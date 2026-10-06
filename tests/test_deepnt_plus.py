"""Run with:  python tests/test_deepnt_plus.py   (or pytest).  Uses only synthetic data."""
import dataclasses
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepnt_plus import active
from deepnt_plus.data import NODE_FEATURES, load_dataset
from deepnt_plus.layers import GATConv
from deepnt_plus.metrics import calibration_metrics, regression_metrics
from deepnt_plus.model import DeepNTPlus, ModelConfig
from deepnt_plus.train import (TrainConfig, causal_blocks, eval_episode, evaluate, predict_frame, sample_episode,
                               score, train_model)
from synthetic_dataset import make

_CACHE = {}


def data():
    if 'd' not in _CACHE:
        proc = make()
        _CACHE['d'] = load_dataset(proc) + (proc,)
    return _CACHE['d'][:2]


def small_model(norm, **kw):
    torch.manual_seed(0)
    return DeepNTPlus(ModelConfig(node_dim=len(NODE_FEATURES), d=32, mem_dim=16, **kw), norm)


def test_data_loading():
    graphs, norm = data()
    assert len(graphs) == 15
    assert [g.split for g in graphs.values()].count('train') == 10
    for g in graphs.values():
        assert g.A.shape == (g.n_paths, g.edge_uv.shape[0])
        assert torch.allclose(g.A.sum(1), g.hops)
        assert torch.isfinite(g.x).all() and torch.isfinite(g.y).all()
        assert g.observed.sum() > 0 and (~g.observed).sum() > 0
    assert norm['fit_on'] == 'train graphs only'


def test_leave_monitor_out_split_and_monitor_balance():
    from collections import Counter
    data()
    proc = _CACHE['d'][2]
    std_graphs, _ = load_dataset(proc)
    for split in ('train', 'test'):                       # Dataset_Build must give every monitor to train and test
        mons = {m for g in std_graphs.values() if g.split == split for m in g.monitor}
        assert mons == {m for g in std_graphs.values() for m in g.monitor}, (split, mons)
    graphs, norm = load_dataset(proc, holdout_monitor='dac-bd')
    dom = {g.gid: Counter(g.monitor).most_common(1)[0][0] for g in graphs.values()}
    tests = [g for g in graphs.values() if g.split == 'test']
    assert tests and all(dom[g.gid] == 'dac-bd' for g in tests)
    assert all(dom[g.gid] != 'dac-bd' for g in graphs.values() if g.split != 'test')
    assert any(g.split == 'val' for g in graphs.values()) and any(g.split == 'train' for g in graphs.values())
    assert norm['fit_on'] == 'train graphs only'


def test_robust_metrics_present():
    m = regression_metrics([100, 200, 400], [110, 180, 500])
    assert {'MdAPE', 'MAE_log'} <= set(m) and m['MdAPE'] == 10.0


def test_gat_attention_is_a_distribution():
    torch.manual_seed(0)
    layer = GATConv(6, 8, heads=2)
    h = torch.ones(5, 6)                                   # identical nodes -> attention weights must average to lin(h)
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    out = layer(h, ei)
    assert torch.allclose(out, layer.lin(h) + layer.bias, atol=1e-5)


def test_ridge_matches_closed_form():
    graphs, norm = data()
    torch.manual_seed(1)
    A, y = torch.rand(6, 20).round().double(), torch.rand(6).double() * 100
    x0, alpha = torch.rand(20).double() * 10, 10.0
    dual = x0 + A.T @ torch.linalg.solve(A @ A.T + alpha * torch.eye(6).double(), y - A @ x0)
    primal = torch.linalg.solve(A.T @ A + alpha * torch.eye(20).double(), A.T @ y + alpha * x0)
    assert torch.allclose(dual, primal, atol=1e-6)


def test_no_target_leakage():
    """Changing the RTT of the hidden (target) paths must not change any prediction."""
    graphs, norm = data()
    g = next(g for g in graphs.values() if g.split == 'test')
    ctx, tgt = eval_episode(g)
    for kw in ({}, {'temporal': True}):
        model = small_model(norm, **kw).eval()
        base = model(g, ctx, tgt, g.y[ctx])['y_hat']
        y2 = g.y.clone()
        y2[tgt] = y2[tgt] * 7 + 123
        again = model(dataclasses.replace(g, y=y2), ctx, tgt, y2[ctx])['y_hat']
        assert torch.equal(base, again), f"target RTT leaked into the prediction ({kw})"


def test_causality_of_temporal_evaluation():
    """A measurement made after a target's time block must not influence that target's prediction."""
    graphs, norm = data()
    g = next(g for g in graphs.values() if g.split == 'val')
    cfg = TrainConfig(causal=True, n_blocks=6)
    model = small_model(norm, temporal=True).eval()
    before = predict_frame(model, g, cfg).set_index('path_id')
    ctx, _ = eval_episode(g)
    last = ctx[torch.argmax(g.t[ctx])]
    y2 = g.y.clone()
    y2[last] = y2[last] * 5
    after = predict_frame(model, dataclasses.replace(g, y=y2), cfg).set_index('path_id')
    t_by_id = dict(zip(g.path_ids, g.t.tolist()))
    early = [p for p in before.index if t_by_id[p] <= float(g.t[last])]
    assert len(early) > 0
    blocks = causal_blocks(g, ctx, eval_episode(g)[1], cfg.n_blocks)
    unaffected = [g.path_ids[i] for c, t in blocks if not (c == last).any() for i in t.tolist()]
    assert unaffected, "test needs at least one block that cannot see the perturbed measurement"
    assert np.allclose(before.loc[unaffected, 'y_pred'], after.loc[unaffected, 'y_pred'])


def test_all_variants_forward_and_cold_start():
    graphs, norm = data()
    g = next(iter(graphs.values()))
    ctx, tgt = eval_episode(g)
    for kw in ({}, {'temporal': True}, {'gnn_kind': 'sage', 'tomography': False, 'convex': False, 'uncertainty': False},
               {'gnn_kind': 'gat', 'tomography': False, 'convex': False}, {'convex': False},
               {'context_features': False}, {'has_loss': True}):
        model = small_model(norm, **kw).eval()
        for c in (ctx, ctx[:0]):                         # normal and empty context
            out = model(g, c, tgt, g.y[c])
            assert out['y_hat'].shape == tgt.shape and torch.isfinite(out['y_hat']).all(), kw
            assert (out['logvar'] is not None) == model.cfg.uncertainty


def test_training_beats_median_on_learnable_data():
    graphs, norm = data()
    model = small_model(norm)
    cfg = TrainConfig(epochs=40, patience=40)
    hist = train_model(model, graphs, cfg, seed=0)
    assert hist['train_loss'].iloc[-1] < hist['train_loss'].iloc[0]
    f = evaluate(model, graphs, 'val', cfg)
    med = float(np.median(torch.cat([g.y[g.ok] for g in graphs.values() if g.split == 'train']).numpy()))
    mae_model = np.abs(f['y_pred'] - f['y_true']).mean()
    mae_median = np.abs(med - f['y_true']).mean()
    assert mae_model < 0.8 * mae_median, (mae_model, mae_median)


def test_outliers_excluded_everywhere():
    graphs, norm = data()
    for g in graphs.values():
        ctx, tgt = eval_episode(g)
        assert g.ok[ctx].all() and g.ok[tgt].all()
        ep = sample_episode(g, np.random.default_rng(0), TrainConfig())
        assert g.ok[ep[0]].all() and g.ok[ep[1]].all()


def test_metrics_and_calibration():
    rng = np.random.default_rng(0)
    mu, sd = rng.normal(5, 1, 20000), np.full(20000, 0.3)
    y = np.expm1(mu + rng.normal(0, 0.3, 20000))
    cal = calibration_metrics(y, mu, sd)
    assert cal['ECE_reg'] < 0.02 and abs(cal['coverage_95'] - 0.95) < 0.01
    assert regression_metrics([1, 2, 3], [1, 2, 3])['MAE'] == 0


def test_active_learning_runs():
    graphs, norm = data()
    model = small_model(norm)
    cfg = TrainConfig()
    curves = active.run(model, graphs, cfg, split='test', rounds=3, mc=4, random_seeds=2)
    assert set(curves['strategy']) == set(active.STRATEGIES)
    assert (curves.groupby(['graph_id', 'strategy'])['measured'].max() > 0).all()
    mean, auc = active.summarise(curves)
    assert len(auc) == len(active.STRATEGIES)


def test_run_all_writes_everything():
    from deepnt_plus.run import run_all
    graphs, norm, proc = _CACHE.get('d') or (None, None, None)
    if proc is None:
        data()
        proc = _CACHE['d'][2]
    out = Path(tempfile.mkdtemp())
    pred, summary = run_all(proc, out, seeds=(0,), epochs=3, variants=['GraphSAGE', 'DeepNT (static)', 'DeepNT+ (full)'],
                            ablations=False, fast=True, log=lambda *_: None)
    for name in ('predictions.csv', 'metrics_summary.csv', 'history.csv', 'config.json', 'active_learning.csv'):
        assert (out / name).exists(), name
    needed = {'graph_id', 'path_id', 'source', 'destination', 'observed', 'y_true', 'y_pred', 'split'}
    assert needed <= set(pred.columns)
    assert {'Median', 'Random forest', 'Convex tomography (classical)', 'DeepNT+ (full)'} <= set(pred['model'])
    assert (pred['observed'] == 0).all() and set(pred['split']) <= {'val', 'test'}


if __name__ == "__main__":
    failed = 0
    for name, fn in [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
