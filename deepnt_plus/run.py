"""Experiment runner: baselines, GNN baselines, DeepNT (static), DeepNT+ (full) and ablations over several seeds.

Everything is written under <out_dir>: config.json, predictions.csv, metrics_*.csv, history.csv, active learning.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import active
from .baselines import run_convex_baseline, run_structural_baselines
from .data import NODE_FEATURES, load_dataset
from .model import DeepNTPlus, ModelConfig
from .train import TrainConfig, evaluate, score, train_model

# name -> (model overrides, causal)
VARIANTS = {
    'GraphSAGE': (dict(gnn_kind='sage', tomography=False, convex=False, uncertainty=False), False),
    'GAT': (dict(gnn_kind='gat', tomography=False, convex=False, uncertainty=False), False),
    'DeepNT (static)': (dict(), False),
    'DeepNT (static, causal)': (dict(), True),
    'DeepNT+ (full)': (dict(temporal=True), True),
}
ABLATIONS = {
    'DeepNT+ - temporal branch': (dict(temporal=False), True),
    'DeepNT+ - convex fusion': (dict(temporal=True, convex=False), True),
    'DeepNT+ - tomography layer': (dict(temporal=True, tomography=False, convex=False), True),
    'DeepNT+ - uncertainty (MSE only)': (dict(temporal=True, uncertainty=False), True),
    'DeepNT+ - context features': (dict(temporal=True, context_features=False), True),
}


def build(name, table, graphs, norm, has_loss, fast=False):
    overrides, causal = table[name]
    mcfg = ModelConfig(node_dim=len(NODE_FEATURES), has_loss=has_loss, **({'d': 32, 'mem_dim': 16} if fast else {}), **overrides)
    return mcfg, causal


def run_variant(name, table, graphs, norm, seed, epochs, has_loss, log=print, fast=False, mc=20):
    mcfg, causal = build(name, table, graphs, norm, has_loss, fast)
    tcfg = TrainConfig(epochs=epochs, causal=causal)
    torch.manual_seed(seed)
    model = DeepNTPlus(mcfg, norm)
    hist = train_model(model, graphs, tcfg, seed, log=log)
    frames = []
    for split in ('val', 'test'):
        f = evaluate(model, graphs, split, tcfg, mc=mc)
        f['model'], f['seed'] = name, seed
        frames.append(f)
    hist['model'], hist['seed'] = name, seed
    return model, tcfg, pd.concat(frames, ignore_index=True), hist


def summarise(pred, by=('model', 'split'), dist='laplace'):
    """Per model/split metrics; for several seeds also the mean and std across seeds."""
    rows = []
    for keys, grp in pred.groupby(list(by) + (['seed'] if 'seed' in pred else [])):
        rows.append({**dict(zip(list(by) + (['seed'] if 'seed' in pred else []), keys)), **score(grp, dist=dist)})
    per_seed = pd.DataFrame(rows)
    if 'seed' not in per_seed:
        return per_seed
    metrics = [c for c in per_seed.columns if c not in list(by) + ['seed']]
    mean = per_seed.groupby(list(by))[metrics].mean().add_suffix('_mean')
    std = per_seed.groupby(list(by))[metrics].std().add_suffix('_std')
    return pd.concat([mean, std], axis=1).reset_index()


def run_all(processed_dir, out_dir, seeds=(0, 1, 2), epochs=150, variants=None, ablations=True,
            active_learning=True, fast=False, holdout_monitor=None, loss_type='laplace', use_geo=False, log=print):
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs, norm = load_dataset(processed_dir, holdout_monitor, use_geo=use_geo)
    has_loss = bool(any((~torch.isnan(g.loss)).any() for g in graphs.values()))
    log(f"{len(graphs)} graphs | packet-loss labels available: {has_loss}")

    preds, hists = [], []
    for name in (variants or list(VARIANTS)) + (list(ABLATIONS) if ablations else []):
        table = VARIANTS if name in VARIANTS else ABLATIONS
        for seed in seeds:
            log(f"--- {name} | seed {seed}")
            model, tcfg, pred, hist = run_variant(name, table, graphs, norm, seed, epochs, has_loss, log, fast)
            preds.append(pred)
            hists.append(hist)
            if name == 'DeepNT (static)' and seed == seeds[0]:
                static_model, static_cfg = model, tcfg
                torch.save(model.state_dict(), out_dir / 'deepnt_static.pt')
            if name == 'DeepNT+ (full)' and seed == seeds[0]:
                torch.save(model.state_dict(), out_dir / 'deepnt_plus_full.pt')

    neural = pd.concat(preds, ignore_index=True)
    structural = run_structural_baselines(graphs, seeds[0])
    convex = run_convex_baseline(graphs, norm)
    classic = pd.concat([structural, convex], ignore_index=True)
    classic = classic[classic['split'].isin(['val', 'test'])].assign(seed=seeds[0])
    all_pred = pd.concat([neural, classic], ignore_index=True)

    all_pred.to_csv(out_dir / 'predictions.csv', index=False)
    pd.concat(hists, ignore_index=True).to_csv(out_dir / 'history.csv', index=False)
    summary = summarise(all_pred, dist=loss_type)
    summary.to_csv(out_dir / 'metrics_summary.csv', index=False)

    if active_learning and 'DeepNT (static)' in (variants or VARIANTS):
        curves = active.run(static_model, graphs, static_cfg, split='test', mc=20)
        curves.to_csv(out_dir / 'active_learning.csv', index=False)

    config = {'seeds': list(seeds), 'epochs': epochs, 'fast': fast, 'loss_type': loss_type, 'normalization': norm,
              'variants': {n: {'model': ModelConfig(node_dim=len(NODE_FEATURES), has_loss=has_loss,
                                                    **VARIANTS.get(n, ABLATIONS.get(n))[0]).to_dict(),
                               'causal': VARIANTS.get(n, ABLATIONS.get(n))[1]}
                           for n in (variants or list(VARIANTS)) + (list(ABLATIONS) if ablations else [])},
              'train': TrainConfig(epochs=epochs, loss_type=loss_type).to_dict(), 'graph_split': {g.gid: g.split for g in graphs.values()},
              'protocol': {'context': 'observed paths (fixed mask) at val/test; random masks per epoch in training',
                           'targets': 'hidden (observed=0) paths that are not RTT outliers',
                           'train_label_scope': TrainConfig().train_label_scope,
                           'packet_loss': 'not trained: no labels' if not has_loss else 'trained where labelled'},
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'numpy': np.__version__, 'pandas': pd.__version__},
              'holdout_monitor': holdout_monitor, 'processed_dir': str(processed_dir)}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    return all_pred, summary
