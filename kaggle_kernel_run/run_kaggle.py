import sys
import os
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm as _norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("🛡️ KAGGLE GPU AUTHENTIC BENCHMARK SUITE: 100% LEAKAGE-FREE EVALUATION")
print("   (Zero Geographical Distance, Pure Topological & Traffic Features Only)")
print("=" * 80)
print(f"PyTorch Version : {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available  : {cuda_avail}")
device = torch.device("cuda:0" if cuda_avail else "cpu")
if cuda_avail:
    print(f"GPU Accelerator : {torch.cuda.get_device_name(0)}")
    print(f"Total GPU VRAM  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print("=" * 80)

# 1. Locate dataset
found_root = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "deepnt_plus" in dirs and "data" in dirs:
        found_root = Path(root)
        break

if not found_root:
    for root, dirs, files in os.walk("."):
        if "deepnt_plus" in dirs and "data" in dirs:
            found_root = Path(root)
            break

if not found_root:
    found_root = Path("/kaggle/input/datasets/atikmorshedsohan/deepnt-plus-network-tomography-suite")

print(f"✅ Authentic Suite Root: {found_root}")
sys.path.insert(0, str(found_root))

from deepnt_plus import data, layers
from deepnt_plus.model import DeepNTPlus, ModelConfig
from deepnt_plus.tgn import TemporalBranch

# 2. Authentic Graph Loader (Strictly without geo_distance_km, dynamic norm statistics, genuine timestamps)
def load_authentic_graphs(proc_dir, use_geo=False):
    proc = Path(proc_dir)
    nodes = pd.read_csv(proc / 'node_features.csv', dtype={'graph_id': str, 'node_id': str})
    edges = pd.read_csv(proc / 'edge_features.csv', dtype={'graph_id': str, 'edge_id': str, 'src': str, 'dst': str})
    paths = pd.read_csv(proc / 'paths_all.csv', dtype={'graph_id': str, 'path_id': str}, keep_default_na=False)
    truth = pd.read_csv(proc / 'path_ground_truth.csv', dtype={'path_id': str}, keep_default_na=False).set_index('path_id')

    # Parse authentic timestamps from measurement_time UTC (reject synthetic 2026-01-01 placeholders in IPv6)
    t_all = pd.to_datetime(paths['measurement_time'].replace('', np.nan), errors='coerce', utc=True)
    is_synthetic_ts = bool(t_all.notna().any() and (t_all.min().year == 2026 and t_all.min().month == 1 and t_all.min().day == 1))
    has_real_timestamps = bool(t_all.notna().any() and (t_all.max() > t_all.min()) and not is_synthetic_ts)
    if has_real_timestamps:
        t0 = t_all.min()
        paths['t_hours'] = (t_all - t0).dt.total_seconds() / 3600.0
    else:
        paths['t_hours'] = 0.0

    split_of = paths.groupby('graph_id')['split'].first().to_dict()
    graph_ids = sorted(split_of)

    # Standardize nodes (Degree, Centralities, PageRank, etc.)
    node_frame = nodes.copy()
    for c in data.NODE_FEATURES:
        node_frame[c] = pd.to_numeric(node_frame[c], errors='coerce')
        if c in data.LOG_COLUMNS:
            node_frame[c] = np.log1p(node_frame[c].clip(lower=0))
    train_nodes = node_frame[node_frame['graph_id'].map(split_of) == 'train']
    node_stats = {c: (float(train_nodes[c].mean()), float(train_nodes[c].std() or 1.0)) for c in data.NODE_FEATURES}

    # Standardize edges (Log Path Count only: 100% structural, Zero Target Leakage)
    edge_log = np.log1p(edges['path_count_all'].astype(float))
    train_edges = edge_log[edges['graph_id'].map(split_of) == 'train']
    edge_stats = (float(train_edges.mean()), float(train_edges.std() or 1.0))

    has_geo = bool(use_geo and ('geo_distance_km' in edges.columns and edges['geo_distance_km'].notna().any()))
    if has_geo:
        geo_log = np.log1p(edges['geo_distance_km'].astype(float).clip(lower=0))
        train_geo = geo_log[edges['graph_id'].map(split_of) == 'train']
        geo_stats = (float(train_geo.mean()), float(train_geo.std() or 1.0))

    graphs = {}
    for gid in graph_ids:
        nf = node_frame[node_frame['graph_id'] == gid].reset_index(drop=True)
        ef = edges[edges['graph_id'] == gid].reset_index(drop=True)
        pf = paths[paths['graph_id'] == gid].reset_index(drop=True)
        idx = {n: i for i, n in enumerate(nf['node_id'])}

        feats = np.stack([data._standardise(nf[c].fillna(node_stats[c][0]).to_numpy(float), *node_stats[c])
                          for c in data.NODE_FEATURES], axis=1)
        uv = np.array([[idx[a], idx[b]] for a, b in zip(ef['src'], ef['dst'])], dtype=np.int64)

        feat_count = data._standardise(np.log1p(ef['path_count_all'].astype(float)).to_numpy(), *edge_stats)
        if has_geo:
            feat_geo = data._standardise(np.log1p(ef['geo_distance_km'].astype(float).clip(lower=0)).to_numpy(), *geo_stats)
            e_feat = np.stack([feat_count, feat_geo], axis=1)
        else:
            e_feat = feat_count[:, None]  # shape: [E, 1]

        A = sp.load_npz(proc / 'tomography' / f'A_{gid}.npz').toarray().astype(np.float32)

        seqs = [[idx[n] for n in s.split('|')] for s in pf['path_nodes']]
        L = max(map(len, seqs))
        pad = np.zeros((len(seqs), L), dtype=np.int64)
        mask = np.zeros((len(seqs), L), dtype=bool)
        Q = np.zeros((len(seqs), len(nf)), dtype=np.float32)
        for i, s in enumerate(seqs):
            pad[i, :len(s)] = s
            mask[i, :len(s)] = True
            Q[i, s] = 1.0

        hops = pf['path_length'].to_numpy(float)
        t = pf['t_hours'].fillna(pf['t_hours'].median() if has_real_timestamps else 0.0).fillna(0.0).to_numpy(float)

        gt = truth.loc[pf['path_id']]
        y = torch.tensor(pd.to_numeric(gt['rtt_ms']).to_numpy(float), dtype=torch.float32)
        obs = torch.tensor(pf['observed'].to_numpy(int) == 1, dtype=torch.bool)
        ok = torch.tensor(~gt['quality_flag'].str.startswith('rtt_outlier').to_numpy(bool), dtype=torch.bool)

        e_idx = np.concatenate([uv.T, uv.T[[1, 0]]], axis=1)
        graphs[gid] = data.GraphData(
            gid=gid, split=split_of[gid], x=torch.tensor(feats, dtype=torch.float32),
            node_ids=list(nf['node_id']), edge_uv=torch.tensor(uv, dtype=torch.int64),
            edge_feat=torch.tensor(e_feat, dtype=torch.float32),
            edge_index=torch.tensor(e_idx, dtype=torch.int64),
            A=torch.tensor(A, dtype=torch.float32), Q=torch.tensor(Q, dtype=torch.float32),
            node_pad=torch.tensor(pad, dtype=torch.int64), node_mask=torch.tensor(mask, dtype=torch.bool),
            path_ids=list(pf['path_id']), sources=list(pf['source_node']), destinations=list(pf['destination_node']),
            t=torch.tensor(t, dtype=torch.float32), hops=torch.tensor(hops, dtype=torch.float32),
            y=y, observed=obs, ok=ok, monitor=list(pf['monitor_id']), loss=torch.full((len(pf),), float('nan'))
        )

    # Dynamic normalization strictly derived from observed training paths
    train_y = torch.cat([g.y[g.ok & g.observed] for g in graphs.values() if g.split == 'train'])
    train_hops = torch.cat([g.hops[g.ok & g.observed] for g in graphs.values() if g.split == 'train'])
    log_y = torch.log1p(train_y)
    rates = torch.log1p(train_y / train_hops.clamp(min=1.0))
    norm = {
        'fit_on': 'train graphs only',
        'log_rtt_mean': float(log_y.mean()),
        'log_rtt_std': float(log_y.std() or 1.0),
        'rate_mean': float(rates.mean()),
        'rate_std': float(rates.std() or 1.0),
        'prior_edge_ms': float((train_y / train_hops.clamp(min=1.0)).median().clamp(min=1.0)),
        'edge_dim': 2 if has_geo else 1,
        'has_real_timestamps': has_real_timestamps
    }
    return graphs, norm

# 3. GPU In-Memory Graph Caching (eliminates per-episode dense A matrix PCIe copy overhead)
def cache_graphs_on_device(graphs, target_device):
    cached = {}
    for gid, g in graphs.items():
        cached[gid] = data.GraphData(**{
            k: v.to(target_device) if isinstance(v, torch.Tensor) else v
            for k, v in g.__dict__.items()
        })
    return cached

# 4. Fair Tuned Convex Baseline (Validation-Tuned Residual Ridge with Edge Prior)
def evaluate_convex(graphs, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0), causal=False, n_blocks=6):
    """Fair classical / causal convex tomography baseline:
    Regularizes toward the observed context median per-edge delay x0 (warm start).
    Tuned over alpha grid on the validation split, then evaluated on the test split."""
    val_graphs = [g for g in graphs.values() if g.split == 'val']
    test_graphs = [g for g in graphs.values() if g.split == 'test']

    def solve_graph(g, alpha, is_causal):
        ctx = torch.nonzero(g.observed & g.ok).squeeze(-1)
        tgt = torch.nonzero(~g.observed & g.ok).squeeze(-1)
        A = g.A.cpu().numpy()
        y = g.y.cpu().numpy()
        hops = g.hops.cpu().numpy()

        if not is_causal:
            c_idx = ctx.cpu().numpy()
            t_idx = tgt.cpu().numpy()
            if len(c_idx) == 0:
                med = float(np.median(y[c_idx])) if len(c_idx) else 50.0
                return t_idx, np.full(len(t_idx), med)

            Ac = np.hstack([A[c_idx], np.ones((len(c_idx), 1))])
            edge_ms = float(np.median(y[c_idx] / np.maximum(hops[c_idx], 1.0)))
            x0 = np.concatenate([np.full(A.shape[1], edge_ms), [0.0]])
            resid = y[c_idx] - Ac @ x0
            K = Ac @ Ac.T + alpha * np.eye(len(c_idx))
            x_sol = np.clip(x0 + Ac.T @ np.linalg.solve(K, resid), 0, None)
            At = np.hstack([A[t_idx], np.ones((len(t_idx), 1))])
            return t_idx, At @ x_sol
        else:
            t_order = tgt[torch.argsort(g.t[tgt])]
            all_t, all_p = [], []
            for chunk in torch.tensor_split(t_order, n_blocks):
                if not chunk.numel():
                    continue
                c_sub = ctx[g.t[ctx] < g.t[chunk].min()].cpu().numpy()
                t_sub = chunk.cpu().numpy()
                all_t.extend(t_sub)
                if len(c_sub) == 0:
                    med = float(np.median(y[ctx.cpu().numpy()])) if ctx.numel() else 50.0
                    all_p.extend(np.full(len(t_sub), med))
                else:
                    Ac = np.hstack([A[c_sub], np.ones((len(c_sub), 1))])
                    edge_ms = float(np.median(y[c_sub] / np.maximum(hops[c_sub], 1.0)))
                    x0 = np.concatenate([np.full(A.shape[1], edge_ms), [0.0]])
                    resid = y[c_sub] - Ac @ x0
                    K = Ac @ Ac.T + alpha * np.eye(len(c_sub))
                    x_sol = np.clip(x0 + Ac.T @ np.linalg.solve(K, resid), 0, None)
                    At = np.hstack([A[t_sub], np.ones((len(t_sub), 1))])
                    all_p.extend(At @ x_sol)
            return np.array(all_t), np.array(all_p)

    # 1. Validation Tuning
    val_errors = {}
    for a in alphas:
        errs = []
        for g in val_graphs:
            t_idx, y_pred = solve_graph(g, a, causal)
            errs.extend(np.abs(g.y[t_idx].cpu().numpy() - y_pred))
        val_errors[a] = float(np.mean(errs)) if errs else float('inf')

    best_alpha = min(val_errors, key=val_errors.get)

    # 2. Test Evaluation with best alpha
    trues, preds = [], []
    for g in test_graphs:
        t_idx, y_pred = solve_graph(g, best_alpha, causal)
        trues.extend(g.y[t_idx].cpu().numpy())
        preds.extend(y_pred)

    y_t, y_p = np.array(trues), np.array(preds)
    abs_err = np.abs(y_t - y_p)
    ss_tot = np.sum((y_t - np.mean(y_t))**2)
    return {
        'MAE': float(np.mean(abs_err)),
        'RMSE': float(np.sqrt(np.mean(abs_err**2))),
        'R2': float(1.0 - np.sum(abs_err**2) / (ss_tot if ss_tot > 0 else 1.0)),
        'MdAPE': float(np.median(abs_err / np.maximum(y_t, 1e-5)) * 100),
        'best_alpha': best_alpha,
        'val_MAE': val_errors[best_alpha]
    }

# 5. Non-Graph Baselines (Median, Hop-count with Intercept)
def evaluate_simple_baselines(graphs):
    train_y = np.concatenate([g.y[g.ok & g.observed].cpu().numpy() for g in graphs.values() if g.split == 'train'])
    train_hops = np.concatenate([g.hops[g.ok & g.observed].cpu().numpy() for g in graphs.values() if g.split == 'train'])
    test_y = np.concatenate([g.y[~g.observed & g.ok].cpu().numpy() for g in graphs.values() if g.split == 'test'])
    test_hops = np.concatenate([g.hops[~g.observed & g.ok].cpu().numpy() for g in graphs.values() if g.split == 'test'])

    ss_tot = np.sum((test_y - np.mean(test_y))**2)

    # 1. Median
    med = float(np.median(train_y))
    mae_med = float(np.mean(np.abs(test_y - med)))
    r2_med = float(1.0 - np.sum((test_y - med)**2) / ss_tot)

    # 2. Hop-count regression with INTERCEPT (y = w1 * hops + w0)
    X_train_hop = np.stack([train_hops, np.ones_like(train_hops)], axis=1)
    params, _, _, _ = np.linalg.lstsq(X_train_hop, train_y, rcond=None)
    pred_hop = np.clip(test_hops * params[0] + params[1], 0, None)
    mae_hop = float(np.mean(np.abs(test_y - pred_hop)))
    r2_hop = float(1.0 - np.sum((test_y - pred_hop)**2) / ss_tot)

    return {
        'Median': {'MAE': mae_med, 'R2': r2_med},
        'Hop-count (Linear w/ Intercept)': {'MAE': mae_hop, 'R2': r2_hop, 'slope': float(params[0]), 'intercept': float(params[1])}
    }

# 6. Authentic Training & Evaluation Engine (with GPU Graph Caching & 5 Seeds)
def train_and_eval_authentic(graphs_cached, norm, model_name='DeepNT+ (Full Dynamic, Ours)',
                             epochs=40, seeds=[0, 1, 2, 3, 4], loss_type='laplace'):
    """Trains and evaluates models using authentic library architecture,
    caching graphs on GPU to avoid PCIe bottleneck, with full 5-seed evaluation."""
    seed_metrics = []
    final_preds, final_trues, final_sigmas = [], [], []

    # Map model name to ModelConfig
    if model_name == 'DeepNT+ (Full Dynamic, Ours)':
        cfg = ModelConfig(d=64, gnn_kind='split', tomography=True, convex=True, temporal=True, uncertainty=True)
    elif model_name == 'DeepNT (Static + Convex Gate)':
        cfg = ModelConfig(d=64, gnn_kind='split', tomography=True, convex=True, temporal=False, uncertainty=True)
    elif model_name == 'DeepNT (Neural Tomo, No Convex)':
        cfg = ModelConfig(d=64, gnn_kind='split', tomography=True, convex=False, temporal=False, uncertainty=True)
    elif model_name == 'GAT':
        cfg = ModelConfig(d=64, gnn_kind='gat', tomography=False, convex=False, temporal=False, uncertainty=False)
    elif model_name == 'GraphSAGE':
        cfg = ModelConfig(d=64, gnn_kind='sage', tomography=False, convex=False, temporal=False, uncertainty=False)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    train_ids = [k for k, g in graphs_cached.items() if g.split == 'train']
    val_ids = [k for k, g in graphs_cached.items() if g.split == 'val']
    test_ids = [k for k, g in graphs_cached.items() if g.split == 'test']

    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        model = DeepNTPlus(cfg, norm).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        best_val, best_state = float('inf'), None

        for epoch in range(1, epochs + 1):
            model.train()
            # 6 episodes per graph per epoch
            for _ in range(6):
                for gid in rng.permutation(train_ids):
                    g = graphs_cached[gid]
                    pool = np.flatnonzero(g.ok.cpu().numpy())
                    rng.shuffle(pool)
                    n_ctx = max(2, int(round(rng.uniform(0.25, 0.55) * len(pool))))
                    ctx = torch.tensor(np.sort(pool[:n_ctx]), device=device)
                    tgt = torch.tensor(np.sort(pool[n_ctx:]), device=device)

                    out = model(g, ctx, tgt, g.y[ctx])
                    z, mu = model.z_of(g.y[tgt]), out['mu_z']

                    if out['logvar'] is not None:
                        lv = out['logvar']
                        # Laplace NLL loss
                        loss = (torch.exp(-lv) * torch.abs(z - mu) + lv).mean() + 0.5 * torch.abs(z - mu).mean()
                    else:
                        loss = torch.abs(z - mu).mean()

                    if model.cfg.tomography and ctx.numel():
                        loss += 0.5 * torch.abs(model.z_of(out['y_add_ctx']) - model.z_of(g.y[ctx])).mean()

                    # Residual regularization
                    if model.cfg.convex and 'y_convex' in out:
                        loss += 0.05 * torch.abs(model.z_of(out['y_hat']) - model.z_of(out['y_convex'])).mean()

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            sched.step()

            # Multi-mask smoothed validation
            model.eval()
            with torch.no_grad():
                val_maes = []
                for gid in val_ids:
                    g = graphs_cached[gid]
                    ctx = torch.nonzero(g.observed & g.ok).squeeze(-1)
                    tgt = torch.nonzero(~g.observed & g.ok).squeeze(-1)
                    out = model(g, ctx, tgt, g.y[ctx])
                    val_maes.append((out['y_hat'] - g.y[tgt]).abs().mean().item())

                    v_pool = np.flatnonzero(g.ok.cpu().numpy())
                    val_rng = np.random.default_rng(3000 + epoch)
                    for _ in range(2):
                        val_rng.shuffle(v_pool)
                        v_n_ctx = max(2, int(round(val_rng.uniform(0.25, 0.55) * len(v_pool))))
                        v_ctx = torch.tensor(np.sort(v_pool[:v_n_ctx]), device=device)
                        v_tgt = torch.tensor(np.sort(v_pool[v_n_ctx:]), device=device)
                        v_out = model(g, v_ctx, v_tgt, g.y[v_ctx])
                        val_maes.append((v_out['y_hat'] - g.y[v_tgt]).abs().mean().item())

                mean_val = float(np.mean(val_maes)) if val_maes else float('inf')
                if mean_val < best_val:
                    best_val = mean_val
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate best checkpoint on test set
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            s_preds, s_trues, s_sigmas = [], [], []
            for gid in test_ids:
                g = graphs_cached[gid]
                ctx = torch.nonzero(g.observed & g.ok).squeeze(-1)
                tgt = torch.nonzero(~g.observed & g.ok).squeeze(-1)
                out = model(g, ctx, tgt, g.y[ctx])
                s_preds.extend(out['y_hat'].cpu().numpy())
                s_trues.extend(g.y[tgt].cpu().numpy())

                if out['logvar'] is not None:
                    # True Laplace std: sigma = sqrt(2) * b = sqrt(2) * exp(lv) * log_rtt_std
                    sigma = np.sqrt(2.0) * torch.exp(out['logvar']) * model.log_rtt_std
                    s_sigmas.extend(sigma.cpu().numpy())

            y_t, y_p = np.array(s_trues), np.array(s_preds)
            abs_err = np.abs(y_t - y_p)
            rel_err = abs_err / np.maximum(y_t, 1e-5)
            mae = float(np.mean(abs_err))
            rmse = float(np.sqrt(np.mean(abs_err**2)))
            ss_tot = np.sum((y_t - np.mean(y_t))**2)
            r2 = float(1.0 - np.sum(abs_err**2) / (ss_tot if ss_tot > 0 else 1.0))
            mdape = float(np.median(rel_err) * 100)

            cov90, ece = np.nan, np.nan
            if s_sigmas:
                z = np.log1p(y_t)
                mu = np.log1p(np.clip(y_p, 0, None))
                sd = np.array(s_sigmas)
                b = sd / np.sqrt(2.0)
                # Consistent Laplace interval coverage
                cov90 = float(np.mean(np.abs(z - mu) <= -b * np.log(0.10)))
                ece = float(np.mean([abs(np.mean(np.abs(z - mu) <= -b * np.log(1 - l)) - l) for l in [0.5, 0.8, 0.9, 0.95]]))

            seed_metrics.append({'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MdAPE': mdape, 'Cov90': cov90, 'ECE': ece})
            if seed == seeds[0]:
                final_preds, final_trues, final_sigmas = s_preds, s_trues, s_sigmas

    df_m = pd.DataFrame(seed_metrics)
    return {
        'MAE_mean': df_m['MAE'].mean(), 'MAE_std': df_m['MAE'].std() if len(df_m) > 1 else 0.0,
        'RMSE_mean': df_m['RMSE'].mean(), 'RMSE_std': df_m['RMSE'].std() if len(df_m) > 1 else 0.0,
        'R2_mean': df_m['R2'].mean(), 'R2_std': df_m['R2'].std() if len(df_m) > 1 else 0.0,
        'MdAPE_mean': df_m['MdAPE'].mean(),
        'Cov90_mean': df_m['Cov90'].mean(),
        'ECE_mean': df_m['ECE'].mean(),
        'trues': final_trues, 'preds': final_preds, 'sigmas': final_sigmas
    }

# 7. Execute Authentic Benchmark Suite
def main():
    datasets = [
        ('IPv4 (10-Graphs, New)', found_root / 'data' / 'ipv4_10graphs'),
        ('IPv6 (10-Graphs, New)', found_root / 'data' / 'ipv6_10graphs'),
        ('IPv4 (run_03, Dense)', found_root / 'data' / 'processed'),
        ('IPv4 (run_02, 15-Graphs)', found_root / 'data' / 'processed_15graphs_backup'),
    ]

    summary_rows = []
    scatter_records = {}
    calibration_records = {}
    EVAL_SEEDS = [0, 1, 2, 3, 4]  # 5-seed academic standard

    for name, path in datasets:
        if not path.exists():
            print(f"⚠️ Skipping {name}: path {path} not found")
            continue
        print(f"\n{'='*75}")
        print(f"🌟 EVALUATING AUTHENTIC (LEAKAGE-FREE) DATASET: {name}")
        print(f"{'='*75}")

        graphs, norm = load_authentic_graphs(path, use_geo=False)
        graphs_cached = cache_graphs_on_device(graphs, device)
        has_temporal = norm.get('has_real_timestamps', False)

        if not has_temporal:
            print("⚠️ [Dataset Notice]: This dataset lacks fine-grained synchronized traceroute timestamps.")
            print("   In accordance with thesis methodology, it is evaluated strictly under static/non-causal regime")
            print("   to prevent artificial temporal artifacts.")

        # 1. Non-Graph Baselines
        simples = evaluate_simple_baselines(graphs)
        for s_name, s_res in simples.items():
            print(f"  [{s_name:32s}] MAE: {s_res['MAE']:.2f} ms | R2: {s_res['R2']*100:.1f}%")
            summary_rows.append({
                'Dataset': name, 'Model': s_name, 'Family': 'Non-Graph',
                'MAE': s_res['MAE'], 'MAE_std': 0.0, 'R2': s_res['R2']*100, 'Cov90': np.nan, 'ECE': np.nan
            })

        # 2. Fair Classical Convex Tomography (Validation-Tuned Ridge with Warm Start)
        conv_res = evaluate_convex(graphs, causal=False)
        print(f"  [{'Convex Tomography (Tuned Ridge)':32s}] MAE: {conv_res['MAE']:.2f} ms | R2: {conv_res['R2']*100:.1f}% | (alpha={conv_res['best_alpha']})")
        summary_rows.append({
            'Dataset': name, 'Model': 'Convex Tomography (Tuned Ridge)', 'Family': 'Analytical Tomography',
            'MAE': conv_res['MAE'], 'MAE_std': 0.0, 'R2': conv_res['R2']*100, 'Cov90': np.nan, 'ECE': np.nan
        })

        # 3. Standard GNNs (GAT, GraphSAGE)
        for gnn_type, display in [('GAT', 'GAT (Pure Spatial GNN)'), ('GraphSAGE', 'GraphSAGE (Pure Spatial GNN)')]:
            t0 = time.time()
            res = train_and_eval_authentic(graphs_cached, norm, model_name=gnn_type, epochs=35, seeds=EVAL_SEEDS)
            print(f"  [{display:32s}] MAE: {res['MAE_mean']:.2f} ± {res['MAE_std']:.2f} ms | R2: {res['R2_mean']*100:.1f}% ({time.time()-t0:.1f}s)")
            summary_rows.append({
                'Dataset': name, 'Model': display, 'Family': 'Standard GNN',
                'MAE': res['MAE_mean'], 'MAE_std': res['MAE_std'], 'R2': res['R2_mean']*100, 'Cov90': np.nan, 'ECE': np.nan
            })

        # 4. DeepNT (Neural Tomo, No Convex) - Pure spatial neural tomography ablation
        t0 = time.time()
        res_noconv = train_and_eval_authentic(graphs_cached, norm, model_name='DeepNT (Neural Tomo, No Convex)', epochs=35, seeds=EVAL_SEEDS)
        print(f"  [{'DeepNT (Neural Tomo, No Convex)':32s}] MAE: {res_noconv['MAE_mean']:.2f} ± {res_noconv['MAE_std']:.2f} ms | R2: {res_noconv['R2_mean']*100:.1f}% ({time.time()-t0:.1f}s)")
        summary_rows.append({
            'Dataset': name, 'Model': 'DeepNT (Neural Tomo, No Convex)', 'Family': 'Neural Tomography',
            'MAE': res_noconv['MAE_mean'], 'MAE_std': res_noconv['MAE_std'], 'R2': res_noconv['R2_mean']*100,
            'Cov90': res_noconv['Cov90_mean']*100, 'ECE': res_noconv['ECE_mean']
        })

        # 5. DeepNT (Static + Convex Gate)
        t0 = time.time()
        res_static_conv = train_and_eval_authentic(graphs_cached, norm, model_name='DeepNT (Static + Convex Gate)', epochs=35, seeds=EVAL_SEEDS)
        print(f"  [{'DeepNT (Static + Convex Gate)':32s}] MAE: {res_static_conv['MAE_mean']:.2f} ± {res_static_conv['MAE_std']:.2f} ms | R2: {res_static_conv['R2_mean']*100:.1f}% ({time.time()-t0:.1f}s)")
        summary_rows.append({
            'Dataset': name, 'Model': 'DeepNT (Static + Convex Gate)', 'Family': 'Hybrid Tomography',
            'MAE': res_static_conv['MAE_mean'], 'MAE_std': res_static_conv['MAE_std'], 'R2': res_static_conv['R2_mean']*100,
            'Cov90': res_static_conv['Cov90_mean']*100, 'ECE': res_static_conv['ECE_mean']
        })

        # 6. DeepNT+ (Full Dynamic: Temporal + Convex) - Run if genuine temporal timestamps exist
        if has_temporal:
            t0 = time.time()
            res_full = train_and_eval_authentic(graphs_cached, norm, model_name='DeepNT+ (Full Dynamic, Ours)', epochs=35, seeds=EVAL_SEEDS)
            print(f"  [{'DeepNT+ (Full Dynamic, Ours)':32s}] MAE: {res_full['MAE_mean']:.2f} ± {res_full['MAE_std']:.2f} ms | R2: {res_full['R2_mean']*100:.1f}% | Cov90: {res_full['Cov90_mean']*100:.1f}% | ECE: {res_full['ECE_mean']:.3f} ({time.time()-t0:.1f}s) 🥇")
            summary_rows.append({
                'Dataset': name, 'Model': 'DeepNT+ (Full Dynamic, Ours)', 'Family': 'Dynamic Hybrid Tomography',
                'MAE': res_full['MAE_mean'], 'MAE_std': res_full['MAE_std'], 'R2': res_full['R2_mean']*100,
                'Cov90': res_full['Cov90_mean']*100, 'ECE': res_full['ECE_mean']
            })
            best_for_plots = res_full
        else:
            best_for_plots = res_static_conv

        scatter_records[name] = (best_for_plots['trues'], best_for_plots['preds'], conv_res['MAE'])
        calibration_records[name] = (best_for_plots['trues'], best_for_plots['preds'], best_for_plots['sigmas'])

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("results_10graphs_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_dir / "authentic_master_benchmark_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("📊 100% AUTHENTIC MASTER BENCHMARK TABLE (LEAKAGE-FREE)")
    print("=" * 80)
    print(df_sum[['Dataset', 'Model', 'MAE', 'MAE_std', 'R2', 'Cov90', 'ECE']].to_string(index=False))
    print("=" * 80)

    # 8. Visual Plots
    print("\n🎨 Generating 100% authentic publication-quality charts...")

    # Figure 1: Authentic Master Comparison Bar Chart (10-Graphs IPv4 & IPv6)
    plot_datasets = [d for d in ['IPv4 (10-Graphs, New)', 'IPv6 (10-Graphs, New)'] if d in df_sum['Dataset'].values]
    if plot_datasets:
        fig, axes = plt.subplots(1, len(plot_datasets), figsize=(16, 5.5))
        if len(plot_datasets) == 1:
            axes = [axes]
        for idx, d_name in enumerate(plot_datasets):
            sub = df_sum[df_sum['Dataset'] == d_name]
            mods = sub['Model'].tolist()
            maes = sub['MAE'].tolist()
            stds = sub['MAE_std'].tolist()
            x = np.arange(len(mods))
            colors = ['#bdc3c7', '#95a5a6', '#7f8c8d', '#e67e22', '#3498db', '#9b59b6', '#2ecc71', '#1abc9c'][:len(mods)]

            axes[idx].bar(x, maes, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2, width=0.55)
            axes[idx].set_title(f'{d_name} - Authentic Test MAE', fontsize=12, fontweight='bold', pad=10)
            axes[idx].set_ylabel('MAE (ms) [Lower is Better]', fontsize=11)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(mods, rotation=30, ha='right', fontsize=8.5)
            axes[idx].grid(axis='y', linestyle='--', alpha=0.5)
            for i, (m, s) in enumerate(zip(maes, stds)):
                txt = f"{m:.1f}" if s == 0 else f"{m:.1f}±{s:.1f}"
                axes[idx].text(i, m + s + 1.2, txt, ha='center', fontweight='bold', fontsize=8.5)

        plt.tight_layout()
        fig.savefig(out_dir / "authentic_10graphs_comparison.png", dpi=300)
        fig.savefig(out_dir / "authentic_master_comparison.png", dpi=300)
        plt.close(fig)
        print("✅ Saved authentic_10graphs_comparison.png")

    # Figure 2: Actual vs Predicted Delay Scatter (10-Graphs)
    if plot_datasets:
        fig, axes = plt.subplots(1, len(plot_datasets), figsize=(14, 6))
        if len(plot_datasets) == 1:
            axes = [axes]
        for idx, d_name in enumerate(plot_datasets):
            if d_name in scatter_records:
                yt, yp, _ = scatter_records[d_name]
                max_v = max(450, float(np.percentile(yt, 98)) * 1.1)
                axes[idx].scatter(yt, yp, alpha=0.45, color='#2980b9', edgecolors='none', s=35)
                axes[idx].plot([0, max_v], [0, max_v], 'r--', linewidth=2, label='Ideal y = x')
                axes[idx].set_title(f'Authentic DeepNT+: {d_name}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Ground Truth RTT (ms)', fontsize=11)
                axes[idx].set_ylabel('Predicted RTT (ms)', fontsize=11)
                axes[idx].set_xlim(0, max_v)
                axes[idx].set_ylim(0, max_v)
                axes[idx].grid(True, linestyle='--', alpha=0.5)
                axes[idx].legend(fontsize=11)

        plt.tight_layout()
        fig.savefig(out_dir / "authentic_delay_scatter.png", dpi=300)
        plt.close(fig)
        print("✅ Saved authentic_delay_scatter.png")

    # Figure 3: Authentic Laplace Uncertainty Calibration
    calib_key = 'IPv4 (10-Graphs, New)' if 'IPv4 (10-Graphs, New)' in calibration_records else next(iter(calibration_records), None)
    if calib_key and calibration_records[calib_key][2]:
        fig, ax = plt.subplots(figsize=(7, 6))
        yt, yp, sigs = calibration_records[calib_key]
        z = np.log1p(np.array(yt))
        mu = np.log1p(np.clip(np.array(yp), 0, None))
        b = np.array(sigs) / np.sqrt(2.0)

        nominal = [50, 80, 90, 95]
        levels = [0.50, 0.80, 0.90, 0.95]
        emp = [np.mean(np.abs(z - mu) <= -b * np.log(1 - l)) * 100 for l in levels]

        ax.plot([40, 100], [40, 100], 'k--', linewidth=2, label='Perfect Calibration (Diagonal)')
        ax.plot(nominal, emp, 'o-', color='#27ae60', linewidth=2.5, markersize=8, label=f'DeepNT+ Laplace Coverage ({calib_key})')
        for nom, val in zip(nominal, emp):
            ax.annotate(f"{val:.1f}%", (nom, val), textcoords="offset points", xytext=(-10, 10), fontweight='bold', color='#27ae60')

        ax.set_title('Authentic Uncertainty Calibration (Laplace Likelihood)', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Nominal Confidence Level (%)', fontsize=12)
        ax.set_ylabel('Empirical Coverage (%)', fontsize=12)
        ax.set_xlim(45, 100)
        ax.set_ylim(40, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=11)

        plt.tight_layout()
        fig.savefig(out_dir / "authentic_uncertainty_calibration.png", dpi=300)
        plt.close(fig)
        print("✅ Saved authentic_uncertainty_calibration.png")

    print("\n" + "=" * 80)
    print("🎉 100% AUTHENTIC ZERO-LEAKAGE BENCHMARKS COMPLETED!")
    print("=" * 80)

if __name__ == '__main__':
    main()
