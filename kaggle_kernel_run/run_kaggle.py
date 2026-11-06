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

# 2. Authentic Graph Loader (Strictly without geo_distance_km)
def load_authentic_graphs(proc_dir):
    proc = Path(proc_dir)
    nodes = pd.read_csv(proc / 'node_features.csv', dtype={'graph_id': str, 'node_id': str})
    edges = pd.read_csv(proc / 'edge_features.csv', dtype={'graph_id': str, 'edge_id': str, 'src': str, 'dst': str})
    paths = pd.read_csv(proc / 'paths_all.csv', dtype={'graph_id': str, 'path_id': str}, keep_default_na=False)
    truth = pd.read_csv(proc / 'path_ground_truth.csv', dtype={'path_id': str}, keep_default_na=False).set_index('path_id')

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

    graphs = {}
    for gid in graph_ids:
        nf = node_frame[node_frame['graph_id'] == gid].reset_index(drop=True)
        ef = edges[edges['graph_id'] == gid].reset_index(drop=True)
        pf = paths[paths['graph_id'] == gid].reset_index(drop=True)
        idx = {n: i for i, n in enumerate(nf['node_id'])}

        feats = np.stack([data._standardise(nf[c].fillna(node_stats[c][0]).to_numpy(float), *node_stats[c])
                          for c in data.NODE_FEATURES], axis=1)
        uv = np.array([[idx[a], idx[b]] for a, b in zip(ef['src'], ef['dst'])], dtype=np.int64)

        # Single structural edge feature: log1p(path_count_all)
        feat_count = data._standardise(np.log1p(ef['path_count_all'].astype(float)).to_numpy(), *edge_stats)
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
        t = np.zeros(len(pf), dtype=np.float32)

        gt = truth.loc[pf['path_id']]
        y = torch.tensor(gt['rtt_ms'].to_numpy(float), dtype=torch.float32)
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

    train_y = torch.cat([g.y[g.ok & g.observed] for g in graphs.values() if g.split == 'train'])
    norm = {
        'log_rtt_mean': float(torch.log1p(train_y).mean()),
        'log_rtt_std': float(torch.log1p(train_y).std() or 1.0),
        'rate_mean': 1.5,
        'rate_std': 1.0,
        'prior_edge_ms': float(torch.median(train_y / 14.0).clamp(min=1.0))
    }
    return graphs, norm

# 3. Authentic Model Architecture
class DeepNTPlusAuthentic(nn.Module):
    def __init__(self, norm, edge_dim=1, d=64, gnn_kind='split', tomography=True, convex=True):
        super().__init__()
        self.d = d
        self.tomography = tomography
        self.convex = convex
        for k in ('log_rtt_mean', 'log_rtt_std', 'rate_mean', 'rate_std', 'prior_edge_ms'):
            self.register_buffer(k, torch.tensor(float(norm[k])))

        self.gnn = layers.GNN(gnn_kind, 8 + 2, d, layers=2, dropout=0.1)
        self.edge_mlp = nn.Sequential(nn.Linear(2 * d + edge_dim + 1, d), nn.ReLU(), nn.Linear(d, 1)) if tomography else None
        self.bias_raw = nn.Parameter(torch.zeros(())) if tomography else None
        self.path_enc = layers.PathEncoder(d, heads=4, layers=1, dropout=0.1)
        self.res_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.res_head[-1].weight)
        nn.init.zeros_(self.res_head[-1].bias)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.e ** 10 - 1)))
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.var_head = nn.Sequential(nn.Linear(d + 1, d), nn.ReLU(), nn.Linear(d, 1))

    def z_of(self, y_ms):
        return (torch.log1p(y_ms) - self.log_rtt_mean) / self.log_rtt_std

    def forward(self, g, ctx, tgt, y_ctx):
        A, N = g.A, g.x.size(0)
        n_ctx, n_t = ctx.numel(), tgt.numel()

        cnt, mean_r = torch.zeros(N, device=g.x.device), torch.zeros(N, device=g.x.device)
        if n_ctx:
            Qc = g.Q[ctx]
            cnt = Qc.sum(0)
            rates = (torch.log1p(y_ctx / g.hops[ctx].clamp(min=1)) - self.rate_mean) / self.rate_std
            mean_r = (Qc.T @ rates) / cnt.clamp(min=1)
        
        h = self.gnn(torch.cat([g.x, torch.log1p(cnt).unsqueeze(-1), mean_r.unsqueeze(-1)], -1), g.edge_index)
        cov_e = A[ctx].sum(0) if n_ctx else torch.zeros(A.size(1), device=A.device)
        Z = self.path_enc(h, g.node_pad, g.node_mask)
        Zt = Z[tgt]
        cov_p = ((A[tgt] * (cov_e > 0).float()).sum(1) / g.hops[tgt]) if n_ctx else torch.zeros(n_t, device=A.device)

        if not self.tomography:
            # Standard GNN baseline prediction (e.g. GAT / GraphSAGE)
            mu_z = self.res_head(Zt).squeeze(-1)
            y_hat = torch.expm1((mu_z * self.log_rtt_std + self.log_rtt_mean).clamp(max=12.0))
            return {'y_hat': y_hat, 'mu_z': mu_z, 'logvar': None, 'y_add_ctx': torch.zeros(0, device=A.device)}

        hu, hv = h[g.edge_uv[:, 0]], h[g.edge_uv[:, 1]]
        e_in = torch.cat([hu + hv, (hu - hv).abs(), g.edge_feat, torch.log1p(cov_e).unsqueeze(-1)], -1)
        x_e = F.softplus(self.edge_mlp(e_in).squeeze(-1)) * self.prior_edge_ms
        b = F.softplus(self.bias_raw) * self.prior_edge_ms

        y_add = A[tgt] @ x_e + b
        base = torch.log1p(y_add)
        y_g = torch.expm1((base + self.log_rtt_std * self.res_head(Zt).squeeze(-1)).clamp(max=12.0))

        y_c = y_g
        if self.convex and n_ctx:
            Ac = torch.cat([A[ctx], torch.ones(n_ctx, 1, device=A.device)], 1)
            ctx_rate = (y_ctx / g.hops[ctx].clamp(min=1.0)).median().clamp(min=0.1)
            scale = (ctx_rate / self.prior_edge_ms).clamp(0.2, 5.0)
            x0 = torch.cat([x_e * scale, b.view(1)])
            alpha = F.softplus(self.alpha_raw) + 1e-3
            resid = y_ctx - Ac @ x0
            K = Ac @ Ac.T + alpha * torch.eye(n_ctx, device=A.device)
            x_ref = (x0 + Ac.T @ torch.linalg.solve(K, resid)).clamp(min=0)
            y_c = torch.cat([A[tgt], torch.ones(n_t, 1, device=A.device)], 1) @ x_ref
            rel = (resid.abs() / y_ctx.clamp(min=1)).mean()
            gate_in = torch.stack([cov_p, torch.full((n_t,), math.log1p(n_ctx) / 5, device=A.device), rel.expand(n_t)], -1)
            g_res = torch.sigmoid(self.gate(gate_in).squeeze(-1))
            y_hat = y_c + g_res * (y_g - y_c)
        else:
            y_hat = y_g

        logvar = self.var_head(torch.cat([Zt, cov_p.unsqueeze(-1)], -1)).squeeze(-1).clamp(-6, 3)
        return {'y_hat': y_hat, 'mu_z': self.z_of(y_hat.clamp(min=0)), 'logvar': logvar,
                'y_add_ctx': (A[ctx] @ x_e + b) if n_ctx else torch.zeros(0, device=A.device)}

# 4. Authentic Classical & Causal Convex Tomography Baselines
def evaluate_convex(graphs, causal=False, n_blocks=6):
    preds, trues = [], []
    for gid, g in graphs.items():
        if g.split != 'test':
            continue
        ctx = torch.nonzero(g.observed & g.ok).squeeze(-1)
        tgt = torch.nonzero(~g.observed & g.ok).squeeze(-1)
        A = g.A
        
        if not causal:
            # 100% full context (classical)
            y_ctx = g.y[ctx]
            n_ctx, n_t = ctx.numel(), tgt.numel()
            Ac = torch.cat([A[ctx], torch.ones(n_ctx, 1)], 1)
            K = Ac @ Ac.T + 1e-2 * torch.eye(n_ctx)
            x_sol = (Ac.T @ torch.linalg.solve(K, y_ctx)).clamp(min=0)
            y_hat = torch.cat([A[tgt], torch.ones(n_t, 1)], 1) @ x_sol
            preds.extend(y_hat.numpy())
            trues.extend(g.y[tgt].numpy())
        else:
            # Causal block evaluation
            order = tgt[torch.argsort(g.t[tgt])]
            for chunk in torch.tensor_split(order, n_blocks):
                if not chunk.numel():
                    continue
                c_sub = ctx[g.t[ctx] < g.t[chunk].min()]
                trues.extend(g.y[chunk].numpy())
                if c_sub.numel() == 0:
                    med = float(g.y[ctx].median())
                    preds.extend(np.full(chunk.numel(), med))
                else:
                    y_sub = g.y[c_sub]
                    Ac = torch.cat([A[c_sub], torch.ones(c_sub.numel(), 1)], 1)
                    K = Ac @ Ac.T + 1e-2 * torch.eye(c_sub.numel())
                    x_sol = (Ac.T @ torch.linalg.solve(K, y_sub)).clamp(min=0)
                    y_chunk = torch.cat([A[chunk], torch.ones(chunk.numel(), 1)], 1) @ x_sol
                    preds.extend(y_chunk.numpy())

    y_t, y_p = np.array(trues), np.array(preds)
    abs_err = np.abs(y_t - y_p)
    ss_tot = np.sum((y_t - np.mean(y_t))**2)
    return {
        'MAE': float(np.mean(abs_err)), 'RMSE': float(np.sqrt(np.mean(abs_err**2))),
        'R2': float(1.0 - np.sum(abs_err**2) / (ss_tot if ss_tot > 0 else 1.0)),
        'MdAPE': float(np.median(abs_err / np.maximum(y_t, 1e-5)) * 100)
    }

# 5. Non-Graph Baselines (Median, Hop-count)
def evaluate_simple_baselines(graphs):
    train_y = np.concatenate([g.y[g.ok & g.observed].numpy() for g in graphs.values() if g.split == 'train'])
    test_y = np.concatenate([g.y[~g.observed & g.ok].numpy() for g in graphs.values() if g.split == 'test'])
    test_hops = np.concatenate([g.hops[~g.observed & g.ok].numpy() for g in graphs.values() if g.split == 'test'])
    
    med = float(np.median(train_y))
    mae_med = float(np.mean(np.abs(test_y - med)))
    ss_tot = np.sum((test_y - np.mean(test_y))**2)
    r2_med = float(1.0 - np.sum((test_y - med)**2) / ss_tot)

    # Hop-count
    train_hops = np.concatenate([g.hops[g.ok & g.observed].numpy() for g in graphs.values() if g.split == 'train'])
    slope = float(np.linalg.lstsq(train_hops[:, None], train_y, rcond=None)[0][0])
    pred_hop = np.clip(test_hops * slope, 0, None)
    mae_hop = float(np.mean(np.abs(test_y - pred_hop)))
    r2_hop = float(1.0 - np.sum((test_y - pred_hop)**2) / ss_tot)

    return {
        'Median': {'MAE': mae_med, 'R2': r2_med},
        'Hop-count': {'MAE': mae_hop, 'R2': r2_hop}
    }

# 6. Authentic Training & Evaluation Engine
def train_and_eval_authentic(graphs, norm, model_kind='deepnt_plus', epochs=35, seeds=[0, 1, 2]):
    seed_metrics = []
    final_preds, final_trues, final_sigmas = [], [], []

    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        
        if model_kind == 'deepnt_plus':
            model = DeepNTPlusAuthentic(norm, edge_dim=1, d=64, gnn_kind='split', tomography=True, convex=True).to(device)
        elif model_kind == 'deepnt_static':
            model = DeepNTPlusAuthentic(norm, edge_dim=1, d=64, gnn_kind='split', tomography=True, convex=False).to(device)
        elif model_kind == 'gat':
            model = DeepNTPlusAuthentic(norm, edge_dim=1, d=64, gnn_kind='gat', tomography=False, convex=False).to(device)
        elif model_kind == 'graphsage':
            model = DeepNTPlusAuthentic(norm, edge_dim=1, d=64, gnn_kind='sage', tomography=False, convex=False).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        train_ids = [k for k, g in graphs.items() if g.split == 'train']
        val_ids = [k for k, g in graphs.items() if g.split == 'val']
        test_ids = [k for k, g in graphs.items() if g.split == 'test']

        best_val, best_state = float('inf'), None
        for epoch in range(1, epochs + 1):
            model.train()
            # 6 episodes per graph per epoch
            for _ in range(6):
                for gid in rng.permutation(train_ids):
                    g = graphs[gid]
                    pool = np.flatnonzero(g.ok.numpy())
                    rng.shuffle(pool)
                    # Centered context fraction (0.25, 0.55)
                    n_ctx = max(2, int(round(rng.uniform(0.25, 0.55) * len(pool))))
                    ctx = torch.tensor(np.sort(pool[:n_ctx]), device=device)
                    tgt = torch.tensor(np.sort(pool[n_ctx:]), device=device)

                    g_dev = data.GraphData(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.__dict__.items()})
                    out = model(g_dev, ctx, tgt, g_dev.y[ctx])
                    z, mu = model.z_of(g_dev.y[tgt]), out['mu_z']
                    
                    if out['logvar'] is not None:
                        lv = out['logvar']
                        loss = (torch.exp(-lv) * torch.abs(z - mu) + lv).mean() + 0.5 * torch.abs(z - mu).mean()
                    else:
                        loss = torch.abs(z - mu).mean()

                    if model.tomography and ctx.numel():
                        loss += 0.5 * torch.abs(model.z_of(out['y_add_ctx']) - model.z_of(g_dev.y[ctx])).mean()

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            sched.step()

            # Multi-mask smoothed validation (3 masks)
            model.eval()
            with torch.no_grad():
                val_maes = []
                for gid in val_ids:
                    g = graphs[gid]
                    ctx = torch.nonzero(g.observed & g.ok).squeeze(-1).to(device)
                    tgt = torch.nonzero(~g.observed & g.ok).squeeze(-1).to(device)
                    g_dev = data.GraphData(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.__dict__.items()})
                    out = model(g_dev, ctx, tgt, g_dev.y[ctx])
                    val_maes.append((out['y_hat'] - g_dev.y[tgt]).abs().mean().item())

                    v_pool = np.flatnonzero(g.ok.numpy())
                    val_rng = np.random.default_rng(3000 + epoch)
                    for _ in range(2):
                        val_rng.shuffle(v_pool)
                        v_n_ctx = max(2, int(round(val_rng.uniform(0.25, 0.55) * len(v_pool))))
                        v_ctx = torch.tensor(np.sort(v_pool[:v_n_ctx]), device=device)
                        v_tgt = torch.tensor(np.sort(v_pool[v_n_ctx:]), device=device)
                        v_out = model(g_dev, v_ctx, v_tgt, g_dev.y[v_ctx])
                        val_maes.append((v_out['y_hat'] - g_dev.y[v_tgt]).abs().mean().item())

                mean_val = float(np.mean(val_maes))
                if mean_val < best_val:
                    best_val = mean_val
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate best model on test set
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            s_preds, s_trues, s_sigmas = [], [], []
            for gid in test_ids:
                g = graphs[gid]
                ctx = torch.nonzero(g.observed & g.ok).squeeze(-1).to(device)
                tgt = torch.nonzero(~g.observed & g.ok).squeeze(-1).to(device)
                g_dev = data.GraphData(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.__dict__.items()})
                out = model(g_dev, ctx, tgt, g_dev.y[ctx])
                s_preds.extend(out['y_hat'].cpu().numpy())
                s_trues.extend(g_dev.y[tgt].cpu().numpy())
                
                if out['logvar'] is not None:
                    # True Laplace std: sigma = sqrt(2) * b = sqrt(2) * exp(lv)
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
                cov90 = float(np.mean(np.abs(z - mu) <= -b * np.log(0.10)))
                ece = float(np.mean([abs(np.mean(np.abs(z - mu) <= -b * np.log(1 - l)) - l) for l in [0.5, 0.8, 0.9, 0.95]]))

            seed_metrics.append({'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MdAPE': mdape, 'Cov90': cov90, 'ECE': ece})
            if seed == 0:
                final_preds, final_trues, final_sigmas = s_preds, s_trues, s_sigmas

    df_m = pd.DataFrame(seed_metrics)
    return {
        'MAE_mean': df_m['MAE'].mean(), 'MAE_std': df_m['MAE'].std(),
        'RMSE_mean': df_m['RMSE'].mean(), 'RMSE_std': df_m['RMSE'].std(),
        'R2_mean': df_m['R2'].mean(), 'R2_std': df_m['R2'].std(),
        'MdAPE_mean': df_m['MdAPE'].mean(),
        'Cov90_mean': df_m['Cov90'].mean(),
        'ECE_mean': df_m['ECE'].mean(),
        'trues': final_trues, 'preds': final_preds, 'sigmas': final_sigmas
    }

# 7. Execute Authentic Benchmark Suite
datasets = [
    ('IPv4 (run_03, Dense)', found_root / 'data' / 'processed'),
    ('IPv4 (run_02, 15-Graphs)', found_root / 'data' / 'processed_15graphs_backup'),
    ('IPv6 (Global Ark)', found_root / 'data' / 'ipv6')
]

summary_rows = []
scatter_records = {}
calibration_records = {}

for name, path in datasets:
    print(f"\n{'='*75}")
    print(f"🌟 EVALUATING AUTHENTIC (LEAKAGE-FREE) DATASET: {name}")
    print(f"{'='*75}")

    graphs, norm = load_authentic_graphs(path)

    # 1. Non-Graph Baselines
    simples = evaluate_simple_baselines(graphs)
    for s_name, s_res in simples.items():
        print(f"  [{s_name:18s}] MAE: {s_res['MAE']:.2f} ms | R2: {s_res['R2']*100:.1f}%")
        summary_rows.append({
            'Dataset': name, 'Model': s_name, 'Family': 'Non-Graph',
            'MAE': s_res['MAE'], 'MAE_std': 0.0, 'R2': s_res['R2']*100, 'Cov90': np.nan, 'ECE': np.nan
        })

    # 2. Classical Convex Tomography
    conv_res = evaluate_convex(graphs, causal=False)
    print(f"  [{'Convex (Classical)':18s}] MAE: {conv_res['MAE']:.2f} ms | R2: {conv_res['R2']*100:.1f}% | RMSE: {conv_res['RMSE']:.2f} ms")
    summary_rows.append({
        'Dataset': name, 'Model': 'Convex (Classical)', 'Family': 'Analytical Tomography',
        'MAE': conv_res['MAE'], 'MAE_std': 0.0, 'R2': conv_res['R2']*100, 'Cov90': np.nan, 'ECE': np.nan
    })

    # 3. Standard GNNs (GAT, GraphSAGE)
    for gnn_type, display in [('gat', 'GAT'), ('graphsage', 'GraphSAGE')]:
        t0 = time.time()
        res = train_and_eval_authentic(graphs, norm, model_kind=gnn_type, epochs=35, seeds=[0, 1, 2])
        print(f"  [{display:18s}] MAE: {res['MAE_mean']:.2f} ± {res['MAE_std']:.2f} ms | R2: {res['R2_mean']*100:.1f}% ({time.time()-t0:.1f}s)")
        summary_rows.append({
            'Dataset': name, 'Model': display, 'Family': 'Standard GNN',
            'MAE': res['MAE_mean'], 'MAE_std': res['MAE_std'], 'R2': res['R2_mean']*100, 'Cov90': np.nan, 'ECE': np.nan
        })

    # 4. DeepNT (Static, Tomography only)
    t0 = time.time()
    res_static = train_and_eval_authentic(graphs, norm, model_kind='deepnt_static', epochs=35, seeds=[0, 1, 2])
    print(f"  [{'DeepNT (Static)':18s}] MAE: {res_static['MAE_mean']:.2f} ± {res_static['MAE_std']:.2f} ms | R2: {res_static['R2_mean']*100:.1f}% ({time.time()-t0:.1f}s)")
    summary_rows.append({
        'Dataset': name, 'Model': 'DeepNT (Static)', 'Family': 'Neural Tomography',
        'MAE': res_static['MAE_mean'], 'MAE_std': res_static['MAE_std'], 'R2': res_static['R2_mean']*100,
        'Cov90': res_static['Cov90_mean']*100, 'ECE': res_static['ECE_mean']
    })

    # 5. DeepNT+ (Full Improved, Authentic)
    t0 = time.time()
    res_full = train_and_eval_authentic(graphs, norm, model_kind='deepnt_plus', epochs=35, seeds=[0, 1, 2])
    print(f"  [{'DeepNT+ (Full, Ours)':18s}] MAE: {res_full['MAE_mean']:.2f} ± {res_full['MAE_std']:.2f} ms | R2: {res_full['R2_mean']*100:.1f}% | Cov90: {res_full['Cov90_mean']*100:.1f}% | ECE: {res_full['ECE_mean']:.3f} ({time.time()-t0:.1f}s) 🥇")
    summary_rows.append({
        'Dataset': name, 'Model': 'DeepNT+ (Full, Ours)', 'Family': 'Hybrid Tomography',
        'MAE': res_full['MAE_mean'], 'MAE_std': res_full['MAE_std'], 'R2': res_full['R2_mean']*100,
        'Cov90': res_full['Cov90_mean']*100, 'ECE': res_full['ECE_mean']
    })

    if name.startswith('IPv4 (run_03'):
        scatter_records['run_03'] = (res_full['trues'], res_full['preds'], conv_res['MAE'])
        calibration_records['run_03'] = (res_full['trues'], res_full['preds'], res_full['sigmas'])
    if name.startswith('IPv4 (run_02'):
        scatter_records['run_02'] = (res_full['trues'], res_full['preds'], conv_res['MAE'])
        calibration_records['run_02'] = (res_full['trues'], res_full['preds'], res_full['sigmas'])

out_dir = Path("/kaggle/working")
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

# Figure 1: Authentic Master Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
for idx, d_name in enumerate(['IPv4 (run_03, Dense)', 'IPv4 (run_02, 15-Graphs)']):
    sub = df_sum[df_sum['Dataset'] == d_name]
    mods = sub['Model'].tolist()
    maes = sub['MAE'].tolist()
    stds = sub['MAE_std'].tolist()
    x = np.arange(len(mods))
    colors = ['#bdc3c7', '#95a5a6', '#7f8c8d', '#e67e22', '#3498db', '#9b59b6', '#2ecc71'][:len(mods)]

    axes[idx].bar(x, maes, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2, width=0.55)
    axes[idx].set_title(f'{d_name} - Authentic Test MAE', fontsize=13, fontweight='bold', pad=10)
    axes[idx].set_ylabel('MAE (ms) [Lower is Better]', fontsize=11)
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(mods, rotation=25, ha='right', fontsize=9.5)
    axes[idx].grid(axis='y', linestyle='--', alpha=0.5)
    for i, (m, s) in enumerate(zip(maes, stds)):
        txt = f"{m:.1f}" if s == 0 else f"{m:.1f}±{s:.1f}"
        axes[idx].text(i, m + s + 1.2, txt, ha='center', fontweight='bold', fontsize=9.5)

plt.tight_layout()
fig.savefig(out_dir / "authentic_master_comparison.png", dpi=300)
plt.close(fig)
print("✅ Saved authentic_master_comparison.png")

# Figure 2: Actual vs Predicted Delay Scatter
if 'run_03' in scatter_records:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (k_name, title) in enumerate([('run_03', 'IPv4 Dense (run_03)'), ('run_02', 'IPv4 Balanced 15-Graphs (run_02)')]):
        if k_name in scatter_records:
            yt, yp, _ = scatter_records[k_name]
            axes[idx].scatter(yt, yp, alpha=0.45, color='#2980b9', edgecolors='none', s=35)
            axes[idx].plot([0, 450], [0, 450], 'r--', linewidth=2, label='Ideal y = x')
            axes[idx].set_title(f'Authentic DeepNT+: {title}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Ground Truth RTT (ms)', fontsize=11)
            axes[idx].set_ylabel('Predicted RTT (ms)', fontsize=11)
            axes[idx].set_xlim(0, 450)
            axes[idx].set_ylim(0, 450)
            axes[idx].grid(True, linestyle='--', alpha=0.5)
            axes[idx].legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(out_dir / "authentic_delay_scatter.png", dpi=300)
    plt.close(fig)
    print("✅ Saved authentic_delay_scatter.png")

# Figure 3: Authentic Laplace Uncertainty Calibration
if 'run_02' in calibration_records:
    fig, ax = plt.subplots(figsize=(7, 6))
    yt, yp, sigs = calibration_records['run_02']
    z = np.log1p(np.array(yt))
    mu = np.log1p(np.clip(np.array(yp), 0, None))
    b = np.array(sigs) / np.sqrt(2.0)

    nominal = [50, 80, 90, 95]
    levels = [0.50, 0.80, 0.90, 0.95]
    emp = [np.mean(np.abs(z - mu) <= -b * np.log(1 - l)) * 100 for l in levels]

    ax.plot([40, 100], [40, 100], 'k--', linewidth=2, label='Perfect Calibration (Diagonal)')
    ax.plot(nominal, emp, 'o-', color='#27ae60', linewidth=2.5, markersize=8, label='DeepNT+ Authentic Laplace Coverage')
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
