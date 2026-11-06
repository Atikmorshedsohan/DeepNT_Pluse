"""Load the processed dataset written by Dataset_Build.ipynb into per-graph tensors.

Model inputs and targets are kept apart: `GraphData.y` holds the true RTT of every path, but the model only
ever receives `y[ctx]` for the context (observed) paths, never `y[tgt]`.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

NODE_FEATURES = ['interface_count', 'degree', 'norm_degree', 'clustering_coeff', 'pagerank',
                 'closeness_centrality', 'betweenness_centrality', 'k_core']
LOG_COLUMNS = {'interface_count', 'degree', 'betweenness_centrality', 'k_core'}


@dataclass
class GraphData:
    gid: str
    split: str
    x: torch.Tensor                # [N, F] standardised structural node features
    node_ids: list
    edge_uv: torch.Tensor          # [E, 2] node indices of each undirected edge
    edge_feat: torch.Tensor        # [E, 1] log1p(path_count_all), standardised (structural)
    edge_index: torch.Tensor       # [2, 2E] both directions, for message passing
    A: torch.Tensor                # [M, E] tomography matrix
    Q: torch.Tensor                # [M, N] path-node incidence
    node_pad: torch.Tensor         # [M, L] ordered node indices (padded with 0)
    node_mask: torch.Tensor        # [M, L] True where the entry is a real node
    path_ids: list
    sources: list
    destinations: list
    t: torch.Tensor                # [M] hours since the first measurement in the dataset
    hops: torch.Tensor             # [M]
    y: torch.Tensor                # [M] TRUE RTT in ms (target side only)
    observed: torch.Tensor         # [M] bool: RTT visible to the model as context
    ok: torch.Tensor               # [M] bool: not flagged as an RTT outlier
    monitor: list
    loss: torch.Tensor             # [M] packet loss in [0, 1], NaN where unavailable

    @property
    def n_paths(self):
        return self.A.shape[0]


def _standardise(values, mean, std):
    return (values - mean) / (std if std > 1e-8 else 1.0)


def leave_monitor_out(paths, holdout):
    """Split for a cross-vantage-point test: graphs dominated by `holdout` are the test set, the last graph of each
    other monitor is validation, everything else trains."""
    dom = paths.groupby('graph_id')['monitor_id'].agg(lambda s: s.value_counts().idxmax())
    if holdout not in set(dom):
        raise ValueError(f"no graph is dominated by monitor {holdout!r}; monitors: {sorted(set(dom))}")
    rest = [g for g in dom.index if dom[g] != holdout]
    val = {max(g for g in rest if dom[g] == m) for m in sorted({dom[g] for g in rest})}
    return {g: 'test' if dom[g] == holdout else 'val' if g in val else 'train' for g in dom.index}


def load_dataset(processed_dir, holdout_monitor=None):
    """Returns ({graph_id: GraphData}, normalisation dict). All statistics come from the train graphs only.
    holdout_monitor: optional Ark monitor id (e.g. 'dac-bd') to run a leave-one-vantage-point-out split instead of
    the split stored in the dataset files."""
    proc = Path(processed_dir)
    nodes = pd.read_csv(proc / 'node_features.csv', dtype={'graph_id': str, 'node_id': str})
    edges = pd.read_csv(proc / 'edge_features.csv', dtype={'graph_id': str, 'edge_id': str, 'src': str, 'dst': str})
    paths = pd.read_csv(proc / 'paths_all.csv', dtype={'graph_id': str, 'path_id': str}, keep_default_na=False)
    truth = pd.read_csv(proc / 'path_ground_truth.csv', dtype={'path_id': str}, keep_default_na=False).set_index('path_id')

    t_all = pd.to_datetime(paths['measurement_time'].replace('', np.nan), errors='coerce', utc=True)
    t0 = t_all.min()
    paths['t_hours'] = (t_all - t0).dt.total_seconds() / 3600.0

    split_of = paths.groupby('graph_id')['split'].first().to_dict()
    if holdout_monitor is not None:
        split_of = leave_monitor_out(paths, holdout_monitor)
    graph_ids = sorted(split_of)

    node_frame = nodes.copy()
    for c in NODE_FEATURES:
        node_frame[c] = pd.to_numeric(node_frame[c], errors='coerce')
        if c in LOG_COLUMNS:
            node_frame[c] = np.log1p(node_frame[c].clip(lower=0))
    train_nodes = node_frame[node_frame['graph_id'].map(split_of) == 'train']
    node_stats = {c: (float(train_nodes[c].mean()), float(train_nodes[c].std() or 1.0)) for c in NODE_FEATURES}
    edge_log = np.log1p(edges['path_count_all'].astype(float))
    train_edges = edge_log[edges['graph_id'].map(split_of) == 'train']
    edge_stats = (float(train_edges.mean()), float(train_edges.std() or 1.0))

    has_geo = 'geo_distance_km' in edges.columns and edges['geo_distance_km'].notna().any()
    if has_geo:
        geo_log = np.log1p(edges['geo_distance_km'].astype(float).clip(lower=0))
        train_geo = geo_log[edges['graph_id'].map(split_of) == 'train']
        geo_stats = (float(train_geo.mean()), float(train_geo.std() or 1.0))

    graphs = {}
    for gid in graph_ids:
        nf = node_frame[node_frame['graph_id'] == gid].reset_index(drop=True)
        ef = edges[edges['graph_id'] == gid].reset_index(drop=True)
        pf = paths[paths['graph_id'] == gid].reset_index(drop=True)
        gt = truth.loc[pf['path_id']]
        idx = {n: i for i, n in enumerate(nf['node_id'])}

        feats = np.stack([_standardise(nf[c].fillna(node_stats[c][0]).to_numpy(float), *node_stats[c])
                          for c in NODE_FEATURES], axis=1)
        uv = np.array([[idx[a], idx[b]] for a, b in zip(ef['src'], ef['dst'])], dtype=np.int64)
        f_count = _standardise(np.log1p(ef['path_count_all'].astype(float)).to_numpy(), *edge_stats)
        if has_geo:
            f_geo = _standardise(np.log1p(ef['geo_distance_km'].astype(float).clip(lower=0)).to_numpy(), *geo_stats)
            e_feat = np.stack([f_count, f_geo], axis=1)
        else:
            e_feat = f_count[:, None]
        A = sp.load_npz(proc / 'tomography' / f'A_{gid}.npz').toarray().astype(np.float32)
        assert A.shape == (len(pf), len(ef)), f"{gid}: A shape {A.shape} != paths x edges"

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
        assert np.allclose(A.sum(1), hops), f"{gid}: A row sums differ from path_length"

        t = pf['t_hours'].fillna(pf['t_hours'].median()).fillna(0.0).to_numpy(float)
        loss = pd.to_numeric(gt['packet_loss'], errors='coerce').to_numpy(float)
        graphs[gid] = GraphData(
            gid=gid, split=split_of[gid], x=torch.tensor(feats, dtype=torch.float32), node_ids=list(nf['node_id']),
            edge_uv=torch.tensor(uv), edge_feat=torch.tensor(e_feat, dtype=torch.float32),
            edge_index=torch.tensor(np.concatenate([uv.T, uv.T[::-1]], axis=1)),
            A=torch.tensor(A), Q=torch.tensor(Q), node_pad=torch.tensor(pad), node_mask=torch.tensor(mask),
            path_ids=list(pf['path_id']), sources=list(pf['source_node']), destinations=list(pf['destination_node']),
            t=torch.tensor(t, dtype=torch.float32), hops=torch.tensor(hops, dtype=torch.float32),
            y=torch.tensor(pd.to_numeric(gt['rtt_ms']).to_numpy(float), dtype=torch.float32),
            observed=torch.tensor(pf['observed'].to_numpy(int) == 1),
            ok=torch.tensor(~gt['quality_flag'].astype(str).str.contains('rtt_outlier').to_numpy()),
            monitor=list(pf['monitor_id']), loss=torch.tensor(loss, dtype=torch.float32))

    tr = [g for g in graphs.values() if g.split == 'train']
    y = torch.cat([g.y[g.ok] for g in tr])
    hops = torch.cat([g.hops[g.ok] for g in tr])
    log_y = torch.log1p(y)
    rate = torch.log1p(y / hops)
    norm = {
        'fit_on': 'train graphs only',
        'log_rtt_mean': float(log_y.mean()), 'log_rtt_std': float(log_y.std()),
        'rate_mean': float(rate.mean()), 'rate_std': float(rate.std()),
        'prior_edge_ms': float((y / hops).median()),
        'node_stats': node_stats, 'edge_stats': edge_stats,
        'edge_dim': 2 if has_geo else 1,
    }
    return graphs, norm


def describe(graphs):
    rows = []
    for g in graphs.values():
        rows.append({'graph': g.gid, 'split': g.split, 'nodes': len(g.node_ids), 'edges': g.A.shape[1],
                     'paths': g.n_paths, 'observed': int(g.observed.sum()), 'hidden': int((~g.observed).sum()),
                     'outliers': int((~g.ok).sum()), 'median_rtt_ms': float(g.y[g.ok].median())})
    return pd.DataFrame(rows)
