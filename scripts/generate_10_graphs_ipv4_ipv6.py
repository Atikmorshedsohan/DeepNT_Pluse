"""Generate 10 IPv4 Graphs and 10 IPv6 Graphs for DeepNT+ Network Tomography
Strictly enforces:
- 10 graphs per protocol (graph_01 to graph_10)
- Balanced split: 6 train / 2 val / 2 test
- Tomography matrix A contract: A.shape == (len(paths), len(edges)) and A.sum(axis=1) == path_length
- Complete feature tables: node_features, edge_features, paths_all, path_ground_truth
- Fully verifiable via deepnt_plus.data.load_dataset()
"""
import sys
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
SRC_IPV4 = BASE_DIR / "kaggle_staging" / "data" / "processed_15graphs_backup"
SRC_IPV6 = BASE_DIR / "kaggle_staging" / "data" / "ipv6"

OUT_IPV4 = BASE_DIR / "kaggle_staging" / "data" / "ipv4_10graphs"
OUT_IPV6 = BASE_DIR / "kaggle_staging" / "data" / "ipv6_10graphs"

# Also sync to DeepNT_Output if present
THESIS_DIR = Path(r"C:\Users\KINGDOM\Desktop\Thesis\DeepNT_Output\reconstructed_dataset")
SYNC_IPV4 = THESIS_DIR / "ipv4_10graphs"
SYNC_IPV6 = THESIS_DIR / "ipv6_10graphs"

SEED = 2026
OBSERVATION_RATE = 0.45
RTT_MIN_MS, RTT_MAX_MS = 0.05, 1000.0

SPLIT_10 = {
    'graph_01': 'train', 'graph_02': 'train', 'graph_03': 'train',
    'graph_04': 'train', 'graph_05': 'train', 'graph_06': 'train',
    'graph_07': 'val',   'graph_08': 'val',
    'graph_09': 'test',  'graph_10': 'test'
}

def build_protocol_10_graphs(name, paths_src, gt_src, graph_allocations, out_dir, sync_dir=None):
    print("\n" + "=" * 80)
    print(f"🚀 GENERATING 10 {name} GRAPHS (graph_01 to graph_10)")
    print("=" * 80)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (out_dir / "tomography").mkdir(parents=True, exist_ok=True)
    (out_dir / "tgn").mkdir(parents=True, exist_ok=True)
    
    # 1. Assign each path to one of the 10 graphs
    assigned_rows = []
    for gid in sorted(SPLIT_10.keys()):
        paths_chunk = graph_allocations[gid]
        for it in paths_chunk:
            it['target_graph_id'] = gid
            it['split'] = SPLIT_10[gid]
            assigned_rows.append(it)
            
    df_assigned = pd.DataFrame(assigned_rows)
    print(f"Total paths allocated to 10 graphs: {len(df_assigned):,}")
    print(df_assigned.groupby(['target_graph_id', 'split', 'monitor_id']).size().unstack(fill_value=0))
    
    # 2. Build NetworkX graph per graph_id
    graphs_nx = {}
    path_rows = []
    
    for gid in sorted(SPLIT_10.keys()):
        g_df = df_assigned[df_assigned['target_graph_id'] == gid].reset_index(drop=True)
        G = nx.Graph()
        items = []
        for _, r in g_df.iterrows():
            nodes = str(r['path_nodes']).split('|')
            edges = [tuple(e.split('-')) for e in str(r['path_edges']).split('|')]
            for a, b in edges:
                if not G.has_edge(a, b):
                    G.add_edge(a, b, edge_id=f"{a}-{b}", path_count=0)
                G[a][b]['path_count'] += 1
            items.append({
                'nodes': nodes, 'edges': edges,
                'rtt_ms': float(r['rtt_ms']), 'rtt_raw': float(r.get('rtt_raw', r['rtt_ms'])),
                'rtt_raw_unit': r.get('rtt_raw_unit', 'ms'),
                'monitor_id': r['monitor_id'],
                'measurement_time': r.get('measurement_time', ''),
                'measurement_method': r.get('measurement_method', 'scamper_trace'),
                'raw_id': r.get('raw_measurement_id', r.get('path_id', '')),
                'rtt_source': r.get('rtt_source', 'destination_rtt'),
                'quality_flag': r.get('quality_flag', 'ok'),
            })
            
        graphs_nx[gid] = G
        n_obs = max(1, int(round(OBSERVATION_RATE * len(items))))
        obs_idx = np.zeros(len(items), dtype=int)
        g_num = int(gid.split('_')[1])
        obs_idx[np.random.default_rng(SEED + g_num).choice(len(items), n_obs, replace=False)] = 1
        
        for k, it in enumerate(items):
            rtt_val = it['rtt_ms']
            outlier = [] if RTT_MIN_MS <= rtt_val <= RTT_MAX_MS else (['rtt_outlier_high'] if rtt_val > RTT_MAX_MS else ['rtt_outlier_low'])
            q_flag = ';'.join([f for f in str(it['quality_flag']).split(';') if f != 'ok'] + outlier) or 'ok'
            path_rows.append({
                'graph_id': gid, 'path_id': f"{gid}_p{k:05d}", 'split': SPLIT_10[gid],
                'source_node': it['nodes'][0], 'destination_node': it['nodes'][-1],
                'path_nodes': '|'.join(it['nodes']),
                'path_edges': '|'.join(f"{a}-{b}" for a, b in it['edges']),
                'path_length': len(it['edges']),
                'measurement_time': it['measurement_time'], 'monitor_id': it['monitor_id'],
                'measurement_method': it['measurement_method'], 'observed': int(obs_idx[k]),
                'raw_measurement_id': it['raw_id'], 'rtt_source': it['rtt_source'],
                'quality_flag': q_flag,
                'rtt_ms': rtt_val, 'rtt_raw': it['rtt_raw'], 'rtt_raw_unit': it['rtt_raw_unit'],
            })
            
    full_df = pd.DataFrame(path_rows)
    
    # 3. Compute Node and Edge Features
    print(f"\nComputing topological features for 10 {name} graphs...")
    node_rows = []
    for gid in sorted(SPLIT_10.keys()):
        G = graphs_nx[gid]
        n = G.number_of_nodes()
        deg = dict(G.degree())
        clus = nx.clustering(G)
        pr = nx.pagerank(G)
        core = nx.core_number(G)
        clo = nx.closeness_centrality(G) if n <= 3500 else {}
        bet = nx.betweenness_centrality(G, k=min(n, 200), seed=SEED)
        
        for node in G:
            node_rows.append({
                'graph_id': gid, 'node_id': node, 'interface_count': 1,
                'degree': deg[node], 'norm_degree': deg[node] / max(n - 1, 1),
                'clustering_coeff': clus[node], 'pagerank': pr[node],
                'closeness_centrality': clo.get(node, np.nan),
                'betweenness_centrality': bet[node], 'k_core': core[node],
                'asn': pd.NA, 'country': pd.NA, 'latitude': np.nan, 'longitude': np.nan
            })
    node_features_df = pd.DataFrame(node_rows)
    
    obs_counts = Counter()
    for r in full_df[full_df['observed'] == 1].itertuples():
        for e in r.path_edges.split('|'):
            obs_counts[(r.graph_id, e)] += 1
            
    edge_rows = []
    for gid in sorted(SPLIT_10.keys()):
        G = graphs_nx[gid]
        for _, _, d in sorted(G.edges(data=True), key=lambda x: x[2]['edge_id']):
            parts = d['edge_id'].split('-')
            edge_rows.append({
                'graph_id': gid, 'edge_id': d['edge_id'], 'src': parts[0], 'dst': parts[1],
                'itdk_link_id': '', 'itdk_link_ids': '',
                'capacity': np.nan, 'utilization': np.nan, 'base_delay': np.nan, 'loss_rate': np.nan,
                'path_count_all': d['path_count'],
                'observed_measurement_count': obs_counts[(gid, d['edge_id'])],
            })
    edge_features_df = pd.DataFrame(edge_rows)
    
    # 4. Build and Save Tomography A Matrices
    STRUCT_COLS = ['graph_id', 'path_id', 'split', 'source_node', 'destination_node',
                   'path_nodes', 'path_edges', 'path_length', 'measurement_time',
                   'monitor_id', 'measurement_method', 'observed', 'raw_measurement_id',
                   'rtt_source', 'quality_flag']
                   
    paths_all_df = full_df[STRUCT_COLS]
    gt_df = full_df[['path_id', 'graph_id', 'split', 'observed', 'rtt_ms', 'rtt_raw', 'rtt_raw_unit', 'rtt_source', 'quality_flag']].assign(
        packet_loss=np.nan, loss_flag='unavailable_no_probe_counts'
    )
    
    paths_obs_df = paths_all_df[paths_all_df['observed'] == 1].merge(
        gt_df[['path_id', 'rtt_ms', 'rtt_raw', 'rtt_raw_unit', 'packet_loss', 'loss_flag']], on='path_id'
    )
    paths_unobs_df = paths_all_df[paths_all_df['observed'] == 0]
    
    A_blocks, row_idx, col_idx = [], [], []
    for gid in sorted(SPLIT_10.keys()):
        prow = paths_all_df[paths_all_df['graph_id'] == gid].reset_index(drop=True)
        ecol = edge_features_df[edge_features_df['graph_id'] == gid].reset_index(drop=True)
        col_map = {e: i for i, e in enumerate(ecol['edge_id'])}
        
        r_i, c_i = [], []
        for i, pe in enumerate(prow['path_edges']):
            for e in pe.split('|'):
                r_i.append(i)
                c_i.append(col_map[e])
        A = sp.csr_matrix((np.ones(len(r_i), dtype=np.int8), (r_i, c_i)), shape=(len(prow), len(ecol)))
        assert (np.asarray(A.sum(axis=1)).ravel() == prow['path_length'].to_numpy()).all(), f"A check failed on {gid}"
        assert A.shape == (len(prow), len(ecol)), f"A shape failed on {gid}"
        
        sp.save_npz(out_dir / "tomography" / f"A_{gid}.npz", A)
        A_blocks.append(A)
        row_idx += [(gid, p, i) for i, p in enumerate(prow['path_id'])]
        col_idx += [(gid, e, i) for i, e in enumerate(ecol['edge_id'])]
        
    sp.save_npz(out_dir / "tomography" / "A_matrix.npz", sp.block_diag(A_blocks, format='csr'))
    pd.DataFrame(row_idx, columns=['graph_id', 'path_id', 'row']).to_csv(out_dir / "tomography" / "A_rows.csv", index=False)
    pd.DataFrame(col_idx, columns=['graph_id', 'edge_id', 'col']).to_csv(out_dir / "tomography" / "A_cols.csv", index=False)
    
    # Save CSV files
    paths_all_df.to_csv(out_dir / "paths_all.csv", index=False)
    paths_obs_df.to_csv(out_dir / "paths_observed.csv", index=False)
    paths_unobs_df.to_csv(out_dir / "paths_unobserved.csv", index=False)
    gt_df.to_csv(out_dir / "path_ground_truth.csv", index=False)
    node_features_df.to_csv(out_dir / "node_features.csv", index=False)
    edge_features_df.to_csv(out_dir / "edge_features.csv", index=False)
    
    # Save statistics
    stats_rows = []
    for gid in sorted(SPLIT_10.keys()):
        G = graphs_nx[gid]
        n_p = len(paths_all_df[paths_all_df['graph_id'] == gid])
        n_o = len(paths_obs_df[paths_obs_df['graph_id'] == gid])
        stats_rows.append({
            'graph_id': gid, 'split': SPLIT_10[gid],
            'nodes': G.number_of_nodes(), 'edges': G.number_of_edges(),
            'total_paths': n_p, 'observed_paths': n_o,
            'unobserved_paths': n_p - n_o
        })
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(out_dir / "graph_statistics.csv", index=False)
    
    print("\n✅ 10 Graphs Statistics Summary:")
    print(df_stats.to_string(index=False))
    
    # If sync dir provided, copy
    if sync_dir:
        sync_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copytree(out_dir, sync_dir, dirs_exist_ok=True)
        print(f"✅ Synced to: {sync_dir}")
        
    return df_stats


def main():
    t0 = time.time()
    
    # --- PART 1: IPv4 10 GRAPHS ---
    print("\n" + "=" * 80)
    print("1. LOADING & PARTITIONING IPv4 PATHS INTO 10 BALANCED GRAPHS")
    print("=" * 80)
    
    paths_v4 = pd.read_csv(SRC_IPV4 / "paths_all.csv")
    gt_v4 = pd.read_csv(SRC_IPV4 / "path_ground_truth.csv")
    merged_v4 = paths_v4.merge(gt_v4[['path_id', 'rtt_ms', 'rtt_raw', 'rtt_raw_unit', 'quality_flag']], on='path_id')
    
    # Group by monitor and sort chronologically/path_id
    monitors_v4 = sorted(merged_v4['monitor_id'].unique())
    print(f"IPv4 Monitors: {monitors_v4}")
    
    # We allocate paths across 10 graphs:
    # dac-bd (478 paths): 4 graphs (graph_01, graph_02, graph_03 [train], graph_07 [val])
    # abz2-uk (431 paths): 3 graphs (graph_04, graph_05 [train], graph_08 [val])
    # acc-gh (417 paths): 3 graphs (graph_06 [train], graph_09, graph_10 [test])
    allocations_v4 = {gid: [] for gid in SPLIT_10.keys()}
    
    dac_paths = merged_v4[merged_v4['monitor_id'] == 'dac-bd'].to_dict('records')
    dac_splits = np.array_split(dac_paths, 4)
    allocations_v4['graph_01'].extend(dac_splits[0])
    allocations_v4['graph_02'].extend(dac_splits[1])
    allocations_v4['graph_03'].extend(dac_splits[2])
    allocations_v4['graph_07'].extend(dac_splits[3])
    
    abz_paths = merged_v4[merged_v4['monitor_id'] == 'abz2-uk'].to_dict('records')
    abz_splits = np.array_split(abz_paths, 3)
    allocations_v4['graph_04'].extend(abz_splits[0])
    allocations_v4['graph_05'].extend(abz_splits[1])
    allocations_v4['graph_08'].extend(abz_splits[2])
    
    acc_paths = merged_v4[merged_v4['monitor_id'] == 'acc-gh'].to_dict('records')
    acc_splits = np.array_split(acc_paths, 3)
    allocations_v4['graph_06'].extend(acc_splits[0])
    allocations_v4['graph_09'].extend(acc_splits[1])
    allocations_v4['graph_10'].extend(acc_splits[2])
    
    build_protocol_10_graphs("IPv4", paths_v4, gt_v4, allocations_v4, OUT_IPV4, SYNC_IPV4)
    
    # --- PART 2: IPv6 10 GRAPHS ---
    print("\n" + "=" * 80)
    print("2. LOADING & PARTITIONING IPv6 PATHS INTO 10 BALANCED GRAPHS")
    print("=" * 80)
    
    paths_v6 = pd.read_csv(SRC_IPV6 / "paths_all.csv")
    gt_v6 = pd.read_csv(SRC_IPV6 / "path_ground_truth.csv")
    merged_v6 = paths_v6.merge(gt_v6[['path_id', 'rtt_ms', 'rtt_raw', 'rtt_raw_unit', 'quality_flag']], on='path_id')
    
    # 5 monitors, each 1,500 paths:
    # scq-es  -> graph_01, graph_02 (train)
    # prg-cz  -> graph_03, graph_04 (train)
    # cdg4-fr -> graph_05, graph_06 (train)
    # san-us  -> graph_07, graph_08 (val)
    # iad-us  -> graph_09, graph_10 (test)
    allocations_v6 = {gid: [] for gid in SPLIT_10.keys()}
    
    mapping_v6_monitors = {
        'scq-es':  ('graph_01', 'graph_02'),
        'prg-cz':  ('graph_03', 'graph_04'),
        'cdg4-fr': ('graph_05', 'graph_06'),
        'san-us':  ('graph_07', 'graph_08'),
        'iad-us':  ('graph_09', 'graph_10')
    }
    
    for mon, (g_a, g_b) in mapping_v6_monitors.items():
        mon_paths = merged_v6[merged_v6['monitor_id'] == mon].to_dict('records')
        p_a, p_b = np.array_split(mon_paths, 2)
        allocations_v6[g_a].extend(p_a)
        allocations_v6[g_b].extend(p_b)
        
    build_protocol_10_graphs("IPv6", paths_v6, gt_v6, allocations_v6, OUT_IPV6, SYNC_IPV6)
    
    # --- PART 3: TEST WITH NATIVE DeepNT+ LOADER ---
    print("\n" + "=" * 80)
    print("3. VERIFYING BOTH DATASETS WITH NATIVE DeepNT+ LOADER")
    print("=" * 80)
    from deepnt_plus.data import load_dataset, describe
    
    print("\n--- Verifying IPv4 10-Graph Dataset ---")
    graphs_v4, norm_v4 = load_dataset(OUT_IPV4, use_geo=False)
    desc_v4 = describe(graphs_v4)
    print(desc_v4.to_string(index=False))
    
    print("\n--- Verifying IPv6 10-Graph Dataset ---")
    graphs_v6, norm_v6 = load_dataset(OUT_IPV6, use_geo=False)
    desc_v6 = describe(graphs_v6)
    print(desc_v6.to_string(index=False))
    
    print("\n" + "=" * 80)
    print(f"🎉 SUCCESS! All 10 IPv4 Graphs and 10 IPv6 Graphs generated and verified in {time.time() - t0:.1f}s.")
    print("=" * 80)

if __name__ == '__main__':
    main()
