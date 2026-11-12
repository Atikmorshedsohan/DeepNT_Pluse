"""GeoLite2 IPv4/IPv6 geolocation lookup and Haversine edge-distance utilities.

Resolves IPv4 ITDK nodes (via CAIDA supervised trace IP alignment) and IPv6
router IPs against MaxMind GeoLite2-City CSV tables, populates ``country``,
``latitude``, and ``longitude`` on nodes, and adds ``geo_distance_km`` to
``edge_features.csv``.
"""
from __future__ import annotations

import csv
import ipaddress
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_GEOLITE_DIR = Path(r"C:\Users\KINGDOM\Desktop\Thesis\DataSet\GeoLite2-City-CSV_20261110")
DEFAULT_SUPERVISED_CSV = Path(r"C:\Users\KINGDOM\Desktop\Thesis\DeepNT_Output\real_rtt\caida_real_rtt_supervised.csv")

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle Haversine distance in kilometers, or NaN if either endpoint lacks coordinates."""
    if any(v is None or pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (float(lat1), float(lon1), float(lat2), float(lon2)))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def load_locations_map(geolite_dir: Path = DEFAULT_GEOLITE_DIR) -> Dict[str, str]:
    """Load geoname_id -> country_iso_code from GeoLite2-City-Locations-en.csv."""
    loc_path = Path(geolite_dir) / "GeoLite2-City-Locations-en.csv"
    loc_map: Dict[str, str] = {}
    with open(loc_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = row.get("geoname_id", "")
            cc = row.get("country_iso_code", "")
            if gid and cc:
                loc_map[gid] = cc
    return loc_map


def lookup_cidr_blocks(
    ips: Iterable[str],
    blocks_csv: Path,
    loc_map: Dict[str, str],
    is_v6: bool,
) -> Dict[str, Tuple[Optional[str], float, float, float]]:
    """Lookup each IP string against a sorted, disjoint GeoLite2 CIDR blocks CSV.

    Returns dict: ip -> (country_iso_code_or_None, latitude, longitude, accuracy_radius).
    Missing coordinates are stored as ``np.nan``.
    """
    ip_cls = ipaddress.IPv6Address if is_v6 else ipaddress.IPv4Address
    shift = 112 if is_v6 else 24
    max_bits = 128 if is_v6 else 32
    full_mask = (1 << max_bits) - 1

    query_ints: List[Tuple[str, int]] = []
    target_buckets = set()
    for ip in set(ips):
        try:
            ip_int = int(ip_cls(str(ip).strip()))
        except ValueError:
            continue
        query_ints.append((ip, ip_int))
        target_buckets.add(ip_int >> shift)

    if not query_ints:
        return {}

    starts: List[int] = []
    ends: List[int] = []
    meta: List[Tuple[Optional[str], float, float, float]] = []

    with open(blocks_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_net = header.index("network")
        idx_geo = header.index("geoname_id")
        idx_reg = header.index("registered_country_geoname_id")
        idx_rep = header.index("represented_country_geoname_id")
        idx_lat = header.index("latitude")
        idx_lon = header.index("longitude")
        idx_acc = header.index("accuracy_radius")

        for row in reader:
            net_str = row[idx_net]
            addr_str, prefix_str = net_str.split("/")
            prefix_len = int(prefix_str)
            ip_int = int(ip_cls(addr_str))
            host_bits = max_bits - prefix_len
            net_mask = full_mask ^ ((1 << host_bits) - 1) if host_bits > 0 else full_mask
            net_start = ip_int & net_mask

            # Safe bucket prefilter: supernets covering multiple top-level buckets are always kept.
            if prefix_len >= (max_bits - shift) and (net_start >> shift) not in target_buckets:
                continue

            net_end = net_start + (1 << host_bits) - 1

            gid = row[idx_geo] or row[idx_reg] or row[idx_rep]
            country = loc_map.get(gid) if gid else None

            lat_s, lon_s, acc_s = row[idx_lat], row[idx_lon], row[idx_acc]
            lat = float(lat_s) if lat_s else float("nan")
            lon = float(lon_s) if lon_s else float("nan")
            acc = float(acc_s) if acc_s else float("inf")

            starts.append(net_start)
            ends.append(net_end)
            meta.append((country, lat, lon, acc))

    results: Dict[str, Tuple[Optional[str], float, float, float]] = {}
    for ip, ip_int in query_ints:
        pos = bisect_right(starts, ip_int) - 1
        if pos >= 0 and ip_int <= ends[pos]:
            results[ip] = meta[pos]
        else:
            results[ip] = (None, float("nan"), float("nan"), float("inf"))
    return results


def build_ipv4_node_ip_map(
    paths_df: pd.DataFrame,
    supervised_csv: Path = DEFAULT_SUPERVISED_CSV,
) -> Dict[str, List[str]]:
    """Map IPv4 ITDK node_id -> sorted list of observed router interface IPs."""
    sup = pd.read_csv(supervised_csv, usecols=["measurement_id", "source_ip", "path_ips"])
    sup_map = {
        str(r.measurement_id): (str(r.source_ip).strip(), str(r.path_ips).split())
        for r in sup.itertuples(index=False)
    }

    node_ips: Dict[str, set] = defaultdict(set)
    for row in paths_df[["raw_measurement_id", "path_nodes"]].itertuples(index=False):
        mid = str(row.raw_measurement_id)
        if mid not in sup_map:
            continue
        src_ip, path_ips = sup_map[mid]
        raw_seq = [src_ip] + path_ips
        dedup_seq = [raw_seq[0]]
        for ip in raw_seq[1:]:
            if ip != dedup_seq[-1]:
                dedup_seq.append(ip)

        nodes = str(row.path_nodes).split("|")
        if len(dedup_seq) == len(nodes):
            for node_id, ip in zip(nodes, dedup_seq):
                node_ips[node_id].add(ip)

    return {nid: sorted(ips) for nid, ips in node_ips.items()}


def resolve_node_geo_map(
    node_to_ips: Dict[str, List[str]],
    ip_geo: Dict[str, Tuple[Optional[str], float, float, float]],
) -> Dict[str, Tuple[Optional[str], float, float]]:
    """Aggregate interface-IP geolocation onto each graph node_id.

    For nodes with multiple observed interface IPs, chooses the valid coordinate
    with minimum GeoLite2 ``accuracy_radius`` and the modal country code.
    """
    node_geo: Dict[str, Tuple[Optional[str], float, float]] = {}
    for node_id, ips in node_to_ips.items():
        countries = []
        valid_coords = []
        for ip in ips:
            cc, lat, lon, acc = ip_geo.get(ip, (None, float("nan"), float("nan"), float("inf")))
            if cc:
                countries.append(cc)
            if not math.isnan(lat) and not math.isnan(lon):
                valid_coords.append((acc, lat, lon))

        country = Counter(countries).most_common(1)[0][0] if countries else None
        if valid_coords:
            valid_coords.sort(key=lambda item: item[0])
            _, lat, lon = valid_coords[0]
        else:
            lat, lon = float("nan"), float("nan")
        node_geo[node_id] = (country, lat, lon)
    return node_geo


def compute_ipv4_and_ipv6_node_geo(
    ipv4_paths_dirs: Iterable[Path] = (),
    ipv6_node_dirs: Iterable[Path] = (),
    geolite_dir: Path = DEFAULT_GEOLITE_DIR,
    supervised_csv: Path = DEFAULT_SUPERVISED_CSV,
) -> Tuple[Dict[str, Tuple[Optional[str], float, float]], Dict[str, Tuple[Optional[str], float, float]]]:
    """Resolve both IPv4 and IPv6 node_ids against GeoLite2-City tables once."""
    loc_map = load_locations_map(geolite_dir)

    # IPv4
    v4_node_geo: Dict[str, Tuple[Optional[str], float, float]] = {}
    v4_node_ips: Dict[str, set] = defaultdict(set)
    for d in ipv4_paths_dirs:
        p = Path(d) / "paths_all.csv"
        if p.exists():
            m = build_ipv4_node_ip_map(pd.read_csv(p, usecols=["raw_measurement_id", "path_nodes"]), supervised_csv)
            for nid, ips in m.items():
                v4_node_ips[nid].update(ips)
    if v4_node_ips:
        all_v4_ips = {ip for ips in v4_node_ips.values() for ip in ips}
        v4_ip_geo = lookup_cidr_blocks(
            all_v4_ips, Path(geolite_dir) / "GeoLite2-City-Blocks-IPv4.csv", loc_map, is_v6=False
        )
        v4_node_geo = resolve_node_geo_map({k: sorted(v) for k, v in v4_node_ips.items()}, v4_ip_geo)

    # IPv6 (node_id is directly the IPv6 address string)
    v6_node_geo: Dict[str, Tuple[Optional[str], float, float]] = {}
    v6_nodes = set()
    for d in ipv6_node_dirs:
        p = Path(d) / "node_features.csv"
        if p.exists():
            v6_nodes.update(pd.read_csv(p, usecols=["node_id"])["node_id"].astype(str).unique())
    if v6_nodes:
        v6_ip_geo = lookup_cidr_blocks(
            v6_nodes, Path(geolite_dir) / "GeoLite2-City-Blocks-IPv6.csv", loc_map, is_v6=True
        )
        v6_node_geo = resolve_node_geo_map({nid: [nid] for nid in v6_nodes}, v6_ip_geo)

    return v4_node_geo, v6_node_geo


def enrich_dataset_dir(
    dataset_dir: Path,
    node_geo: Dict[str, Tuple[Optional[str], float, float]],
) -> dict:
    """Populate country/latitude/longitude on node_features.csv and geo_distance_km on edge_features.csv."""
    dataset_dir = Path(dataset_dir)
    nf_path = dataset_dir / "node_features.csv"
    ef_path = dataset_dir / "edge_features.csv"

    nf = pd.read_csv(nf_path)
    ef = pd.read_csv(ef_path)

    nids = nf["node_id"].astype(str)
    nf["country"] = nids.map(lambda nid: node_geo.get(nid, (None, float("nan"), float("nan")))[0])
    nf["latitude"] = nids.map(lambda nid: node_geo.get(nid, (None, float("nan"), float("nan")))[1])
    nf["longitude"] = nids.map(lambda nid: node_geo.get(nid, (None, float("nan"), float("nan")))[2])

    # Compute Haversine distance per edge using resolved node coordinates.
    coords = {
        nid: (geo[1], geo[2])
        for nid, geo in node_geo.items()
    }
    dists = []
    for src, dst in zip(ef["src"].astype(str), ef["dst"].astype(str)):
        lat1, lon1 = coords.get(src, (float("nan"), float("nan")))
        lat2, lon2 = coords.get(dst, (float("nan"), float("nan")))
        dists.append(haversine_km(lat1, lon1, lat2, lon2))
    ef["geo_distance_km"] = dists

    nf.to_csv(nf_path, index=False)
    ef.to_csv(ef_path, index=False)

    valid_d = pd.Series(dists).dropna()
    return {
        "dir": str(dataset_dir),
        "node_rows": len(nf),
        "unique_nodes": int(nids.nunique()),
        "country_pct": float(nf["country"].notna().mean() * 100.0),
        "latlon_pct": float((nf["latitude"].notna() & nf["longitude"].notna()).mean() * 100.0),
        "edge_rows": len(ef),
        "geo_edge_pct": float(ef["geo_distance_km"].notna().mean() * 100.0),
        "median_km": float(valid_d.median()) if len(valid_d) else float("nan"),
        "max_km": float(valid_d.max()) if len(valid_d) else float("nan"),
    }
