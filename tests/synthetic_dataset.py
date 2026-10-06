"""Builds a synthetic ITDK + traceroute set in the real file formats and runs Dataset_Build.ipynb on it.

Used only to test the model code. RTT is additive over routers (so tomography has a real signal) with a mild
daily drift (so the temporal branch has something to find). None of this is thesis data.
"""
import json
import math
import random
import re
import tempfile
from pathlib import Path

import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / "Dataset_Build.ipynb"


def make(tmp=None, n_routers=140, n_traceroutes=1800, seed=7):
    rng = random.Random(seed)
    tmp = Path(tmp or tempfile.mkdtemp())
    out, itdk = tmp / "DeepNT_Output", tmp / "DataSet"
    (out / "real_rtt").mkdir(parents=True, exist_ok=True)
    itdk.mkdir(parents=True, exist_ok=True)

    parent = {k: (rng.choice([1, 2, 3]) if k < 14 else rng.randint(4, k - 1)) for k in range(4, n_routers + 1)}
    delay = {k: rng.uniform(1.0, 12.0) for k in range(1, n_routers + 1)}
    ip1, ip2 = (lambda k: f"20.{k}.0.1"), (lambda k: f"20.{k}.0.2")
    nodes = ["# nodes", "# start time: 1742404243"] + [f"node N{k}:  {ip1(k)} {ip2(k)}" for k in range(1, n_routers + 1)]
    links = ["# start time: 1742404243 (2025-03-19 17:10:43 UTC)",
             "#   -P /d/cycle-20250224/abz2-uk.team-probing.c012517.20250224.warts.gz",
             "#   -P /d/cycle-20250309/dac-bd.team-probing.c012600.20250309.warts.gz", "#"]
    links += [f"link L{i}:  N{p}:{ip1(p)} N{c}:{ip1(c)}" for i, (c, p) in enumerate(parent.items(), 1)]
    perm = {k: 1 + (k * 7919) % 100003 for k in range(0, n_routers + 10)}      # scramble IDs like real ITDK
    scramble = lambda text: re.sub(r"\bN(\d+)\b", lambda m: f"N{perm[int(m.group(1))]}", text)
    (itdk / "midar-iff-snmp.nodes").write_text(scramble("\n".join(nodes)) + "\n")
    (itdk / "midar-iff-snmp.links").write_text(scramble("\n".join(links)) + "\n")

    def route(k):
        r = [k]
        while r[-1] > 3:
            r.append(parent[r[-1]])
        return r[::-1]

    leaves = [k for k in range(4, n_routers + 1) if k not in parent.values()]
    rows = []
    for i in range(n_traceroutes):
        leaf = rng.choice(leaves)
        path = route(leaf)
        hops = [ip1(n) for n in path[1:-1]] + [ip2(leaf)]
        nums = list(range(1, len(hops) + 1))
        resp = rng.random() > 0.02
        if rng.random() < 0.06 and len(hops) > 2:
            hops[rng.randrange(len(hops) - 1)] = "99.9.9.9"
        t = pd.Timestamp("2025-03-10 04:00:00") + pd.Timedelta(seconds=rng.randrange(0, 160000))
        drift = 1 + 0.15 * math.sin(2 * math.pi * (t.hour + t.minute / 60) / 24)
        rtt = 1000 * drift * (2 * sum(delay[n] for n in path[1:]) + rng.gauss(0, 1.5))
        if i in (11, 12):
            rtt = 9_000_000
        wf = ".abz2-uk.team-probing.c012585.20250310.warts" if path[0] == 1 else ".dac-bd.team-probing.c012587.20250310.warts"
        rows.append(dict(measurement_id=f"{wf}:{i}", warts_file=wf, source_ip=ip2(path[0]), destination_ip=ip2(leaf),
                         start_time=str(t), hop_numbers="|".join(map(str, nums)), path_ips="|".join(hops),
                         destination_responded=resp, destination_hop_number=len(path) if resp else None,
                         destination_rtt_ms=rtt if resp else None))
    pd.DataFrame(rows).to_csv(out / "real_rtt" / "caida_real_rtt_supervised.csv", index=False)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    root = 'r"C:\\Users\\KINGDOM\\Desktop\\Thesis"'
    assert root in codes[0]
    codes[0] = codes[0].replace(root, 'r"' + str(tmp) + '"')
    ns = {}
    for c in codes:
        exec(c, ns)
    return out / "data" / "processed"


if __name__ == "__main__":
    print(make())
