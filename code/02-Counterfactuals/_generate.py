"""Generate the `power-grid` project network (deterministic).

A regional electrical transmission grid (static -- no time dimension):
  - ~300 bus nodes (kind = "generator" | "substation" | "load")
Edges are undirected transmission lines, weighted by line `capacity_mw`, each
with a `length_km`. A companion `regions.csv` lists the three control areas.

Design parameters (the only record of the planted structure):
  - INTERTIE: ONE high-capacity interconnect line ties region C (a remote
    generation area) to the rest of the grid. It carries very high edge
    betweenness; removing it severs C from the load centers.
  - GEN_LOAD_MISMATCH: generators are concentrated in region C (a remote
    renewable cluster) while load concentrates in regions A and B, so power must
    traverse a few long high-capacity corridors -- the critical path.
  - RADIAL_TAILS: region B has radial (tree-like, single-feed) tails that lose
    power if one upstream line fails; region A's core is meshed (looped,
    redundant) and does not.
  - BRIDGE_SUB: a hidden bridge substation with high betweenness but modest
    degree sits between two otherwise weakly connected clusters inside region A.

Run:
    python data/projects/power-grid/_generate.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 42

# region sizes (sum ~ 300)
N_A = 130      # meshed urban load region
N_B = 110      # mixed region with radial tails
N_C = 60       # remote generation region
REGIONS = ["A", "B", "C"]

# --- planted parameters -----------------------------------------------------
VOLT_LEVELS = [69, 138, 230, 345, 500]
INTERTIE_CAP = 1800           # MW on the single C<->grid intertie
CORRIDOR_CAP = 1500           # MW on the long generation corridors
MESH_EXTRA_EDGES = 110        # extra loops added to region A's core (meshing)
RADIAL_FRACTION = 0.55        # fraction of region B that is radial tails


def main() -> None:
    rng = np.random.default_rng(SEED)

    # region centers on a 0-100 map; C is remote (far right)
    centers = {"A": (28, 55), "B": (52, 30), "C": (90, 70)}

    rows = []
    nid = 1

    def make_node(region, kind, cx, cy, spread, cap_lo, cap_hi):
        nonlocal nid
        x = float(np.clip(cx + rng.normal(0, spread), 0, 100))
        y = float(np.clip(cy + rng.normal(0, spread), 0, 100))
        node_id = f"BUS{nid:04d}"
        nid += 1
        cap = int(rng.integers(cap_lo, cap_hi)) if cap_hi > cap_lo else 0
        rows.append({
            "node_id": node_id, "kind": kind, "region": region,
            "label": f"{kind.capitalize()} {node_id[3:]}",
            "x": round(x, 2), "y": round(y, 2),
            "capacity_mw": cap,
            "voltage_kv": int(rng.choice(VOLT_LEVELS,
                              p=[0.30, 0.30, 0.20, 0.13, 0.07])),
        })
        return node_id

    # ----- region A: meshed urban load + a few generators ------------------
    a_ids = []
    for _ in range(N_A):
        roll = rng.random()
        if roll < 0.08:
            kind, cap = "generator", (150, 600)
        elif roll < 0.45:
            kind, cap = "substation", (0, 1)
        else:
            kind, cap = "load", (40, 320)        # capacity_mw = peak load draw
        a_ids.append(make_node("A", kind, *centers["A"], 11, *cap))

    # ----- region B: mixed, with radial tails ------------------------------
    b_ids = []
    for _ in range(N_B):
        roll = rng.random()
        if roll < 0.06:
            kind, cap = "generator", (120, 450)
        elif roll < 0.40:
            kind, cap = "substation", (0, 1)
        else:
            kind, cap = "load", (30, 260)
        b_ids.append(make_node("B", kind, *centers["B"], 12, *cap))

    # ----- region C: remote GENERATION cluster (renewables) ----------------
    c_ids = []
    for _ in range(N_C):
        roll = rng.random()
        if roll < 0.62:
            kind, cap = "generator", (200, 900)   # lots of generation
        elif roll < 0.85:
            kind, cap = "substation", (0, 1)
        else:
            kind, cap = "load", (10, 80)          # very little load out here
        c_ids.append(make_node("C", kind, *centers["C"], 9, *cap))

    nodes = pd.DataFrame(rows)
    pos = {r.node_id: (r.x, r.y) for r in nodes.itertuples()}

    def dist(a, b):
        (ax, ay), (bx, by) = pos[a], pos[b]
        return float(np.hypot(ax - bx, ay - by))

    edges_set: dict[tuple[str, str], dict] = {}

    def add_line(a, b, cap, kind="line"):
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in edges_set:
            return
        d = dist(a, b)
        edges_set[key] = {
            "from_id": key[0], "to_id": key[1],
            "capacity_mw": int(cap),
            "length_km": round(d * 1.4 + rng.uniform(0, 2), 2),
        }

    def nearest_within(node, pool, k):
        ds = sorted((dist(node, p), p) for p in pool if p != node)
        return [p for _, p in ds[:k]]

    # ----- region A core: build a connected MESH (redundant loops) ---------
    # First a spanning backbone (nearest-neighbor chain), then many extra loops.
    a_sorted = sorted(a_ids, key=lambda n: pos[n][0] + pos[n][1])
    for i in range(1, len(a_sorted)):
        # connect each to a nearby earlier node (keeps it connected + local)
        prev = nearest_within(a_sorted[i], a_sorted[:i], 1)[0]
        add_line(a_sorted[i], prev, rng.integers(200, 700))
    # extra meshing edges among nearby A nodes -> loops, triangles, redundancy
    added = 0
    attempts = 0
    while added < MESH_EXTRA_EDGES and attempts < MESH_EXTRA_EDGES * 8:
        attempts += 1
        u = rng.choice(a_ids)
        cand = nearest_within(u, a_ids, 6)
        v = rng.choice(cand)
        key = (u, v) if u < v else (v, u)
        if key not in edges_set:
            add_line(u, v, rng.integers(250, 800))
            added += 1

    # ----- region B: a meshed sub-core PLUS radial (tree) tails ------------
    n_radial = int(N_B * RADIAL_FRACTION)
    b_core = b_ids[:N_B - n_radial]
    b_tails = b_ids[N_B - n_radial:]
    # core: small mesh
    b_core_sorted = sorted(b_core, key=lambda n: pos[n][0] + pos[n][1])
    for i in range(1, len(b_core_sorted)):
        prev = nearest_within(b_core_sorted[i], b_core_sorted[:i], 1)[0]
        add_line(b_core_sorted[i], prev, rng.integers(180, 600))
    for _ in range(18):
        u = rng.choice(b_core)
        v = rng.choice(nearest_within(u, b_core, 5))
        add_line(u, v, rng.integers(200, 550))
    # tails: each radial node hangs off exactly ONE upstream node (a tree) ->
    # single feed, no redundancy. Attach to nearest already-connected B node.
    connected_b = list(b_core)
    for t in b_tails:
        parent = nearest_within(t, connected_b, 1)[0]
        add_line(t, parent, rng.integers(60, 200))   # thinner radial feeders
        connected_b.append(t)

    # ----- region C: internal collector network (radial-ish to a C hub) ----
    # choose a C substation as the collector hub
    c_subs = [n for n in c_ids
              if nodes.loc[nodes.node_id == n, "kind"].iloc[0] == "substation"]
    c_hub = c_subs[0] if c_subs else c_ids[0]
    c_sorted = sorted(c_ids, key=lambda n: dist(n, c_hub))
    connected_c = [c_hub]
    for n in c_sorted:
        if n == c_hub:
            continue
        parent = nearest_within(n, connected_c, 1)[0]
        add_line(n, parent, rng.integers(150, 500))
        connected_c.append(n)
    # a little meshing inside C so it isn't a pure tree
    for _ in range(10):
        u = rng.choice(c_ids)
        v = rng.choice(nearest_within(u, c_ids, 4))
        add_line(u, v, rng.integers(150, 450))

    # ----- A <-> B coupling: a few normal-capacity ties (redundant) --------
    # pick the closest A/B node pairs and connect a handful (so A-B is robust)
    ab_pairs = sorted(
        ((dist(a, b), a, b) for a in a_ids for b in b_ids),
        key=lambda t: t[0])[:6]
    for _, a, b in ab_pairs:
        add_line(a, b, rng.integers(400, 900))

    # ----- THE GENERATION CORRIDORS + SINGLE INTERTIE to region C ----------
    # Region C connects to the rest of the grid through ONE intertie line into a
    # region-B gateway substation, plus long high-capacity corridors carry C's
    # generation toward the A/B load centers. The intertie is the sole electrical
    # path from C to everything else.
    # gateway on the grid side: the region-B node closest to the C hub
    gateway = min(b_ids, key=lambda n: dist(n, c_hub))
    # the single intertie line (very high capacity, long)
    add_line(c_hub, gateway, INTERTIE_CAP)
    # long corridors: from gateway deep into A's core (high-capacity backbone)
    a_core_targets = sorted(a_ids, key=lambda n: dist(n, gateway))[:3]
    corridor_prev = gateway
    for tgt in a_core_targets:
        add_line(corridor_prev, tgt, CORRIDOR_CAP)
        corridor_prev = tgt

    # ----- BRIDGE SUBSTATION inside region A -------------------------------
    # Split region A loosely into two halves by x; route nearly all cross-half
    # A traffic through ONE substation with modest degree (a hidden bridge).
    a_x = {n: pos[n][0] for n in a_ids}
    median_x = np.median(list(a_x.values()))
    left = [n for n in a_ids if a_x[n] < median_x]
    right = [n for n in a_ids if a_x[n] >= median_x]
    # remove any existing left-right A edges to force the bottleneck
    to_drop = []
    for key in list(edges_set.keys()):
        u, v = key
        if u in a_x and v in a_x:
            lu = u in left
            lv = v in left
            if lu != lv:
                to_drop.append(key)
    # keep the bridge edges only: pick a bridge substation near the divide
    a_subs = [n for n in a_ids
              if nodes.loc[nodes.node_id == n, "kind"].iloc[0] == "substation"]
    bridge = min(a_subs, key=lambda n: abs(a_x[n] - median_x))
    for key in to_drop:
        del edges_set[key]
    # connect the bridge to ONE node on each side (deliberately modest degree)
    bl = nearest_within(bridge, left, 1)
    br = nearest_within(bridge, right, 1)
    for n in bl + br:
        add_line(bridge, n, rng.integers(300, 700))

    # repair: dropping the cross-divide A edges can orphan a few nodes. Reconnect
    # any now-isolated A node to its nearest SAME-SIDE neighbor (never across the
    # divide), so the bridge stays the sole left<->right path but no bus is dead.
    incident = set()
    for (u, v) in edges_set:
        incident.add(u)
        incident.add(v)
    for n in a_ids:
        if n in incident:
            continue
        side = left if n in left else right
        side_pool = [m for m in side if m != n and m != bridge]
        parent = nearest_within(n, side_pool, 1)[0]
        add_line(n, parent, rng.integers(200, 600))

    edges = pd.DataFrame(list(edges_set.values()))

    # ----- regions lookup table --------------------------------------------
    regions = pd.DataFrame({
        "region": REGIONS,
        "name": ["Metro East Control Area", "Central Valley Control Area",
                 "High Plains Control Area"],
        "center_x": [centers[r][0] for r in REGIONS],
        "center_y": [centers[r][1] for r in REGIONS],
    })

    nodes.to_csv(HERE / "nodes.csv", index=False)
    edges.to_csv(HERE / "edges.csv", index=False)
    regions.to_csv(HERE / "regions.csv", index=False)
    kc = nodes.kind.value_counts()
    print(f"power-grid: {len(nodes)} buses "
          f"({kc.get('generator',0)} generators + {kc.get('substation',0)} substations + "
          f"{kc.get('load',0)} loads), {len(edges)} lines, {len(regions)} regions.")


if __name__ == "__main__":
    main()
