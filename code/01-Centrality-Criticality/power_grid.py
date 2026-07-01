"""
SYSEN 5470 Week 1 - Think Like a Graph
Power Grid Centrality and Criticality Analysis

Capacity-only version:
- Edge weight: capacity_mw
- Weighted shortest-path distance: 1 / capacity_mw
- length_km is kept as a descriptive line attribute only
- Betweenness values are normalized and rounded to three decimals

Run from Command Prompt in the same folder as nodes.csv, edges.csv, regions.csv:
    python week1_power_grid_capacity_only_analysis.py

Outputs are saved in:
    week1_outputs_capacity_only/
"""

from pathlib import Path
import json
import random

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# 0. Settings
# -----------------------------
DATA_DIR = Path(".")
OUT_DIR = Path("power_grid_outputs_capacity_only_normalized_betweenness")
TOP_K = 5
RANDOM_TRIALS = 300
RANDOM_SEED = 5470

random.seed(RANDOM_SEED)
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "graphics_data").mkdir(exist_ok=True)


# -----------------------------
# 1. Load data
# -----------------------------
nodes = pd.read_csv(DATA_DIR / "nodes.csv")
edges = pd.read_csv(DATA_DIR / "edges.csv")
regions = pd.read_csv(DATA_DIR / "regions.csv")

# Keep a clean copy of node attributes by node_id.
node_attr = nodes.set_index("node_id").to_dict("index")


# -----------------------------
# 2. Build undirected weighted graph
# -----------------------------
G = nx.Graph()

# Add buses as nodes.
for _, row in nodes.iterrows():
    G.add_node(row["node_id"], **row.to_dict())

# Add transmission lines as undirected edges.
# capacity_mw is the physical edge weight.
# distance_capacity = 1 / capacity_mw is used for weighted shortest paths.
for _, row in edges.iterrows():
    cap = float(row["capacity_mw"])
    G.add_edge(
        row["from_id"],
        row["to_id"],
        capacity_mw=cap,
        length_km=float(row["length_km"]),
        distance_capacity=1.0 / cap,
    )


# -----------------------------
# 3. Helper functions
# -----------------------------
def largest_component_nodes(graph):
    """Return the set of nodes in the largest connected component."""
    components = list(nx.connected_components(graph))
    if not components:
        return set()
    return max(components, key=len)


def damage_metrics(graph):
    """Measure damage after a node or line removal.

    Anything outside the largest connected component is treated as isolated
    from the main grid. This is a graph measure, not an AC power-flow model.
    """
    if graph.number_of_nodes() == 0:
        return {
            "largest_component_size": 0,
            "n_components": 0,
            "isolated_nodes": 0,
            "isolated_load_mw": 0.0,
            "isolated_generation_mw": 0.0,
            "avg_shortest_path_lcc": None,
        }

    components = list(nx.connected_components(graph))
    lcc = max(components, key=len)
    isolated = set(graph.nodes()) - set(lcc)

    isolated_load = sum(
        graph.nodes[n].get("capacity_mw", 0)
        for n in isolated
        if graph.nodes[n].get("kind") == "load"
    )
    isolated_generation = sum(
        graph.nodes[n].get("capacity_mw", 0)
        for n in isolated
        if graph.nodes[n].get("kind") == "generator"
    )

    # Average shortest path is omitted from the repeated outage loop to keep
    # this script fast and simple for Command Prompt execution.
    avg_path = None

    return {
        "largest_component_size": len(lcc),
        "n_components": len(components),
        "isolated_nodes": len(isolated),
        "isolated_load_mw": isolated_load,
        "isolated_generation_mw": isolated_generation,
        "avg_shortest_path_lcc": avg_path,
    }


def remove_nodes_and_measure(node_list):
    """Remove selected buses and measure graph damage."""
    H = G.copy()
    H.remove_nodes_from(node_list)
    return damage_metrics(H)


def remove_edges_and_measure(edge_list):
    """Remove selected lines and measure graph damage."""
    H = G.copy()
    H.remove_edges_from(edge_list)
    return damage_metrics(H)


def edge_key(u, v):
    """Canonical undirected edge key."""
    return tuple(sorted([u, v]))


# -----------------------------
# 4. Basic graph summary
# -----------------------------
base_damage = damage_metrics(G)
degrees = dict(G.degree())
strengths = dict(G.degree(weight="capacity_mw"))

summary = pd.DataFrame([
    {
        "n_buses": G.number_of_nodes(),
        "n_lines": G.number_of_edges(),
        "density": nx.density(G),
        "n_components": nx.number_connected_components(G),
        "largest_component_size": base_damage["largest_component_size"],
        "degree_min": min(degrees.values()),
        "degree_median": pd.Series(degrees).median(),
        "degree_mean": pd.Series(degrees).mean(),
        "degree_max": max(degrees.values()),
        "total_generation_mw": nodes.loc[nodes["kind"] == "generator", "capacity_mw"].sum(),
        "total_load_mw": nodes.loc[nodes["kind"] == "load", "capacity_mw"].sum(),
        "total_line_capacity_mw": edges["capacity_mw"].sum(),
    }
])
summary.to_csv(OUT_DIR / "01_graph_summary.csv", index=False)

# Generation/load by region answers: where is power made vs used?
region_power = (
    nodes.pivot_table(
        index="region",
        columns="kind",
        values="capacity_mw",
        aggfunc="sum",
        fill_value=0,
    )
    .reset_index()
)
for col in ["generator", "load", "substation"]:
    if col not in region_power.columns:
        region_power[col] = 0
region_power["net_generation_minus_load_mw"] = region_power["generator"] - region_power["load"]
region_power = region_power.merge(regions, on="region", how="left")
region_power.to_csv(OUT_DIR / "02_region_generation_load_balance.csv", index=False)


# -----------------------------
# 5. Single-node and double-node joins
# -----------------------------
# Single-node join: add attributes of the from_id endpoint.
from_nodes = nodes[["node_id", "kind", "region", "capacity_mw", "voltage_kv", "label", "x", "y"]].rename(
    columns={
        "node_id": "from_id",
        "kind": "from_kind",
        "region": "from_region",
        "capacity_mw": "from_bus_capacity_mw",
        "voltage_kv": "from_voltage_kv",
        "label": "from_label",
        "x": "from_x",
        "y": "from_y",
    }
)
single_join = edges.merge(from_nodes, on="from_id", how="left")
single_join.to_csv(OUT_DIR / "03_single_node_join_from_endpoint.csv", index=False)

# Double-node join: add attributes of the to_id endpoint too.
to_nodes = nodes[["node_id", "kind", "region", "capacity_mw", "voltage_kv", "label", "x", "y"]].rename(
    columns={
        "node_id": "to_id",
        "kind": "to_kind",
        "region": "to_region",
        "capacity_mw": "to_bus_capacity_mw",
        "voltage_kv": "to_voltage_kv",
        "label": "to_label",
        "x": "to_x",
        "y": "to_y",
    }
)
joined = single_join.merge(to_nodes, on="to_id", how="left")

# Canonical labels for an undirected network.
joined["region_pair"] = joined.apply(lambda r: "-".join(sorted([r["from_region"], r["to_region"]])), axis=1)
joined["kind_pair"] = joined.apply(lambda r: "-".join(sorted([r["from_kind"], r["to_kind"]])), axis=1)
joined["voltage_pair"] = joined.apply(lambda r: "-".join(map(str, sorted([r["from_voltage_kv"], r["to_voltage_kv"]]))), axis=1)
joined["inter_region"] = joined["from_region"] != joined["to_region"]
joined["edge_key"] = joined.apply(lambda r: "--".join(edge_key(r["from_id"], r["to_id"])), axis=1)
joined.to_csv(OUT_DIR / "04_double_node_join_full_edge_context.csv", index=False)


# -----------------------------
# 6. Aggregation by region and bus kind
# -----------------------------
region_summary = (
    joined.groupby("region_pair", as_index=False)
    .agg(
        line_count=("capacity_mw", "size"),
        total_capacity_mw=("capacity_mw", "sum"),
        mean_capacity_mw=("capacity_mw", "mean"),
        max_capacity_mw=("capacity_mw", "max"),
        total_length_km=("length_km", "sum"),
        mean_length_km=("length_km", "mean"),
        inter_region=("inter_region", "max"),
    )
    .sort_values(["inter_region", "region_pair"], ascending=[False, True])
)
region_summary.to_csv(OUT_DIR / "05_region_pair_summary.csv", index=False)

kind_summary = (
    joined.groupby("kind_pair", as_index=False)
    .agg(
        line_count=("capacity_mw", "size"),
        total_capacity_mw=("capacity_mw", "sum"),
        mean_capacity_mw=("capacity_mw", "mean"),
        total_length_km=("length_km", "sum"),
    )
    .sort_values("line_count", ascending=False)
)
kind_summary.to_csv(OUT_DIR / "06_kind_pair_summary.csv", index=False)


# -----------------------------
# 7. Centrality
# -----------------------------
# Betweenness: unweighted uses fewest hops; capacity-weighted uses 1/capacity_mw.
# normalized=True scales values to the 0-1 range, making bus-to-bus and
# line-to-line comparisons easier to read. Values are rounded to 3 decimals
# before export.
bet_unweighted = nx.betweenness_centrality(G, normalized=True, weight=None)
bet_capacity = nx.betweenness_centrality(G, normalized=True, weight="distance_capacity")

node_centrality = pd.DataFrame([
    {
        "node_id": n,
        "label": G.nodes[n].get("label"),
        "kind": G.nodes[n].get("kind"),
        "region": G.nodes[n].get("region"),
        "bus_capacity_mw": G.nodes[n].get("capacity_mw"),
        "voltage_kv": G.nodes[n].get("voltage_kv"),
        "degree": degrees[n],
        "strength_incident_capacity_mw": strengths[n],
        "betweenness_unweighted": bet_unweighted[n],
        "betweenness_weighted_capacity": bet_capacity[n],
        "x": G.nodes[n].get("x"),
        "y": G.nodes[n].get("y"),
    }
    for n in G.nodes()
])
node_centrality["degree_rank"] = node_centrality["degree"].rank(ascending=False, method="min")
node_centrality["strength_rank"] = node_centrality["strength_incident_capacity_mw"].rank(ascending=False, method="min")
node_centrality["betweenness_capacity_rank"] = node_centrality["betweenness_weighted_capacity"].rank(ascending=False, method="min")
node_centrality[["betweenness_unweighted", "betweenness_weighted_capacity"]] = node_centrality[["betweenness_unweighted", "betweenness_weighted_capacity"]].round(3)
node_centrality.to_csv(OUT_DIR / "07_node_centrality_capacity_weighted.csv", index=False)

edge_bet_unweighted = nx.edge_betweenness_centrality(G, normalized=True, weight=None)
edge_bet_capacity = nx.edge_betweenness_centrality(G, normalized=True, weight="distance_capacity")

edge_cent_rows = []
for _, r in joined.iterrows():
    u, v = r["from_id"], r["to_id"]
    k = edge_key(u, v)
    edge_cent_rows.append({
        **r.to_dict(),
        "edge_betweenness_unweighted": edge_bet_unweighted.get(k, edge_bet_unweighted.get((v, u))),
        "edge_betweenness_weighted_capacity": edge_bet_capacity.get(k, edge_bet_capacity.get((v, u))),
    })
edge_centrality = pd.DataFrame(edge_cent_rows)
edge_centrality["capacity_rank"] = edge_centrality["capacity_mw"].rank(ascending=False, method="min")
edge_centrality["length_rank"] = edge_centrality["length_km"].rank(ascending=False, method="min")
edge_centrality["edge_betweenness_capacity_rank"] = edge_centrality["edge_betweenness_weighted_capacity"].rank(ascending=False, method="min")
edge_centrality[["edge_betweenness_unweighted", "edge_betweenness_weighted_capacity"]] = edge_centrality[["edge_betweenness_unweighted", "edge_betweenness_weighted_capacity"]].round(3)
edge_centrality.to_csv(OUT_DIR / "08_edge_centrality_capacity_weighted.csv", index=False)


# -----------------------------
# 8. Single-line outage test
# -----------------------------
single_edge_rows = []
for u, v, data in G.edges(data=True):
    metrics = remove_edges_and_measure([(u, v)])
    row = {
        "from_id": u,
        "to_id": v,
        "edge_key": "--".join(edge_key(u, v)),
        "capacity_mw": data["capacity_mw"],
        "length_km": data["length_km"],
        **metrics,
    }
    # add joined endpoint info
    match = joined[joined["edge_key"] == row["edge_key"]].iloc[0]
    row.update({
        "region_pair": match["region_pair"],
        "kind_pair": match["kind_pair"],
        "from_region": match["from_region"],
        "to_region": match["to_region"],
        "from_kind": match["from_kind"],
        "to_kind": match["to_kind"],
        "edge_betweenness_weighted_capacity": float(
            edge_centrality.loc[edge_centrality["edge_key"] == row["edge_key"], "edge_betweenness_weighted_capacity"].iloc[0]
        ),
    })
    single_edge_rows.append(row)

single_edge_outages = pd.DataFrame(single_edge_rows)
single_edge_outages.to_csv(OUT_DIR / "09_single_edge_outage_results.csv", index=False)


# -----------------------------
# 9. Targeted removal tests
# -----------------------------
top_degree_nodes = node_centrality.sort_values("degree", ascending=False).head(TOP_K)["node_id"].tolist()
top_strength_nodes = node_centrality.sort_values("strength_incident_capacity_mw", ascending=False).head(TOP_K)["node_id"].tolist()
top_bet_nodes = node_centrality.sort_values("betweenness_weighted_capacity", ascending=False).head(TOP_K)["node_id"].tolist()

def top_edges_from(df, sort_col):
    return [tuple(x.split("--")) for x in df.sort_values(sort_col, ascending=False).head(TOP_K)["edge_key"]]

top_capacity_edges = top_edges_from(edge_centrality, "capacity_mw")
top_length_edges = top_edges_from(edge_centrality, "length_km")
top_edge_bet_edges = top_edges_from(edge_centrality, "edge_betweenness_weighted_capacity")

removal_rows = []
for rule, assets, metrics in [
    ("none_intact_network", "none", damage_metrics(G)),
    ("top_5_buses_by_degree", "; ".join(top_degree_nodes), remove_nodes_and_measure(top_degree_nodes)),
    ("top_5_buses_by_strength_capacity", "; ".join(top_strength_nodes), remove_nodes_and_measure(top_strength_nodes)),
    ("top_5_buses_by_weighted_betweenness_capacity", "; ".join(top_bet_nodes), remove_nodes_and_measure(top_bet_nodes)),
    ("top_5_lines_by_capacity", "; ".join(["--".join(e) for e in top_capacity_edges]), remove_edges_and_measure(top_capacity_edges)),
    ("top_5_lines_by_length_descriptive", "; ".join(["--".join(e) for e in top_length_edges]), remove_edges_and_measure(top_length_edges)),
    ("top_5_lines_by_weighted_edge_betweenness_capacity", "; ".join(["--".join(e) for e in top_edge_bet_edges]), remove_edges_and_measure(top_edge_bet_edges)),
]:
    removal_rows.append({"rule": rule, "assets_removed": assets, **metrics})

# Random baselines for top-k bus and line removals.
all_nodes = list(G.nodes())
all_edges = list(G.edges())
random_node_metrics = []
random_edge_metrics = []
for _ in range(RANDOM_TRIALS):
    random_node_metrics.append(remove_nodes_and_measure(random.sample(all_nodes, TOP_K)))
    random_edge_metrics.append(remove_edges_and_measure(random.sample(all_edges, TOP_K)))

for label, rows in [("random_5_buses_mean", random_node_metrics), ("random_5_lines_mean", random_edge_metrics)]:
    df = pd.DataFrame(rows)
    removal_rows.append({
        "rule": label,
        "assets_removed": f"mean of {RANDOM_TRIALS} random trials",
        "largest_component_size": df["largest_component_size"].mean(),
        "n_components": df["n_components"].mean(),
        "isolated_nodes": df["isolated_nodes"].mean(),
        "isolated_load_mw": df["isolated_load_mw"].mean(),
        "isolated_generation_mw": df["isolated_generation_mw"].mean(),
        "avg_shortest_path_lcc": df["avg_shortest_path_lcc"].mean(),
    })

removal_results = pd.DataFrame(removal_rows)
removal_results.to_csv(OUT_DIR / "10_targeted_removal_results.csv", index=False)


# -----------------------------
# 10. Maintenance recommendation table
# -----------------------------
worst_load_line = single_edge_outages.sort_values("isolated_load_mw", ascending=False).iloc[0]
worst_gen_line = single_edge_outages.sort_values("isolated_generation_mw", ascending=False).iloc[0]
top_capacity_line = edge_centrality.sort_values("capacity_mw", ascending=False).iloc[0]
top_edge_bet_line = edge_centrality.sort_values("edge_betweenness_weighted_capacity", ascending=False).iloc[0]
top_degree_bus = node_centrality.sort_values("degree", ascending=False).iloc[0]
top_strength_bus = node_centrality.sort_values("strength_incident_capacity_mw", ascending=False).iloc[0]
top_bet_bus = node_centrality.sort_values("betweenness_weighted_capacity", ascending=False).iloc[0]

budget = pd.DataFrame([
    {
        "rule": "degree",
        "recommended_asset": top_degree_bus["node_id"],
        "asset_type": "bus",
        "reason": "most direct line connections",
        "report_interpretation": "protects an obvious local hub",
    },
    {
        "rule": "bus_strength_capacity",
        "recommended_asset": top_strength_bus["node_id"],
        "asset_type": "bus",
        "reason": "largest incident MW capacity",
        "report_interpretation": "protects a bus attached to large capacity",
    },
    {
        "rule": "bus_weighted_betweenness_capacity",
        "recommended_asset": top_bet_bus["node_id"],
        "asset_type": "bus",
        "reason": "lies on many capacity-preferred shortest paths",
        "report_interpretation": "protects a hidden bridge bus",
    },
    {
        "rule": "line_capacity",
        "recommended_asset": top_capacity_line["edge_key"],
        "asset_type": "line",
        "reason": "highest line capacity_mw",
        "report_interpretation": "protects the visible high-capacity export corridor",
    },
    {
        "rule": "edge_weighted_betweenness_capacity",
        "recommended_asset": top_edge_bet_line["edge_key"],
        "asset_type": "line",
        "reason": "line lies on many capacity-preferred paths",
        "report_interpretation": "protects a structurally load-bearing line",
    },
    {
        "rule": "single_line_outage_isolated_load",
        "recommended_asset": worst_load_line["edge_key"],
        "asset_type": "line",
        "reason": "one-line outage isolates the most load MW",
        "report_interpretation": "best choice if the goal is to avoid load isolation",
    },
    {
        "rule": "single_line_outage_stranded_generation",
        "recommended_asset": worst_gen_line["edge_key"],
        "asset_type": "line",
        "reason": "one-line outage strands the most generation MW",
        "report_interpretation": "best choice if the goal is to avoid stranding remote generation",
    },
])
budget.to_csv(OUT_DIR / "11_maintenance_budget_recommendations.csv", index=False)


# -----------------------------
# 11. Figure-ready and AI-graphics-ready data
# -----------------------------
# Figure 1: region capacity matrix
region_matrix = joined.pivot_table(
    index="from_region",
    columns="to_region",
    values="capacity_mw",
    aggfunc="sum",
    fill_value=0,
)
# Symmetric matrix for undirected interpretation
all_regions = sorted(nodes["region"].unique())
sym = pd.DataFrame(0.0, index=all_regions, columns=all_regions)
for _, r in joined.iterrows():
    a, b = r["from_region"], r["to_region"]
    sym.loc[a, b] += r["capacity_mw"]
    if a != b:
        sym.loc[b, a] += r["capacity_mw"]
sym.to_csv(OUT_DIR / "graphics_data" / "figure1_region_capacity_matrix.csv")

plt.figure(figsize=(5.5, 4.5))
plt.imshow(sym.values)
plt.xticks(range(len(sym.columns)), sym.columns)
plt.yticks(range(len(sym.index)), sym.index)
plt.colorbar(label="Total line capacity (MW)")
plt.title("Figure 1. Transmission capacity by region pair")
for i in range(len(sym.index)):
    for j in range(len(sym.columns)):
        plt.text(j, i, f"{sym.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "12_fig1_region_capacity_heatmap.png", dpi=200)
plt.close()

# Figure 2: outage damage bars
fig2 = pd.DataFrame([
    {"case": "Highest capacity line", "asset": top_capacity_line["edge_key"], "isolated_load_mw": float(single_edge_outages.loc[single_edge_outages["edge_key"] == top_capacity_line["edge_key"], "isolated_load_mw"].iloc[0])},
    {"case": "Highest edge betweenness", "asset": top_edge_bet_line["edge_key"], "isolated_load_mw": float(single_edge_outages.loc[single_edge_outages["edge_key"] == top_edge_bet_line["edge_key"], "isolated_load_mw"].iloc[0])},
    {"case": "Worst single-line load outage", "asset": worst_load_line["edge_key"], "isolated_load_mw": float(worst_load_line["isolated_load_mw"])},
])
fig2.to_csv(OUT_DIR / "graphics_data" / "figure2_single_line_outage_bars.csv", index=False)

plt.figure(figsize=(7, 4.2))
plt.bar(fig2["case"], fig2["isolated_load_mw"])
plt.ylabel("Isolated load (MW)")
plt.title("Figure 2. Single-line outage damage by selection rule")
plt.xticks(rotation=20, ha="right")
for i, v in enumerate(fig2["isolated_load_mw"]):
    plt.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "13_fig2_single_line_outage_damage.png", dpi=200)
plt.close()

# Figure 3 data: top hidden buses for an AI graphic
hidden_buses = node_centrality.sort_values("betweenness_weighted_capacity", ascending=False).head(10)
hidden_buses.to_csv(OUT_DIR / "graphics_data" / "figure3_top_hidden_bottleneck_buses.csv", index=False)

# Map data for AI or external plotting.
node_centrality.to_csv(OUT_DIR / "graphics_data" / "ai_map_nodes_with_centrality.csv", index=False)
edge_centrality.to_csv(OUT_DIR / "graphics_data" / "ai_map_edges_with_centrality.csv", index=False)

# Storyboard for AI graphics generation.
storyboard = {
    "theme": "Think Like a Graph: capacity is not redundancy",
    "main_question": "Are the most visible grid assets the best places to protect, or do hidden bottlenecks create more cascading-failure exposure?",
    "weighting_rule": "capacity-only weighted distance = 1 / capacity_mw; length_km is descriptive only",
    "recommended_graphics": [
        {
            "figure": "Figure 1",
            "title": "Where power transfer is structurally concentrated",
            "data_file": "graphics_data/figure1_region_capacity_matrix.csv",
            "visual_type": "region-pair heatmap",
            "message": "Generation is concentrated in Region C while load is concentrated in A/B; sparse inter-region capacity carries the mismatch.",
        },
        {
            "figure": "Figure 2",
            "title": "The worst single-line load outage is not the biggest line",
            "data_file": "graphics_data/figure2_single_line_outage_bars.csv",
            "visual_type": "bar chart",
            "message": "Capacity identifies the prominent export corridor, but outage damage depends on graph position.",
        },
        {
            "figure": "Figure 3 optional",
            "title": "Hidden bottleneck buses",
            "data_file": "graphics_data/figure3_top_hidden_bottleneck_buses.csv",
            "visual_type": "ranked callout table or network callout",
            "message": "BUS0161 is important because it lies on many high-capacity preferred paths, not because it is a high-degree hub.",
        },
    ],
}
with open(OUT_DIR / "graphics_data" / "ai_graphics_storyboard.json", "w", encoding="utf-8") as f:
    json.dump(storyboard, f, indent=2)


# -----------------------------
# 12. Write readable summary
# -----------------------------
with open(OUT_DIR / "00_READ_ME_FIRST.txt", "w", encoding="utf-8") as f:
    f.write("SYSEN 5470 - Power Grid Capacity-Only Analysis\n")
    f.write("====================================================\n\n")
    f.write("Main question:\n")
    f.write("Are the most visible grid assets the best places to protect, or do hidden bottlenecks create more cascading-failure exposure?\n\n")
    f.write("Weighting rule:\n")
    f.write("Weighted distance = 1 / capacity_mw. length_km is kept as a descriptive attribute only. Betweenness is normalized and rounded to three decimals.\n\n")
    f.write("Basic graph:\n")
    f.write(f"- Buses: {G.number_of_nodes()}\n")
    f.write(f"- Lines: {G.number_of_edges()}\n")
    f.write(f"- Components: {nx.number_connected_components(G)}\n\n")
    f.write("Power made vs used by region:\n")
    f.write(region_power.to_string(index=False))
    f.write("\n\n")
    f.write("Most important-looking vs structurally important assets:\n")
    f.write(f"- Top weighted-betweenness bus: {top_bet_bus['node_id']} ({top_bet_bus['label']}), degree {top_bet_bus['degree']}, region {top_bet_bus['region']}\n")
    f.write(f"- Highest-capacity line: {top_capacity_line['edge_key']}, capacity {top_capacity_line['capacity_mw']:.0f} MW, length {top_capacity_line['length_km']:.2f} km\n")
    f.write(f"- Highest weighted-edge-betweenness line: {top_edge_bet_line['edge_key']}, capacity {top_edge_bet_line['capacity_mw']:.0f} MW\n")
    f.write(f"- Worst single-line outage by isolated load: {worst_load_line['edge_key']}, isolated load {worst_load_line['isolated_load_mw']:.0f} MW\n")
    f.write(f"- Worst single-line outage by stranded generation: {worst_gen_line['edge_key']}, stranded generation {worst_gen_line['isolated_generation_mw']:.0f} MW\n\n")
    f.write("Most useful files for report writing:\n")
    f.write("- 02_region_generation_load_balance.csv\n")
    f.write("- 04_double_node_join_full_edge_context.csv\n")
    f.write("- 05_region_pair_summary.csv\n")
    f.write("- 07_node_centrality_capacity_weighted.csv\n")
    f.write("- 08_edge_centrality_capacity_weighted.csv\n")
    f.write("- 09_single_edge_outage_results.csv\n")
    f.write("- 10_targeted_removal_results.csv\n")
    f.write("- 11_maintenance_budget_recommendations.csv\n")
    f.write("- graphics_data/ for AI graphics inputs\n")

print("Done. Outputs saved in:", OUT_DIR)
print("Open:", OUT_DIR / "00_READ_ME_FIRST.txt")
