import textwrap
import time
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Main metric:
#   Weighted average path length using distance = 1 / capacity_mw.
#   Lower is better. Negative intervention-minus-baseline change is beneficial.
#
# Secondary metric:
#   Stranded load MW = load in components with no generator.
#
# Default:
#   R = 50 for quick screening.
#   Change R = 2000 for final report results.
#
# Outputs:
#   outputs_final_counterfactual/
#     tables...
#     figures_polished/
#       PNG and PDF figures
#       00_figure_index.csv
#
# Figure names follow the suggested report flow: 01_main_..., 02_main_..., A1_appendix_...
# =============================================================================


# -----------------------------
# User settings
# -----------------------------
SEED = 5470

# Use 50 for quick screening. Change to 2000 for final report.
R = 2000

BC_EDGE = ("BUS0161", "BUS0252")

# Candidate B-C reinforcement sizes, MW added to the sampled B-C line.
BC_UPGRADE_MW_GRID = [0, 250, 500, 750, 1000, 1500, 2000, 3000]

# Alternate path out of Region C.
ALT_EDGE = ("BUS0252", "BUS0014")
ALT_EDGE_CAPACITY_MW = 600

# Practical threshold for action.
PRACTICAL_THRESHOLD_PCT = -2.0

OUT_DIR = Path("outputs_final_counterfactual")
FIG_DIR = OUT_DIR / "figures_polished"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# -----------------------------
# Figure settings
# -----------------------------
FIG_DPI = 300

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": FIG_DPI,
    "font.size": 12,
    "axes.titlesize": 17,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 20,
})


FIGURE_INDEX = []


def add_figure_record(filename, title, suggested_use):
    FIGURE_INDEX.append({
        "filename_png": f"{filename}.png",
        "filename_pdf": f"{filename}.pdf",
        "title": title,
        "suggested_use": suggested_use,
    })


def wrap_label(label, width=28):
    return "\n".join(textwrap.wrap(str(label), width=width))


def format_axis(ax):
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def format_axis_xgrid(ax):
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, filename, title, suggested_use):
    png_path = FIG_DIR / f"{filename}.png"
    pdf_path = FIG_DIR / f"{filename}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    add_figure_record(filename, title, suggested_use)


# -----------------------------
# Data helpers
# -----------------------------
def load_data():
    nodes_path = Path("nodes.csv")
    edges_path = Path("edges.csv")

    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError("Put nodes.csv and edges.csv in the same folder as this script.")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    required_nodes = {"node_id", "kind", "region", "capacity_mw"}
    required_edges = {"from_id", "to_id", "capacity_mw"}

    missing_nodes = required_nodes - set(nodes.columns)
    missing_edges = required_edges - set(edges.columns)

    if missing_nodes:
        raise ValueError(f"nodes.csv missing required columns: {sorted(missing_nodes)}")
    if missing_edges:
        raise ValueError(f"edges.csv missing required columns: {sorted(missing_edges)}")

    node_ids = set(nodes["node_id"])
    for bus in list(BC_EDGE) + list(ALT_EDGE):
        if bus not in node_ids:
            raise ValueError(f"{bus} not found in nodes.csv. Check intervention endpoints.")

    existing_edges = {tuple(sorted((r["from_id"], r["to_id"]))) for _, r in edges.iterrows()}
    if tuple(sorted(BC_EDGE)) not in existing_edges:
        raise ValueError(f"B-C reinforcement edge {BC_EDGE} not found in edges.csv.")

    return nodes, edges


def build_graph(nodes_df, edges_df):
    edge_cols = edges_df[["from_id", "to_id", "capacity_mw"]].copy()
    graph = ig.Graph.DataFrame(
        edges=edge_cols,
        directed=False,
        vertices=nodes_df,
        use_vids=False,
    )
    capacities = edge_cols["capacity_mw"].astype(float).to_numpy()
    graph.es["capacity_mw"] = capacities.tolist()
    graph.es["cost"] = (1.0 / np.maximum(capacities, 1e-9)).tolist()
    return graph


def weighted_apl(graph):
    return float(graph.average_path_length(weights=graph.es["cost"], directed=False))


def stranded_load_mw(graph):
    comps = graph.connected_components(mode="weak")
    kinds = graph.vs["kind"]
    caps = graph.vs["capacity_mw"]

    stranded = 0.0
    for comp in comps:
        has_generator = any(kinds[i] == "generator" for i in comp)
        if not has_generator:
            stranded += sum(float(caps[i]) for i in comp if kinds[i] == "load")

    return float(stranded)


def add_capacity_to_edge(graph, edge, add_mw):
    g = graph.copy()
    eid = g.get_eid(edge[0], edge[1], directed=False, error=False)
    if eid == -1:
        raise ValueError(f"Edge not found for reinforcement: {edge}")

    new_capacity = float(g.es[eid]["capacity_mw"]) + float(add_mw)
    g.es[eid]["capacity_mw"] = new_capacity
    g.es[eid]["cost"] = 1.0 / max(new_capacity, 1e-9)
    return g


def add_alternate_edge(graph, edge, capacity_mw):
    g = graph.copy()
    u, v = edge
    names = set(g.vs["name"])

    if u not in names or v not in names:
        raise ValueError(f"Alternate-path endpoint not found: {edge}")

    eid = g.get_eid(u, v, directed=False, error=False)

    if eid == -1:
        g.add_edge(u, v)
        eid = g.get_eid(u, v, directed=False)
        g.es[eid]["capacity_mw"] = float(capacity_mw)
        g.es[eid]["cost"] = 1.0 / max(float(capacity_mw), 1e-9)
    else:
        new_capacity = float(g.es[eid]["capacity_mw"]) + float(capacity_mw)
        g.es[eid]["capacity_mw"] = new_capacity
        g.es[eid]["cost"] = 1.0 / max(new_capacity, 1e-9)

    return g


# -----------------------------
# Summary tables
# -----------------------------
def graph_summary(nodes, edges):
    g = build_graph(nodes, edges)

    degrees = np.array(g.degree(), dtype=float)
    strengths = np.array(g.strength(weights=g.es["capacity_mw"]), dtype=float)
    comps = g.connected_components(mode="weak")
    cap = edges["capacity_mw"].astype(float)
    density = (2 * g.ecount()) / (g.vcount() * (g.vcount() - 1))

    row = {
        "nodes": g.vcount(),
        "edges": g.ecount(),
        "density": density,
        "components": len(comps),
        "largest_component_nodes": max(comps.sizes()),
        "degree_min": degrees.min(),
        "degree_median": np.median(degrees),
        "degree_mean": degrees.mean(),
        "degree_max": degrees.max(),
        "strength_median_mw": np.median(strengths),
        "strength_mean_mw": strengths.mean(),
        "strength_max_mw": strengths.max(),
        "line_capacity_min_mw": cap.min(),
        "line_capacity_median_mw": cap.median(),
        "line_capacity_mean_mw": cap.mean(),
        "line_capacity_max_mw": cap.max(),
        "baseline_weighted_apl_observed": weighted_apl(g),
        "baseline_stranded_load_mw_observed": stranded_load_mw(g),
    }

    return pd.DataFrame([row])


def node_kind_summary(nodes):
    return (
        nodes.groupby("kind")
        .agg(
            node_count=("node_id", "size"),
            total_capacity_mw=("capacity_mw", "sum"),
            median_capacity_mw=("capacity_mw", "median"),
            mean_capacity_mw=("capacity_mw", "mean"),
        )
        .reset_index()
        .sort_values("kind")
    )


def region_generation_load_summary(nodes):
    rows = []
    for region, sub in nodes.groupby("region"):
        gen = sub.loc[sub["kind"] == "generator", "capacity_mw"].sum()
        load = sub.loc[sub["kind"] == "load", "capacity_mw"].sum()
        rows.append({
            "region": region,
            "generation_mw": gen,
            "load_mw": load,
            "net_generation_minus_load_mw": gen - load,
            "node_count": len(sub),
            "generator_count": int((sub["kind"] == "generator").sum()),
            "load_count": int((sub["kind"] == "load").sum()),
            "substation_count": int((sub["kind"] == "substation").sum()),
        })
    return pd.DataFrame(rows).sort_values("region")


def region_pair_capacity_summary(nodes, edges):
    node_region = nodes.set_index("node_id")["region"].to_dict()
    df = edges.copy()
    df["from_region"] = df["from_id"].map(node_region)
    df["to_region"] = df["to_id"].map(node_region)
    df["region_pair"] = df.apply(
        lambda r: "-".join(sorted([str(r["from_region"]), str(r["to_region"])])),
        axis=1,
    )
    out = (
        df.groupby("region_pair")
        .agg(
            line_count=("capacity_mw", "size"),
            total_capacity_mw=("capacity_mw", "sum"),
            mean_capacity_mw=("capacity_mw", "mean"),
            median_capacity_mw=("capacity_mw", "median"),
            max_capacity_mw=("capacity_mw", "max"),
        )
        .reset_index()
        .sort_values("region_pair")
    )
    return out


def centrality_summary(nodes, edges):
    g = build_graph(nodes, edges)
    degree = np.array(g.degree(), dtype=float)
    strength = np.array(g.strength(weights=g.es["capacity_mw"]), dtype=float)
    bet = np.array(g.betweenness(weights=g.es["cost"], directed=False), dtype=float)

    bet_norm = bet / bet.max() if bet.max() > 0 else bet

    df = pd.DataFrame({
        "node_id": g.vs["name"],
        "kind": g.vs["kind"],
        "region": g.vs["region"],
        "degree": degree,
        "strength_mw": strength,
        "weighted_betweenness_norm": bet_norm,
    })
    return df.sort_values("weighted_betweenness_norm", ascending=False)


# -----------------------------
# Monte Carlo optimizer
# -----------------------------
def run_optimizer_mc(nodes, edges, R, seed):
    rng = np.random.default_rng(seed)
    base_capacity = edges["capacity_mw"].clip(lower=1).astype(float).to_numpy()

    records = []
    t0 = time.time()

    for rep in range(1, R + 1):
        sampled_edges = edges.copy()
        sampled_capacity = rng.poisson(lam=base_capacity).astype(float)
        sampled_edges["capacity_mw"] = np.clip(sampled_capacity, 1.0, None)

        g0 = build_graph(nodes, sampled_edges)
        base_apl = weighted_apl(g0)
        base_stranded = stranded_load_mw(g0)

        for add_mw in BC_UPGRADE_MW_GRID:
            g_int = add_capacity_to_edge(g0, BC_EDGE, add_mw)
            int_apl = weighted_apl(g_int)
            int_stranded = stranded_load_mw(g_int)

            records.append({
                "replicate": rep,
                "strategy": "BC_capacity_sweep",
                "option": f"BC_plus_{int(add_mw)}MW",
                "bc_added_mw": add_mw,
                "alternate_edge_capacity_mw": 0,
                "baseline_weighted_apl": base_apl,
                "intervention_weighted_apl": int_apl,
                "delta_weighted_apl": int_apl - base_apl,
                "pct_delta_weighted_apl": (int_apl - base_apl) / base_apl * 100.0,
                "baseline_stranded_load_mw": base_stranded,
                "intervention_stranded_load_mw": int_stranded,
                "delta_stranded_load_mw": int_stranded - base_stranded,
            })

        g_alt = add_alternate_edge(g0, ALT_EDGE, ALT_EDGE_CAPACITY_MW)
        alt_apl = weighted_apl(g_alt)
        alt_stranded = stranded_load_mw(g_alt)

        records.append({
            "replicate": rep,
            "strategy": "alternate_path",
            "option": f"ALT_{ALT_EDGE[0]}_{ALT_EDGE[1]}_{int(ALT_EDGE_CAPACITY_MW)}MW",
            "bc_added_mw": np.nan,
            "alternate_edge_capacity_mw": ALT_EDGE_CAPACITY_MW,
            "baseline_weighted_apl": base_apl,
            "intervention_weighted_apl": alt_apl,
            "delta_weighted_apl": alt_apl - base_apl,
            "pct_delta_weighted_apl": (alt_apl - base_apl) / base_apl * 100.0,
            "baseline_stranded_load_mw": base_stranded,
            "intervention_stranded_load_mw": alt_stranded,
            "delta_stranded_load_mw": alt_stranded - base_stranded,
        })

        if rep % 10 == 0 or rep == R:
            elapsed = time.time() - t0
            print(f"  completed {rep:,}/{R:,} replicates in {elapsed:,.1f}s", flush=True)

    return pd.DataFrame(records)


def summarize_effects(effects):
    rows = []

    for option, sub in effects.groupby("option"):
        delta = sub["delta_weighted_apl"].to_numpy()
        pct = sub["pct_delta_weighted_apl"].to_numpy()
        stranded = sub["delta_stranded_load_mw"].to_numpy()

        ci_delta = np.percentile(delta, [2.5, 97.5])
        ci_pct = np.percentile(pct, [2.5, 97.5])
        ci_stranded = np.percentile(stranded, [2.5, 97.5])
        first = sub.iloc[0]

        rows.append({
            "strategy": first["strategy"],
            "option": option,
            "R": len(sub),
            "bc_added_mw": first["bc_added_mw"],
            "alternate_edge_capacity_mw": first["alternate_edge_capacity_mw"],
            "mean_delta_weighted_apl": delta.mean(),
            "ci95_low_delta_weighted_apl": ci_delta[0],
            "ci95_high_delta_weighted_apl": ci_delta[1],
            "mean_pct_delta_weighted_apl": pct.mean(),
            "ci95_low_pct_delta_weighted_apl": ci_pct[0],
            "ci95_high_pct_delta_weighted_apl": ci_pct[1],
            "mean_delta_stranded_load_mw": stranded.mean(),
            "ci95_low_delta_stranded_load_mw": ci_stranded[0],
            "ci95_high_delta_stranded_load_mw": ci_stranded[1],
            "ci_crosses_zero": bool(ci_delta[0] <= 0 <= ci_delta[1]),
            "statistically_beneficial": bool(ci_delta[1] < 0),
            "clears_practical_threshold": bool(pct.mean() <= PRACTICAL_THRESHOLD_PCT),
        })

    return pd.DataFrame(rows).sort_values(["strategy", "mean_delta_weighted_apl"])


def build_decision_table(summary):
    bc = summary[summary["strategy"] == "BC_capacity_sweep"].copy()
    alt = summary[summary["strategy"] == "alternate_path"].copy()
    bc_nonzero = bc[bc["bc_added_mw"] > 0].copy()

    rows = []

    def add_row(label, df):
        if df.empty:
            rows.append({
                "decision_item": label,
                "option": "none_found",
                "mean_pct_delta_weighted_apl": np.nan,
                "ci95_low_pct_delta_weighted_apl": np.nan,
                "ci95_high_pct_delta_weighted_apl": np.nan,
                "statistically_beneficial": False,
                "clears_practical_threshold": False,
            })
        else:
            r = df.iloc[0]
            rows.append({
                "decision_item": label,
                "option": r["option"],
                "mean_pct_delta_weighted_apl": r["mean_pct_delta_weighted_apl"],
                "ci95_low_pct_delta_weighted_apl": r["ci95_low_pct_delta_weighted_apl"],
                "ci95_high_pct_delta_weighted_apl": r["ci95_high_pct_delta_weighted_apl"],
                "statistically_beneficial": r["statistically_beneficial"],
                "clears_practical_threshold": r["clears_practical_threshold"],
            })

    add_row("best_BC_upgrade_by_mean_APL", bc_nonzero.sort_values("mean_delta_weighted_apl").head(1))
    add_row("smallest_BC_upgrade_with_CI_below_zero", bc_nonzero[bc_nonzero["statistically_beneficial"]].sort_values("bc_added_mw").head(1))
    add_row("smallest_BC_upgrade_clearing_practical_threshold", bc_nonzero[bc_nonzero["clears_practical_threshold"]].sort_values("bc_added_mw").head(1))
    add_row("alternate_path_option", alt)

    return pd.DataFrame(rows)


# -----------------------------
# Polished figures
# -----------------------------
def make_polished_figures(nodes, edges, effects, summary, region_pairs, node_summary):
    bc = summary[summary["strategy"] == "BC_capacity_sweep"].copy().sort_values("bc_added_mw")
    alt = summary[summary["strategy"] == "alternate_path"].copy()
    bc_nonzero = bc[bc["bc_added_mw"] > 0].copy()

    best_bc_option = bc_nonzero.sort_values("mean_delta_weighted_apl").iloc[0]["option"]
    alt_option = alt.iloc[0]["option"]

    best_bc_effects = effects[effects["option"] == best_bc_option]
    alt_effects = effects[effects["option"] == alt_option]

    # -------------------------------------------------------------------------
    # Recommended Figure 1: Full network with zoomed intervention inset
    # -------------------------------------------------------------------------
    title = "Full network with zoomed B-C and alternate-path intervention view"

    # Main panel keeps the whole network. Right panel gives a large, readable
    # zoomed view of the B-C bottleneck and alternate-path corridor.
    g = build_graph(nodes, edges)
    layout = np.array(g.layout_fruchterman_reingold(weights=g.es["capacity_mw"], niter=900).coords)
    node_names = list(g.vs["name"])
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    node_region = dict(zip(nodes["node_id"], nodes["region"]))
    node_kind = dict(zip(nodes["node_id"], nodes["kind"]))

    edge_key_to_capacity = {
        tuple(sorted((r["from_id"], r["to_id"]))): float(r["capacity_mw"])
        for _, r in edges.iterrows()
    }
    bc_observed_capacity = edge_key_to_capacity.get(tuple(sorted(BC_EDGE)), np.nan)

    key_buses = sorted(set(list(BC_EDGE) + list(ALT_EDGE)))
    key_indices = [name_to_idx[b] for b in key_buses]

    # Context nodes for zoom: intervention buses plus immediate neighbors.
    context_indices = set(key_indices)
    for idx in key_indices:
        context_indices.update(g.neighbors(idx))
    context_indices = sorted(context_indices)
    context_set = set(context_indices)

    # Zoom window around key buses and one-hop context.
    xy = layout[context_indices, :]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xspan = max(xmax - xmin, 0.1)
    yspan = max(ymax - ymin, 0.1)
    xpad = 0.38 * xspan
    ypad = 0.42 * yspan
    zoom_xlim = (xmin - xpad, xmax + xpad)
    zoom_ylim = (ymin - ypad, ymax + ypad)

    fig = plt.figure(figsize=(17.0, 9.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    # -------------------------
    # Left: full network context
    # -------------------------
    for e in g.es:
        u, v = e.tuple
        ax_full.plot(
            [layout[u, 0], layout[v, 0]],
            [layout[u, 1], layout[v, 1]],
            linewidth=0.42,
            alpha=0.14,
            zorder=1,
        )

    for region in sorted(nodes["region"].dropna().unique()):
        idx = [name_to_idx[n] for n in nodes.loc[nodes["region"] == region, "node_id"]]
        ax_full.scatter(layout[idx, 0], layout[idx, 1], s=16, alpha=0.55, label=f"Region {region}", zorder=2)

    # Highlight intervention edges on full network without labels.
    for edge, style, lw, label in [
        (BC_EDGE, "-", 3.4, "Existing B-C bottleneck"),
        (ALT_EDGE, "--", 3.4, "Proposed alternate path"),
    ]:
        u, v = edge
        i, j = name_to_idx[u], name_to_idx[v]
        ax_full.plot(
            [layout[i, 0], layout[j, 0]],
            [layout[i, 1], layout[j, 1]],
            linestyle=style,
            linewidth=lw,
            label=label,
            zorder=5,
        )

    for bus in key_buses:
        idx = name_to_idx[bus]
        ax_full.scatter([layout[idx, 0]], [layout[idx, 1]], s=95, edgecolor="black", linewidth=1.1, zorder=6)

    # Show the zoom box on the full network.
    rect = plt.Rectangle(
        (zoom_xlim[0], zoom_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        zoom_ylim[1] - zoom_ylim[0],
        fill=False,
        linewidth=2.0,
        linestyle=":",
        zorder=7,
    )
    ax_full.add_patch(rect)

    ax_full.set_title("Full 300-bus network context", pad=12)
    ax_full.set_axis_off()
    ax_full.legend(loc="lower left", frameon=True, ncol=2)

    # -------------------------
    # Right: zoomed intervention view
    # -------------------------
    for e in g.es:
        u, v = e.tuple
        if u in context_set and v in context_set:
            ax_zoom.plot(
                [layout[u, 0], layout[v, 0]],
                [layout[u, 1], layout[v, 1]],
                linewidth=0.9,
                alpha=0.28,
                zorder=1,
            )

    context_names = [node_names[i] for i in context_indices]
    context_df = nodes[nodes["node_id"].isin(context_names)].copy()

    for region in sorted(context_df["region"].dropna().unique()):
        idx = [name_to_idx[n] for n in context_df.loc[context_df["region"] == region, "node_id"]]
        ax_zoom.scatter(layout[idx, 0], layout[idx, 1], s=58, alpha=0.68, label=f"Region {region}", zorder=2)

    # Highlight the two strategies in the zoom.
    def plot_strategy_edge(ax, edge, label, linestyle="-", linewidth=5.2):
        u, v = edge
        i, j = name_to_idx[u], name_to_idx[v]
        ax.plot(
            [layout[i, 0], layout[j, 0]],
            [layout[i, 1], layout[j, 1]],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.96,
            label=label,
            zorder=6,
        )

    plot_strategy_edge(ax_zoom, BC_EDGE, "A: reinforce existing B-C line", linestyle="-", linewidth=5.6)
    plot_strategy_edge(ax_zoom, ALT_EDGE, "B: add alternate path", linestyle="--", linewidth=5.2)

    for bus in key_buses:
        idx = name_to_idx[bus]
        ax_zoom.scatter(
            [layout[idx, 0]],
            [layout[idx, 1]],
            s=260,
            edgecolor="black",
            linewidth=1.8,
            zorder=8,
        )

    # Labels are intentionally placed in the margins of the zoom panel,
    # away from the nodes. Arrows point to the buses.
    label_positions = {
        BC_EDGE[0]: (0.06, 0.88),   # BUS0161
        BC_EDGE[1]: (0.78, 0.88),   # BUS0252
        ALT_EDGE[0]: (0.78, 0.88),  # shared endpoint if same as BUS0252
        ALT_EDGE[1]: (0.08, 0.12),  # BUS0014
    }

    for bus in key_buses:
        idx = name_to_idx[bus]
        tx, ty = label_positions.get(bus, (0.10, 0.50))
        ax_zoom.annotate(
            f"{bus}\nRegion {node_region.get(bus, '?')}\n{node_kind.get(bus, '?')}",
            xy=(layout[idx, 0], layout[idx, 1]),
            xycoords="data",
            xytext=(tx, ty),
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=10.2,
            weight="bold",
            linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.36", fc="white", ec="black", lw=0.85, alpha=0.97),
            arrowprops=dict(arrowstyle="-", lw=1.2, alpha=0.9, shrinkA=5, shrinkB=8),
            zorder=10,
        )

    # Strategy label boxes placed at top/bottom of zoom, separate from node labels.
    ax_zoom.text(
        0.50,
        0.985,
        f"Strategy A: reinforce {BC_EDGE[0]}-{BC_EDGE[1]} | observed {bc_observed_capacity:,.0f} MW | tested +{min(BC_UPGRADE_MW_GRID):,.0f} to +{max(BC_UPGRADE_MW_GRID):,.0f} MW",
        transform=ax_zoom.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="black", lw=0.75, alpha=0.96),
        zorder=11,
    )

    ax_zoom.text(
        0.50,
        0.015,
        f"Strategy B: add alternate path {ALT_EDGE[0]}-{ALT_EDGE[1]} | {ALT_EDGE_CAPACITY_MW:,.0f} MW",
        transform=ax_zoom.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.2,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="black", lw=0.75, alpha=0.96),
        zorder=11,
    )

    ax_zoom.set_xlim(*zoom_xlim)
    ax_zoom.set_ylim(*zoom_ylim)
    ax_zoom.set_title("Zoomed intervention corridor", pad=12)
    ax_zoom.set_axis_off()

    fig.suptitle("Figure 1. Full Network with Zoomed View of Tested Upgrade Strategies", y=1.02)
    save_figure(fig, "01_main_intervention_map", title, "Main report: full network plus zoomed B-C and alternate-path intervention map")

    # -------------------------------------------------------------------------
    # Recommended Figure 2: Baseline APL distribution
    # -------------------------------------------------------------------------
    title = "Baseline weighted average path length under capacity uncertainty"
    baseline = effects.drop_duplicates("replicate")[["replicate", "baseline_weighted_apl"]]
    fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
    ax.hist(baseline["baseline_weighted_apl"], bins=28, alpha=0.85)
    ax.axvline(baseline["baseline_weighted_apl"].mean(), linestyle="--", linewidth=2, label=f"Mean = {baseline['baseline_weighted_apl'].mean():.5f}")
    ax.set_title("Figure 2. Baseline Weighted APL Distribution", pad=14)
    ax.set_xlabel("Baseline weighted average path length")
    ax.set_ylabel("Monte Carlo replicates")
    ax.legend(loc="best", frameon=True)
    format_axis(ax)
    save_figure(fig, "A1_appendix_baseline_weighted_apl_distribution", title, "Appendix or method check: show uncertainty model before interventions")

    # -------------------------------------------------------------------------
    # Recommended Figure 3: Capacity sweep with confidence intervals
    # -------------------------------------------------------------------------
    title = "B-C reinforcement capacity sweep"
    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    x = bc["bc_added_mw"].astype(float)
    y = bc["mean_pct_delta_weighted_apl"]
    low = bc["mean_pct_delta_weighted_apl"] - bc["ci95_low_pct_delta_weighted_apl"]
    high = bc["ci95_high_pct_delta_weighted_apl"] - bc["mean_pct_delta_weighted_apl"]

    ax.errorbar(x, y, yerr=[low, high], fmt="o-", capsize=6, linewidth=2.5, markersize=7)
    ax.axhline(0, linestyle="--", linewidth=2)
    ax.axhline(PRACTICAL_THRESHOLD_PCT, linestyle=":", linewidth=2.5)
    ax.set_title("Figure 3. B-C Reinforcement Capacity Sweep", pad=14)
    ax.set_xlabel("Additional capacity on BUS0161-BUS0252, MW")
    ax.set_ylabel("% change in weighted average path length\nnegative is better")
    ax.text(
        0.02,
        0.03,
        f"Reference lines: 0% = no improvement; {PRACTICAL_THRESHOLD_PCT:.0f}% = practical threshold",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
    )
    format_axis(ax)
    save_figure(fig, "02_main_bc_capacity_sweep_ci", title, "Main report: B-C optimization result")

    # -------------------------------------------------------------------------
    # Recommended Figure 4: Best B-C vs alternate path histogram
    # -------------------------------------------------------------------------
    title = "Distribution of best B-C reinforcement versus alternate path"
    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    ax.hist(best_bc_effects["pct_delta_weighted_apl"], bins=24, alpha=0.58, label=f"Best B-C: {best_bc_option}")
    ax.hist(alt_effects["pct_delta_weighted_apl"], bins=24, alpha=0.58, label=f"Alternate path: {alt_option}")
    ax.axvline(0, linestyle="--", linewidth=2)
    ax.axvline(PRACTICAL_THRESHOLD_PCT, linestyle=":", linewidth=2.5)
    ax.set_title("Figure 4. Monte Carlo Effect Distribution", pad=14)
    ax.set_xlabel("% change in weighted average path length\nnegative is better")
    ax.set_ylabel("Monte Carlo replicates")
    ax.legend(loc="best", frameon=True)
    format_axis(ax)
    save_figure(fig, "03_main_best_bc_vs_alternate_histogram", title, "Main report: intervention comparison distribution")

    # -------------------------------------------------------------------------
    # Recommended Figure 5: All options mean effect with CI
    # -------------------------------------------------------------------------
    title = "Mean effect and 95% confidence interval for all tested options"
    compare = pd.concat([bc, alt], ignore_index=True).sort_values("mean_pct_delta_weighted_apl", ascending=True)
    fig_height = max(6.8, 0.55 * len(compare) + 2.0)
    fig, ax = plt.subplots(figsize=(12.5, fig_height), constrained_layout=True)
    ypos = np.arange(len(compare))
    means = compare["mean_pct_delta_weighted_apl"].to_numpy()
    left_err = means - compare["ci95_low_pct_delta_weighted_apl"].to_numpy()
    right_err = compare["ci95_high_pct_delta_weighted_apl"].to_numpy() - means
    labels = [wrap_label(x, 30) for x in compare["option"]]

    ax.errorbar(means, ypos, xerr=[left_err, right_err], fmt="o", capsize=5, linewidth=2.2, markersize=7)
    ax.axvline(0, linestyle="--", linewidth=2)
    ax.axvline(PRACTICAL_THRESHOLD_PCT, linestyle=":", linewidth=2.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Figure 5. Mean Intervention Effect with 95% CI", pad=14)
    ax.set_xlabel("% change in weighted average path length\nnegative is better")
    format_axis_xgrid(ax)
    save_figure(fig, "04_main_all_options_mean_ci", title, "Main report or appendix: all intervention comparison")

    # -------------------------------------------------------------------------
    # Figure 6: Boxplot of all interventions
    # -------------------------------------------------------------------------
    title = "Boxplot of all intervention effect distributions"
    ordered = []
    labels = []
    for add_mw in BC_UPGRADE_MW_GRID:
        opt = f"BC_plus_{int(add_mw)}MW"
        sub = effects[effects["option"] == opt]["pct_delta_weighted_apl"]
        if len(sub) > 0:
            ordered.append(sub)
            labels.append(f"B-C\n+{add_mw} MW")
    ordered.append(alt_effects["pct_delta_weighted_apl"])
    labels.append("Alt\npath")

    fig_width = max(12.5, 0.75 * len(labels) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0), constrained_layout=True)
    ax.boxplot(ordered, vert=True, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0)
    ax.axhline(0, linestyle="--", linewidth=2)
    ax.axhline(PRACTICAL_THRESHOLD_PCT, linestyle=":", linewidth=2.5)
    ax.set_title("Figure 6. Distribution of Effects Across Upgrade Options", pad=14)
    ax.set_xlabel("Intervention")
    ax.set_ylabel("% change in weighted average path length\nnegative is better")
    format_axis(ax)
    save_figure(fig, "A2_appendix_all_interventions_boxplot", title, "Appendix: robustness and distribution view")

    # -------------------------------------------------------------------------
    # Figure 7: Diminishing returns
    # -------------------------------------------------------------------------
    title = "Diminishing returns from adding B-C capacity"
    bc2 = bc[bc["bc_added_mw"] > 0].copy().sort_values("bc_added_mw")
    bc2["incremental_improvement_pct_points"] = (
        bc2["mean_pct_delta_weighted_apl"].shift(1) - bc2["mean_pct_delta_weighted_apl"]
    )
    bc2.iloc[0, bc2.columns.get_loc("incremental_improvement_pct_points")] = np.nan

    fig, ax = plt.subplots(figsize=(11.5, 6.6), constrained_layout=True)
    ax.plot(bc2["bc_added_mw"], bc2["incremental_improvement_pct_points"], marker="o", linewidth=2.5, markersize=7)
    ax.axhline(0, linestyle="--", linewidth=2)
    ax.set_title("Figure 7. Diminishing Returns from Extra B-C Capacity", pad=14)
    ax.set_xlabel("Additional capacity on BUS0161-BUS0252, MW")
    ax.set_ylabel("Extra improvement compared with previous tested size\npercentage points")
    format_axis(ax)
    save_figure(fig, "A3_appendix_bc_diminishing_returns", title, "Appendix or discussion: marginal benefit of more MW")

    # -------------------------------------------------------------------------
    # Figure 8: Region-pair capacity heatmap
    # -------------------------------------------------------------------------
    title = "Region-pair capacity heatmap"
    pairs = region_pairs.copy()
    regions = sorted({x for pair in pairs["region_pair"] for x in pair.split("-")})
    mat = pd.DataFrame(0.0, index=regions, columns=regions)
    for r in pairs.itertuples():
        a, b = r.region_pair.split("-")
        mat.loc[a, b] = r.total_capacity_mw
        mat.loc[b, a] = r.total_capacity_mw

    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    im = ax.imshow(np.log1p(mat.values), aspect="equal")
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions)
    ax.set_title("Figure 8. Region-Pair Transmission Capacity", pad=14)
    for i in range(len(regions)):
        for j in range(len(regions)):
            ax.text(j, i, f"{mat.values[i, j]:,.0f}", ha="center", va="center", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("log1p(total capacity MW)")
    save_figure(fig, "A4_appendix_region_pair_capacity_heatmap", title, "Network context: regional bottleneck structure")

    # -------------------------------------------------------------------------
    # Figure 9: Degree vs weighted betweenness
    # -------------------------------------------------------------------------
    title = "Degree versus normalized capacity-weighted betweenness"
    top = node_summary.head(8)
    fig, ax = plt.subplots(figsize=(10.5, 6.6), constrained_layout=True)
    ax.scatter(node_summary["degree"], node_summary["weighted_betweenness_norm"], alpha=0.65, s=55)
    for r in top.itertuples():
        ax.annotate(
            r.node_id,
            (r.degree, r.weighted_betweenness_norm),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Figure 9. Visible Connectivity vs Structural Bottleneck Role", pad=14)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Normalized capacity-weighted betweenness")
    format_axis(ax)
    save_figure(fig, "A5_appendix_degree_vs_betweenness", title, "Network context: why centrality matters")

    # -------------------------------------------------------------------------
    # Figure 10: Line capacity distribution
    # -------------------------------------------------------------------------
    title = "Transmission line capacity distribution"
    fig, ax = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    cap = edges["capacity_mw"].astype(float)
    ax.hist(cap, bins=32, alpha=0.85)
    ax.axvline(cap.median(), linestyle="--", linewidth=2, label=f"Median = {cap.median():.0f} MW")
    ax.axvline(cap.max(), linestyle=":", linewidth=2.5, label=f"Max = {cap.max():.0f} MW")
    ax.set_title("Figure 10. Transmission Line Capacity Distribution", pad=14)
    ax.set_xlabel("Line capacity, MW")
    ax.set_ylabel("Number of lines")
    ax.legend(loc="best", frameon=True)
    format_axis(ax)
    save_figure(fig, "A6_appendix_line_capacity_histogram", title, "Appendix: edge capacity context")

    # -------------------------------------------------------------------------
    # Figure 11: Executive composite
    # -------------------------------------------------------------------------
    title = "Executive composite summary"
    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(best_bc_effects["pct_delta_weighted_apl"], bins=24, alpha=0.58, label=f"Best B-C: {best_bc_option}")
    ax1.hist(alt_effects["pct_delta_weighted_apl"], bins=24, alpha=0.58, label="Alternate path")
    ax1.axvline(0, linestyle="--", linewidth=2)
    ax1.axvline(PRACTICAL_THRESHOLD_PCT, linestyle=":", linewidth=2.5)
    ax1.set_title("Effect distribution")
    ax1.set_xlabel("% APL change")
    ax1.set_ylabel("Replicates")
    ax1.legend(fontsize=9)
    format_axis(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(x, y, yerr=[low, high], fmt="o-", capsize=5, linewidth=2.4, markersize=6)
    ax2.axhline(0, linestyle="--", linewidth=2)
    ax2.axhline(PRACTICAL_THRESHOLD_PCT, linestyle=":", linewidth=2.5)
    ax2.set_title("B-C capacity sweep")
    ax2.set_xlabel("Added B-C MW")
    ax2.set_ylabel("% APL change")
    format_axis(ax2)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    lines = ["B-C CAPACITY OPTIMIZER SUMMARY", "-" * 96]
    for row in compare.head(6).itertuples():
        lines.append(
            f"{row.option}: mean {row.mean_pct_delta_weighted_apl:.3f}%, "
            f"95% CI [{row.ci95_low_pct_delta_weighted_apl:.3f}%, "
            f"{row.ci95_high_pct_delta_weighted_apl:.3f}%], "
            f"real={row.statistically_beneficial}, practical={row.clears_practical_threshold}"
        )
    lines.append("-" * 96)
    lines.append(f"Best B-C option by mean APL: {best_bc_option}")
    lines.append(f"Alternate path option: {alt_option}")
    lines.append(f"R = {R:,}; negative percent change means shorter weighted structural routes.")
    ax3.text(0.01, 0.94, "\n".join(lines), family="monospace", fontsize=11.5, va="top")

    fig.suptitle("Counterfactual Upgrade Comparison: B-C Capacity Sweep and Alternate Path", y=1.02)
    save_figure(fig, "A7_appendix_executive_composite", title, "Appendix or presentation summary")


# -----------------------------
# Main
# -----------------------------
def main():
    nodes, edges = load_data()

    print("Loaded data.")
    print(f"  Nodes: {len(nodes):,}")
    print(f"  Edges: {len(edges):,}")
    print(f"  Monte Carlo R: {R:,}")
    print(f"  B-C edge: {BC_EDGE[0]}-{BC_EDGE[1]}")
    print(f"  B-C upgrade grid: {BC_UPGRADE_MW_GRID}")
    print(f"  Alternate path: {ALT_EDGE[0]}-{ALT_EDGE[1]} at {ALT_EDGE_CAPACITY_MW:,} MW")
    print()

    gsum = graph_summary(nodes, edges)
    kind_summary = node_kind_summary(nodes)
    region_balance = region_generation_load_summary(nodes)
    region_pairs = region_pair_capacity_summary(nodes, edges)
    node_centrality = centrality_summary(nodes, edges)

    gsum.to_csv(OUT_DIR / "01_graph_summary.csv", index=False)
    kind_summary.to_csv(OUT_DIR / "02_node_kind_summary.csv", index=False)
    region_balance.to_csv(OUT_DIR / "03_region_generation_load_summary.csv", index=False)
    region_pairs.to_csv(OUT_DIR / "04_region_pair_capacity_summary.csv", index=False)
    node_centrality.to_csv(OUT_DIR / "05_node_centrality_summary.csv", index=False)

    effects = run_optimizer_mc(nodes, edges, R=R, seed=SEED)
    summary = summarize_effects(effects)
    decision = build_decision_table(summary)

    effects.to_csv(OUT_DIR / "06_bc_capacity_optimizer_effects.csv", index=False)
    summary.to_csv(OUT_DIR / "07_bc_capacity_optimizer_summary.csv", index=False)
    decision.to_csv(OUT_DIR / "08_bc_capacity_optimizer_decision_table.csv", index=False)

    make_polished_figures(nodes, edges, effects, summary, region_pairs, node_centrality)

    pd.DataFrame(FIGURE_INDEX).to_csv(FIG_DIR / "00_figure_index.csv", index=False)

    print("\nGraph summary:")
    print(gsum.to_string(index=False))

    print("\nRegion generation/load summary:")
    print(region_balance.to_string(index=False))

    print("\nTop 10 weighted-betweenness buses:")
    print(node_centrality.head(10).to_string(index=False))

    print("\nB-C capacity optimizer summary:")
    print(summary.to_string(index=False))

    print("\nDecision table:")
    print(decision.to_string(index=False))

    print("\nSaved tables in:")
    print(f"  {OUT_DIR.resolve()}")

    print("\nSaved polished figures in:")
    print(f"  {FIG_DIR.resolve()}")
    for p in sorted(FIG_DIR.iterdir()):
        print(f"  - {p.name}")

    print("\nRecommended main-report figures:")
    print("  1. 01_main_intervention_map.png")
    print("  2. 02_main_bc_capacity_sweep_ci.png")
    print("  3. 03_main_best_bc_vs_alternate_histogram.png")
    print("  4. 04_main_all_options_mean_ci.png")
    print("\nRecommended appendix figures:")
    print("  A1. A1_appendix_baseline_weighted_apl_distribution.png")
    print("  A2. A2_appendix_all_interventions_boxplot.png")
    print("  A3. A3_appendix_bc_diminishing_returns.png")
    print("  A4. A4_appendix_region_pair_capacity_heatmap.png")
    print("  A5. A5_appendix_degree_vs_betweenness.png")

    print("\nNext decision:")
    print("  Review the R=50 figures first.")
    print("  If the ranking and endpoint choices look right, change R = 50 to R = 2000 and rerun.")


if __name__ == "__main__":
    main()
