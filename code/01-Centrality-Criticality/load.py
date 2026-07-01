"""Load the `power-grid` project network (Python).

Reads the CSVs in this folder and builds an undirected, weighted python-igraph
object: a regional electrical transmission grid of buses and lines. Run it
straight (``python load.py``) for a quick summary, or import ``load_grid()`` into
your own script.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import igraph as ig

HERE = Path(__file__).resolve().parent


def load_nodes() -> pd.DataFrame:
    """Node table: one row per bus."""
    return pd.read_csv(HERE / "nodes.csv")


def load_edges() -> pd.DataFrame:
    """Edge table: one row per transmission line."""
    return pd.read_csv(HERE / "edges.csv")


def load_regions() -> pd.DataFrame:
    """Control-area lookup table (join onto nodes by ``region``)."""
    return pd.read_csv(HERE / "regions.csv")


def load_grid(nodes: pd.DataFrame | None = None,
              edges: pd.DataFrame | None = None) -> ig.Graph:
    """Build the undirected, weighted grid graph (edge weight = ``capacity_mw``).

    The graph is a single static snapshot (no time dimension).
    """
    if nodes is None:
        nodes = load_nodes()
    if edges is None:
        edges = load_edges()
    g = ig.Graph.DataFrame(edges, directed=False, vertices=nodes, use_vids=False)
    g.es["weight"] = g.es["capacity_mw"]
    return g


if __name__ == "__main__":
    print("\n🔌 power-grid (Python)")
    print("   Undirected transmission grid; lines weighted by capacity (MW).\n")

    nodes = load_nodes()
    edges = load_edges()
    g = load_grid(nodes, edges)

    kinds = nodes["kind"].value_counts()
    print(f"✅ Loaded {len(nodes)} buses "
          f"({kinds.get('generator',0)} generators, {kinds.get('substation',0)} substations, "
          f"{kinds.get('load',0)} loads) and {len(edges)} lines.")
    print(f"🔗 Directed: {g.is_directed()} | total line capacity: "
          f"{edges['capacity_mw'].sum():,} MW")
    print("🎉 Graph ready. Object `g` is an undirected, weighted igraph.")
