"""Offline LCC extraction for the four existing synthetic_v1 families.

Uses XGI node/edge incidence connectivity, not a graph expansion. No graph is
regenerated or filtered for being small. Ties retain the first component in
ascending source-node order. Empty edges have no incident node component and
are excluded. Source generator metadata is retained as provenance unchanged.

XGI API: https://xgi.readthedocs.io/en/latest/api/algorithms/xgi.algorithms.connected.html
"""

import argparse
import csv
from pathlib import Path

import torch
import xgi
from tqdm import tqdm

if not __package__:
    # PyCharm Run executes this file directly, without a package context.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Bert"

from .data import FAMILIES, SPLITS


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = REPO_ROOT / "Bert" / "datasets"


def extract_lcc(raw):
    """Return a relabeled LCC tensor dictionary, retaining other source fields."""
    num_nodes = int(raw["num_nodes"])
    index = raw["hyperedge_index"]
    inferred = int(index[1].max()) + 1 if index.shape[1] else 0
    num_edges = int(raw.get("num_hyperedges", inferred))
    members = [set() for _ in range(num_edges)]
    for node, edge in index.t().tolist():
        members[edge].add(node)
    graph = xgi.Hypergraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from({edge: sorted(group) for edge, group in enumerate(members) if group})
    component = xgi.largest_connected_component(graph) if num_nodes else set()
    node_map = {node: i for i, node in enumerate(sorted(component))}
    retained = [group for group in members if group and group <= component]
    pairs = [(node_map[node], edge) for edge, group in enumerate(retained)
             for node in sorted(group)]
    result = dict(raw)
    result.update(num_nodes=len(component), num_hyperedges=len(retained),
                  hyperedge_index=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous(),
                  x=torch.empty((len(component), 0), dtype=torch.float32))
    stats = {"original_num_nodes": num_nodes, "lcc_num_nodes": len(component),
             "original_num_hyperedges": num_edges, "lcc_num_hyperedges": len(retained),
             "lcc_node_ratio": len(component) / max(num_nodes, 1)}
    return result, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DATASETS / "synthetic_v1")
    parser.add_argument("--output_dir", type=Path, default=DATASETS / "synthetic_v1_lcc")
    args = parser.parse_args(argv)
    # Resolve relative arguments from the repository, not PyCharm's working dir.
    source = (REPO_ROOT / args.input_dir).resolve()
    destination = (REPO_ROOT / args.output_dir).resolve()
    print(f"LCC input directory: {source}", flush=True)
    print(f"LCC output directory: {destination}", flush=True)
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("input_dir and output_dir must be separate directory trees")
    paths = [(split, family, path) for split in SPLITS for family in FAMILIES
             for path in sorted((source / split / family).rglob("*.pt"))]
    if not paths:
        parser.error(
            f"No .pt files found under: {source}\n"
            f"Expected layout: {source / 'train' / 'random' / '*.pt'}\n"
            "Check that input_dir points to the existing synthetic_v1 data directory."
        )
    destination.mkdir(parents=True, exist_ok=True)
    fields = ("path", "split", "family", "original_num_nodes", "lcc_num_nodes",
              "original_num_hyperedges", "lcc_num_hyperedges", "lcc_node_ratio")
    # Exclusive output creation keeps both original data and prior LCC runs intact.
    with (destination / "lcc_stats.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, family, path in tqdm(
            paths, desc="LCC", dynamic_ncols=True, ascii=True,
            bar_format="{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}] |{bar}|",
        ):
            raw = torch.load(path, map_location="cpu", weights_only=True)
            result, stats = extract_lcc(raw)
            relative = path.relative_to(source)
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as graph_stream:
                torch.save(result, graph_stream)
            writer.writerow(dict(path=relative.as_posix(), split=split, family=family, **stats))


if __name__ == "__main__":
    main()
