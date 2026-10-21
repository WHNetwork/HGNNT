"""Generate and save synthetic_v1 with XGI, without compiling or training.

Requires numpy, torch, tqdm and an XGI version exposing random_nested_hypergraph.
Default corpus: per family 3200 train / 400 val / 400 test / 400 OOD graphs.
IID nodes: 64..256; OOD nodes: 288..384 (a disjoint size distribution).
Edge counts and sizes are natural generator outcomes, not exact constraints.
No edges are truncated to force a count or cardinality distribution.

API reference: https://xgi.readthedocs.io/en/latest/api/generators/xgi.generators.random.html
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import xgi
from tqdm import tqdm


FAMILIES = ("random", "chung_lu", "dcsbm", "watts_strogatz", "nested")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "datasets" / "synthetic_v1"


def degree_sequence(nodes, total, sigma, rng):
    """Positive integer expected degrees with exactly the requested stub total.

    All callers choose edge sizes/counts so total >= len(nodes). Lognormal
    weights vary degree heterogeneity; clipping bounds the sampled hub weights.
    This constructs generator parameters, not hypergraph incidences.
    """
    weights = np.clip(rng.lognormal(0.0, sigma, len(nodes)), 0.1, 6.0)
    degrees = 1 + rng.multinomial(total - len(nodes), weights / weights.sum())
    return {int(node): int(degree) for node, degree in zip(nodes, degrees)}


def generate_hypergraph(family, seed, ood=False):
    """One independent XGI draw and its actual arguments as Python metadata."""
    # Parameter draws and XGI wiring use separate reproducible random streams.
    rng = np.random.default_rng([seed, 1])
    n = int(rng.integers(288, 385) if ood else rng.integers(64, 257))
    target = int(rng.integers(128, 385) if ood else rng.integers(64, 257))
    params = {"n": n, "seed": seed}

    if family == "random":
        # XGI order = cardinality - 1. Different subsets/mixtures of sizes give
        # uniform, narrow and broad distributions without enumerating all edges.
        # Sizes <= 7 keep combination-index sampling probabilities well resolved
        # even for the largest OOD graph; other families supply larger edges.
        sizes = sorted(rng.choice(np.arange(2, 8), size=int(rng.integers(1, 7)),
                                  replace=False).tolist())
        alpha = float(rng.uniform(0.3, 3.0))
        weights = 0.2 / len(sizes) + 0.8 * rng.dirichlet([alpha] * len(sizes))
        ps = [float(target * weight / math.comb(n, size))
              for size, weight in zip(sizes, weights)]
        orders = [size - 1 for size in sizes]
        graph = xgi.fast_random_hypergraph(n, ps, order=orders, seed=seed)
        params.update(ps=ps, order=orders, mixture_alpha=alpha)

    elif family in ("chung_lu", "dcsbm"):
        # k2 are EXPECTED sizes: realized sizes need not equal these integers.
        # Min size 4 (CL) or 5 (DCSBM) ensures enough stubs for positive k1.
        low, high = (8, 20) if ood else (4 if family == "chung_lu" else 5, 12)
        sizes = rng.integers(low, high + 1, target)
        k2 = {edge: int(size) for edge, size in enumerate(sizes)}
        sigma = float(rng.uniform(1.2, 1.6) if ood else rng.uniform(0.15, 1.0))
        params.update(k2=k2, degree_sigma=sigma, degree_weight_clip=[0.1, 6.0])
        if family == "chung_lu":
            k1 = degree_sequence(range(n), int(sizes.sum()), sigma, rng)
            graph = xgi.chung_lu_hypergraph(k1, k2, seed=seed)
            params.update(k1=k1)
        else:
            communities = int(rng.integers(5, 9) if ood else rng.integers(2, 5))
            node_groups = rng.permutation(np.arange(n) % communities)
            edge_groups = rng.permutation(np.arange(target) % communities)
            g1 = {node: int(group) for node, group in enumerate(node_groups)}
            g2 = {edge: int(group) for edge, group in enumerate(edge_groups)}
            # Match node and edge community stub volumes BEFORE forming omega.
            volumes = np.array([sum(k2[e] for e in k2 if g2[e] == c)
                                for c in range(communities)], dtype=np.int64)
            k1 = {}
            for c in range(communities):
                nodes = np.flatnonzero(node_groups == c)
                k1.update(degree_sequence(nodes, int(volumes[c]), sigma, rng))
            mixing = float(rng.uniform(0.55, 0.85) if ood else rng.uniform(0.05, 0.35))
            # Both marginals equal volumes. sum(k1) = sum(k2) = sum(omega).
            # mixing=0 is block diagonal; mixing=1 is degree-based random mixing.
            mass = volumes.astype(float)
            omega = ((1 - mixing) * np.diag(mass)
                     + mixing * np.outer(mass, mass) / mass.sum())
            graph = xgi.dcsbm_hypergraph(k1, k2, g1, g2, omega, seed=seed)
            params.update(k1=k1, g1=g1, g2=g2, omega=omega.tolist(),
                          num_communities=communities, mixing=mixing)

    elif family == "watts_strogatz":
        d = int(rng.integers(13, 21) if ood else rng.integers(2, 13))
        # Current XGI ring_lattice emits n*k/2 edges, not n*k/d.
        # Use even k and constrain the lattice at its source, without trimming.
        cap = 512 if ood else 256
        choices = [k for k in (2, 4, 6, 8) if n * k // 2 <= cap]
        k = int(rng.choice(choices))
        l = int(rng.integers(0, d - 1))
        p = float(rng.uniform(0.65, 0.95) if ood
                  else np.exp(rng.uniform(np.log(0.005), np.log(0.35))))
        graph = xgi.watts_strogatz_hypergraph(n, d, k, l, p, seed=seed)
        # XGI rewiring can repeat a sampled node; set conversion below preserves
        # the resulting membership rather than inventing replacement members.
        params.update(d=d, k=k, l=l, p=p)

    elif family == "nested":
        d = int(rng.integers(7, 9) if ood else rng.integers(3, 7))
        target = int(rng.integers(256, 449) if ood else rng.integers(96, 241))
        # At most 2**d-d-1 edges per facet, INCLUDING the facet itself.
        # IID d<=6, OOD d<=8: even before deduplication there are <=494 edges.
        per_facet = 2**d - d - 1
        facets = max(1, round(target / per_facet))
        epsilon = float(rng.uniform(0.95, 1.0) if ood else rng.uniform(0.2, 0.85))
        graph = xgi.random_nested_hypergraph(n, facets, d, epsilon, seed=seed)
        params.update(m=facets, d=d, epsilon=epsilon)

    else:
        raise ValueError(f"Unknown hypergraph family: {family}")

    graph.add_nodes_from(range(n))  # retain all requested nodes, including isolates
    if family in ("random", "chung_lu", "dcsbm", "nested"):
        params["target_num_hyperedges"] = target
    params["xgi_version"] = xgi.__version__
    return graph, params


def to_tensor_data(graph, family, seed, params):
    """Relabel locally and remove exact duplicate member sets, never partial ones."""
    node_ids = {node: i for i, node in enumerate(sorted(graph.nodes))}
    members = {tuple(sorted({node_ids[node] for node in edge}))
               for edge in graph.edges.members()}
    # Canonical ordering makes serialization independent of XGI edge IDs/order.
    members = sorted(members, key=lambda edge: (len(edge), edge))
    pairs = [(node, edge) for edge, group in enumerate(members) for node in group]
    index = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()
    return {
        "num_nodes": len(node_ids),
        "num_hyperedges": len(members),
        "hyperedge_index": index,
        "x": torch.empty((len(node_ids), 0), dtype=torch.float32),
        "family": family,
        "seed": seed,
        "generator_params": params,
    }, [len(edge) for edge in members]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples_per_family", type=int, default=4000)
    parser.add_argument("--ood_per_family", type=int, default=400)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.samples_per_family < 1 or args.ood_per_family < 0:
        parser.error("seed/OOD count must be nonnegative; IID count must be positive")

    # Remainder goes to train; default counts are exactly 3200/400/400/400.
    held_out = args.samples_per_family // 10
    splits = (("train", args.samples_per_family - 2 * held_out),
              ("val", held_out), ("test", held_out), ("ood", args.ood_per_family))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ("path", "split", "family", "seed", "num_nodes", "num_hyperedges",
              "mean_hyperedge_size", "max_hyperedge_size", "mean_node_degree",
              "generator_params")
    # Exclusive creation prevents overwriting an existing experiment's manifest
    # or graph files. Use a new output directory for another corpus.
    total = len(FAMILIES) * (args.samples_per_family + args.ood_per_family)
    # One global bar avoids nested-bar rendering issues. Keep counts/timing before
    # the flexible ASCII bar, with no long postfix that tqdm might truncate.
    with (args.output_dir / "manifest.csv").open("x", newline="", encoding="utf-8") as stream, tqdm(
        total=total, unit="graph", dynamic_ncols=True, ascii=True,
        bar_format="{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}] |{bar}|",
    ) as progress:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        sample_id = 0
        for family in FAMILIES:
            for split, count in splits:
                folder = args.output_dir / split / family
                folder.mkdir(parents=True, exist_ok=True)
                progress.set_description_str(f"{split}/{family}")
                for index in range(count):
                    # Distinct seeds across ALL families and splits; independent
                    # draws, no replicated templates or copied graph structures.
                    seed = args.seed + sample_id
                    sample_id += 1
                    graph, params = generate_hypergraph(family, seed, ood=split == "ood")
                    data, sizes = to_tensor_data(graph, family, seed, params)
                    path = folder / f"{family}_{index:05d}.pt"
                    with path.open("xb") as graph_stream:
                        torch.save(data, graph_stream)
                    incidences = data["hyperedge_index"].shape[1]
                    writer.writerow({
                        "path": path.relative_to(args.output_dir).as_posix(),
                        "split": split, "family": family, "seed": seed,
                        "num_nodes": data["num_nodes"],
                        "num_hyperedges": data["num_hyperedges"],
                        "mean_hyperedge_size": incidences / max(len(sizes), 1),
                        "max_hyperedge_size": max(sizes, default=0),
                        "mean_node_degree": incidences / data["num_nodes"],
                        "generator_params": json.dumps(params, sort_keys=True),
                    })
                    # Count only graphs whose tensor file and manifest row were written.
                    progress.update(1)
                stream.flush()
                progress.write(f"{split}/{family}: saved {count} graphs")


if __name__ == "__main__":
    main()
