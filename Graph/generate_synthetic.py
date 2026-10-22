"""Generate independently seeded NetworkX graph corpora without trimming edges."""

import argparse
import csv
import json
import random
from pathlib import Path
import networkx as nx
import torch
from tqdm import tqdm

if not __package__:
    # PyCharm Run executes this file directly, without a package context.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Graph"

from .data import DATA_ROOT, FAMILIES, SPLITS
from .ir import compile_graph
from .data import pairs_index


def generate_graph(family, seed, node_range, max_edges, max_retries=1000):
    rng = random.Random(seed)
    for retry in range(max_retries):
        n = rng.randint(*node_range)
        wiring_seed = rng.randrange(2**32)
        params = dict(n=n, wiring_seed=wiring_seed)
        if family == "er":
            p = rng.uniform(1.5, 9.0) / (n - 1)
            graph = nx.fast_gnp_random_graph(n, min(p, 1.0), seed=wiring_seed)
            params.update(p=min(p, 1.0))
        elif family == "ba":
            m = rng.randint(1, min(5, n - 1))
            graph = nx.barabasi_albert_graph(n, m, seed=wiring_seed)
            params.update(m=m)
        elif family == "ws":
            k = rng.choice([k for k in (2, 4, 6, 8) if k < n])
            p = 10 ** rng.uniform(-2.5, 0)
            graph = nx.watts_strogatz_graph(n, k, p, seed=wiring_seed)
            params.update(k=k, p=p)
        else:
            blocks = rng.randint(2, min(5, n))
            sizes = [1] * blocks
            for _ in range(n - blocks):
                sizes[rng.randrange(blocks)] += 1
            within = rng.uniform(0.08, 0.35)
            between = rng.uniform(0.001, 0.035)
            probs = [[within if i == j else between for j in range(blocks)] for i in range(blocks)]
            graph = nx.stochastic_block_model(sizes, probs, seed=wiring_seed)
            params.update(sizes=sizes, probabilities=probs)
        if graph.number_of_edges() > max_edges:
            continue
        permutation = list(range(n))
        rng.shuffle(permutation)
        graph = nx.relabel_nodes(graph, dict(enumerate(permutation)), copy=True)
        params.update(retries=retry, relabeling=permutation, networkx_version=nx.__version__)
        return graph, params
    raise RuntimeError(f"{family} seed={seed}: no graph <= {max_edges} edges after {max_retries} draws")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, default=DATA_ROOT / "synthetic_v1")
    p.add_argument("--seed", type=int, default=0)
    for split, count in zip(SPLITS, (3200, 400, 400, 400)):
        p.add_argument(f"--{split}_per_family", type=int, default=count)
    p.add_argument("--iid_nodes", type=int, nargs=2, default=(32, 96))
    p.add_argument("--ood_nodes", type=int, nargs=2, default=(112, 160))
    p.add_argument("--max_edges", type=int, default=512)
    p.add_argument("--max_retries", type=int, default=1000)
    a = p.parse_args(argv)
    if a.seed < 0 or a.max_edges < 0 or a.max_retries < 1:
        p.error("invalid seed/edge budget/retry count")
    if any(lo < 4 or hi < lo for lo, hi in (a.iid_nodes, a.ood_nodes)):
        p.error("node ranges must satisfy 4 <= minimum <= maximum")
    counts = {s: getattr(a, f"{s}_per_family") for s in SPLITS}
    if any(c < 0 for c in counts.values()):
        p.error("sample counts must be nonnegative")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    fields = ("path", "split", "family", "seed", "num_nodes", "num_edges", "mean_degree",
              "max_degree", "clustering_coefficient", "num_connected_components", "retries", "generator_params")
    with (a.output_dir / "manifest.csv").open("x", newline="", encoding="utf-8") as f, tqdm(
            total=4 * sum(counts.values()), unit="graph") as bar:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        sample_id = 0
        for split in SPLITS:
            for family in FAMILIES:
                folder = a.output_dir / split / family
                folder.mkdir(parents=True)
                for i in range(counts[split]):
                    seed = a.seed + sample_id
                    sample_id += 1
                    graph, params = generate_graph(family, seed,
                        a.ood_nodes if split == "ood" else a.iid_nodes, a.max_edges, a.max_retries)
                    n, e = graph.number_of_nodes(), graph.number_of_edges()
                    program = compile_graph(n, pairs_index(graph.edges()))
                    path = folder / f"{family}_{i:05d}.pt"
                    with path.open("xb") as stream:
                        torch.save(dict(num_nodes=n, edge_index=program.to_edge_index(),
                                        x=torch.empty((n, 0)), family=family, seed=seed,
                                        generator_params=params), stream)
                    writer.writerow(dict(path=path.relative_to(a.output_dir).as_posix(), split=split,
                        family=family, seed=seed, num_nodes=n, num_edges=e, mean_degree=2*e/n,
                        max_degree=max(dict(graph.degree()).values()),
                        clustering_coefficient=nx.average_clustering(graph),
                        num_connected_components=nx.number_connected_components(graph),
                        retries=params["retries"], generator_params=json.dumps(params, sort_keys=True)))
                    bar.set_description(f"{split}/{family}")
                    bar.update()
                f.flush()


if __name__ == "__main__":
    main()
