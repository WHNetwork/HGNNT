"""HIR-SP ordinary-graph labels and topology-preserving sequence views.

Graph algorithms in this file construct supervision/negative filters only.
None of these statistics enter the sequence encoder.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import random

import networkx as nx
import torch

from .data import FAMILIES, load_graph, split_paths
from .ir import EdgeInstruction, GraphIRProgram, compile_graph


@dataclass
class StructuralGraph:
    path: str
    family: str
    program: GraphIRProgram
    graph: nx.Graph
    signature: tuple
    triangle: list
    bridge: list
    properties: dict


def load_record(path, max_length):
    """Accept the existing num_nodes/edge_index .pt schema, not hyperedges."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if any(key in raw for key in ("hyperedges", "hyperedge_index", "incidence_index")):
        raise ValueError(f"{path}: higher-order input is unsupported")
    data = load_graph(path)
    program = compile_graph(data.num_nodes, data.edge_index)
    graph = nx.Graph()
    graph.add_nodes_from(range(program.num_nodes))
    graph.add_edges_from(e.endpoints for e in program.instructions)
    isolates = list(nx.isolates(graph))
    length = 5 * program.num_edges + len(isolates) + 1
    if program.num_nodes > 65536 or length > max_length:
        raise ValueError(f"{path}: nodes={program.num_nodes}, token_length={length}, limit={max_length}; no truncation")
    triangles = nx.triangles(graph)
    bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
    # Explicit original instruction alignment, independent of NetworkX iteration.
    triangle = [int(bool(set(graph[u]) & set(graph[v]))) for u, v in
                (e.endpoints for e in program.instructions)]
    bridge = [int(tuple(sorted(e.endpoints)) in bridges) for e in program.instructions]
    components = nx.number_connected_components(graph)
    properties = dict(components=min(components, 3)-1 if components else None,
                      graph_triangle=int(any(triangles.values())), bipartite=int(nx.is_bipartite(graph)))
    # Unequal isomorphism invariants CERTIFY non-isomorphism. Equal signatures
    # are never assumed to be different graphs for contrastive negatives.
    signature = (program.num_nodes, program.num_edges, tuple(sorted(dict(graph.degree()).values())),
                 tuple(sorted(triangles.values())), tuple(sorted(map(len, nx.connected_components(graph)))))
    family = path.parent.name if path.parent.name in FAMILIES else str(raw.get("family", "unknown"))
    return StructuralGraph(str(path.resolve()), family, program, graph, signature, triangle, bridge, properties)


def discover(root, seed, max_graphs):
    """Use existing splits; fall back only for a flat corpus without split dirs."""
    root = Path(root)
    if all(any((root / split).rglob("*.pt")) for split in ("train", "val")):
        return {s: split_paths(root, s, max_graphs, seed,
                              families=("ba", "ws", "sbm") if s == "train" else FAMILIES)
                for s in ("train", "val", "test", "ood")}, "existing"
    # Preserve any explicit held-out splits when train/val are incomplete.
    held_out = {s: split_paths(root, s, max_graphs, seed) for s in ("test", "ood")}
    paths = sorted(path for path in root.rglob("*.pt")
                   if path.relative_to(root).parts[0] not in ("test", "ood"))
    random.Random(seed).shuffle(paths)
    cut = max(1, int(.8 * len(paths)))
    parts = dict(train=paths[:cut], val=paths[cut:], **held_out)
    parts["train"] = [p for p in parts["train"] if p.parent.name != "er"]
    if max_graphs is not None:
        parts = {s: rows[:max_graphs] for s, rows in parts.items()}
    return parts, "seeded_80_20_fallback_preserving_held_out"


def prepare_corpus(root, seed, max_graphs, max_length):
    paths, mode = discover(root, seed, max_graphs)
    records, buckets, removed = {}, {}, []
    # Train has precedence, then validation, test, OOD. Remove exact isomorphic
    # copies both within and across splits BEFORE generating any views.
    for split, files in paths.items():
        records[split] = []
        for path in files:
            item = load_record(path, max_length)
            if split == "train" and item.family == "er":
                continue
            matches = buckets.setdefault(item.signature, [])
            duplicate = next((old for old in matches if nx.is_isomorphic(old.graph, item.graph)), None)
            if duplicate is not None:
                removed.append(dict(path=item.path, duplicate_of=duplicate.path, split=split))
                continue
            matches.append(item)
            records[split].append(item)
    if not records["train"] or not records["val"]:
        raise ValueError("HIR-SP requires nonempty train and validation splits after exact deduplication")
    manifest = dict(root=str(Path(root).resolve()), split_mode=mode, seed=seed,
                    negative_policy="only unequal isomorphism invariants are certified negatives",
                    deduplication="exact NetworkX isomorphism within invariant buckets, all selected splits",
                    duplicates_removed=removed, splits={})
    for split, items in records.items():
        manifest["splits"][split] = [dict(path=i.path, family=i.family, nodes=i.program.num_nodes,
            edges=i.program.num_edges, token_length=5*i.program.num_edges+len(list(nx.isolates(i.graph)))+1,
            graph_labels=i.properties, triangle_counts=[i.triangle.count(c) for c in (0, 1)],
            bridge_counts=[i.bridge.count(c) for c in (0, 1)]) for i in items]
    manifest["distributions"] = {}
    for split, items in records.items():
        manifest["distributions"][split] = {}
        for family in sorted({item.family for item in items}):
            rows = [item for item in items if item.family == family]
            summary = dict(graphs=len(rows))
            for name, values in (("nodes", [i.program.num_nodes for i in rows]),
                                 ("edges", [i.program.num_edges for i in rows]),
                                 ("tokens", [5*i.program.num_edges+len(list(nx.isolates(i.graph)))+1 for i in rows])):
                summary[name] = dict(minimum=min(values), maximum=max(values), mean=sum(values)/len(values))
            summary["graph_classes"] = {task: [sum(i.properties[task] == c for i in rows) for c in range(k)]
                                        for task, k in (("components", 3), ("graph_triangle", 2), ("bipartite", 2))}
            summary["edge_classes"] = {task: [sum(getattr(i, task).count(c) for i in rows) for c in (0, 1)]
                                       for task in ("triangle", "bridge")}
            manifest["distributions"][split][family] = summary
    return records, manifest


def sample_node_pairs(item, limit, rng):
    """Bounded-source BFS with class reservoirs, not enumeration of all pairs.

    At most min(N, ceil(sqrt(limit))) BFS sources; each has bounded targets.
    Rare classes may be missed: report candidate and selected counts explicitly.
    """
    n = item.program.num_nodes
    if n < 2 or limit < 1:
        return [], [], [0]*4
    # The source budget grows with the requested sample budget, not N. In
    # particular the default 128-pair budget no longer traverses every source
    # of the existing 32--96-node corpus and materializes all pair candidates.
    sources = rng.sample(range(n), min(n, math.ceil(math.sqrt(limit))))
    pools = [set() for _ in range(4)]
    for u in sources:
        distances = nx.single_source_shortest_path_length(item.graph, u)
        groups = [[], [], [], []]
        for v in range(n):
            if u == v:
                continue
            d = distances.get(v)
            label = 3 if d is None else 0 if d == 1 else 1 if d == 2 else 2
            groups[label].append(v)
        for label, targets in enumerate(groups):
            for v in rng.sample(targets, min(len(targets), limit)):
                pools[label].add(tuple(sorted((u, v))))
    pools = [rng.sample(sorted(pool), min(len(pool), limit)) for pool in pools]
    counts = list(map(len, pools))
    selected = []
    for row in range(limit):
        for label, pool in enumerate(pools):
            if row < len(pool) and len(selected) < limit:
                selected.append((pool[row], label))
    rng.shuffle(selected)
    return [p for p, _ in selected], [y for _, y in selected], counts


def second_view(program, rng):
    """Return view-row -> original-edge-row; node rows never change."""
    order = list(range(program.num_edges))
    rng.shuffle(order)
    rows = []
    for row, original in enumerate(order):
        endpoints = program.instructions[original].endpoints
        rows.append(EdgeInstruction(row, endpoints[::-1] if rng.random() < .5 else endpoints))
    return GraphIRProgram(program.num_nodes, program.node_symbols, tuple(rows)), order
