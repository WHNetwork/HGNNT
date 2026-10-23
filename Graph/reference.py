"""Deterministic reference parsing of tokenized, visible GraphIR only."""

from dataclasses import dataclass
from enum import IntEnum
import heapq
import math

import torch
from torch import Tensor
from .tokenizer import TokenType, TokenizedProgram


class RelationType(IntEnum):
    SELF = 0
    INSTRUCTION = 1
    REF_TO_NODE = 2
    NODE_TO_REF = 3
    GRAPH_READ = 4


@dataclass(frozen=True)
class ReferenceGraph:
    # All positions share one index space: IR tokens, node slots, GRAPH.
    token_types: Tensor
    node_positions: Tensor  # Rows follow TokenizedProgram.node_symbols, including isolates.
    edge_positions: Tensor  # Rows follow edge_symbols; each points at its EDGE token.
    ref_positions: Tensor
    ref_nodes: Tensor
    graph_position: int
    # Row 0 is the query; row 1 is the key/value it may read.
    relation_index: Tensor
    relation_types: Tensor
    node_features: Tensor


def _structure_features(adjacency):
    """Log degree, local clustering and log core, with no hidden edges.

    Degree-oriented triangle enumeration avoids enumerating all neighbor pairs
    around a high-degree node. Heap peeling computes core numbers.
    """
    n = len(adjacency)
    degree = [len(neighbors) for neighbors in adjacency]
    forward = [set() for _ in adjacency]
    for u, neighbors in enumerate(adjacency):
        for v in neighbors:
            if (degree[u], u) < (degree[v], v):
                forward[u].add(v)
    triangles = [0] * n
    for u, neighbors in enumerate(forward):
        for v in neighbors:
            for w in neighbors & forward[v]:
                triangles[u] += 1
                triangles[v] += 1
                triangles[w] += 1
    remaining = degree.copy()
    heap = [(d, u) for u, d in enumerate(degree)]
    heapq.heapify(heap)
    core = [0] * n
    removed = [False] * n
    level = 0
    while heap:
        d, u = heapq.heappop(heap)
        if removed[u] or d != remaining[u]:
            continue
        removed[u] = True
        level = max(level, d)
        core[u] = level
        for v in adjacency[u]:
            if not removed[v]:
                remaining[v] -= 1
                heapq.heappush(heap, (remaining[v], v))
    return [(math.log1p(d),
             2 * triangles[u] / (d * (d - 1)) if d > 1 else 0.,
             math.log1p(core[u])) for u, d in enumerate(degree)]


def parse_reference_graph(program: TokenizedProgram, device=None, dtype=torch.float32,
                          max_tokens=None, max_relations=None) -> ReferenceGraph:
    """Resolve symbols as indices, never as numeric model features.

    Same-node references communicate through their shared slot, rather than a
    quadratic clique of occurrences. GRAPH is a reader only, never a key for
    other positions. Relation types describe topology, not separate networks.
    """
    token_count, n = len(program.tokens), len(program.node_symbols)
    graph_position = token_count + n
    length = graph_position + 1
    relation_count = (length + sum((end-start) * (end-start-1)
                                  for start, end in program.instruction_spans)
                      + 2 * sum(len(rows) for rows in program.symbol_occurrences)
                      + n + len(program.edge_symbols))
    for name, count, limit in (("tokens", length, max_tokens),
                               ("relations", relation_count, max_relations)):
        if limit is not None and count > limit:
            raise ValueError(f"Visible GraphIR requires {count} {name}, exceeds max_{name}={limit}; "
                             "increase the resource limit; structure is never truncated")
    node_rows = {symbol: row for row, symbol in enumerate(program.node_symbols)}
    adjacency = [set() for _ in range(n)]
    refs, ref_nodes, edge_positions = [], [], []
    queries, keys, types = list(range(length)), list(range(length)), [int(RelationType.SELF)] * length

    def connect(query, key, relation):
        queries.append(query)
        keys.append(key)
        types.append(int(relation))

    for start, end in program.instruction_spans:
        endpoints = []
        for position in range(start, end):
            token = program.tokens[position]
            if token.token_type == TokenType.EDGE:
                # The EDGE position IS the edge entity; no extra edge slot.
                edge_positions.append(position)
            elif token.token_type == TokenType.NODE_REF:
                row = node_rows[token.node_symbol]
                endpoints.append(row)
                refs.append(position)
                ref_nodes.append(row)
                connect(position, token_count + row, RelationType.REF_TO_NODE)
                connect(token_count + row, position, RelationType.NODE_TO_REF)
            for other in range(start, end):
                if position != other:
                    connect(position, other, RelationType.INSTRUCTION)
        u, v = endpoints
        adjacency[u].add(v)
        adjacency[v].add(u)
    node_positions = list(range(token_count, graph_position))
    for position in (*node_positions, *edge_positions):
        connect(graph_position, position, RelationType.GRAPH_READ)

    def index(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    token_types = [int(token.token_type) for token in program.tokens]
    token_types.extend([int(TokenType.NODE_SLOT)] * n + [int(TokenType.GRAPH)])
    return ReferenceGraph(index(token_types), index(node_positions), index(edge_positions),
                          index(refs), index(ref_nodes), graph_position,
                          index([queries, keys]), index(types),
                          torch.tensor(_structure_features(adjacency), device=device,
                                       dtype=dtype).reshape(n, 3))
