"""Fixed five-token instructions and explicit endpoint/symbol bindings."""

from dataclasses import dataclass
from enum import IntEnum
from .ir import GraphIRProgram


class TokenType(IntEnum):
    EDGE = 0
    CONNECT = 1
    NODE_REF = 2
    END_EDGE = 3
    PAD = 4


@dataclass(frozen=True)
class GraphToken:
    token_type: TokenType
    edge_symbol: int
    node_symbol: int | None = None


@dataclass(frozen=True)
class EdgeEndpoints:
    nodes: tuple[int, int]
    token_positions: tuple[int, int]
    edge_token_position: int


@dataclass(frozen=True)
class TokenizedProgram:
    tokens: tuple[GraphToken, ...]
    instruction_spans: tuple[tuple[int, int], ...]
    node_symbols: tuple[int, ...]
    symbol_occurrences: tuple[tuple[int, ...], ...]
    edge_symbols: tuple[int, ...]
    edge_endpoints: tuple[EdgeEndpoints, ...]


class GraphTokenizer:
    def __call__(self, program: GraphIRProgram) -> TokenizedProgram:
        tokens, spans, bindings = [], [], []
        occurrences = [[] for _ in program.node_symbols]
        for edge in program.instructions:
            start = len(tokens)
            u, v = edge.endpoints
            tokens.extend((GraphToken(TokenType.EDGE, edge.edge_symbol),
                           GraphToken(TokenType.CONNECT, edge.edge_symbol),
                           GraphToken(TokenType.NODE_REF, edge.edge_symbol, u),
                           GraphToken(TokenType.NODE_REF, edge.edge_symbol, v),
                           GraphToken(TokenType.END_EDGE, edge.edge_symbol)))
            occurrences[u].append(start + 2)
            occurrences[v].append(start + 3)
            spans.append((start, start + 5))
            bindings.append(EdgeEndpoints((u, v), (start + 2, start + 3), start))
        return TokenizedProgram(tuple(tokens), tuple(spans), program.node_symbols,
                                tuple(map(tuple, occurrences)),
                                tuple(e.edge_symbol for e in program.instructions), tuple(bindings))
