"""Persistent token/node/edge states with five native graph attention stages."""

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .embedding import GraphEmbedding
from .tokenizer import GraphTokenizer


@dataclass
class EncoderOutput:
    node_embeddings: Tensor
    edge_embeddings: Tensor
    graph_embedding: Tensor


class SDPAttention(nn.Module):
    """Dense bidirectional attention; SDPA chooses its supported device backend."""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads, self.width, self.dropout = heads, d // heads, dropout
        self.q, self.k, self.v, self.out = (nn.Linear(d, d) for _ in range(4))

    def forward(self, query, context):
        b, qlen, d = query.shape
        def project(layer, value):
            return layer(value).reshape(b, -1, self.heads, self.width).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            project(self.q, query), project(self.k, context), project(self.v, context),
            dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        return self.out(attended.transpose(1, 2).reshape(b, qlen, d))


class AttentionUpdate(nn.Module):
    def __init__(self, d, heads, dropout, expansion):
        super().__init__()
        self.attention = SDPAttention(d, heads, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, expansion*d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(expansion*d, d))
        self.drop = nn.Dropout(dropout)

    def finish(self, query, delta):
        query = self.norm1(query + self.drop(delta))
        return self.norm2(query + self.drop(self.ffn(query)))

    def forward(self, query, context):
        return self.finish(query, self.attention(query, context))


class GraphIRBlock(nn.Module):
    def __init__(self, d, heads, dropout, expansion):
        super().__init__()
        self.instruction_attention = AttentionUpdate(d, heads, dropout, expansion)
        self.symbol_alignment = AttentionUpdate(d, heads, dropout, expansion)
        self.endpoint_attention = AttentionUpdate(d, heads, dropout, expansion)
        self.full_edge_attention = AttentionUpdate(d, heads, dropout, expansion)
        self.node_feedback = AttentionUpdate(d, heads, dropout, expansion)
        self.symbol_projection = nn.Linear(d, d)
        self.injection_norm = nn.LayerNorm(d)
        self.endpoint_projection = nn.Linear(d, d)
        self.endpoint_norm = nn.LayerNorm(d)
        self.feedback_symbol = nn.Linear(d, d)
        self.feedback_edge = nn.Linear(d, d)
        self.feedback_gate = nn.Linear(3*d, d)
        self.feedback_norm = nn.LayerNorm(d)

    @staticmethod
    def bound_update(module, symbols, source, groups):
        """Batch equal-degree queries, keeping each node's context separate.

        Empty-context nodes receive only the same residual/norm/FFN update.
        index_copy is out-of-place and preserves gradients through all updates.
        """
        delta = torch.zeros_like(symbols)
        for rows, positions in groups:
            query = symbols[rows].unsqueeze(1)
            context = source[positions]
            update = module.attention(query, context).squeeze(1)
            delta = delta.index_copy(0, rows, update)
        return module.finish(symbols, delta)

    def edge_instruction_self_attention(self, tokens, symbols, refs, nodes, edges):
        injected = self.injection_norm(tokens[refs] + self.symbol_projection(symbols[nodes]))
        tokens = tokens.index_copy(0, refs, injected)
        if len(edges):
            local = tokens.reshape(-1, 5, tokens.shape[-1])
            tokens = self.instruction_attention(local, local).reshape_as(tokens)
        return tokens

    def node_symbol_alignment(self, tokens, symbols, occurrence_groups):
        return self.bound_update(self.symbol_alignment, symbols, tokens, occurrence_groups)

    def endpoint_aware_edge_cross_attention(self, tokens, symbols, edges, bindings, ref_positions, previous):
        if not len(edges):
            return previous
        context = self.endpoint_norm(tokens[ref_positions] + self.endpoint_projection(symbols[bindings]))
        # Previous global edge states entered these EDGE tokens via feedback,
        # and have now passed through the current instruction Transformer.
        query = tokens[edges]
        return self.endpoint_attention(query.unsqueeze(1), context).squeeze(1)

    def full_edge_self_attention(self, edges):
        if not len(edges):
            return edges
        return self.full_edge_attention(edges.unsqueeze(0), edges.unsqueeze(0)).squeeze(0)

    def context_feedback(self, tokens, symbols, edge_states, incident_groups, refs, nodes, owners, edges):
        symbols = self.bound_update(self.node_feedback, symbols, edge_states, incident_groups)
        local = tokens[refs]
        node_context = self.feedback_symbol(symbols[nodes])
        edge_context = self.feedback_edge(edge_states[owners])
        gate = torch.sigmoid(self.feedback_gate(torch.cat((local, node_context, edge_context), -1)))
        updated = self.feedback_norm(local + gate * (node_context + edge_context))
        tokens = tokens.index_copy(0, refs, updated)
        # EDGE tokens carry global context into the next instruction stage.
        tokens = tokens.index_copy(0, edges, edge_states)
        return tokens, symbols

    def forward(self, tokens, symbols, edge_states, links):
        refs, nodes, owners, edges, bindings, ref_positions, occurrences, incidents = links
        tokens = self.edge_instruction_self_attention(tokens, symbols, refs, nodes, edges)
        symbols = self.node_symbol_alignment(tokens, symbols, occurrences)
        edge_states = self.endpoint_aware_edge_cross_attention(
            tokens, symbols, edges, bindings, ref_positions, edge_states)
        edge_states = self.full_edge_self_attention(edge_states)
        tokens, symbols = self.context_feedback(
            tokens, symbols, edge_states, incidents, refs, nodes, owners, edges)
        return tokens, symbols, edge_states


class AttentionReadout(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, values):
        if not len(values):
            return values.new_zeros(values.shape[-1])
        return (self.score(values).softmax(dim=0) * values).sum(0)


class GraphIREncoder(nn.Module):
    """No position or ID embeddings; endpoint permutations are attention permutations.

    In evaluation, endpoint swapping leaves edges/readout unchanged (up to floating
    point reduction order). Training dropout preserves this symmetry in distribution.
    max_edges is a diagnostic limit, never a truncation or alternate attention path.
    """
    def __init__(self, hidden_dim=256, num_blocks=4, num_heads=4, dropout=0.1,
                 ffn_expansion=4, max_edges=20000, feature_dim=None):
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_blocks < 1 or ffn_expansion < 1 or max_edges < 0 or not 0 <= dropout < 1:
            raise ValueError("invalid block count, expansion, edge limit, or dropout")
        self.config = dict(hidden_dim=hidden_dim, num_blocks=num_blocks, num_heads=num_heads,
                           dropout=dropout, ffn_expansion=ffn_expansion, max_edges=max_edges,
                           feature_dim=feature_dim)
        self.hidden_dim = hidden_dim
        self.tokenizer = GraphTokenizer()
        self.embedding = GraphEmbedding(hidden_dim, feature_dim)
        self.blocks = nn.ModuleList([GraphIRBlock(hidden_dim, num_heads, dropout, ffn_expansion)
                                    for _ in range(num_blocks)])
        self.node_readout, self.edge_readout = AttentionReadout(hidden_dim), AttentionReadout(hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def add_feature_adapter(self, feature_dim):
        self.embedding.add_feature_adapter(feature_dim)
        self.config["feature_dim"] = feature_dim

    @staticmethod
    def _links(program, device):
        def index(values):
            return torch.tensor(values, dtype=torch.long, device=device)
        bindings = index([b.nodes for b in program.edge_endpoints]).reshape(-1, 2)
        positions = index([b.token_positions for b in program.edge_endpoints]).reshape(-1, 2)
        edge_positions = index([b.edge_token_position for b in program.edge_endpoints])
        refs, nodes = positions.flatten(), bindings.flatten()
        owners = torch.arange(len(bindings), device=device).repeat_interleave(2)
        incident = [[] for _ in program.node_symbols]
        for e, binding in enumerate(program.edge_endpoints):
            for node in binding.nodes:
                incident[node].append(e)
        def group(rows):
            groups = {}
            for node, values in enumerate(rows):
                if values:
                    groups.setdefault(len(values), []).append((node, values))
            return tuple((index([r for r, _ in g]), index([v for _, v in g])) for g in groups.values())
        return refs, nodes, owners, edge_positions, bindings, positions, group(program.symbol_occurrences), group(incident)

    def forward(self, program, x=None):
        if program.num_edges > self.config["max_edges"]:
            raise ValueError(f"Graph has {program.num_edges} edges, exceeds max_edges={self.config['max_edges']}; "
                             "full O(E^2) attention is required. Explicitly increase --max_edges on suitable hardware.")
        tokenized = self.tokenizer(program)
        tokens, symbols = self.embedding(tokenized, x)
        links = self._links(tokenized, tokens.device)
        edges = tokens[links[3]]
        for block in self.blocks:
            tokens, symbols, edges = block(tokens, symbols, edges, links)
        graph = self.fusion(torch.cat((self.node_readout(symbols), self.edge_readout(edges))))
        return EncoderOutput(symbols, edges, graph)
