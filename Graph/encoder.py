"""A single sparse graph-guided Transformer over IR tokens and node slots."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .embedding import GraphEmbedding
from .reference import parse_reference_graph
from .tokenizer import GraphTokenizer

ARCHITECTURE_VERSION = "graph_guided_sparse_v2"


@dataclass
class EncoderOutput:
    node_embeddings: Tensor
    edge_embeddings: Tensor
    graph_embedding: Tensor


class SparseGraphAttention(nn.Module):
    """Standard Q/K/V attention evaluated only on (query, key) relations."""
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.heads = num_heads
        self.width = hidden_dim // num_heads
        self.dropout = dropout
        self.q, self.k, self.v, self.out = (nn.Linear(hidden_dim, hidden_dim) for _ in range(4))

    def forward(self, states, relation_index):
        length, dim = states.shape
        query, key = relation_index

        def project(layer):
            return layer(states).reshape(length, self.heads, self.width)

        q, k, v = project(self.q), project(self.k), project(self.v)
        scores = (q[query] * k[key]).sum(-1) / math.sqrt(self.width)
        # Accumulate softmax in fp32 for reduced precision inputs. No L x L tensor.
        scores = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
        groups = query[:, None].expand(-1, self.heads)
        maxima = scores.new_full((length, self.heads), -torch.inf)
        maxima.scatter_reduce_(0, groups, scores.detach(), reduce="amax", include_self=True)
        weights = (scores - maxima[query]).exp()
        totals = weights.new_zeros((length, self.heads)).scatter_add_(0, groups, weights)
        weights = F.dropout(weights / totals[query], p=self.dropout, training=self.training)
        messages = weights.to(v.dtype).unsqueeze(-1) * v[key]
        attended = v.new_zeros((length, self.heads, self.width)).index_add_(0, query, messages)
        return self.out(attended.reshape(length, dim))


class GraphTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, expansion):
        super().__init__()
        self.attention = SparseGraphAttention(hidden_dim, num_heads, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, expansion * hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(expansion * hidden_dim, hidden_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, states, relation_index):
        states = self.norm1(states + self.drop(self.attention(states, relation_index)))
        return self.norm2(states + self.drop(self.ffn(states)))


class GraphIREncoder(nn.Module):
    """No node ID or absolute position embeddings; GRAPH only reads objects."""
    def __init__(self, hidden_dim=256, num_blocks=4, num_heads=4, dropout=0.1,
                 ffn_expansion=4, feature_dim=None, max_tokens=None, max_relations=None,
                 architecture_version=ARCHITECTURE_VERSION):
        super().__init__()
        if architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(f"Unsupported encoder architecture {architecture_version!r}; "
                             f"expected {ARCHITECTURE_VERSION!r}. Old five-stage checkpoints cannot be loaded.")
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_blocks < 1 or ffn_expansion < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid block count, expansion, or dropout")
        if any(limit is not None and limit < 1 for limit in (max_tokens, max_relations)):
            raise ValueError("optional token/relation limits must be positive")
        self.config = dict(hidden_dim=hidden_dim, num_blocks=num_blocks, num_heads=num_heads,
                           dropout=dropout, ffn_expansion=ffn_expansion, feature_dim=feature_dim,
                           max_tokens=max_tokens, max_relations=max_relations,
                           architecture_version=architecture_version)
        self.hidden_dim = hidden_dim
        self.tokenizer = GraphTokenizer()
        self.embedding = GraphEmbedding(hidden_dim, feature_dim)
        self.blocks = nn.ModuleList([GraphTransformerBlock(hidden_dim, num_heads, dropout, ffn_expansion)
                                    for _ in range(num_blocks)])

    def add_feature_adapter(self, feature_dim):
        self.embedding.add_feature_adapter(feature_dim)
        self.config["feature_dim"] = feature_dim

    def forward(self, program, x=None):
        tokenized = self.tokenizer(program)
        weight = self.embedding.type_projection.weight
        reference = parse_reference_graph(tokenized, device=weight.device, dtype=weight.dtype,
                                          max_tokens=self.config["max_tokens"],
                                          max_relations=self.config["max_relations"])
        states = self.embedding(reference, x)
        for block in self.blocks:
            states = block(states, reference.relation_index)
        return EncoderOutput(states[reference.node_positions], states[reference.edge_positions],
                             states[reference.graph_position])
