"""Syntax-only tokens and shared degree/optional-feature symbol initialization."""

import torch
from torch import nn
from .tokenizer import TokenType


class GraphEmbedding(nn.Module):
    def __init__(self, hidden_dim, feature_dim=None):
        super().__init__()
        self.syntax = nn.Embedding(len(TokenType), hidden_dim)
        self.shared_node = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.shared_node, std=0.02)
        self.structural_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(),
                                            nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.feature_mlp = None
        self.register_parameter("gamma", None)
        if feature_dim is not None:
            self.add_feature_adapter(feature_dim)

    def add_feature_adapter(self, feature_dim):
        if feature_dim <= 0 or self.feature_mlp is not None:
            raise ValueError("adapter requires positive feature_dim and can only be initialized once")
        d = self.shared_node.numel()
        self.feature_mlp = nn.Sequential(nn.Linear(feature_dim, d), nn.GELU(), nn.Linear(d, d))
        self.feature_mlp.to(device=self.shared_node.device, dtype=self.shared_node.dtype)
        self.gamma = nn.Parameter(self.shared_node.new_tensor(0.1))

    def forward(self, program, x=None):
        device = self.shared_node.device
        ids = torch.tensor([int(t.token_type) for t in program.tokens], device=device, dtype=torch.long)
        tokens = self.syntax(ids)
        # Counts are exclusively from the tokenizer's VISIBLE reference bindings.
        degree = self.shared_node.new_tensor([len(p) for p in program.symbol_occurrences])
        nodes = self.shared_node + self.structural_mlp(degree.log1p().unsqueeze(-1))
        if x is not None:
            if self.feature_mlp is None or x.ndim != 2 or x.shape[0] != len(degree):
                raise ValueError("features require a configured adapter and one row per node")
            nodes = nodes + self.gamma * self.feature_mlp(x.to(device=device, dtype=nodes.dtype))
        return tokens, self.norm(nodes)
