"""Type projection plus visible structure and optional external node features."""

from torch import nn
from torch.nn import functional as F
from .tokenizer import TokenType


class GraphEmbedding(nn.Module):
    def __init__(self, hidden_dim, feature_dim=None):
        super().__init__()
        self.type_projection = nn.Linear(len(TokenType), hidden_dim, bias=False)
        self.structure_projection = nn.Linear(3, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.feature_projection = None
        if feature_dim is not None:
            self.add_feature_adapter(feature_dim)

    def add_feature_adapter(self, feature_dim):
        if feature_dim <= 0 or self.feature_projection is not None:
            raise ValueError("feature projection requires a positive dimension and may be configured once")
        weight = self.type_projection.weight
        self.feature_projection = nn.Linear(feature_dim, weight.shape[0], bias=False).to(
            device=weight.device, dtype=weight.dtype)

    def forward(self, reference, x=None):
        weight = self.type_projection.weight
        types = F.one_hot(reference.token_types, num_classes=len(TokenType)).to(dtype=weight.dtype)
        states = self.type_projection(types)
        structure = self.structure_projection(reference.node_features)
        # All occurrences of one symbol read the same initial structural row.
        states = states.index_add(0, reference.ref_positions, structure[reference.ref_nodes])
        nodes = structure
        if x is not None:
            projection = self.feature_projection
            if (projection is None or x.ndim != 2
                    or x.shape != (len(nodes), projection.in_features)):
                raise ValueError("x requires a configured feature projection and shape [num_nodes, feature_dim]")
            nodes = nodes + projection(x.to(device=weight.device, dtype=weight.dtype))
        states = states.index_add(0, reference.node_positions, nodes)
        return self.norm(states)
