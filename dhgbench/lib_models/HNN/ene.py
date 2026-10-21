"""Minimal edge-state ENE encoder and separate node classification adapter."""

from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add


def _mean(messages, index, size):
    sums = scatter_add(messages, index, dim=0, dim_size=size)
    counts = scatter_add(messages.new_ones(index.numel()), index, dim=0, dim_size=size)
    return sums / counts.clamp_min(1).unsqueeze(-1)


class ENELayer(nn.Module):
    """One E -> N -> E update; node messages are temporary, not layer state."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, edges, hyperedge_index, num_nodes):
        node, edge = hyperedge_index
        messages = _mean(edges[edge], node, num_nodes)
        gathered = _mean(messages[node], edge, edges.shape[0])
        return F.relu(self.linear(gathered))


class ENEEncoder(nn.Module):
    """Task-independent encoder returning only [M, hidden_dim] edge states.

    Edge rows retain their input IDs, including empty ID slots. An optional
    num_hyperedges preserves declared trailing empty edges. No incidence or
    self-loop augmentation is performed here.
    """

    def __init__(self, num_features, hidden_dim=128, num_layers=1, dropout=0.5):
        super().__init__()
        if num_layers not in (1, 2):
            raise ValueError('ENE v0 supports 1 or 2 ENE layers')
        if hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError('ENE requires positive hidden_dim and dropout in [0, 1)')
        self.output_dim = hidden_dim
        self.dropout = dropout
        self.input_projection = nn.Linear(num_features, hidden_dim, bias=False)
        self.layers = nn.ModuleList(ENELayer(hidden_dim) for _ in range(num_layers))

    def reset_parameters(self):
        self.input_projection.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, hyperedge_index, num_hyperedges=None):
        node, edge = hyperedge_index
        num_edges = int(num_hyperedges) if num_hyperedges is not None else 0
        if edge.numel():
            num_edges = max(num_edges, int(edge.max()) + 1)
        # Z0 = D_e^-1 H^T X W_in. Zero-degree edges remain zero.
        edges = _mean(self.input_projection(x)[node], edge, num_edges)
        edges = F.dropout(edges, p=self.dropout, training=self.training)
        for position, layer in enumerate(self.layers):
            edges = layer(edges, hyperedge_index, x.shape[0])
            if position + 1 < len(self.layers):
                edges = F.dropout(edges, p=self.dropout, training=self.training)
        return edges


class ENENodeClassifier(nn.Module):
    """Node-task adapter: incident-edge mean followed by a linear classifier."""

    def __init__(self, num_features, num_targets, args):
        super().__init__()
        hidden_dim = int(getattr(args, 'MLP_hidden', 128))
        self.encoder = ENEEncoder(num_features, hidden_dim,
                                  int(getattr(args, 'All_num_layers', 1)),
                                  float(getattr(args, 'dropout', 0.5)))
        self.classifier = nn.Linear(hidden_dim, num_targets)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        index = data.hyperedge_index
        edges = self.encoder(data.x, index, getattr(data, 'num_hyperedges', None))
        nodes = _mean(edges[index[1]], index[0], data.x.shape[0])
        return self.classifier(nodes), edges
