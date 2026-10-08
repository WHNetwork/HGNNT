"""Centrality-guided Edge–Node–Edge node classifier."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add


class CENE(nn.Module):
    MODES = ('ENE', 'ENE-Excl', 'CENE', 'CENE-Shuffle')

    def __init__(self, num_features, num_targets, args):
        super().__init__()
        self.mode = getattr(args, 'cene_mode', 'CENE')
        if self.mode not in self.MODES:
            raise ValueError(f'Unknown CENE mode: {self.mode}')
        hidden = int(getattr(args, 'MLP_hidden', 128))
        decoder_hidden = int(getattr(args, 'decoder_hidden', hidden))
        self.dropout = float(getattr(args, 'dropout', 0.5))
        self.shuffle_seed = int(getattr(args, 'cene_shuffle_seed', 0))
        self.lin_in = nn.Linear(num_features, hidden)
        self.lin_self = nn.Linear(hidden, hidden, bias=False)
        self.lin_msg = nn.Linear(hidden, hidden, bias=False)
        self.lin_res = nn.Linear(hidden, hidden, bias=False)
        self.lin_edge = nn.Linear(hidden, hidden, bias=False)
        self.classifier = nn.Sequential(nn.Linear(hidden, decoder_hidden), nn.ReLU(),
                                        nn.Dropout(self.dropout), nn.Linear(decoder_hidden, num_targets))
        self.lambda_param = nn.Parameter(torch.zeros(()))
        self._structure_key = None
        self._structure = None
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        with torch.no_grad():
            self.lambda_param.zero_()

    def _get_structure(self, data):
        index = data.hyperedge_index
        n = data.x.shape[0]
        key = (id(index), index.data_ptr(), index._version, index.device, n)
        if key == self._structure_key:
            return self._structure
        if index.numel() == 0:
            node = index.new_empty((0,))
            edge = index.new_empty((0,))
            count = index.new_zeros(n)
            num_edges = 0
        else:
            node, raw_edge = index[0], index[1]
            _, edge = torch.unique(raw_edge, sorted=True, return_inverse=True)
            # Treat repeated (node, hyperedge) entries as one incidence.
            incidence = torch.unique(torch.stack((node, edge), dim=1), dim=0)
            node, edge = incidence[:, 0], incidence[:, 1]
            sizes = torch.bincount(edge, minlength=int(edge.max()) + 1)
            keep = sizes[edge] > 1
            node, edge = node[keep], edge[keep]
            if edge.numel():
                _, edge = torch.unique(edge, sorted=True, return_inverse=True)
                num_edges = int(edge.max()) + 1
            else:
                num_edges = 0
            count = torch.bincount(node, minlength=n)
        score = torch.log1p(count.to(dtype=torch.float32))
        std = score.std(unbiased=False)
        score = (score - score.mean()) / std.clamp_min(1e-12) if n else score
        if self.mode == 'CENE-Shuffle':
            generator = torch.Generator(device='cpu').manual_seed(self.shuffle_seed)
            permutation = torch.randperm(n, generator=generator).to(score.device)
            score = score[permutation]
        self._structure_key = key
        self._structure = node, edge, count, score, num_edges
        return self._structure

    def edge_messages(self, z, node, edge, degree, score, num_nodes):
        """Aggregate E→N→E messages for the already filtered incidence list."""
        num_edges = z.shape[0]
        if edge.numel() == 0:
            return z.new_zeros(z.shape)
        sums = scatter_add(z[edge], node, dim=0, dim_size=num_nodes)
        exclude = self.mode != 'ENE'
        valid = degree[node] > (1 if exclude else 0)
        if exclude:
            messages = (sums[node] - z[edge]) / (degree[node] - 1).clamp_min(1).unsqueeze(-1)
        else:
            messages = sums[node] / degree[node].clamp_min(1).unsqueeze(-1)
        if self.mode in ('CENE', 'CENE-Shuffle'):
            weights = torch.exp(torch.clamp(self.lambda_param * score[node].to(z.dtype), -20, 20))
        else:
            weights = z.new_ones(node.shape)
        weights = weights * valid.to(z.dtype)
        numerator = scatter_add(messages * weights.unsqueeze(-1), edge, dim=0, dim_size=num_edges)
        denominator = scatter_add(weights, edge, dim=0, dim_size=num_edges)
        return numerator / denominator.clamp_min(1e-12).unsqueeze(-1)

    def forward(self, data):
        node, edge, degree, score, num_edges = self._get_structure(data)
        h = F.relu(self.lin_in(data.x))
        h = F.dropout(h, self.dropout, training=self.training)
        if num_edges:
            z = scatter_add(h[node], edge, dim=0, dim_size=num_edges)
            edge_size = torch.bincount(edge, minlength=num_edges).clamp_min(1)
            z = z / edge_size.unsqueeze(-1)
            message = self.edge_messages(z, node, edge, degree, score, h.shape[0])
            z = F.relu(self.lin_self(z) + self.lin_msg(message))
            z = F.dropout(z, self.dropout, training=self.training)
            gathered = scatter_add(z[edge], node, dim=0, dim_size=h.shape[0])
            gathered = gathered / degree.clamp_min(1).unsqueeze(-1)
        else:
            gathered = torch.zeros_like(h)
        h = F.relu(self.lin_res(h) + self.lin_edge(gathered))
        return self.classifier(h), None
