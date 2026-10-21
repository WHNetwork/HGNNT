"""Shared structural Call Site encoder, REF fusion and incidence mean readout.

IDs locate symbols only. H-IR sorting and REF selection can still depend on
node numbering; removing ID embeddings does not guarantee equivariance.
"""

from bisect import bisect_left

import torch
from torch import nn

from hir import ADD, DROP, EXTEND, NUM_TYPES, ORDER, PATCH, REF, compile_hypergraph


# Fixed across datasets: 1, 2, 3-4, 5-8, ..., 65-128, >128 members.
ORDER_BUCKET_BOUNDS = (1, 2, 4, 8, 16, 32, 64, 128)


class HIRCall(nn.Module):
    def __init__(self, num_targets, args, data):
        super().__init__()
        if args.task_type != 'node_cls':
            raise ValueError('HIRCall supports node_cls only')
        embedding_dim = int(getattr(args, 'hircall_embedding_dim', 128))
        hidden_size = int(getattr(args, 'hircall_hidden_size', 128))
        layers = int(getattr(args, 'hircall_num_layers', 2))
        dropout = float(getattr(args, 'hircall_dropout', 0.3))
        self.call_batch_size = int(getattr(args, 'hircall_batch_size', 256))
        if min(embedding_dim, hidden_size, layers, self.call_batch_size) < 1:
            raise ValueError('HIRCall dimensions, layers and batch size must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('HIRCall dropout must be in [0, 1)')
        self.output_dim = 2 * hidden_size
        # This table has only compiler operator/type symbols, never graph IDs.
        self.token_embedding = nn.Embedding(NUM_TYPES, embedding_dim)
        descriptor_dim = 1 + len(ORDER_BUCKET_BOUNDS) + 1
        self.node_projection = nn.Sequential(nn.Linear(descriptor_dim, embedding_dim), nn.ReLU())
        self.order_projection = nn.Linear(1, embedding_dim)
        self.ref_projection = nn.Linear(1, embedding_dim, bias=False)
        self.empty_set = nn.Parameter(torch.zeros(embedding_dim))
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.ref_fusion = nn.Sequential(nn.Linear(2 * self.output_dim, self.output_dim), nn.ReLU())
        self.classifier = nn.Linear(self.output_dim, num_targets)
        # Only topology-derived tensors are cached, never learned activations.
        for name in ('_incidence', '_adds', '_drops', '_modes'):
            self.register_buffer(name, torch.empty(0, dtype=torch.long), persistent=False)
        for name in ('_descriptors', '_orders', '_ref_flags', '_degree'):
            self.register_buffer(name, torch.empty(0), persistent=False)
        self._structure_key = None
        self.reset_parameters()
        self._prepare_structure(data)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.GRU)):
                module.reset_parameters()
        with torch.no_grad():
            self.empty_set.zero_()

    def _prepare_structure(self, data):
        index = data.hyperedge_index
        num_nodes = int(data.num_nodes)
        declared = getattr(data, 'hircall_num_hyperedges', getattr(data, 'num_hyperedges', None))
        num_edges = int(declared) if declared is not None else None
        # Counts with arbitrary source IDs are not a declared 0..M-1 domain.
        if num_edges is not None and index.numel() and int(index[1].max()) >= num_edges:
            num_edges = None
        key = (id(index), index.data_ptr(), index._version, num_nodes, num_edges)
        if key == self._structure_key:
            return
        program = compile_hypergraph(num_nodes, index.detach().cpu(), num_edges)
        incidence = program.to_hyperedge_index()
        degree = torch.bincount(incidence[0], minlength=num_nodes).float()
        bucket_count = len(ORDER_BUCKET_BOUNDS) + 1
        edge_buckets = torch.tensor([bisect_left(ORDER_BUCKET_BOUNDS, len(edge))
                                     for edge in program.hyperedges], dtype=torch.long)
        histogram = torch.zeros(num_nodes * bucket_count)
        bins = incidence[0] * bucket_count + edge_buckets[incidence[1]]
        histogram.index_add_(0, bins, torch.ones(bins.numel()))
        histogram = histogram.reshape(num_nodes, bucket_count) / degree.clamp_min(1).unsqueeze(-1)
        descriptors = torch.cat((degree.log1p().unsqueeze(-1), histogram), dim=-1)

        def set_index(field):
            pairs = [(node, ir.edge_id) for ir in program.instructions for node in getattr(ir, field)]
            return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()

        device = self.token_embedding.weight.device
        dtype = self.token_embedding.weight.dtype
        self._incidence = incidence.to(device)
        self._adds = set_index('add').to(device)
        self._drops = set_index('drop').to(device)
        self._modes = torch.tensor([ir.mode for ir in program.instructions], dtype=torch.long, device=device)
        self._descriptors = descriptors.to(device=device, dtype=dtype)
        self._degree = degree.to(device=device, dtype=dtype)
        self._orders = torch.tensor([ir.order for ir in program.instructions],
                                    dtype=dtype, device=device).log1p().unsqueeze(-1)
        self._ref_flags = torch.tensor([ir.ref >= 0 for ir in program.instructions],
                                       dtype=dtype, device=device).unsqueeze(-1)
        self.program = program
        # Canonical row -> source ID after removal of the loader's N offset.
        self.source_edge_ids = program.source_edge_ids
        self._structure_key = key
        self._structure_index = index

    def _set_context(self, descriptions, membership):
        num_edges = len(self.program.hyperedges)
        sums = descriptions.new_zeros((num_edges, descriptions.shape[-1])).index_add(
            0, membership[1], descriptions[membership[0]])
        count = torch.bincount(membership[1], minlength=num_edges).to(descriptions.dtype)
        means = sums / count.clamp_min(1).unsqueeze(-1)
        return torch.where(count.unsqueeze(-1) > 0, means, self.empty_set)

    def encode(self, data):
        """Return original-node-order Z_V and canonical-edge-order Z_E.

        Every call uses five structural tokens, independent of cardinality.
        source_edge_ids maps canonical rows to source edge IDs, not vice versa.
        """
        self._prepare_structure(data)
        descriptions = self.node_projection(self._descriptors)
        add_context = self._set_context(descriptions, self._adds)
        drop_context = self._set_context(descriptions, self._drops)
        num_edges = len(self.program.hyperedges)
        calls = []
        for offset in range(0, num_edges, self.call_batch_size):
            selection = slice(offset, offset + self.call_batch_size)
            # Fixed semantic order: OP, ORDER, REF_CONTEXT, DROP_CONTEXT, ADD_CONTEXT.
            sequence = torch.stack((
                self.token_embedding(self._modes[selection]),
                self.token_embedding.weight[ORDER] + self.order_projection(self._orders[selection]),
                self.token_embedding.weight[REF] + self.ref_projection(self._ref_flags[selection]),
                self.token_embedding.weight[DROP] + drop_context[selection],
                self.token_embedding.weight[ADD] + add_context[selection],
            ), dim=1)
            hidden, _ = self.gru(sequence)
            calls.append(hidden.mean(dim=1))
        if calls:
            initial = torch.cat(calls, dim=0)
            fused = []
            for ir, current in zip(self.program.instructions, initial.unbind(0)):
                if ir.mode in (EXTEND, PATCH):
                    # Compiler REF targets are earlier canonical rows. Python
                    # lists keep shared references differentiable without any
                    # in-place writes to tensors used by previous operations.
                    current = self.ref_fusion(torch.cat((current, fused[ir.ref]), dim=-1))
                fused.append(current)
            edges = torch.stack(fused)
        else:
            edges = descriptions.new_zeros((0, self.output_dim))
        nodes = edges.new_zeros((self.program.num_nodes, self.output_dim)).index_add(
            0, self._incidence[0], edges[self._incidence[1]])
        # Real incidence includes inherited REF members and excludes PATCH drops.
        # Isolated nodes have a shared zero representation; empty edges have
        # valid call embeddings but no incidence contributions.
        nodes = nodes / self._degree.clamp_min(1).unsqueeze(-1)
        return nodes, edges

    def forward(self, data):
        nodes, _ = self.encode(data)
        return self.classifier(nodes), None
