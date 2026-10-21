"""Structure-only H-IR token embeddings and bidirectional GRU readout."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from hir import END_EDGE, PAD, chunk_hir, compile_hypergraph


class HIRSeqGRU(nn.Module):
    def __init__(self, num_targets, args, data):
        super().__init__()
        if args.task_type != 'node_cls':
            raise ValueError('HIRSeqGRU supports node_cls only')
        embedding_dim = int(getattr(args, 'hirseq_embedding_dim', 128))
        hidden_size = int(getattr(args, 'hirseq_hidden_size', 128))
        layers = int(getattr(args, 'hirseq_num_layers', 2))
        dropout = float(getattr(args, 'hirseq_dropout', 0.3))
        self.max_chunk_length = int(getattr(args, 'hirseq_max_chunk_length', 512))
        self.chunk_batch_size = int(getattr(args, 'hirseq_chunk_batch_size', 16))
        if min(embedding_dim, hidden_size, layers, self.chunk_batch_size) < 1:
            raise ValueError('HIRSeqGRU dimensions, layers and chunk batch size must be positive')
        if not 0 <= dropout < 1 or self.max_chunk_length < 6:
            raise ValueError('HIRSeqGRU requires dropout in [0, 1) and chunk length >= 6')
        self.output_dim = 2 * hidden_size
        self._structure_key = None
        self._prepare_structure(data)
        # Reserve ORDER 0..N, including empty and full-cardinality edges.
        vocabulary_size = self.program.space.order(self.program.num_nodes) + 1
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=PAD)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.node_attention = nn.Linear(self.output_dim, 1, bias=False)
        self.isolated_node = nn.Parameter(torch.zeros(self.output_dim))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.output_dim, num_targets)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.gru.reset_parameters()
        self.node_attention.reset_parameters()
        self.classifier.reset_parameters()
        with torch.no_grad():
            self.isolated_node.zero_()

    def _prepare_structure(self, data):
        index = data.hyperedge_index
        num_nodes = int(data.num_nodes)
        # The benchmark's initialization counts only nonempty edges; retain
        # the loader's declared count separately for explicit empty edges.
        declared_edges = getattr(data, 'hirseqgru_num_hyperedges',
                                 getattr(data, 'num_hyperedges', None))
        num_edges = int(declared_edges) if declared_edges is not None else None
        # Standalone callers may supply a count with arbitrary source IDs,
        # rather than the compiler's explicit 0..M-1 empty-edge domain.
        if num_edges is not None and index.numel() and int(index[1].max()) >= num_edges:
            num_edges = None
        key = (id(index), index.data_ptr(), index._version, num_nodes, num_edges)
        if key == self._structure_key:
            return
        program = compile_hypergraph(num_nodes, index.detach().cpu(), num_edges)
        if hasattr(self, 'program') and (
            program.num_nodes != self.program.num_nodes
            or len(program.hyperedges) != len(self.program.hyperedges)
        ):
            raise ValueError('Create a new HIRSeqGRU for a different node/edge vocabulary')
        chunks, _ = chunk_hir(program, self.max_chunk_length)
        node_start = program.space.node(0)
        node_stop = program.space.node(num_nodes)
        readouts = []
        for chunk in chunks:
            node_positions, node_ids, edge_positions, edge_ids, weights = [], [], [], [], []
            for start, edge, contribution in chunk.readouts:
                end = chunk.tokens.index(END_EDGE, start)
                edge_positions.append(end)
                edge_ids.append(edge)
                weights.append(contribution)
                # Only formal records contribute. Context copies still enter
                # the GRU, but their NODE tokens are excluded from pooling.
                # DROP operands are real mentions, never new memberships.
                for position in range(start, end):
                    token = chunk.tokens[position]
                    if node_start <= token < node_stop:
                        node_positions.append(position)
                        node_ids.append(token - node_start)
            readouts.append((node_positions, node_ids, edge_positions, edge_ids, weights))
        self.program, self.chunks, self._readouts = program, chunks, tuple(readouts)
        self._structure_key = key
        self._structure_index = index  # Keep tensor identity alive for the cache.

    def _pool_nodes(self, states, node_ids):
        num_nodes = self.program.num_nodes
        if not states:
            return self.isolated_node.expand(num_nodes, -1)
        states = torch.cat(states)
        ids = torch.tensor(node_ids, dtype=torch.long, device=states.device)
        scores = self.node_attention(states).squeeze(-1)
        # One numerically stable scalar softmax per node across ALL chunks.
        maxima = scores.new_full((num_nodes,), -torch.inf)
        maxima.scatter_reduce_(0, ids, scores.detach(), reduce='amax', include_self=True)
        weights = (scores - maxima[ids]).exp()
        totals = scores.new_zeros(num_nodes).index_add(0, ids, weights)
        pooled = states.new_zeros((num_nodes, self.output_dim)).index_add(
            0, ids, states * (weights / totals[ids]).unsqueeze(-1))
        return torch.where(totals.unsqueeze(-1) > 0, pooled, self.isolated_node)

    def encode(self, data):
        """Return (Z_V, Z_E) in original node and canonical hyperedge order.

        ``program.source_edge_ids`` maps each canonical output row back to its
        source edge ID (after removing the loader's bipartite offset). Nodes
        inherited through REF do not receive an extra NODE readout for that
        derived edge unless they occur explicitly in its lowered record.
        No features, labels or reconstructed memberships enter
        the encoder. A model's learned vocabulary belongs to one graph size.
        """
        self._prepare_structure(data)
        device = self.embedding.weight.device
        node_states, node_ids, edge_states, edge_ids, edge_weights = [], [], [], [], []
        for offset in range(0, len(self.chunks), self.chunk_batch_size):
            batch = self.chunks[offset:offset + self.chunk_batch_size]
            sequences = [torch.tensor(chunk.tokens, dtype=torch.long, device=device)
                         for chunk in batch]
            lengths = torch.tensor([len(chunk.tokens) for chunk in batch], dtype=torch.long)
            tokens = pad_sequence(sequences, batch_first=True, padding_value=PAD)
            packed = pack_padded_sequence(self.embedding(tokens), lengths,
                                          batch_first=True, enforce_sorted=False)
            encoded, _ = self.gru(packed)
            hidden, _ = pad_packed_sequence(encoded, batch_first=True)
            for row in range(len(batch)):
                positions, nodes, ends, edges, weights = self._readouts[offset + row]
                if positions:
                    node_states.append(hidden[row, positions])
                    node_ids.extend(nodes)
                if ends:
                    edge_states.append(hidden[row, ends])
                    edge_ids.extend(edges)
                    edge_weights.extend(weights)
        nodes = self._pool_nodes(node_states, node_ids)
        edges = self.embedding.weight.new_zeros((len(self.program.hyperedges), self.output_dim))
        if edge_states:
            states = torch.cat(edge_states)
            ids = torch.tensor(edge_ids, dtype=torch.long, device=device)
            weights = states.new_tensor(edge_weights)
            edges = edges.index_add(0, ids, states * weights.unsqueeze(-1))
            # Empty edges have a shared, deterministic zero representation.
            nonempty = torch.tensor([bool(edge) for edge in self.program.hyperedges],
                                    dtype=torch.bool, device=device)
            edges = torch.where(nonempty.unsqueeze(-1), edges, torch.zeros_like(edges))
        return nodes, edges

    def forward(self, data):
        nodes, _ = self.encode(data)
        return self.classifier(self.dropout(nodes)), None
