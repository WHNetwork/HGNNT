"""Structure-only HIR-GraphBERT. No feature tensor or dataset ID vocabulary.

The encoder boundary is StructureInput, containing ONLY visible token metadata.
The unchanged H-IR compiler/lowering live in hir.py; input-only GLOBAL and MASK
symbols are not part of its serialized vocabulary.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hir import (EDGE, EDGE_ID, END_EDGE, EXTEND, NODE, NUM_TYPES, ORDER,
                 PAD, PATCH, REF, chunk_hir, compile_hypergraph)


MASK_INSTRUCTION, GLOBAL = NUM_TYPES, NUM_TYPES + 1
RELATION_NAMES = ('same_hyperedge', 'same_node', 'reference_dependency', 'global')
HIRGRAPHBERT_DEFAULTS = {
    'hidden_size': 256, 'num_hidden_layers': 4, 'num_attention_heads': 4,
    'intermediate_size': 1024, 'dropout': 0.2, 'max_seq_length': 512,
    'chunk_batch_size': 16, 'gradient_checkpointing': True,
}


def model_config(args):
    return {key: default if getattr(args, 'hir_' + key, None) is None
            else getattr(args, 'hir_' + key) for key, default in HIRGRAPHBERT_DEFAULTS.items()}


@dataclass(frozen=True)
class GraphSequence:
    types: tuple
    nodes: tuple
    payload_edges: tuple
    owners: tuple
    references: tuple
    orders: tuple
    # position, edge ID, fragment contribution. Context copies have no readout.
    readouts: tuple = ()


@dataclass(frozen=True)
class StructureInput:
    num_nodes: int
    num_edges: int
    sequences: tuple
    node_codes: tuple
    edge_codes: tuple

    @property
    def token_count(self):
        return sum(len(sequence.types) + 1 for sequence in self.sequences)


def decode_chunk(program, chunk, edge_mapping=None):
    """Map every numeric token by TokenSpace ranges, including REF payloads.

    Never recover memberships here: attention uses only tokens in this chunk.
    In particular, the logical program's hyperedges cannot supply hidden links.
    """
    space = program.space
    mapping = edge_mapping or tuple(range(space.num_edges))
    types, nodes, payloads, owners, refs = [], [], [], [], []
    position = 0
    while position < len(chunk.tokens):
        end = chunk.tokens.index(END_EDGE, position)
        record = chunk.tokens[position:end + 1]
        if record[0] != EDGE:
            raise ValueError('Expected a complete H-IR instruction or fragment')
        owner = mapping[record[1] - space.edge(0)]
        reference = -1
        if record[3] in (EXTEND, PATCH):
            if record[4] != REF:
                raise ValueError('Missing visible REF payload')
            reference = mapping[record[5] - space.edge(0)]
        for token in record:
            node, payload = -1, -1
            if space.node(0) <= token < space.edge(0):
                kind, node = NODE, token - space.node(0)
            elif space.edge(0) <= token < space.order(0):
                kind, payload = EDGE_ID, mapping[token - space.edge(0)]
            elif token >= space.order(0):
                kind = ORDER
            else:
                kind = token
            types.append(kind)
            nodes.append(node)
            payloads.append(payload)
            owners.append(owner)
            refs.append(reference)
        position = end + 1
    return GraphSequence(tuple(types), tuple(nodes), tuple(payloads), tuple(owners), tuple(refs),
                         chunk.orders, tuple((pos, mapping[edge], weight) for pos, edge, weight in chunk.readouts))


def structure_input(program, chunks, max_length, node_codes=None, edge_codes=None,
                    edge_mapping=None, num_edges=None, masked_edges=(), masked_anchors=None):
    """Build an encoder input, adding feature-free queries for absent nodes.

    Codes are local identity permutations, shared across all chunks/branches of
    a sample. Callers keep source/global IDs outside this object.
    """
    n = program.num_nodes
    e = len(program.hyperedges) if num_edges is None else num_edges
    sequences = [decode_chunk(program, chunk, edge_mapping) for chunk in chunks]
    for edge in masked_edges:
        # Input-only partial instruction: one visible member anchors the query
        # to the remaining graph. No original syntax/ORDER/REF/length survives.
        anchors = tuple((masked_anchors or {}).get(edge, ()))
        width = 1 + len(anchors)
        empty = (-1,) * width
        sequences.append(GraphSequence((MASK_INSTRUCTION,) + (NODE,) * len(anchors), (-1,) + anchors,
                                       empty, (edge,) * width, empty, empty, ((0, edge, 1.),)))
    present = {node for sequence in sequences for node in sequence.nodes if node >= 0}
    absent = [node for node in range(n) if node not in present]
    for start in range(0, len(absent), max_length - 1):
        ids = tuple(absent[start:start + max_length - 1])
        empty = (-1,) * len(ids)
        sequences.append(GraphSequence((NODE,) * len(ids), ids, empty, empty, empty, empty))
    if any(len(sequence.types) + 1 > max_length for sequence in sequences):
        raise ValueError('H-IR chunk plus GLOBAL exceeds max_seq_length')
    return StructureInput(n, e, tuple(sequences),
                          tuple(range(1, n + 1)) if node_codes is None else tuple(node_codes),
                          tuple(range(1, e + 1)) if edge_codes is None else tuple(edge_codes))


def graph_relations(types, nodes, owners, references, valid):
    """Four visible relation channels and a hard graph-guided attention mask."""
    same_edge = (owners[:, :, None] == owners[:, None, :]) & (owners[:, :, None] >= 0)
    same_node = (nodes[:, :, None] == nodes[:, None, :]) & (nodes[:, :, None] >= 0)
    dependency = (((owners[:, :, None] == references[:, None, :]) & (owners[:, :, None] >= 0)) |
                  ((references[:, :, None] == owners[:, None, :]) & (references[:, :, None] >= 0)))
    global_channel = (types[:, :, None] == GLOBAL) | (types[:, None, :] == GLOBAL)
    channels = torch.stack((same_edge, same_node, dependency, global_channel), dim=1)
    diagonal = torch.eye(types.shape[1], dtype=torch.bool, device=types.device).unsqueeze(0)
    # Padding queries attend GLOBAL as well, so SDPA never sees an all-masked row.
    allowed = (channels.any(dim=1) | diagonal) & valid[:, None, :]
    return channels, allowed


class GraphGuidedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config['num_attention_heads']
        self.width = config['hidden_size'] // self.heads
        self.qkv = nn.Linear(config['hidden_size'], 3 * config['hidden_size'])
        self.output = nn.Linear(config['hidden_size'], config['hidden_size'])
        # Explicitly counted extra structural attention parameters: 4 per head.
        self.relation_bias = nn.Parameter(torch.zeros(self.heads, len(RELATION_NAMES)))
        self.dropout = config['dropout']

    def forward(self, hidden, channels, allowed):
        batch, length, _ = hidden.shape
        query, key, value = self.qkv(hidden).reshape(batch, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        bias = torch.einsum('hr,brij->bhij', self.relation_bias.to(query.dtype), channels.to(query.dtype))
        bias = bias.masked_fill(~allowed[:, None], float('-inf'))
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=bias,
                                                 dropout_p=self.dropout if self.training else 0.)
        return self.output(attended.transpose(1, 2).reshape(batch, length, -1))


class GraphTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config['hidden_size']
        self.attention = GraphGuidedAttention(config)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, config['intermediate_size']), nn.GELU(),
                                 nn.Dropout(config['dropout']), nn.Linear(config['intermediate_size'], hidden))
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, hidden, channels, allowed):
        hidden = hidden + self.dropout(self.attention(self.norm1(hidden), channels, allowed))
        return hidden + self.dropout(self.ffn(self.norm2(hidden)))


class AttentionPooling(nn.Module):
    """Shared, stable segment softmax over NODE occurrences across ALL chunks."""
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, values, ids, count, contributions=None):
        if not len(values):
            return values.new_zeros((count, values.shape[-1]))
        logits = self.score(values).squeeze(-1).float()
        if contributions is not None:
            logits = logits + contributions.float().clamp_min(1e-12).log()
        maximum = logits.new_full((count,), -float('inf'))
        maximum.scatter_reduce_(0, ids, logits.detach(), reduce='amax', include_self=True)
        weights = (logits - maximum[ids]).exp()
        denominator = weights.new_zeros(count).index_add(0, ids, weights)
        weights = (weights / denominator[ids]).to(values.dtype)
        return values.new_zeros((count, values.shape[-1])).index_add(0, ids, values * weights[:, None])


@dataclass
class EncodedStructure:
    nodes: torch.Tensor
    edges: torch.Tensor
    node_occurrences: torch.Tensor


class HIRGraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        hidden = config['hidden_size']
        if (hidden < 2 or hidden % config['num_attention_heads'] or config['max_seq_length'] < 7
                or min(config['chunk_batch_size'], config['num_hidden_layers'], config['intermediate_size']) < 1):
            raise ValueError('Invalid HIRGraphBERT width, heads, layers, or chunk size')
        self.chunk_batch_size = config['chunk_batch_size']
        self.use_checkpointing = config['gradient_checkpointing']
        self.tokens = nn.Embedding(NUM_TYPES + 2, hidden, padding_idx=PAD)
        self.positions = nn.Embedding(config['max_seq_length'], hidden)
        self.node_identity = nn.Linear(hidden, hidden, bias=False)
        self.edge_identity = nn.Linear(hidden, hidden, bias=False)
        self.order_projection = nn.Linear(1, hidden, bias=False)
        self.input_norm, self.output_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(config['dropout'])
        self.layers = nn.ModuleList(GraphTransformerLayer(config) for _ in range(config['num_hidden_layers']))
        self.node_pool, self.edge_pool = AttentionPooling(hidden), AttentionPooling(hidden)
        self.register_buffer('frequencies', torch.exp(-math.log(10000.) * torch.arange(0, hidden, 2) / hidden),
                             persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.tokens.weight[PAD].zero_()
            for layer in self.layers:
                layer.attention.relation_bias.zero_()

    def _identity(self, codes):
        phase = codes.clamp_min(0).float().unsqueeze(-1) * self.frequencies
        result = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)[..., :self.config['hidden_size']]
        return result * (codes >= 0).unsqueeze(-1)

    def _run(self, types, nodes, payload_codes, owner_codes, orders, owners, references, valid,
             node_codes, rows, columns):
        hidden = (self.tokens(types) + self.positions(torch.arange(types.shape[1], device=types.device))[None]
                  + self.node_identity(self._identity(node_codes))
                  + self.edge_identity(self._identity(owner_codes) + self._identity(payload_codes))
                  + self.order_projection(orders.clamp_min(0).float().log1p().unsqueeze(-1)))
        hidden = self.dropout(self.input_norm(hidden))
        channels, allowed = graph_relations(types, nodes, owners, references, valid)
        for layer in self.layers:
            hidden = layer(hidden, channels, allowed)
        return self.output_norm(hidden[rows, columns])

    def encode_many(self, inputs, max_batch_tokens=32768):
        """Batch independent sequences; identities/graphs never cross owners."""
        device, hidden = self.tokens.weight.device, self.config['hidden_size']
        requests = [(owner, sequence) for owner, view in enumerate(inputs) for sequence in view.sequences]
        requests.sort(key=lambda item: len(item[1].types))
        groups, group = [], []
        for request in requests:
            length = len(request[1].types) + 1
            if length > max_batch_tokens:
                raise ValueError('A graph sequence exceeds the forward token budget')
            if group and (len(group) >= self.chunk_batch_size or length * (len(group) + 1) > max_batch_tokens):
                groups.append(group)
                group = []
            group.append(request)
        if group:
            groups.append(group)
        node_values, node_ids = [[] for _ in inputs], [[] for _ in inputs]
        edge_values, edge_ids, edge_weights = [[] for _ in inputs], [[] for _ in inputs], [[] for _ in inputs]
        calls = []
        for group in groups:
            width = max(len(sequence.types) + 1 for _, sequence in group)
            shape = (len(group), width)
            types = torch.full(shape, PAD, dtype=torch.long)
            nodes, payload_codes, owner_codes, orders, owners, references, codes = [
                torch.full(shape, -1, dtype=torch.long) for _ in range(7)]
            valid = torch.zeros(shape, dtype=torch.bool)
            locations, outputs, own_tokens = [], [], {}
            for row, (owner, sequence) in enumerate(group):
                view = inputs[owner]
                length = len(sequence.types) + 1
                types[row, 0], valid[row, :length] = GLOBAL, True
                types[row, 1:length] = torch.tensor(sequence.types)
                for destination, source in ((nodes, sequence.nodes), (owners, sequence.owners),
                                            (references, sequence.references), (orders, sequence.orders)):
                    destination[row, 1:length] = torch.tensor(source)
                codes[row, 1:length] = torch.tensor([view.node_codes[node] if node >= 0 else -1 for node in sequence.nodes])
                owner_codes[row, 1:length] = torch.tensor([view.edge_codes[edge] if edge >= 0 else -1 for edge in sequence.owners])
                payload_codes[row, 1:length] = torch.tensor([view.edge_codes[edge] if edge >= 0 else -1 for edge in sequence.payload_edges])
                for position, node in enumerate(sequence.nodes):
                    if node >= 0:
                        locations.append((row, position + 1))
                        outputs.append((owner, 'node', node, 1.))
                for position, edge, weight in sequence.readouts:
                    locations.append((row, position + 1))
                    outputs.append((owner, 'edge', edge, weight))
                own_tokens[owner] = own_tokens.get(owner, 0) + length
            if not locations:
                continue
            rows, columns = torch.tensor(locations, dtype=torch.long, device=device).unbind(1)
            args = [value.to(device) for value in (types, nodes, payload_codes, owner_codes, orders,
                                                  owners, references, valid, codes)] + [rows, columns]
            if self.use_checkpointing and self.training and torch.is_grad_enabled():
                selected = checkpoint(self._run, *args, use_reentrant=False)
            else:
                selected = self._run(*args)
            for owner in own_tokens:
                for kind, values, indices, weights in (
                        ('node', node_values, node_ids, None), ('edge', edge_values, edge_ids, edge_weights)):
                    slots = [i for i, output in enumerate(outputs) if output[0] == owner and output[1] == kind]
                    if slots:
                        values[owner].append(selected[torch.tensor(slots, device=device)])
                        indices[owner].extend(outputs[i][2] for i in slots)
                        if weights is not None:
                            weights[owner].extend(outputs[i][3] for i in slots)
            calls.append({'tokens': sum(own_tokens.values()), 'padded_tokens': len(group) * width,
                          'sequences': len(group), 'max_sequence_length': width, 'owner_tokens': own_tokens})
        result = []
        for owner, view in enumerate(inputs):
            values = torch.cat(node_values[owner]) if node_values[owner] else self.tokens.weight.new_empty((0, hidden))
            ids = torch.tensor(node_ids[owner], dtype=torch.long, device=device)
            nodes_out = self.node_pool(values, ids, view.num_nodes)
            values = torch.cat(edge_values[owner]) if edge_values[owner] else self.tokens.weight.new_empty((0, hidden))
            edges_out = self.edge_pool(values, torch.tensor(edge_ids[owner], dtype=torch.long, device=device), view.num_edges,
                                       torch.tensor(edge_weights[owner], device=device))
            result.append(EncodedStructure(nodes_out, edges_out, torch.bincount(ids, minlength=view.num_nodes)))
        return result, calls

    def forward(self, view):
        return self.encode_many([view])[0][0]

    def parameter_counts(self):
        structural = sum(layer.attention.relation_bias.numel() for layer in self.layers)
        return {'encoder': sum(p.numel() for p in self.parameters()), 'structural_attention': structural,
                'node_identity_projection': sum(p.numel() for p in self.node_identity.parameters()),
                'edge_identity_projection': sum(p.numel() for p in self.edge_identity.parameters()),
                'node_attention_pooling': sum(p.numel() for p in self.node_pool.parameters()),
                'edge_attention_pooling': sum(p.numel() for p in self.edge_pool.parameters())}


def compile_data(data):
    # num_nodes is required. Accessing data.x, even for a fallback, is unnecessary.
    count = getattr(data, 'num_hyperedges', None)
    count = None if count is None else int(count)
    index = data.hyperedge_index.detach().cpu()
    if count is not None and index.numel() and int(index[1].max()) >= count:
        if len(index[1].unique()) != count:
            raise ValueError('Explicit empty edges require dense source edge IDs')
        count = None
    return compile_hypergraph(int(data.num_nodes), index, count)


class HIRGraphBERT(nn.Module):
    """Classifier consumes only NODE states, including isolated NODE queries."""
    def __init__(self, num_classes, args):
        super().__init__()
        self.model_config = model_config(args)
        self.num_classes = num_classes
        self.encoder = HIRGraphEncoder(self.model_config)
        hidden = self.model_config['hidden_size']
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                        nn.Dropout(self.model_config['dropout']), nn.Linear(hidden, num_classes))
        self.program = self.chunks = self.visible_input = None
        self.compilation_stats = {}
        self._source_index = None
        self.identity_seed = getattr(args, 'seed', getattr(args, 'seed_offset', 0))
        self.forward_token_budget = getattr(args, 'bert_max_batch_tokens', 32768)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def prepare(self, data):
        index, n, count = data.hyperedge_index, int(data.num_nodes), getattr(data, 'num_hyperedges', None)
        count = None if count is None else int(count)
        key = (index._version, n, count, self.identity_seed)
        if self._source_index is index and getattr(self, '_source_key', None) == key:
            return
        self.program = compile_data(data)
        self.chunks, stats = chunk_hir(self.program, self.model_config['max_seq_length'] - 1)
        generator = torch.Generator().manual_seed(self.identity_seed)
        nodes = (torch.randperm(n, generator=generator) + 1).tolist()
        edges = (torch.randperm(len(self.program.hyperedges), generator=generator) + 1).tolist()
        self.visible_input = structure_input(self.program, self.chunks, self.model_config['max_seq_length'], nodes, edges)
        self.compilation_stats = {**self.program.stats, **stats, 'encoder_tokens': self.visible_input.token_count,
                                  'input_mode': 'Structure-only'}
        self._source_index, self._source_key = index, key

    def encode_nodes(self, data):
        self.prepare(data)
        return self.encoder.encode_many([self.visible_input], self.forward_token_budget)[0][0].nodes

    def forward(self, data):
        nodes = self.encode_nodes(data)
        self.last_node_variance = nodes.detach().float().var(0, unbiased=False).mean().item() if len(nodes) else 0.
        return self.classifier(nodes), None

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('kind') != 'hirgraphbert_node_classifier':
            raise ValueError('Expected a HIRGraphBERT node classifier checkpoint')
        args = SimpleNamespace(**{'hir_' + key: value for key, value in saved['model_config'].items()}, seed=saved['seed'])
        model = cls(saved['num_classes'], args).to(device)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        return model.eval(), saved


def load_encoder(model, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('kind') != 'hirgraphbert_pretrained_encoder' or saved.get('input_mode') != 'Structure-only':
        raise ValueError('Expected a Structure-only HIRGraphBERT encoder checkpoint')
    # Runtime packing/checkpointing/dropout are not weight shapes or semantics.
    shape_keys = ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'max_seq_length')
    if any(model.model_config[key] != saved['model_config'][key] for key in shape_keys):
        raise ValueError('Incompatible HIRGraphBERT encoder architecture')
    state = saved['shared_encoder_state_dict']
    result = model.encoder.load_state_dict(state, strict=True)
    return {'loaded_keys': list(state), 'missing_keys': result.missing_keys, 'unexpected_keys': result.unexpected_keys,
            'pretrain_seed': saved['seed'], 'best_epoch': saved['best_epoch'], 'source': str(path)}
