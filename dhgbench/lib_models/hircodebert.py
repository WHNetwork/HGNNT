"""Hierarchical, semantic-aware Transformer encoder for visible H-IR.

HIR-CodeBERT keeps the deterministic H-IR compiler and its bounded FRAGMENT
lowering.  Stage 1 encodes one visible instruction (or fragment) at a time;
Stage 2 encodes canonical instruction identities with typed relations derived
only from the tokens present in the supplied StructureInput.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hir import (ADD, DIRECT, DROP, EDGE, EDGE_ID, END_EDGE, EXTEND, FRAGMENT, NODE,
                 NUM_TYPES, ORDER, PAD, PATCH, REF, chunk_hir)
from lib_models.hirgraphbert import (MASK_INSTRUCTION, AttentionPooling,
                                     EncodedStructure, compile_data, structure_input)


ROLE_NONE, ROLE_SUMMARY, ROLE_OWNER, ROLE_OPCODE, ROLE_REFERENCE = range(5)
ROLE_DIRECT, ROLE_ADD, ROLE_DROP, ROLE_MASK, ROLE_ISOLATED = range(5, 10)
NUM_ROLES = 10
SEMANTIC_RELATION_NAMES = (
    'references', 'referenced_by', 'shared_node', 'same_opcode',
    'shared_direct_operand', 'shared_add_operand', 'shared_drop_operand',
)

HIRCODEBERT_DEFAULTS = {
    'hidden_size': 256, 'num_hidden_layers': 4, 'local_num_hidden_layers': 2,
    'semantic_num_hidden_layers': 2, 'num_attention_heads': 4,
    'intermediate_size': 1024, 'dropout': 0.2, 'max_seq_length': 512,
    'chunk_batch_size': 16, 'gradient_checkpointing': True,
}


def model_config(args):
    return {key: default if getattr(args, 'hir_' + key, None) is None
            else getattr(args, 'hir_' + key) for key, default in HIRCODEBERT_DEFAULTS.items()}


@dataclass(frozen=True)
class VisibleInstruction:
    """One instruction occurrence reconstructed from visible token metadata."""
    types: tuple
    nodes: tuple
    payload_edges: tuple
    owners: tuple
    references: tuple
    orders: tuple
    roles: tuple
    owner: int
    reference: int
    opcode: int
    contribution: float


def _roles(types):
    """Classify operands without recovering any hidden hyperedge membership."""
    result = [ROLE_NONE] * len(types)
    operand_role = ROLE_NONE
    for position, kind in enumerate(types):
        if kind == EDGE:
            result[position] = ROLE_NONE
        elif kind == ORDER:
            result[position] = ROLE_NONE
        elif kind in (DIRECT, EXTEND, PATCH, FRAGMENT):
            result[position] = ROLE_OPCODE
            operand_role = ROLE_DIRECT if kind in (DIRECT, FRAGMENT) else ROLE_NONE
        elif kind == REF:
            result[position] = ROLE_REFERENCE
        elif kind == DROP:
            result[position] = ROLE_NONE
            operand_role = ROLE_DROP
        elif kind == ADD:
            result[position] = ROLE_NONE
            operand_role = ROLE_ADD
        elif kind == NODE:
            result[position] = operand_role if operand_role != ROLE_NONE else ROLE_ISOLATED
        elif kind == MASK_INSTRUCTION:
            result[position] = ROLE_MASK
        elif kind == END_EDGE:
            operand_role = ROLE_NONE
    # The first edge payload names the defining instruction; later payloads are REF operands.
    seen_edge = False
    for position, kind in enumerate(types):
        if kind == EDGE_ID:
            result[position] = ROLE_OWNER if not seen_edge else ROLE_REFERENCE
            seen_edge = True
    return tuple(result)


def visible_instructions(view):
    """Split StructureInput sequences into deterministic, visible instructions.

    Context copies and FRAGMENT records remain separate occurrences.  Synthetic
    masked-edge queries and isolated NODE queries are represented as local units
    too, so every visible NODE token receives a Stage-1 state.
    """
    instructions = []
    for sequence in view.sequences:
        readouts = {position: (edge, weight) for position, edge, weight in sequence.readouts}
        position = 0
        while position < len(sequence.types):
            if sequence.types[position] == EDGE:
                try:
                    end = sequence.types.index(END_EDGE, position) + 1
                except ValueError as error:
                    raise ValueError('Visible H-IR instruction is missing END_EDGE') from error
            else:
                # MASK queries and absent-node queries are deliberately not H-IR syntax.
                end = len(sequence.types)
            fields = tuple(value[position:end] for value in (
                sequence.types, sequence.nodes, sequence.payload_edges, sequence.owners,
                sequence.references, sequence.orders))
            owner = next((edge for edge in fields[3] if edge >= 0), -1)
            reference = next((edge for edge in fields[4] if edge >= 0), -1)
            opcode = next((kind for kind in fields[0] if kind in (DIRECT, EXTEND, PATCH, FRAGMENT)),
                          MASK_INSTRUCTION if MASK_INSTRUCTION in fields[0] else -1)
            output = readouts.get(position)
            contribution = float(output[1]) if output is not None and output[0] == owner else 0.
            instructions.append(VisibleInstruction(
                *fields, _roles(fields[0]), owner, reference, opcode, contribution))
            position = end
    return tuple(instructions)


def semantic_relations(view, instructions):
    """Build typed instruction relations from this view's visible operands only."""
    count = view.num_edges
    references = [set() for _ in range(count)]
    opcodes = [set() for _ in range(count)]
    operands = [[set() for _ in range(count)] for _ in range(3)]
    all_nodes = [set() for _ in range(count)]
    role_slot = {ROLE_DIRECT: 0, ROLE_ADD: 1, ROLE_DROP: 2}
    for instruction in instructions:
        owner = instruction.owner
        if not 0 <= owner < count:
            continue
        if 0 <= instruction.reference < count:
            references[owner].add(instruction.reference)
        if instruction.opcode >= 0:
            opcodes[owner].add(instruction.opcode)
        for node, role in zip(instruction.nodes, instruction.roles):
            if node < 0:
                continue
            all_nodes[owner].add(node)
            if role in role_slot:
                operands[role_slot[role]][owner].add(node)
    channels = torch.zeros((len(SEMANTIC_RELATION_NAMES), count, count), dtype=torch.bool)
    for first in range(count):
        for second in references[first]:
            channels[0, first, second] = True
            channels[1, second, first] = True
        for second in range(first + 1, count):
            flags = [bool(all_nodes[first] & all_nodes[second]),
                     bool(opcodes[first] & opcodes[second])]
            flags.extend(bool(role[first] & role[second]) for role in operands)
            for channel, flag in enumerate(flags, start=2):
                if flag:
                    channels[channel, first, second] = channels[channel, second, first] = True
    return channels


class TransformerLayer(nn.Module):
    def __init__(self, config, relation_count=0):
        super().__init__()
        hidden, heads = config['hidden_size'], config['num_attention_heads']
        self.heads, self.width = heads, hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.output = nn.Linear(hidden, hidden)
        self.relation_bias = nn.Parameter(torch.zeros(heads, relation_count)) if relation_count else None
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, config['intermediate_size']), nn.GELU(),
                                 nn.Dropout(config['dropout']), nn.Linear(config['intermediate_size'], hidden))
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, hidden, valid, channels=None):
        normalized = self.norm1(hidden)
        batch, length, _ = normalized.shape
        query, key, value = self.qkv(normalized).reshape(
            batch, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        if self.relation_bias is not None:
            bias = torch.einsum('hr,brij->bhij', self.relation_bias.to(query.dtype), channels.to(query.dtype))
        else:
            bias = torch.zeros((batch, 1, 1, length), dtype=query.dtype, device=query.device)
        # All visible instructions may exchange information; typed channels bias
        # the dependency/operand paths. Padding keys remain inaccessible.
        bias = bias.masked_fill(~valid[:, None, None, :], float('-inf'))
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=bias,
                                                  dropout_p=self.dropout.p if self.training else 0.)
        hidden = hidden + self.dropout(self.output(attended.transpose(1, 2).reshape(batch, length, -1)))
        return hidden + self.dropout(self.ffn(self.norm2(hidden)))


class HIRCodeEncoder(nn.Module):
    """Two-stage H-IR encoder with local token and global instruction states."""
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        hidden, heads = config['hidden_size'], config['num_attention_heads']
        if (hidden < 2 or hidden % heads or config['max_seq_length'] < 7
                or min(config['chunk_batch_size'], config['local_num_hidden_layers'],
                       config['semantic_num_hidden_layers'], config['intermediate_size']) < 1
                or config['num_hidden_layers'] != (config['local_num_hidden_layers'] +
                                                   config['semantic_num_hidden_layers'])):
            raise ValueError('Invalid HIR-CodeBERT width, heads, layer split, or sequence limit')
        self.chunk_batch_size = config['chunk_batch_size']
        self.use_checkpointing = config['gradient_checkpointing']
        self.tokens = nn.Embedding(NUM_TYPES + 2, hidden, padding_idx=PAD)
        self.positions = nn.Embedding(config['max_seq_length'], hidden)
        self.roles = nn.Embedding(NUM_ROLES, hidden)
        self.local_summary = nn.Parameter(torch.zeros(1, 1, hidden))
        self.node_identity = nn.Linear(hidden, hidden, bias=False)
        self.edge_identity = nn.Linear(hidden, hidden, bias=False)
        self.order_projection = nn.Linear(1, hidden, bias=False)
        self.local_norm, self.semantic_norm, self.output_norm = (nn.LayerNorm(hidden) for _ in range(3))
        self.dropout = nn.Dropout(config['dropout'])
        self.local_layers = nn.ModuleList(
            TransformerLayer(config) for _ in range(config['local_num_hidden_layers']))
        self.semantic_layers = nn.ModuleList(
            TransformerLayer(config, len(SEMANTIC_RELATION_NAMES))
            for _ in range(config['semantic_num_hidden_layers']))
        self.instruction_pool = AttentionPooling(hidden)
        self.node_fusion = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.node_pool = AttentionPooling(hidden)
        self.register_buffer('frequencies', torch.exp(-math.log(10000.) * torch.arange(0, hidden, 2) / hidden),
                             persistent=False)
        self._analysis_view = self._analysis_instructions = self._analysis_relations = None
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
        nn.init.normal_(self.local_summary, std=0.02)
        with torch.no_grad():
            self.tokens.weight[PAD].zero_()
            for layer in self.semantic_layers:
                layer.relation_bias.zero_()

    def _identity(self, codes):
        phase = codes.clamp_min(0).float().unsqueeze(-1) * self.frequencies
        result = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)[..., :self.config['hidden_size']]
        return result * (codes >= 0).unsqueeze(-1)

    def _local_run(self, types, roles, node_codes, owner_codes, payload_codes, orders, valid):
        length = types.shape[1]
        hidden = (self.tokens(types) + self.positions(torch.arange(length, device=types.device))[None]
                  + self.roles(roles) + self.node_identity(self._identity(node_codes))
                  + self.edge_identity(self._identity(owner_codes) + self._identity(payload_codes))
                  + self.order_projection(orders.clamp_min(0).float().log1p().unsqueeze(-1)))
        summary = torch.cat((self.local_summary.expand(hidden.shape[0], -1, -1),
                             hidden.new_zeros((hidden.shape[0], hidden.shape[1] - 1, hidden.shape[2]))), dim=1)
        hidden = hidden + summary
        hidden = self.dropout(self.local_norm(hidden))
        for layer in self.local_layers:
            hidden = layer(hidden, valid)
        return self.output_norm(hidden)

    def _semantic_run(self, hidden, valid, channels):
        hidden = self.dropout(self.semantic_norm(hidden))
        for layer in self.semantic_layers:
            hidden = layer(hidden, valid, channels)
        return self.output_norm(hidden)

    def encode_many(self, inputs, max_batch_tokens=32768):
        device, hidden_size = self.tokens.weight.device, self.config['hidden_size']
        if len(inputs) == 1 and inputs[0] is self._analysis_view:
            by_view = [self._analysis_instructions]
            cached_relations = [self._analysis_relations]
        else:
            by_view = [visible_instructions(view) for view in inputs]
            cached_relations = [semantic_relations(view, instructions)
                                for view, instructions in zip(inputs, by_view)]
            if len(inputs) == 1:
                self._analysis_view, self._analysis_instructions = inputs[0], by_view[0]
                self._analysis_relations = cached_relations[0]
        requests = [(owner, index, instruction) for owner, instructions in enumerate(by_view)
                    for index, instruction in enumerate(instructions)]
        requests.sort(key=lambda item: len(item[2].types))
        groups, group = [], []
        for request in requests:
            length = len(request[2].types) + 1
            if length > max_batch_tokens:
                raise ValueError('A local H-IR instruction exceeds the forward token budget')
            if group and (len(group) >= self.chunk_batch_size or length * (len(group) + 1) > max_batch_tokens):
                groups.append(group)
                group = []
            group.append(request)
        if group:
            groups.append(group)

        instruction_states = [[None] * len(instructions) for instructions in by_view]
        node_values, node_ids, node_edges = [[] for _ in inputs], [[] for _ in inputs], [[] for _ in inputs]
        calls = []
        for group in groups:
            width = max(len(instruction.types) + 1 for _, _, instruction in group)
            shape = (len(group), width)
            types = torch.full(shape, PAD, dtype=torch.long)
            role_ids = torch.full(shape, ROLE_NONE, dtype=torch.long)
            node_codes = torch.full(shape, -1, dtype=torch.long)
            owner_codes = torch.full(shape, -1, dtype=torch.long)
            payload_codes = torch.full(shape, -1, dtype=torch.long)
            orders = torch.full(shape, -1, dtype=torch.long)
            valid = torch.zeros(shape, dtype=torch.bool)
            node_locations, node_outputs, own_tokens = [], [], {}
            for row, (owner, index, instruction) in enumerate(group):
                view, length = inputs[owner], len(instruction.types) + 1
                valid[row, :length] = True
                role_ids[row, 0] = ROLE_SUMMARY
                types[row, 1:length] = torch.tensor(instruction.types)
                role_ids[row, 1:length] = torch.tensor(instruction.roles)
                orders[row, 1:length] = torch.tensor(instruction.orders)
                node_codes[row, 1:length] = torch.tensor(
                    [view.node_codes[node] if node >= 0 else -1 for node in instruction.nodes])
                owner_codes[row, 1:length] = torch.tensor(
                    [view.edge_codes[edge] if edge >= 0 else -1 for edge in instruction.owners])
                payload_codes[row, 1:length] = torch.tensor(
                    [view.edge_codes[edge] if edge >= 0 else -1 for edge in instruction.payload_edges])
                for position, node in enumerate(instruction.nodes):
                    if node >= 0:
                        node_locations.append((row, position + 1))
                        node_outputs.append((owner, node, instruction.owner))
                own_tokens[owner] = own_tokens.get(owner, 0) + length
            tensors = [value.to(device) for value in (
                types, role_ids, node_codes, owner_codes, payload_codes, orders, valid)]
            if self.use_checkpointing and self.training and torch.is_grad_enabled():
                encoded = checkpoint(self._local_run, *tensors, use_reentrant=False)
            else:
                encoded = self._local_run(*tensors)
            for row, (owner, index, _) in enumerate(group):
                instruction_states[owner][index] = encoded[row, 0]
            for location, (owner, node, edge) in zip(node_locations, node_outputs):
                node_values[owner].append(encoded[location])
                node_ids[owner].append(node)
                node_edges[owner].append(edge)
            calls.append({'tokens': sum(own_tokens.values()), 'padded_tokens': len(group) * width,
                          'sequences': len(group), 'max_sequence_length': width, 'owner_tokens': own_tokens})

        local_edges, edge_valids = [], []
        for view, instructions, states in zip(inputs, by_view, instruction_states):
            owned = [(state, instruction.owner, instruction.contribution)
                     for state, instruction in zip(states, instructions)
                     if instruction.owner >= 0 and instruction.contribution > 0]
            present = torch.zeros(view.num_edges, dtype=torch.bool, device=device)
            if owned:
                values = torch.stack([state for state, _, _ in owned])
                ids = torch.tensor([edge for _, edge, _ in owned], dtype=torch.long, device=device)
                contributions = torch.tensor([weight for _, _, weight in owned], device=device)
                local_edges.append(self.instruction_pool(values, ids, view.num_edges, contributions))
                present[ids] = True
            else:
                local_edges.append(self.tokens.weight.new_zeros((view.num_edges, hidden_size)))
            edge_valids.append(present)

        if local_edges and max((len(value) for value in local_edges), default=0):
            width = max(len(value) for value in local_edges)
            semantic = self.tokens.weight.new_zeros((len(inputs), width, hidden_size))
            valid = torch.zeros((len(inputs), width), dtype=torch.bool, device=device)
            channels = torch.zeros((len(inputs), len(SEMANTIC_RELATION_NAMES), width, width),
                                   dtype=torch.bool, device=device)
            for row, (values, relations, present) in enumerate(zip(local_edges, cached_relations, edge_valids)):
                semantic[row, :len(values)] = values
                valid[row, :len(values)] = present
                channels[row, :, :len(values), :len(values)] = relations.to(device)
                if not present.any():
                    valid[row, 0] = True
            if self.use_checkpointing and self.training and torch.is_grad_enabled():
                semantic = checkpoint(self._semantic_run, semantic, valid, channels, use_reentrant=False)
            else:
                semantic = self._semantic_run(semantic, valid, channels)
            edge_outputs = [semantic[row, :view.num_edges] * edge_valids[row][:, None]
                            for row, view in enumerate(inputs)]
        else:
            edge_outputs = local_edges

        results = []
        for owner, view in enumerate(inputs):
            if node_values[owner]:
                local = torch.stack(node_values[owner])
                edge_ids = torch.tensor(node_edges[owner], dtype=torch.long, device=device)
                context = local.new_zeros(local.shape)
                attached = edge_ids >= 0
                if attached.any():
                    context[attached] = edge_outputs[owner][edge_ids[attached]]
                fused = self.node_fusion(torch.cat((local, context), dim=-1))
                ids = torch.tensor(node_ids[owner], dtype=torch.long, device=device)
                nodes = self.node_pool(fused, ids, view.num_nodes)
                occurrences = torch.bincount(ids, minlength=view.num_nodes)
            else:
                nodes = self.tokens.weight.new_zeros((view.num_nodes, hidden_size))
                occurrences = torch.zeros(view.num_nodes, dtype=torch.long, device=device)
            results.append(EncodedStructure(nodes, edge_outputs[owner], occurrences))
        return results, calls

    def forward(self, view):
        return self.encode_many([view])[0][0]

    def parameter_counts(self):
        structural = sum(layer.relation_bias.numel() for layer in self.semantic_layers)
        return {'encoder': sum(p.numel() for p in self.parameters()),
                'structural_attention': structural,
                'local_transformer': sum(p.numel() for layer in self.local_layers for p in layer.parameters()),
                'semantic_transformer': sum(p.numel() for layer in self.semantic_layers for p in layer.parameters()),
                'node_attention_pooling': sum(p.numel() for p in self.node_pool.parameters()),
                'instruction_attention_pooling': sum(p.numel() for p in self.instruction_pool.parameters())}


class HIRCodeBERT(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        self.model_config = model_config(args)
        self.num_classes = num_classes
        self.encoder = HIRCodeEncoder(self.model_config)
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
        index, count = data.hyperedge_index, getattr(data, 'num_hyperedges', None)
        count = None if count is None else int(count)
        key = (index._version, int(data.num_nodes), count, self.identity_seed)
        if self._source_index is index and getattr(self, '_source_key', None) == key:
            return
        self.program = compile_data(data)
        self.chunks, stats = chunk_hir(self.program, self.model_config['max_seq_length'] - 1)
        generator = torch.Generator().manual_seed(self.identity_seed)
        nodes = (torch.randperm(self.program.num_nodes, generator=generator) + 1).tolist()
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
        if saved.get('kind') != 'hircodebert_node_classifier':
            raise ValueError('Expected a HIR-CodeBERT node classifier checkpoint')
        args = SimpleNamespace(**{'hir_' + key: value for key, value in saved['model_config'].items()}, seed=saved['seed'])
        model = cls(saved['num_classes'], args).to(device)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        return model.eval(), saved


def load_encoder(model, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('kind') != 'hircodebert_pretrained_encoder' or saved.get('input_mode') != 'Structure-only':
        raise ValueError('Expected a Structure-only HIR-CodeBERT encoder checkpoint')
    shape_keys = ('hidden_size', 'local_num_hidden_layers', 'semantic_num_hidden_layers',
                  'num_attention_heads', 'intermediate_size', 'max_seq_length')
    if any(model.model_config[key] != saved['model_config'][key] for key in shape_keys):
        raise ValueError('Incompatible HIR-CodeBERT encoder architecture')
    state = saved['shared_encoder_state_dict']
    result = model.encoder.load_state_dict(state, strict=True)
    return {'loaded_keys': list(state), 'missing_keys': result.missing_keys,
            'unexpected_keys': result.unexpected_keys, 'pretrain_seed': saved['seed'],
            'best_epoch': saved['best_epoch'], 'source': str(path)}
