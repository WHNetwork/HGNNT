"""Batch independent H-IR packets through the existing HIREncoder BERT.

Only the BERT sequence dimension is combined: local IDs, incidences, adapters,
attention masks and prediction targets remain owned by their original packets.
No parameters or checkpoint keys are introduced here.
"""

from collections import defaultdict

import torch
from torch.utils.checkpoint import checkpoint

from hir import EDGE_ID, NODE, ORDER, PAD


def encode_packets(encoder, batches, projected, weights, node_batch_size, max_batch_tokens):
    requests, query_sizes = [], []
    main_enabled = weights['membership'] > 0 or weights['feature'] > 0
    for owner, batch in enumerate(batches):
        for branch, chunks in (('main', batch.view.chunks if main_enabled else ()),
                               ('relation', batch.view.relation_chunks if weights['relation'] > 0
                                and len(batch.relation_pairs) else ())):
            requests.extend((owner, branch, chunk) for chunk in chunks)
        candidates = batch.membership_pairs[:, 0].unique(sorted=True).tolist() if weights['membership'] > 0 else []
        query_sizes.append(len(candidates))
        requests.extend((owner, 'query', (index, node)) for index, node in enumerate(candidates))

    def length(request):
        return 1 if request[1] == 'query' else len(request[2].tokens)

    # Length buckets prevent short tails/queries from padding to a long chunk.
    requests.sort(key=lambda request: (request[1] == 'query', length(request)))
    groups, group, tokens = [], [], 0
    for request in requests:
        size = length(request)
        query = request[1] == 'query'
        count_limit = node_batch_size if query else encoder.chunk_batch_size
        padded = size * (len(group) + 1)
        if group and (query != (group[0][1] == 'query') or len(group) >= count_limit
                      or tokens + size > max_batch_tokens or padded > max_batch_tokens
                      or padded > 1.25 * (tokens + size)):
            groups.append(group)
            group, tokens = [], 0
        if size > max_batch_tokens:
            raise ValueError('A BERT sequence exceeds bert_max_batch_tokens')
        group.append(request)
        tokens += size
    if group:
        groups.append(group)

    device, hidden = projected[0].device, projected[0].shape[-1]
    features = [torch.cat((value, value.new_zeros((1, hidden)))) for value in projected]
    states, targets, contributions = defaultdict(list), defaultdict(list), defaultdict(list)
    calls = []
    for group in groups:
        width = max(map(length, group))
        types = torch.full((len(group), width), PAD, dtype=torch.long)
        payload = torch.zeros_like(types)
        numeric = torch.zeros_like(types, dtype=torch.bool)
        orders, mask = torch.zeros_like(types), torch.zeros_like(types)
        node_ids = torch.empty_like(types)
        locations, outputs, output_weights = [], [], []
        owned_tokens, owned_padded = defaultdict(int), defaultdict(int)
        for row, (owner, branch, record) in enumerate(group):
            size = length(group[row])
            mask[row, :size] = 1
            node_ids[row].fill_(len(projected[owner]))
            owned_tokens[owner] += size
            owned_padded[owner] += width
            if branch == 'query':
                index, node = record
                types[row, 0], payload[row, 0], node_ids[row, 0] = NODE, node, node
                numeric[row, 0] = True
                readouts = ((0, index, 1.),)
            else:
                program = batches[owner].view.program
                space = program.space
                raw = torch.tensor(record.tokens, dtype=torch.long)
                is_node = (raw >= space.node(0)) & (raw < space.edge(0))
                is_edge = (raw >= space.edge(0)) & (raw < space.order(0))
                is_order = raw >= space.order(0)
                types[row, :size] = torch.where(is_node, NODE, torch.where(is_edge, EDGE_ID, torch.where(is_order, ORDER, raw)))
                payload[row, :size] = torch.where(is_node, raw - space.node(0), torch.where(
                    is_edge, raw - space.edge(0), torch.where(is_order, raw - space.order(0), 0)))
                numeric[row, :size] = is_node | is_edge | is_order
                node_ids[row, :size] = torch.where(is_node, raw - space.node(0), len(projected[owner]))
                orders[row, :size] = torch.tensor(record.orders)
                readouts = record.readouts
            for position, target, weight in readouts:
                locations.append((row, position))
                outputs.append((owner, branch, target))
                output_weights.append(weight)
        types, payload, numeric, orders, mask, node_ids = [value.to(device) for value in
                                                         (types, payload, numeric, orders, mask, node_ids)]
        phase = payload.unsqueeze(-1).to(projected[0].dtype) * encoder.frequencies
        identity = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)[..., :hidden]
        identity = identity * numeric.unsqueeze(-1)
        # This stacks feature EMBEDDINGS for independent sequences, never raw
        # graph feature matrices or graph IDs in a shared incidence namespace.
        feature_rows = torch.stack([features[owner][node_ids[row]] for row, (owner, _, _) in enumerate(group)])
        inputs = (encoder.bert.embeddings.word_embeddings(types) + identity + feature_rows
                  + encoder.order_projection(orders.to(projected[0].dtype).log1p().unsqueeze(-1)))
        positions = torch.tensor(locations, dtype=torch.long, device=device)
        arguments = (inputs, mask, positions[:, 0], positions[:, 1])
        if encoder.use_checkpointing and encoder.training and torch.is_grad_enabled():
            encoded = checkpoint(encoder._encode_readouts, *arguments, use_reentrant=False)
        else:
            encoded = encoder._encode_readouts(*arguments)
        # Group readouts by their owner after the one shared BERT call.
        slots = defaultdict(list)
        for index, (owner, branch, target) in enumerate(outputs):
            key = (owner, branch)
            slots[key].append(index)
            targets[key].append(target)
            contributions[key].append(output_weights[index])
        for key, indices in slots.items():
            states[key].append(encoded[torch.tensor(indices, device=device)])
        calls.append({'tokens': sum(owned_tokens.values()), 'padded_tokens': len(group) * width,
                      'sequences': len(group), 'max_sequence_length': width,
                      'owner_tokens': dict(owned_tokens), 'owner_padded_tokens': dict(owned_padded)})
    result = {}
    for owner, batch in enumerate(batches):
        for branch in ('main', 'relation', 'query'):
            key = (owner, branch)
            count = query_sizes[owner] if branch == 'query' else len(batch.view.program.hyperedges)
            if states[key]:
                values = torch.cat(states[key])
                indices = torch.tensor(targets[key], dtype=torch.long, device=device)
                factors = torch.tensor(contributions[key], dtype=values.dtype, device=device)
                result[key] = values.new_zeros((count, hidden)).index_add(0, indices, values * factors.unsqueeze(-1))
            else:
                result[key] = projected[owner].new_zeros((count, hidden))
    return result, calls
