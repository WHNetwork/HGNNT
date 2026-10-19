"""Three structural objectives sharing one HIRGraphEncoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from lib_models.hirgraphbert import HIRGraphEncoder


class HIRGraphPretrainer(nn.Module):
    def __init__(self, config, weights=None, encoder_cls=HIRGraphEncoder):
        super().__init__()
        self.model_config = dict(config)
        self.weights = weights or {'membership': 1., 'relation': .5, 'hir': 1.}
        self.encoder = encoder_cls(config)
        hidden = config['hidden_size']
        self.membership_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.relation_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.hir_context = nn.MultiheadAttention(hidden, config['num_attention_heads'], batch_first=True)
        self.hir_norm = nn.LayerNorm(hidden)
        self.mode_head, self.order_head = nn.Linear(hidden, 3), nn.Linear(hidden, hidden)
        self.reference_query, self.reference_key = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.no_reference = nn.Parameter(torch.zeros(1, hidden))
        self.payload_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    @staticmethod
    def pair_features(first, second):
        return torch.cat((first + second, (first - second).abs(), first * second), dim=-1)

    @staticmethod
    def weighted_mean(values, weights):
        weights = weights.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1e-12)

    def _hir_loss(self, encoded, targets):
        device = encoded.nodes.device
        edge_ids = torch.tensor(targets.edges, device=device)
        query = encoded.edges[edge_ids]
        visible = torch.ones(len(encoded.edges), dtype=torch.bool, device=device)
        visible[list(targets.hidden_edges)] = False
        context = torch.cat((encoded.nodes, encoded.edges[visible]))
        # Only visible NODE/EDGE states from this independently masked branch.
        if not len(context):
            context = self.no_reference
        attended, _ = self.hir_context(query[None], context[None], context[None], need_weights=False)
        state = self.hir_norm(query + attended[0])
        mode_logits = self.mode_head(state)
        # Exact numeric ORDER prediction over 0..N with a shared analytic code,
        # rather than a graph-size-specific learned table or a lossy order bin.
        order_codes = self.encoder._identity(torch.arange(len(encoded.nodes) + 1, device=device))
        order_logits = self.order_head(state) @ order_codes.t().to(state.dtype) / math.sqrt(state.shape[-1])
        mode_labels = torch.tensor(targets.modes, device=device)
        order_labels = torch.tensor(targets.orders, device=device)
        losses, correct, details = [], [], {'hir_mode_acc': 0., 'hir_order_acc': 0.,
                                          'hir_mode_count': 0, 'hir_order_count': 0}
        # Fragment payload recovery has no full-instruction MODE/ORDER label.
        # Exclude ignored fields explicitly: CE over all ignored labels is NaN.
        for name, logits, labels in (('mode', mode_logits, mode_labels), ('order', order_logits, order_labels)):
            usable = labels >= 0
            if usable.any():
                losses.append(F.cross_entropy(logits[usable].float(), labels[usable]))
                accuracy = (logits[usable].argmax(-1) == labels[usable]).float().mean()
                correct.append(accuracy)
                details['hir_' + name + '_acc'] = accuracy.detach().item()
                details['hir_' + name + '_count'] = int(usable.sum().item())
        keys = torch.cat((self.no_reference, encoded.edges))
        reference_logits = self.reference_query(state) @ self.reference_key(keys).t() / math.sqrt(state.shape[-1])
        # Removed definitions have zero/masked states, not valid REF candidates.
        candidates = torch.cat((visible.new_ones(1), visible))
        reference_logits = reference_logits.masked_fill(~candidates[None], float('-inf'))
        reference_labels = torch.tensor([ref + 1 if ref >= 0 else 0 if ref == -1 else -100
                                         for ref in targets.references], device=device)
        usable = reference_labels >= 0
        if usable.any():
            losses.append(F.cross_entropy(reference_logits[usable].float(), reference_labels[usable]))
            correct.append((reference_logits[usable].argmax(-1) == reference_labels[usable]).float().mean())
        if targets.payloads:
            target_index = {edge: i for i, edge in enumerate(targets.edges)}
            rows = torch.tensor([target_index[edge] for edge, _, _ in targets.payloads], device=device)
            nodes = torch.tensor([node for _, node, _ in targets.payloads], device=device)
            labels = torch.tensor([label for _, _, label in targets.payloads], device=device)
            logits = self.payload_head(self.pair_features(state[rows], encoded.nodes[nodes]))
            losses.append(F.cross_entropy(logits.float(), labels))
            correct.append((logits.argmax(-1) == labels).float().mean())
        if not losses:
            raise ValueError('Masked H-IR target has no hidden supervised fields')
        return torch.stack(losses).mean(), torch.stack(correct).mean(), details

    def forward_packets(self, batches, max_batch_tokens=32768):
        views, ownership = [], []
        for owner, batch in enumerate(batches):
            for name in ('main', 'relation', 'hir'):
                view = getattr(batch, name)
                if view is not None:
                    views.append(view)
                    ownership.append((owner, name))
        encoded, calls = self.encoder.encode_many(views, max_batch_tokens)
        states = dict(zip(ownership, encoded))
        # Report packet-owned real tokens, merging the three independent views.
        for call in calls:
            packet_tokens = {}
            for view, count in call['owner_tokens'].items():
                owner = ownership[view][0]
                packet_tokens[owner] = packet_tokens.get(owner, 0) + count
            call['owner_tokens'] = packet_tokens
        device = self.no_reference.device
        totals, metrics = [], []
        for owner, batch in enumerate(batches):
            main = states.get((owner, 'main'))
            # A memory packet can hold a complete relation/HIR branch alone.
            # This only schedules the existing independent forwards differently;
            # encoder architecture, branch inputs and objective weights stay put.
            anchor = next(states[owner, name] for name in ('main', 'relation', 'hir') if (owner, name) in states)
            zero = anchor.nodes.sum() * 0. + anchor.edges.sum() * 0.
            total, row = zero, {}
            for task in ('membership', 'relation', 'hir'):
                loss, accuracy, count = zero, zero.detach(), 0
                if task == 'membership' and len(batch.membership_labels):
                    nodes, edges = batch.membership_pairs.to(device).unbind(1)
                    logits = self.membership_head(self.pair_features(main.nodes[nodes], main.edges[edges])).squeeze(-1)
                    labels = batch.membership_labels.to(device).float()
                    loss = self.weighted_mean(F.binary_cross_entropy_with_logits(logits.float(), labels, reduction='none'),
                                              batch.target_weights[task])
                    accuracy = self.weighted_mean(((logits >= 0) == labels.bool()).float(), batch.target_weights[task])
                    count = len(labels)
                elif task == 'relation' and len(batch.relation_labels):
                    edges = states[owner, 'relation'].edges
                    first, second = edges[batch.relation_pairs.to(device)].unbind(1)
                    logits = self.relation_head(self.pair_features(first, second))
                    labels = batch.relation_labels.to(device)
                    loss = self.weighted_mean(F.cross_entropy(logits.float(), labels, reduction='none'), batch.target_weights[task])
                    accuracy = self.weighted_mean((logits.argmax(-1) == labels).float(), batch.target_weights[task])
                    count = len(labels)
                elif task == 'hir' and batch.hir_targets.edges:
                    loss, accuracy, details = self._hir_loss(states[owner, 'hir'], batch.hir_targets)
                    row.update(details)
                    count = len(batch.hir_targets.edges)
                scale = batch.task_scales[task]
                total = total + self.weights[task] * scale * loss
                row[task + '_loss'], row[task + '_acc'] = loss.detach().item(), accuracy.detach().item()
                row[task + '_count'] = count
            row.setdefault('hir_mode_acc', 0.)
            row.setdefault('hir_order_acc', 0.)
            row.setdefault('hir_mode_count', 0)
            row.setdefault('hir_order_count', 0)
            row['node_repr_variance'] = (main.nodes.detach().float().var(0, unbiased=False).mean().item()
                                         if main is not None and len(main.nodes) else 0.)
            row['loss'] = total.detach().item()
            totals.append(total)
            metrics.append(row)
        return totals, metrics, calls

    def forward(self, batch):
        losses, metrics, _ = self.forward_packets([batch])
        return losses[0], metrics[0]
