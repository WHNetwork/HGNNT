"""Three self-supervised heads around the actual downstream HIREncoder."""

import torch
from torch import nn
from torch.nn import functional as F

from lib_models.HNN.hirbert import HIREncoder
from lib_utils.hirbert_pretrain_utils import edge_to_node, representation_stats


class FeatureDecoder(nn.Module):
    def __init__(self, features, hidden, statistics):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, features))
        self.kind, positive, center, scale = statistics
        self.register_buffer('positive', positive)
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)


class HIRPretrainer(nn.Module):
    def __init__(self, num_features, config, weights, feature_statistics, node_batch_size=256,
                 dataset_specs=None):
        super().__init__()
        self.encoder = HIREncoder(1 if dataset_specs else num_features, config)
        self.model_config, self.weights = dict(config), dict(weights)
        self.num_features, self.node_batch_size = num_features, node_batch_size
        hidden = config['hidden_size']
        self.membership_head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.feature_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, num_features))
        self.relation_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        for name in ('membership', 'feature', 'relation'):
            getattr(self, name + '_head').requires_grad_(weights[name] > 0)
        self.feature_kind, pos_weight, center, scale = feature_statistics
        # These statistics/heads belong to SSL only, not to HIREncoder transfer.
        self.register_buffer('pos_weight', pos_weight)
        self.register_buffer('feature_center', center)
        self.register_buffer('feature_scale', scale)
        self.dataset_specs = dataset_specs
        if dataset_specs:
            # There is exactly one BERT/order/token encoder. Feature dimensions
            # are owned by explicit dataset adapters, never concatenated.
            self.encoder.feature_projection = nn.Identity()
            del self.feature_head
            self.adapters = nn.ModuleDict({name: nn.Linear(size, hidden)
                                           for name, (size, _) in dataset_specs.items()})
            self.feature_decoders = nn.ModuleDict({name: FeatureDecoder(size, hidden, stats)
                                                   for name, (size, stats) in dataset_specs.items()})
            self.feature_decoders.requires_grad_(weights['feature'] > 0)

    def encode(self, view, relation=False):
        projection = self.adapters[view.dataset] if self.dataset_specs else self.encoder.feature_projection
        projected = projection(view.x)
        # A node-wide mask is applied AFTER projection as well, removing bias.
        projected = projected.index_fill(0, view.masked_nodes, 0)
        program = view.relation_program if relation else view.program
        chunks = view.relation_chunks if relation else view.chunks
        edges = self.encoder(projected, program, chunks)
        nodes, degree = edge_to_node(edges, view.incidence, len(view.x))
        return projected, edges, nodes, degree

    @staticmethod
    def target_mean(values, batch, task):
        if batch.target_weights is None:
            return values.mean()
        weights = batch.target_weights[task].to(values)
        return (values * weights).sum() / weights.sum().clamp_min(1e-12)

    def forward_packets(self, batches, max_batch_tokens):
        """One bounded autograd graph, batched BERT calls, independent losses."""
        from lib_models.HNN.hirbert_packet_batch import encode_packets

        projected = []
        for batch in batches:
            view = batch.view
            projection = self.adapters[view.dataset] if self.dataset_specs else self.encoder.feature_projection
            projected.append(projection(view.x).index_fill(0, view.masked_nodes, 0))
        encoded, calls = encode_packets(self.encoder, batches, projected, self.weights,
                                        self.node_batch_size, max_batch_tokens)
        losses, metrics = [], []
        for owner, batch in enumerate(batches):
            states = {branch: encoded[(owner, branch)] for branch in ('main', 'relation', 'query')}
            states['projected'] = projected[owner]
            loss, values = self.forward(batch, states)
            losses.append(loss)
            metrics.append(values)
        return losses, metrics, calls

    def forward(self, batch, states=None):
        view = batch.view
        # A zero-valued differentiable anchor handles genuinely absent examples
        # (e.g. feature targets losing their last context through corruption).
        zero = self.encoder.order_projection.weight.sum() * 0
        losses = dict.fromkeys(('membership', 'feature', 'relation'), zero)
        metrics = {'membership_acc': 0.0, 'feature_metric': 0.0, 'relation_acc': 0.0,
                   'membership_count': len(batch.membership_labels), 'feature_count': len(batch.feature_nodes),
                   'relation_count': len(batch.relation_labels), 'skipped_feature_nodes': batch.skipped_feature_nodes}
        main_enabled = self.weights['membership'] > 0 or self.weights['feature'] > 0
        if main_enabled:
            if states is None:
                projected, edges, nodes, _ = self.encode(view)
            else:
                projected, edges = states['projected'], states['main']
                nodes, _ = edge_to_node(edges, view.incidence, len(view.x))
            metrics.update(representation_stats(edges, view.incidence, len(view.x)))
        if self.weights['membership'] > 0 and len(batch.membership_pairs):
            candidates, inverse = batch.membership_pairs[:, 0].unique(sorted=True, return_inverse=True)
            queries = (self.encoder.encode_node_queries(projected, candidates, self.node_batch_size)
                       if states is None else states['query'])
            node = (queries + nodes[candidates])[inverse]
            edge = edges[batch.membership_pairs[:, 1]]
            inputs = torch.cat((node, edge, node * edge, (node - edge).abs()), dim=-1)
            logits = self.membership_head(inputs).squeeze(-1)
            # Equal aggregate weight for positive/negative classes even if the
            # configurable negative ratio differs from one.
            positive = batch.membership_labels.sum()
            negative = len(batch.membership_labels) - positive
            pos_weight = negative / positive.clamp_min(1) if batch.membership_pos_weight is None else logits.new_tensor(batch.membership_pos_weight)
            losses['membership'] = self.target_mean(F.binary_cross_entropy_with_logits(
                logits, batch.membership_labels, pos_weight=pos_weight, reduction='none'), batch, 'membership')
            metrics['membership_acc'] = self.target_mean(
                ((logits.detach() >= 0) == batch.membership_labels.bool()).float(), batch, 'membership').item()
        if self.weights['feature'] > 0 and len(batch.feature_nodes):
            # Only visible incidence-derived BERT messages reach this head.
            decoder = self.feature_decoders[view.dataset] if self.dataset_specs else None
            prediction = (decoder.head if decoder else self.feature_head)(nodes[batch.feature_nodes])
            target = batch.feature_targets
            kind = decoder.kind if decoder else self.feature_kind
            if kind == 'binary':
                losses['feature'] = self.target_mean(F.binary_cross_entropy_with_logits(
                    prediction, target, pos_weight=decoder.positive if decoder else self.pos_weight,
                    reduction='none').mean(-1), batch, 'feature')
                positive = target > 0
                # Positive-entry recall exposes the trivial all-zero solution.
                metrics['feature_metric'] = ((prediction.detach() >= 0) & positive).sum().item() / max(1, positive.sum().item())
            else:
                normalized = ((target - decoder.center) / decoder.scale if decoder
                              else (target - self.feature_center) / self.feature_scale)
                losses['feature'] = self.target_mean(F.mse_loss(prediction, normalized, reduction='none').mean(-1), batch, 'feature')
                metrics['feature_metric'] = losses['feature'].detach().sqrt().item()
        if self.weights['relation'] > 0 and len(batch.relation_pairs):
            if states is None:
                _, relation_edges, _, _ = self.encode(view, relation=True)
            else:
                relation_edges = states['relation']
            first, second = relation_edges[batch.relation_pairs].unbind(dim=1)
            inputs = torch.cat((first + second, (first - second).abs(), first * second), dim=-1)
            logits = self.relation_head(inputs)
            losses['relation'] = self.target_mean(F.cross_entropy(logits, batch.relation_labels, reduction='none'), batch, 'relation')
            metrics['relation_acc'] = self.target_mean(
                (logits.detach().argmax(-1) == batch.relation_labels).float(), batch, 'relation').item()
            if not main_enabled:
                metrics.update(representation_stats(relation_edges, view.incidence, len(view.x)))
        scales = batch.task_scales or dict.fromkeys(losses, 1.0)
        total = sum(self.weights[key] * scales[key] * losses[key] for key in losses)
        metrics.update({key + '_loss': value.detach().item() for key, value in losses.items()})
        metrics['loss'] = total.detach().item()
        return total, metrics
