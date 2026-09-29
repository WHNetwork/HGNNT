import torch
import torch.nn as nn
import torch.nn.functional as F

from lib_models.HNN.edgnn import EquivSetConv, JumpLinkConv
from lib_models.HNN.mlp import MLP


class OrderSplitEDHNN(nn.Module):
    """EDHNN with cardinality-specific propagation and shared parameters."""

    def __init__(self, num_features, num_targets, args):
        super().__init__()
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = args.dropout
        self.norm = args.normalization

        self.in_channels = num_features
        self.hidden_channels = args.MLP_hidden
        self.output_channels = num_targets

        self.mlp1_layers = args.MLP_num_layers
        self.mlp2_layers = args.MLP_num_layers if args.MLP2_num_layers < 0 else args.MLP2_num_layers
        self.mlp3_layers = args.MLP_num_layers if args.MLP3_num_layers < 0 else args.MLP3_num_layers
        self.num_layers = args.All_num_layers
        self.edconv_type = args.edconv_type
        self.order_fusion = args.order_fusion
        self.max_exact_order = args.max_exact_order
        self.order_heads = args.order_heads

        if self.max_exact_order < 2:
            raise ValueError('max_exact_order must be at least 2')
        if self.order_fusion == 'cross_attn':
            if self.order_heads <= 0 or self.hidden_channels % self.order_heads != 0:
                raise ValueError(
                    'OrderSplitEDHNN requires a positive order_heads that divides '
                    f'MLP_hidden for cross_attn, got {self.order_heads} and '
                    f'{self.hidden_channels}'
                )

        self.lin_in = nn.Linear(num_features, self.hidden_channels)
        if self.edconv_type == 'EquivSet':
            self.conv = EquivSetConv(
                self.hidden_channels,
                self.hidden_channels,
                mlp1_layers=self.mlp1_layers,
                mlp2_layers=self.mlp2_layers,
                mlp3_layers=self.mlp3_layers,
                alpha=args.alpha,
                aggr=args.aggregate,
                dropout=self.dropout,
                normalization=self.norm,
                input_norm=args.AllSet_input_norm,
            )
        elif self.edconv_type == 'JumpLink':
            self.conv = JumpLinkConv(
                self.hidden_channels,
                self.hidden_channels,
                mlp_layers=self.mlp1_layers,
                alpha=args.alpha,
                aggr=args.aggregate,
            )
        else:
            raise ValueError(f'Unsupported EDConv type: {self.edconv_type}')

        self.decoder = MLP(
            in_channels=self.hidden_channels,
            hidden_channels=args.decoder_hidden,
            out_channels=self.output_channels,
            num_layers=args.decoder_num_layer,
            dropout=self.dropout,
            Normalization=self.norm,
            InputNorm=False,
        )

        self.order_embedding = None
        self.attn_scorer = None
        self.cross_attention = None
        self.cross_attention_norm = None
        if self.order_fusion in ['attn', 'cross_attn']:
            # Exact orders 2..max_exact_order plus one overflow bucket.
            self.order_embedding = nn.Embedding(self.max_exact_order, self.hidden_channels)
        if self.order_fusion == 'attn':
            self.attn_scorer = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.Tanh(),
                nn.Linear(self.hidden_channels, 1),
            )
        elif self.order_fusion == 'cross_attn':
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_channels,
                num_heads=self.order_heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.cross_attention_norm = nn.LayerNorm(self.hidden_channels)
        elif self.order_fusion != 'mean':
            raise ValueError(f'Unsupported order fusion: {self.order_fusion}')

        self._order_cache_signature = None
        self._cached_order_layers = None
        self._printed_order_statistics = False

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.conv.reset_parameters()
        self.decoder.reset_parameters()
        if hasattr(self.act, 'reset_parameters'):
            self.act.reset_parameters()
        if self.order_embedding is not None:
            self.order_embedding.reset_parameters()
        if self.attn_scorer is not None:
            for module in self.attn_scorer:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        if self.cross_attention is not None:
            if self.cross_attention.in_proj_weight is not None:
                nn.init.xavier_uniform_(self.cross_attention.in_proj_weight)
            else:
                nn.init.xavier_uniform_(self.cross_attention.q_proj_weight)
                nn.init.xavier_uniform_(self.cross_attention.k_proj_weight)
                nn.init.xavier_uniform_(self.cross_attention.v_proj_weight)
            if self.cross_attention.in_proj_bias is not None:
                nn.init.constant_(self.cross_attention.in_proj_bias, 0.)
            nn.init.xavier_uniform_(self.cross_attention.out_proj.weight)
            if self.cross_attention.out_proj.bias is not None:
                nn.init.constant_(self.cross_attention.out_proj.bias, 0.)
            if self.cross_attention.bias_k is not None:
                nn.init.xavier_normal_(self.cross_attention.bias_k)
            if self.cross_attention.bias_v is not None:
                nn.init.xavier_normal_(self.cross_attention.bias_v)
            self.cross_attention_norm.reset_parameters()

    @staticmethod
    def _structure_signature(hyperedge_index):
        device = hyperedge_index.device
        return (
            hyperedge_index.data_ptr(),
            tuple(hyperedge_index.shape),
            device.type,
            device.index,
            hyperedge_index._version,
        )

    def _embedding_index(self, order):
        if order > self.max_exact_order:
            return self.max_exact_order - 1
        return order - 2

    def _build_order_layers(self, hyperedge_index, num_nodes=None):
        vertex, edge = hyperedge_index[0], hyperedge_index[1]
        if num_nodes is None:
            num_nodes = int(vertex.max().item()) + 1 if vertex.numel() > 0 else 0
        _, inverse, cardinalities = torch.unique(
            edge, sorted=True, return_inverse=True, return_counts=True
        )
        incidence_cardinality = cardinalities[inverse]

        distribution = {}
        for size in torch.unique(cardinalities, sorted=True).tolist():
            distribution[int(size)] = int((cardinalities == size).sum().item())

        layers = []
        self_loop_mask = incidence_cardinality == 1
        for order in range(2, self.max_exact_order + 1):
            real_count = int((cardinalities == order).sum().item())
            if real_count == 0:
                continue
            order_incidence_mask = incidence_cardinality == order
            mask = self_loop_mask | order_incidence_mask
            layer_edge_original = edge[mask]
            _, layer_edge = torch.unique(
                layer_edge_original, sorted=True, return_inverse=True
            )
            participation_mask = torch.zeros(
                num_nodes, dtype=torch.bool, device=vertex.device
            )
            participation_mask[vertex[order_incidence_mask]] = True
            layers.append({
                'order': order,
                'embedding_index': self._embedding_index(order),
                'vertex': vertex[mask],
                'edge': layer_edge,
                'participation_mask': participation_mask,
                'num_order_hyperedges': real_count,
                'fallback': False,
            })

        overflow_count = int((cardinalities > self.max_exact_order).sum().item())
        if overflow_count > 0:
            order_incidence_mask = incidence_cardinality > self.max_exact_order
            mask = self_loop_mask | order_incidence_mask
            layer_edge_original = edge[mask]
            _, layer_edge = torch.unique(
                layer_edge_original, sorted=True, return_inverse=True
            )
            participation_mask = torch.zeros(
                num_nodes, dtype=torch.bool, device=vertex.device
            )
            participation_mask[vertex[order_incidence_mask]] = True
            layers.append({
                'order': self.max_exact_order + 1,
                'embedding_index': self._embedding_index(self.max_exact_order + 1),
                'vertex': vertex[mask],
                'edge': layer_edge,
                'participation_mask': participation_mask,
                'num_order_hyperedges': overflow_count,
                'fallback': False,
            })

        if not layers:
            # With no cardinality >= 2, preserve EDHNN behavior by propagating
            # once over the complete incidence structure (including an empty one).
            _, remapped_edge = torch.unique(edge, sorted=True, return_inverse=True)
            layers.append({
                'order': 2,
                'embedding_index': self._embedding_index(2),
                'vertex': vertex,
                'edge': remapped_edge,
                'participation_mask': torch.ones(
                    num_nodes, dtype=torch.bool, device=vertex.device
                ),
                'num_order_hyperedges': 0,
                'fallback': True,
            })

        return layers, distribution

    def _print_order_statistics(self, distribution, layers):
        print('[OrderSplitEDHNN] Hyperedge cardinality distribution:')
        for size, count in distribution.items():
            print(f'  size={size}: {count} hyperedges')
        print('\n[OrderSplitEDHNN] Active order layers:')
        for layer in layers:
            if layer['fallback']:
                print('  fallback=all: 0 hyperedges (no cardinality >= 2)')
            else:
                order = layer['order']
                label = f'>{self.max_exact_order}' if order > self.max_exact_order else str(order)
                print(f'  order={label}: {layer["num_order_hyperedges"]} hyperedges')

    def _get_order_layers(self, hyperedge_index, num_nodes):
        signature = self._structure_signature(hyperedge_index) + (num_nodes,)
        if signature != self._order_cache_signature:
            layers, distribution = self._build_order_layers(hyperedge_index, num_nodes)
            self._cached_order_layers = layers
            self._order_cache_signature = signature
            if not self._printed_order_statistics:
                self._print_order_statistics(distribution, layers)
                self._printed_order_statistics = True
        return self._cached_order_layers

    @staticmethod
    def _effective_participation(participation):
        effective_mask = participation.clone()
        no_valid = ~effective_mask.any(dim=1)
        effective_mask[no_valid] = True
        return effective_mask

    def _masked_mean(self, H, participation):
        effective_mask = self._effective_participation(participation)
        mask = effective_mask.unsqueeze(-1).to(H.dtype)
        return (H * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _fuse_orders(self, order_embeddings, layers):
        # H contains one replica per active order: [N, L, D].
        H = torch.stack(order_embeddings, dim=1)
        participation = torch.stack(
            [layer['participation_mask'] for layer in layers], dim=1
        ).to(device=H.device)
        if self.order_fusion == 'mean':
            return self._masked_mean(H, participation)

        order_indices = torch.tensor(
            [layer['embedding_index'] for layer in layers],
            dtype=torch.long,
            device=H.device,
        )
        Z = H + self.order_embedding(order_indices).unsqueeze(0)
        effective_mask = self._effective_participation(participation)
        if self.order_fusion == 'attn':
            score = self.attn_scorer(Z)
            score = score.masked_fill(
                ~effective_mask.unsqueeze(-1), torch.finfo(score.dtype).min
            )
            alpha = torch.softmax(score, dim=1)
            return (alpha * H).sum(dim=1)

        attended, _ = self.cross_attention(
            Z,
            Z,
            H,
            key_padding_mask=~effective_mask,
            need_weights=False,
        )
        H_cross = self.cross_attention_norm(H + attended)
        return self._masked_mean(H_cross, effective_mask)

    def forward(self, data):
        layers = self._get_order_layers(data.hyperedge_index, data.x.shape[-2])

        x_base = F.relu(self.lin_in(data.x))
        x_base = F.dropout(x_base, p=self.dropout, training=self.training)
        x0 = x_base

        order_embeddings = []
        for layer in layers:
            x_r = x_base
            for _ in range(self.num_layers):
                x_r, _ = self.conv(x_r, layer['vertex'], layer['edge'], x0)
                x_r = self.act(x_r)
                x_r = F.dropout(x_r, p=self.dropout, training=self.training)
            order_embeddings.append(x_r)

        x = self._fuse_orders(order_embeddings, layers)
        output = self.decoder(x)
        return output, None
