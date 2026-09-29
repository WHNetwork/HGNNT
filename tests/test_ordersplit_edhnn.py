import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dhgbench'))

from lib_models.HNN.ordersplit_edhnn import OrderSplitEDHNN  # noqa: E402


def make_args(order_fusion):
    return SimpleNamespace(
        activation='relu',
        dropout=0.0,
        normalization='ln',
        MLP_hidden=8,
        MLP_num_layers=1,
        MLP2_num_layers=1,
        MLP3_num_layers=1,
        All_num_layers=1,
        edconv_type='EquivSet',
        alpha=0.2,
        aggregate='mean',
        AllSet_input_norm=True,
        decoder_hidden=8,
        decoder_num_layer=1,
        order_fusion=order_fusion,
        max_exact_order=4,
        order_heads=2,
    )


def make_data():
    incidences = []
    for node, edge in enumerate([10, 11, 12, 13, 14, 15]):
        incidences.append((node, edge))
    incidences.extend((node, 20) for node in [0, 1])
    incidences.extend((node, 30) for node in [0, 1, 2])
    incidences.extend((node, 40) for node in [0, 1, 2, 3])
    incidences.extend((node, 50) for node in [0, 1, 2, 3, 4])
    hyperedge_index = torch.tensor(incidences, dtype=torch.long).t().contiguous()
    return SimpleNamespace(x=torch.randn(6, 5), hyperedge_index=hyperedge_index)


class OrderSplitEDHNNTest(unittest.TestCase):
    def test_grouping_and_contiguous_remapping(self):
        data = make_data()
        model = OrderSplitEDHNN(5, 3, make_args('mean'))

        layers, distribution = model._build_order_layers(data.hyperedge_index)

        self.assertEqual(distribution, {1: 6, 2: 1, 3: 1, 4: 1, 5: 1})
        self.assertEqual([layer['order'] for layer in layers], [2, 3, 4, 5])
        self.assertEqual(
            [layer['num_order_hyperedges'] for layer in layers],
            [1, 1, 1, 1],
        )
        self.assertEqual([layer['vertex'].numel() for layer in layers], [8, 9, 10, 11])
        expected_participation = [
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
        for layer, expected in zip(layers, expected_participation):
            self.assertTrue(torch.equal(
                layer['participation_mask'].cpu(), torch.tensor(expected)
            ))
        for layer in layers:
            unique_edges = torch.unique(layer['edge'], sorted=True)
            expected = torch.arange(unique_edges.numel(), dtype=unique_edges.dtype)
            self.assertTrue(torch.equal(unique_edges.cpu(), expected))
            self.assertEqual(unique_edges.numel(), 7)  # six self-loops + one order edge

    def test_mean_fusion_masks_non_participating_orders(self):
        model = OrderSplitEDHNN(5, 3, make_args('mean'))
        order_embeddings = [
            torch.tensor([[1., 1.], [10., 10.], [100., 100.]]),
            torch.tensor([[3., 3.], [20., 20.], [200., 200.]]),
            torch.tensor([[9., 9.], [30., 30.], [300., 300.]]),
        ]
        participation_by_order = [
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ]
        layers = [
            {'participation_mask': torch.tensor(mask)}
            for mask in participation_by_order
        ]

        fused = model._fuse_orders(order_embeddings, layers)

        expected = torch.tensor([[2., 2.], [20., 20.], [200., 200.]])
        self.assertTrue(torch.equal(fused, expected))

    def test_all_fusions_forward_backward_without_mutating_input(self):
        for fusion in ['mean', 'attn', 'cross_attn']:
            with self.subTest(fusion=fusion):
                data = make_data()
                original_hyperedge_index = data.hyperedge_index.clone()
                model = OrderSplitEDHNN(5, 3, make_args(fusion))

                output, edge_output = model(data)
                self.assertEqual(output.shape, (6, 3))
                self.assertIsNone(edge_output)
                output.sum().backward()
                self.assertIsNotNone(model.lin_in.weight.grad)
                self.assertTrue(any(
                    parameter.grad is not None for parameter in model.conv.parameters()
                ))
                if fusion == 'attn':
                    self.assertIsNotNone(model.attn_scorer[0].weight.grad)
                if fusion == 'cross_attn':
                    self.assertIsNotNone(model.cross_attention.in_proj_weight.grad)
                if model.order_embedding is not None:
                    self.assertIsNotNone(model.order_embedding.weight.grad)
                self.assertTrue(torch.equal(data.hyperedge_index, original_hyperedge_index))


if __name__ == '__main__':
    unittest.main()
