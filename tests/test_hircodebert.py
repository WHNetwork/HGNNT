import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1] / 'dhgbench'
sys.path.insert(0, str(ROOT))

from hir import EXTEND, FRAGMENT, compile_hypergraph, chunk_hir
from hircodebert_cli import parse_args
from lib_models.hircodebert import (ROLE_ADD, HIRCodeEncoder, semantic_relations,
                                    visible_instructions)
from lib_models.hirgraphbert import structure_input


def program(edges, num_nodes=None):
    num_nodes = num_nodes if num_nodes is not None else max((node for edge in edges for node in edge), default=-1) + 1
    pairs = [(node, edge) for edge, members in enumerate(edges) for node in members]
    index = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()
    return compile_hypergraph(num_nodes, index, len(edges))


def config(**updates):
    result = dict(hidden_size=16, num_hidden_layers=2, local_num_hidden_layers=1,
                  semantic_num_hidden_layers=1, num_attention_heads=4,
                  intermediate_size=32, dropout=0., max_seq_length=16,
                  chunk_batch_size=4, gradient_checkpointing=True)
    result.update(updates)
    return result


class HIRCodeBERTContracts(unittest.TestCase):
    def test_visible_semantics_find_reference_and_operand_roles(self):
        compiled = program([tuple(range(5)), tuple(range(6))], 8)
        view = structure_input(compiled, chunk_hir(compiled, 63)[0], 64)
        instructions = visible_instructions(view)
        extend = next(item for item in instructions if item.opcode == EXTEND)
        self.assertEqual((extend.owner, extend.reference), (1, 0))
        self.assertIn(ROLE_ADD, extend.roles)
        channels = semantic_relations(view, instructions)
        self.assertTrue(channels[0, 1, 0])
        self.assertTrue(channels[1, 0, 1])

    def test_fragment_records_stay_local_and_recombine_by_edge(self):
        compiled = program([tuple(range(30))], 32)
        chunks, _ = chunk_hir(compiled, 10)
        view = structure_input(compiled, chunks, 11)
        instructions = visible_instructions(view)
        self.assertGreater(len(instructions), 1)
        self.assertTrue(all(item.opcode == FRAGMENT for item in instructions if item.owner == 0))
        self.assertEqual({item.owner for item in instructions if item.owner >= 0}, {0})
        self.assertAlmostEqual(sum(item.contribution for item in instructions), 1.)

    def test_masked_view_semantics_do_not_read_backing_membership(self):
        first = program([(0, 1, 2), (2, 3)], 5)
        second = program([(0, 3, 4), (1, 4)], 5)
        first_view = structure_input(first, (), 16, num_edges=2, masked_edges=(0,), masked_anchors={0: (0,)})
        second_view = structure_input(second, (), 16, num_edges=2, masked_edges=(0,), masked_anchors={0: (0,)})
        self.assertEqual(first_view, second_view)
        first_instructions = visible_instructions(first_view)
        second_instructions = visible_instructions(second_view)
        self.assertEqual(first_instructions, second_instructions)
        self.assertTrue(torch.equal(semantic_relations(first_view, first_instructions),
                                    semantic_relations(second_view, second_instructions)))

    def test_two_stages_and_node_readout_receive_gradients(self):
        compiled = program([(0, 1), (0, 1, 2), (0, 2)], 4)
        view = structure_input(compiled, chunk_hir(compiled, 15)[0], 16)
        model = HIRCodeEncoder(config())
        encoded = model(view)
        self.assertEqual(encoded.nodes.shape, (4, 16))
        self.assertEqual(encoded.edges.shape, (3, 16))
        (encoded.nodes.square().sum() + encoded.edges.square().sum()).backward()
        self.assertGreater(model.local_layers[0].qkv.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.semantic_layers[0].qkv.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.node_fusion[0].weight.grad.abs().sum().item(), 0.)

    def test_cli_uses_v1_output_and_two_plus_two_layers(self):
        args = parse_args(['--device=cpu', '--amp=off', '--gpu_target_memory_gb=1',
                           '--gpu_memory_budget_gb=2'])
        self.assertEqual(args.model_label, 'HIR-CodeBERT')
        self.assertEqual(args.output.name, 'hircodebert_v1')
        self.assertEqual((args.hir_hidden_size, args.hir_num_attention_heads), (256, 4))
        self.assertEqual((args.hir_local_num_hidden_layers, args.hir_semantic_num_hidden_layers), (2, 2))


if __name__ == '__main__':
    unittest.main()
