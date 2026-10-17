"""Contract tests for token mapping, visible attention, masking and transfer.

These are deterministic CPU unit tests, not an epoch-training/smoke runner.
"""

import json
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dhgbench'))

from hir import (DIRECT, EDGE, EDGE_ID, EXTEND, FRAGMENT, NODE, ORDER, PATCH, REF,
                 chunk_hir, compile_hypergraph, parse_hir)
from hirgraphbert_cli import parse_args
from lib_models.hirgraphbert import (GLOBAL, MASK_INSTRUCTION, AttentionPooling, HIRGraphBERT,
                                     HIRGraphEncoder, HIRGRAPHBERT_DEFAULTS, decode_chunk, graph_relations,
                                     load_encoder, structure_input)
from lib_models.hirgraphbert_pretrain import HIRGraphPretrainer
from lib_utils.hirbert_microbatch import MicroBatchPolicy
from lib_utils.hirbert_pretrain_utils import UnlabeledGraph, cpu_state, rng_state, restore_rng
from lib_utils.hirgraphbert_data import StructureSampler, load_graph, masked_hir_input, structure_graph
from lib_utils.hirgraphbert_training import _node_operation, _transaction


def program(edges=None, n=32):
    edges = edges if edges is not None else [range(4), range(5), range(8), range(1, 9)]
    pairs = [(node, edge) for edge, members in enumerate(edges) for node in members]
    index = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()
    return compile_hypergraph(n, index, len(edges))


def config(**kwargs):
    return {**HIRGRAPHBERT_DEFAULTS, 'hidden_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 4,
            'intermediate_size': 32, 'dropout': 0., 'max_seq_length': 16, 'chunk_batch_size': 2, **kwargs}


def model_args(**kwargs):
    return SimpleNamespace(**{'hir_' + key: value for key, value in config(**kwargs).items()})


class FeatureForbidden:
    def __init__(self, p):
        self.num_nodes = p.num_nodes
        self.num_hyperedges = len(p.hyperedges)
        self.hyperedge_index = p.to_hyperedge_index()

    @property
    def x(self):
        raise AssertionError('Raw features must never be read')


class HIRGraphBERTContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_token_ranges_reference_payloads_and_unchanged_roundtrip(self):
        p = program(n=12)
        self.assertEqual(parse_hir(p.tokens, 12, 4).hyperedges, p.hyperedges)
        self.assertEqual([ir.mode for ir in p.instructions], [DIRECT, EXTEND, EXTEND, PATCH])
        chunks = chunk_hir(p, 128)[0]
        decoded = decode_chunk(p, chunks[0])
        self.assertIn(NODE, decoded.types)
        self.assertIn(EDGE_ID, decoded.types)
        self.assertIn(ORDER, decoded.types)
        self.assertIn(REF, decoded.types)
        for token, kind, node, payload in zip(chunks[0].tokens, decoded.types, decoded.nodes, decoded.payload_edges):
            if kind == NODE:
                self.assertEqual(token, p.space.node(node))
            elif kind == EDGE_ID:
                self.assertEqual(token, p.space.edge(payload))
            elif kind == ORDER:
                self.assertGreaterEqual(token, p.space.order(0))
        self.assertEqual(decoded.references[decoded.owners.index(3)], 2)
        # Decoding cannot use the program's recovered memberships as an oracle.
        forged = replace(p, hyperedges=((),) * len(p.hyperedges))
        self.assertEqual(decoded, decode_chunk(forged, chunks[0]))

    def test_attention_uses_only_visible_relations_and_global(self):
        types = torch.tensor([[GLOBAL, EDGE, NODE, EDGE, NODE, EDGE, NODE, EDGE, NODE]])
        nodes = torch.tensor([[-1, -1, 0, -1, 0, -1, 2, -1, 3]])
        owners = torch.tensor([[-1, 0, 0, 1, 1, 2, 2, 3, 3]])
        refs = torch.tensor([[-1, -1, -1, -1, -1, 1, 1, -1, -1]])
        channels, mask = graph_relations(types, nodes, owners, refs, torch.ones_like(types, dtype=torch.bool))
        self.assertTrue(channels[0, 0, 1, 2])  # same edge
        self.assertTrue(channels[0, 1, 2, 4])  # same node
        self.assertTrue(channels[0, 2, 3, 6])  # REF dependency
        self.assertTrue(mask[0, 2, 0])
        self.assertFalse(mask[0, 2, 8])
        # Removing a visible identity removes its equality relation immediately.
        nodes[0, 4] = -1
        changed, _ = graph_relations(types, nodes, owners, refs, torch.ones_like(types, dtype=torch.bool))
        self.assertFalse(changed[0, 1, 2, 4])

    def test_no_features_node_outputs_isolates_and_no_global_id_parameters(self):
        p = program(n=12)
        data = FeatureForbidden(p)
        model = HIRGraphBERT(3, model_args()).eval()
        logits, aux = model(data)
        self.assertIsNone(aux)
        self.assertEqual(logits.shape, (12, 3))
        self.assertTrue(torch.isfinite(logits).all())
        nodes = model.encode_nodes(data)
        self.assertFalse(torch.allclose(nodes[10], nodes[11]))
        self.assertGreater(model.last_node_variance, 0)
        self.assertFalse(any('feature' in key for key in model.state_dict()))
        before = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        model(FeatureForbidden(program([range(12), (39,)], n=40)))
        self.assertEqual(before, {key: tuple(value.shape) for key, value in model.state_dict().items()})
        stripped = structure_graph(data, 'no_features')
        self.assertEqual(stripped.x.shape, (12, 0))

    def test_shared_attention_pooling_matches_manual_and_propagates_gradients(self):
        pool = AttentionPooling(4)
        values = torch.randn(7, 4, requires_grad=True)
        ids = torch.tensor([0, 1, 0, 2, 1, 0, 2])
        actual = pool(values, ids, 3)
        expected = torch.stack([(pool.score(values[ids == node]).softmax(0) * values[ids == node]).sum(0)
                                for node in range(3)])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        actual.square().sum().backward()
        self.assertTrue((values.grad.abs().sum(-1) > 0).all())
        self.assertGreater(pool.score.weight.grad.abs().sum().item(), 0)

    def test_fragments_cross_chunk_nodes_and_independent_packet_batching(self):
        p = program([range(22), range(1, 23), (23,)], n=25)
        chunks, stats = chunk_hir(p, 10)
        self.assertGreater(stats['long_edge_fragments'], 0)
        self.assertTrue(any(FRAGMENT in chunk.tokens for chunk in chunks))
        view = structure_input(p, chunks, 16)
        encoder = HIRGraphEncoder(config()).eval()
        all_nodes = {node for sequence in view.sequences for node in sequence.nodes if node >= 0}
        self.assertEqual(all_nodes, set(range(25)))
        first, _ = encoder.encode_many([view], 256)
        self.assertTrue((first[0].node_occurrences > 0).all())
        self.assertGreater(first[0].node_occurrences[1].item(), 1)
        encoder.chunk_batch_size = 8
        joint, calls = encoder.encode_many([view, view], 256)
        self.assertTrue(torch.allclose(first[0].nodes, joint[0].nodes, atol=1e-6))
        self.assertTrue(torch.allclose(joint[0].nodes, joint[1].nodes, atol=1e-6))
        self.assertTrue(all(call['padded_tokens'] <= 256 and call['max_sequence_length'] <= 16 for call in calls))

    def test_anchored_instruction_masks_hidden_fields_and_copies(self):
        first = program([range(4), range(5), range(8), range(11)], n=12)
        second = program([range(4), range(5), range(8), range(1, 11)], n=12)
        codes, edges = tuple(range(1, 13)), (4, 2, 1, 3)
        view1, targets1 = masked_hir_input(first, [3], 16, codes, edges, random.Random(7), anchors={3: 1})
        view2, targets2 = masked_hir_input(second, [3], 16, codes, edges, random.Random(7), anchors={3: 1})
        self.assertNotEqual(targets1.modes, targets2.modes)
        self.assertNotEqual(targets1.orders, targets2.orders)
        # Different true mode/order/payloads produce EXACTLY the same encoder input.
        self.assertEqual(view1, view2)
        masked = [sequence for sequence in view1.sequences if MASK_INSTRUCTION in sequence.types]
        self.assertEqual(len(masked), 1)
        self.assertEqual(masked[0].types, (MASK_INSTRUCTION, NODE))
        self.assertEqual(masked[0].nodes, (-1, 1))
        self.assertEqual(masked[0].orders, (-1, -1))
        self.assertEqual(masked[0].references, (-1, -1))
        self.assertTrue(all(node != 1 for _, node, _ in targets1.payloads))
        duplicate = program([range(4), range(4), range(8), range(1, 9)], n=12)
        visible, labels = masked_hir_input(duplicate, [0], 16, codes, edges, random.Random(7))
        self.assertEqual(labels.hidden_edges, (0, 1))
        for sequence in visible.sequences:
            if MASK_INSTRUCTION not in sequence.types:
                self.assertFalse(set(sequence.owners) & {0, 1})
                self.assertFalse(set(sequence.references) & {0, 1})

    def test_checkpointing_outputs_and_encoder_gradients(self):
        p = program(n=12)
        view = structure_input(p, chunk_hir(p, 15)[0], 16)
        checked = HIRGraphEncoder(config(gradient_checkpointing=True))
        plain = HIRGraphEncoder(config(gradient_checkpointing=False))
        plain.load_state_dict(checked.state_dict())
        results = []
        for encoder in (checked, plain):
            output = encoder(view).nodes
            results.append(output.detach())
            (output[:, 0].square().sum()).backward()
        self.assertTrue(torch.allclose(*results, atol=1e-6))
        for (name, a), (_, b) in zip(checked.named_parameters(), plain.named_parameters()):
            if a.grad is not None or b.grad is not None:
                self.assertIsNotNone(a.grad, name)
                self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-5), name)
        for name in ('layers.0.attention.relation_bias', 'layers.0.attention.qkv.weight', 'tokens.weight',
                     'node_identity.weight', 'order_projection.weight', 'node_pool.score.weight'):
            gradient = dict(checked.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertEqual(HIRGraphEncoder(dict(HIRGRAPHBERT_DEFAULTS)).parameter_counts()['structural_attention'], 64)

    def sample_batch(self):
        args = parse_args(['--device=cpu', '--amp=off', '--local_min_edges=4', '--local_max_edges=9',
                           '--hir_max_seq_length=16',
                           '--membership_mask_ratio=0.5', '--hir_mask_ratio=1', '--hir_hidden_size=16',
                           '--hir_num_hidden_layers=2', '--hir_num_attention_heads=4', '--hir_intermediate_size=32',
                           '--hir_dropout=0', '--hir_chunk_batch_size=2'])
        p = program([range(4), range(5), range(8), range(1, 9), (20, 21), (21, 22), (23, 24), range(18, 23), range(4)])
        sampler = StructureSampler(UnlabeledGraph(torch.randn(32, 9), p, 'unit'), args)
        batches = []
        for index in range(5):
            packets = list(sampler.sample('train', 1, index))
            main = next(packet for packet in packets if packet.main is not None)
            relation = next((packet for packet in packets if packet.relation is not None), None)
            hir = next((packet for packet in packets if packet.hir is not None), None)
            if (relation is not None and hir is not None and len(main.membership_labels)
                    and any(mode >= 0 for mode in hir.hir_targets.modes)):
                batches.append(replace(main, relation=relation.relation, hir=hir.hir,
                                       relation_pairs=relation.relation_pairs, relation_labels=relation.relation_labels,
                                       hir_targets=hir.hir_targets,
                                       task_scales={key: sum(packet.task_scales[key] for packet in packets)
                                                    for key in main.task_scales}))
        self.assertTrue(batches)
        return args, sampler, batches[0]

    def test_three_tasks_gradients_local_identity_and_hidden_membership(self):
        args, sampler, batch = self.sample_batch()
        self.assertEqual(sampler.graph.x.shape[1], 0)
        self.assertFalse(set(sampler.train_hir) & set(sampler.valid_hir))
        self.assertEqual(len(set(batch.main.node_codes)), batch.main.num_nodes)
        self.assertTrue(all(code >= 1 for code in batch.main.node_codes))
        self.assertEqual(batch.main.node_codes, batch.hir.node_codes)
        removals, _, _, _ = sampler.legacy.plan('train', 1)
        for (node, edge), label in zip(batch.membership_pairs.tolist(), batch.membership_labels.tolist()):
            source_node, source_edge = batch.node_ids[node], batch.source_edge_ids[edge]
            self.assertEqual(source_node in sampler.graph.program.hyperedges[source_edge], bool(label))
            if label:
                # A DROP NODE can still occur in syntax after recompilation;
                # it does not assert positive visible membership.
                self.assertIn(source_node, removals[source_edge])
                self.assertNotIn((source_node, source_edge), sampler.legacy.val_members)
            else:
                self.assertNotIn(source_node, sampler.legacy._negative_nodes(source_edge))
        model = HIRGraphPretrainer(config())
        loss, metrics = model(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for task in ('membership', 'relation', 'hir'):
            self.assertGreater(metrics[task + '_count'], 0)
            self.assertGreater(metrics[task + '_loss'], 0)
            self.assertTrue(0 <= metrics[task + '_acc'] <= 1)
        for name in ('membership_head.0.weight', 'relation_head.0.weight', 'mode_head.weight', 'order_head.weight',
                     'reference_query.weight', 'payload_head.0.weight', 'encoder.layers.0.attention.relation_bias'):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        # Validation corruption is fixed across epoch indices.
        a = list(sampler.sample('valid', 1, 0))
        b = list(sampler.sample('valid', 90, 0))
        for name in ('main', 'relation', 'hir'):
            self.assertEqual([getattr(batch, name) for batch in a], [getattr(batch, name) for batch in b])
        self.assertEqual([batch.hir_targets for batch in a], [batch.hir_targets for batch in b])

    def test_microbatch_gradient_equivalence_and_node_oom_replay(self):
        args, _, batch = self.sample_batch()
        # Production transaction helper, without starting any epoch runner.
        sampler = SimpleNamespace(sample=lambda *unused: iter([batch]))
        first = HIRGraphPretrainer(config())
        second = HIRGraphPretrainer(config())
        second.load_state_dict(first.state_dict())
        for model, size in ((first, 1), (second, 2)):
            policy = MicroBatchPolicy(args)
            policy.state.packets = size
            results, profile = _transaction(model, {'unit': sampler}, [('unit', 0), ('unit', 1)],
                                            args, policy, torch.cuda.amp.GradScaler(enabled=False), None, 'train', 1)
            self.assertEqual(len(results), 2)
            self.assertEqual(profile['micro_batches'], 2 if size == 1 else 1)
        for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
            if a.grad is not None or b.grad is not None:
                self.assertIsNotNone(a.grad, name)
                self.assertTrue(torch.allclose(a.grad, b.grad, atol=2e-5, rtol=2e-4), name)
        node_model = HIRGraphBERT(3, model_args())
        data = FeatureForbidden(program(n=12))
        data.y = torch.arange(12) % 3
        masks = {'train': torch.ones(12, dtype=torch.bool)}
        policy = MicroBatchPolicy(args)
        policy.state.packets, policy.state.chunk_batch_size = 1, 2
        optimizer = torch.optim.AdamW(node_model.parameters())
        original = node_model.forward
        attempts = []

        def once(*values):
            attempts.append(1)
            if len(attempts) == 1:
                raise torch.cuda.OutOfMemoryError('unit injected CUDA out of memory')
            return original(*values)

        with patch.object(node_model, 'forward', side_effect=once):
            result = _node_operation(node_model, data, masks, args, policy, None,
                                     optimizer, torch.cuda.amp.GradScaler(enabled=False))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(policy.state.chunk_batch_size, 1)
        self.assertGreater(result['train_loss'], 0)
        self.assertIsNotNone(node_model.classifier[0].weight.grad)

    def test_strict_shared_checkpoint_transfer_and_classification_roundtrip(self):
        source = HIRGraphPretrainer(config())
        model = HIRGraphBERT(3, model_args())
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'encoder.pt'
            torch.save({'kind': 'hirgraphbert_pretrained_encoder', 'input_mode': 'Structure-only',
                        'model_config': config(), 'shared_encoder_state_dict': cpu_state(source.encoder),
                        'seed': 7, 'best_epoch': 11}, checkpoint)
            report = load_encoder(model, checkpoint)
            self.assertFalse(report['missing_keys'])
            for name, value in source.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, model.encoder.state_dict()[name]))
            # Shared encoder transfer has no graph name/feature-dimension restrictions.
            data = FeatureForbidden(program(n=12))
            model.eval()
            logits = model(data)[0]
            final = Path(folder) / 'classifier.pt'
            torch.save({'kind': 'hirgraphbert_node_classifier', 'model_config': config(), 'num_classes': 3,
                        'model_state_dict': cpu_state(model), 'seed': 0}, final)
            loaded, _ = HIRGraphBERT.from_checkpoint(final)
            self.assertTrue(torch.equal(logits, loaded(data)[0]))
            model.train()
            F.cross_entropy(model(data)[0], torch.arange(12) % 3).backward()
            self.assertGreater(model.classifier[0].weight.grad.abs().sum().item(), 0)
            broken = torch.load(checkpoint, weights_only=True)
            broken['shared_encoder_state_dict'].pop('tokens.weight')
            torch.save(broken, checkpoint)
            with self.assertRaises(RuntimeError):
                load_encoder(model, checkpoint)
            with self.assertRaises(ValueError):
                load_encoder(HIRGraphBERT(3, model_args(max_seq_length=32)), checkpoint)

    def test_empty_graphs_and_explicit_empty_edges(self):
        for n, edges in ((0, []), (5, []), (5, [(), (), (0,)])):
            model = HIRGraphBERT(3, model_args()).eval()
            logits, _ = model(FeatureForbidden(program(edges, n)))
            self.assertEqual(logits.shape, (n, 3))
            self.assertTrue(torch.isfinite(logits).all())

    def test_microbatch_oom_policy_and_serializable_rng(self):
        args = parse_args(['--device=cpu'])
        policy = MicroBatchPolicy(args)
        for expected in (4, 2, 1):
            self.assertTrue(policy.reduce())
            self.assertEqual(policy.state.packets, expected)
        self.assertTrue(policy.reduce())
        self.assertEqual(policy.state.chunk_batch_size, 8)
        state = rng_state()
        expected = torch.randn(3)
        restore_rng(state)
        self.assertTrue(torch.equal(expected, torch.randn(3)))
        other = MicroBatchPolicy(args)
        other.load_state_dict(policy.state_dict())
        self.assertEqual(policy.state_dict(), other.state_dict())

    def test_cpu_loader_strips_features_and_pretraining_never_reads_labels(self):
        class Raw:
            x = SimpleNamespace(shape=(6, 999))
            num_hyperedges = 2
            edge_index = torch.tensor([[0, 1, 2, 6, 6, 7], [6, 6, 7, 0, 1, 2]])

            @property
            def y(self):
                raise AssertionError('SSL loader must not read labels')

        args = parse_args(['--device=cpu'])
        dataset = SimpleNamespace(data=Raw(), processed_paths=['unit-cache'])
        fake = SimpleNamespace(dataset_Hypergraph=lambda **unused: dataset)
        with patch.dict(sys.modules, {'lib_dataset.convert_datasets_to_pygDataset': fake}):
            graph = load_graph(args)
        self.assertEqual(graph.x.shape, (6, 0))
        self.assertEqual(set(graph.program.hyperedges), {(0, 1), (2,)})

    def test_local_identity_consistency_across_fragment_packets(self):
        args = parse_args(['--device=cpu', '--amp=off', '--local_min_edges=2', '--local_max_edges=4',
                           '--hir_max_seq_length=16',
                           '--membership_mask_ratio=0.5'])
        p = program([range(90), range(1, 91), range(92), (92, 93), (93, 94), (95, 96)], n=100)
        sampler = StructureSampler(UnlabeledGraph(torch.empty(100, 0), p, 'fragments'), args)
        batches = list(sampler.sample('train', 1, 0))
        self.assertGreater(len(batches), 1)
        nodes, edges, repeated = {}, {}, 0
        for batch in batches:
            view = next(getattr(batch, name) for name in ('main', 'relation', 'hir')
                        if getattr(batch, name) is not None)
            for original, code in zip(batch.node_ids, view.node_codes):
                if original in nodes:
                    self.assertEqual(code, nodes[original])
                    repeated += 1
                nodes[original] = code
            for original, code in zip(batch.source_edge_ids, view.edge_codes):
                if original in edges:
                    self.assertEqual(code, edges[original])
                edges[original] = code
        self.assertGreater(repeated, 0)

    def test_main_method_defaults_preserve_old_model_configuration(self):
        from parameter_parser import method_config, set_task_args
        from lib_models import _semi_methods_
        self.assertIn('HIRGraphBERT', _semi_methods_)
        old = method_config(SimpleNamespace(method='HIRBERT', is_default=False, dname='cora'))
        new = method_config(SimpleNamespace(method='HIRGraphBERT', is_default=False, dname='citeseer',
                                            task_type='node_cls', use_bench_prop=True))
        new = set_task_args(new)
        self.assertEqual(old.hir_max_seq_length, 256)
        self.assertEqual((new.hir_num_hidden_layers, new.hir_hidden_size, new.hir_num_attention_heads), (4, 256, 4))
        self.assertEqual(new.hir_max_seq_length, 512)
        self.assertFalse(new.add_self_loop)

    def test_report_pairs_new_seeds_and_keeps_historical_results_separate(self):
        from benchmark_hirgraphbert import write_reports
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, previous = root / 'new', root / 'previous'
            output.mkdir()
            previous.mkdir()
            historical = previous / 'metrics.json'
            historical.write_text(json.dumps({'dataset': 'cora', 'variant': 'feature_only', 'seed': 0, 'test_acc': 66.}),
                                  encoding='utf-8')
            original = historical.read_bytes()
            for variant, accuracies in (('scratch', [70., 74.]), ('pretrained', [72., 78.])):
                for seed, accuracy in enumerate(accuracies):
                    path = output / 'cora' / variant / f'seed_{seed}'
                    path.mkdir(parents=True)
                    value = {'completed': True, 'model': 'HIR-GraphBERT', 'input_mode': 'Structure-only',
                             'dataset': 'cora', 'variant': variant, 'seed': seed, 'test_acc': accuracy, 'val_acc': accuracy,
                             'node_repr_variance': .2, 'parameter_count': 123,
                             'parameter_counts': {'structural_attention': 64},
                             'peak_gpu_memory_mb': 0., 'peak_gpu_reserved_memory_mb': 0.,
                             'config': {'variant': variant, 'seed': seed}}
                    (path / 'metrics.json').write_text(json.dumps(value), encoding='utf-8')
            write_reports(SimpleNamespace(output=output, reference_results=[previous]))
            result = json.loads((output / 'experiment_report.json').read_text(encoding='utf-8'))
            self.assertEqual(len(result['runs']), 4)
            self.assertEqual(len(result['historical_references']), 1)
            self.assertEqual(result['paired_comparisons'][0]['pretrained_minus_scratch_pp_mean'], 3.)
            scratch = next(row for row in result['summary'] if row['variant'] == 'scratch')
            self.assertEqual(scratch['test_acc_mean'], 72.)
            self.assertEqual(scratch['test_acc_std'], 2.)
            self.assertEqual(historical.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
