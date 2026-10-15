import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dhgbench'))
from hir import compile_hypergraph
from lib_models.HNN.hirbert import HIRBERT


def make_args(**updates):
    values = dict(
        method='HIRBERT', task_type='node_cls', embedding_mode=False,
        hir_hidden_size=16, hir_num_hidden_layers=1, hir_num_attention_heads=4,
        hir_intermediate_size=32, hir_dropout=0.1, hir_max_seq_length=16,
        hir_chunk_batch_size=2, hir_gradient_checkpointing=True,
        device='cpu', epochs=1, lr=0.001, wd=0.0, clip_grad=False,
        display_step=1, num_seeds=1, seed_offset=0, is_perturbed=False,
        train_prop=0.5, valid_prop=0.25, eval_verbose=False,
        save_results=True, dname='synthetic_hir', is_default=False,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def make_data(edges=None, n=24):
    if edges is None:
        edges = [(0,), (0,), range(4), range(5), range(8), range(1, 9), range(22)]
    pairs = [(v, i) for i, edge in enumerate(edges) for v in edge]
    return SimpleNamespace(
        x=torch.randn(n, 5, generator=torch.Generator().manual_seed(81)),
        y=torch.arange(n) % 3,
        hyperedge_index=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        num_nodes=n, num_hyperedges=len(edges), num_features=5, num_classes=3,
        name='synthetic_hir', device='cpu',
    )


class HIRBERTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()

    def tearDown(self):
        self.stdout.__exit__(None, None, None)

    def test_bert_forward_logits_and_all_parameter_gradients(self):
        from transformers import BertModel
        from lib_models import _semi_methods_
        from lib_utils.exp_agent import parse_model
        self.assertIn('HIRBERT', _semi_methods_)
        data = make_data()
        model = parse_model(make_args(), data)
        self.assertIsInstance(model.encoder.bert, BertModel)
        self.assertEqual(model.encoder.bert.config.hidden_size, 16)
        original = data.hyperedge_index.clone()
        logits, auxiliary = model(data)
        self.assertEqual(logits.shape, (24, 3))
        self.assertIsNone(auxiliary)
        F.cross_entropy(logits[:12], data.y[:12]).backward()
        for name, param in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
        for param in (model.encoder.feature_projection.weight,
                      model.encoder.order_projection.weight,
                      model.encoder.bert.encoder.layer[0].attention.self.query.weight,
                      model.classifier[0].weight):
            self.assertGreater(param.grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(data.hyperedge_index, original))

    def test_cpu_step_and_features_reach_bert_and_isolated_nodes(self):
        data = make_data()
        model = HIRBERT(5, 3, make_args(hir_dropout=0.0))
        model.prepare(data)
        data.x.requires_grad_()
        features = model.encoder.feature_projection(data.x)
        edge_states = model.encoder(features, model.program, model.chunks)
        edge_states[:, 0].sum().backward()
        self.assertGreater(data.x.grad[:22].abs().sum().item(), 0)
        self.assertTrue(torch.equal(data.x.grad[22:], torch.zeros_like(data.x.grad[22:])))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        before = model.classifier[-1].weight.detach().clone()
        logits, _ = model(data)
        F.cross_entropy(logits, data.y).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier[-1].weight))
        model.eval()
        with torch.no_grad():
            original = model(data)[0]
            data.x = data.x.detach().clone()
            data.x[23] += 10
            changed = model(data)[0]
        self.assertFalse(torch.allclose(original[23], changed[23]))

    def test_empty_hypergraph_singletons_and_declared_empty_edges(self):
        for edges in ([], [(0,), (1,), (1,)], [(), (0,), ()]):
            with self.subTest(edges=edges):
                data = make_data(edges=edges, n=6)
                model = HIRBERT(5, 3, make_args())
                logits, _ = model(data)
                self.assertEqual(logits.shape, (6, 3))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertEqual(len(model.program.hyperedges), len(edges))
                F.cross_entropy(logits, data.y).backward()
        data = make_data(edges=[], n=0)
        self.assertEqual(HIRBERT(5, 3, make_args())(data)[0].shape, (0, 3))

    def test_bounded_bert_inputs_and_all_long_edge_members(self):
        data = make_data(edges=[range(22), range(1, 23), (23,)], n=24)
        model = HIRBERT(5, 3, make_args(hir_max_seq_length=9, hir_chunk_batch_size=2))
        shapes = []

        def record_shape(module, args, kwargs):
            shapes.append(kwargs['inputs_embeds'].shape)

        handle = model.encoder.bert.register_forward_pre_hook(record_shape, with_kwargs=True)
        logits, _ = model(data)
        F.cross_entropy(logits, data.y).backward()
        handle.remove()
        self.assertTrue(shapes)
        self.assertTrue(all(s[0] <= 2 and s[1] <= 9 for s in shapes))
        self.assertGreater(model.compilation_stats['long_edge_fragments'], 0)
        self.assertEqual(len(model.program.hyperedges), 3)
        self.assertEqual({e for c in model.chunks for _, e, _ in c.readouts}, {0, 1, 2})

    def test_checkpointing_preserves_outputs_and_gradients(self):
        data = make_data()
        torch.manual_seed(12)
        checked = HIRBERT(5, 3, make_args(hir_dropout=0.0))
        plain = HIRBERT(5, 3, make_args(hir_dropout=0.0, hir_gradient_checkpointing=False))
        plain.load_state_dict(checked.state_dict())
        outputs = []
        for model in (checked, plain):
            output = model(data)[0]
            outputs.append(output.detach())
            F.cross_entropy(output, data.y).backward()
        self.assertTrue(torch.allclose(*outputs, atol=1e-6))
        for (name, first), (_, second) in zip(checked.named_parameters(), plain.named_parameters()):
            with self.subTest(parameter=name):
                self.assertTrue(torch.allclose(first.grad, second.grad, atol=1e-6, rtol=1e-4))

    def test_seed_reproducibility_and_cache_contains_only_structure(self):
        data = make_data()

        def step(seed):
            torch.manual_seed(seed)
            model = HIRBERT(5, 3, make_args())
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            F.cross_entropy(model(data)[0], data.y).backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                return model, model(data)[0]

        first, expected = step(13)
        again, actual = step(13)
        other, changed = step(14)
        self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(torch.equal(expected, changed))
        self.assertEqual(first.program.tokens, other.program.tokens)
        old_program = first.program
        first(data)
        self.assertIs(first.program, old_program)
        # Feature updates must be visible despite structural caching.
        data.x = data.x + 1
        self.assertFalse(torch.allclose(first(data)[0], expected))
        data.hyperedge_index[0, 0] = 23
        first(data)
        self.assertIsNot(first.program, old_program)

    def test_hirbert_config_cli_and_task_scope(self):
        from parameter_parser import method_config, parameter_parser, set_task_args
        command = ['main.py', '--method=HIRBERT', '--task_type=node_cls', '--dname=cora',
                   '--epochs=1', '--hir_hidden_size=32', '--hir_dropout=0.0', '--device=cpu']
        with patch.object(sys, 'argv', command):
            args = set_task_args(method_config(parameter_parser()))
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.hir_hidden_size, 32)
        self.assertEqual(args.hir_dropout, 0)
        self.assertEqual(args.hir_num_hidden_layers, 4)
        self.assertFalse(args.add_self_loop)
        args.task_type = 'edge_pred'
        with self.assertRaises(ValueError):
            set_task_args(args)

    def test_only_hirbert_skips_automatic_self_loops(self):
        from torch_geometric.data import Data
        from lib_dataset.preprocessing import data_processing
        # Dataset bipartite representation: real singleton {0}, real edge {1,2}.
        index = torch.tensor([[0, 1, 2, 3, 4, 4], [3, 4, 4, 0, 1, 2]])
        base = Data(x=torch.randn(3, 5), y=torch.tensor([0, 1, 0]), edge_index=index,
                    n_x=3, num_hyperedges=2)
        for method, expected_count in [('HIRBERT', 2), ('EDHNN', 4)]:
            args = make_args(method=method, add_self_loop=True)
            result = data_processing(args, SimpleNamespace(data=base, sens=None))
            program = compile_hypergraph(3, result.hyperedge_index)
            self.assertEqual(len(program.hyperedges), expected_count)
            self.assertIn((0,), program.hyperedges)
            self.assertIn((1, 2), program.hyperedges)
            self.assertTrue(torch.equal(result.x, base.x))
            self.assertTrue(torch.equal(result.y, base.y))
        self.assertTrue(torch.equal(base.edge_index, index))

    def test_node_cls_two_seeds_artifacts_and_checkpoint_reload(self):
        from lib_dataset.data_base import HyperDataset
        from lib_utils.exp_agent import ExpAgent
        data = make_data(edges=[(), (0,), (0,), range(4), range(5), range(8), range(1, 9), range(22)])
        data.generate_random_split = lambda **kwargs: HyperDataset.generate_random_split(data, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            args = make_args(results_dir=directory, num_seeds=2, seed_offset=3)
            agent = ExpAgent(args)
            agent.node_cls_train_eval(data)
            root = Path(directory)
            with (root / 'node_cls_runs.csv').open(newline='', encoding='utf-8') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row['seed']) for row in rows], [3, 4])
            self.assertEqual({row['method'] for row in rows}, {'HIRBERT'})
            self.assertTrue((root / 'node_cls_summary.csv').exists())
            checkpoints = sorted(root.glob('hirbert/*/*/seed_*/checkpoint.pt'))
            self.assertEqual(len(checkpoints), 2)
            saved_parameters = []
            for path in checkpoints:
                loaded, graph = HIRBERT.from_checkpoint(path)
                with torch.no_grad():
                    expected = loaded(data)[0]
                    actual = loaded(graph)[0]
                self.assertTrue(torch.equal(expected, actual))
                saved = torch.load(path, weights_only=True)
                # Compare the reloaded model to the final recorded split metrics.
                for i, split in enumerate(('train', 'valid', 'test')):
                    mask = saved['masks'][split]
                    acc = float((actual[mask].argmax(-1) == graph.y[mask]).float().mean())
                    self.assertAlmostEqual(100 * acc, saved['metrics']['acc'][i], delta=1e-5)
                saved_parameters.append(saved['model_state_dict']['classifier.3.weight'])
                with (path.parent / 'training.csv').open(newline='', encoding='utf-8') as file:
                    history = list(csv.DictReader(file))
                self.assertEqual(len(history), 1)
                self.assertTrue(float(history[0]['train_loss']) > 0)
                self.assertTrue(float(history[0]['val_loss']) > 0)
                for i, split in enumerate(('train', 'val', 'test')):
                    self.assertAlmostEqual(float(history[0][split + '_acc']),
                                           saved['metrics']['acc'][i], delta=1e-5)
                stats = json.loads((path.parent / 'compilation.json').read_text())
                self.assertEqual(stats['num_hyperedges'], data.num_hyperedges)
                self.assertLessEqual(stats['max_chunk_length'], 16)
                metrics = json.loads((path.parent / 'metrics.json').read_text())
                self.assertGreater(metrics['parameter_count'], 0)
                self.assertGreater(metrics['train_time'], 0)
            self.assertFalse(torch.equal(*saved_parameters))


if __name__ == '__main__':
    unittest.main()
