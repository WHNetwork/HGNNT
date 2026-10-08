import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dhgbench'))
from lib_models.HNN.cene import CENE


def args(mode='CENE', seed=0):
    return SimpleNamespace(cene_mode=mode, cene_shuffle_seed=seed,
                           MLP_hidden=4, decoder_hidden=5, dropout=0.0,
                           task_type='node_cls', method='CENE', embedding_mode=False)


def data(index=None, n=5):
    if index is None:
        index = [[0, 1, 1, 2, 2, 3, 4], [10, 10, 20, 20, 30, 30, 40]]
    return SimpleNamespace(x=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10,
                           hyperedge_index=torch.tensor(index, dtype=torch.long),
                           num_features=3, num_classes=2)


class CENETest(unittest.TestCase):
    def test_forward_backward_and_registration(self):
        from lib_models import _semi_methods_
        from lib_utils.exp_agent import parse_model
        self.assertIn('CENE', _semi_methods_)
        sample = data()
        model = parse_model(args(), sample)
        logits, extra = model(sample)
        self.assertEqual(logits.shape, (5, 2))
        self.assertIsNone(extra)
        logits.square().sum().backward()
        self.assertIsNotNone(model.lin_in.weight.grad)
        self.assertIsNotNone(model.lambda_param.grad)
        self.assertTrue(torch.isfinite(logits).all())

    def test_lambda_zero_is_uniform(self):
        sample = data()
        weighted = CENE(3, 2, args('CENE'))
        uniform = CENE(3, 2, args('ENE-Excl'))
        uniform.load_state_dict(weighted.state_dict())
        self.assertTrue(torch.allclose(weighted(sample)[0], uniform(sample)[0], atol=1e-6))
        weighted.lambda_param.data.fill_(-1)
        self.assertTrue(torch.isfinite(weighted(sample)[0]).all())

    def test_exclusion_and_degree_one(self):
        node = torch.tensor([0, 1, 1, 2, 3, 4])
        edge = torch.tensor([0, 0, 1, 1, 2, 2])
        degree = torch.tensor([1, 2, 1, 1, 1])
        z = torch.tensor([[2.0], [6.0], [10.0]])
        score = torch.zeros(5)
        excl = CENE(3, 2, args('ENE-Excl'))
        incl = CENE(3, 2, args('ENE'))
        self.assertTrue(torch.allclose(excl.edge_messages(z, node, edge, degree, score, 5).flatten(),
                                        torch.tensor([6.0, 2.0, 0.0])))
        self.assertTrue(torch.allclose(incl.edge_messages(z, node, edge, degree, score, 5).flatten(),
                                        torch.tensor([3.0, 5.0, 10.0])))

    def test_centrality_changes_messages_and_has_gradient(self):
        # e0={0,1}, e1={0,2}, e2={1,3}, e3={1,4}.
        # In e0, shared nodes 0 and 1 have degrees 2 and 3,
        # and exclusion messages 2 and (6+10)/2=8 respectively.
        sample = data([[0, 1, 0, 2, 1, 3, 1, 4],
                       [0, 0, 1, 1, 2, 2, 3, 3]])
        model = CENE(3, 2, args('CENE'))
        node, edge, degree, score, _ = model._get_structure(sample)
        self.assertEqual(degree[:2].tolist(), [2, 3])
        z = torch.tensor([[0.0], [2.0], [6.0], [10.0]])
        uniform = CENE(3, 2, args('ENE-Excl')).edge_messages(
            z, node, edge, degree, score, 5)
        zero = model.edge_messages(z, node, edge, degree, score, 5)
        self.assertTrue(torch.allclose(zero, uniform))
        self.assertAlmostEqual(zero[0, 0].item(), 5.0)
        zero[0, 0].backward()
        self.assertIsNotNone(model.lambda_param.grad)
        self.assertTrue(torch.isfinite(model.lambda_param.grad))
        self.assertGreater(model.lambda_param.grad.item(), 0)
        with torch.no_grad():
            model.lambda_param.fill_(1.0)
        changed = model.edge_messages(z, node, edge, degree, score, 5)
        self.assertGreater(changed[0, 0].item(), zero[0, 0].item())

    def test_singletons_empty_isolated_and_noncontiguous_ids(self):
        sample = data()
        original = sample.hyperedge_index.clone()
        model = CENE(3, 2, args())
        model.eval()
        node, edge, degree, _, count = model._get_structure(sample)
        self.assertEqual(count, 3)
        self.assertEqual(degree.tolist(), [1, 2, 2, 1, 0])
        self.assertEqual(edge.unique().tolist(), [0, 1, 2])
        model(sample)
        self.assertTrue(torch.equal(sample.hyperedge_index, original))
        self.assertTrue(torch.isfinite(model(sample)[0]).all())
        empty = data([[], []])
        singleton = data([[0, 2], [7, 99]])
        self.assertTrue(torch.isfinite(model(empty)[0]).all())
        self.assertTrue(torch.isfinite(model(singleton)[0]).all())
        self.assertEqual(model._get_structure(singleton)[-1], 0)
        regular = data([[0, 1, 2, 3], [5, 5, 5, 5]], n=4)
        self.assertTrue(torch.isfinite(model(regular)[0]).all())
        self.assertTrue(torch.equal(model._get_structure(regular)[3], torch.zeros(4)))
        duplicate = data([[0, 0, 1], [5, 5, 5]])
        self.assertEqual(model._get_structure(duplicate)[2].tolist(), [1, 1, 0, 0, 0])

    def test_node_and_edge_renumbering_equivariance(self):
        sample = data()
        model = CENE(3, 2, args())
        model.eval()
        expected = model(sample)[0]
        permutation = torch.tensor([2, 4, 0, 3, 1])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(5)
        changed = SimpleNamespace(x=sample.x[permutation],
                                  hyperedge_index=torch.stack((inverse[sample.hyperedge_index[0]],
                                                               100 - sample.hyperedge_index[1])),
                                  num_features=3, num_classes=2)
        self.assertTrue(torch.allclose(model(changed)[0], expected[permutation], atol=1e-6))

    def test_ablations_shuffle_and_reset(self):
        sample = data()
        for mode in CENE.MODES:
            model = CENE(3, 2, args(mode, 7))
            self.assertEqual(model.mode, mode)
            self.assertTrue(torch.isfinite(model(sample)[0]).all())
            model.lambda_param.data.fill_(3)
            model.reset_parameters()
            self.assertEqual(model.lambda_param.item(), 0)
        shuffled = CENE(3, 2, args('CENE-Shuffle', 7))
        first = shuffled._get_structure(sample)[3].clone()
        self.assertTrue(torch.equal(first, shuffled._get_structure(sample)[3]))
        self.assertTrue(torch.equal(first, CENE(3, 2, args('CENE-Shuffle', 7))._get_structure(sample)[3]))


if __name__ == '__main__':
    unittest.main()
