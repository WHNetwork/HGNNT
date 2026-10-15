import random
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dhgbench'))
from hir import (ADD, DIRECT, DROP, EDGE, END_EDGE, EXTEND, FRAGMENT, PATCH,
                 REF, EdgeIR, TokenSpace, chunk_hir, compile_hypergraph, parse_hir)


def incidence(edges, ids=None):
    if ids is None:
        ids = range(len(edges))
    pairs = [(node, edge_id) for edge_id, edge in zip(ids, edges) for node in edge]
    return [[p[0] for p in pairs], [p[1] for p in pairs]]


class HIRTest(unittest.TestCase):
    def round_trip(self, n, edges, explicit=False):
        program = compile_hypergraph(n, incidence(edges), len(edges) if explicit else None)
        parsed = parse_hir(program.tokens, n, len(program.hyperedges))
        expected = Counter(tuple(sorted(set(e))) for e in edges if e or explicit)
        self.assertEqual(Counter(parsed.hyperedges), expected)
        self.assertEqual(parsed.hyperedges, program.hyperedges)
        self.assertEqual(parsed.instructions, program.instructions)
        return program

    def test_direct_extend_patch_and_multilevel_dependencies(self):
        program = self.round_trip(10, [range(4), range(5), range(8), range(1, 9)])
        self.assertEqual([ir.mode for ir in program.instructions], [DIRECT, EXTEND, EXTEND, PATCH])
        self.assertEqual([ir.ref for ir in program.instructions], [-1, 0, 1, 2])
        self.assertEqual(program.instructions[-1].drop, (0,))
        self.assertEqual(program.instructions[-1].add, (8,))
        self.assertEqual(program.stats['reference_ratio'], 0.75)

    def test_same_order_overlap_disjoint_edges_and_cost_ties(self):
        program = self.round_trip(30, [(0, 1), (1, 2), (8, 9), (9, 10, 11), range(15, 19)])
        self.assertTrue(all(ir.mode == DIRECT for ir in program.instructions))
        tied = self.round_trip(10, [range(3), range(4)])
        self.assertEqual(tied.instructions[-1].mode, DIRECT)
        tied_patch = self.round_trip(10, [range(6), range(1, 7)])
        self.assertEqual(tied_patch.instructions[-1].mode, DIRECT)
        # Two equally good lower-order subsets: smallest canonical ID wins.
        tie_ref = self.round_trip(7, [range(4), (0, 1, 2, 4), range(5)])
        self.assertEqual(tie_ref.instructions[-1].ref, 0)

    def test_duplicate_incidence_and_edge_multiplicity(self):
        edges = [(0, 0, 1), (0, 1), tuple(range(8)), tuple(range(8))]
        program = self.round_trip(10, edges)
        self.assertEqual(len(program.hyperedges), 4)
        self.assertEqual(program.stats['duplicate_incidences_removed'], 1)
        self.assertEqual(program.stats['duplicate_hyperedges_preserved'], 2)
        self.assertEqual(program.instructions[-1].mode, PATCH)

    def test_empty_singletons_isolates_and_explicit_empty_edges(self):
        program = self.round_trip(7, [])
        self.assertEqual(program.tokens, ())
        chunks, stats = chunk_hir(program, 16)
        self.assertEqual(chunks, ())
        self.assertEqual(stats['max_chunk_length'], 0)
        self.round_trip(0, [])
        self.round_trip(7, [(0,), (3,), (3,), (1, 2)])
        self.round_trip(7, [(), (0,), (), (1, 2)], explicit=True)
        self.round_trip(0, [(), ()], explicit=True)

    def test_incidence_matrix_keeps_isolates_and_duplicate_columns(self):
        import torch
        p = self.round_trip(8, [(0,), (0, 1), (0, 1), ()], explicit=True)
        parsed = parse_hir(p.tokens, 8, 4)
        matrix = parsed.incidence_matrix().to_dense()
        self.assertEqual(matrix.shape, (8, 4))
        self.assertTrue(torch.equal(matrix[2:], torch.zeros(6, 4)))
        recompiled = compile_hypergraph(8, parsed.to_hyperedge_index(), 4)
        self.assertEqual(recompiled.tokens, p.tokens)

    def test_input_permutations_and_noncontiguous_source_ids(self):
        edges = [range(8), (0,), range(4), range(1, 9), range(5), range(5)]
        index = incidence(edges, [99, 3, 12, 701, 22, 38])
        expected = compile_hypergraph(12, index)
        for seed in range(10):
            rng = random.Random(seed)
            permutation = list(range(len(index[0])))
            rng.shuffle(permutation)
            source_ids = sorted(set(index[1]))
            changed_ids = rng.sample(range(1000, 2000), len(source_ids))
            mapping = dict(zip(source_ids, changed_ids))
            shuffled = [[index[0][i] for i in permutation],
                        [mapping[index[1][i]] for i in permutation]]
            actual = compile_hypergraph(12, shuffled)
            self.assertEqual(expected.tokens, actual.tokens)

    def test_invalid_incidence(self):
        for index in ([[0], []], [[-1], [0]], [[3], [0]], [[0], [-1]], [[0.5], [0]]):
            with self.subTest(index=index), self.assertRaises(ValueError):
                compile_hypergraph(3, index)
        with self.assertRaises(ValueError):
            compile_hypergraph(3, [[0], [3]], num_hyperedges=2)

    def test_invalid_references_and_patch_membership(self):
        space = TokenSpace(12, 2)
        base = EdgeIR(0, 8, DIRECT, add=tuple(range(8))).tokens(space)
        invalid = [
            EdgeIR(1, 8, PATCH, ref=1, drop=(0,), add=(8,)),  # self/future
            EdgeIR(1, 8, PATCH, ref=0, drop=(10,), add=(8,)),
            EdgeIR(1, 8, PATCH, ref=0, drop=(0,), add=(1,)),
            EdgeIR(1, 8, EXTEND, ref=0, add=()),
            EdgeIR(1, 2, PATCH, ref=0, drop=tuple(range(8)), add=(9, 10)),
        ]
        for ir in invalid:
            with self.subTest(ir=ir), self.assertRaises(ValueError):
                parse_hir(base + ir.tokens(space), 12, 2)
        bad_ref = list(base + EdgeIR(1, 9, EXTEND, 0, add=(8,)).tokens(space))
        bad_ref[len(base) + 5] = space.edge(4)
        with self.assertRaises(ValueError):
            parse_hir(bad_ref, 12, 2)

    def test_bad_nodes_orders_and_incomplete_syntax(self):
        p = self.round_trip(5, [(0, 1)])
        variants = [p.tokens[:-1], p.tokens + (END_EDGE,)]
        for position, value in [(1, p.space.edge(4)), (2, p.space.order(3)),
                                (3, FRAGMENT), (4, p.space.node(5)), (5, p.space.node(0))]:
            bad = list(p.tokens)
            bad[position] = value
            variants.append(bad)
        for tokens in variants:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                parse_hir(tokens, 5, 1)

    def test_random_round_trip(self):
        for seed in range(60):
            rng = random.Random(seed)
            n = rng.randrange(0, 25)
            edges = [tuple(rng.sample(range(n), rng.randrange(n + 1)))
                     for _ in range(rng.randrange(0, 35))]
            if edges:
                edges.append(edges[0])
            with self.subTest(seed=seed):
                self.round_trip(n, edges, explicit=True)

    def check_chunks(self, program, limit):
        chunks, stats = chunk_hir(program, limit)
        contributions, fragments = defaultdict(float), defaultdict(list)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.tokens), limit)
            self.assertEqual(len(chunk.tokens), len(chunk.orders))
            local, position = {}, 0
            readouts = {pos: (edge, weight) for pos, edge, weight in chunk.readouts}
            while position < len(chunk.tokens):
                start = position
                self.assertEqual(chunk.tokens[position], EDGE)
                edge = chunk.tokens[position + 1] - program.space.edge(0)
                order = chunk.tokens[position + 2] - program.space.order(0)
                mode = chunk.tokens[position + 3]
                position += 4
                if mode in (DIRECT, FRAGMENT):
                    nodes = []
                    while chunk.tokens[position] != END_EDGE:
                        nodes.append(chunk.tokens[position] - program.space.node(0))
                        position += 1
                    members = tuple(nodes)
                else:
                    self.assertEqual(chunk.tokens[position], REF)
                    ref = chunk.tokens[position + 1] - program.space.edge(0)
                    self.assertIn(ref, local)  # no dangling/cross-chunk references
                    position += 2
                    drops, adds = [], []
                    if mode == PATCH:
                        self.assertEqual(chunk.tokens[position], DROP)
                        position += 1
                        while chunk.tokens[position] != ADD:
                            drops.append(chunk.tokens[position] - program.space.node(0))
                            position += 1
                    self.assertEqual(chunk.tokens[position], ADD)
                    position += 1
                    while chunk.tokens[position] != END_EDGE:
                        adds.append(chunk.tokens[position] - program.space.node(0))
                        position += 1
                    members = tuple(sorted((set(local[ref]) - set(drops)) | set(adds)))
                position += 1
                if mode == FRAGMENT:
                    fragments[edge].extend(members)
                    self.assertEqual(order, len(program.hyperedges[edge]))
                else:
                    self.assertEqual(members, program.hyperedges[edge])
                    self.assertEqual(len(members), order)
                local[edge] = members
                if start in readouts:
                    target, weight = readouts[start]
                    self.assertEqual(target, edge)
                    contributions[edge] += weight
        self.assertEqual(set(contributions), set(range(len(program.hyperedges))))
        for weight in contributions.values():
            self.assertAlmostEqual(weight, 1.0)
        for edge, members in fragments.items():
            self.assertEqual(tuple(members), program.hyperedges[edge])
        return stats

    def test_long_edges_reference_chains_and_chunk_bounds(self):
        p = self.round_trip(90, [range(4), range(5), range(8), range(1, 9),
                                 range(60), range(1, 61)])
        for limit in (6, 14, 32, 128, 256):
            with self.subTest(limit=limit):
                stats = self.check_chunks(p, limit)
                if limit == 14:
                    self.assertGreater(stats['direct_fallbacks'], 0)
                    self.assertGreater(stats['long_edge_fragments'], 0)
                if limit == 32:
                    self.assertGreater(stats['context_definitions'], 0)
        with self.assertRaises(ValueError):
            chunk_hir(p, 5)

    def test_many_edges_are_not_truncated(self):
        p = self.round_trip(130, [(i, i + 1) for i in range(120)])
        stats = self.check_chunks(p, 32)
        self.assertGreater(stats['num_chunks'], 20)
        self.assertEqual(chunk_hir(p, 32), chunk_hir(p, 32))


if __name__ == '__main__':
    unittest.main()
