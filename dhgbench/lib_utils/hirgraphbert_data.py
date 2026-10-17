"""Structure-only graph IO and leak-free SSL views using existing local sampling."""

import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from hir import (ADD, DIRECT, DROP, END_EDGE, EXTEND, FRAGMENT, PATCH, REF,
                 EdgeIR, chunk_hir, compile_hypergraph)
from lib_models.hirgraphbert import structure_input
from lib_utils.hirbert_local_sampling import LocalPretrainSampler, chunk_members
from lib_utils.hirbert_pretrain_utils import UnlabeledGraph, split_examples


TASKS = ('membership', 'relation', 'hir')
MODES = (DIRECT, EXTEND, PATCH)


def structure_graph(data, name):
    """Strip features at the CPU boundary; legacy sampler sees a zero-width x."""
    from lib_models.hirgraphbert import compile_data
    return UnlabeledGraph(torch.empty((int(data.num_nodes), 0)), compile_data(data), name)


def load_graph(args, labels=False):
    """Reuse DHG-Bench's dataset cache, without its feature/model preprocessing.

    Only shape metadata is read from x. Even when the cache stores features,
    neither the returned graph nor anything moved to the GPU contains them.
    Labels are accessed only by the explicitly supervised caller.
    """
    from lib_dataset.convert_datasets_to_pygDataset import dataset_Hypergraph

    base = Path(args.data_root)
    name = args.dname
    if name in ('cora', 'citeseer', 'pubmed'):
        raw_path = base / 'trad_data' / 'cocitation'
    elif name in ('coauthor_cora', 'coauthor_dblp'):
        raw_path = base / 'trad_data' / 'coauthorship'
    elif name == 'yelp':
        raw_path = base / 'trad_data' / 'yelp'
    elif name in ('bail', 'credit', 'german'):
        raw_path = base / 'fair_data'
    elif name in ('actor', 'amazon', 'pokec', 'twitch'):
        raw_path = base / 'hete_data'
    elif name in ('amazon_review', 'magpm', 'magpm_mini', 'trivago', 'ogbn_mag'):
        raw_path = base / 'large_data'
    else:
        raw_path = base / 'trad_data'
    noisy = name in ('walmart-trips-100', 'house-committees-100', 'walmart-trips', 'house-committees')
    dataset = dataset_Hypergraph(name=name, root=str(base / 'pyg_data' / 'hypergraph_dataset_updated'),
                                p2raw=str(raw_path), feature_noise=args.feature_noise if noisy else None)
    raw = dataset.data
    n = int(raw.x.shape[0])
    count = getattr(raw, 'num_hyperedges', None)
    index = raw.edge_index.detach().cpu().long()
    # Cached graphs use the symmetric bipartite [V|E; E|V] representation.
    index = index[:, index[0] < n].clone()
    if index.numel():
        if int(index[1].min()) >= n:
            index[1] -= n
        count = int(index[1].max()) + 1 if count is None else int(count)
    else:
        count = 0 if count is None else int(count)
    data = SimpleNamespace(num_nodes=n, num_hyperedges=count, hyperedge_index=index)
    graph = structure_graph(data, name)
    graph.source_paths = {'raw': str(raw_path), 'processed': str(dataset.processed_paths[0])}
    print(f'[Structure-only data] {name}: nodes={n} edges={len(graph.program.hyperedges)} cache={dataset.processed_paths[0]}')
    if not labels:
        return graph
    y = raw.y.detach().cpu().long().reshape(-1)
    # Match benchmark label normalization, including fair/heterogeneous datasets.
    known = y >= 0
    if known.any():
        y[known] = torch.unique(y[known], sorted=True, return_inverse=True)[1]
    return SimpleNamespace(**vars(data), y=y, num_classes=int(y[known].max()) + 1 if known.any() else 0,
                           name=name)


@dataclass(frozen=True)
class HIRTargets:
    edges: tuple
    modes: tuple
    orders: tuple
    references: tuple
    # (masked edge, candidate node, NONE/ADD/DROP label)
    payloads: tuple
    hidden_edges: tuple


def hir_snapshot(program, chunks=None):
    """Recover visible definitions and actual OUTPUT instructions in local IDs.

    Packet recompilation stores empty memberships for reference-only context
    edges and partial memberships for fragments. Its instructions consequently
    are not the instructions actually fed to the encoder. Read the lowered
    stream instead, including its copied context, without consulting true edges.
    """
    if chunks is None:
        return ({edge: set(members) for edge, members in enumerate(program.hyperedges)},
                {ir.edge_id: ir for ir in program.instructions})
    members, outputs, fragments, fragment_orders = defaultdict(set), {}, defaultdict(set), {}
    for chunk in chunks:
        definitions, position = {}, 0
        readouts = {offset for offset, _, _ in chunk.readouts}
        while position < len(chunk.tokens):
            end = chunk.tokens.index(END_EDGE, position)
            record = chunk.tokens[position:end + 1]
            edge, order, mode = record[1] - program.space.edge(0), record[2] - program.space.order(0), record[3]
            body, reference, drops = record[4:-1], -1, ()
            if mode in (DIRECT, FRAGMENT):
                adds = tuple(token - program.space.node(0) for token in body)
                current = set(adds)
            else:
                if mode not in (EXTEND, PATCH) or body[0] != REF:
                    raise ValueError('Invalid visible H-IR instruction')
                reference = body[1] - program.space.edge(0)
                add_position = body.index(ADD)
                if mode == PATCH:
                    if body[2] != DROP:
                        raise ValueError('Missing visible DROP marker')
                    drops = tuple(token - program.space.node(0) for token in body[3:add_position])
                adds = tuple(token - program.space.node(0) for token in body[add_position + 1:])
                current = (definitions[reference] - set(drops)) | set(adds)
            definitions[edge] = current
            members[edge].update(current)
            if position in readouts:
                if mode == FRAGMENT:
                    fragments[edge].update(adds)
                    fragment_orders[edge] = order
                else:
                    outputs[edge] = EdgeIR(edge, order, mode, reference, drops, adds)
            position = end + 1
    for edge, nodes in fragments.items():
        if edge not in outputs and len(nodes) == fragment_orders[edge]:
            # Only a COMPLETE fragment union has known payload negatives. A
            # missing member of a partial span cannot be labeled NONE: it may
            # be an actual member in a different packet. Keep partial fragments
            # as visible context, but do not invent H-IR supervision for them.
            outputs[edge] = EdgeIR(edge, len(nodes), FRAGMENT, add=tuple(sorted(nodes)))
    return dict(members), outputs


def masked_hir_input(program, target_edges, max_length, node_codes, edge_codes, rng, payload_count=32,
                     chunks=None, duplicate_groups=None, anchors=None, max_targets=None):
    """Recover hidden fields conditional on one visible member per instruction.

    The member must connect to a kept definition. All target definitions,
    fragments, equal-edge copies and incoming references are removed/recompiled.
    A constant-size MASK + NODE anchor replaces them. The anchor's own operation
    is never a payload label. This is an input view, not a new H-IR grammar.
    Labels describe the actual Membership-corrupted stream BEFORE this mask.
    """
    members, instructions = hir_snapshot(program, chunks)
    duplicate_groups = duplicate_groups or {}

    def hidden_edges(selected):
        identical = {tuple(sorted(members[edge])) for edge in selected}
        families = {duplicate_groups[edge] for edge in selected if edge in duplicate_groups}
        return {edge for edge, nodes in members.items() if tuple(sorted(nodes)) in identical
                or (edge in duplicate_groups and duplicate_groups[edge] in families)}

    def output_members(edge):
        return set(instructions[edge].add) if instructions[edge].mode == FRAGMENT else members[edge]

    selected = []
    for edge in target_edges:
        if edge not in instructions or (selected and edge in hidden_edges(selected)):
            continue  # Reference-only definitions are not prediction targets.
        proposal = selected + [edge]
        hidden = hidden_edges(proposal)
        context_nodes = set().union(*(nodes for other, nodes in members.items() if other not in hidden))
        # At least one member stays hidden, and every query retains an actual
        # structural link. Do not add a target that removes another's context.
        if all(len(output_members(target)) >= 2 and output_members(target) & context_nodes for target in proposal):
            selected.append(edge)
            if max_targets is not None and len(selected) >= max_targets:
                break
    if not selected:
        return None, HIRTargets((), (), (), (), (), ())
    hidden = hidden_edges(selected)
    kept = sorted(set(members) - hidden)
    context_nodes = set().union(*(members[edge] for edge in kept))
    visible_anchors = {}
    for edge in selected:
        # For a fragment, anchor a member actually present in the output span,
        # not a member available only in an auxiliary context copy.
        available = output_members(edge) & context_nodes
        if not available:
            raise ValueError('Masked H-IR target has no visible output anchor')
        chosen = anchors[edge] if anchors is not None and edge in anchors else rng.choice(sorted(available))
        if chosen not in available:
            raise ValueError('Masked H-IR anchor is not a visible shared member')
        visible_anchors[edge] = (chosen,)
    targets = tuple(instructions[edge] for edge in selected)
    index = torch.tensor([(node, new) for new, edge in enumerate(kept) for node in sorted(members[edge])],
                         dtype=torch.long).reshape(-1, 2).t().contiguous()
    context = compile_hypergraph(program.num_nodes, index, len(kept))
    mapping = tuple(kept[old] for old in context.source_edge_ids)
    chunks = list(chunk_hir(context, max_length - 1)[0])
    # Independent of the target's original ordinal/canonical position.
    rng.shuffle(chunks)
    view = structure_input(context, chunks, max_length, node_codes, edge_codes, mapping,
                           len(program.hyperedges), selected, visible_anchors)
    payloads = []
    for ir in targets:
        visible = set(visible_anchors[ir.edge_id])
        positive = [node for node in ir.add + ir.drop if node not in visible]
        rng.shuffle(positive)
        chosen = set(positive[:max(1, payload_count // 2)])
        operations = set(ir.add + ir.drop) | visible
        complement = [node for node in range(program.num_nodes) if node not in operations]
        chosen.update(rng.sample(complement, min(len(complement), max(1, payload_count - len(chosen)))))
        adds, drops = set(ir.add), set(ir.drop)
        payloads.extend((ir.edge_id, node, 1 if node in adds else 2 if node in drops else 0) for node in sorted(chosen))
    labels = HIRTargets(tuple(ir.edge_id for ir in targets),
                        tuple(MODES.index(ir.mode) if ir.mode != FRAGMENT and (ir.ref < 0 or ir.ref in kept)
                              else -100 for ir in targets),
                        tuple(ir.order if ir.mode != FRAGMENT else -100 for ir in targets),
                        tuple(ir.ref if ir.ref in kept else -1 if ir.ref < 0 and ir.mode != FRAGMENT else -2 for ir in targets),
                        tuple(payloads), tuple(sorted(hidden)))
    return view, labels


class StructureLocalSampler(LocalPretrainSampler):
    """Reuse legacy targets/mappings, without legacy token-budget admission."""
    def __init__(self, graph, args):
        super().__init__(graph, args)
        # Removing every member leaves an anonymous EDGE with no structural
        # condition for Membership prediction. Reserve one visible member per
        # duplicate family; singleton/universal edges are context only.
        rng = random.Random(args.pretrain_seed + 211)
        pool = []
        for group in self.groups:
            edge = group[0]
            if 1 < len(self.edges[edge]) < graph.num_nodes:
                anchor = rng.choice(self.edges[edge])
                pool.extend((node, edge) for node in self.edges[edge] if node != anchor)
        self.train_members, self.val_members = split_examples(pool, args.ssl_val_ratio, rng)
        self._valid_negative_counts = Counter(edge for _, edge in self.val_members)
        self._valid_negative_nodes = {}

    def _select_edges(self, root, priority, removals, rng):
        """Bound neighborhood growth by graph scope, never by encoded size."""
        selected = list(dict.fromkeys([root] + (list(priority[:2]) if priority else [])))
        if len(selected) > self.args.local_max_edges:
            selected = list(priority[:2])
        desired = rng.randint(self.args.local_min_edges, self.args.local_max_edges)
        visible = {}

        def members(edge):
            if edge not in visible:
                visible[edge] = set(self.edges[edge]) - removals.get(edge, set())
            return visible[edge]

        nodes = set().union(*(members(edge) for edge in selected))
        queue, seen = deque((edge, 0) for edge in selected), set(selected)
        while queue and len(selected) < desired:
            edge, depth = queue.popleft()
            if depth >= self.args.local_hops:
                continue
            candidates = list(self._neighbors(edge, removals).items())
            rng.shuffle(candidates)
            candidates.sort(key=lambda item: (-item[1] / max(1, min(len(members(edge)), len(members(item[0])))),
                                               -item[1]))
            for other, _ in candidates:
                if other in seen:
                    continue
                seen.add(other)
                if len(nodes | members(other)) > self.args.local_max_nodes:
                    continue
                selected.append(other)
                nodes.update(members(other))
                queue.append((other, depth + 1))
                if len(selected) >= desired:
                    break
        # Mandatory supervised endpoints stay whole, including large edges.
        return selected

    def sample(self, split, epoch, sample_index):
        """Materialize one full CPU sample; GPU execution groups its chunks.

        No token/chunk budget changes its neighborhood, targets or compiler
        output. Validation and OOM replays use the same deterministic inputs.
        """
        epoch = epoch if split == 'train' else 0
        removals, hidden_members, _, _ = self.plan(split, epoch)
        rng = random.Random(self.args.pretrain_seed + epoch * 104729 + sample_index * 9176
                            + (0 if split == 'train' else 7919))
        anchors, relation_order = self._epoch_orders(split, epoch, hidden_members, (), removals)
        if not anchors:
            raise ValueError('No usable local self-supervised target edges')
        root, priority = anchors[sample_index % len(anchors)], None
        if relation_order:
            labels = sorted(relation_order)
            rotation = (epoch - 1) * self.args.local_samples_per_graph if split == 'train' else 0
            label = labels[(sample_index + rotation) % len(labels)]
            near = set(self._neighbors(root, removals)) | {root}
            entries = relation_order[label]
            close = [pair for pair in entries if pair[0] in near and pair[1] in near]
            touching = [pair for pair in entries if root in pair[:2]]
            candidates = touching or close or entries
            priority = candidates[(sample_index // len(labels)) % len(candidates)]
        selected = self._select_edges(root, priority, removals, rng)
        counts = SimpleNamespace(targets=self.args.local_target_count, feature_targets=0,
                                 relation_targets=self.args.local_relation_targets)
        membership, pairs, labels, _ = self._targets(
            selected, priority, relation_order, hidden_members, (), removals, counts, rng, split)
        selected = sorted(selected)
        visible = {edge: tuple(node for node in self.edges[edge] if node not in removals.get(edge, ()))
                   for edge in selected}
        global_nodes = sorted({node for node, _, _ in membership} |
                              {node for values in visible.values() for node in values})
        node_index = {node: index for index, node in enumerate(global_nodes)}
        incidence = torch.tensor([(node_index[node], edge) for edge, original in enumerate(selected)
                                  for node in visible[original]], dtype=torch.long).reshape(-1, 2).t().contiguous()
        full = compile_hypergraph(len(global_nodes), incidence, len(selected))
        original_edges = [selected[source] for source in full.source_edge_ids]
        chunks = chunk_hir(full, self.args.hir_max_seq_length)[0]
        context = [chunk_members(full, chunk) for chunk in chunks]
        batch = self._packet(full, chunks, context, tuple(range(len(chunks))), global_nodes, original_edges,
                             membership, pairs, labels, (), set(), self.args.hir_max_seq_length)
        # All fragments belong to this one sample, so each target is owned once.
        batch.target_weights = {'membership': torch.ones(len(batch.membership_labels)),
                                'relation': torch.ones(len(batch.relation_labels))}
        batch.task_scales = {task: float(len(weights) > 0) for task, weights in batch.target_weights.items()}
        batch.task_scales['context'] = 1.
        yield batch

    def _negative_nodes(self, edge):
        if edge not in self._valid_negative_nodes:
            size = min(self.graph.num_nodes - len(self.edges[edge]),
                       math.ceil(self._valid_negative_counts[edge] * self.args.negative_ratio))
            rng = random.Random(self.args.pretrain_seed + 65537 * (edge + 1))
            chosen = set()
            for _ in range(size * 8):
                node = rng.randrange(self.graph.num_nodes)
                if node not in self._edge_sets[edge]:
                    chosen.add(node)
                if len(chosen) == size:
                    break
            if len(chosen) < size:
                pool = [node for node in range(self.graph.num_nodes)
                        if node not in self._edge_sets[edge] and node not in chosen]
                chosen.update(rng.sample(pool, size - len(chosen)))
            self._valid_negative_nodes[edge] = chosen
        return self._valid_negative_nodes[edge]

    def _targets(self, selected, priority, relation_order, hidden_members, feature_pool, removals, limits, rng, split):
        membership, pairs, labels, features = super()._targets(
            selected, priority, relation_order, hidden_members, feature_pool, removals, limits, rng, split)
        # Random negatives in the legacy implementation could supervise the
        # exact same (node, edge) in train and validation. Reserve a fixed bank
        # per edge, then resample only this task's negative part.
        positives = [target for target in membership if target[2] == 1.]
        edges = [edge for _, edge, _ in positives]
        rng.shuffle(edges)
        counts = Counter(edges[index % len(edges)] for index in range(math.ceil(len(edges) * self.args.negative_ratio)))
        membership = list(positives)
        for edge, count in sorted(counts.items()):
            heldout = self._negative_nodes(edge)
            if split != 'train':
                nodes = rng.sample(sorted(heldout), min(count, len(heldout)))
            else:
                count = min(count, self.graph.num_nodes - len(self.edges[edge]) - len(heldout))
                nodes = set()
                for _ in range(count * 8):
                    node = rng.randrange(self.graph.num_nodes)
                    if node not in self._edge_sets[edge] and node not in heldout:
                        nodes.add(node)
                    if len(nodes) == count:
                        break
                if len(nodes) < count:
                    pool = [node for node in range(self.graph.num_nodes)
                            if node not in self._edge_sets[edge] and node not in heldout and node not in nodes]
                    nodes.update(rng.sample(pool, count - len(nodes)))
            membership.extend((node, edge, 0.) for node in sorted(nodes))
        return membership, pairs, labels, features


@dataclass
class StructureSSLBatch:
    main: object
    relation: object
    hir: object
    membership_pairs: torch.Tensor
    membership_labels: torch.Tensor
    relation_pairs: torch.Tensor
    relation_labels: torch.Tensor
    hir_targets: HIRTargets
    target_weights: dict
    task_scales: dict
    dataset: str
    node_ids: tuple
    source_edge_ids: tuple

    @property
    def view_costs(self):
        return {name: (view.token_count, view.num_nodes, len(view.sequences))
                for name in ('main', 'relation', 'hir') if (view := getattr(self, name)) is not None}

    @property
    def cost(self):
        costs = list(self.view_costs.values())
        return (sum(cost[0] for cost in costs), max((cost[1] for cost in costs), default=0),
                sum(cost[2] for cost in costs))


def task_packets(batch):
    """Schedule each independent view once, retaining all chunks and targets.

    There is no admission budget here. Encoder.encode_many groups complete
    chunks under the existing forward cap and pools across the whole view.
    """
    for name in ('main', 'relation', 'hir'):
        if getattr(batch, name) is None:
            continue
        main, relation, hir = name == 'main', name == 'relation', name == 'hir'
        scales = {**batch.task_scales,
                  'membership': batch.task_scales['membership'] if main else 0.,
                  'relation': batch.task_scales['relation'] if relation else 0.,
                  'hir': batch.task_scales['hir'] if hir else 0.,
                  'context': batch.task_scales['context'] if main else 0.}
        yield replace(batch, main=batch.main if main else None, relation=batch.relation if relation else None,
                      hir=batch.hir if hir else None,
                      membership_pairs=batch.membership_pairs if main else batch.membership_pairs[:0],
                      membership_labels=batch.membership_labels if main else batch.membership_labels[:0],
                      relation_pairs=batch.relation_pairs if relation else batch.relation_pairs[:0],
                      relation_labels=batch.relation_labels if relation else batch.relation_labels[:0],
                      hir_targets=batch.hir_targets if hir else HIRTargets((), (), (), (), (), ()),
                      task_scales=scales)


class StructureSampler:
    """Reuse duplicate-safe masks, complete neighborhoods and target mappings.

    Global source IDs remain only in metadata/split bookkeeping. The encoder
    receives newly permuted local codes on each sample, consistently across
    chunks and the three branches, without a trainable global identity table.
    """
    def __init__(self, graph, args):
        self.graph, self.args = graph, args
        # Discard caller-provided features as well (e.g. in a supplied test graph).
        self.graph = replace(graph, x=torch.empty((graph.num_nodes, 0)))
        legacy_args = SimpleNamespace(**{**vars(args), 'tasks': ['membership', 'relation'],
                                        'hir_max_seq_length': args.hir_max_seq_length - 1})
        self.legacy = StructureLocalSampler(self.graph, legacy_args)
        rng = random.Random(args.pretrain_seed + 6163)
        groups = list(self.legacy.groups)
        rng.shuffle(groups)
        count = min(len(groups) - 1, max(1, round(len(groups) * args.ssl_val_ratio)))
        self.valid_hir = {edge for group in groups[:count] for edge in group}
        self.train_hir = set(range(len(graph.program.hyperedges))) - self.valid_hir

    def summary(self):
        old = self.legacy.summary()
        return {key: value for key, value in old.items() if not key.startswith('feature_')} | {
            'hir_train_edges': len(self.train_hir), 'hir_valid_edges': len(self.valid_hir)}

    def sample(self, split, epoch, sample_index):
        rng = random.Random(self.args.pretrain_seed + (epoch * 1000003 if split == 'train' else 7919) + sample_index * 9176)
        allowed = self.train_hir if split == 'train' else self.valid_hir
        # Share one local identity code across all chunks and task views.
        packets = list(self.legacy.sample(split, epoch, sample_index))
        sample_nodes = sorted({node for batch in packets for node in batch.view.node_ids.tolist()})
        sample_edges = sorted({edge for batch in packets for edge in batch.view.program.source_edge_ids})
        node_values, edge_values = list(range(1, len(sample_nodes) + 1)), list(range(1, len(sample_edges) + 1))
        rng.shuffle(node_values)
        rng.shuffle(edge_values)
        node_identity, edge_identity = dict(zip(sample_nodes, node_values)), dict(zip(sample_edges, edge_values))
        prepared = []
        for batch in packets:
            program, chunks = batch.view.program, list(batch.view.chunks)
            node_codes = [node_identity[node] for node in batch.view.node_ids.tolist()]
            edge_codes = [edge_identity[edge] for edge in program.source_edge_ids]
            rng.shuffle(chunks)
            main = structure_input(program, chunks, self.args.hir_max_seq_length, node_codes, edge_codes)
            relation = None
            # The relation label describes the ORIGINAL pair. A full visible
            # endpoint after Membership masking is a meaningful denoising view;
            # an arbitrary local fragment is not a complete endpoint.
            complete = {ir.edge_id for ir in batch.view.relation_program.instructions
                        if ir.mode != FRAGMENT and ir.order > 0}
            usable = torch.tensor([int(first) in complete and int(second) in complete
                                   for first, second in batch.relation_pairs.tolist()], dtype=torch.bool)
            relation_pairs, relation_labels = batch.relation_pairs[usable], batch.relation_labels[usable]
            if len(relation_labels):
                relation_chunks = list(batch.view.relation_chunks)
                rng.shuffle(relation_chunks)
                relation = structure_input(batch.view.relation_program, relation_chunks, self.args.hir_max_seq_length,
                                           node_codes, edge_codes)
            outputs = {edge for chunk in batch.view.chunks for _, edge, _ in chunk.readouts}
            candidates = [edge for edge, source in enumerate(program.source_edge_ids)
                          if source in allowed and edge in outputs and len(program.hyperedges[edge]) >= 2]
            rng.shuffle(candidates)
            target_count = min(self.args.local_hir_targets, max(1, math.ceil(len(candidates) * self.args.hir_mask_ratio)))
            hir, hir_targets = None, HIRTargets((), (), (), (), (), ())
            if candidates:
                families = {edge: self.legacy.edge_group[source] for edge, source in enumerate(program.source_edge_ids)}
                hir, hir_targets = masked_hir_input(program, candidates, self.args.hir_max_seq_length, node_codes,
                                                    edge_codes, rng, self.args.hir_payload_candidates,
                                                    chunks=batch.view.chunks, duplicate_groups=families,
                                                    max_targets=target_count)
            weights = {key: value for key, value in batch.target_weights.items() if key in ('membership', 'relation')}
            prepared.append(StructureSSLBatch(main, relation, hir, batch.membership_pairs, batch.membership_labels,
                                             relation_pairs, relation_labels, hir_targets, weights,
                                             {key: batch.task_scales[key] for key in ('membership', 'relation', 'context')} |
                                             {'hir': 0.}, self.graph.dataset,
                                             tuple(batch.view.node_ids.tolist()), program.source_edge_ids))
        # Renormalize only targets actually retained in this logical sample.
        # Dropped partial relation endpoints and packets without H-IR targets
        # must not silently dilute the corresponding loss.
        def relation_keys(batch):
            return [tuple(sorted((batch.source_edge_ids[first], batch.source_edge_ids[second])))
                    for first, second in batch.relation_pairs.tolist()]

        relations = Counter(key for batch in prepared for key in relation_keys(batch))
        hir_packets = sum(bool(batch.hir_targets.edges) for batch in prepared)
        for batch in prepared:
            weights = torch.tensor([1. / relations[key] for key in relation_keys(batch)], dtype=torch.float32)
            batch.target_weights['relation'] = weights
            batch.task_scales['relation'] = weights.sum().item() / max(1, len(relations))
            batch.task_scales['hir'] = 1. / hir_packets if batch.hir_targets.edges else 0.
            yield from task_packets(batch)
