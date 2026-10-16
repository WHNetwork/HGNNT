"""CPU corruption plans and bounded, complete-fragment H-IR training views."""

import math
import random
from collections import Counter, deque
from dataclasses import dataclass, replace

import torch

from hir import ADD, DIRECT, DROP, EDGE_ID, END_EDGE, EXTEND, FRAGMENT, PATCH, REF, EdgeIR, HIRChunk, chunk_hir, compile_hypergraph
from lib_utils.hirbert_pretrain_utils import PretrainBatch, PretrainSampler, VisibleView, direct_program, relation_label


class LocalBudgetError(ValueError):
    pass


@dataclass
class LocalLimits:
    tokens: int
    nodes: int
    chunks: int
    targets: int
    chunk_batch_size: int
    edges: int = 128
    feature_targets: int = 32
    relation_targets: int = 32
    sequence_length: int = 512

    def shrink(self):
        self.tokens = max(96, self.tokens // 2)
        self.nodes = max(32, self.nodes // 2)
        self.targets = max(1, self.targets // 2)
        self.chunk_batch_size = max(1, self.chunk_batch_size // 2)
        self.edges = max(2, self.edges // 2)
        self.feature_targets = max(1, self.feature_targets // 2)
        self.relation_targets = max(1, self.relation_targets // 2)


def chunk_members(program, chunk):
    """Interpret copied definitions and fragments, not just NODE occurrences."""
    definitions, output, position = {}, {}, 0
    targets = {offset for offset, _, _ in chunk.readouts}
    while position < len(chunk.tokens):
        end = chunk.tokens.index(END_EDGE, position)
        edge = chunk.tokens[position + 1] - program.space.edge(0)
        mode, body = chunk.tokens[position + 3], chunk.tokens[position + 4:end]
        if mode in (DIRECT, FRAGMENT):
            members = {token - program.space.node(0) for token in body}
        else:
            if body[0] != REF:
                raise ValueError('Missing local reference')
            members = set(definitions[body[1] - program.space.edge(0)])
            add = body.index(ADD)
            if mode == PATCH:
                members.difference_update(token - program.space.node(0) for token in body[3:add])
            elif mode != EXTEND:
                raise ValueError('Unknown local H-IR mode')
            members.update(token - program.space.node(0) for token in body[add + 1:])
        definitions[edge] = members
        if position in targets:
            output.setdefault(edge, set()).update(members)
        position = end + 1
    return output


def move_batch(batch, device):
    # Programs, chunks, original IDs and all deterministic caches stay on CPU.
    view = replace(batch.view, x=batch.view.x.to(device), incidence=batch.view.incidence.to(device),
                   masked_nodes=batch.view.masked_nodes.to(device))
    weights = {key: value.to(device) for key, value in batch.target_weights.items()} if batch.target_weights else None
    return replace(batch, view=view, target_weights=weights, **{key: getattr(batch, key).to(device) for key in (
        'membership_pairs', 'membership_labels', 'feature_nodes', 'feature_targets', 'relation_pairs', 'relation_labels')})


def target_keys(batch):
    """Stable CPU identities, also used to deduplicate fragment supervision stats."""
    nodes, edges = batch.view.node_ids.tolist(), batch.view.program.source_edge_ids
    return {
        'membership': [(nodes[node], edges[edge], label) for (node, edge), label in
                       zip(batch.membership_pairs.tolist(), batch.membership_labels.tolist())],
        'feature': [nodes[node] for node in batch.feature_nodes.tolist()],
        'relation': [tuple(sorted((edges[first], edges[second]))) for first, second in batch.relation_pairs.tolist()],
    }


class LocalPretrainSampler(PretrainSampler):
    def __init__(self, graph, args):
        super().__init__(graph, args)
        self.by_node = {}
        for edge, members in enumerate(self.edges):
            for node in members:
                self.by_node.setdefault(node, []).append(edge)
        self._plan_key, self._plan = None, None
        self._orders_key, self._orders = None, None
        self._edge_sets = [frozenset(edge) for edge in self.edges]
        self._heldout_relations = {tuple(sorted((self.edge_group[first], self.edge_group[second])))
                                  for bank in self.val_relations.values() for first, second, _ in bank}

    def plan(self, split, epoch):
        key = (split, epoch if split == 'train' else 0)
        if key != self._plan_key:
            rng = random.Random(self.args.pretrain_seed + (epoch * 1000003 if split == 'train' else 7919))
            members = (rng.sample(self.train_members, max(1, math.ceil(len(self.train_members) * self.args.membership_mask_ratio)))
                       if self.train_members and split == 'train' else list(self.val_members))
            features = (rng.sample(self.train_features, max(1, math.ceil(len(self.train_features) * self.args.feature_mask_ratio)))
                        if self.train_features and split == 'train' else list(self.val_features))
            removals, by_edge = {}, {}
            for node, edge in set(self.val_members) | set(members):
                for duplicate in self.groups[self.edge_group[edge]]:
                    removals.setdefault(duplicate, set()).add(node)
            for node, edge in members:
                by_edge.setdefault(edge, []).append(node)
            self._plan = (removals, by_edge, set(self.val_features) | set(features), features)
            self._plan_key = key
        return self._plan

    def _epoch_orders(self, split, epoch, hidden_members, feature_pool, removals):
        key = (split, epoch)
        if key != self._orders_key:
            rng = random.Random(self.args.pretrain_seed + epoch * 1000003 + (7919 if split != 'train' else 0))
            # Cycle shuffled target edges without replacement before repeating.
            anchors = sorted(hidden_members)
            if not anchors and feature_pool:
                anchors = sorted({edge for node in feature_pool for edge in self.by_node[node]
                                  if node not in removals.get(edge, ())})
            if not anchors:
                anchors = [edge for edge, nodes in enumerate(self.edges) if nodes]
            rng.shuffle(anchors)
            bank = self.train_relations if split == 'train' else self.val_relations
            relations = {label: rng.sample(entries, len(entries)) for label, entries in bank.items()}
            self._orders_key, self._orders = key, (anchors, relations)
        return self._orders

    def _neighbors(self, edge, removals):
        neighbors = Counter()
        for node in self.edges[edge]:
            if node not in removals.get(edge, ()):
                neighbors.update(other for other in self.by_node[node]
                                 if other != edge and node not in removals.get(other, ()))
        return neighbors

    def _select_edges(self, root, priority, removals, limits, rng):
        """Grow through visible shared nodes, checking actual compiled size.

        Relation endpoints are supervised seeds (including disjoint pairs), not
        unrelated filler. Budgets/connected-component size override soft goals.
        A mandatory long target edge is kept whole on CPU and fragmented later.
        """
        selected = list(dict.fromkeys([root] + (list(priority[:2]) if priority else [])))
        if len(selected) > limits.edges:
            selected = list(priority[:2])
        desired = rng.randint(min(self.args.local_min_edges, limits.edges), limits.edges)
        soft_tokens = min(self.args.local_target_tokens, limits.tokens)
        visible_cache = {}

        def visible(edge):
            if edge not in visible_cache:
                visible_cache[edge] = set(self.edges[edge]) - removals.get(edge, set())
            return visible_cache[edge]
        nodes = set().union(*(visible(edge) for edge in selected))
        mandatory = len(selected)

        def compiled_size():
            global_nodes = sorted(set().union(*(visible(edge) for edge in selected)))
            mapping = {node: i for i, node in enumerate(global_nodes)}
            index = torch.tensor([(mapping[node], i) for i, edge in enumerate(selected) for node in visible(edge)],
                                 dtype=torch.long).reshape(-1, 2).t().contiguous()
            program = compile_hypergraph(len(global_nodes), index, len(selected))
            chunks = chunk_hir(program, self.args.hir_max_seq_length)[0] if self.tasks & {'membership', 'feature'} else ()
            if 'relation' in self.tasks:
                chunks += chunk_hir(direct_program(program), self.args.hir_max_seq_length)[0]
            return sum(len(chunk.tokens) for chunk in chunks), len(chunks)

        token_count, chunk_count = compiled_size()
        if token_count > limits.tokens or chunk_count > limits.chunks or len(nodes) > limits.nodes:
            # Mandatory large edges use COMPLETE fragment streams, not member
            # truncation. Do not compound them with extra context edges.
            return selected
        checked = mandatory
        queue = deque((edge, 0) for edge in selected)
        seen = set(selected)
        while queue and len(selected) < desired:
            edge, depth = queue.popleft()
            if depth >= self.args.local_hops:
                continue
            candidates = list(self._neighbors(edge, removals).items())
            rng.shuffle(candidates)
            # Visible containment/large overlap precedes weak intersections.
            candidates.sort(key=lambda item: (-item[1] / max(1, min(len(visible(edge)), len(visible(item[0])))), -item[1]))
            for other, _ in candidates:
                if other in seen:
                    continue
                seen.add(other)
                members = visible(other)
                if len(nodes | members) > limits.nodes:
                    continue
                selected.append(other)
                nodes.update(members)
                queue.append((other, depth + 1))
                # Amortize CPU compilation over a few additions; estimate-free
                # backoff removes whole optional edges, never their members.
                if len(selected) - checked >= 8 or len(selected) >= desired:
                    token_count, chunk_count = compiled_size()
                    while len(selected) > mandatory and (token_count > limits.tokens or chunk_count > limits.chunks):
                        selected.pop()
                        token_count, chunk_count = compiled_size()
                        if token_count <= limits.tokens and chunk_count <= limits.chunks:
                            return selected
                    checked = len(selected)
                    if len(selected) >= desired or (len(selected) >= self.args.local_min_edges and token_count >= soft_tokens):
                        return selected
        token_count, chunk_count = compiled_size()
        while len(selected) > mandatory and (token_count > limits.tokens or chunk_count > limits.chunks):
            selected.pop()
            token_count, chunk_count = compiled_size()
        return selected

    def _targets(self, selected, priority, relation_order, hidden_members, feature_pool, removals, limits, rng, split):
        positives = []
        pools = [(edge, rng.sample(hidden_members[edge], len(hidden_members[edge])))
                 for edge in selected if edge in hidden_members]
        rng.shuffle(pools)
        # Spread positives over different edges before taking another member.
        while pools and len(positives) < limits.targets:
            remaining = []
            for edge, nodes in pools:
                positives.append((nodes.pop(), edge, 1.))
                if nodes:
                    remaining.append((edge, nodes))
                if len(positives) == limits.targets:
                    break
            pools = remaining
        membership = list(positives)
        negative_edges = [edge for _, edge, _ in positives]
        rng.shuffle(negative_edges)
        negative_counts = Counter(negative_edges[index % len(negative_edges)]
                                  for index in range(math.ceil(len(positives) * self.args.negative_ratio)))
        for edge, count in sorted(negative_counts.items()):
            negatives = set()
            count = min(count, self.graph.num_nodes - len(self.edges[edge]))
            for _ in range(count * 8):
                node = rng.randrange(self.graph.num_nodes)
                if node not in self._edge_sets[edge]:
                    negatives.add(node)
                if len(negatives) == count:
                    break
            if len(negatives) < count:
                pool = [node for node in range(self.graph.num_nodes)
                        if node not in self._edge_sets[edge] and node not in negatives]
                negatives.update(rng.sample(pool, count - len(negatives)))
            membership.extend((node, edge, 0.) for node in sorted(negatives))
        contextual = {node for edge in selected for node in self.edges[edge] if node not in removals.get(edge, ())}
        features = sorted(set(feature_pool) & contextual)
        rng.shuffle(features)
        features = features[:limits.feature_targets]
        banks = {label: [] for label in relation_order}
        if split == 'train':
            # Fresh local pairs greatly increase supervision beyond the global
            # reservoir. All held-out duplicate-group pairs remain excluded.
            for i, first in enumerate(selected):
                for second in selected[i + 1:]:
                    group = tuple(sorted((self.edge_group[first], self.edge_group[second])))
                    label = relation_label(self._edge_sets[first], self._edge_sets[second])
                    if group not in self._heldout_relations and label in banks:
                        banks[label].append((first, second, label))
        else:
            chosen = set(selected)
            banks = {label: [pair for pair in entries if pair[0] in chosen and pair[1] in chosen]
                     for label, entries in relation_order.items()}
        for entries in banks.values():
            rng.shuffle(entries)
        relations = [priority] if priority else []
        labels = list(banks)
        if priority:
            start = labels.index(priority[2]) + 1
            labels = labels[start:] + labels[:start]
        while len(relations) < limits.relation_targets and any(banks.values()):
            for label in labels:
                while banks[label]:
                    pair = banks[label].pop()
                    if pair not in relations and (pair[1], pair[0], pair[2]) not in relations:
                        relations.append(pair)
                        break
                if len(relations) == limits.relation_targets:
                    break
        return membership, [(first, second) for first, second, _ in relations], [label for _, _, label in relations], features

    def sample(self, split, epoch, sample_index, limits):
        """CPU neighborhood -> complete H-IR -> bounded packets, never truncation."""
        epoch = epoch if split == 'train' else 0
        removals, hidden_members, masked_features, feature_pool = self.plan(split, epoch)
        rng = random.Random(self.args.pretrain_seed + epoch * 104729 + sample_index * 9176
                            + (0 if split == 'train' else 7919))
        anchors, relation_order = self._epoch_orders(split, epoch, hidden_members, feature_pool, removals)
        if not anchors:
            raise ValueError('No usable local self-supervised target edges')
        root, priority = anchors[sample_index % len(anchors)], None
        if relation_order:
            labels = sorted(relation_order)
            rotation = (epoch - 1) * self.args.local_samples_per_graph * self.args.pretrain_steps_per_epoch if split == 'train' else 0
            label = labels[(sample_index + rotation) % len(labels)]
            near = set(self._neighbors(root, removals)) | {root}
            entries = relation_order[label]
            close = [pair for pair in entries if pair[0] in near and pair[1] in near]
            touching = [pair for pair in entries if root in pair[:2]]
            candidates = touching or close or entries
            priority = candidates[(sample_index // len(labels)) % len(candidates)]
        # Validation targets/neighborhoods are fixed even if an OOM changes
        # packet size later. Only their fragmentation/packing may then change.
        selection_limits = limits if split == 'train' else LocalLimits(
            self.args.local_max_tokens, self.args.local_max_nodes, self.args.local_max_chunks,
            self.args.local_target_count, self.args.hir_chunk_batch_size, self.args.local_max_edges,
            self.args.local_feature_targets, self.args.local_relation_targets, self.args.hir_max_seq_length)
        selected = self._select_edges(root, priority, removals, selection_limits, rng)
        membership, relation_pairs, relation_labels, features = self._targets(
            selected, priority, relation_order, hidden_members, feature_pool, removals, selection_limits, rng, split)
        selected = sorted(selected)
        visible = {edge: tuple(node for node in self.edges[edge] if node not in removals.get(edge, ())) for edge in selected}
        candidates = {node for node, _, _ in membership}
        global_nodes = sorted(candidates | {node for members in visible.values() for node in members})
        node_index = {node: index for index, node in enumerate(global_nodes)}
        index = torch.tensor([(node_index[node], edge) for edge, original in enumerate(selected)
                              for node in visible[original]], dtype=torch.long).reshape(-1, 2).t().contiguous()
        full = compile_hypergraph(len(global_nodes), index, len(selected))
        original_edges = [selected[source] for source in full.source_edge_ids]
        # Sequence length is independent of edge count and node budget. Actual
        # nodes/tokens/chunks are checked separately on each materialized packet.
        sequence = self.args.hir_max_seq_length
        while True:
            chunks = chunk_hir(full, sequence)[0]
            context = [chunk_members(full, chunk) for chunk in chunks]

            def packet(ids, materialize_features=True):
                return self._packet(full, chunks, context, ids, global_nodes, original_edges,
                                    membership, relation_pairs, relation_labels, features, masked_features, sequence,
                                    materialize_features)

            try:
                packets = self._packets(chunks, context, original_edges, relation_pairs, packet, limits, rng)
                break
            except LocalBudgetError:
                if sequence == 6:
                    raise
                sequence = max(6, sequence // 2)
        # Normalize EACH unique target across all its fragment appearances.
        # A long edge or repeatedly included context cannot dominate the loss.
        occurrences = {task: Counter() for task in ('membership', 'feature', 'relation')}
        for ids in packets:
            for task, keys in target_keys(packet(ids, False)).items():
                occurrences[task].update(keys)
        for ids in packets:
            batch = packet(ids)
            batch.target_weights = {task: torch.tensor([1 / occurrences[task][key] for key in keys])
                                    for task, keys in target_keys(batch).items()}
            batch.task_scales = {task: weights.sum().item() / max(1, len(occurrences[task]))
                                 for task, weights in batch.target_weights.items()}
            batch.task_scales['context'] = 1 / len(packets)
            positives = sum(key[2] == 1 for key in occurrences['membership'])
            batch.membership_pos_weight = (len(occurrences['membership']) - positives) / max(1, positives)
            yield batch

    def _packets(self, chunks, context, original_edges, relations, packet, limits, rng):
        """Pack connected chunks, co-locating relation endpoints with real readouts."""
        per_edge = {}
        for index, outputs in enumerate(context):
            for edge in outputs:
                per_edge.setdefault(original_edges[edge], []).append(index)
        nodes = [set().union(*outputs.values()) for outputs in context]
        uncovered, packets, paired = set(range(len(chunks))), [], set()

        def append(seed):
            ids = set(seed)
            if not self._fits(packet(tuple(sorted(ids)), False), limits):
                raise LocalBudgetError('Local budget cannot fit target chunks/queries; lower targets or increase packet budget')
            present_nodes = set().union(*(nodes[i] for i in ids))
            present_edges = {edge for i in ids for edge in context[i]}
            # Prefer genuinely shared nodes; fragments of the same edge are
            # also related. Never add disconnected edges just to fill tokens.
            remaining = set(uncovered) - ids
            while remaining:
                ranked = [(i, len(nodes[i] & present_nodes) + int(bool(set(context[i]) & present_edges))) for i in remaining]
                ranked = [(i, score) for i, score in ranked if score]
                rng.shuffle(ranked)
                ranked.sort(key=lambda item: -item[1])
                for index, _ in ranked:
                    remaining.remove(index)
                    trial = tuple(sorted(ids | {index}))
                    if self._fits(packet(trial, False), limits):
                        ids.add(index)
                        present_nodes.update(nodes[index])
                        present_edges.update(context[index])
                        break
                else:
                    break
            ids = tuple(sorted(ids))
            packets.append(ids)
            uncovered.difference_update(ids)
            outputs = {original_edges[edge] for i in ids for edge in context[i]}
            paired.update((first, second) for first, second in relations if first in outputs and second in outputs)

        for first, second in relations:
            if (first, second) not in paired:
                shared = set(per_edge[first]) & set(per_edge[second])
                seed = [min(shared)] if shared else [rng.choice(per_edge[first]), rng.choice(per_edge[second])]
                append(seed)
        while uncovered:
            append([min(uncovered)])
        return packets

    @staticmethod
    def _fits(packet, limits):
        tokens = sum(len(chunk.tokens) for chunk in packet.view.chunks + packet.view.relation_chunks)
        tokens += packet.membership_pairs[:, 0].unique().numel()
        return tokens <= limits.tokens and len(packet.view.x) <= limits.nodes and (
            len(packet.view.chunks) + len(packet.view.relation_chunks) <= limits.chunks) and all(
                len(chunk.tokens) <= limits.sequence_length for chunk in packet.view.chunks + packet.view.relation_chunks)

    def _packet(self, full, chunks, context, ids, global_nodes, original_edges,
                membership, relations, labels, feature_targets, masked_features, sequence, materialize_features=True):
        members, used_nodes, present = {}, set(), set()
        for index in ids:
            chunk = chunks[index]
            for edge, values in context[index].items():
                members.setdefault(edge, set()).update(values)
            used_nodes.update(token - full.space.node(0) for token in chunk.tokens
                              if full.space.node(0) <= token < full.space.edge(0))
            present.update(token - full.space.edge(0) for token in chunk.tokens
                           if full.space.edge(0) <= token < full.space.order(0))
        output_edges = {original_edges[edge] for edge in members}
        membership = [target for target in membership if target[1] in output_edges]
        usable_relations = [(pair, label) for pair, label in zip(relations, labels)
                            if all(edge in output_edges for edge in pair)]
        global_index = {node: index for index, node in enumerate(global_nodes)}
        used_nodes.update(global_index[node] for node, _, _ in membership)
        local_nodes = sorted(used_nodes)
        node_mapping = {old: new for new, old in enumerate(local_nodes)}
        global_ids = [global_nodes[node] for node in local_nodes]
        global_to_local = {node: i for i, node in enumerate(global_ids)}
        present = sorted(present)
        present_index = {edge: index for index, edge in enumerate(present)}
        incidence = [(node_mapping[node], present_index[edge]) for edge, values in members.items() for node in values]
        incidence = torch.tensor(incidence, dtype=torch.long).reshape(-1, 2).t().contiguous()
        program = compile_hypergraph(len(local_nodes), incidence, len(present))
        edge_mapping = {present[old]: new for new, old in enumerate(program.source_edge_ids)}
        original_to_compiled = {original_edges[old]: new for old, new in edge_mapping.items()}
        totals = {}
        for index in ids:
            for _, edge, weight in chunks[index].readouts:
                totals[edge] = totals.get(edge, 0.) + weight
        lowered = []
        for index in ids:
            chunk, tokens = chunks[index], []
            for token in chunk.tokens:
                if full.space.node(0) <= token < full.space.edge(0):
                    token = program.space.node(node_mapping[token - full.space.node(0)])
                elif full.space.edge(0) <= token < full.space.order(0):
                    token = program.space.edge(edge_mapping[token - full.space.edge(0)])
                elif token >= full.space.order(0):
                    token = program.space.order(token - full.space.order(0))
                tokens.append(token)
            lowered.append(HIRChunk(tuple(tokens), chunk.orders,
                                    tuple((position, edge_mapping[edge], weight / totals[edge])
                                          for position, edge, weight in chunk.readouts)))
        orders = {edge_mapping[old]: len(full.hyperedges[old]) for old in edge_mapping}
        # Reference-only definitions need IDs for token remapping but have no
        # readout/incidence. They must not become empty-edge relation examples.
        direct = tuple(EdgeIR(edge_mapping[old], orders[edge_mapping[old]],
                              FRAGMENT if len(values) < orders[edge_mapping[old]] else DIRECT,
                              add=tuple(sorted(node_mapping[node] for node in values)))
                       for old, values in sorted(members.items()))
        relation_program = replace(program, instructions=direct,
                                   tokens=tuple(token for ir in direct for token in ir.tokens(program.space)))
        relation_chunks = chunk_hir(relation_program, sequence)[0] if usable_relations else ()
        # Budget probes and target normalization need only shape/structure.
        # Avoid repeatedly copying thousands of feature columns while packing.
        x = self.graph.x[global_ids].clone() if materialize_features else torch.empty((len(global_ids), 0))
        masked = [i for i, node in enumerate(global_ids) if node in masked_features]
        x[masked] = 0
        actual_nodes = {global_nodes[node] for values in members.values() for node in values}
        feature_nodes = [node for node in feature_targets if node in actual_nodes]
        source_ids = tuple(original_edges[present[old]] for old in program.source_edge_ids)
        program = replace(program, source_edge_ids=source_ids)
        relation_program = replace(relation_program, source_edge_ids=source_ids)

        def pairs(values):
            return torch.tensor(values, dtype=torch.long).reshape(-1, 2)

        view = VisibleView(x, program, tuple(lowered), relation_program, relation_chunks,
                           program.to_hyperedge_index(), torch.tensor(masked, dtype=torch.long),
                           original_to_compiled, self.graph.dataset, torch.tensor(global_ids))
        return PretrainBatch(view,
                             pairs([(global_to_local[node], original_to_compiled[edge]) for node, edge, _ in membership]),
                             torch.tensor([label for _, _, label in membership]),
                             torch.tensor([global_to_local[node] for node in feature_nodes], dtype=torch.long),
                             self.graph.x[feature_nodes].clone() if materialize_features else torch.empty((len(feature_nodes), 0)),
                             pairs([(original_to_compiled[first], original_to_compiled[second])
                                    for (first, second), _ in usable_relations]),
                             torch.tensor([label for _, label in usable_relations], dtype=torch.long),
                             len(feature_targets) - len(feature_nodes))
