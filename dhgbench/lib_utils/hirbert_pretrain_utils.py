"""Label-free corruption, sampling, checkpoint and optimization utilities."""

import csv
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from hir import DIRECT, EdgeIR, chunk_hir, compile_hypergraph


TASKS = ('membership', 'feature', 'relation')
RELATIONS = ('containment', 'overlap', 'disjoint')


@dataclass
class UnlabeledGraph:
    x: torch.Tensor
    program: object
    dataset: str
    source_paths: dict = None

    @property
    def num_features(self):
        return self.x.shape[1]

    @property
    def num_nodes(self):
        return self.x.shape[0]


def unlabeled_graph(data, dataset):
    """The pretraining boundary: never inspect y, num_classes or class masks."""
    count = getattr(data, 'num_hyperedges', None)
    count = None if count is None else int(count)
    index = data.hyperedge_index.detach().cpu()
    if count is not None and index.numel() and int(index[1].max()) >= count:
        if len(index[1].unique()) != count:
            raise ValueError('Explicit empty edges require dense source IDs')
        count = None
    return UnlabeledGraph(data.x.detach().cpu(), compile_hypergraph(data.x.shape[0], index, count), dataset)


def edge_to_node(edges, incidence, num_nodes):
    nodes, edge_ids = incidence.to(edges.device)
    result = edges.new_zeros((num_nodes, edges.shape[-1])).index_add(0, nodes, edges[edge_ids])
    degree = torch.bincount(nodes, minlength=num_nodes)
    return result / degree.clamp_min(1).unsqueeze(-1), degree


def representation_stats(edges, incidence, num_nodes):
    nodes, degree = edge_to_node(edges.detach(), incidence, num_nodes)
    context = nodes[degree > 0]
    return {
        'edge_repr_variance': edges.detach().var(dim=0, unbiased=False).mean().item() if len(edges) else 0.0,
        'node_repr_variance': context.var(dim=0, unbiased=False).mean().item() if len(context) else 0.0,
    }


def direct_program(program):
    """Task C's alternate lowering only; the compiler and its output are untouched."""
    instructions = tuple(EdgeIR(i, len(members), DIRECT, add=members)
                         for i, members in enumerate(program.hyperedges))
    return replace(program, instructions=instructions,
                   tokens=tuple(token for ir in instructions for token in ir.tokens(program.space)))


@dataclass
class VisibleView:
    x: torch.Tensor
    program: object
    chunks: tuple
    relation_program: object
    relation_chunks: tuple
    incidence: torch.Tensor
    masked_nodes: torch.Tensor
    original_to_compiled: torch.Tensor
    dataset: str = ''
    node_ids: torch.Tensor = None


@dataclass
class PretrainBatch:
    view: VisibleView
    membership_pairs: torch.Tensor
    membership_labels: torch.Tensor
    feature_nodes: torch.Tensor
    feature_targets: torch.Tensor
    relation_pairs: torch.Tensor
    relation_labels: torch.Tensor
    skipped_feature_nodes: int
    task_scales: dict = None
    target_weights: dict = None
    membership_pos_weight: float = None


def split_examples(examples, fraction, rng):
    examples = list(examples)
    rng.shuffle(examples)
    if len(examples) < 2:
        raise ValueError('At least two self-supervised examples are required for train/validation')
    size = min(len(examples) - 1, max(1, round(len(examples) * fraction)))
    return examples[size:], examples[:size]


def relation_label(first, second):
    # Containment is symmetric and includes equality (including duplicate edges).
    if first <= second or second <= first:
        return 0
    return 1 if first & second else 2


def balanced_relation_bank(groups, edges, limit, fraction, rng, max_pairs=50000):
    """Reservoir sampling per class, without materializing all edge pairs.

    Equivalent pairs of duplicate groups never cross the SSL train/val split.
    Small graphs are enumerated; large graphs use bounded candidate sampling.
    """
    representatives = [ids[0] for ids in groups]
    sets = [frozenset(edges[index]) for index in representatives]
    banks, counts = [[] for _ in RELATIONS], [0] * len(RELATIONS)
    capacity = max(2, math.ceil(limit / (1 - fraction)))
    if len(sets) * (len(sets) + 1) // 2 <= max_pairs:
        candidates = ((i, j) for i in range(len(sets)) for j in range(i, len(sets))
                      if i != j or len(groups[i]) > 1)
    else:
        # Mix overlapping candidates, duplicate groups and uniform pairs. Avoid
        # quadratic enumeration for optional large graphs such as coauthor_dblp.
        postings = {}
        for i, members in enumerate(sets):
            for node in members:
                postings.setdefault(node, []).append(i)
        sampled = {(i, i) for i, group in enumerate(groups) if len(group) > 1}
        for attempt in range(max_pairs * 3):
            if len(sampled) >= max_pairs:
                break
            i = rng.randrange(len(sets))
            if attempt % 2 and sets[i]:
                node = rng.choice(tuple(sets[i]))
                j = rng.choice(postings[node])
            else:
                j = rng.randrange(len(sets))
            if i != j:
                sampled.add(tuple(sorted((i, j))))
        candidates = sorted(sampled)
    for i, j in candidates:
            first = sets[i]
            second = sets[j]
            label = relation_label(first, second)
            pair = (representatives[i], groups[j][1] if i == j else representatives[j], label)
            counts[label] += 1
            if len(banks[label]) < capacity:
                banks[label].append(pair)
            else:
                slot = rng.randrange(counts[label])
                if slot < capacity:
                    banks[label][slot] = pair
    train, valid = {}, {}
    for label, bank in enumerate(banks):
        # A class with one example cannot support held-out validation.
        if len(bank) >= 2:
            train[label], valid[label] = split_examples(bank, fraction, rng)
    if not train:
        raise ValueError('Relation prediction needs at least two distinct pairs in an available class')
    return train, valid, dict(zip(RELATIONS, counts))


class PretrainSampler:
    """Only this target sampler retains original structure/features.

    The model receives a fresh VisibleView and detached targets. Validation
    memberships/features are hidden in EVERY training view. Dynamic training
    corruptions are seeded by epoch/step; validation corruptions are fixed.
    """

    def __init__(self, graph, args):
        self.graph, self.args = graph, args
        self.tasks = set(args.tasks)
        self.edges = graph.program.hyperedges
        duplicates = {}
        for edge, members in enumerate(self.edges):
            duplicates.setdefault(members, []).append(edge)
        self.groups = list(duplicates.values())
        self.edge_group = {edge: group for group, ids in enumerate(self.groups) for edge in ids}
        rng = random.Random(args.pretrain_seed)
        self.train_members, self.val_members = [], []
        self.train_features, self.val_features = [], []
        self.train_relations, self.val_relations, self.relation_counts = {}, {}, {}
        if 'membership' in self.tasks:
            pool = [(node, ids[0]) for ids in self.groups for node in self.edges[ids[0]]
                    if len(self.edges[ids[0]]) < graph.num_nodes]
            self.train_members, self.val_members = split_examples(pool, args.ssl_val_ratio, rng)
        if 'feature' in self.tasks:
            active = sorted({node for edge in self.edges for node in edge})
            self.train_features, self.val_features = split_examples(active, args.ssl_val_ratio, rng)
        if 'relation' in self.tasks:
            self.train_relations, self.val_relations, self.relation_counts = balanced_relation_bank(
                self.groups, self.edges, args.relation_pairs_per_class, args.ssl_val_ratio, rng,
                getattr(args, 'relation_max_pairs', 50000))

    def feature_statistics(self):
        x = self.graph.x
        kind = self.args.feature_loss
        if kind == 'auto':
            kind = 'binary' if ((x == 0) | (x == 1)).all().item() else 'continuous'
        rows = x[self.train_features] if self.train_features else x
        positive = rows.sum(0)
        pos_weight = ((len(rows) - positive) / positive.clamp_min(1)).clamp(1, self.args.max_pos_weight)
        return kind, pos_weight, rows.mean(0), rows.std(0, unbiased=False).clamp_min(1e-4)

    def summary(self):
        return {
            'membership_train': len(self.train_members), 'membership_valid': len(self.val_members),
            'feature_train': len(self.train_features), 'feature_valid': len(self.val_features),
            'relation_available_pairs': self.relation_counts,
            'relation_train': {RELATIONS[k]: len(v) for k, v in self.train_relations.items()},
            'relation_valid': {RELATIONS[k]: len(v) for k, v in self.val_relations.items()},
            'duplicate_groups': sum(len(ids) > 1 for ids in self.groups),
        }

    def batch(self, split, epoch=0, step=0):
        training = split == 'train'
        rng = random.Random(self.args.pretrain_seed + (1000003 * epoch + 9176 * step if training else 7919))
        selected = []
        if self.train_members:
            selected = (rng.sample(self.train_members, max(1, math.ceil(len(self.train_members) * self.args.membership_mask_ratio)))
                        if training else list(self.val_members))
        hidden = set(self.val_members) | set(selected)
        removals = {}
        for node, edge in hidden:
            for duplicate in self.groups[self.edge_group[edge]]:
                removals.setdefault(duplicate, set()).add(node)
        # Original canonical edge IDs become SOURCE IDs, including emptied edges.
        pairs = [(node, edge) for edge, members in enumerate(self.edges) for node in members
                 if node not in removals.get(edge, ())]
        index = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()
        program = compile_hypergraph(self.graph.num_nodes, index, len(self.edges))
        inverse = torch.empty(len(self.edges), dtype=torch.long)
        for compiled, original in enumerate(program.source_edge_ids):
            inverse[original] = compiled
        chunks, _ = chunk_hir(program, self.args.hir_max_seq_length)
        relation = direct_program(program)
        relation_chunks = chunk_hir(relation, self.args.hir_max_seq_length)[0] if 'relation' in self.tasks else ()
        feature_nodes = (rng.sample(self.train_features, max(1, math.ceil(len(self.train_features) * self.args.feature_mask_ratio)))
                         if training and self.train_features else list(self.val_features))
        masked_nodes = sorted(set(self.val_features) | set(feature_nodes))
        device = self.graph.x.device
        visible_x = self.graph.x.clone()
        visible_x[masked_nodes] = 0
        visible_index = program.to_hyperedge_index().to(device)
        contextual = set(visible_index[0].tolist())
        usable = [node for node in feature_nodes if node in contextual]
        feature_ids = torch.tensor(usable, dtype=torch.long, device=device)
        positives = [(node, int(inverse[edge])) for node, edge in selected]
        negatives = []
        for _ in range(math.ceil(len(selected) * self.args.negative_ratio)):
            _, edge = rng.choice(selected)
            members = set(self.edges[edge])
            node = rng.randrange(self.graph.num_nodes)
            for attempt in range(32):
                if node not in members:
                    break
                node = rng.randrange(self.graph.num_nodes)
            if node in members:
                node = rng.choice([node for node in range(self.graph.num_nodes) if node not in members])
            negatives.append((node, int(inverse[edge])))
        relations = []
        bank = self.train_relations if training else self.val_relations
        if bank:
            count = min(self.args.relation_pairs_per_class, min(map(len, bank.values())))
            for entries in bank.values():
                relations.extend(rng.sample(entries, count))
            rng.shuffle(relations)

        def pair_tensor(values):
            return torch.tensor(values, dtype=torch.long, device=device).reshape(-1, 2)

        view = VisibleView(visible_x, program, chunks, relation, relation_chunks, visible_index,
                           torch.tensor(masked_nodes, dtype=torch.long, device=device), inverse.to(device))
        return PretrainBatch(
            view, pair_tensor(positives + negatives),
            torch.tensor([1.] * len(positives) + [0.] * len(negatives), device=device),
            feature_ids, self.graph.x[feature_ids].detach().clone(),
            pair_tensor([(int(inverse[first]), int(inverse[second])) for first, second, _ in relations]),
            torch.tensor([label for _, _, label in relations], dtype=torch.long, device=device),
            len(feature_nodes) - len(usable),
        )


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def training_arguments(args):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if not key.startswith('_')}


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def save_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def completed_metrics(directory, config, names, metrics_name='metrics.json'):
    directory = Path(directory)
    try:
        metrics = json.loads((directory / metrics_name).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if metrics.get('config') != config:
        raise ValueError(f'Different experiment configuration in {directory}; choose another --output')
    if metrics.get('completed') and all((directory / name).is_file() for name in names):
        return metrics
    return None


def rng_state():
    numpy = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate(), 'numpy': (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4])}


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))


def grad_norm(module):
    terms = [parameter.grad.detach().float().square().sum() for parameter in module.parameters()
             if parameter.grad is not None]
    return torch.stack(terms).sum().sqrt().item() if terms else 0.0


def make_scheduler(optimizer, kind, epochs, warmup):
    def factor(step):
        if kind == 'none':
            return 1.0
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, epochs - warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress)) if kind == 'cosine' else 1 - progress
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def write_history(path, history):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def check_encoder_compatibility(saved, config, features, dataset):
    if saved.get('kind') != 'hirbert_pretrained_encoder':
        raise ValueError('Expected a HIRBERT pretrained encoder checkpoint')
    if config != saved['model_config']:
        raise ValueError('Incompatible encoder model configuration')
    if 'shared_encoder_state_dict' in saved:
        dimensions = saved['feature_dimensions']
        if dataset in dimensions and dimensions[dataset] != features:
            raise ValueError('Incompatible dataset adapter input feature dimension')
    elif features != saved['num_features'] or (dataset is not None and saved['dataset_info']['dataset'] != dataset):
        raise ValueError('Incompatible input feature dimension or pretraining dataset')


def load_encoder(model, path, dataset=None):
    """Strict, complete transfer; caller MUST reset its classifier BEFORE this."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if dataset is None and len(saved.get('dataset_adapters', {})) > 1:
        raise ValueError('Specify dataset when loading a multi-graph feature adapter')
    dataset = dataset or saved['dataset_info'].get('dataset')
    check_encoder_compatibility(saved, model.model_config, model.num_features, dataset)
    adapter_loaded = False
    if 'shared_encoder_state_dict' in saved:
        state = dict(saved['shared_encoder_state_dict'])
        adapter_loaded = dataset in saved['dataset_adapters'] and saved['adapter_trained_samples'].get(dataset, 0) > 0
        adapter = (saved['dataset_adapters'][dataset] if adapter_loaded
                   else model.encoder.feature_projection.state_dict())
        state.update({'feature_projection.' + key: value for key, value in adapter.items()})
        loaded = list(saved['shared_encoder_state_dict'])
        if adapter_loaded:
            loaded += ['feature_projection.' + key for key in adapter]
    else:
        state, loaded, adapter_loaded = saved['encoder_state_dict'], list(saved['encoder_state_dict']), True
    model.encoder.load_state_dict(state, strict=True)
    return {'source': str(Path(path).resolve()), 'loaded_keys': loaded,
            'missing_keys': [], 'unexpected_keys': [], 'best_epoch': saved['best_epoch'],
            'pretrain_seed': saved['seed'], 'pretrain_config': saved['pretrain_config'],
            'adapter_dataset': dataset, 'adapter_loaded': adapter_loaded,
            'adapter_initialization': 'pretrained' if adapter_loaded else 'random_unseen_or_untrained_dataset',
            'failed_pretrain_datasets': saved.get('failed_datasets', [])}
