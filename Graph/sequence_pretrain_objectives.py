"""Discardable HIR-SP heads, per-graph normalized losses, and metrics."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

TASKS = dict(node=4, triangle=2, bridge=2, components=3, graph_triangle=2, bipartite=2)
LOSS_NAMES = ("node", "triangle", "bridge", "components", "graph_triangle", "bipartite",
              "node_view", "edge_view", "contrast")


def mlp(dimension, outputs):
    return nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, outputs))


class StructuralHeads(nn.Module):
    def __init__(self, hidden_dim, projection_dim=128):
        super().__init__()
        self.node = mlp(3*hidden_dim, 4)
        self.edge = nn.ModuleDict({name: mlp(hidden_dim, 2) for name in ("triangle", "bridge")})
        self.graph = nn.ModuleDict({name: mlp(hidden_dim, classes) for name, classes in TASKS.items()
                                   if name in ("components", "graph_triangle", "bipartite")})
        self.projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, projection_dim))

    def node_logits(self, nodes, pairs):
        u, v = nodes[pairs[:, 0]], nodes[pairs[:, 1]]
        return self.node(torch.cat((u+v, (u-v).abs(), u*v), dim=-1))


def class_weights(records):
    """Fixed training-only inverse-frequency weights; missing classes get 0."""
    counts = {task: [0]*classes for task, classes in TASKS.items() if task != "node"}
    for item in records:
        for task in ("triangle", "bridge"):
            for y in getattr(item, task):
                counts[task][y] += 1
        for task, y in item.properties.items():
            if y is not None:
                counts[task][y] += 1
    weights = {}
    for task, distribution in counts.items():
        present = sum(n > 0 for n in distribution)
        weights[task] = [sum(distribution)/(present*n) if n else 0. for n in distribution]
    return weights, counts


def supervised_losses(heads, output, item, pairs, labels, weights, meter, coefficients):
    device = output.graph_embedding.device
    losses = {}

    def record(task, logits, truth):
        if not len(truth):
            return
        target = torch.tensor(truth, device=device, dtype=torch.long)
        # Validation-only unseen classes must still have a defined loss. Zero
        # training weight applies only to training; evaluation is unweighted.
        weight = torch.tensor(weights[task], device=device, dtype=torch.float32) if weights and task in weights else None
        # reduction='none' preserves graph-class weighting even for the one
        # label per graph; default weighted CE would cancel its own weight.
        loss = F.cross_entropy(logits.float(), target, weight=weight, reduction="none").mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite {task} loss for {item.path}")
        losses[task] = loss
        meter.add_prediction(task, logits, target)

    if coefficients["node"] > 0 and pairs:
        indices = torch.tensor(pairs, device=device, dtype=torch.long)
        record("node", heads.node_logits(output.node_embeddings, indices), labels)
    if coefficients["edge"] > 0 and item.program.num_edges:
        for task in ("triangle", "bridge"):
            record(task, heads.edge[task](output.edge_embeddings), getattr(item, task))
    for task, y in item.properties.items():
        if coefficients["graph"] > 0 and y is not None:
            record(task, heads.graph[task](output.graph_embedding.unsqueeze(0)), [y])
    return losses


def contrastive_loss(heads, first, second, items, temperature):
    """Symmetric cross-view InfoNCE, excluding all uncertified negatives."""
    a = F.normalize(heads.projection(torch.stack(first)).float(), dim=-1)
    b = F.normalize(heads.projection(torch.stack(second)).float(), dim=-1)
    scores = a @ b.t() / temperature
    losses, negatives = [], []
    for i, item in enumerate(items):
        safe = [j for j, other in enumerate(items) if item.signature != other.signature]
        negatives.append(len(safe))
        if not safe:
            continue
        columns = [i] + safe
        losses.extend((-F.log_softmax(scores[i, columns], dim=0)[0],
                       -F.log_softmax(scores[columns, i], dim=0)[0]))
    return (torch.stack(losses).mean() if losses else None), negatives


def view_loss(a, b):
    return (1-F.cosine_similarity(a.float(), b.float(), dim=-1)).mean() if len(a) else None


@torch.no_grad()
def representation_summary(values, max_rows=256):
    """Bounded diagnostics; rank is centered entropy rank, never a probe."""
    values = values.detach().float()[:max_rows]
    if not len(values):
        return dict(rows=0, variance=None, mean_cosine=None, effective_rank=None)
    centered = values-values.mean(0)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    p = singular[singular > 0]/total if total > 0 else singular[:0]
    rank = float((-(p*p.log()).sum()).exp()) if len(p) else 0.
    valid = values[values.norm(dim=-1) > 1e-12]
    if len(valid) > 1:
        normalized = F.normalize(valid, dim=-1)
        cosine = float((normalized.sum(0).square().sum()-len(valid))/(len(valid)*(len(valid)-1)))
    else:
        cosine = None
    return dict(rows=len(values), variance=float(centered.square().mean()), mean_cosine=cosine,
                effective_rank=rank, row_cap=max_rows)


class EpochMetrics:
    def __init__(self):
        self.losses = {name: [] for name in LOSS_NAMES + ("total", "edge", "graph", "view")}
        self.predictions = {name: [] for name in TASKS}
        self.targets = {name: [] for name in TASKS}
        self.confusions = {name: torch.zeros((classes, classes), dtype=torch.long)
                           for name, classes in TASKS.items()}
        self.valid_samples = {name: 0 for name in LOSS_NAMES}
        self.skipped_batches = 0
        self.skipped_graphs = 0
        self.negative_counts = []
        self.representations = {name: [] for name in ("node", "edge", "graph")}
        self.node_candidate_counts = [0]*4

    def add_prediction(self, task, logits, labels):
        classes = TASKS[task]
        truth = labels.detach().cpu()
        prediction = logits.detach().argmax(-1).cpu()
        self.confusions[task] += torch.bincount(truth*classes+prediction,
                                               minlength=classes*classes).reshape(classes, classes)
        # Multiclass metrics need only confusion counts. Binary AP retains
        # compact CPU tensors, not millions of Python probability objects.
        if classes == 2:
            self.predictions[task].append(logits.detach().float().softmax(-1)[:, 1].cpu())
            self.targets[task].append(truth)
        self.valid_samples[task] += len(labels)

    def add_loss(self, name, value):
        if value is not None:
            self.losses[name].append(float(value.detach()))

    def add_representation(self, output):
        for name, values in (("node", output.node_embeddings), ("edge", output.edge_embeddings),
                             ("graph", output.graph_embedding.unsqueeze(0))):
            remaining = 256-sum(len(x) for x in self.representations[name])
            if remaining > 0:
                self.representations[name].append(values[:remaining].detach().float().cpu())

    def result(self):
        result = {name+"_loss": sum(v)/len(v) if v else None for name, v in self.losses.items()}
        for name, classes in TASKS.items():
            matrix = self.confusions[name]
            counts = matrix.sum(1).tolist()
            if not sum(counts):
                result[name] = dict(counts=counts, accuracy=None, macro_accuracy=None,
                                    per_class_accuracy=[None]*classes, macro_f1=None, f1=None, ap=None)
                continue
            per_class = [int(matrix[c, c])/counts[c] if counts[c] else None for c in range(classes)]
            present = [c for c in range(classes) if counts[c]]
            f1 = [2*int(matrix[c, c])/int(matrix[c].sum()+matrix[:, c].sum())
                  if int(matrix[c].sum()+matrix[:, c].sum()) else 0. for c in range(classes)]
            result[name] = dict(counts=counts, accuracy=float(matrix.diag().sum())/sum(counts),
                macro_accuracy=sum(per_class[c] for c in present)/len(present), per_class_accuracy=per_class,
                macro_f1=sum(f1[c] for c in present)/len(present), f1=f1[1] if classes == 2 else None,
                ap=float(average_precision_score(torch.cat(self.targets[name]).numpy(),
                                                torch.cat(self.predictions[name]).numpy()))
                if classes == 2 and all(counts) else None,
                macro_policy="ground-truth present classes only; absent-class accuracy is null")
        result.update(valid_samples=self.valid_samples, loss_observation_counts={k: len(v) for k, v in self.losses.items()},
                      skipped_batches=self.skipped_batches, skipped_graphs=self.skipped_graphs,
                      contrast_no_negative_anchors=sum(n == 0 for n in self.negative_counts),
                      contrast_effective_graphs=sum(n > 0 for n in self.negative_counts),
                      contrast_directional_anchors=2*sum(n > 0 for n in self.negative_counts),
                      contrast_negative_count=sum(self.negative_counts),
                      mean_contrast_negatives=sum(self.negative_counts)/len(self.negative_counts)
                      if self.negative_counts else None, node_candidate_counts=self.node_candidate_counts)
        result["aggregation"] = "subtask losses: equal graph means; combined/contrast/total losses: batch means; metrics: valid samples"
        result["skip_reasons"] = dict(no_enabled_objective_with_valid_samples=self.skipped_batches,
            contrast_graphs_without_certified_negatives=sum(n == 0 for n in self.negative_counts))
        result["representation"] = {name: representation_summary(torch.cat(rows)) if rows else None
                                     for name, rows in self.representations.items()}
        return result


def combine_losses(losses, coefficients):
    """Subtask means are already per graph; unavailable losses remain None."""
    def summed(names):
        values = [losses[n] for n in names if losses.get(n) is not None]
        return sum(values) if values else None
    groups = dict(node=losses.get("node"), edge=summed(("triangle", "bridge")),
                  graph=summed(("components", "graph_triangle", "bipartite")),
                  contrast=losses.get("contrast"), view=summed(("node_view", "edge_view")))
    active = [value*coefficients[name] for name, value in groups.items()
              if value is not None and coefficients[name] > 0]
    return (sum(active) if active else None), groups


def cosine_schedule(step, total_steps, warmup_ratio):
    warmup = int(total_steps*warmup_ratio)
    if step < warmup:
        return (step+1)/max(warmup, 1)
    return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, total_steps-warmup))))
