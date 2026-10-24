"""Independent HIR-SP V2 frozen linear probes on ordinary simple graphs.

Run from the repository root (epoch values default to the logged, zero-based field):
python -m Graph.sequence_frozen_probe --checkpoint Graph/runs/epoch18/encoder_best.pt --data_dir Graph/datasets/synthetic_v1 --output_dir Graph/runs/frozen_probe --seeds 0 1 2

Only this entry point trains probes. Encoder inputs are the unmodified topology,
with x=None. Labels, feature scaling and target scaling never enter the encoder.
Validation reuses pretraining validation graphs and therefore measures transfer
to NEW TASKS, not unseen graphs. Only test/ood results measure held-out graphs.
Dependencies are the existing HIR-SP torch, transformers, networkx and sklearn.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import statistics

import networkx as nx
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
import torch
from torch import nn
from torch.nn import functional as F

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Graph"

from .data import FAMILIES, split_paths
from .sequence_encoder import (
    ARCHITECTURE_VERSION, INITIALIZATION_VERSION, HIRSequenceEncoder,
)
from .sequence_pretrain_data import load_record
from .sequence_tokenizer import HIRSequenceTokenizer


KNOWN = ("ba", "ws", "sbm")
# K-core is ordinal and has an open-ended support: regression avoids inventing
# arbitrary bins and permits evaluation of core indices absent from probe train.
TASKS = {
    "node_kcore": ("node", "regression"),
    "node_articulation": ("node", "classification"),
    "edge_betweenness": ("edge", "regression"),
    "graph_efficiency": ("graph", "regression"),
    "graph_four_cycles": ("graph", "regression"),
}
CONFIG_KEYS = {
    "size", "id_encoding", "digit_dim", "feature_dim", "max_length",
    "attention_dropout", "architecture_version", "initialization_version",
}


def dump(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def relative_entry(manifest, entry):
    """Relocate the recorded dataset root without guessing by basename."""
    source = str(entry["path"]).replace("\\", "/")
    root = str(manifest["root"]).replace("\\", "/").rstrip("/")
    if not source.lower().startswith(root.lower() + "/"):
        raise ValueError(f"manifest path is outside its recorded root: {source}")
    relative = PurePosixPath(source[len(root) + 1:])
    if relative.is_absolute() or any(p in ("..", ".") for p in relative.parts):
        raise ValueError(f"invalid manifest path: {source}")
    if len(relative.parts) != 3 or relative.parts[1] != entry["family"]:
        raise ValueError(f"expected split/family/file.pt in manifest: {source}")
    return relative.as_posix()


def inspect_checkpoint(path, expected_epoch):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: checkpoint must be a dictionary")
    config = checkpoint.get("encoder_config")
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"{path}: missing or unknown V2 encoder configuration fields")
    if (checkpoint.get("architecture_version") != ARCHITECTURE_VERSION
            or config["architecture_version"] != ARCHITECTURE_VERSION
            or checkpoint.get("initialization_version") != INITIALIZATION_VERSION
            or config["initialization_version"] != INITIALIZATION_VERSION):
        raise ValueError(f"{path}: requires HIRSequenceEncoder V2 with scaled ID initialization")
    if (config["size"] not in ("small", "base")
            or config["id_encoding"] not in ("hex", "lookup")
            or type(config["digit_dim"]) is not int or config["digit_dim"] < 1
            or type(config["max_length"]) is not int or config["max_length"] < 1
            or config["feature_dim"] is not None
            or not isinstance(config["attention_dropout"], (float, int))
            or not 0 <= config["attention_dropout"] < 1):
        raise ValueError(f"{path}: invalid or non-topology-only HIR-SP configuration")
    tokenizer = HIRSequenceTokenizer(config["max_length"])
    if checkpoint.get("tokenizer_config") != tokenizer.config:
        raise ValueError(f"{path}: tokenizer configuration mismatch")
    if checkpoint.get("pretraining_version") != "hir_sp_v1":
        raise ValueError(f"{path}: only HIR-SP v1 weights are accepted")
    if type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] != expected_epoch:
        raise ValueError(f"{path}: expected saved epoch={expected_epoch}, got {checkpoint.get('epoch')}")
    if not isinstance(checkpoint.get("encoder_state_dict"), dict):
        raise ValueError(f"{path}: missing encoder_state_dict")
    manifest = checkpoint.get("data_manifest")
    if not isinstance(manifest, dict) or manifest.get("split_mode") != "existing":
        raise ValueError(f"{path}: requires an explicit existing train/val split manifest; no fallback split")
    normalized = {}
    for split in ("train", "val", "test", "ood"):
        entries = manifest.get("splits", {}).get(split)
        if not isinstance(entries, list):
            raise ValueError(f"{path}: missing manifest split {split}")
        normalized[split] = sorted(
            [dict(entry, path=relative_entry(manifest, entry)) for entry in entries],
            key=lambda entry: entry["path"],
        )
        paths = [e["path"] for e in normalized[split]]
        if len(set(paths)) != len(paths) or any(PurePosixPath(p).parts[0] != split for p in paths):
            raise ValueError(f"{path}: invalid/duplicate manifest split paths")
    return dict(path=str(path.resolve()), epoch=checkpoint["epoch"], config=config,
                tokenizer_config=checkpoint["tokenizer_config"], splits=normalized)


def prepare_records(root, metadata):
    """Use ALL recorded training/validation graphs as the held-out exclusion set.

    Exact isomorphism also removes relabeled copies, with invariant buckets
    used only to reduce comparisons. Equal invariants alone never imply equality.
    No random split, node/edge split within a graph, or ER fitting is permitted.
    """
    root = root.resolve()
    maximum = metadata["config"]["max_length"]
    records = {s: [] for s in ("train", "val", "test", "ood")}
    buckets, removed = {}, []

    def add(item, split):
        bucket = buckets.setdefault(item.signature, [])
        duplicate = next((old for old in bucket if nx.is_isomorphic(item.graph, old.graph)), None)
        if duplicate:
            removed.append(dict(path=item.path, split=split, duplicate_of=duplicate.path))
            return False
        bucket.append(item)
        return True

    for split in ("train", "val"):
        for entry in metadata["splits"][split]:
            item = load_record(root / entry["path"], maximum)
            if item.family != entry["family"] or item.family not in FAMILIES:
                raise ValueError(f"manifest family mismatch: {item.path}")
            actual = dict(nodes=item.program.num_nodes, edges=item.program.num_edges,
                          token_length=5 * item.program.num_edges + nx.number_of_isolates(item.graph) + 1,
                          graph_labels=item.properties,
                          triangle_counts=[item.triangle.count(c) for c in (0, 1)],
                          bridge_counts=[item.bridge.count(c) for c in (0, 1)])
            if any(entry.get(k) != value for k, value in actual.items()):
                raise ValueError(f"graph no longer agrees with pretraining manifest: {item.path}")
            if split == "train" and item.family == "er":
                raise ValueError("checkpoint contains ER pretraining graphs; ER cannot be claimed as unseen")
            if not add(item, split):
                raise ValueError("recorded train/val contain isomorphic overlap; refusing to change the original split")
            if item.family in KNOWN:
                records[split].append(item)
    for split in ("train", "val"):
        if any(not any(i.family == family for i in records[split]) for family in KNOWN):
            raise ValueError(f"recorded {split} must contain BA, WS and SBM")
    # Discover held-out files even when the pretraining manifest capped its
    # unused test/ood lists. ER validation graphs are never used as test data.
    for split in ("test", "ood"):
        for path in split_paths(root, split):
            item = load_record(path, maximum)
            if add(item, split):
                records[split].append(item)
    if not any(i.family == "er" for s in ("test", "ood") for i in records[s]):
        raise ValueError("no independent ER test/ood graphs remain after exact overlap exclusion")
    if not any(i.family in KNOWN for s in ("test", "ood") for i in records[s]):
        raise ValueError("no independent BA/WS/SBM test/ood graphs remain")
    manifest = dict(
        root=str(root), duplicates_removed=removed,
        exclusion="all checkpoint train/val graphs, including ER val; exact isomorphism",
        validation_role="pretraining-seen graphs, new labels; probe selection only",
        er_role="test/ood only; never probe fitting, normalization or selection",
        splits={s: [dict(path=i.path, family=i.family, nodes=i.program.num_nodes,
                         edges=i.program.num_edges) for i in items] for s, items in records.items()},
    )
    return records, manifest


def structural_labels(item):
    graph = item.graph
    core = nx.core_number(graph)
    cuts = set(nx.articulation_points(graph))
    between = {tuple(sorted(edge)): value for edge, value in
               nx.edge_betweenness_centrality(graph, normalized=True, weight=None).items()}
    # Every unordered pair of opposite vertices contributes choose(common, 2).
    # Each undirected simple cycle of length four has two opposite pairs.
    # Chords are allowed: K4 contains three four-cycles, not zero induced ones.
    opposite = Counter()
    for node in graph:
        neighbors = sorted(graph[node])
        for offset, u in enumerate(neighbors):
            for v in neighbors[offset + 1:]:
                opposite[u, v] += 1
    cycles = sum(c * (c - 1) // 2 for c in opposite.values()) // 2
    return {
        "node_kcore": [core[u] for u in range(item.program.num_nodes)],
        "node_articulation": [int(u in cuts) for u in range(item.program.num_nodes)],
        "edge_betweenness": [between[tuple(sorted(e.endpoints))] for e in item.program.instructions],
        "graph_efficiency": [nx.global_efficiency(graph)] if graph.number_of_nodes() >= 2 else [],
        "graph_four_cycles": [cycles],
    }


def distribution(values, task):
    y = np.asarray(values, dtype=np.float64)
    result = dict(n=len(y))
    if TASKS[task][1] == "classification":
        counts = [int((y == c).sum()) for c in (0, 1)]
        return dict(result, counts=counts, missing_classes=[c for c in (0, 1) if not counts[c]])
    result.update(missing_classes=None, class_note="regression; no finite class vocabulary")
    if len(y):
        result.update(min=float(y.min()), max=float(y.max()), mean=float(y.mean()),
                      std=float(y.std()), zero_count=int((y == 0).sum()),
                      quantiles=np.quantile(y, [0, .25, .5, .75, 1]).tolist())
        if task in ("node_kcore", "graph_four_cycles"):
            result["integer_counts"] = dict(sorted(Counter(map(int, y)).items()))
    return result


@dataclass
class Rows:
    x: np.ndarray
    y: torch.Tensor
    families: list
    indices: np.ndarray

    def features(self, selection):
        """Read only the selected rows; tasks share the same level's memmap."""
        if isinstance(selection, torch.Tensor):
            selection = selection.numpy()
        return torch.from_numpy(self.x[self.indices[selection]])


def extract(encoder, records, labels, cache_dir, binding):
    """Write each level once, one graph at a time, without a CPU feature list.

    Caches are created exclusively in this run's directory and never reopened
    from another run. The binding records the source checkpoint, encoding
    configuration, actual split manifest and (for random weights) encoder seed.
    """
    cache_dir.mkdir(parents=True, exist_ok=False)
    dump(cache_dir / "binding.json", binding)
    encoder.requires_grad_(False)
    encoder.eval()
    dimension = encoder.hidden_dim
    result = {}
    for split, items in records.items():
        sizes = dict(node=sum(i.program.num_nodes for i in items),
                     edge=sum(i.program.num_edges for i in items), graph=len(items))
        maps = {
            level: np.lib.format.open_memmap(cache_dir / f"{split}_{level}.npy", mode="w+",
                                            dtype=np.float32, shape=(count, dimension))
            if count else np.empty((0, dimension), dtype=np.float32)
            for level, count in sizes.items()
        }
        counts = {task: sum(len(labels[i.path][task]) for i in items) for task in TASKS}
        result[split] = {
            task: Rows(maps[level], torch.empty(counts[task], dtype=torch.float64), [],
                       np.empty(counts[task], dtype=np.int64))
            for task, (level, _) in TASKS.items()
        }
        offsets = dict.fromkeys(sizes, 0)
        positions = dict.fromkeys(TASKS, 0)
        for item in items:
            with torch.inference_mode():
                output = encoder(item.program, x=None)
            representations = dict(node=output.node_embeddings, edge=output.edge_embeddings,
                                   graph=output.graph_embedding.unsqueeze(0))
            lengths = dict(node=item.program.num_nodes, edge=item.program.num_edges, graph=1)
            for level, representation in representations.items():
                feature = representation.detach().float().cpu().numpy()
                if feature.shape != (lengths[level], dimension) or not np.isfinite(feature).all():
                    raise ValueError(f"invalid/alignment-mismatched {level} embeddings: {item.path}")
                start = offsets[level]
                maps[level][start:start + len(feature)] = feature
            for task, (level, _) in TASKS.items():
                target = labels[item.path][task]
                if not target:
                    continue
                if len(target) != lengths[level]:
                    raise ValueError(f"mismatched {task} labels: {item.path}")
                rows, start = result[split][task], positions[task]
                stop = start + len(target)
                rows.indices[start:stop] = np.arange(offsets[level], offsets[level] + len(target))
                rows.y[start:stop] = torch.as_tensor(target, dtype=torch.float64)
                rows.families.extend([item.family] * len(target))
                positions[task] = stop
            for level in offsets:
                offsets[level] += lengths[level]
            del output, representations, representation, feature
        for feature_map in maps.values():
            if isinstance(feature_map, np.memmap):
                feature_map.flush()
    return result


def feature_statistics(rows, batch_size):
    """Population moments from probe TRAIN only, with bounded feature memory."""
    count = 0
    mean = torch.zeros(rows.x.shape[1], dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    for start in range(0, len(rows.y), batch_size):
        values = rows.features(slice(start, start + batch_size)).double()
        variance, batch_mean = torch.var_mean(values, dim=0, unbiased=False)
        size = len(values)
        delta = batch_mean - mean
        total = count + size
        m2 += variance * size + delta.square() * (count * size / total)
        mean += delta * (size / total)
        count = total
    std = (m2 / count).sqrt().float()
    return mean.float(), torch.where(std >= 1e-6, std, torch.ones_like(std))


def predict_batches(head, rows, mean, std, batch_size):
    """Keep only scalar predictions in RAM, rather than normalized embeddings."""
    predicted = torch.empty(len(rows.y))
    for start in range(0, len(rows.y), batch_size):
        selection = slice(start, start + batch_size)
        predicted[selection] = head((rows.features(selection) - mean) / std).squeeze(-1)
    return predicted


def metrics(truth, prediction, classification):
    y, p = np.asarray(truth), np.asarray(prediction)
    result = dict(n=len(y), not_applicable={})
    names = (("accuracy", "balanced_accuracy", "macro_f1", "positive_f1", "auroc", "ap")
             if classification else ("mae", "rmse", "r2", "pearson", "spearman"))
    if not len(y):
        result.update({name: None for name in names})
        result["not_applicable"] = {name: "no samples" for name in names}
        return result
    if not np.isfinite(p).all():
        raise FloatingPointError("nonfinite probe predictions")
    if classification:
        predicted = (p >= .5).astype(int)
        counts = [int((y == c).sum()) for c in (0, 1)]
        result.update(accuracy=float(accuracy_score(y, predicted)), counts=counts,
                      missing_classes=[c for c in (0, 1) if not counts[c]],
                      confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1]).tolist())
        if min(counts) == 0:
            for name in names[1:]:
                result[name] = None
                result["not_applicable"][name] = "requires both ground-truth classes"
        else:
            result.update(
                balanced_accuracy=float(np.mean([(predicted[y == c] == c).mean() for c in (0, 1)])),
                macro_f1=float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
                positive_f1=float(f1_score(y, predicted, zero_division=0)),
                auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
            )
    else:
        result.update(mae=float(mean_absolute_error(y, p)), rmse=math.sqrt(mean_squared_error(y, p)))
        variable_truth = len(y) >= 2 and np.ptp(y) > 0
        if variable_truth:
            result["r2"] = float(r2_score(y, p))
        else:
            result["r2"] = None
            result["not_applicable"]["r2"] = "fewer than two samples or constant target"
        for name in ("pearson", "spearman"):
            if not variable_truth or np.ptp(p) == 0:
                result[name] = None
                result["not_applicable"][name] = "fewer than two samples or constant target/prediction"
            else:
                # Average ranks preserve ties, including repeated zero labels.
                a, b = (average_ranks(y), average_ranks(p)) if name == "spearman" else (y, p)
                result[name] = float(np.corrcoef(a, b)[0, 1])
    return result


def average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def fit_probe(data, task, seed, args):
    """One affine layer, train-only scaling, fixed AdamW and minibatch order.

    Each task selects its epoch by equal-family BA/WS/SBM validation loss.
    Four-cycle count uses an affine probe in log1p target space; reported
    metrics are inverse-transformed to original counts, with no clipping.
    All other regression targets use identity followed by train-only z-score.
    """
    train, val = data["train"][task], data["val"][task]
    classification = TASKS[task][1] == "classification"
    if not len(train.y) or not len(val.y):
        return None, "empty probe training or validation task"
    if classification and len(torch.unique(train.y)) != 2:
        return None, "probe training has a missing binary class"
    for family in KNOWN:
        if family not in val.families:
            return None, f"no validation labels for {family}"
    torch.manual_seed(seed)
    # Probe fitting is CPU-only; the encoder device affects extraction only.
    # This also keeps the same probe arithmetic for CUDA and CPU encoders.
    head = nn.Linear(train.x.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    mean, std = feature_statistics(train, args.batch_size)
    raw_target = torch.log1p(train.y) if task == "graph_four_cycles" else train.y
    target_mean = raw_target.mean() if not classification else torch.tensor(0.)
    target_std = raw_target.std(unbiased=False) if not classification else torch.tensor(1.)
    if target_std < 1e-6:
        target_std = torch.ones_like(target_std)
    ty = ((raw_target - target_mean) / target_std).float()
    raw_val = torch.log1p(val.y) if task == "graph_four_cycles" else val.y
    vy = ((raw_val - target_mean) / target_std).float()
    loss_fn = F.binary_cross_entropy_with_logits if classification else F.mse_loss
    family_indices = [torch.tensor([i for i, f in enumerate(val.families) if f == family]) for family in KNOWN]
    best, selected_epoch, state = math.inf, None, None
    history = []
    for epoch in range(args.probe_epochs):
        head.train()
        order = torch.randperm(len(ty), generator=generator)
        total = 0.
        for indices in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            features = (train.features(indices) - mean) / std
            loss = loss_fn(head(features).squeeze(-1), ty[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite probe loss: {task}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        head.eval()
        with torch.no_grad():
            vp = predict_batches(head, val, mean, std, args.batch_size)
            score = sum(loss_fn(vp[idx], vy[idx]).item() for idx in family_indices) / len(KNOWN)
        if not math.isfinite(score):
            raise FloatingPointError(f"nonfinite validation loss: {task}")
        history.append(dict(epoch=epoch, train_loss=total / len(ty), validation_family_macro_loss=score))
        if score < best:
            best, selected_epoch = score, epoch
            state = {name: value.detach().clone() for name, value in head.state_dict().items()}
    head.load_state_dict(state, strict=True)
    results = []
    with torch.no_grad():
        for split in ("val", "test", "ood"):
            rows = data[split][task]
            predicted = predict_batches(head, rows, mean, std, args.batch_size)
            if classification:
                predicted = predicted.sigmoid()
            else:
                predicted = predicted.double() * target_std + target_mean
                if task == "graph_four_cycles":
                    predicted = predicted.double().expm1()
            for family in FAMILIES:
                indices = [i for i, f in enumerate(rows.families) if f == family]
                result = metrics(rows.y[indices].numpy(), predicted[indices].numpy(), classification)
                results.append(dict(split=split, family=family, metrics=result))
    artifact = dict(state_dict=state, feature_mean=mean, feature_std=std,
                    target_mean=target_mean, target_std=target_std, task=task,
                    target_transform="log1p" if task == "graph_four_cycles" else "identity",
                    selected_epoch=selected_epoch, validation_loss=best, seed=seed)
    return dict(results=results, artifact=artifact, history=history), None


def summarize(rows):
    """Across-seed means/SD plus paired differences on identical seeds.

    Never pool incompatible task metrics; each family/split/task is separate.
    Positive oriented_gain means the first model is better. SD is descriptive,
    not a claim of statistical significance for a small number of seeds.
    """
    groups = {}
    for row in rows:
        for metric, value in row["metrics"].items():
            if metric == "n" or isinstance(value, (dict, list)):
                continue
            key = row["split"], row["family"], row["task"], metric
            groups.setdefault(key, {}).setdefault(row["model"], {})[row["seed"]] = value
    summaries, comparisons = [], []

    def stats(values):
        return dict(mean=statistics.mean(values) if values else None,
                    std=statistics.stdev(values) if len(values) > 1 else None, valid_seeds=len(values))

    for (split, family, task, metric), models in sorted(groups.items()):
        common = dict(split=split, family=family, task=task, metric=metric)
        for model, values in models.items():
            summaries.append(dict(common, model=model, **stats([v for v in values.values() if v is not None])))
        for left, right in (("pretrained_v2", "random_v2"),):
            a, b = models.get(left, {}), models.get(right, {})
            seeds = sorted(s for s in a.keys() & b.keys() if a[s] is not None and b[s] is not None)
            differences = [a[s] - b[s] for s in seeds]
            sign = -1 if metric in ("mae", "rmse") else 1
            comparisons.append(dict(common, left=left, right=right, paired_seeds=seeds,
                                    raw_difference=stats(differences),
                                    oriented_gain=stats([sign * d for d in differences])))
    return dict(per_model=summaries, paired_comparisons=comparisons,
                note="SD requires >=2 seeds; gains compare paired seeds, not statistical significance")


def write_csv(path, rows):
    fields = ["model", "seed", "task", "split", "family", "selected_epoch", "status", "metric", "value", "reason"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for metric, value in row["metrics"].items():
                if isinstance(value, (dict, list)):
                    continue
                writer.writerow(dict({k: row[k] for k in fields[:7]}, metric=metric, value=value,
                                     reason=row["metrics"]["not_applicable"].get(metric, "")))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--epoch_numbering", choices=("logged", "completed"), default="logged",
                   help="logged: stored epoch 18; completed: stored epoch 17")
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0", help="encoder extraction device; probes train on CPU")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=.001)
    p.add_argument("--weight_decay", type=float, default=.01)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.probe_epochs < 1 or args.batch_size < 1 or min(args.seeds) < 0
            or len(set(args.seeds)) != len(args.seeds)
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        p.error("invalid seeds, counts or optimizer parameters")
    offset = int(args.epoch_numbering == "completed")
    metadata = {"pretrained_v2": inspect_checkpoint(args.checkpoint, 18 - offset)}
    a = metadata["pretrained_v2"]
    records, manifest = prepare_records(args.data_dir, a)
    labels = {i.path: structural_labels(i) for items in records.values() for i in items}
    coverage = {}
    for split, items in records.items():
        coverage[split] = {}
        for family in FAMILIES:
            family_items = [i for i in items if i.family == family]
            coverage[split][family] = {
                task: dict(distribution([v for i in family_items for v in labels[i.path][task]], task),
                           graphs=len(family_items),
                           undefined_graphs=sum(not labels[i.path][task] for i in family_items))
                for task in TASKS
            }
    # Exclusive run directory prevents mixing or overwriting seeds/results.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump(args.output_dir / "config.json", dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoints=metadata, tasks=TASKS,
        protocol="frozen eval-mode V2, all rows, one affine head/task, train-only z-score; fixed AdamW",
        selection="per-task minimum equal-family BA/WS/SBM validation loss; no test/ER selection",
        labels=dict(node_kcore="integer coreness, regression", node_articulation="binary cut vertex",
                    edge_betweenness="exact undirected, normalized=True, unweighted",
                    graph_efficiency="unweighted global efficiency; unreachable pairs contribute zero; n<2 undefined",
                    graph_four_cycles="undirected non-induced simple four-cycles; log1p regression, raw-count metrics"),
        excluded_pretraining_labels=["distance bins", "edge triangle", "bridge", "components",
                                     "graph triangle", "bipartite"],
        missing_class_policy="skip binary probe if train misses a class; null two-class metrics on single-class eval",
        seed_policy="same samples; each seed controls random encoder, probe init and minibatch permutation",
    ))
    dump(args.output_dir / "data_manifest.json", manifest)
    dump(args.output_dir / "label_coverage.json", coverage)
    all_rows = []
    # Trained encoders are extracted once; random encoders vary with seed.
    for model in ("random_v2", "pretrained_v2"):
        cached = None
        for seed in args.seeds:
            if cached is None or model == "random_v2":
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                if model == "random_v2":
                    config = a["config"]
                    encoder = HIRSequenceEncoder(
                        size=config["size"], id_encoding=config["id_encoding"],
                        digit_dim=config["digit_dim"], feature_dim=config["feature_dim"],
                        max_length=config["max_length"], attention_dropout=config["attention_dropout"],
                    ).to(args.device)
                else:
                    encoder = HIRSequenceEncoder.from_checkpoint(metadata[model]["path"], args.device)
                if encoder.config != a["config"] or encoder.tokenizer.config != a["tokenizer_config"]:
                    raise ValueError("encoder construction differs from validated checkpoint configuration")
                print(f"Extracting frozen {model}, seed={seed}", flush=True)
                cache_name = f"{model}_seed{seed}" if model == "random_v2" else model
                cached = extract(encoder, records, labels, args.output_dir / "cache" / cache_name,
                                 dict(model=model, encoder_seed=seed if model == "random_v2" else None,
                                      checkpoint=a, data_manifest=manifest,
                                      encoder_config=encoder.config, tokenizer_config=encoder.tokenizer.config))
                del encoder
            for task in TASKS:
                fitted, reason = fit_probe(cached, task, seed, args)
                if fitted is not None:
                    torch.save(fitted["artifact"], args.output_dir / f"{model}_seed{seed}_{task}.pt")
                    dump(args.output_dir / f"{model}_seed{seed}_{task}_history.json", fitted["history"])
                    task_rows = fitted["results"]
                else:
                    task_rows = []
                    for split in ("val", "test", "ood"):
                        for family in FAMILIES:
                            metric = metrics([], [], TASKS[task][1] == "classification")
                            metric["n"] = cached[split][task].families.count(family)
                            metric["not_applicable"] = {k: reason for k in metric if k not in ("n", "not_applicable")}
                            task_rows.append(dict(split=split, family=family, metrics=metric))
                for row in task_rows:
                    row.update(model=model, seed=seed, task=task,
                               selected_epoch=fitted["artifact"]["selected_epoch"] if fitted else None,
                               status="ok" if fitted else reason)
                    all_rows.append(row)
                print(f"Probe {model}, seed={seed}, task={task}: {reason or 'complete'}", flush=True)
            dump(args.output_dir / "metrics.json", all_rows)
    summary = summarize(all_rows)
    dump(args.output_dir / "comparison.json", summary)
    write_csv(args.output_dir / "metrics.csv", all_rows)
    # Flat, directly comparable model means and paired gains for each metric.
    dump(args.output_dir / "summary.json", summary["per_model"])
    print(f"Completed frozen probes: {args.output_dir.resolve() / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
