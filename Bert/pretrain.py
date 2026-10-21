"""Structure-only masked membership pretraining on existing LCC data splits.

Run from the repository root, for example:
python -m Bert.pretrain --output_dir runs/hirbert_structure

No LCC extraction or repartitioning happens here. Validation masks are prepared
once in memory; test/OOD masks are evaluated only after loading the best model.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from tqdm import tqdm

if not __package__:
    # Support PyCharm Run without depending on its working directory.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Bert"

from .data import FAMILIES, SPLITS, Hypergraph, load_graph
from .encoder import HIRBertEncoder
from .hir import HIRProgram, compile_hypergraph


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "Bert" / "datasets" / "synthetic_v1_lcc"
PROGRESS_FORMAT = "{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}] |{bar}|"


@dataclass
class MembershipBatch:
    program: HIRProgram
    positive_pairs: Tensor  # [P, 2], columns node symbol / original edge symbol
    negative_pairs: Tensor  # [P, 2], absent from the FULL original graph


def sample_masked_membership(graph: Hypergraph, mask_ratio: float,
                             generator: torch.Generator) -> MembershipBatch | None:
    """Hide positives BEFORE compilation; no full-order/structure side channel.

    Greedily hide up to round(ratio * order) members per eligible edge in random
    edge/member order. BOTH endpoints keep >=1 visible incidence. Constraints
    can reduce the realized mask ratio. Negatives are original nonmembers drawn
    from active nodes, with replacement. No valid pairs means skip this graph.
    Full memberships/degrees are used only for sampling, never Encoder input.
    """
    if not 0 < mask_ratio <= 1:
        raise ValueError("mask_ratio must be in (0, 1]")
    members = [set() for _ in range(graph.num_hyperedges)]
    for node, edge in graph.hyperedge_index.t().tolist():
        members[edge].add(node)
    remaining_degree = [0] * graph.num_nodes
    for group in members:
        for node in group:
            remaining_degree[node] += 1
    active = {node for node, degree in enumerate(remaining_degree) if degree > 0}
    hidden = [set() for _ in members]
    positives, negatives = [], []
    for edge in torch.randperm(graph.num_hyperedges, generator=generator).tolist():
        group = members[edge]
        nonmembers = sorted(active - group)
        if len(group) < 2 or not nonmembers:
            continue
        count = min(len(group) - 1, max(1, round(mask_ratio * len(group))))
        ordered = sorted(group)
        for index in torch.randperm(len(ordered), generator=generator).tolist():
            node = ordered[index]
            if remaining_degree[node] > 1:
                hidden[edge].add(node)
                remaining_degree[node] -= 1
                positives.append((node, edge))
            if len(hidden[edge]) == count:
                break
        draws = torch.randint(len(nonmembers), (len(hidden[edge]),), generator=generator).tolist()
        negatives.extend((nonmembers[i], edge) for i in draws)
    if not positives:
        return None
    # No initially active node became isolated, so active above is also exactly
    # the active-node set in this final visible structure. Never take LCC here.
    visible = [(node, edge) for edge, group in enumerate(members)
               for node in sorted(group - hidden[edge])]
    visible_index = torch.tensor(visible, dtype=torch.long).reshape(-1, 2).t().contiguous()
    program = compile_hypergraph(graph.num_nodes, visible_index,
                                 num_hyperedges=graph.num_hyperedges)
    return MembershipBatch(program, torch.tensor(positives, dtype=torch.long),
                           torch.tensor(negatives, dtype=torch.long))


class MaskedMembershipPretrainer(nn.Module):
    """A removable pair scorer/loss; the Encoder knows nothing about this task."""

    def __init__(self, encoder: HIRBertEncoder):
        super().__init__()
        self.encoder = encoder
        d = encoder.hidden_dim
        self.membership_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, batch: MembershipBatch) -> Tensor:
        device = next(self.parameters()).device
        encoded = self.encoder(batch.program)
        pairs = torch.cat((batch.positive_pairs, batch.negative_pairs)).to(device)
        nodes = encoded.node_embeddings[pairs[:, 0]]
        edges = encoded.hyperedge_embeddings[pairs[:, 1]]
        logits = self.membership_head(torch.cat((nodes, edges, nodes * edges), dim=-1)).squeeze(-1)
        labels = torch.cat((torch.ones(len(batch.positive_pairs), device=device),
                            torch.zeros(len(batch.negative_pairs), device=device)))
        return self.loss(logits, labels)


def split_paths(data_dir, split, limit=None, seed=0):
    """Select fixed family-balanced prefixes of seeded permutations, not new splits.

    Using the same seed gives nested subsets at 800 / 3200 / 12800 graphs.
    Only the four existing families are considered; nested is not added.
    """
    groups = [sorted((data_dir / split / family).rglob("*.pt")) for family in FAMILIES]
    if limit is None:
        return [path for group in groups for path in group]
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for index, group in enumerate(groups):
        quota = limit // len(FAMILIES) + int(index < limit % len(FAMILIES))
        if len(group) < quota:
            raise ValueError(f"{split}/{FAMILIES[index]} has fewer than {quota} graphs")
        order = torch.randperm(len(group), generator=generator).tolist()
        selected.extend(group[i] for i in order[:quota])
    return sorted(selected)


def masked_batches(paths, mask_ratio, seed, description):
    """Deterministic evaluation masks; no variants are saved to disk."""
    generator = torch.Generator().manual_seed(seed)
    for path in tqdm(paths, desc=description, dynamic_ncols=True, ascii=True,
                     bar_format=PROGRESS_FORMAT):
        batch = sample_masked_membership(load_graph(path), mask_ratio, generator)
        if batch is not None:
            yield batch


def evaluate(model, batches):
    model.eval()
    total, pairs, graphs = 0.0, 0, 0
    with torch.no_grad():
        for batch in batches:
            count = 2 * len(batch.positive_pairs)
            total += model(batch).item() * count
            pairs += count
            graphs += 1
    if not pairs:
        raise ValueError("evaluation split has no valid masked membership samples")
    return total / pairs, graphs, pairs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Transformer depth in each of the local and global stages")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path,
                        default=Path(__file__).resolve().parent / "runs" / "structure_v1")
    parser.add_argument("--max_train_graphs", type=int, choices=(800, 3200, 12800))
    parser.add_argument("--max_val_graphs", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.data_dir = (REPO_ROOT / args.data_dir).resolve()
    args.output_dir = (REPO_ROOT / args.output_dir).resolve()
    if args.epochs < 1 or args.seed < 0 or not 0 < args.mask_ratio <= 1:
        raise ValueError("epochs must be positive, seed nonnegative, mask_ratio in (0, 1]")
    if args.max_val_graphs is not None and args.max_val_graphs < 1:
        raise ValueError("max_val_graphs must be positive")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    limits = {"train": args.max_train_graphs, "val": args.max_val_graphs}
    paths = {split: split_paths(args.data_dir, split, limits.get(split), args.seed)
             for split in SPLITS}
    if any(not group for group in paths.values()):
        raise ValueError("data_dir must contain the existing train/val/test/ood family directories")
    validation_batches = list(masked_batches(paths["val"], args.mask_ratio, args.seed + 1, "val masks"))
    if not validation_batches:
        raise ValueError("validation graphs have no valid masked membership samples")
    encoder = HIRBertEncoder(args.hidden_dim, args.num_layers, args.num_heads)
    model = MaskedMembershipPretrainer(encoder).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_encoder.pt"
    best_validation = float("inf")
    print("Existing splits: " + ", ".join(f"{split}={len(group)}" for split, group in paths.items()))
    with (args.output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_loss", "validation_loss", "train_graphs", "validation_graphs"))
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_total, train_pairs, train_graphs = 0.0, 0, 0
            order = torch.randperm(len(paths["train"]), generator=generator).tolist()
            for index in tqdm(order, desc=f"train {epoch}/{args.epochs}",
                              dynamic_ncols=True, ascii=True, bar_format=PROGRESS_FORMAT):
                graph = load_graph(paths["train"][index])
                batch = sample_masked_membership(graph, args.mask_ratio, generator)
                if batch is None:
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = model(batch)
                loss.backward()
                optimizer.step()
                count = 2 * len(batch.positive_pairs)
                train_total += loss.detach().item() * count
                train_pairs += count
                train_graphs += 1
            if not train_pairs:
                raise ValueError("training graphs have no valid masked membership samples")
            validation_loss, validation_graphs, _ = evaluate(model, tqdm(
                validation_batches, desc="val", dynamic_ncols=True, ascii=True,
                bar_format=PROGRESS_FORMAT))
            train_loss = train_total / train_pairs
            writer.writerow((epoch, train_loss, validation_loss, train_graphs, validation_graphs))
            stream.flush()
            print(f"epoch={epoch} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}")
            if validation_loss < best_validation:
                best_validation = validation_loss
                # Encoder and membership head remain independently loadable.
                torch.save({"encoder_config": encoder.config,
                            "encoder_state_dict": encoder.state_dict(),
                            "membership_head_state_dict": model.membership_head.state_dict(),
                            "epoch": epoch, "validation_loss": validation_loss,
                            "seed": args.seed, "mask_ratio": args.mask_ratio,
                            "validation_mask_seed": args.seed + 1,
                            "train_files": [p.relative_to(args.data_dir).as_posix() for p in paths["train"]],
                            "validation_files": [p.relative_to(args.data_dir).as_posix() for p in paths["val"]]},
                           checkpoint_path)
    # Test/OOD do not influence model selection and are evaluated exactly once.
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.membership_head.load_state_dict(checkpoint["membership_head_state_dict"])
    with (args.output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "loss", "graphs", "pairs", "mask_seed"))
        for split, mask_seed in (("test", args.seed + 2), ("ood", args.seed + 3)):
            loss, graphs, pairs = evaluate(model, masked_batches(
                paths[split], args.mask_ratio, mask_seed, split))
            writer.writerow((split, loss, graphs, pairs, mask_seed))
            print(f"best_epoch={checkpoint['epoch']} {split}_loss={loss:.6f} graphs={graphs}")


if __name__ == "__main__":
    main()
