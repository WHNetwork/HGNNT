"""Independent HIRBERT ablations with validation-selected checkpoints.

Accuracies use percentages (0--100), as in the existing DHG-Bench results.
No existing model, compiler, preprocessing, or training entry point is changed.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from lib_dataset import _single_datasets_
from lib_models.HNN.hirbert import HIRBERT, HIRBERT_DEFAULTS
from lib_utils.utils import fix_seed
from parameter_parser import method_config, str2bool


ROOT = Path(__file__).resolve().parent
VARIANTS = ('full', 'feature_only', 'no_feature_skip', 'structure_only')
LABELS = dict(zip(VARIANTS, ('Full HIRBERT', 'Feature Only', 'No Feature Skip', 'Structure Only')))
HISTORY_FIELDS = ('epoch', 'train_loss', 'train_eval_loss', 'val_loss', 'train_acc', 'val_acc')
RUN_FIELDS = (
    'dataset', 'variant', 'seed', 'best_epoch', 'last_epoch', 'stopped_early',
    'train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss',
    'last_train_acc', 'last_val_acc', 'last_test_acc',
    'last_train_loss', 'last_val_loss', 'last_test_loss', 'train_time', 'parameter_count',
)
SUMMARY_FIELDS = (
    'dataset', 'variant', 'num_seeds', 'test_acc_mean', 'test_acc_std',
    'val_acc_mean', 'val_acc_std', 'best_epoch_mean', 'train_time_mean', 'parameter_count',
)
COMPARISON_FIELDS = (
    'dataset', 'variant', 'num_paired_seeds', 'full_test_acc_mean',
    'variant_test_acc_mean', 'full_minus_variant_pp', 'delta_std_pp',
)


def model_config(args):
    return {key: default if getattr(args, 'hir_' + key, None) is None
            else getattr(args, 'hir_' + key) for key, default in HIRBERT_DEFAULTS.items()}


class FeatureOnly(nn.Module):
    """The existing projection/MLP architecture, with one hidden input branch.

    BERT is never constructed. The MLP has the same hidden width, GELU,
    dropout and output layer as HIRBERT; its first layer accepts H rather
    than 2H inputs because the edge-to-node branch is absent.
    """

    def __init__(self, num_features, num_classes, args):
        super().__init__()
        self.model_config = model_config(args)
        self.num_features, self.num_classes = num_features, num_classes
        hidden = self.model_config['hidden_size']
        self.feature_projection = nn.Linear(num_features, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(self.model_config['dropout']), nn.Linear(hidden, num_classes),
        )
        self.compilation_stats = {}

    def reset_parameters(self):
        self.feature_projection.reset_parameters()
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, data):
        return self.classifier(self.feature_projection(data.x)), None


class WithoutFeatureSkip(HIRBERT):
    """Reuse HIREncoder and unchanged H-IR compilation and node readout."""

    def __init__(self, num_features, num_classes, args, structure_only=False):
        super().__init__(num_features, num_classes, args)
        self.structure_only = structure_only
        hidden = self.model_config['hidden_size']
        self.classifier[0] = nn.Linear(hidden, hidden)
        if structure_only:
            # This projection is never called or passed to the optimizer/count.
            self.encoder.feature_projection.requires_grad_(False)

    def forward(self, data):
        self.prepare(data)
        if self.structure_only:
            # Only the node count is used by prepare(); no values from data.x
            # enter the encoder. Zeros also avoid introducing projection bias.
            features = self.encoder.order_projection.weight.new_zeros(
                (self.program.num_nodes, self.model_config['hidden_size']))
        else:
            features = self.encoder.feature_projection(data.x)
        edges = self.encoder(features, self.program, self.chunks)
        node_ids, edge_ids = self._incidence.to(features.device)
        messages = torch.zeros_like(features).index_add(0, node_ids, edges[edge_ids])
        degrees = torch.bincount(node_ids, minlength=features.shape[0]).clamp_min(1)
        messages = messages / degrees.unsqueeze(-1)
        return self.classifier(messages), None


def make_model(variant, num_features, num_classes, args):
    if variant == 'full':
        return HIRBERT(num_features, num_classes, args)
    if variant == 'feature_only':
        return FeatureOnly(num_features, num_classes, args)
    if variant in ('no_feature_skip', 'structure_only'):
        return WithoutFeatureSkip(num_features, num_classes, args, variant == 'structure_only')
    raise ValueError(f'Unknown variant: {variant}')


@dataclass
class EarlyStopping:
    patience: int
    best_accuracy: float = -float('inf')
    best_loss: float = float('inf')
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, val_accuracy, val_loss, epoch):
        improved = (val_accuracy, -val_loss) > (self.best_accuracy, -self.best_loss)
        if improved:
            self.best_accuracy, self.best_loss = val_accuracy, val_loss
            self.best_epoch, self.bad_epochs = epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


def load_data(args):
    """Use the benchmark loader and HIRBERT preprocessing, including no added loops."""
    from lib_dataset.data_base import HyperDataset
    from lib_dataset.preprocessing import data_processing

    loader_args = SimpleNamespace(
        method='HIRBERT', task_type='node_cls', dname=args.dname, device=args.device,
        feature_noise=args.feature_noise, add_self_loop=False,
    )
    # HyperDataset's data paths are relative to dhgbench, even when this script
    # is invoked from the repository root or another directory.
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        data = HyperDataset(loader_args)
        data = data_processing(loader_args, data)
        data._initialization_()
    finally:
        os.chdir(previous)
    return data


def make_masks(data, args, seed):
    masks = data.generate_random_split(
        train_ratio=args.train_prop, val_ratio=args.valid_prop, seed=seed)
    if any(not mask.any().item() for mask in masks.values()):
        raise ValueError('Train, validation and test splits must all be nonempty')
    return masks


def run_config(args):
    """Compare settings directly when extending an existing experiment directory."""
    return {
        'model_config': model_config(args), 'optimizer': 'Adam',
        'lr': args.lr, 'wd': args.wd, 'max_epochs': args.epochs, 'patience': args.patience,
        'selection_metric': 'val_accuracy', 'tie_breaker': 'val_loss',
        'train_prop': args.train_prop, 'valid_prop': args.valid_prop,
        'feature_noise': args.feature_noise,
    }


@torch.no_grad()
def evaluate(model, data, masks, splits=('train', 'valid')):
    model.eval()
    logits, _ = model(data)
    result = {}
    for split in splits:
        mask = masks[split]
        prefix = 'val' if split == 'valid' else split
        result[prefix + '_loss'] = F.nll_loss(
            F.log_softmax(logits[mask], dim=-1), data.y[mask]).item()
        result[prefix + '_acc'] = 100 * (
            logits[mask].argmax(-1) == data.y[mask]).double().mean().item()
    return result


def load_checkpoint(path, device='cpu'):
    """Reload any ablation for inference on the same dataset (returned separately)."""
    saved = torch.load(path, map_location=device, weights_only=True)
    args = SimpleNamespace(**{'hir_' + key: value for key, value in saved['model_config'].items()})
    model = make_model(saved['variant'], saved['num_features'], saved['num_classes'], args).to(device)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    return model, saved


def train_run(data, masks, args, variant, seed, directory):
    """Train from scratch; test labels are evaluated only after selection ends."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A completion marker belongs only to a fully finished run.
    (directory / 'metrics.json').unlink(missing_ok=True)
    fix_seed(seed)
    model = make_model(variant, data.num_features, data.num_classes, args).to(args.device)
    model.reset_parameters()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_count = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
    stopping = EarlyStopping(args.patience)
    checkpoint_path = directory / 'best_checkpoint.pt'
    config = run_config(args)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    with (directory / 'training.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = F.nll_loss(F.log_softmax(logits[masks['train']], dim=-1), data.y[masks['train']])
            loss.backward()
            optimizer.step()
            current = evaluate(model, data, masks)
            row = {'epoch': epoch, **current, 'train_eval_loss': current['train_loss'],
                   'train_loss': loss.item()}
            writer.writerow(row)
            stream.flush()
            if stopping.update(current['val_acc'], current['val_loss'], epoch):
                torch.save({
                    'dataset': args.dname, 'variant': variant, 'seed': seed, 'epoch': epoch,
                    'val_acc': current['val_acc'], 'val_loss': current['val_loss'],
                    'num_features': data.num_features, 'num_classes': data.num_classes,
                    'model_config': model.model_config, 'run_config': config,
                    'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    'masks': {key: mask.detach().cpu() for key, mask in masks.items()},
                }, checkpoint_path)
            if epoch == 1 or epoch % args.display_step == 0 or stopping.should_stop:
                print(f'[{args.dname}/{variant}/seed_{seed}] epoch={epoch} '
                      f'train_loss={row["train_loss"]:.4f} val_loss={current["val_loss"]:.4f} '
                      f'train_acc={current["train_acc"]:.2f} val_acc={current["val_acc"]:.2f}')
            if stopping.should_stop:
                break
    if torch.device(args.device).type == 'cuda':
        torch.cuda.synchronize(args.device)
    train_time = time.perf_counter() - started
    # Both test evaluations happen after the entire validation-only loop.
    last = evaluate(model, data, masks, ('train', 'valid', 'test'))
    saved = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    model.load_state_dict(saved['model_state_dict'])
    best = evaluate(model, data, masks, ('train', 'valid', 'test'))
    metrics = {
        'dataset': args.dname, 'variant': variant, 'seed': seed,
        'best_epoch': stopping.best_epoch, 'best_val_acc': best['val_acc'],
        'best_val_loss': best['val_loss'], 'last_epoch': epoch,
        'stopped_early': epoch < args.epochs, **best,
        **{'last_' + key: value for key, value in last.items()},
        'parameter_count': parameter_count, 'train_time': train_time,
        'model_config': model.model_config, 'run_config': config,
        'compilation_stats': model.compilation_stats,
        'accuracy_unit': 'percent', 'completed': True,
    }
    # Written last: interrupted runs are restarted rather than mistaken for complete.
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'[{variant}/seed_{seed}] best_epoch={stopping.best_epoch} '
          f'best_val_acc={best["val_acc"]:.2f} test_acc={best["test_acc"]:.2f} '
          f'last_test_acc={last["test_acc"]:.2f} parameters={parameter_count}')
    return metrics


def completed_run(directory, expected_config=None):
    directory = Path(directory)
    paths = [directory / name for name in ('training.csv', 'metrics.json', 'best_checkpoint.pt')]
    if not all(path.is_file() and path.stat().st_size for path in paths):
        return None
    try:
        metrics = json.loads(paths[1].read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return None
    if not metrics.get('completed'):
        return None
    if expected_config is not None and metrics['run_config'] != expected_config:
        raise ValueError(f'{directory} uses different experiment settings; choose a different --output directory')
    return metrics


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def collect_runs(output):
    rows = []
    for path in sorted(Path(output).glob('*/*/seed_*/metrics.json')):
        result = completed_run(path.parent)
        if result is not None:
            rows.append(result)
    return rows


def write_reports(output, plots=True):
    """Include prior completed seeds/variants when a subset is run later."""
    output = Path(output)
    rows = collect_runs(output)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['dataset'], row['variant']].append(row)
    summaries, comparisons = [], []
    for (dataset, variant), group in grouped.items():
        summaries.append({
            'dataset': dataset, 'variant': variant, 'num_seeds': len(group),
            'test_acc_mean': mean(row['test_acc'] for row in group),
            'test_acc_std': pstdev(row['test_acc'] for row in group),
            'val_acc_mean': mean(row['val_acc'] for row in group),
            'val_acc_std': pstdev(row['val_acc'] for row in group),
            'best_epoch_mean': mean(row['best_epoch'] for row in group),
            'train_time_mean': mean(row['train_time'] for row in group),
            'parameter_count': group[0]['parameter_count'],
        })
        if variant == 'full':
            continue
        full = {row['seed']: row for row in grouped.get((dataset, 'full'), [])}
        paired = [(full[row['seed']]['test_acc'], row['test_acc'])
                  for row in group if row['seed'] in full]
        if paired:
            deltas = [first - second for first, second in paired]
            comparisons.append({
                'dataset': dataset, 'variant': variant, 'num_paired_seeds': len(paired),
                'full_test_acc_mean': mean(first for first, _ in paired),
                'variant_test_acc_mean': mean(second for _, second in paired),
                'full_minus_variant_pp': mean(deltas), 'delta_std_pp': pstdev(deltas),
            })
    write_csv(output / 'ablation_runs.csv', rows, RUN_FIELDS)
    write_csv(output / 'ablation_summary.csv', summaries, SUMMARY_FIELDS)
    write_csv(output / 'ablation_comparison.csv', comparisons, COMPARISON_FIELDS)
    if plots and rows:
        plot_results(output, rows, summaries)
    return rows


def plot_results(output, rows, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for dataset in sorted({row['dataset'] for row in rows}):
        histories = defaultdict(list)
        for row in rows:
            if row['dataset'] == dataset:
                path = output / dataset / row['variant'] / f'seed_{row["seed"]}' / 'training.csv'
                with path.open(newline='', encoding='utf-8') as stream:
                    histories[row['variant']].append(list(csv.DictReader(stream)))
        for quantity, ylabel in (('acc', 'Accuracy (%)'), ('loss', 'NLL loss')):
            figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            for axis, split in zip(axes, ('train', 'val')):
                for variant in VARIANTS:
                    by_epoch = defaultdict(list)
                    for history in histories[variant]:
                        for row in history:
                            by_epoch[int(row['epoch'])].append(float(row[f'{split}_{quantity}']))
                    if by_epoch:
                        epochs = sorted(by_epoch)
                        axis.plot(epochs, [mean(by_epoch[epoch]) for epoch in epochs], label=LABELS[variant])
                axis.set(xlabel='Epoch', ylabel=ylabel, title='Train' if split == 'train' else 'Validation')
                axis.grid(alpha=0.2)
                axis.legend()
            figure.suptitle(f'{dataset}: mean over seeds with a recorded epoch')
            figure.savefig(output / dataset / f'{"accuracy" if quantity == "acc" else "loss"}_curves.png', dpi=150)
            plt.close(figure)
        summary = {row['variant']: row for row in summaries if row['dataset'] == dataset}
        variants = [variant for variant in VARIANTS if variant in summary]
        figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        axis.bar([LABELS[variant] for variant in variants],
                 [summary[variant]['test_acc_mean'] for variant in variants],
                 yerr=[summary[variant]['test_acc_std'] for variant in variants], capsize=5)
        axis.set(ylabel='Test accuracy (%)', ylim=(0, 100),
                 title=f'{dataset}: validation-selected checkpoints (mean +/- population std)')
        figure.savefig(output / dataset / 'test_accuracy.png', dpi=150)
        plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dname', choices=_single_datasets_, default='cora')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--seed_offset', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--selection_metric', choices=['val_accuracy'], default='val_accuracy')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'hirbert_ablation')
    parser.add_argument('--variants', '--variant', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--feature_noise', default='0.6')
    parser.add_argument('--display_step', type=int, default=20)
    for key, default in HIRBERT_DEFAULTS.items():
        parser.add_argument('--hir_' + key, type=str2bool if isinstance(default, bool) else type(default))
    args = parser.parse_args(argv)
    if min(args.num_seeds, args.epochs, args.patience, args.display_step) < 1:
        parser.error('num_seeds, epochs, patience and display_step must be positive')
    if not (0 < args.train_prop < 1 and 0 < args.valid_prop < 1
            and args.train_prop + args.valid_prop < 1):
        parser.error('train_prop and valid_prop must leave nonempty train/valid/test splits')
    args.method, args.is_default = 'HIRBERT', False
    args._hirbert_cli_overrides = {
        key: value for key, value in vars(args).items()
        if (key.startswith('hir_') or key in ('epochs', 'lr', 'wd')) and value is not None
    }
    args = method_config(args)
    args.output = args.output.resolve()
    args.variants = list(dict.fromkeys(args.variants))
    return args


def run_experiments(args, data=None):
    args.output.mkdir(parents=True, exist_ok=True)
    expected_config = run_config(args)
    # Reject mixed settings before training any new variant in this dataset.
    for path in (args.output / args.dname).glob('*/seed_*/metrics.json'):
        completed_run(path.parent, expected_config)
    if data is None:
        data = load_data(args)
    for seed in range(args.seed_offset, args.seed_offset + args.num_seeds):
        # Generated once, before all variants, with the existing benchmark API.
        masks = make_masks(data, args, seed)
        for variant in args.variants:
            directory = args.output / args.dname / variant / f'seed_{seed}'
            if completed_run(directory, expected_config) is not None:
                print(f'[skip complete] {args.dname}/{variant}/seed_{seed}')
                continue
            train_run(data, masks, args, variant, seed, directory)
            write_reports(args.output, plots=False)
    rows = write_reports(args.output)
    print(f'[Ablation results] {args.output}')
    return rows


def main(argv=None):
    run_experiments(parse_args(argv))


if __name__ == '__main__':
    main()
