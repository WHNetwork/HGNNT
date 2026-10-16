"""One command: unlabeled H-IR pretraining, transfer, five downstream controls."""

import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from benchmark_hirbert_ablation import EarlyStopping, evaluate, load_data, make_masks, make_model, model_config, write_csv
from lib_utils.hirbert_pretrain_utils import (
    atomic_save, check_encoder_compatibility, completed_metrics, cpu_state, grad_norm, load_encoder, make_scheduler,
    representation_stats, restore_rng, rng_state, save_json, training_arguments, write_history,
)
from lib_utils.utils import fix_seed
from pretrain_hirbert import build_parser, configure_args, load_unlabeled_data, run_pretraining


DEFAULT_VARIANTS = ('full_scratch', 'full_pretrained', 'no_skip_scratch', 'no_skip_pretrained', 'feature_only')
VARIANTS = DEFAULT_VARIANTS + ('full_pretrained_frozen', 'no_skip_pretrained_frozen')
LABELS = {key: key.replace('_', ' ').title() for key in VARIANTS}
COMPARISONS = (('full_pretrained', 'full_scratch'), ('no_skip_pretrained', 'no_skip_scratch'),
               ('full_pretrained', 'feature_only'))


def parse_args(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--seed_offset', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--scheduler', choices=('cosine', 'linear', 'none'), default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument('--pretrained_checkpoint', type=Path)
    args = configure_args(parser.parse_args(argv), parser)
    args.train_prop, args.valid_prop = 0.5, 0.25
    args.variants = list(dict.fromkeys(args.variants))
    if min(args.num_seeds, args.epochs, args.patience) < 1 or min(args.seed_offset, args.warmup_epochs) < 0:
        parser.error('Invalid seed, epochs, patience or warmup')
    if args.lr <= 0 or args.wd < 0:
        parser.error('Learning rate must be positive and weight decay nonnegative')
    return args


def base_variant(variant):
    if variant == 'feature_only':
        return variant
    return 'no_feature_skip' if variant.startswith('no_skip') else 'full'


def finetune_config(args, variant, data, pretrained=None):
    return {
        'variant': variant, 'model_config': model_config(args), 'optimizer': 'AdamW',
        'lr': args.lr, 'wd': args.wd, 'epochs': args.epochs, 'patience': args.patience,
        'scheduler': args.scheduler, 'warmup_epochs': args.warmup_epochs, 'grad_clip': args.grad_clip,
        'train_prop': args.train_prop, 'valid_prop': args.valid_prop, 'feature_noise': args.feature_noise,
        'dataset': args.dname, 'num_nodes': data.num_nodes, 'num_features': data.num_features,
        'num_classes': data.num_classes,
        'pretrained': pretrained if 'pretrained' in variant else None,
    }


def initialize_model(data, args, variant, checkpoint=None):
    model = make_model(base_variant(variant), data.num_features, data.num_classes, args).to(args.device)
    # This is intentionally BEFORE strict encoder transfer. The old Trainer is
    # never called. Scratch and pretrained controls have the same head init.
    model.reset_parameters()
    report = None
    if 'pretrained' in variant:
        if checkpoint is None:
            raise ValueError('A pretrained encoder checkpoint is required')
        report = load_encoder(model, checkpoint, args.dname)
        print(f'[{variant}] strictly loaded {len(report["loaded_keys"])} encoder tensors; '
              f'missing={report["missing_keys"]}, unexpected={report["unexpected_keys"]}')
    if variant.endswith('_frozen'):
        model.encoder.requires_grad_(False)
    return model, report


def evaluate_with_diagnostics(model, data, masks, splits=('train', 'valid')):
    diagnostics = {'node_repr_variance': 0.0, 'edge_repr_variance': 0.0}

    def record(encoder, inputs, edges):
        diagnostics.update(representation_stats(edges, model._incidence, data.num_nodes))

    handle = model.encoder.register_forward_hook(record) if hasattr(model, 'encoder') else None
    try:
        result = evaluate(model, data, masks, splits)
    finally:
        if handle is not None:
            handle.remove()
    return {**result, **diagnostics}


def load_finetuned(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    args = SimpleNamespace(**{'hir_' + key: value for key, value in saved['model_config'].items()})
    model = make_model(base_variant(saved['variant']), saved['num_features'], saved['num_classes'], args).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if saved['variant'].endswith('_frozen'):
        model.encoder.requires_grad_(False)
    return model.eval(), saved


def finetune_run(data, masks, args, variant, seed, directory, checkpoint=None, pretrained=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = finetune_config(args, variant, data, pretrained)
    done = completed_metrics(directory, config, ('training.csv', 'best_checkpoint.pt'))
    if done is not None:
        print(f'[skip complete finetune] {variant}/seed_{seed}')
        return done
    fix_seed(seed)
    model, report = initialize_model(data, args, variant, checkpoint)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd)
    scheduler = make_scheduler(optimizer, args.scheduler, args.epochs, min(args.warmup_epochs, args.epochs))
    stopping = EarlyStopping(args.patience)
    history, best_state, elapsed = [], None, 0.0
    last_path = directory / 'last_checkpoint.pt'
    if last_path.exists():
        if not args.resume:
            raise ValueError('Incomplete finetuning exists; enable --resume or choose another --output')
        saved = torch.load(last_path, map_location=args.device, weights_only=True)
        if saved['config'] != config:
            raise ValueError('Finetuning resume configuration differs')
        for split, mask in masks.items():
            if not torch.equal(mask.cpu(), saved['masks'][split].cpu()):
                raise ValueError('Finetuning resume masks differ')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        history, best_state = saved['history'], saved['best_model_state_dict']
        stopping, elapsed, report = EarlyStopping(**saved['stopping']), saved['elapsed'], saved['encoder_load_report']
        restore_rng(saved['rng_state'])
        print(f'[resume finetune] {variant}/seed_{seed} after epoch {saved["epoch"]}')
    for epoch in range(len(history) + 1, args.epochs + 1):
        if stopping.should_stop:
            break
        started = time.perf_counter()
        model.train()
        if variant.endswith('_frozen'):
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(data)
        loss = F.nll_loss(F.log_softmax(logits[masks['train']], dim=-1), data.y[masks['train']])
        if not math.isfinite(loss.item()):
            raise FloatingPointError('Non-finite finetuning loss')
        loss.backward()
        encoder_norm = grad_norm(model.encoder) if hasattr(model, 'encoder') else 0.0
        bert_norm = grad_norm(model.encoder.bert) if hasattr(model, 'encoder') else 0.0
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        current = evaluate_with_diagnostics(model, data, masks)
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], **current,
               'train_eval_loss': current['train_loss'], 'train_loss': loss.item(),
               'encoder_grad_norm': encoder_norm, 'bert_grad_norm': bert_norm}
        history.append(row)
        improved = stopping.update(current['val_acc'], current['val_loss'], epoch)
        if improved:
            best_state = cpu_state(model)
        scheduler.step()
        elapsed += time.perf_counter() - started
        atomic_save({'config': config, 'variant': variant, 'seed': seed, 'epoch': epoch,
                     'model_state_dict': cpu_state(model), 'best_model_state_dict': best_state,
                     'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                     'rng_state': rng_state(), 'masks': {key: value.cpu() for key, value in masks.items()},
                     'history': history, 'stopping': asdict(stopping), 'elapsed': elapsed,
                     'encoder_load_report': report}, last_path)
        write_history(directory / 'training.csv', history)
        if improved:
            atomic_save({'training_config': training_arguments(args), 'model_state_dict': best_state, 'variant': variant, 'seed': seed,
                         'epoch': stopping.best_epoch, 'model_config': model.model_config,
                         'num_features': data.num_features, 'num_classes': data.num_classes,
                         'masks': {key: value.cpu() for key, value in masks.items()},
                         'encoder_load_report': report, 'config': config}, directory / 'best_checkpoint.pt')
        if epoch == 1 or epoch % args.display_step == 0:
            print(f'[{variant}/seed_{seed}] epoch={epoch} val_acc={current["val_acc"]:.2f} '
                  f'val_loss={current["val_loss"]:.4f} encoder_grad={encoder_norm:.4g}')
    # Test labels are read only after validation-only training/model selection.
    last_metrics = evaluate_with_diagnostics(model, data, masks, ('train', 'valid', 'test'))
    best_path = directory / 'best_checkpoint.pt'
    # Re-export from the committed resume snapshot if artifact writing was interrupted.
    atomic_save({'training_config': training_arguments(args), 'model_state_dict': best_state, 'variant': variant, 'seed': seed,
                 'epoch': stopping.best_epoch, 'model_config': model.model_config,
                 'num_features': data.num_features, 'num_classes': data.num_classes,
                 'masks': {key: value.cpu() for key, value in masks.items()},
                 'encoder_load_report': report, 'config': config}, best_path)
    saved = torch.load(best_path, map_location=args.device, weights_only=True)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    best = evaluate_with_diagnostics(model, data, masks, ('train', 'valid', 'test'))
    write_history(directory / 'training.csv', history)
    result = {
        'dataset': args.dname, 'variant': variant, 'seed': seed, 'completed': True, 'config': config,
        'best_epoch': stopping.best_epoch, 'best_val_acc': best['val_acc'], 'last_epoch': history[-1]['epoch'],
        'stopped_early': history[-1]['epoch'] < args.epochs, **best,
        **{'last_' + key: value for key, value in last_metrics.items()},
        'parameter_count': sum(p.numel() for p in parameters), 'train_time': elapsed,
        'encoder_grad_norm_mean': mean(row['encoder_grad_norm'] for row in history),
        'bert_grad_norm_mean': mean(row['bert_grad_norm'] for row in history),
        'encoder_load_report': report, 'encoder_frozen': variant.endswith('_frozen'),
        'accuracy_unit': 'percent',
    }
    save_json(result, directory / 'metrics.json')
    print(f'[{variant}/seed_{seed}] best_epoch={stopping.best_epoch} test_acc={best["test_acc"]:.2f}')
    return result


def write_reports(output, plots=True):
    rows = []
    for path in sorted(Path(output).glob('*/*/seed_*/metrics.json')):
        metrics = json.loads(path.read_text(encoding='utf-8'))
        if metrics.get('variant') in VARIANTS:
            done = completed_metrics(path.parent, metrics['config'], ('training.csv', 'best_checkpoint.pt'))
            if done:
                rows.append(done)
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['variant']].append(row)
    summaries, comparisons = [], []
    for (dataset, variant), group in groups.items():
        summaries.append({
            'dataset': dataset, 'variant': variant, 'num_seeds': len(group),
            'test_acc_mean': mean(row['test_acc'] for row in group),
            'test_acc_std': pstdev(row['test_acc'] for row in group),
            'val_acc_mean': mean(row['val_acc'] for row in group),
            'val_acc_std': pstdev(row['val_acc'] for row in group),
            'best_epoch_mean': mean(row['best_epoch'] for row in group),
            'train_time_mean': mean(row['train_time'] for row in group), 'parameter_count': group[0]['parameter_count'],
            'encoder_grad_norm_mean': mean(row['encoder_grad_norm_mean'] for row in group),
            'node_repr_variance_mean': mean(row['node_repr_variance'] for row in group),
            'edge_repr_variance_mean': mean(row['edge_repr_variance'] for row in group),
        })
    for dataset in sorted({row['dataset'] for row in rows}):
        for first, second in COMPARISONS:
            one = {row['seed']: row['test_acc'] for row in groups.get((dataset, first), [])}
            two = {row['seed']: row['test_acc'] for row in groups.get((dataset, second), [])}
            shared = sorted(one.keys() & two.keys())
            if shared:
                deltas = [one[seed] - two[seed] for seed in shared]
                comparisons.append({'dataset': dataset, 'first': first, 'second': second,
                                    'num_paired_seeds': len(shared), 'difference_pp_mean': mean(deltas),
                                    'difference_pp_std': pstdev(deltas)})
    fields = ('dataset', 'variant', 'seed', 'best_epoch', 'last_epoch', 'train_acc', 'val_acc', 'test_acc',
              'last_train_acc', 'last_val_acc', 'last_test_acc', 'train_loss', 'val_loss',
              'parameter_count', 'train_time', 'encoder_grad_norm_mean', 'bert_grad_norm_mean',
              'node_repr_variance', 'edge_repr_variance', 'encoder_frozen')
    write_csv(output / 'finetune_runs.csv', rows, fields)
    write_csv(output / 'finetune_summary.csv', summaries,
              list(summaries[0]) if summaries else ('dataset', 'variant', 'num_seeds', 'test_acc_mean', 'test_acc_std'))
    write_csv(output / 'finetune_comparison.csv', comparisons,
              ('dataset', 'first', 'second', 'num_paired_seeds', 'difference_pp_mean', 'difference_pp_std'))
    if plots and rows:
        plot_finetune(output, rows, summaries)
    return rows


def plot_finetune(output, rows, summaries):
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
        for field, ylabel in (('acc', 'Accuracy (%)'), ('loss', 'NLL loss')):
            figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
            for axis, split in zip(axes, ('train', 'val')):
                for variant, histories_for_variant in histories.items():
                    values = defaultdict(list)
                    for history in histories_for_variant:
                        for row in history:
                            values[int(row['epoch'])].append(float(row[f'{split}_{field}']))
                    epochs = sorted(values)
                    axis.plot(epochs, [mean(values[epoch]) for epoch in epochs], label=LABELS[variant])
                axis.set(xlabel='Finetuning epoch', ylabel=ylabel, title=split)
                axis.legend(fontsize=8)
            figure.suptitle(f'{dataset}: means over seeds with recorded epochs')
            figure.savefig(output / dataset / f'finetune_{field}_curves.png', dpi=150)
            plt.close(figure)
        subset = [row for row in summaries if row['dataset'] == dataset]
        figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
        axis.bar([LABELS[row['variant']] for row in subset], [row['test_acc_mean'] for row in subset],
                 yerr=[row['test_acc_std'] for row in subset], capsize=5)
        axis.tick_params(axis='x', labelrotation=15)
        axis.set(ylabel='Test accuracy (%)', ylim=(0, 100), title=f'{dataset}: validation-selected mean +/- std')
        figure.savefig(output / dataset / 'finetune_test_accuracy.png', dpi=150)
        plt.close(figure)


def run_benchmark(args, data=None):
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint, provenance = None, None
    if any('pretrained' in variant for variant in args.variants):
        pretrain_input = data
        if data is None and args.pretrain_mode == 'single_graph':
            pretrain_input = load_unlabeled_data(SimpleNamespace(**{**vars(args), 'dname': args.pretrain_datasets[0], 'device': 'cpu'}))
        checkpoint = args.pretrained_checkpoint or run_pretraining(
            pretrain_input, args)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('kind') != 'hirbert_pretrained_encoder' or not saved.get('completed'):
            raise ValueError('A completed pretrained encoder checkpoint is required')
        provenance = {key: saved[key] for key in ('model_config', 'num_features', 'dataset_info',
                                                 'seed', 'best_epoch', 'best_val_loss', 'pretrain_config')}
        provenance.update({key: saved[key] for key in ('feature_dimensions', 'pretrain_datasets', 'adapter_trained_samples', 'failed_datasets')
                           if key in saved})
    # The classification loader/label metadata are needed only after Stage 1.
    data = load_data(args) if data is None else data
    if provenance is not None:
        check_encoder_compatibility(saved, model_config(args), data.num_features, args.dname)
    for variant in args.variants:
        expected = finetune_config(args, variant, data, provenance)
        for path in (args.output / args.dname / variant).glob('seed_*/metrics.json'):
            completed_metrics(path.parent, expected, ('training.csv', 'best_checkpoint.pt'))
    for seed in range(args.seed_offset, args.seed_offset + args.num_seeds):
        masks = make_masks(data, args, seed)
        for variant in args.variants:
            finetune_run(data, masks, args, variant, seed,
                         args.output / args.dname / variant / f'seed_{seed}', checkpoint, provenance)
            write_reports(args.output, plots=False)
    result = write_reports(args.output)
    print(f'[H-IR pretraining benchmark] {args.output}')
    return result


def main(argv=None):
    run_benchmark(parse_args(argv))


if __name__ == '__main__':
    main()
