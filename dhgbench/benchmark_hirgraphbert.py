"""One command: joint SSL, paired Scratch/Pretrained seeds and historical references."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from hirgraphbert_cli import parse_args
from lib_utils.hirbert_pretrain_utils import save_json
from lib_utils.hirgraphbert_data import load_graph
from lib_utils.hirgraphbert_training import finetune_run, run_pretraining, write_csv


def historical_results(roots, output):
    """Read old results as references, without rewriting or rerunning old models."""
    rows, seen = [], set()
    for root in roots:
        paths = ([root] if root.is_file() else sorted(root.rglob('metrics.json')) +
                 sorted(path for path in root.rglob('*summary.csv') if path.name in
                        ('node_cls_summary.csv', 'ablation_summary.csv', 'finetune_summary.csv')))
        for path in paths:
            resolved = path.resolve()
            if resolved in seen or output.resolve() in resolved.parents:
                continue
            seen.add(resolved)
            if path.suffix == '.json':
                result = json.loads(path.read_text(encoding='utf-8'))
                variant = result.get('variant', '')
                if result.get('model') == 'HIR-GraphBERT' or variant not in (
                        'full', 'full_scratch', 'full_pretrained', 'feature_only'):
                    continue
                rows.append({'dataset': result.get('dataset', path.parent.parent.parent.name),
                             'model': 'Feature Only' if variant == 'feature_only' else 'HIRBERT',
                             'variant': variant, 'seed': result.get('seed', ''), 'test_acc': result.get('test_acc', ''),
                             'input_mode': 'Feature-only' if variant == 'feature_only' else 'Feature + Structure',
                             'source': str(path), 'status': 'historical reference'})
            elif path.suffix == '.csv':
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    for result in csv.DictReader(stream):
                        method = result.get('method', result.get('variant', ''))
                        if method.lower() in ('hirbert', 'full', 'full_scratch', 'full_pretrained', 'feature_only', 'feature only'):
                            feature = method.lower() in ('feature_only', 'feature only')
                            rows.append({**result, 'model': 'Feature Only' if feature else 'HIRBERT',
                                         'input_mode': 'Feature-only' if feature else 'Feature + Structure',
                                         'source': str(path), 'status': 'historical reference'})
    return rows


def write_reports(args):
    rows = []
    for path in sorted(args.output.glob('*/*/seed_*/metrics.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        if result.get('completed') and result.get('model') == 'HIR-GraphBERT':
            rows.append(result)
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['variant']].append(row)
    summaries, comparisons = [], []
    for (dataset, variant), group in groups.items():
        protocols = [{key: value for key, value in row['config'].items() if key != 'seed'} for row in group]
        if any(protocol != protocols[0] for protocol in protocols[1:]):
            raise ValueError(f'Different seed protocols in {dataset}/{variant}; use separate output directories')
        summaries.append({'dataset': dataset, 'model': 'HIR-GraphBERT', 'input_mode': 'Structure-only',
                          'variant': variant, 'num_seeds': len(group), 'seeds': ','.join(str(row['seed']) for row in group),
                          'test_acc_mean': mean(row['test_acc'] for row in group),
                          'test_acc_std': pstdev(row['test_acc'] for row in group),
                          'val_acc_mean': mean(row['val_acc'] for row in group),
                          'val_acc_std': pstdev(row['val_acc'] for row in group),
                          'node_repr_variance_mean': mean(row['node_repr_variance'] for row in group),
                          'parameter_count': group[0]['parameter_count'],
                          'structural_attention_parameters': group[0]['parameter_counts']['structural_attention'],
                          'peak_gpu_memory_mb': max(row['peak_gpu_memory_mb'] for row in group),
                          'peak_gpu_reserved_memory_mb': max(row['peak_gpu_reserved_memory_mb'] for row in group)})
    for dataset in sorted({row['dataset'] for row in rows}):
        scratch = {row['seed']: row['test_acc'] for row in groups[dataset, 'scratch']}
        pretrained = {row['seed']: row['test_acc'] for row in groups[dataset, 'pretrained']}
        seeds = sorted(scratch.keys() & pretrained.keys())
        if seeds:
            deltas = [pretrained[seed] - scratch[seed] for seed in seeds]
            comparisons.append({'dataset': dataset, 'input_mode': 'Structure-only', 'num_paired_seeds': len(seeds),
                                'pretrained_minus_scratch_pp_mean': mean(deltas), 'difference_pp_std': pstdev(deltas)})
    references = historical_results(args.reference_results, args.output)
    write_csv(args.output / 'finetune_runs.csv', [{key: value for key, value in row.items()
                                                if key not in ('config', 'encoder_load_report', 'parameter_counts')} for row in rows])
    write_csv(args.output / 'finetune_summary.csv', summaries)
    write_csv(args.output / 'paired_comparison.csv', comparisons)
    write_csv(args.output / 'historical_references.csv', references)
    save_json({'model': 'HIR-GraphBERT', 'input_mode': 'Structure-only', 'accuracy_unit': 'percent',
               'std_definition': 'population standard deviation across recorded seeds', 'runs': rows,
               'summary': summaries, 'paired_comparisons': comparisons, 'historical_references': references},
              args.output / 'experiment_report.json')
    lines = ['# HIR-GraphBERT — Structure-only', '',
             'Encoder/classifier inputs contain H-IR structure only. Accuracies are percentages; standard deviation is across seeds.', '',
             '| Dataset | Configuration | Seeds | Test accuracy (mean ± std) | Extra structural attention parameters |',
             '| --- | --- | --- | --- | --- |']
    for row in summaries:
        lines.append(f'| {row["dataset"]} | {row["variant"].title()} | {row["seeds"]} | '
                     f'{row["test_acc_mean"]:.2f} ± {row["test_acc_std"]:.2f} | {row["structural_attention_parameters"]} |')
    lines.extend(['', '## Historical references', '',
                  'Original HIRBERT and Feature Only files remain in their source directories. '
                  'Historical runs may use different training/split protocols and are not paired with the new experiments.', ''])
    if references:
        lines.extend(['| Dataset | Model | Variant | Test accuracy | Source |', '| --- | --- | --- | --- | --- |'])
        for row in references:
            lines.append(f'| {row.get("dataset", row.get("dname", ""))} | {row["model"]} | {row.get("variant", "")} | '
                         f'{row.get("test_acc", row.get("test_acc_mean", ""))} | {row["source"]} |')
    else:
        lines.append('No HIRBERT/Feature Only metrics were found under the configured reference paths. No historical scores are fabricated.')
    (args.output / 'EXPERIMENT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows


def run_benchmark(args):
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.pretrained_checkpoint
    if args.mode == 'pretrain' or (args.mode == 'all' and 'pretrained' in args.variants and checkpoint is None):
        checkpoint = run_pretraining(args)
    if args.mode == 'pretrain':
        return checkpoint
    data = load_graph(args, labels=True)
    for seed in range(args.seed_offset, args.seed_offset + args.num_seeds):
        for variant in args.variants:
            finetune_run(data, args, variant, seed, checkpoint)
            write_reports(args)
    rows = write_reports(args)
    print(f'[Structure-only experiment report] {args.output / "EXPERIMENT_REPORT.md"}')
    return rows


def main(argv=None):
    return run_benchmark(parse_args(argv))


if __name__ == '__main__':
    main()
