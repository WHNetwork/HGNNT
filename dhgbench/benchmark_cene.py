"""Reproducible node-classification comparison. Training requires --execute."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

from lib_dataset import _single_datasets_
from lib_models import _semi_methods_


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ['cora', 'coauthor_cora', 'coauthor_dblp', 'NTU2012', 'house-committees-100']
ABLATIONS = ['ENE', 'ENE-Excl', 'CENE', 'CENE-Shuffle']
METHODS = list(_semi_methods_) + ['EDHNN-Depth2'] + [f'CENE:{mode}' for mode in ABLATIONS if mode != 'CENE']
CLI_UNAVAILABLE = {'walmart-trips', 'house-committees', 'magpm_mini'}


def config_status(method, dataset):
    base = 'EDHNN' if method == 'EDHNN-Depth2' else method.split(':')[0]
    path = ROOT / 'lib_yamls' / 'node_yamls' / f'config_{base.lower()}.yaml'
    if not path.is_file():
        return f'missing config: {path.name}'
    try:
        configs = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        return f'invalid config: {exc}'
    if not isinstance(configs, dict):
        return 'invalid YAML mapping'
    selected = dataset if dataset in configs else 'default'
    if selected not in configs:
        return f'no {dataset} configuration'
    if not isinstance(configs[selected], dict):
        return f'invalid {selected} configuration'
    return None


def uses_default(method, dataset):
    base = 'EDHNN' if method == 'EDHNN-Depth2' else method.split(':')[0]
    path = ROOT / 'lib_yamls' / 'node_yamls' / f'config_{base.lower()}.yaml'
    return dataset not in yaml.safe_load(path.read_text(encoding='utf-8'))


def read_rows(path):
    if not path.is_file():
        return []
    try:
        with path.open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream, strict=True))
    except (OSError, UnicodeError, csv.Error):
        return []


def valid_result(run_dir, dataset, method, seed, command):
    """A seed is complete only when its single result and command match."""
    rows = read_rows(run_dir / 'node_cls_runs.csv')
    if len(rows) != 1:
        return None
    row = rows[0]
    base = 'EDHNN' if method == 'EDHNN-Depth2' else method.split(':')[0]
    try:
        accuracy = float(row['test_acc'])
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
            return None
        if (row['dataset'] != dataset or row['method'] != base
                or int(row['seed']) != seed or None in row
                or any(value is None for value in row.values())):
            return None
        if base == 'CENE':
            mode = method.split(':', 1)[1] if ':' in method else 'CENE'
            if row.get('cene_mode') != mode:
                return None
        historical = json.loads((run_dir / 'command.json').read_text(encoding='utf-8'))
        if historical != command:
            return None
    except (KeyError, ValueError, TypeError, OSError, UnicodeError):
        return None
    return row


def build_command(dataset, method, seed, device, run_dir):
    base = 'EDHNN' if method == 'EDHNN-Depth2' else method.split(':')[0]
    command = [sys.executable, str(ROOT / 'main.py'), f'--dname={dataset}',
               '--task_type=node_cls', f'--method={base}', '--num_seeds=1',
               f'--seed_offset={seed}', f'--device={device}',
               '--cene_benchmark', f'--results_dir={run_dir}']
    if uses_default(method, dataset):
        command.append('--is_default=True')
    if method == 'EDHNN-Depth2':
        command.append('--edhnn_depth_match')
    if base == 'OrderSplitEDHNN':
        command.append('--order_fusion=cross_attn')
    if base == 'CENE':
        mode = method.split(':', 1)[1] if ':' in method else 'CENE'
        command.extend([f'--cene_mode={mode}', f'--cene_shuffle_seed={seed}'])
    return command


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def numeric(rows, field):
    values = []
    for row in rows:
        try:
            value = float(row.get(field, ''))
        except (ValueError, TypeError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean_std(values):
    if not values:
        return '', ''
    mean = sum(values) / len(values)
    return mean, (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def report(output, datasets, methods, seeds, failures, commands):
    runs = []
    summary = []
    comparisons = []
    for dataset in datasets:
        for method in methods:
            group = []
            for seed in range(seeds):
                run_dir = output / 'seeds' / dataset / method.replace(':', '_') / str(seed)
                command = commands.get((dataset, method, seed))
                result = valid_result(run_dir, dataset, method, seed, command) if command else None
                if result:
                    row = dict(result)
                    row['base_method'] = row['method']
                    row['method'] = method
                    group.append(row)
                    runs.append(row)
            acc_mean, acc_std = mean_std(numeric(group, 'test_acc'))
            f1_mean, f1_std = mean_std(numeric(group, 'test_f1_score'))
            time_mean, time_std = mean_std(numeric(group, 'train_time'))
            params = numeric(group, 'parameter_count')
            status = 'complete' if len(group) == seeds else ('partial' if group else 'missing')
            reason = next((f['reason'] for f in failures if f['dataset'] == dataset and f['method'] == method), '')
            if reason and not group:
                status = 'unsupported' if reason.startswith(('missing config', 'no ', 'invalid', 'dataset ')) else 'failed'
            item = dict(dataset=dataset, method=method, completed_seeds=len(group), expected_seeds=seeds,
                        test_acc_mean=acc_mean, test_acc_std=acc_std,
                        test_f1_mean=f1_mean, test_f1_std=f1_std,
                        train_time_mean=time_mean, train_time_std=time_std,
                        parameter_count=int(params[0]) if params else '', status=status, reason=reason)
            summary.append(item)
            comparisons.append(item.copy())
    edhnn = {row['dataset']: row['test_acc_mean'] for row in comparisons
             if row['method'] == 'EDHNN' and row['status'] == 'complete'}
    for row in comparisons:
        ref = edhnn.get(row['dataset'])
        row['delta_edhnn_pp'] = row['test_acc_mean'] - ref if ref != '' and ref is not None and row['status'] == 'complete' else ''
    run_fields = list(dict.fromkeys(key for row in runs for key in row)) or ['dataset', 'method', 'seed']
    summary_fields = list(summary[0]) if summary else ['dataset', 'method', 'status']
    write_rows(output / 'cene_runs.csv', runs, run_fields)
    write_rows(output / 'cene_summary.csv', summary, summary_fields)
    write_rows(output / 'cene_comparison.csv', comparisons, summary_fields + ['delta_edhnn_pp'])
    write_rows(output / 'cene_failures.csv', failures, ['dataset', 'method', 'seed', 'reason'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='required to start training')
    parser.add_argument('--dry-run', action='store_true', help='print commands without training')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--all-datasets', action='store_true')
    parser.add_argument('--num-seeds', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'cene_benchmark')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.num_seeds < 1:
        parser.error('--num-seeds must be positive')
    datasets = list(dict.fromkeys(_single_datasets_ if args.all_datasets else args.datasets))
    unknown = set(datasets) - set(_single_datasets_)
    if unknown:
        parser.error(f'not node_cls datasets: {sorted(unknown)}')
    if not args.execute and not args.dry_run:
        parser.error('specify --execute to train or --dry-run to inspect commands')
    failures = []
    commands = {}
    for dataset in datasets:
        for method in METHODS:
            if dataset in CLI_UNAVAILABLE or config_status(method, dataset):
                continue
            for seed in range(args.num_seeds):
                run_dir = args.output / 'seeds' / dataset / method.replace(':', '_') / str(seed)
                commands[dataset, method, seed] = build_command(dataset, method, seed, args.device, run_dir)
    for dataset in datasets:
        for method in METHODS:
            if dataset in CLI_UNAVAILABLE:
                reason = f'dataset {dataset} is absent from parameter_parser --dname choices'
                failures.append(dict(dataset=dataset, method=method, seed='', reason=reason))
                print(f'SKIP {dataset} {method}: {reason}')
                continue
            reason = config_status(method, dataset)
            if reason:
                failures.append(dict(dataset=dataset, method=method, seed='', reason=reason))
                print(f'SKIP {dataset} {method}: {reason}')
                continue
            for seed in range(args.num_seeds):
                run_dir = args.output / 'seeds' / dataset / method.replace(':', '_') / str(seed)
                command = commands[dataset, method, seed]
                if valid_result(run_dir, dataset, method, seed, command):
                    print(f'DONE {dataset} {method} seed={seed}')
                    continue
                print(subprocess.list2cmdline(command), flush=True)
                if args.dry_run:
                    continue
                try:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    # ResultLogger appends: remove only this seed's old CSV files.
                    for filename in ('node_cls_runs.csv', 'node_cls_summary.csv'):
                        (run_dir / filename).unlink(missing_ok=True)
                    (run_dir / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
                    with (run_dir / 'stdout.log').open('w', encoding='utf-8') as stream:
                        with subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT) as process:
                            try:
                                returncode = process.wait()
                            except KeyboardInterrupt:
                                process.terminate()
                                process.wait()
                                raise
                    if returncode:
                        # A failed process cannot leave a resumable success row.
                        (run_dir / 'node_cls_runs.csv').unlink(missing_ok=True)
                    if returncode or not valid_result(run_dir, dataset, method, seed, command):
                        lines = (run_dir / 'stdout.log').read_text(encoding='utf-8', errors='replace').splitlines()
                        detail = next((line.strip() for line in reversed(lines) if line.strip()), 'no output')
                        raise RuntimeError(f'exit code {returncode}; invalid or missing result: {detail}; see {run_dir / "stdout.log"}')
                except KeyboardInterrupt:
                    (run_dir / 'node_cls_runs.csv').unlink(missing_ok=True)
                    failures.append(dict(dataset=dataset, method=method, seed=seed, reason='interrupted by user'))
                    report(args.output, datasets, METHODS, args.num_seeds, failures, commands)
                    print('Stopped. Re-run the same command to resume.', flush=True)
                    return
                except (OSError, RuntimeError) as exc:
                    reason = str(exc)
                    failures.append(dict(dataset=dataset, method=method, seed=seed, reason=reason))
                    print(f'FAIL {dataset} {method} seed={seed}: {reason}', flush=True)
    if args.execute and not args.dry_run:
        report(args.output, datasets, METHODS, args.num_seeds, failures, commands)


if __name__ == '__main__':
    main()
