"""Sequential ENE dataset workers and batch-local result aggregation.

This module uses only the standard library until configuration is requested.
Accuracy values in the worker protocol and batch CSVs are fractions (0..1).
"""

import copy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import traceback


SUMMARY_FIELDS = [
    'batch_id', 'run_id', 'dataset', 'status', 'method',
    'num_seeds_expected', 'num_seeds_completed', 'layers', 'epochs',
    'dropout', 'weight_decay', 'hidden_dim', 'learning_rate',
    'train_acc_mean', 'train_acc_std', 'val_acc_mean', 'val_acc_std',
    'test_acc_mean', 'test_acc_std', 'avg_train_time', 'error_message', 'log_path',
]
RUN_FIELDS = ['batch_id', 'run_id', 'dataset', 'status', 'seed', 'best_epoch',
              'best_val_acc', 'train_acc', 'val_acc', 'test_acc', 'train_time']
CUDA_OOM = re.compile(r'\bCUDA\s+(?:error:\s*)?out of memory\b|^torch\.cuda\.OutOfMemoryError:', re.I | re.M)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _config(args):
    return dict(num_seeds_expected=args.num_seeds, seed_offset=args.seed_offset,
                layers=getattr(args, 'All_num_layers', ''), epochs=args.epochs,
                dropout=args.dropout, weight_decay=args.wd,
                hidden_dim=getattr(args, 'MLP_hidden', ''), learning_rate=args.lr)


def configure_worker(args):
    path = Path(args.ene_batch_output)
    state = json.loads(path.read_text(encoding='utf-8'))
    state['config'] = _config(args)
    _write_json(path, state)


def record_seed(args, run_id, seed, metrics, train_time, best_epoch, best_val_acc):
    path = Path(args.ene_batch_output)
    state = json.loads(path.read_text(encoding='utf-8'))
    # The existing Evaluator reports percentages. Convert exactly once here.
    train, valid, test = (float(value) / 100.0 for value in metrics['acc'])
    state['run_id'] = run_id
    state['runs'].append(dict(dataset=args.dname, run_id=run_id, seed=seed,
                              best_epoch=best_epoch, best_val_acc=best_val_acc,
                              train_acc=train, val_acc=valid, test_acc=test,
                              train_time=float(train_time)))
    _write_json(path, state)


def run_worker(args, run_single):
    path = Path(args.ene_batch_output)
    _write_json(path, dict(dataset=args.dname, status='RUNNING', run_id='', runs=[], config={}))
    try:
        run_single(args)
    except Exception as error:
        torch = sys.modules.get('torch')
        oom_type = getattr(getattr(torch, 'cuda', None), 'OutOfMemoryError', ())
        is_oom = isinstance(error, oom_type) or bool(CUDA_OOM.search(str(error)))
        state = json.loads(path.read_text(encoding='utf-8'))
        state.update(status='OOM' if is_oom else 'ERROR',
                     error_message=f'{type(error).__name__}: {error}')
        _write_json(path, state)
        traceback.print_exc()
        return 1
    state = json.loads(path.read_text(encoding='utf-8'))
    state['status'] = 'SUCCESS'
    _write_json(path, state)
    return 0


def _child_arguments(arguments):
    """Keep explicit training options; strip scheduling and output options."""
    removed = ('--all_datasets', '--datasets', '--dname', '--results_dir', '--ene_batch_output')
    result = []
    position = 0
    while position < len(arguments):
        token = arguments[position]
        option = token.split('=', 1)[0]
        name = next((flag for flag in removed if option.startswith('--') and flag.startswith(option)), None)
        if name is None:
            result.append(token)
            position += 1
            continue
        position += 1
        if name == '--datasets':
            while position < len(arguments) and not arguments[position].startswith('--'):
                position += 1
        elif name != '--all_datasets' and '=' not in token:
            position += 1
    return result


def _write_csv(path, fields, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _print_table(rows):
    headers = ['Dataset', 'Status', 'Seeds', 'Train Acc', 'Valid Acc', 'Test Acc', 'Time']
    table = [headers]
    ordered = sorted((row for row in rows if row['status'] == 'SUCCESS'),
                     key=lambda row: row['test_acc_mean'], reverse=True)
    ordered += [row for row in rows if row['status'] != 'SUCCESS']
    for row in ordered:
        values = []
        for split in ('train', 'val', 'test'):
            values.append(f"{100 * row[split + '_acc_mean']:.2f}±{100 * row[split + '_acc_std']:.2f}%"
                          if row['status'] == 'SUCCESS' else '—')
        table.append([row['dataset'], row['status'],
                      f"{row['num_seeds_completed']}/{row['num_seeds_expected']}", *values,
                      f"{row['avg_train_time']:.2f}s" if row['status'] == 'SUCCESS' else '—'])
    widths = [max(len(line[column]) for line in table) for column in range(len(headers))]
    for line in table:
        print(' | '.join(value.ljust(width) for value, width in zip(line, widths)))


def run_batch(args):
    from lib_dataset import _single_datasets_
    from parameter_parser import method_config

    datasets = list(dict.fromkeys(args.datasets if args.datasets is not None else _single_datasets_))
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = Path(args.results_dir).resolve() / 'ene_batch' / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'logs').mkdir()
    (directory / 'datasets').mkdir()
    main_path = Path(__file__).resolve().parents[1] / 'main.py'
    passthrough = _child_arguments(args._raw_cli)
    summaries, runs = [], []

    def save():
        _write_csv(directory / 'summary.csv', SUMMARY_FIELDS, summaries)
        _write_csv(directory / 'runs.csv', RUN_FIELDS, runs)

    save()
    print(f'[ENE batch] {directory}', flush=True)
    for position, dataset in enumerate(datasets, 1):
        label = re.sub(r'[^A-Za-z0-9_.-]', '_', dataset)
        log_path = directory / 'logs' / f'{position:03d}_{label}.log'
        row = dict(batch_id=batch_id, run_id='', dataset=dataset, status='ERROR', method='ENE',
                   num_seeds_expected=args.num_seeds, num_seeds_completed=0,
                   log_path=str(log_path), error_message='')
        try:
            configured = copy.deepcopy(args)
            configured.dname = dataset
            configured._ene_cli_overrides = dict(args._ene_cli_overrides, dname=dataset)
            configured = method_config(configured)
            row.update(_config(configured))
            if dataset not in _single_datasets_ or dataset not in args._dname_choices:
                row.update(status='UNSUPPORTED', error_message='Not registered for node_cls or absent from CLI dname choices')
                log_path.write_text(row['error_message'] + '\n', encoding='utf-8')
            else:
                worker_dir = directory / 'datasets' / f'{position:03d}_{label}'
                worker_dir.mkdir()
                output = worker_dir / 'worker.json'
                command = [sys.executable, '-u', str(main_path), *passthrough,
                           f'--dname={dataset}', f'--results_dir={worker_dir}',
                           f'--ene_batch_output={output}']
                print(f'[ENE batch {position}/{len(datasets)}] {dataset}', flush=True)
                with log_path.open('w', encoding='utf-8') as log:
                    log.write('Command: ' + repr(command) + '\n')
                    log.flush()
                    process = subprocess.run(command, cwd=main_path.parent, stdout=log,
                                             stderr=subprocess.STDOUT, check=False,
                                             env=dict(os.environ, PYTHONIOENCODING='utf-8'))
                state = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
                row.update(state.get('config', {}))
                run_id = state.get('run_id', '')
                completed = [run for run in state.get('runs', [])
                             if run_id and run.get('run_id') == run_id and run.get('dataset') == dataset]
                expected = list(range(configured.seed_offset, configured.seed_offset + row['num_seeds_expected']))
                complete = (bool(expected) and sorted(run['seed'] for run in completed) == expected
                            and len(completed) == len(state.get('runs', []))
                            and state.get('dataset') == dataset)
                row.update(run_id=run_id, num_seeds_completed=len(completed))
                status = state.get('status')
                if process.returncode == 0 and status == 'SUCCESS' and complete:
                    row['status'] = 'SUCCESS'
                    for split in ('train', 'val', 'test'):
                        values = [run[split + '_acc'] for run in completed]
                        row[split + '_acc_mean'] = statistics.mean(values)
                        row[split + '_acc_std'] = statistics.pstdev(values)
                    row['avg_train_time'] = statistics.mean(run['train_time'] for run in completed)
                else:
                    # A killed process without explicit CUDA OOM evidence is ERROR.
                    text = log_path.read_text(encoding='utf-8', errors='replace')
                    row['status'] = 'OOM' if status == 'OOM' or CUDA_OOM.search(text) else 'ERROR'
                    row['error_message'] = state.get('error_message') or (
                        f'Child exit={process.returncode}; completed {len(completed)}/{len(expected)} seeds. '
                        + (text.strip().splitlines()[-1] if text.strip() else 'No worker result.'))
                runs.extend(dict(run, batch_id=batch_id, status=row['status']) for run in completed)
        except Exception as error:
            row.update(status='ERROR', error_message=f'{type(error).__name__}: {error}')
            with log_path.open('a', encoding='utf-8') as log:
                traceback.print_exc(file=log)
        finally:
            summaries.append(row)
            save()
        print(f"[ENE batch] {dataset}: {row['status']} ({row['num_seeds_completed']}/{row['num_seeds_expected']})", flush=True)
    _print_table(summaries)
    print(f'[ENE batch] summary.csv and runs.csv: {directory}')
