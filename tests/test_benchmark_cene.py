"""Resume/report regression tests; subprocesses are mocked, no training."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dhgbench'))
import benchmark_cene as bench


class BenchmarkCENEResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.run_dir = self.output / 'seeds' / 'cora' / 'EDHNN' / '0'
        self.run_dir.mkdir(parents=True)
        self.command = ['python', 'main.py', '--seed_offset=0']
        self.row = dict(dataset='cora', method='EDHNN', seed='0', test_acc='75.0',
                        parameter_count='100', lambda_value='', train_time='1.2')

    def save(self, rows=None, command=None):
        bench.write_rows(self.run_dir / 'node_cls_runs.csv',
                         [self.row] if rows is None else rows, list(self.row))
        (self.run_dir / 'command.json').write_text(
            json.dumps(self.command if command is None else command), encoding='utf-8')

    def valid(self):
        return bench.valid_result(self.run_dir, 'cora', 'EDHNN', 0, self.command)

    def test_result_and_command_must_match(self):
        self.assertIsNone(self.valid())
        self.save()
        self.assertIsNotNone(self.valid())
        for field, value in [('dataset', 'pubmed'), ('method', 'CENE'), ('seed', '1'),
                             ('test_acc', ''), ('test_acc', 'nan'), ('test_acc', 'inf'),
                             ('test_acc', 'bad')]:
            with self.subTest(field=field, value=value):
                self.save([dict(self.row, **{field: value})])
                self.assertIsNone(self.valid())
        self.save(command=['different command'])
        self.assertIsNone(self.valid())
        self.save([self.row, self.row])
        self.assertIsNone(self.valid())

    def test_corrupt_and_incomplete_files_are_not_complete(self):
        self.save()
        (self.run_dir / 'command.json').write_text('{broken', encoding='utf-8')
        self.assertIsNone(self.valid())
        self.save()
        (self.run_dir / 'node_cls_runs.csv').write_bytes(b'\xff')
        self.assertIsNone(self.valid())
        self.save()
        (self.run_dir / 'node_cls_runs.csv').write_text(
            'dataset,method,seed,test_acc\ncora,EDHNN,0\n', encoding='utf-8')
        self.assertIsNone(self.valid())

    def test_report_uses_only_valid_seeds_and_leaves_f1_empty(self):
        self.save()
        second = self.output / 'seeds' / 'cora' / 'EDHNN' / '1'
        second.mkdir()
        bench.write_rows(second / 'node_cls_runs.csv', [dict(self.row, seed='1', test_acc='nan')], list(self.row))
        (second / 'command.json').write_text(json.dumps(self.command), encoding='utf-8')
        commands = {('cora', 'EDHNN', seed): self.command for seed in range(2)}
        bench.report(self.output, ['cora'], ['EDHNN'], 2, [], commands)
        summary = bench.read_rows(self.output / 'cene_summary.csv')[0]
        self.assertEqual(summary['completed_seeds'], '1')
        self.assertEqual(summary['expected_seeds'], '2')
        self.assertEqual(summary['test_acc_mean'], '75.0')
        self.assertEqual(summary['test_f1_mean'], '')
        self.assertEqual(summary['status'], 'partial')

    def test_main_cleans_old_rows_continues_after_failure_and_resumes(self):
        self.save([self.row, self.row])

        def launch(command, **kwargs):
            directory = Path(next(item.split('=', 1)[1] for item in command if item.startswith('--results_dir=')))
            self.assertFalse((directory / 'node_cls_runs.csv').exists())
            method = next(item.split('=', 1)[1] for item in command if item.startswith('--method='))
            process = mock.MagicMock()
            process.__enter__.return_value = process
            if method == 'HGNN':
                process.wait.return_value = 1
            else:
                bench.write_rows(directory / 'node_cls_runs.csv', [self.row], list(self.row))
                process.wait.return_value = 0
            return process

        argv = ['--execute', '--datasets', 'cora', '--num-seeds', '1',
                '--device', 'cpu', '--output', str(self.output)]
        with mock.patch.object(bench, 'METHODS', ['HGNN', 'EDHNN']), \
                mock.patch.object(bench, 'config_status', return_value=None), \
                mock.patch.object(bench, 'uses_default', return_value=False), \
                mock.patch.object(bench.subprocess, 'Popen', side_effect=launch) as popen, \
                mock.patch.object(bench, 'tqdm') as tqdm, \
                contextlib.redirect_stdout(io.StringIO()):
            progress = tqdm.return_value.__enter__.return_value
            bench.main(argv)
            self.assertEqual(popen.call_count, 2)
            self.assertEqual(tqdm.call_args.kwargs['total'], 2)
            self.assertEqual(tqdm.call_args.kwargs['initial'], 0)
            self.assertEqual(progress.update.call_args_list, [mock.call(1), mock.call(1)])
            self.assertEqual(progress.set_postfix.call_args.kwargs['Completed'], 1)
            self.assertEqual(progress.set_postfix.call_args.kwargs['Failed'], 1)
            self.assertEqual(progress.set_postfix.call_args.kwargs['Remaining'], 0)
            self.assertEqual(len(bench.read_rows(self.run_dir / 'node_cls_runs.csv')), 1)
            self.assertEqual(len(bench.read_rows(self.output / 'cene_failures.csv')), 1)
            progress.reset_mock()
            bench.main(argv)
            self.assertEqual(popen.call_count, 3)  # Retry failure, skip matching EDHNN.
            self.assertEqual(tqdm.call_args.kwargs['initial'], 1)
            progress.update.assert_called_once_with(1)
            bench.main([item.replace('cpu', 'cuda:1') for item in argv])
            self.assertEqual(popen.call_count, 5)  # Different command requires rerun.

    def test_progress_total_excludes_skips_and_updates_on_success(self):
        def launch(command, **kwargs):
            options = dict(item[2:].split('=', 1) for item in command if item.startswith('--') and '=' in item)
            directory = Path(options['results_dir'])
            row = dict(self.row, dataset=options['dname'], method=options['method'], seed=options['seed_offset'])
            bench.write_rows(directory / 'node_cls_runs.csv', [row], list(row))
            process = mock.MagicMock()
            process.__enter__.return_value = process
            process.wait.return_value = 0
            return process

        def config_status(method, dataset):
            return 'no pubmed configuration' if (method, dataset) == ('HGNN', 'pubmed') else None

        with mock.patch.object(bench, 'METHODS', ['EDHNN', 'HGNN']), \
                mock.patch.object(bench, 'config_status', side_effect=config_status), \
                mock.patch.object(bench, 'uses_default', return_value=False), \
                mock.patch.object(bench.subprocess, 'Popen', side_effect=launch) as popen, \
                mock.patch.object(bench, 'tqdm') as tqdm:
            bench.main(['--execute', '--datasets', 'cora', 'pubmed', '--num-seeds', '2',
                        '--device', 'cpu', '--output', str(self.output)])
            progress = tqdm.return_value.__enter__.return_value
            self.assertEqual(tqdm.call_args.kwargs['total'], 6)  # 3 supported pairs × 2 seeds.
            self.assertEqual(popen.call_count, 6)
            self.assertEqual(progress.update.call_args_list, [mock.call(1)] * 6)
            self.assertEqual(progress.set_postfix.call_args.kwargs['Completed'], 6)
            self.assertEqual(progress.set_postfix.call_args.kwargs['Remaining'], 0)
            tqdm.write.assert_any_call('SKIP pubmed HGNN: no pubmed configuration')
            self.assertFalse(any('--method=' in call.args[0] for call in tqdm.write.call_args_list))
            tqdm.return_value.__exit__.assert_called_once()

    def test_dry_run_prints_commands_without_training_or_progress(self):
        with mock.patch.object(bench, 'METHODS', ['EDHNN']), \
                mock.patch.object(bench, 'config_status', return_value=None), \
                mock.patch.object(bench, 'uses_default', return_value=False), \
                mock.patch.object(bench.subprocess, 'Popen') as popen, \
                mock.patch.object(bench, 'tqdm') as tqdm:
            bench.main(['--execute', '--dry-run', '--datasets', 'cora', '--num-seeds', '2',
                        '--output', str(self.output)])
            popen.assert_not_called()
            tqdm.assert_not_called()
            self.assertEqual(tqdm.write.call_count, 2)
            self.assertIn('--seed_offset=1', tqdm.write.call_args.args[0])
            self.assertFalse((self.output / 'cene_summary.csv').exists())

    def test_interrupt_closes_progress_and_preserves_completed_result(self):
        with mock.patch.object(bench, 'uses_default', return_value=False):
            command = bench.build_command('cora', 'EDHNN', 0, 'cpu', self.run_dir)
        self.save(command=command)
        process = mock.MagicMock()
        process.__enter__.return_value = process
        process.wait.side_effect = [KeyboardInterrupt, 0]
        with mock.patch.object(bench, 'METHODS', ['EDHNN']), \
                mock.patch.object(bench, 'config_status', return_value=None), \
                mock.patch.object(bench, 'uses_default', return_value=False), \
                mock.patch.object(bench.subprocess, 'Popen', return_value=process), \
                mock.patch.object(bench, 'tqdm') as tqdm:
            bench.main(['--execute', '--datasets', 'cora', '--num-seeds', '2',
                        '--device', 'cpu', '--output', str(self.output)])
            process.terminate.assert_called_once()
            tqdm.return_value.__exit__.assert_called_once()
            self.assertIsNotNone(bench.valid_result(self.run_dir, 'cora', 'EDHNN', 0, command))
            self.assertEqual(len(bench.read_rows(self.output / 'cene_runs.csv')), 1)


if __name__ == '__main__':
    unittest.main()
