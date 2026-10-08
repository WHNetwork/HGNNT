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
                contextlib.redirect_stdout(io.StringIO()):
            bench.main(argv)
            self.assertEqual(popen.call_count, 2)
            self.assertEqual(len(bench.read_rows(self.run_dir / 'node_cls_runs.csv')), 1)
            self.assertEqual(len(bench.read_rows(self.output / 'cene_failures.csv')), 1)
            bench.main(argv)
            self.assertEqual(popen.call_count, 3)  # Retry failure, skip matching EDHNN.
            bench.main([item.replace('cpu', 'cuda:1') for item in argv])
            self.assertEqual(popen.call_count, 5)  # Different command requires rerun.


if __name__ == '__main__':
    unittest.main()
