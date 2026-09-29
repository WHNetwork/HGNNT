import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dhgbench'))

from lib_utils.result_logger import (  # noqa: E402
    RUN_FIELDS,
    SUMMARY_FIELDS,
    ResultLogger,
)


def make_args(results_dir, method='OrderSplitEDHNN', save_results=True):
    return SimpleNamespace(
        save_results=save_results,
        results_dir=results_dir,
        dname='house-committees-100',
        task_type='node_cls',
        method=method,
        num_seeds=2,
        lr=0.001,
        wd=0.0,
        epochs=200,
        dropout=0.5,
        All_num_layers=2,
        MLP_hidden=512,
        MLP_num_layers=1,
        decoder_hidden=256,
        decoder_num_layer=1,
        alpha=0.2,
        aggregate='mean',
        normalization='ln',
        activation='prelu',
        order_fusion='attn',
        max_exact_order=8,
        order_heads=4,
    )


DATA = SimpleNamespace(
    num_nodes=100,
    num_hyperedges=25,
    num_features=16,
    num_classes=3,
)

METRICS = {
    'acc': [0.8, 0.7, 0.6],
    'f1_score': [0.75, 0.65, 0.55],
}

STATISTICS = {
    'acc': ([0.85, 0.75, 0.65], [0.05, 0.05, 0.05]),
    'f1_score': ([0.8, 0.7, 0.6], [0.05, 0.05, 0.05]),
}


class ResultLoggerTest(unittest.TestCase):
    def test_create_headers_append_and_method_specific_fields(self):
        with tempfile.TemporaryDirectory() as results_dir:
            logger = ResultLogger(
                make_args(results_dir),
                'run_ordersplit',
                '2026-09-29T09:15:30.123456',
            )
            logger.log_node_cls_seed(0, METRICS, 1.2, DATA)
            logger.log_node_cls_seed(1, METRICS, 1.4, DATA)
            logger.log_node_cls_summary(STATISTICS, 1.3, 0.1)

            with open(logger.runs_path, newline='', encoding='utf-8') as file:
                runs_reader = csv.DictReader(file)
                self.assertEqual(runs_reader.fieldnames, RUN_FIELDS)
                runs = list(runs_reader)
            with open(logger.summary_path, newline='', encoding='utf-8') as file:
                summary_reader = csv.DictReader(file)
                self.assertEqual(summary_reader.fieldnames, SUMMARY_FIELDS)
                summaries = list(summary_reader)

            self.assertEqual(len(runs), 2)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(runs[0]['run_id'], 'run_ordersplit')
            self.assertEqual(runs[0]['order_fusion'], 'attn')
            self.assertEqual(runs[0]['max_exact_order'], '8')
            self.assertEqual(runs[0]['order_heads'], '4')
            self.assertEqual(runs[0]['train_auc'], '')
            self.assertEqual(summaries[0]['test_acc_mean'], '0.65')
            self.assertEqual(summaries[0]['order_fusion'], 'attn')

            edhnn_logger = ResultLogger(
                make_args(results_dir, method='EDHNN'),
                'run_edhnn',
                '2026-09-29T10:00:00.000000',
            )
            edhnn_logger.log_node_cls_seed(0, {'acc': [0.9, 0.8, 0.7]}, 1.0, DATA)
            edhnn_logger.log_node_cls_summary(
                {'acc': ([0.9, 0.8, 0.7], [0.0, 0.0, 0.0])},
                1.0,
                0.0,
            )

            with open(logger.runs_path, newline='', encoding='utf-8') as file:
                appended_runs = list(csv.DictReader(file))
            with open(logger.summary_path, newline='', encoding='utf-8') as file:
                appended_summaries = list(csv.DictReader(file))

            self.assertEqual(len(appended_runs), 3)
            self.assertEqual(len(appended_summaries), 2)
            self.assertEqual(appended_runs[-1]['run_id'], 'run_edhnn')
            self.assertEqual(appended_runs[-1]['order_fusion'], '')
            self.assertEqual(appended_runs[-1]['max_exact_order'], '')
            self.assertEqual(appended_runs[-1]['order_heads'], '')
            self.assertEqual(appended_summaries[-1]['order_fusion'], '')
            self.assertEqual(appended_summaries[-1]['max_exact_order'], '')
            self.assertEqual(appended_summaries[-1]['order_heads'], '')

    def test_disabled_logger_creates_no_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = str(Path(temp_dir) / 'results')
            logger = ResultLogger(
                make_args(results_dir, save_results=False),
                'disabled_run',
                '2026-09-29T11:00:00.000000',
            )

            logger.log_node_cls_seed(0, METRICS, 1.0, DATA)
            logger.log_node_cls_summary(STATISTICS, 1.0, 0.0)

            self.assertFalse(Path(results_dir).exists())


if __name__ == '__main__':
    unittest.main()
