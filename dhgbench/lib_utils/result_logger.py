import csv
import os


NODE_CLS_METRICS = ['acc', 'f1_score', 'auc', 'parity', 'equality']
SPLITS = ['train', 'val', 'test']

MODEL_CONFIG_FIELDS = [
    'lr',
    'wd',
    'epochs',
    'dropout',
    'All_num_layers',
    'MLP_hidden',
    'MLP_num_layers',
    'decoder_hidden',
    'decoder_num_layer',
    'alpha',
    'aggregate',
    'normalization',
    'activation',
    'order_fusion',
    'max_exact_order',
    'order_heads',
]

RUN_FIELDS = [
    'run_id',
    'timestamp',
    'dataset',
    'task_type',
    'method',
    'seed',
] + [
    f'{split}_{metric}'
    for metric in NODE_CLS_METRICS
    for split in SPLITS
] + [
    'train_time',
    'num_nodes',
    'num_hyperedges',
    'num_features',
    'num_classes',
    'num_seeds',
] + MODEL_CONFIG_FIELDS

SUMMARY_FIELDS = [
    'run_id',
    'timestamp',
    'dataset',
    'task_type',
    'method',
    'num_seeds',
] + [
    f'{split}_{metric}_{stat}'
    for metric in NODE_CLS_METRICS
    for split in SPLITS
    for stat in ['mean', 'std']
] + [
    'avg_train_time',
    'std_train_time',
] + MODEL_CONFIG_FIELDS


class ResultLogger:
    def __init__(self, args, run_id, timestamp):
        self.args = args
        self.run_id = run_id
        self.timestamp = timestamp
        self.enabled = getattr(args, 'save_results', True)
        self.results_dir = getattr(args, 'results_dir', './results')
        self.runs_path = os.path.join(self.results_dir, 'node_cls_runs.csv')
        self.summary_path = os.path.join(self.results_dir, 'node_cls_summary.csv')

        if self.enabled:
            os.makedirs(self.results_dir, exist_ok=True)

    def _base_row(self):
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'dataset': getattr(self.args, 'dname', ''),
            'task_type': getattr(self.args, 'task_type', ''),
            'method': getattr(self.args, 'method', ''),
            'num_seeds': getattr(self.args, 'num_seeds', ''),
        }

    def _config_row(self):
        row = {
            field: getattr(self.args, field, '')
            for field in MODEL_CONFIG_FIELDS
        }
        if getattr(self.args, 'method', '') != 'OrderSplitEDHNN':
            row['order_fusion'] = ''
            row['max_exact_order'] = ''
            row['order_heads'] = ''
        return row

    @staticmethod
    def _metric_value(metrics, metric, index):
        values = metrics.get(metric)
        if values is None:
            return ''
        try:
            return values[index]
        except (IndexError, TypeError):
            return ''

    @staticmethod
    def _append_row(path, fieldnames, row):
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({field: row.get(field, '') for field in fieldnames})

    def log_node_cls_seed(self, seed, metrics, train_time, data):
        if not self.enabled:
            return

        row = self._base_row()
        row['seed'] = seed
        for metric in NODE_CLS_METRICS:
            for index, split in enumerate(SPLITS):
                row[f'{split}_{metric}'] = self._metric_value(metrics, metric, index)
        row.update({
            'train_time': train_time,
            'num_nodes': getattr(data, 'num_nodes', ''),
            'num_hyperedges': getattr(data, 'num_hyperedges', ''),
            'num_features': getattr(data, 'num_features', ''),
            'num_classes': getattr(data, 'num_classes', ''),
        })
        row.update(self._config_row())
        self._append_row(self.runs_path, RUN_FIELDS, row)

    def log_node_cls_summary(
        self,
        metric_statistics,
        avg_train_time,
        std_train_time,
    ):
        if not self.enabled:
            return

        row = self._base_row()
        for metric in NODE_CLS_METRICS:
            statistics = metric_statistics.get(metric)
            for index, split in enumerate(SPLITS):
                if statistics is None:
                    mean, std = '', ''
                else:
                    means, stds = statistics
                    mean = means[index] if index < len(means) else ''
                    std = stds[index] if index < len(stds) else ''
                row[f'{split}_{metric}_mean'] = mean
                row[f'{split}_{metric}_std'] = std
        row['avg_train_time'] = avg_train_time
        row['std_train_time'] = std_train_time
        row.update(self._config_row())
        self._append_row(self.summary_path, SUMMARY_FIELDS, row)
