"""Label-free joint H-IR pretraining with dataset adapters and local views."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from benchmark_hirbert_ablation import ROOT, model_config
from lib_dataset import _single_datasets_
from lib_models.HNN.hirbert import HIRBERT_DEFAULTS
from lib_models.HNN.hirbert_pretrain import HIRPretrainer
from lib_utils.hirbert_pretrain_utils import TASKS, atomic_save, unlabeled_graph
from parameter_parser import method_config, str2bool

DEFAULT_PRETRAIN_DATASETS = ['cora', 'citeseer', 'coauthor_cora', 'NTU2012', 'house-committees-100']


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dname', default='cora', choices=_single_datasets_)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'hirbert_pretrain')
    parser.add_argument('--pretrain_seed', type=int, default=0)
    parser.add_argument('--pretrain_mode', choices=('multi_graph', 'single_graph'), default='multi_graph')
    parser.add_argument('--pretrain_datasets', nargs='+', choices=_single_datasets_)
    parser.add_argument('--exclude_target_from_pretrain', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--local_max_tokens', type=int, default=8192)
    parser.add_argument('--local_max_nodes', type=int, default=2048)
    parser.add_argument('--local_max_chunks', type=int, default=32)
    parser.add_argument('--local_target_count', type=int, default=16, help='Maximum positive membership targets per sample')
    parser.add_argument('--local_feature_targets', type=int, default=32)
    parser.add_argument('--local_relation_targets', type=int, default=32)
    parser.add_argument('--local_min_edges', type=int, default=64, help='Soft neighborhood size target')
    parser.add_argument('--local_max_edges', type=int, default=128)
    parser.add_argument('--local_hops', type=int, default=3)
    parser.add_argument('--local_target_tokens', type=int, default=6144, help='Soft compiled input size target')
    parser.add_argument('--local_samples_per_graph', type=int, default=128)
    parser.add_argument('--validation_samples_per_graph', type=int, default=64)
    parser.add_argument('--micro_batch_size', type=int, default=8, help='Initial maximum packets per forward/backward')
    parser.add_argument('--micro_batch_max_size', type=int, default=64)
    parser.add_argument('--micro_batch_max_tokens', type=int, default=262144)
    parser.add_argument('--micro_batch_max_nodes', type=int, default=32768)
    parser.add_argument('--micro_batch_max_chunks', type=int, default=1024)
    parser.add_argument('--bert_max_batch_tokens', type=int, default=32768, help='Hard real AND padded token cap per BERT call')
    parser.add_argument('--bert_max_chunk_batch_size', type=int, default=64, help='Adaptive ceiling; initial hir_chunk_batch_size stays 16')
    parser.add_argument('--auto_micro_batch', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--gpu_memory_budget_gb', type=float, default=18.0)
    parser.add_argument('--gpu_target_memory_gb', type=float, default=15.0)
    parser.add_argument('--micro_batch_tune_interval', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Nominal microbatches per optimizer update; losses normalized per logical sample')
    parser.add_argument('--amp', choices=('auto', 'bf16', 'fp16', 'off'), default='auto')
    parser.add_argument('--oom_retries', type=int, default=6)
    parser.add_argument('--relation_max_pairs', type=int, default=50000)
    parser.add_argument('--pretrain_epochs', type=int, default=200)
    parser.add_argument('--pretrain_patience', type=int, default=50)
    parser.add_argument('--pretrain_steps_per_epoch', type=int, default=1)
    parser.add_argument('--pretrain_lr', type=float, default=0.0003)
    parser.add_argument('--pretrain_wd', type=float, default=0.01)
    parser.add_argument('--pretrain_scheduler', choices=('cosine', 'linear', 'none'), default='cosine')
    parser.add_argument('--pretrain_warmup_epochs', type=int, default=10)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--lambda_m', type=float, default=1.0)
    parser.add_argument('--lambda_f', type=float, default=0.5)
    parser.add_argument('--lambda_r', type=float, default=0.5)
    parser.add_argument('--membership_mask_ratio', type=float, default=0.15)
    parser.add_argument('--negative_ratio', type=float, default=1.0)
    parser.add_argument('--feature_mask_ratio', type=float, default=0.15)
    parser.add_argument('--feature_loss', choices=('auto', 'binary', 'continuous'), default='auto')
    parser.add_argument('--max_pos_weight', type=float, default=100.0)
    parser.add_argument('--relation_pairs_per_class', type=int, default=128)
    parser.add_argument('--ssl_val_ratio', type=float, default=0.1)
    parser.add_argument('--node_batch_size', type=int, default=256)
    parser.add_argument('--grad_clip', type=float, default=1.0, help='0 disables gradient clipping')
    parser.add_argument('--resume', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--feature_noise', default='0.6')
    parser.add_argument('--display_step', type=int, default=10)
    for key, default in HIRBERT_DEFAULTS.items():
        # These larger defaults belong ONLY to the SSL entry points. main.py,
        # the original benchmark and the model's defaults remain unchanged.
        parser.add_argument('--hir_' + key, type=str2bool if isinstance(default, bool) else type(default),
                            default={'max_seq_length': 512, 'chunk_batch_size': 16}.get(key))
    return parser


def configure_args(args, parser):
    requested = args.pretrain_datasets or (DEFAULT_PRETRAIN_DATASETS if args.pretrain_mode == 'multi_graph' else [args.dname])
    args.requested_pretrain_datasets = list(dict.fromkeys(requested))
    if args.pretrain_mode == 'single_graph' and len(args.requested_pretrain_datasets) != 1:
        parser.error('single_graph requires exactly one pretraining dataset')
    args.excluded_pretrain_datasets = ([args.dname] if args.exclude_target_from_pretrain
                                     and args.dname in requested else [])
    args.pretrain_datasets = [name for name in args.requested_pretrain_datasets
                             if name not in args.excluded_pretrain_datasets]
    if not args.pretrain_datasets:
        parser.error('No pretraining graph remains after excluding the target')
    if (args.local_max_tokens < 96 or args.local_max_nodes < 32 or args.local_max_chunks < 8
            or min(args.local_target_count, args.local_samples_per_graph, args.validation_samples_per_graph, args.gradient_accumulation_steps,
                   args.relation_max_pairs, args.local_feature_targets, args.local_relation_targets,
                   args.local_min_edges, args.local_target_tokens) < 1 or args.oom_retries < 0
            or args.local_max_edges < max(2, args.local_min_edges) or args.local_hops not in (1, 2, 3, 4)):
        parser.error('Invalid local budget, accumulation steps or OOM retry limit')
    if (min(args.micro_batch_size, args.micro_batch_max_size, args.micro_batch_max_tokens,
            args.micro_batch_max_nodes, args.micro_batch_max_chunks, args.bert_max_batch_tokens, args.bert_max_chunk_batch_size,
            args.micro_batch_tune_interval) < 1 or args.micro_batch_size > args.micro_batch_max_size
            or not 0 < args.gpu_target_memory_gb < args.gpu_memory_budget_gb):
        parser.error('Invalid microbatch or GPU memory budget')
    args.weights = dict(zip(TASKS, (args.lambda_m, args.lambda_f, args.lambda_r)))
    for key in TASKS:
        if args.weights[key] < 0:
            parser.error('Loss weights must be nonnegative')
        if key not in args.tasks:
            args.weights[key] = 0.0
    args.tasks = [key for key in TASKS if args.weights[key] > 0]
    if not args.tasks:
        parser.error('Enable at least one task with a positive loss weight')
    if min(args.pretrain_epochs, args.pretrain_patience, args.pretrain_steps_per_epoch,
           args.relation_pairs_per_class, args.node_batch_size, args.display_step) < 1:
        parser.error('Epochs, patience and batch/sample counts must be positive')
    if not all(0 < value <= 1 for value in (args.membership_mask_ratio, args.feature_mask_ratio)):
        parser.error('Masking ratios must be in (0, 1]')
    if not 0 < args.ssl_val_ratio < 1 or args.negative_ratio <= 0:
        parser.error('ssl_val_ratio must be in (0, 1) and negative_ratio must be positive')
    if args.grad_clip < 0 or args.pretrain_lr <= 0 or args.pretrain_wd < 0 or args.max_pos_weight < 1:
        parser.error('Invalid learning rate, weight decay, clipping or positive weight')
    if args.pretrain_seed < 0 or args.pretrain_warmup_epochs < 0:
        parser.error('Seeds and warmup epochs must be nonnegative')
    args.method, args.is_default = 'HIRBERT', False
    args._hirbert_cli_overrides = {
        key: value for key, value in vars(args).items()
        if (key.startswith('hir_') or key in ('epochs', 'lr', 'wd')) and value is not None
    }
    args = method_config(args)
    if args.bert_max_batch_tokens < args.hir_max_seq_length:
        parser.error('bert_max_batch_tokens must fit at least one full BERT chunk')
    if args.bert_max_chunk_batch_size < args.hir_chunk_batch_size:
        parser.error('bert_max_chunk_batch_size must be at least the initial hir_chunk_batch_size')
    if (args.micro_batch_max_tokens < args.local_max_tokens or args.micro_batch_max_nodes < args.local_max_nodes
            or args.micro_batch_max_chunks < args.local_max_chunks):
        parser.error('Microbatch limits must fit one legal local packet')
    args.output = args.output.resolve()
    return args


def parse_args(argv=None):
    parser = build_parser()
    return configure_args(parser.parse_args(argv), parser)


def pretrain_config(args):
    keys = ('pretrain_seed', 'pretrain_epochs', 'pretrain_patience', 'pretrain_steps_per_epoch',
            'pretrain_lr', 'pretrain_wd', 'pretrain_scheduler', 'pretrain_warmup_epochs',
            'membership_mask_ratio', 'negative_ratio', 'feature_mask_ratio', 'feature_loss',
            'max_pos_weight', 'relation_pairs_per_class', 'ssl_val_ratio', 'node_batch_size',
            'grad_clip', 'feature_noise', 'pretrain_mode', 'pretrain_datasets',
            'requested_pretrain_datasets', 'excluded_pretrain_datasets', 'local_max_tokens',
            'local_max_nodes', 'local_max_chunks', 'local_target_count', 'local_samples_per_graph',
            'local_feature_targets', 'local_relation_targets', 'local_min_edges', 'local_max_edges',
            'local_hops', 'local_target_tokens', 'gradient_accumulation_steps', 'amp', 'oom_retries',
            'relation_max_pairs', 'validation_samples_per_graph', 'micro_batch_size', 'micro_batch_max_size',
            'micro_batch_max_tokens', 'micro_batch_max_nodes', 'micro_batch_max_chunks', 'bert_max_batch_tokens', 'bert_max_chunk_batch_size',
            'auto_micro_batch', 'gpu_memory_budget_gb', 'gpu_target_memory_gb', 'micro_batch_tune_interval')
    return {**{key: getattr(args, key) for key in keys}, 'tasks': args.tasks,
            'weights': args.weights, 'model_config': model_config(args), 'optimizer': 'AdamW'}


def load_unlabeled_data(args):
    """Reuse the DHG-Bench dataset IO without its label-processing wrapper."""
    from lib_dataset.convert_datasets_to_pygDataset import dataset_Hypergraph
    from lib_dataset.preprocessing import ExtractV2E
    from torch_geometric.data import Data

    base = ROOT.parent / 'data'
    if args.dname in ('cora', 'citeseer', 'pubmed'):
        raw_path = base / 'trad_data' / 'cocitation'
    elif args.dname in ('coauthor_cora', 'coauthor_dblp'):
        raw_path = base / 'trad_data' / 'coauthorship'
    elif args.dname == 'yelp':
        raw_path = base / 'trad_data' / 'yelp'
    elif args.dname in ('bail', 'credit', 'german'):
        raw_path = base / 'fair_data'
    elif args.dname in ('actor', 'amazon', 'pokec', 'twitch'):
        raw_path = base / 'hete_data'
    elif args.dname in ('amazon_review', 'magpm', 'magpm_mini', 'trivago', 'ogbn_mag'):
        raw_path = base / 'large_data'
    else:
        raw_path = base / 'trad_data'
    noisy = args.dname in ('walmart-trips-100', 'house-committees-100', 'walmart-trips', 'house-committees')
    dataset = dataset_Hypergraph(name=args.dname, root=str(base / 'pyg_data' / 'hypergraph_dataset_updated'),
                                p2raw=str(raw_path), feature_noise=args.feature_noise if noisy else None)
    raw = dataset.data
    count = getattr(raw, 'num_hyperedges', None)
    if count is None:
        count = int(raw.edge_index[0].max()) - len(raw.x) + 1
    # No y, num_classes, train_percent, train_mask or class split is inspected.
    raw = Data(x=raw.x, edge_index=raw.edge_index, n_x=len(raw.x), num_hyperedges=int(count))
    raw = ExtractV2E(raw)
    if raw.edge_index.numel():
        raw.edge_index[1] -= raw.edge_index[1].min()
    graph = unlabeled_graph(SimpleNamespace(x=raw.x.cpu(), hyperedge_index=raw.edge_index,
                                            num_hyperedges=int(raw.num_hyperedges)), args.dname)
    graph.source_paths = {'raw': str(raw_path), 'processed': str(base / 'pyg_data' / 'hypergraph_dataset_updated')}
    print(f'[pretrain data] {args.dname}: raw={raw_path}; processed={graph.source_paths["processed"]}')
    return graph


def plot_pretrain(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epochs = [row['epoch'] for row in history]
    for axis, split in zip(axes, ('train', 'val')):
        for name in ('loss', 'membership_loss', 'feature_loss', 'relation_loss'):
            axis.plot(epochs, [row[f'{split}_{name}'] for row in history], label=name)
        axis.set(xlabel='Pretraining epoch', ylabel='SSL loss', title=split)
        axis.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_pretraining(data, args, directory=None):
    from lib_utils.hirbert_joint_training import run_joint_pretraining
    return run_joint_pretraining(data, args, directory, loader=load_unlabeled_data,
                                model_factory=HIRPretrainer, save=atomic_save, plot=plot_pretrain)


def main(argv=None):
    args = parse_args(argv)
    path = run_pretraining(None, args)
    print(f'[Pretrained encoder] {path}')


if __name__ == '__main__':
    main()
