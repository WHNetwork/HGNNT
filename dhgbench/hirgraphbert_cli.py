"""Shared arguments for the Structure-only pretraining and comparison runners."""

import argparse
from pathlib import Path

from lib_dataset import _single_datasets_
from lib_models.hirgraphbert import HIRGRAPHBERT_DEFAULTS
from parameter_parser import str2bool


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ['cora', 'citeseer', 'coauthor_cora', 'NTU2012', 'house-committees-100']


def build_parser():
    parser = argparse.ArgumentParser(description='HIR-GraphBERT: Structure-only joint SSL and node classification')
    parser.add_argument('--mode', choices=('all', 'pretrain', 'finetune'), default='all')
    parser.add_argument('--dname', choices=_single_datasets_, default='cora')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--data_root', type=Path, default=ROOT.parent / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'hirgraphbert_structure_only')
    parser.add_argument('--pretrain_datasets', nargs='+', choices=_single_datasets_, default=DEFAULT_DATASETS)
    parser.add_argument('--pretrained_checkpoint', type=Path)
    parser.add_argument('--variants', nargs='+', choices=('scratch', 'pretrained'), default=['scratch', 'pretrained'])
    parser.add_argument('--reference_results', nargs='*', type=Path, default=[ROOT / 'results'])
    parser.add_argument('--pretrain_seed', type=int, default=0)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--seed_offset', type=int, default=0)
    integer_defaults = {
        'pretrain_epochs': 200, 'pretrain_patience': 50, 'epochs': 500, 'patience': 100,
        'local_samples_per_graph': 128, 'validation_samples_per_graph': 64,
        'local_target_count': 16, 'local_relation_targets': 32,
        'local_min_edges': 64, 'local_max_edges': 128, 'local_hops': 3,
        'local_hir_targets': 4, 'hir_payload_candidates': 32,
        'micro_batch_size': 8, 'micro_batch_max_size': 64, 'micro_batch_max_tokens': 262144,
        'micro_batch_max_nodes': 32768, 'micro_batch_max_chunks': 1024,
        'bert_max_batch_tokens': 32768, 'bert_max_chunk_batch_size': 64,
        'micro_batch_tune_interval': 4, 'gradient_accumulation_steps': 4, 'oom_retries': 6,
        'relation_pairs_per_class': 128, 'relation_max_pairs': 50000, 'warmup_epochs': 10, 'display_step': 10,
    }
    for key, default in integer_defaults.items():
        parser.add_argument('--' + key, type=int, default=default)
    parser.add_argument('--local_max_nodes', type=int, default=2048,
                        help='Visible-node cap for optional neighborhood growth; supervised endpoints stay whole')
    floats = {'pretrain_lr': .0003, 'pretrain_wd': .01, 'lr': .0005, 'wd': .01,
              'membership_mask_ratio': .15, 'negative_ratio': 1., 'hir_mask_ratio': .15, 'ssl_val_ratio': .1,
              'lambda_m': 1., 'lambda_r': .5, 'lambda_h': 1., 'train_prop': .5, 'valid_prop': .25,
              'gpu_memory_budget_gb': 18., 'gpu_target_memory_gb': 15., 'grad_clip': 1.}
    for key, default in floats.items():
        parser.add_argument('--' + key, type=float, default=default)
    for key in ('auto_micro_batch', 'resume'):
        parser.add_argument('--' + key, type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--amp', choices=('auto', 'bf16', 'fp16', 'off'), default='bf16')
    parser.add_argument('--scheduler', choices=('cosine', 'linear', 'none'), default='cosine')
    parser.add_argument('--feature_noise', default='0.6', help='Dataset cache selector only; no features enter this model')
    for key, default in HIRGRAPHBERT_DEFAULTS.items():
        parser.add_argument('--hir_' + key, default=default,
                            type=str2bool if isinstance(default, bool) else type(default))
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output, args.data_root = args.output.resolve(), args.data_root.resolve()
    args.pretrain_datasets = list(dict.fromkeys(args.pretrain_datasets))
    args.variants = list(dict.fromkeys(args.variants))
    if min(args.pretrain_seed, args.seed_offset, args.oom_retries, args.warmup_epochs) < 0:
        parser.error('Seeds, retries and warmup must be nonnegative')
    positive = ('pretrain_epochs', 'pretrain_patience', 'epochs', 'patience', 'num_seeds', 'local_samples_per_graph',
                'validation_samples_per_graph', 'local_target_count', 'local_relation_targets', 'local_hir_targets',
                'hir_payload_candidates', 'local_min_edges', 'local_max_edges', 'local_max_nodes', 'micro_batch_size',
                'micro_batch_max_size', 'micro_batch_max_tokens', 'micro_batch_max_nodes', 'micro_batch_max_chunks',
                'bert_max_batch_tokens', 'bert_max_chunk_batch_size', 'micro_batch_tune_interval', 'gradient_accumulation_steps',
                'relation_pairs_per_class', 'relation_max_pairs', 'display_step', 'hir_hidden_size', 'hir_num_hidden_layers',
                'hir_num_attention_heads', 'hir_intermediate_size', 'hir_chunk_batch_size')
    if any(getattr(args, key) < 1 for key in positive):
        parser.error('Epochs and sample/model/batch counts must be positive')
    if (not 7 <= args.hir_max_seq_length <= 512 or args.local_max_edges < max(2, args.local_min_edges)
            or args.local_hops not in (1, 2, 3, 4)):
        parser.error('Invalid local or chunk limits')
    if args.hir_hidden_size < 2 or args.hir_hidden_size % args.hir_num_attention_heads or not 0 <= args.hir_dropout < 1:
        parser.error('Hidden size must be divisible by heads; dropout must be in [0,1)')
    if (min(args.pretrain_lr, args.lr, args.negative_ratio, args.lambda_m, args.lambda_r, args.lambda_h) <= 0
            or min(args.pretrain_wd, args.wd, args.grad_clip) < 0):
        parser.error('Invalid optimizer settings or structural loss weights')
    if not all(0 < value <= 1 for value in (args.membership_mask_ratio, args.hir_mask_ratio)) or not 0 < args.ssl_val_ratio < 1:
        parser.error('Invalid masking or validation fraction')
    if min(args.train_prop, args.valid_prop) <= 0 or args.train_prop + args.valid_prop >= 1:
        parser.error('Invalid supervised split proportions')
    if (args.micro_batch_size > args.micro_batch_max_size or args.bert_max_batch_tokens < args.hir_max_seq_length
            or args.bert_max_chunk_batch_size < args.hir_chunk_batch_size
            or not 0 < args.gpu_target_memory_gb < args.gpu_memory_budget_gb):
        parser.error('Invalid microbatch or GPU memory budgets')
    if args.mode == 'finetune' and 'pretrained' in args.variants and args.pretrained_checkpoint is None:
        parser.error('--mode=finetune with pretrained requires --pretrained_checkpoint')
    return args
