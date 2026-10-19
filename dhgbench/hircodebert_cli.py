"""HIR-CodeBERT command-line configuration."""

from hirgraphbert_cli import build_parser as _build_parser
from hirgraphbert_cli import parse_args as _parse_args


def build_parser():
    return _build_parser('hircodebert')


def parse_args(argv=None):
    return _parse_args(argv, 'hircodebert')
