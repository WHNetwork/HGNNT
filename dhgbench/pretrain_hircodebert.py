"""Entry point for five-graph HIR-CodeBERT pretraining."""

from hircodebert_cli import parse_args
from lib_utils.hirgraphbert_training import run_pretraining


if __name__ == '__main__':
    run_pretraining(parse_args())
