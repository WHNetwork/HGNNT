"""Entry point: five-graph Structure-only HIR-GraphBERT pretraining."""

from hirgraphbert_cli import parse_args
from lib_utils.hirgraphbert_training import run_pretraining


def main(argv=None):
    args = parse_args(argv)
    path = run_pretraining(args)
    print(f'[Structure-only shared encoder] {path}')
    return path


if __name__ == '__main__':
    main()
