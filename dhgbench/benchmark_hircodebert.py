"""Five-graph pretraining and paired Cora evaluation for HIR-CodeBERT."""

from benchmark_hirgraphbert import run_benchmark
from hircodebert_cli import parse_args


def main(argv=None):
    return run_benchmark(parse_args(argv))


if __name__ == '__main__':
    main()
