import subprocess
import sys
from pathlib import Path


# ============================================================
# Experiment settings
# ============================================================

DATASETS = [
    "house-committees-100",
    "cora",
    "coauthor_cora",
    "coauthor_dblp",
    "NTU2012",
]

NUM_SEEDS = 10

MAX_EXACT_ORDER = 8
ORDER_HEADS = 4

DEVICE = "cuda:1"


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent
MAIN_PY = ROOT / "main.py"


def run_experiment(dataset, method, extra_args=None):
    """Run one DHG-Bench experiment."""

    if extra_args is None:
        extra_args = []

    command = [
        sys.executable,
        str(MAIN_PY),

        f"--dname={dataset}",
        "--task_type=node_cls",
        f"--method={method}",
        f"--num_seeds={NUM_SEEDS}",
        f"--device={DEVICE}",
    ]

    command.extend(extra_args)

    print("\n" + "=" * 80)
    print(f"Dataset : {dataset}")
    print(f"Method  : {method}")

    if method == "OrderSplitEDHNN":
        print("Fusion  : cross_attn")

    print("=" * 80)
    print(" ".join(command))
    print("=" * 80 + "\n")

    result = subprocess.run(
        command,
        cwd=ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Experiment failed: dataset={dataset}, method={method}"
        )


def main():

    print("\n")
    print("#" * 80)
    print("EDHNN vs OrderSplitEDHNN-CrossAttention")
    print(f"Datasets   : {len(DATASETS)}")
    print(f"Seeds      : {NUM_SEEDS}")
    print(f"Max order  : {MAX_EXACT_ORDER}")
    print(f"Heads      : {ORDER_HEADS}")
    print("#" * 80)

    for dataset in DATASETS:

        # ----------------------------------------------------
        # 1. Original EDHNN
        # ----------------------------------------------------

        run_experiment(
            dataset=dataset,
            method="EDHNN",
        )

        # ----------------------------------------------------
        # 2. Final OrderSplit + Cross-Attention
        # ----------------------------------------------------

        run_experiment(
            dataset=dataset,
            method="OrderSplitEDHNN",
            extra_args=[
                "--order_fusion=cross_attn",
                f"--max_exact_order={MAX_EXACT_ORDER}",
                f"--order_heads={ORDER_HEADS}",
            ],
        )

    print("\n")
    print("#" * 80)
    print("All experiments finished.")
    print("#" * 80)

    print("\nResults should be available in:")
    print(ROOT / "results" / "node_cls_runs.csv")
    print(ROOT / "results" / "node_cls_summary.csv")


if __name__ == "__main__":
    main()