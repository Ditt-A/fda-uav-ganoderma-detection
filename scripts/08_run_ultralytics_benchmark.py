from __future__ import annotations

"""DEPRECATED legacy Ultralytics-only runner.

The active, audited six-model B0/B1/B2 workflow is
``scripts/run_benchmark_auto.py``.  This file is intentionally blocked so it
cannot accidentally bypass the Method-B audits or trainer-augmentation guards.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    raise SystemExit(
        "DEPRECATED: use scripts/run_benchmark_auto.py; this legacy runner is "
        "not part of the active benchmark pipeline."
    )

    # Kept below only to preserve the historical implementation for review.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
    )

    parser.add_argument(
        "--train-config",
        default="configs/experiment.yaml",
    )

    parser.add_argument(
        "--project",
        default="runs/ultralytics",
    )

    args = parser.parse_args()

    cfg = yaml.safe_load(
        Path(args.config).read_text(
            encoding="utf-8"
        )
    )

    for scenario in cfg["scenarios"]:
        for model in cfg["models"]:
            run_name = (
                f'{scenario["name"]}_'
                f'{model["name"]}'
            )

            command = [
                sys.executable,
                "scripts/05_train_ultralytics.py",
                "--dataset",
                scenario["data"],
                "--model",
                model["weights"],
                "--name",
                run_name,
                "--config",
                args.train_config,
                "--project",
                args.project,
            ]

            print(
                "[RUN]",
                " ".join(command),
                flush=True,
            )

            subprocess.run(
                command,
                check=True,
            )


if __name__ == "__main__":
    main()
