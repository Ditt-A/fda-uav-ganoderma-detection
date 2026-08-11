from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SOURCE_FILES = (
    "scripts/run_benchmark_auto.py",
    "scripts/04_build_scenarios.py",
    "scripts/02b_groupaware_split.py",
    "scripts/03b_tile_small_objects.py",
    "scripts/05b_audit_method_b.py",
    "scripts/rebuild_method_b.py",
    "src/fda.py",
    "configs/method_b.yaml",
    "configs/datasets.yaml",
)


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def archive_if_exists(path: Path, archive_root: Path) -> bool:
    if not path.exists():
        return False
    destination = archive_root / path.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ARCHIVE] {path} -> {destination}")
    shutil.move(str(path), str(destination))
    return True


def snapshot_sources(archive_root: Path) -> None:
    for relative in ACTIVE_SOURCE_FILES:
        source = ROOT / relative
        if not source.exists():
            continue
        destination = archive_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Archive every active Method-B/benchmark artifact, then rebuild and "
            "fail-loudly audit B0/B1/B2. This command never starts model training."
        )
    )
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--method-config", default="configs/method_b.yaml")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = ROOT / "archives" / f"before_b0_b1_b2_{stamp}"
    paths_to_archive = (
        ROOT / "data/processed/harmonized",
        ROOT / "data/processed/tiled",
        ROOT / "data/processed/scenarios",
        ROOT / "data/exports/coco",
        ROOT / "runs/benchmark",
        ROOT / "runs/preflight",
        ROOT / "reports/benchmark",
        ROOT / "reports/environments",
        ROOT / "reports/method_b",
        ROOT / "reports/recovery",
        ROOT / "external/TinyFormer/configs/ganoderma",
    )
    archived = False
    for path in paths_to_archive:
        archived = archive_if_exists(path, archive_root) or archived
    if archived:
        snapshot_sources(archive_root)
        print(f"[ARCHIVE COMPLETE] {archive_root}")
    elif archive_root.exists():
        archive_root.rmdir()

    python = sys.executable
    run(
        [
            python,
            "scripts/00b_recover_roboflow_versions.py",
            "--config",
            args.datasets_config,
        ]
    )
    run(
        [
            python,
            "scripts/02b_groupaware_split.py",
            "--datasets-config",
            args.datasets_config,
            "--method-config",
            args.method_config,
            "--output",
            "data/processed/harmonized",
            "--report-dir",
            "reports/method_b",
        ]
    )
    run(
        [
            python,
            "scripts/03b_tile_small_objects.py",
            "--input",
            "data/processed/harmonized",
            "--output",
            "data/processed/tiled",
            "--config",
            args.method_config,
            "--report-dir",
            "reports/method_b",
        ]
    )
    run(
        [
            python,
            "scripts/04_build_scenarios.py",
            "--input",
            "data/processed/tiled",
            "--output-root",
            "data/processed/scenarios",
            "--config",
            args.method_config,
            "--report-dir",
            "reports/method_b",
        ]
    )
    run(
        [
            python,
            "scripts/05b_audit_method_b.py",
            "--report-dir",
            "reports/method_b",
            "--config",
            args.method_config,
            "--b0",
            "data/processed/scenarios/B0",
            "--b1",
            "data/processed/scenarios/B1",
            "--b2",
            "data/processed/scenarios/B2",
        ]
    )

    print("\n" + "=" * 88)
    print("METHOD B B0/B1/B2 REBUILD COMPLETE")
    print("=" * 88)
    for scenario in ("B0", "B1", "B2"):
        print(f"[DATA] data/processed/scenarios/{scenario}")
    print("[AUDIT] reports/method_b/method_b_audit.json")
    print("[TRAINING] NOT STARTED")
    if archived:
        print(f"[ARCHIVE OLD] {archive_root}")


if __name__ == "__main__":
    main()
