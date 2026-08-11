#!/usr/bin/env python3
"""Train and evaluate DEIMv2-S on the audited B0/B1/B2 benchmark."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_benchmark_auto as benchmark
from common_coco_evaluator import collect_final_reports, evaluate_common_coco, write_metrics

REPOSITORY = ROOT / "external" / "DEIMv2"
RUNS = ROOT / "runs" / "benchmark"
REPORTS = ROOT / "reports" / "benchmark"
PRETRAINED = REPOSITORY / "ckpts" / "deimv2_dinov3_s_coco.pth"
SCENARIOS = ("B0", "B1", "B2")


def deterministic_transforms() -> dict[str, Any]:
    return {
        "type": "Compose",
        "ops": [
            {"type": "Resize", "size": [640, 640]},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {
                "type": "Normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ],
    }


def prepare_config(
    scenario: str,
    coco_root: Path,
    output_directory: Path,
    epochs: int,
    batch: int,
    workers: int,
) -> Path:
    original_model = REPOSITORY / "configs/deimv2/deimv2_dinov3_s_coco.yml"
    original_base = REPOSITORY / "configs/base/deimv2.yml"
    clean_base = REPOSITORY / "configs/base/deimv2_benchmark_clean.yml"
    model = yaml.safe_load(original_model.read_text(encoding="utf-8"))
    base = yaml.safe_load(original_base.read_text(encoding="utf-8"))
    removed_loader = base.pop("train_dataloader", None)
    if removed_loader is None:
        raise RuntimeError("DEIMv2 base config has no train_dataloader to sanitize.")
    clean_base.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    dataset_name = f"ganoderma_{scenario}.yml"
    dataset_path = REPOSITORY / "configs/dataset" / dataset_name
    dataset = {
        "task": "detection",
        "evaluator": {"type": "CocoEvaluator", "iou_types": ["bbox"]},
        "num_classes": 1,
        "remap_mscoco_category": False,
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str((coco_root / "images/train").resolve()),
                "ann_file": str((coco_root / "annotations/instances_train.json").resolve()),
                "return_masks": False,
                "transforms": deterministic_transforms(),
            },
            "shuffle": True,
            "num_workers": workers,
            "drop_last": True,
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": 640,
                "base_size_repeat": None,
                "mixup_prob": 0.0,
                "copyblend_prob": 0.0,
                "stop_epoch": 999999,
            },
            "total_batch_size": batch,
        },
        "val_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str((coco_root / "images/val").resolve()),
                "ann_file": str((coco_root / "annotations/instances_val.json").resolve()),
                "return_masks": False,
                "transforms": deterministic_transforms(),
            },
            "shuffle": False,
            "num_workers": workers,
            "drop_last": False,
            "collate_fn": {"type": "BatchImageCollateFunction"},
            "total_batch_size": batch,
        },
    }
    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")

    clean_includes: list[str] = []
    dataset_inserted = False
    for item in model.get("__include__", []) or []:
        value = str(item)
        if "dataset/" in value:
            if not dataset_inserted:
                clean_includes.append("../dataset/" + dataset_name)
                dataset_inserted = True
            continue
        if "base/dataloader.yml" in value:
            continue
        if "base/deimv2.yml" in value:
            clean_includes.append("../base/deimv2_benchmark_clean.yml")
            continue
        clean_includes.append(value)
    if not dataset_inserted:
        clean_includes.insert(0, "../dataset/" + dataset_name)

    model["__include__"] = clean_includes
    model.pop("train_dataloader", None)
    model.pop("val_dataloader", None)
    model["output_dir"] = str(output_directory.resolve())
    model["epoches"] = epochs
    model["flat_epoch"] = max(4, epochs // 2)
    model["no_aug_epoch"] = 0
    model["eval_spatial_size"] = [640, 640]
    model["print_freq"] = 10
    model["checkpoint_freq"] = 1
    model.setdefault("DEIMCriterion", {}).setdefault("matcher", {})[
        "matcher_change_epoch"
    ] = epochs + 1

    config_directory = REPOSITORY / "configs/ganoderma"
    config_directory.mkdir(parents=True, exist_ok=True)
    config_path = config_directory / f"deimv2_s_{scenario}.yml"
    config_path.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    audit = {
        "scenario": scenario,
        "num_classes": dataset["num_classes"],
        "remap_mscoco_category": dataset["remap_mscoco_category"],
        "trainer_side_augmentation": "disabled",
        "transforms": [item["type"] for item in deterministic_transforms()["ops"]],
        "mixup_prob": 0.0,
        "copyblend_prob": 0.0,
        "multi_scale": False,
        "fixed_input_size": 640,
        "clean_base": str(clean_base),
        "verified": (
            "__include__" not in dataset
            and "train_dataloader" not in base
            and "../base/dataloader.yml" not in clean_includes
        ),
    }
    if not audit["verified"]:
        raise RuntimeError("DEIMv2 clean benchmark config audit failed.")
    (output_directory / "augmentation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return config_path


def completed_epochs(output_directory: Path, fallback: int) -> int:
    log_path = output_directory / "log.txt"
    last = -1
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                last = max(last, int(json.loads(line).get("epoch", -1)))
            except Exception:
                pass
    return last + 1 if last >= 0 else fallback


def inspect_checkpoint(checkpoint: Path) -> tuple[int | None, int | None]:
    try:
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = payload.get("last_epoch")
        state = payload.get("model")
        count = None
        if isinstance(state, dict):
            count = sum(value.numel() for value in state.values() if hasattr(value, "numel"))
        return (int(epoch) + 1 if epoch is not None else None, count)
    except Exception as error:
        print(f"[DEIMV2 PROVENANCE] checkpoint inspection skipped: {error}")
        return None, None


def run_scenario(
    scenario: str,
    environment: str,
    epochs: int,
    batch: int,
    workers: int,
    patience: int,
    seed: int,
    device: str,
) -> None:
    coco_primary = benchmark.export_scenario_to_coco(scenario, "primary_full")
    coco_balanced = benchmark.export_scenario_to_coco(scenario, "balanced_diagnostic")
    run_name = f"{scenario}_deimv2s"
    output_directory = RUNS / run_name
    marker = output_directory / "training_complete.json"
    existing_checkpoints = list(output_directory.glob("*.pth")) if output_directory.exists() else []
    marker_payload = {}
    if marker.exists():
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            marker_payload = {}
    reusable = (
        bool(existing_checkpoints)
        and marker_payload.get("training_completed") is True
        and int(marker_payload.get("epochs_requested", -1)) == epochs
        and int(marker_payload.get("batch", -1)) == batch
    )

    if output_directory.exists() and not reusable:
        benchmark.archive_artifact(output_directory, "PREVIOUS_DEIMV2_RUN")
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = benchmark.write_training_budget(
        run_directory=output_directory,
        model="deimv2s",
        scenario=scenario,
        epochs=epochs,
        batch=batch,
        workers=workers,
        input_size=640,
        grad_accum=1,
        seed=seed,
        patience=patience,
    )
    config_path = prepare_config(
        scenario, coco_primary, output_directory, epochs, batch, workers
    )

    training_log = REPORTS / "logs" / f"{run_name}.log"
    environment_values = os.environ.copy()
    environment_values["CUDA_VISIBLE_DEVICES"] = device
    environment_values["DEIMV2_PATIENCE"] = str(patience)
    environment_values["DEIMV2_MIN_DELTA"] = "0.0"

    if reusable:
        print(f"[DEIMV2 RECOVERY] reusing completed training: {output_directory}")
        training_runtime = float(
            json.loads(marker.read_text(encoding="utf-8")).get("runtime_seconds", 0.0)
        )
    else:
        benchmark.archive_artifact(training_log, "PREVIOUS_DEIMV2_LOG")
        started = time.time()
        benchmark.run_command(
            benchmark.conda_command(
                environment,
                [
                    "python",
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node=1",
                    "--master_port=7785",
                    "train.py",
                    "-c",
                    str(config_path),
                    "--use-amp",
                    "--seed",
                    str(seed),
                    "-t",
                    str(PRETRAINED),
                ],
            ),
            cwd=REPOSITORY,
            log_path=training_log,
            env=environment_values,
        )
        training_runtime = time.time() - started
        marker.write_text(
            json.dumps(
                {
                    "training_completed": True,
                    "epochs_requested": epochs,
                    "batch": batch,
                    "workers": workers,
                    "runtime_seconds": training_runtime,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    checkpoint = benchmark.select_tinyformer_evaluation_checkpoint(output_directory)
    checkpoint_epoch, parameter_count = inspect_checkpoint(checkpoint)
    actual_epochs = completed_epochs(output_directory, checkpoint_epoch or epochs)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "best_epoch": checkpoint_epoch,
            "epochs_completed": actual_epochs,
            "parameter_count": parameter_count,
            "checkpoint": str(checkpoint.resolve()),
            "runtime": training_runtime,
            "optimizer_steps": metadata["batches_per_epoch"] * actual_epochs,
            "effective_samples_seen": metadata["train_images"] * actual_epochs,
            "optimizer_steps_are_planned": False,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for view_index, (view, coco_root) in enumerate(
        (("primary_full", coco_primary), ("balanced_diagnostic", coco_balanced))
    ):
        predictions = output_directory / f"predictions_{view}.json"
        metrics = output_directory / f"common_metrics_{view}.json"
        if predictions.exists() and metrics.exists():
            print(f"[DEIMV2 EVALUATION RECOVERY] {scenario} {view}")
            continue
        test_config = benchmark.prepare_tinyformer_test_config(
            repository_path=REPOSITORY,
            scenario=scenario,
            coco_root=coco_root,
            training_config_path=config_path,
            evaluation_view=view,
        )
        test_values = environment_values.copy()
        test_values["COMMON_COCO_PREDICTIONS"] = str(predictions.resolve())
        test_log = REPORTS / "logs" / f"{run_name}_{view}_test.log"
        benchmark.archive_artifact(test_log, "PREVIOUS_DEIMV2_TEST_LOG")
        benchmark.run_command(
            benchmark.conda_command(
                environment,
                [
                    "python",
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node=1",
                    f"--master_port={7786 + view_index}",
                    "train.py",
                    "-c",
                    str(test_config),
                    "--test-only",
                    "-r",
                    str(checkpoint),
                ],
            ),
            cwd=REPOSITORY,
            log_path=test_log,
            env=test_values,
        )
        if not predictions.exists():
            raise RuntimeError(f"DEIMv2 did not produce predictions: {predictions}")
        rows = evaluate_common_coco(
            coco_root / "annotations/instances_test.json",
            predictions,
            {**metadata, "evaluation_view": view},
        )
        write_metrics(rows, metrics)
    collect_final_reports(RUNS, REPORTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="B0,B1,B2")
    parser.add_argument("--environment", default="ganoderma_bench_tmp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not PRETRAINED.exists():
        raise FileNotFoundError(f"Missing official DEIMv2-S checkpoint: {PRETRAINED}")
    requested = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    invalid = sorted(set(requested) - set(SCENARIOS))
    if invalid:
        raise ValueError(f"Unsupported scenarios: {invalid}")
    benchmark.run_command(
        benchmark.conda_command(
            args.environment,
            ["python", "-m", "pip", "install", "-r", str(REPOSITORY / "requirements.txt")],
        )
    )
    for scenario in requested:
        run_scenario(
            scenario=scenario,
            environment=args.environment,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
