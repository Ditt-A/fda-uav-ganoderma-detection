#!/usr/bin/env python3
"""Create publication-ready figures from completed YOLO training logs.

The script reads only benchmark artifacts and writes figures under reports/figures.
It intentionally does not retrain or alter any experiment result.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "benchmark"
REPORTS = ROOT / "reports" / "benchmark"
FIGURES = ROOT / "reports" / "figures"

MODELS = {
    "yolov10s": "YOLOv10-S",
    "yolo11s": "YOLO11-S",
    "yolov12s": "YOLOv12-S",
    "yolov13s": "YOLOv13-S",
}
SCENARIOS = ("B0", "B1", "B2")
SCENARIO_LABELS = {
    "B0": "B0: originals",
    "B1": "B1: + geometric",
    "B2": "B2: + FDA",
}
PALETTE = {"B0": "#4C78A8", "B1": "#F58518", "B2": "#54A24B"}


def clean_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def column(frame: pd.DataFrame, name: str) -> str | None:
    exact = [item for item in frame.columns if item == name]
    if exact:
        return exact[0]
    normalized = name.replace(" ", "").lower()
    for item in frame.columns:
        if item.replace(" ", "").lower() == normalized:
            return item
    return None


def numeric(frame: pd.DataFrame, name: str) -> pd.Series | None:
    selected = column(frame, name)
    if selected is None:
        return None
    return pd.to_numeric(frame[selected], errors="coerce")


def save_figure(figure: plt.Figure, stem: str) -> list[Path]:
    created: list[Path] = []
    for extension in ("png", "pdf"):
        target = FIGURES / (stem + "." + extension)
        figure.savefig(target, dpi=320, bbox_inches="tight")
        created.append(target)
    plt.close(figure)
    return created


def training_curves(logs: dict[tuple[str, str], pd.DataFrame]) -> list[Path]:
    created: list[Path] = []
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=False, sharey=True)
    for axis, (model_key, model_name) in zip(axes.flat, MODELS.items()):
        for scenario in SCENARIOS:
            frame = logs.get((model_key, scenario))
            if frame is None:
                continue
            x = numeric(frame, "epoch")
            y = numeric(frame, "metrics/mAP50-95(B)")
            if x is None or y is None:
                continue
            axis.plot(x, y, label=SCENARIO_LABELS[scenario], color=PALETTE[scenario], linewidth=2.2)
        axis.set_title(model_name, fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation mAP@50-95")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=True)
    figure.suptitle("YOLO learning curves across augmentation scenarios", fontsize=14, fontweight="bold")
    figure.tight_layout()
    created.extend(save_figure(figure, "yolo_learning_curves_map50_95"))

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=False, sharey=True)
    for axis, (model_key, model_name) in zip(axes.flat, MODELS.items()):
        for scenario in SCENARIOS:
            frame = logs.get((model_key, scenario))
            if frame is None:
                continue
            x = numeric(frame, "epoch")
            precision = numeric(frame, "metrics/precision(B)")
            recall = numeric(frame, "metrics/recall(B)")
            if x is None:
                continue
            if precision is not None:
                axis.plot(x, precision, color=PALETTE[scenario], linewidth=2, linestyle="-", label=f"{scenario} precision")
            if recall is not None:
                axis.plot(x, recall, color=PALETTE[scenario], linewidth=1.7, linestyle="--", label=f"{scenario} recall")
        axis.set_title(model_name, fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation score")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2, frameon=True)
    figure.suptitle("YOLO precision and recall learning curves", fontsize=14, fontweight="bold")
    figure.tight_layout()
    created.extend(save_figure(figure, "yolo_learning_curves_precision_recall"))
    return created


def final_test_comparison() -> list[Path]:
    source = REPORTS / "final_metrics.csv"
    if not source.exists():
        return []
    frame = clean_frame(source)
    needed = {"model", "scenario", "evaluation_view", "mAP50_95"}
    if not needed.issubset(frame.columns):
        return []
    frame = frame[
        frame["model"].isin(MODELS)
        & (frame["evaluation_view"] == "primary_full")
        & frame["scenario"].isin(SCENARIOS)
    ].copy()
    if frame.empty:
        return []
    frame["model_label"] = frame["model"].map(MODELS)
    frame["scenario_label"] = frame["scenario"].map(SCENARIO_LABELS)
    order = [MODELS[item] for item in MODELS]
    figure, axis = plt.subplots(figsize=(12.5, 6.2))
    sns.barplot(
        data=frame,
        x="model_label",
        y="mAP50_95",
        hue="scenario_label",
        order=order,
        hue_order=[SCENARIO_LABELS[item] for item in SCENARIOS],
        palette=[PALETTE[item] for item in SCENARIOS],
        ax=axis,
    )
    axis.set_title("Held-out test performance: YOLO augmentation ablation", fontweight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("mAP@50-95 (primary held-out test)")
    axis.set_ylim(0, max(0.05, float(frame["mAP50_95"].max()) * 1.2))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Training scenario", frameon=True)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    figure.tight_layout()
    return save_figure(figure, "yolo_heldout_test_ablation")


def write_manifest(created: list[Path], logs: dict[tuple[str, str], pd.DataFrame]) -> Path:
    target = FIGURES / "yolo_log_figure_manifest.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact", "kind", "source"])
        writer.writeheader()
        for path in created:
            writer.writerow({"artifact": str(path.relative_to(ROOT)), "kind": "figure", "source": "YOLO results.csv / final_metrics.csv"})
        for (model, scenario), frame in sorted(logs.items()):
            writer.writerow({"artifact": "runs/benchmark/" + scenario + "_" + model + "/results.csv", "kind": "training_log", "source": f"{len(frame)} epoch records"})
    return target


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    logs: dict[tuple[str, str], pd.DataFrame] = {}
    for model in MODELS:
        for scenario in SCENARIOS:
            path = RUNS / (scenario + "_" + model) / "results.csv"
            if path.exists():
                logs[(model, scenario)] = clean_frame(path)
    if not logs:
        raise SystemExit("No YOLO results.csv artifacts found under runs/benchmark.")
    created = training_curves(logs)
    created.extend(final_test_comparison())
    manifest = write_manifest(created, logs)
    print(f"Created {len(created)} report figures in {FIGURES}")
    for path in created:
        print(path)
    print(manifest)


if __name__ == "__main__":
    main()
