#!/usr/bin/env python3
"""Generate report figures for RF-DETR, TinyFormer, DEIMv2, and all models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/benchmark"
REPORTS = ROOT / "reports/benchmark"
FIGURES = ROOT / "reports/figures"
SCENARIOS = ("B0", "B1", "B2")
SCENARIO_LABELS = {
    "B0": "B0: originals",
    "B1": "B1: + geometric",
    "B2": "B2: + FDA",
}
PALETTE = {"B0": "#4C78A8", "B1": "#F58518", "B2": "#54A24B"}
TRANSFORMERS = {
    "rfdetrs": "RF-DETR-S",
    "tinyformers": "TinyFormer-S-PBM",
    "deimv2s": "DEIMv2-S",
}
ALL_MODELS = {
    "yolov10s": "YOLOv10-S",
    "yolo11s": "YOLO11-S",
    "yolov12s": "YOLOv12-S",
    "yolov13s": "YOLOv13-S",
    **TRANSFORMERS,
}


def save(figure: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for extension in ("png", "pdf"):
        target = FIGURES / (stem + "." + extension)
        figure.savefig(target, dpi=320, bbox_inches="tight")
        paths.append(target)
    plt.close(figure)
    return paths


def read_rfdetr(scenario: str) -> pd.DataFrame | None:
    path = RUNS / f"{scenario}_rfdetrs" / "metrics.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    metric = pd.to_numeric(frame.get("val/mAP_50_95"), errors="coerce")
    epoch = pd.to_numeric(frame.get("epoch"), errors="coerce")
    output = pd.DataFrame({"epoch": epoch + 1, "map": metric}).dropna()
    if output.empty:
        return None
    return output.groupby("epoch", as_index=False)["map"].max()


def read_json_log(model: str, scenario: str) -> pd.DataFrame | None:
    path = RUNS / f"{scenario}_{model}" / "log.txt"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        metric = payload.get("test_coco_eval_bbox")
        if not isinstance(metric, list) or not metric:
            continue
        rows.append({"epoch": int(payload.get("epoch", len(rows))) + 1, "map": float(metric[0])})
    return pd.DataFrame(rows) if rows else None


def transformer_learning_curves() -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), sharey=True)
    found = False
    for axis, (model, label) in zip(axes, TRANSFORMERS.items()):
        for scenario in SCENARIOS:
            frame = read_rfdetr(scenario) if model == "rfdetrs" else read_json_log(model, scenario)
            if frame is None:
                continue
            found = True
            axis.plot(
                frame["epoch"],
                frame["map"],
                color=PALETTE[scenario],
                linewidth=2.2,
                label=SCENARIO_LABELS[scenario],
            )
        axis.set_title(label, fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation mAP@50-95")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    if not found:
        plt.close(figure)
        return []
    figure.suptitle("Transformer detector learning curves", fontsize=14, fontweight="bold")
    figure.tight_layout()
    return save(figure, "transformer_learning_curves_map50_95")


def consolidated_test_chart() -> list[Path]:
    source = REPORTS / "final_metrics.csv"
    if not source.exists():
        return []
    frame = pd.read_csv(source)
    required = {"model", "scenario", "evaluation_view", "source", "map50_95"}
    if not required.issubset(frame.columns):
        return []
    frame = frame[
        frame["model"].isin(ALL_MODELS)
        & frame["scenario"].isin(SCENARIOS)
        & (frame["evaluation_view"] == "primary_full")
        & (frame["source"] == "COMBINED")
    ].copy()
    if frame.empty:
        return []
    frame["model_label"] = frame["model"].map(ALL_MODELS)
    frame["scenario_label"] = frame["scenario"].map(SCENARIO_LABELS)
    order = [label for key, label in ALL_MODELS.items() if key in set(frame["model"])]
    figure, axis = plt.subplots(figsize=(15.5, 6.5))
    sns.barplot(
        data=frame,
        x="model_label",
        y="map50_95",
        hue="scenario_label",
        order=order,
        hue_order=[SCENARIO_LABELS[item] for item in SCENARIOS],
        palette=[PALETTE[item] for item in SCENARIOS],
        ax=axis,
    )
    axis.set_title("Held-out test performance across model families", fontweight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("Combined mAP@50-95 (primary test)")
    axis.set_ylim(0, max(0.05, float(frame["map50_95"].max()) * 1.18))
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Training scenario", frameon=True)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    figure.tight_layout()
    return save(figure, "all_models_heldout_test_ablation")


def runtime_chart() -> list[Path]:
    source = REPORTS / "final_metrics.csv"
    if not source.exists():
        return []
    frame = pd.read_csv(source)
    required = {"model", "scenario", "evaluation_view", "source", "runtime"}
    if not required.issubset(frame.columns):
        return []
    frame = frame[
        frame["model"].isin(ALL_MODELS)
        & frame["scenario"].isin(SCENARIOS)
        & (frame["evaluation_view"] == "primary_full")
        & (frame["source"] == "COMBINED")
    ].copy()
    frame["runtime_hours"] = pd.to_numeric(frame["runtime"], errors="coerce") / 3600
    frame = frame.dropna(subset=["runtime_hours"])
    if frame.empty:
        return []
    frame["model_label"] = frame["model"].map(ALL_MODELS)
    frame["scenario_label"] = frame["scenario"].map(SCENARIO_LABELS)
    order = [label for key, label in ALL_MODELS.items() if key in set(frame["model"])]
    figure, axis = plt.subplots(figsize=(15.5, 6.2))
    sns.barplot(
        data=frame,
        x="model_label",
        y="runtime_hours",
        hue="scenario_label",
        order=order,
        hue_order=[SCENARIO_LABELS[item] for item in SCENARIOS],
        palette=[PALETTE[item] for item in SCENARIOS],
        ax=axis,
    )
    axis.set_title("Recorded training runtime by model and scenario", fontweight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("Training runtime (hours)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Training scenario", frameon=True)
    figure.tight_layout()
    return save(figure, "all_models_training_runtime")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.12)
    created = []
    created.extend(transformer_learning_curves())
    created.extend(consolidated_test_chart())
    created.extend(runtime_chart())
    print(f"Created {len(created)} detector report figures.")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
