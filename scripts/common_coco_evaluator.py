from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

SOURCES = ("COMBINED", "A", "B")
FINAL_COLUMNS = (
    "model",
    "scenario",
    "evaluation_view",
    "source",
    "seed",
    "train_images",
    "best_epoch",
    "epochs_completed",
    "precision",
    "recall",
    "f1",
    "map50",
    "map75",
    "map50_95",
    "ar",
    "batch",
    "workers",
    "input_size",
    "parameter_count",
    "flops",
    "checkpoint",
    "runtime",
    "batches_per_epoch",
    "optimizer_steps",
    "effective_samples_seen",
    "budget_mode",
    "score_threshold_for_f1",
    "ground_truth",
    "predictions",
)


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def best_f1_at_iou50(
    ground_truth: dict[str, Any], predictions: list[dict], image_ids: set[int]
) -> dict[str, float]:
    gt_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for ann in ground_truth.get("annotations", []):
        if ann["image_id"] in image_ids and int(ann["category_id"]) == 0:
            gt_by_image[int(ann["image_id"])].append(list(map(float, ann["bbox"])))
    total_gt = sum(len(items) for items in gt_by_image.values())
    matched: dict[int, set[int]] = defaultdict(set)
    true_positives = 0
    false_positives = 0
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 1.0}
    ordered = sorted(
        (
            item for item in predictions
            if int(item["image_id"]) in image_ids and int(item["category_id"]) == 0
        ),
        key=lambda item: (-float(item.get("score", 0.0)), int(item["image_id"])),
    )
    for prediction in ordered:
        image_id = int(prediction["image_id"])
        candidates = gt_by_image.get(image_id, [])
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(candidates):
            if index in matched[image_id]:
                continue
            value = _iou(list(map(float, prediction["bbox"])), target)
            if value > best_iou:
                best_iou, best_index = value, index
        if best_iou >= 0.5:
            true_positives += 1
            matched[image_id].add(best_index)
        else:
            false_positives += 1
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, total_gt)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        if f1 > best["f1"]:
            best = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "threshold": float(prediction.get("score", 0.0)),
            }
    return best


def _coco_from_predictions(coco_gt, predictions: list[dict]):
    from pycocotools.coco import COCO

    coco_dt = COCO()
    coco_dt.dataset = {
        "images": deepcopy(coco_gt.dataset.get("images", [])),
        "categories": deepcopy(coco_gt.dataset.get("categories", [])),
        "annotations": [
            {
                **deepcopy(item),
                "id": index + 1,
                "iscrowd": 0,
                "area": float(item["bbox"][2]) * float(item["bbox"][3]),
            }
            for index, item in enumerate(predictions)
        ],
    }
    coco_dt.createIndex()
    return coco_dt


def evaluate_common_coco(
    ground_truth_path: Path,
    predictions_path: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise TypeError("COCO predictions must be a JSON list")
    valid_ids = {int(item["id"]) for item in ground_truth.get("images", [])}
    for item in predictions:
        if int(item.get("image_id", -1)) not in valid_ids:
            raise RuntimeError(f"Prediction references unknown image_id: {item}")
        if int(item.get("category_id", -1)) != 0:
            raise RuntimeError(f"Prediction category_id must be 0: {item}")
        if len(item.get("bbox", [])) != 4 or not math.isfinite(float(item.get("score", math.nan))):
            raise RuntimeError(f"Invalid prediction: {item}")

    coco_gt = COCO(str(ground_truth_path))
    coco_dt = _coco_from_predictions(coco_gt, predictions)
    images_by_source = defaultdict(set)
    for image in ground_truth.get("images", []):
        source = str(image.get("source") or image["file_name"].split("__", 1)[0]).upper()
        images_by_source[source].add(int(image["id"]))
    images_by_source["COMBINED"] = valid_ids

    rows = []
    for source in SOURCES:
        image_ids = images_by_source[source]
        if not image_ids:
            raise RuntimeError(f"No evaluation images for source={source}")
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = [0]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats
        prf = best_f1_at_iou50(ground_truth, predictions, image_ids)
        row = {column: metadata.get(column) for column in FINAL_COLUMNS}
        row.update(
            {
                "source": source,
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "map50_95": float(stats[0]),
                "map50": float(stats[1]),
                "map75": float(stats[2]),
                "ar": float(stats[8]),
                "score_threshold_for_f1": prf["threshold"],
                "ground_truth": str(ground_truth_path.resolve()),
                "predictions": str(predictions_path.resolve()),
            }
        )
        rows.append(row)
    return rows


def write_metrics(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def collect_final_reports(search_root: Path, report_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(search_root.rglob("common_metrics_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Expected metric row list: {path}")
        rows.extend(payload)
    rows.sort(
        key=lambda row: (
            str(row.get("model")),
            str(row.get("scenario")),
            str(row.get("evaluation_view")),
            str(row.get("source")),
        )
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "final_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with (report_dir / "final_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FINAL_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collect-root", default="runs/benchmark")
    parser.add_argument("--report-dir", default="reports/benchmark")
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    rows = evaluate_common_coco(
        Path(args.ground_truth), Path(args.predictions), metadata
    )
    write_metrics(rows, Path(args.output))
    collect_final_reports(Path(args.collect_root), Path(args.report_dir))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
