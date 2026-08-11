from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (
    iter_images,
    load_yaml,
    matching_label,
    read_yolo_labels,
    save_json,
)
from src.fda import fourier_amplitude_mix

SCENARIOS = ("B0", "B1", "B2")
SPLITS = ("train", "valid", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def source_prefix(path: Path) -> str:
    return path.name.split("__", 1)[0].upper() if "__" in path.name else "UNKNOWN"


def split_dirs(root: Path, split: str, view: str = "primary_full") -> tuple[Path, Path]:
    base = root if view == "primary_full" else root / "balanced_diagnostic"
    return base / split / "images", base / split / "labels"


def split_dataset_stats(root: Path, split: str, view: str = "primary_full") -> dict:
    images_dir, labels_dir = split_dirs(root, split, view)
    images = list(iter_images(images_dir))
    source_counts = Counter()
    boxes_by_source = Counter()
    negatives_by_source = Counter()
    invalid: list[str] = []
    for image_path in images:
        source = source_prefix(image_path)
        source_counts[source] += 1
        label_path = matching_label(image_path, images_dir, labels_dir)
        if not label_path.exists():
            invalid.append(f"missing label: {label_path}")
            continue
        rows = read_yolo_labels(label_path)
        if not rows:
            negatives_by_source[source] += 1
        for row in rows:
            if len(row) != 5 or int(row[0]) != 0:
                invalid.append(f"class/shape invalid: {label_path} {row}")
                continue
            x, y, width, height = map(float, row[1:5])
            if not (
                0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
                and x - width / 2 >= -1e-6
                and y - height / 2 >= -1e-6
                and x + width / 2 <= 1.0 + 1e-6
                and y + height / 2 <= 1.0 + 1e-6
            ):
                invalid.append(f"bbox invalid: {label_path} {row}")
                continue
            boxes_by_source[source] += 1
    return {
        "images": len(images),
        "source_images": dict(source_counts),
        "source_boxes": dict(boxes_by_source),
        "source_negative_images": dict(negatives_by_source),
        "invalid_labels": invalid[:100],
        "invalid_label_count": len(invalid),
    }


def snapshot(root: Path, split: str, view: str) -> dict[str, tuple[str, str]]:
    images_dir, labels_dir = split_dirs(root, split, view)
    result = {}
    for image_path in iter_images(images_dir):
        label_path = matching_label(image_path, images_dir, labels_dir)
        result[image_path.name] = (
            sha256_file(image_path),
            sha256_file(label_path) if label_path.exists() else "MISSING",
        )
    return result


def compare_all(roots: dict[str, Path], split: str, view: str) -> dict:
    snapshots = {name: snapshot(root, split, view) for name, root in roots.items()}
    reference = snapshots["B0"]
    pairs = {
        f"B0_{scenario}": {
            "same_names": set(reference) == set(snapshots[scenario]),
            "same_content": reference == snapshots[scenario],
            "count_b0": len(reference),
            f"count_{scenario.lower()}": len(snapshots[scenario]),
            "only_b0": sorted(set(reference) - set(snapshots[scenario]))[:20],
            f"only_{scenario.lower()}": sorted(set(snapshots[scenario]) - set(reference))[:20],
        }
        for scenario in ("B1", "B2")
    }
    return {
        "same_content": all(item["same_content"] for item in pairs.values()),
        "pairs": pairs,
    }


def cross_split_hash_leakage(root: Path) -> dict[str, list[str]]:
    locations: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        images_dir, _ = split_dirs(root, split)
        for image_path in iter_images(images_dir):
            locations[sha256_file(image_path)].add(split)
    return {
        digest: sorted(splits)
        for digest, splits in locations.items()
        if len(splits) > 1
    }


def median_or_none(values: list[float]):
    return statistics.median(values) if values else None


def mean_or_none(values: list[float]):
    return statistics.mean(values) if values else None


def build_source_annotation_audit(
    b0: Path, report_dir: Path, tile_manifest_path: Path
) -> list[dict]:
    tile_rows = []
    if tile_manifest_path.exists():
        with tile_manifest_path.open("r", newline="", encoding="utf-8") as handle:
            tile_rows = list(csv.DictReader(handle))
    tiling = defaultdict(lambda: {"clipped": 0, "removed": 0})
    for row in tile_rows:
        key = (row["source"], row["split"])
        tiling[key]["clipped"] += int(row.get("boxes_clipped") or 0)
        tiling[key]["removed"] += int(row.get("boxes_removed_min_visibility") or 0)

    output: list[dict] = []
    for source in ("A", "B"):
        for split in SPLITS:
            images_dir, labels_dir = split_dirs(b0, split)
            images = [p for p in iter_images(images_dir) if source_prefix(p) == source]
            widths: list[float] = []
            heights: list[float] = []
            area_ratios: list[float] = []
            touching = Counter()
            coco_sizes = Counter()
            positive_images = 0
            bbox_count = 0
            for image_path in images:
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise RuntimeError(f"Could not read image during audit: {image_path}")
                image_h, image_w = image.shape[:2]
                rows = read_yolo_labels(matching_label(image_path, images_dir, labels_dir))
                positive_images += bool(rows)
                for row in rows:
                    bbox_count += 1
                    cx, cy, width, height = map(float, row[1:5])
                    widths.append(width)
                    heights.append(height)
                    area_ratio = width * height
                    area_ratios.append(area_ratio)
                    sides = {
                        "left": cx - width / 2 <= 1e-6,
                        "right": cx + width / 2 >= 1.0 - 1e-6,
                        "top": cy - height / 2 <= 1e-6,
                        "bottom": cy + height / 2 >= 1.0 - 1e-6,
                    }
                    for side, value in sides.items():
                        touching[side] += bool(value)
                    touching["any"] += any(sides.values())
                    pixel_area = width * image_w * height * image_h
                    coco_sizes[
                        "small" if pixel_area < 32**2 else "medium" if pixel_area < 96**2 else "large"
                    ] += 1
            denom = max(1, bbox_count)
            output.append(
                {
                    "source": source,
                    "split": split,
                    "image_count": len(images),
                    "positive_image_count": positive_images,
                    "negative_image_count": len(images) - positive_images,
                    "bbox_count": bbox_count,
                    "mean_bbox_width": mean_or_none(widths),
                    "median_bbox_width": median_or_none(widths),
                    "mean_bbox_height": mean_or_none(heights),
                    "median_bbox_height": median_or_none(heights),
                    "mean_bbox_area_ratio": mean_or_none(area_ratios),
                    "median_bbox_area_ratio": median_or_none(area_ratios),
                    "fraction_touching_left": touching["left"] / denom,
                    "fraction_touching_right": touching["right"] / denom,
                    "fraction_touching_top": touching["top"] / denom,
                    "fraction_touching_bottom": touching["bottom"] / denom,
                    "fraction_touching_any_border": touching["any"] / denom,
                    "fraction_area_ratio_gt_0_30": sum(x > 0.30 for x in area_ratios) / denom,
                    "coco_small_count_after_export": coco_sizes["small"],
                    "coco_medium_count_after_export": coco_sizes["medium"],
                    "coco_large_count_after_export": coco_sizes["large"],
                    "boxes_clipped_by_tiling": tiling[(source, split)]["clipped"],
                    "boxes_removed_by_min_visibility": tiling[(source, split)]["removed"],
                }
            )
    json_output = {
        "scope": "B0 primary_full (B1/B2 validation and test are asserted identical)",
        "box_dimensions": "normalized YOLO width/height; COCO sizes use pixel area",
        "source_labels_modified": False,
        "rows": output,
    }
    save_json(json_output, report_dir / "source_annotation_audit.json")
    with (report_dir / "source_annotation_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)
    return output


def verify_fda_controls(config: dict, scenario_summary: dict) -> dict:
    fda_cfg = config.get("fda", {})
    bidirectional = bool(fda_cfg.get("bidirectional", True))
    donor_resize = bool(fda_cfg.get("donor_resize", True))
    policy = scenario_summary.get("fda_policy", {})
    failures = []
    if policy.get("bidirectional") is not bidirectional:
        failures.append("scenario builder did not apply fda.bidirectional")
    if policy.get("donor_resize") is not donor_resize:
        failures.append("scenario builder did not apply fda.donor_resize")
    records = [
        row for row in scenario_summary.get("augmentation_manifest", [])
        if row.get("kind") == "FDA"
    ]
    if any(row["content_source"] == row["donor_source"] for row in records):
        failures.append("FDA used a donor from the content source")
    if bidirectional:
        content_sources = {row["content_source"] for row in records}
        if not {"A", "B"}.issubset(content_sources):
            failures.append("bidirectional FDA did not create content copies for both A and B")
    else:
        content = str(fda_cfg.get("one_way_content_source", "")).upper()
        donor = str(fda_cfg.get("one_way_donor_source", "")).upper()
        if not records or any(
            row["content_source"] != content or row["donor_source"] != donor
            for row in records
        ):
            failures.append(f"one-way FDA policy {content}->{donor} was not enforced")

    content = np.zeros((8, 8, 3), dtype=np.uint8)
    donor = np.zeros((4, 4, 3), dtype=np.uint8)
    resize_true_works = False
    resize_false_rejects = False
    try:
        resize_true_works = fourier_amplitude_mix(
            content, donor, donor_resize=True
        ).shape == content.shape
    except Exception:
        pass
    try:
        fourier_amplitude_mix(content, donor, donor_resize=False)
    except ValueError:
        resize_false_rejects = True
    if not resize_true_works or not resize_false_rejects:
        failures.append("donor_resize true/false behavior verification failed")
    return {
        "bidirectional": bidirectional,
        "donor_resize": donor_resize,
        "records_checked": len(records),
        "resize_true_resizes": resize_true_works,
        "resize_false_rejects": resize_false_rejects,
        "active_and_verified": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="reports/method_b")
    parser.add_argument("--config", default="configs/method_b.yaml")
    parser.add_argument("--b0", default="data/processed/scenarios/B0")
    parser.add_argument("--b1", default="data/processed/scenarios/B1")
    parser.add_argument("--b2", default="data/processed/scenarios/B2")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    roots = {"B0": Path(args.b0), "B1": Path(args.b1), "B2": Path(args.b2)}
    config = load_yaml(args.config)
    with (report_dir / "split_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))

    group_splits: dict[str, set[str]] = defaultdict(set)
    original_hash_splits: dict[str, set[str]] = defaultdict(set)
    source_splits = defaultdict(lambda: defaultdict(int))
    for row in manifest:
        group_splits[row["group"]].add(row["assigned_split"])
        original_hash_splits[row["sha256"]].add(row["assigned_split"])
        source_splits[row["source"]][row["assigned_split"]] += 1
    group_leaks = {k: sorted(v) for k, v in group_splits.items() if len(v) > 1}
    original_hash_leaks = {
        k: sorted(v) for k, v in original_hash_splits.items() if len(v) > 1
    }

    scenario_summary = json.loads(
        (report_dir / "scenario_build_summary.json").read_text(encoding="utf-8")
    )
    result = {
        "seed": int(config.get("seed", 42)),
        "group_leakage_count": len(group_leaks),
        "group_leakage": group_leaks,
        "exact_original_image_leakage_count": len(original_hash_leaks),
        "exact_original_image_leakage": original_hash_leaks,
        "source_split_counts_before_tiling": {
            source: dict(counts) for source, counts in source_splits.items()
        },
        "scenarios": {},
        "identity": {},
    }
    for scenario, root in roots.items():
        result["scenarios"][scenario] = {
            split: split_dataset_stats(root, split) for split in SPLITS
        }
    for split in ("valid", "test"):
        result["identity"][f"primary_full_{split}"] = compare_all(
            roots, split, "primary_full"
        )
        result["identity"][f"balanced_diagnostic_{split}"] = compare_all(
            roots, split, "balanced_diagnostic"
        )

    actual_hash_leaks = cross_split_hash_leakage(roots["B0"])
    result["exact_scenario_image_leakage_count"] = len(actual_hash_leaks)
    result["exact_scenario_image_leakage"] = dict(list(actual_hash_leaks.items())[:100])
    result["fda_config_controls"] = verify_fda_controls(config, scenario_summary)
    annotation_rows = build_source_annotation_audit(
        roots["B0"], report_dir, report_dir / "tile_manifest.csv"
    )
    result["source_annotation_audit_rows"] = len(annotation_rows)

    failures: list[str] = []
    if group_leaks:
        failures.append("group leakage != 0")
    if original_hash_leaks or actual_hash_leaks:
        failures.append("exact-image leakage != 0")
    for source in ("A", "B"):
        for split in SPLITS:
            if source_splits[source][split] <= 0:
                failures.append(f"source {source} missing from {split}")
    for scenario in SCENARIOS:
        for split in SPLITS:
            count = result["scenarios"][scenario][split]["invalid_label_count"]
            if count:
                failures.append(f"{scenario}/{split} has {count} invalid labels")
    for key, identity in result["identity"].items():
        if not identity["same_content"]:
            failures.append(f"B0/B1/B2 {key} differs")

    counts = {
        scenario: result["scenarios"][scenario]["train"]["images"]
        for scenario in SCENARIOS
    }
    result["train_count_assertions"] = {
        "B1_equals_2x_B0": counts["B1"] == 2 * counts["B0"],
        "B2_equals_3x_B0": counts["B2"] == 3 * counts["B0"],
        "counts": counts,
    }
    if counts["B1"] != 2 * counts["B0"]:
        failures.append(f"B1 != 2*B0: {counts}")
    if counts["B2"] != 3 * counts["B0"]:
        failures.append(f"B2 != 3*B0: {counts}")
    failures.extend(result["fda_config_controls"]["failures"])
    result["invalid_label_count"] = sum(
        result["scenarios"][scenario][split]["invalid_label_count"]
        for scenario in SCENARIOS
        for split in SPLITS
    )
    result["failures"] = failures
    result["status"] = "PASS" if not failures else "FAIL"
    save_json(result, report_dir / "method_b_audit.json")

    full_valid = result["scenarios"]["B0"]["valid"]["images"]
    full_test = result["scenarios"]["B0"]["test"]["images"]
    balanced_valid = len(snapshot(roots["B0"], "valid", "balanced_diagnostic"))
    balanced_test = len(snapshot(roots["B0"], "test", "balanced_diagnostic"))
    print("\nMETHOD B B0/B1/B2 DATA READINESS")
    print("=" * 36)
    print(f"B0 train: {counts['B0']}")
    print(f"B1 train: {counts['B1']}")
    print(f"B2 train: {counts['B2']}")
    print(f"Full valid: {full_valid}")
    print(f"Full test : {full_test}")
    print(f"Balanced diagnostic valid: {balanced_valid}")
    print(f"Balanced diagnostic test : {balanced_test}")
    for source in ("A", "B"):
        print(
            f"Source {source}: "
            + " / ".join(
                str(result["scenarios"]["B0"][split]["source_images"].get(source, 0))
                for split in SPLITS
            )
            + " (train / valid / test)"
        )
    print(f"Group leakage: {len(group_leaks)}")
    print(f"Exact-image leakage: {len(original_hash_leaks) + len(actual_hash_leaks)}")
    print(f"Invalid labels: {result['invalid_label_count']}")
    for view in ("primary_full", "balanced_diagnostic"):
        label = "full" if view == "primary_full" else "balanced"
        for split in ("valid", "test"):
            passed = result["identity"][f"{view}_{split}"]["same_content"]
            print(f"B0/B1/B2 {label} {split} identical: {'PASS' if passed else 'FAIL'}")
    fda = result["fda_config_controls"]
    print("FDA config controls:")
    print(f"bidirectional: {'active and verified' if fda['active_and_verified'] else 'FAIL'}")
    print(f"donor_resize: {'active and verified' if fda['active_and_verified'] else 'FAIL'}")
    print(f"DATA_AUDIT = {result['status']}")
    print(f"[REPORT] {report_dir / 'method_b_audit.json'}")
    print(f"[REPORT] {report_dir / 'source_annotation_audit.json'}")
    print(f"[REPORT] {report_dir / 'source_annotation_audit.csv'}")
    if failures:
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
