from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (
    find_data_yaml,
    find_split_dir,
    iter_images,
    load_yaml,
    matching_label,
    normalize_class_name,
    parse_names,
    read_yolo_labels,
    save_json,
    save_yaml,
    write_yolo_labels,
)

SPLITS = ("train", "valid", "test")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def label_signature(rows: list[list[float]]) -> tuple:
    return tuple(
        tuple(round(float(x), 8) for x in row)
        for row in sorted(rows)
    )


def strip_roboflow_hash(stem: str) -> str:
    return stem.split(".rf.", 1)[0]


def clean_source_a_name(stem: str) -> str:
    s = strip_roboflow_hash(stem)
    s = re.sub(r"__SRC\d+$", "", s, flags=re.I)
    s = re.sub(r"_(?:PNG|JPG|JPEG)$", "", s, flags=re.I)
    return s


def group_key_a(path: Path) -> str:
    s = clean_source_a_name(path.stem)
    s = re.sub(r"_Split\d+$", "", s, flags=re.I)
    s = re.sub(r"_\d{4,}$", "", s)
    s = re.sub(r"([A-Za-z])\d+$", r"\1", s)
    return f"A::{s}"


def group_key_b(path: Path) -> str:
    s = strip_roboflow_hash(path.stem)
    s = re.sub(r"_(?:jpg|jpeg|png)$", "", s, flags=re.I)
    match = re.match(r"^(.*?)-\d+$", s)
    if match:
        s = match.group(1)
    return f"B::{s}"


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "image"


def map_rows(
    rows: list[list[float]],
    names: dict[int, str],
    positives: set[str],
) -> list[list[float]]:
    mapped = []
    for row in rows:
        old_class = int(row[0])
        old_name = normalize_class_name(
            names.get(old_class, f"class_{old_class}")
        )
        if old_name in positives:
            mapped.append([0, *[float(x) for x in row[1:5]]])
    return mapped


def collect_source(
    source_name: str,
    source_cfg: dict,
) -> tuple[list[dict], list[dict]]:
    root = Path(source_cfg["output"])
    data_yaml_path = find_data_yaml(root)
    dataset_root = data_yaml_path.parent
    names = parse_names(load_yaml(data_yaml_path))
    positives = {
        normalize_class_name(x)
        for x in source_cfg.get("positive_names", [])
    }
    prefix = str(source_cfg["prefix"]).strip().upper()

    if prefix not in {"A", "B"}:
        raise ValueError(
            f"Method B mengharapkan prefix A/B, mendapat {prefix!r}"
        )

    items = []
    duplicate_rows = []
    seen_hashes: dict[str, dict] = {}

    for original_split in SPLITS:
        found = find_split_dir(dataset_root, original_split)
        if not found:
            continue
        images_dir, labels_dir = found

        for image_path in iter_images(images_dir):
            label_path = matching_label(
                image_path,
                images_dir,
                labels_dir,
            )
            rows = read_yolo_labels(label_path)
            mapped = map_rows(rows, names, positives)

            image_hash = sha256_file(image_path)
            signature = label_signature(mapped)

            if image_hash in seen_hashes:
                previous = seen_hashes[image_hash]
                if previous["label_signature"] != signature:
                    raise RuntimeError(
                        "Exact duplicate image memiliki mapped label berbeda:\n"
                        f"  {previous['image_path']}\n"
                        f"  {image_path}"
                    )
                duplicate_rows.append(
                    {
                        "source": prefix,
                        "kept": str(previous["image_path"]),
                        "dropped": str(image_path),
                        "kept_original_split": previous["original_split"],
                        "dropped_original_split": original_split,
                        "sha256": image_hash,
                    }
                )
                continue

            group = (
                group_key_a(image_path)
                if prefix == "A"
                else group_key_b(image_path)
            )

            item = {
                "source": prefix,
                "source_name": source_name,
                "original_split": original_split,
                "image_path": image_path,
                "mapped_rows": mapped,
                "positive_boxes": len(mapped),
                "is_negative": int(len(mapped) == 0),
                "group": group,
                "sha256": image_hash,
                "label_signature": signature,
            }
            seen_hashes[image_hash] = item
            items.append(item)

    if not items:
        raise RuntimeError(f"Tidak ada image untuk {source_name}")

    return items, duplicate_rows


def summarize_groups(items: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[item["group"]].append(item)

    rows = []
    for group, members in sorted(grouped.items()):
        rows.append(
            {
                "group": group,
                "images": len(members),
                "positive_boxes": sum(x["positive_boxes"] for x in members),
                "negative_images": sum(x["is_negative"] for x in members),
            }
        )
    return rows


def assignment_score(
    group_rows: list[dict],
    assignment: tuple[int, ...],
    ratios: tuple[float, float, float],
) -> float:
    totals = {
        "images": sum(x["images"] for x in group_rows),
        "positive_boxes": sum(x["positive_boxes"] for x in group_rows),
        "negative_images": sum(x["negative_images"] for x in group_rows),
    }

    counts = [
        {"images": 0, "positive_boxes": 0, "negative_images": 0, "groups": 0}
        for _ in range(3)
    ]

    for row, split_idx in zip(group_rows, assignment):
        counts[split_idx]["groups"] += 1
        for metric in ("images", "positive_boxes", "negative_images"):
            counts[split_idx][metric] += row[metric]

    if any(x["groups"] == 0 for x in counts):
        return float("inf")

    score = 0.0
    weights = {
        "images": 1.0,
        "positive_boxes": 0.75,
        "negative_images": 0.25,
    }

    for split_idx, ratio in enumerate(ratios):
        for metric, weight in weights.items():
            total = totals[metric]
            if total <= 0:
                continue
            target = max(1.0, total * ratio)
            error = (counts[split_idx][metric] - target) / target
            score += weight * error * error

    positive_group_count = sum(
        1 for row in group_rows if row["positive_boxes"] > 0
    )
    if positive_group_count >= 3:
        for split_idx in (1, 2):
            if counts[split_idx]["positive_boxes"] == 0:
                score += 1000.0

    return score


def choose_group_assignment(
    group_rows: list[dict],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    if len(group_rows) < 3:
        raise RuntimeError(
            "Group-aware split membutuhkan minimal 3 group per source. "
            f"Ditemukan {len(group_rows)}."
        )

    split_names = ("train", "valid", "test")
    n = len(group_rows)
    best_assignment = None
    best_score = float("inf")

    if n <= 11:
        for assignment in itertools.product(range(3), repeat=n):
            if set(assignment) != {0, 1, 2}:
                continue
            score = assignment_score(group_rows, assignment, ratios)
            if score < best_score:
                best_score = score
                best_assignment = assignment
    else:
        rng = random.Random(seed)
        trials = max(50000, n * 5000)

        for _ in range(trials):
            order = list(range(n))
            rng.shuffle(order)

            assignment = [-1] * n
            forced_splits = [0, 1, 2]
            rng.shuffle(forced_splits)

            for idx, split_idx in zip(order[:3], forced_splits):
                assignment[idx] = split_idx

            cumulative = (
                ratios[0],
                ratios[0] + ratios[1],
                1.0,
            )

            for idx in order[3:]:
                u = rng.random()
                if u < cumulative[0]:
                    assignment[idx] = 0
                elif u < cumulative[1]:
                    assignment[idx] = 1
                else:
                    assignment[idx] = 2

            candidate = tuple(assignment)
            score = assignment_score(group_rows, candidate, ratios)

            if score < best_score:
                best_score = score
                best_assignment = candidate

    if best_assignment is None:
        raise RuntimeError("Gagal menemukan group-aware split.")

    return {
        row["group"]: split_names[idx]
        for row, idx in zip(group_rows, best_assignment)
    }


def copy_item(
    item: dict,
    assigned_split: str,
    output_root: Path,
) -> dict:
    image_path: Path = item["image_path"]
    prefix = item["source"]

    raw_stem = (
        clean_source_a_name(image_path.stem)
        if prefix == "A"
        else strip_roboflow_hash(image_path.stem)
    )
    token = hashlib.sha1(
        (
            f"{prefix}|{item['original_split']}|"
            f"{image_path.resolve()}"
        ).encode("utf-8")
    ).hexdigest()[:10]

    new_name = (
        f"{prefix}__{safe_name(raw_stem)}"
        f"__{token}{image_path.suffix.lower()}"
    )

    out_images = output_root / assigned_split / "images"
    out_labels = output_root / assigned_split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    out_image = out_images / new_name
    out_label = out_labels / Path(new_name).with_suffix(".txt").name

    shutil.copy2(image_path, out_image)
    write_yolo_labels(item["mapped_rows"], out_label)

    return {
        "source": prefix,
        "group": item["group"],
        "assigned_split": assigned_split,
        "original_split": item["original_split"],
        "original_image": str(image_path),
        "output_image": str(out_image),
        "positive_boxes": item["positive_boxes"],
        "is_negative": item["is_negative"],
        "sha256": item["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--method-config", default="configs/method_b.yaml")
    parser.add_argument("--output", default="data/processed/harmonized")
    parser.add_argument("--report-dir", default="reports/method_b")
    args = parser.parse_args()

    datasets_cfg = load_yaml(args.datasets_config)
    method_cfg = load_yaml(args.method_config)

    split_cfg = method_cfg.get("group_split", {})
    ratios = (
        float(split_cfg.get("train", 0.70)),
        float(split_cfg.get("valid", 0.15)),
        float(split_cfg.get("test", 0.15)),
    )
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios harus berjumlah 1.0, mendapat {ratios}")

    seed = int(method_cfg.get("seed", 42))
    output_root = Path(args.output)
    report_dir = Path(args.report_dir)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_manifest = []
    all_duplicates = []
    summary = {"ratios": ratios, "sources": {}}

    for source_index, (source_name, source_cfg) in enumerate(
        datasets_cfg["sources"].items()
    ):
        items, duplicates = collect_source(source_name, source_cfg)
        all_duplicates.extend(duplicates)

        group_rows = summarize_groups(items)
        assignment = choose_group_assignment(
            group_rows,
            ratios,
            seed + source_index * 1009,
        )

        for item in items:
            assigned_split = assignment[item["group"]]
            all_manifest.append(
                copy_item(item, assigned_split, output_root)
            )

        prefix = str(source_cfg["prefix"]).upper()
        source_summary = {}
        for split in SPLITS:
            members = [
                x for x in all_manifest
                if x["source"] == prefix
                and x["assigned_split"] == split
            ]
            source_summary[split] = {
                "images": len(members),
                "positive_boxes": sum(x["positive_boxes"] for x in members),
                "negative_images": sum(x["is_negative"] for x in members),
                "groups": sorted({x["group"] for x in members}),
            }

        summary["sources"][prefix] = source_summary

    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)

    for row in all_manifest:
        group_splits[row["group"]].add(row["assigned_split"])
        hash_splits[row["sha256"]].add(row["assigned_split"])

    leaking_groups = {
        k: sorted(v)
        for k, v in group_splits.items()
        if len(v) > 1
    }
    leaking_hashes = {
        k: sorted(v)
        for k, v in hash_splits.items()
        if len(v) > 1
    }

    if leaking_groups:
        raise RuntimeError(
            "GROUP LEAKAGE ditemukan:\n"
            + json.dumps(leaking_groups, indent=2)
        )
    if leaking_hashes:
        raise RuntimeError(
            "EXACT IMAGE LEAKAGE ditemukan:\n"
            + json.dumps(leaking_hashes, indent=2)
        )

    for source, source_summary in summary["sources"].items():
        for split in SPLITS:
            if source_summary[split]["images"] == 0:
                raise RuntimeError(
                    f"Source {source} tidak memiliki image pada split {split}"
                )

    save_yaml(
        {
            "path": str(output_root.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": {0: "BSR"},
            "nc": 1,
        },
        output_root / "data.yaml",
    )

    manifest_path = report_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_manifest[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_manifest)

    duplicate_path = report_dir / "exact_duplicates_dropped.csv"
    if all_duplicates:
        with duplicate_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(all_duplicates[0].keys()),
            )
            writer.writeheader()
            writer.writerows(all_duplicates)
    else:
        duplicate_path.write_text(
            "source,kept,dropped,kept_original_split,dropped_original_split,sha256\n",
            encoding="utf-8",
        )

    summary["total_images"] = len(all_manifest)
    summary["exact_duplicates_dropped"] = len(all_duplicates)
    summary["group_leakage"] = 0
    summary["exact_hash_leakage"] = 0
    save_json(summary, report_dir / "group_split_summary.json")

    print()
    print("=" * 80)
    print("METHOD B - GROUP-AWARE SPLIT")
    print("=" * 80)
    for source, source_summary in summary["sources"].items():
        print(f"\nSOURCE {source}")
        for split in SPLITS:
            x = source_summary[split]
            print(
                f"  {split:5s}: images={x['images']:4d} "
                f"boxes={x['positive_boxes']:4d} "
                f"negative={x['negative_images']:4d} "
                f"groups={len(x['groups']):2d}"
            )
            print("          " + ", ".join(x["groups"]))

    print()
    print("[PASS] group leakage = 0")
    print("[PASS] exact image leakage = 0")
    print(f"[REPORT] {manifest_path}")
    print(f"[REPORT] {report_dir / 'group_split_summary.json'}")


if __name__ == "__main__":
    main()
