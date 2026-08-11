from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (
    iter_images,
    load_yaml,
    matching_label,
    read_image,
    read_yolo_labels,
    save_json,
    save_yaml,
    seed_everything,
    write_yolo_labels,
)

SPLITS = ("train", "valid", "test")


def yolo_to_xyxy(row, width, height):
    cls, cx, cy, bw, bh = row
    return (
        int(cls),
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    )


def xyxy_to_yolo(cls, x1, y1, x2, y2, width, height):
    return [
        cls,
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def positions(length: int, tile: int, overlap: float) -> list[int]:
    """Return a deterministic edge-covering grid."""
    if length <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    last = length - tile
    estimated = max(1, math.ceil(last / stride))
    if estimated == 1:
        return [0, last]
    return sorted(
        set(int(round(i * last / estimated)) for i in range(estimated + 1))
    )


def should_tile(image_name: str, prefixes: list[str]) -> bool:
    return not prefixes or any(
        image_name.startswith(f"{prefix}__") for prefix in prefixes
    )


def stable_rank(seed: int, split: str, record: dict) -> str:
    identity = (
        f"{seed}|{split}|{record['source']}|"
        f"{Path(record['output_image']).name}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/harmonized")
    parser.add_argument("--output", default="data/processed/tiled")
    parser.add_argument("--config", default="configs/method_b.yaml")
    parser.add_argument("--report-dir", default="reports/method_b")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    tcfg = cfg.get("tiling", {})
    enabled = bool(tcfg.get("enabled", True))
    prefixes = [str(x).upper() for x in tcfg.get("source_prefixes", ["A"])]
    tile_w = int(tcfg.get("tile_width", 320))
    tile_h = int(tcfg.get("tile_height", 320))
    overlap = float(tcfg.get("overlap", 0.20))
    min_visibility = float(tcfg.get("min_box_visibility", 0.50))
    keep_empty = bool(tcfg.get("keep_empty_tiles", True))
    train_max_background_ratio = float(
        tcfg.get("train_max_background_to_positive_ratio", 0.5)
    )
    if train_max_background_ratio < 0:
        raise ValueError("train background/positive ratio must be non-negative")

    input_root = Path(args.input)
    output_root = Path(args.output)
    report_dir = Path(args.report_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    diagnostics: dict = {
        "seed": seed,
        "policy": {
            "grid": "deterministic_edge_covering",
            "train_background_selection": "sha256_rank",
            "train_max_background_to_positive_ratio": train_max_background_ratio,
            "validation_test_backgrounds": "all_preserved",
            "positive_centric_alternative": (
                "investigated_not_adopted: changing the grid would change the "
                "primary evaluation distribution; deterministic grid plus a "
                "secondary balanced view separates realism from sensitivity"
            ),
        },
        "splits": {},
    }

    for split in SPLITS:
        in_images = input_root / split / "images"
        in_labels = input_root / split / "labels"
        out_images = output_root / split / "images"
        out_labels = output_root / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        positive_tiles: list[tuple] = []
        empty_tiles: list[tuple] = []
        passthrough: list[dict] = []
        candidate_counts = Counter()
        removed_visibility = Counter()
        clipped_counts = Counter()

        for image_path in iter_images(in_images):
            label_path = matching_label(image_path, in_images, in_labels)
            rows = read_yolo_labels(label_path)
            source = image_path.name.split("__", 1)[0]

            if not enabled or not should_tile(image_path.name, prefixes):
                target_image = out_images / image_path.name
                target_label = out_labels / image_path.with_suffix(".txt").name
                shutil.copy2(image_path, target_image)
                write_yolo_labels(rows, target_label)
                passthrough.append(
                    {
                        "split": split,
                        "source": source,
                        "original_image": str(image_path),
                        "output_image": str(target_image),
                        "tile_x": "",
                        "tile_y": "",
                        "tile_width": "",
                        "tile_height": "",
                        "positive_boxes": len(rows),
                        "boxes_clipped": 0,
                        "boxes_removed_min_visibility": 0,
                        "kind": "passthrough",
                    }
                )
                continue

            image = read_image(image_path)
            h, w = image.shape[:2]
            if w <= tile_w and h <= tile_h:
                target_image = out_images / image_path.name
                target_label = out_labels / image_path.with_suffix(".txt").name
                cv2.imwrite(str(target_image), image)
                write_yolo_labels(rows, target_label)
                passthrough.append(
                    {
                        "split": split,
                        "source": source,
                        "original_image": str(image_path),
                        "output_image": str(target_image),
                        "tile_x": 0,
                        "tile_y": 0,
                        "tile_width": w,
                        "tile_height": h,
                        "positive_boxes": len(rows),
                        "boxes_clipped": 0,
                        "boxes_removed_min_visibility": 0,
                        "kind": "passthrough_small",
                    }
                )
                continue

            boxes = [yolo_to_xyxy(row, w, h) for row in rows]
            for y0 in positions(h, tile_h, overlap):
                for x0 in positions(w, tile_w, overlap):
                    x2t, y2t = min(w, x0 + tile_w), min(h, y0 + tile_h)
                    crop = image[y0:y2t, x0:x2t]
                    ch, cw = crop.shape[:2]
                    tile_rows = []
                    tile_clipped = 0
                    tile_removed = 0

                    for cls, bx1, by1, bx2, by2 in boxes:
                        ix1, iy1 = max(bx1, x0), max(by1, y0)
                        ix2, iy2 = min(bx2, x2t), min(by2, y2t)
                        inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        if inter_area <= 0:
                            continue
                        box_area = max(1e-9, (bx2 - bx1) * (by2 - by1))
                        visibility = inter_area / box_area
                        is_clipped = (
                            abs(ix1 - bx1) > 1e-6
                            or abs(iy1 - by1) > 1e-6
                            or abs(ix2 - bx2) > 1e-6
                            or abs(iy2 - by2) > 1e-6
                        )
                        if visibility < min_visibility:
                            tile_removed += 1
                            removed_visibility[(split, source)] += 1
                            continue
                        if is_clipped:
                            tile_clipped += 1
                            clipped_counts[(split, source)] += 1
                        tile_rows.append(
                            xyxy_to_yolo(
                                cls,
                                ix1 - x0,
                                iy1 - y0,
                                ix2 - x0,
                                iy2 - y0,
                                cw,
                                ch,
                            )
                        )

                    tile_name = (
                        f"{image_path.stem}__tile_x{x0}_y{y0}"
                        f"{image_path.suffix.lower()}"
                    )
                    out_img = out_images / tile_name
                    out_lbl = out_labels / Path(tile_name).with_suffix(".txt").name
                    kind = "positive_tile" if tile_rows else "empty_tile"
                    candidate_counts[(source, kind)] += 1
                    record = {
                        "split": split,
                        "source": source,
                        "original_image": str(image_path),
                        "output_image": str(out_img),
                        "tile_x": x0,
                        "tile_y": y0,
                        "tile_width": cw,
                        "tile_height": ch,
                        "positive_boxes": len(tile_rows),
                        "boxes_clipped": tile_clipped,
                        "boxes_removed_min_visibility": tile_removed,
                        "kind": kind,
                    }
                    item = (crop, out_img, out_lbl, tile_rows, record)
                    if tile_rows:
                        positive_tiles.append(item)
                    elif keep_empty:
                        empty_tiles.append(item)

        candidate_empty_count = len(empty_tiles)
        if split == "train":
            max_empty = math.floor(
                len(positive_tiles) * train_max_background_ratio + 1e-12
            )
            empty_tiles = sorted(
                empty_tiles,
                key=lambda item: stable_rank(seed, split, item[4]),
            )[:max_empty]

        for crop, out_img, out_lbl, rows, record in positive_tiles + empty_tiles:
            if not cv2.imwrite(str(out_img), crop):
                raise RuntimeError(f"Could not write tile: {out_img}")
            write_yolo_labels(rows, out_lbl)
            manifest.append(record)
        manifest.extend(passthrough)

        by_source = defaultdict(lambda: Counter())
        for row in [*positive_tiles, *empty_tiles]:
            rec = row[4]
            by_source[rec["source"]][rec["kind"]] += 1
        for rec in passthrough:
            by_source[rec["source"]]["passthrough"] += 1
            by_source[rec["source"]][
                "positive_passthrough" if rec["positive_boxes"] else "negative_passthrough"
            ] += 1
        diagnostics["splits"][split] = {
            "candidate_positive_tiles": len(positive_tiles),
            "candidate_background_tiles": candidate_empty_count,
            "kept_background_tiles": len(empty_tiles),
            "dropped_background_tiles": candidate_empty_count - len(empty_tiles),
            "by_source": {source: dict(counts) for source, counts in by_source.items()},
            "boxes_clipped": sum(v for (sp, _), v in clipped_counts.items() if sp == split),
            "boxes_removed_min_visibility": sum(
                v for (sp, _), v in removed_visibility.items() if sp == split
            ),
        }

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
    if not manifest:
        raise RuntimeError("Tiling produced an empty dataset")
    manifest_path = report_dir / "tile_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    save_json(diagnostics, report_dir / "tiling_diagnostics.json")

    print("\n" + "=" * 80)
    print("METHOD B - SMALL OBJECT TILING")
    print("=" * 80)
    for split in SPLITS:
        rows = [x for x in manifest if x["split"] == split]
        print(
            f"{split:5s}: total={len(rows):4d} "
            f"A={sum(x['source'] == 'A' for x in rows):4d} "
            f"B={sum(x['source'] == 'B' for x in rows):4d} "
            f"positive={sum(x['positive_boxes'] > 0 for x in rows):4d} "
            f"negative={sum(x['positive_boxes'] == 0 for x in rows):4d}"
        )
    print(f"[PASS] Tiling prefixes = {prefixes}")
    print(f"[PASS] Tile size = {tile_w}x{tile_h}, overlap={overlap:.2f}")
    print("[PASS] Full validation/test background tiles preserved")
    print(f"[REPORT] {manifest_path}")
    print(f"[REPORT] {report_dir / 'tiling_diagnostics.json'}")


if __name__ == "__main__":
    main()
