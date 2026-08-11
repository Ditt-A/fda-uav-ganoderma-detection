from __future__ import annotations

import argparse
import math
import random
import shutil
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
    save_yaml,
    seed_everything,
    write_yolo_labels,
)


def yolo_to_xyxy(row, width, height):
    cls, cx, cy, bw, bh = row
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    return int(cls), x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, width, height):
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return [cls, cx, cy, bw, bh]


def positions(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    pos = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if pos[-1] != last:
        pos.append(last)
    return pos


def should_tile(image_name: str, prefixes: list[str]) -> bool:
    if not prefixes:
        return True
    return any(image_name.startswith(f"{prefix}__") for prefix in prefixes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/harmonized")
    parser.add_argument("--output", default="data/processed/tiled")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    tcfg = cfg.get("tiling", {})
    enabled = bool(tcfg.get("enabled", True))
    prefixes = [str(x) for x in tcfg.get("source_prefixes", [])]
    tile_w = int(tcfg.get("tile_width", 640))
    tile_h = int(tcfg.get("tile_height", 640))
    overlap = float(tcfg.get("overlap", 0.2))
    min_visibility = float(tcfg.get("min_box_visibility", 0.5))
    keep_empty = bool(tcfg.get("keep_empty_tiles", True))
    max_empty_ratio = float(tcfg.get("max_empty_to_positive_ratio", 2.0))

    input_root = Path(args.input)
    output_root = Path(args.output)
    if output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "valid", "test"):
        in_images = input_root / split / "images"
        in_labels = input_root / split / "labels"
        out_images = output_root / split / "images"
        out_labels = output_root / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        positive_tiles = []
        empty_tiles = []

        for image_path in iter_images(in_images):
            label_path = matching_label(image_path, in_images, in_labels)
            rows = read_yolo_labels(label_path)

            if not enabled or not should_tile(image_path.name, prefixes):
                target_image = out_images / image_path.name
                target_label = out_labels / image_path.with_suffix(".txt").name
                shutil.copy2(image_path, target_image)
                shutil.copy2(label_path, target_label) if label_path.exists() else write_yolo_labels([], target_label)
                continue

            image = read_image(image_path)
            h, w = image.shape[:2]

            # Jika citra lebih kecil dari tile, copy apa adanya.
            if w <= tile_w and h <= tile_h:
                target_image = out_images / image_path.name
                target_label = out_labels / image_path.with_suffix(".txt").name
                cv2.imwrite(str(target_image), image)
                write_yolo_labels(rows, target_label)
                continue

            boxes = [yolo_to_xyxy(row, w, h) for row in rows]

            for y0 in positions(h, tile_h, overlap):
                for x0 in positions(w, tile_w, overlap):
                    x1t, y1t = x0, y0
                    x2t, y2t = min(w, x0 + tile_w), min(h, y0 + tile_h)
                    crop = image[y1t:y2t, x1t:x2t]
                    ch, cw = crop.shape[:2]
                    tile_rows = []

                    for cls, bx1, by1, bx2, by2 in boxes:
                        ix1 = max(bx1, x1t)
                        iy1 = max(by1, y1t)
                        ix2 = min(bx2, x2t)
                        iy2 = min(by2, y2t)
                        inter_w = max(0.0, ix2 - ix1)
                        inter_h = max(0.0, iy2 - iy1)
                        inter_area = inter_w * inter_h
                        box_area = max(1e-9, (bx2 - bx1) * (by2 - by1))
                        visibility = inter_area / box_area
                        if visibility < min_visibility or inter_area <= 0:
                            continue
                        tile_rows.append(
                            xyxy_to_yolo(
                                cls,
                                ix1 - x1t, iy1 - y1t,
                                ix2 - x1t, iy2 - y1t,
                                cw, ch,
                            )
                        )

                    stem = image_path.stem
                    tile_name = f"{stem}__x{x0}_y{y0}{image_path.suffix.lower()}"
                    out_img = out_images / tile_name
                    out_lbl = out_labels / Path(tile_name).with_suffix(".txt").name

                    item = (crop, out_img, out_lbl, tile_rows)
                    if tile_rows:
                        positive_tiles.append(item)
                    elif keep_empty:
                        empty_tiles.append(item)

        if split == "train" and keep_empty and positive_tiles:
            random.shuffle(empty_tiles)
            max_empty = int(math.ceil(len(positive_tiles) * max_empty_ratio))
            empty_tiles = empty_tiles[:max_empty]

        for crop, out_img, out_lbl, rows in positive_tiles + empty_tiles:
            cv2.imwrite(str(out_img), crop)
            write_yolo_labels(rows, out_lbl)

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
    print(f"[OK] Tiled dataset: {output_root}")


if __name__ == "__main__":
    main()
