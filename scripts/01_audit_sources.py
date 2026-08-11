from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (
    find_data_yaml,
    find_split_dir,
    image_hash,
    iter_images,
    load_yaml,
    matching_label,
    normalize_class_name,
    parse_names,
    read_yolo_labels,
    save_json,
)


def audit_source(name: str, source: dict) -> dict:
    root = Path(source["output"])
    data_yaml_path = find_data_yaml(root)
    data_yaml = load_yaml(data_yaml_path)
    names = parse_names(data_yaml)

    report = {
        "source": name,
        "root": str(root),
        "data_yaml": str(data_yaml_path),
        "classes": names,
        "splits": {},
    }

    global_hashes = Counter()

    for split in ("train", "valid", "test"):
        found = find_split_dir(data_yaml_path.parent, split)
        if not found:
            report["splits"][split] = {"missing": True}
            continue
        images_dir, labels_dir = found

        class_counts = Counter()
        widths, heights, rel_areas, boxes_per_image = [], [], [], []
        empty_images = 0
        corrupt_images = 0
        image_count = 0

        for image_path in iter_images(images_dir):
            image_count += 1
            global_hashes[image_hash(image_path)] += 1

            image = cv2.imread(str(image_path))
            if image is None:
                corrupt_images += 1
                continue
            h, w = image.shape[:2]
            widths.append(w)
            heights.append(h)

            label_path = matching_label(image_path, images_dir, labels_dir)
            rows = read_yolo_labels(label_path)
            boxes_per_image.append(len(rows))
            if not rows:
                empty_images += 1
            for row in rows:
                class_id, _, _, bw, bh = row
                class_counts[int(class_id)] += 1
                rel_areas.append(float(bw * bh))

        report["splits"][split] = {
            "image_count": image_count,
            "empty_images": empty_images,
            "corrupt_images": corrupt_images,
            "class_box_counts": {
                f"{class_id}:{names.get(class_id, 'UNKNOWN')}": count
                for class_id, count in sorted(class_counts.items())
            },
            "width_median": float(np.median(widths)) if widths else None,
            "height_median": float(np.median(heights)) if heights else None,
            "boxes_per_image_median": float(np.median(boxes_per_image))
            if boxes_per_image else None,
            "relative_box_area_median": float(np.median(rel_areas))
            if rel_areas else None,
        }

    report["exact_duplicate_files"] = sum(v - 1 for v in global_hashes.values() if v > 1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for name, source in cfg["sources"].items():
        report = audit_source(name, source)
        output = Path("reports") / f"{name}_audit.json"
        save_json(report, output)
        print(f"[OK] {output}")
        print(report)


if __name__ == "__main__":
    main()
