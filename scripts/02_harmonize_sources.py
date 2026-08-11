from __future__ import annotations

import argparse
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
    safe_copy,
    save_yaml,
    write_yolo_labels,
)


def harmonize_source(source_name: str, source: dict, output_root: Path) -> None:
    source_root = Path(source["output"])
    data_yaml_path = find_data_yaml(source_root)
    dataset_root = data_yaml_path.parent
    names = parse_names(load_yaml(data_yaml_path))

    positives = {normalize_class_name(x) for x in source.get("positive_names", [])}
    prefix = str(source["prefix"]).strip()

    print(f"[SOURCE] {source_name} classes={names}")

    for split in ("train", "valid", "test"):
        found = find_split_dir(dataset_root, split)
        if not found:
            print(f"[WARN] {source_name}: split {split} tidak ditemukan.")
            continue
        images_dir, labels_dir = found

        out_images = output_root / split / "images"
        out_labels = output_root / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for image_path in iter_images(images_dir):
            rel = image_path.relative_to(images_dir)
            stem = "__".join(rel.with_suffix("").parts)
            new_name = f"{prefix}__{stem}{image_path.suffix.lower()}"
            out_image = out_images / new_name
            out_label = out_labels / Path(new_name).with_suffix(".txt").name

            label_path = matching_label(image_path, images_dir, labels_dir)
            rows = read_yolo_labels(label_path)
            mapped = []

            for row in rows:
                old_class = int(row[0])
                old_name = normalize_class_name(names.get(old_class, f"class_{old_class}"))
                if old_name in positives:
                    mapped.append([0, *row[1:5]])
                # Semua kelas selain target dibuang dari annotations.
                # Citra tetap dipertahankan sehingga objek tersebut menjadi background.

            safe_copy(image_path, out_image)
            write_yolo_labels(mapped, out_label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--output", default="data/processed/harmonized")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for source_name, source in cfg["sources"].items():
        harmonize_source(source_name, source, output_root)

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "BSR"},
        "nc": 1,
    }
    save_yaml(data_yaml, output_root / "data.yaml")
    print(f"[OK] Harmonized dataset: {output_root}")


if __name__ == "__main__":
    main()
