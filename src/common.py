from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val", "validation"),
    "test": ("test",),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_class_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def parse_names(data_yaml: dict) -> dict[int, str]:
    names = data_yaml.get("names", {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError("Field 'names' pada data.yaml harus list atau dict.")


def find_data_yaml(root: str | Path) -> Path:
    root = Path(root)
    candidates = list(root.rglob("data.yaml")) + list(root.rglob("data.yml"))
    if not candidates:
        raise FileNotFoundError(f"Tidak menemukan data.yaml di {root}")
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def find_split_dir(dataset_root: str | Path, canonical_split: str) -> tuple[Path, Path] | None:
    root = Path(dataset_root)
    for alias in SPLIT_ALIASES[canonical_split]:
        direct_images = root / alias / "images"
        direct_labels = root / alias / "labels"
        if direct_images.exists():
            return direct_images, direct_labels

        alt_images = root / "images" / alias
        alt_labels = root / "labels" / alias
        if alt_images.exists():
            return alt_images, alt_labels
    return None


def iter_images(images_dir: str | Path) -> Iterable[Path]:
    images_dir = Path(images_dir)
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def matching_label(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def read_yolo_labels(path: str | Path) -> list[list[float]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            raise ValueError(f"Label tidak valid {path}:{lineno}: {raw}")
        rows.append([float(x) for x in parts[:5]])
    return rows


def write_yolo_labels(rows: list[list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        f"{int(row[0])} " + " ".join(f"{float(x):.8f}" for x in row[1:5])
        for row in rows
    )
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def safe_copy(src: str | Path, dst: str | Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def image_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {path}")
    return img


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
