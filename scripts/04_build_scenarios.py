from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
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
from src.fda import fourier_amplitude_mix

SCENARIOS = ("B0", "B1", "B2")
EVAL_SPLITS = ("valid", "test")


def build_geometric_transform(cfg):
    """Build the only permitted offline spatial policy: H/V flips."""
    aug_cfg = cfg.get("augmentation", {})
    hp = float(aug_cfg.get("horizontal_flip_p", 0.0))
    vp = float(aug_cfg.get("vertical_flip_p", 0.0))
    unsupported = {
        key: float(aug_cfg.get(key, 0.0))
        for key in (
            "rotate90_p",
            "affine_p",
            "scale_limit",
            "translate_limit",
            "rotate_limit",
            "shear_limit",
        )
    }
    active = {key: value for key, value in unsupported.items() if abs(value) > 1e-12}
    if active:
        raise RuntimeError(f"Only H/V flips are permitted; active transforms: {active}")
    if not 0.0 <= hp <= 1.0 or not 0.0 <= vp <= 1.0:
        raise ValueError("horizontal/vertical flip probabilities must be in [0, 1]")

    def transform(*, image, bboxes, class_labels):
        output_image = image.copy()
        output_boxes = [list(map(float, box)) for box in bboxes]
        hflip = random.random() < hp
        vflip = random.random() < vp
        if hflip:
            output_image = cv2.flip(output_image, 1)
            for box in output_boxes:
                box[0] = 1.0 - box[0]
        if vflip:
            output_image = cv2.flip(output_image, 0)
            for box in output_boxes:
                box[1] = 1.0 - box[1]
        for box in output_boxes:
            cx, cy, width, height = box
            if not (
                0.0 <= cx <= 1.0
                and 0.0 <= cy <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            ):
                raise RuntimeError(f"Invalid bbox after flip: {box}")
        return {
            "image": output_image,
            "bboxes": output_boxes,
            "class_labels": list(class_labels),
            "hflip": hflip,
            "vflip": vflip,
        }

    print(f"[METHOD B AUGMENT] HFlip={hp}, VFlip={vp}")
    print("[METHOD B AUGMENT] Rotate/Affine/Translate/Scale/Shear=OFF")
    return transform


def source_prefix(path: Path) -> str:
    return path.name.split("__", 1)[0].upper() if "__" in path.name else "UNKNOWN"


def copy_dataset(input_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite scenario: {output_root}. Archive it first."
        )
    shutil.copytree(input_root, output_root)


def archive_existing_scenarios(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parents[1]
    archive = project_root / "archives" / f"before_b0_b1_b2_{stamp}"
    destination = archive / "data" / "processed" / "scenarios"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_root), str(destination))
    print(f"[ARCHIVE] {output_root} -> {destination}")
    return archive


def save_augmented(image, rows, transform, out_image: Path, out_label: Path) -> dict:
    transformed = transform(
        image=image,
        bboxes=[row[1:5] for row in rows],
        class_labels=[int(row[0]) for row in rows],
    )
    if not cv2.imwrite(str(out_image), transformed["image"]):
        raise RuntimeError(f"Could not write augmented image: {out_image}")
    write_yolo_labels(
        [
            [int(cls), *list(map(float, bbox))]
            for cls, bbox in zip(
                transformed["class_labels"], transformed["bboxes"]
            )
        ],
        out_label,
    )
    return {"hflip": transformed["hflip"], "vflip": transformed["vflip"]}


def stable_index(seed: int, identity: str, length: int) -> int:
    if length <= 0:
        raise ValueError("Cannot select from an empty donor pool")
    digest = hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % length


def write_data_yaml(scenario: Path, balanced: bool = False) -> None:
    suffix = "balanced_diagnostic" if balanced else None
    data = {
        "path": str(scenario.resolve()),
        "train": "train/images",
        "val": f"{suffix}/valid/images" if suffix else "valid/images",
        "test": f"{suffix}/test/images" if suffix else "test/images",
        "names": {0: "BSR"},
        "nc": 1,
    }
    save_yaml(data, scenario / ("data_balanced.yaml" if balanced else "data.yaml"))


def balanced_names(root: Path, split: str, ratio: float, seed: int) -> list[str]:
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    positives: list[Path] = []
    negatives_by_source: dict[str, list[Path]] = defaultdict(list)
    positives_by_source = Counter()
    for image_path in iter_images(images_dir):
        rows = read_yolo_labels(matching_label(image_path, images_dir, labels_dir))
        source = source_prefix(image_path)
        if rows:
            positives.append(image_path)
            positives_by_source[source] += 1
        else:
            negatives_by_source[source].append(image_path)

    selected = list(positives)
    for source, candidates in sorted(negatives_by_source.items()):
        limit = int(positives_by_source[source] * ratio)
        ranked = sorted(
            candidates,
            key=lambda p: hashlib.sha256(
                f"{seed}|balanced|{split}|{source}|{p.name}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[:limit])
    return sorted(path.name for path in selected)


def build_balanced_views(
    roots: dict[str, Path], ratio: float, seed: int
) -> dict[str, dict]:
    """Select once from B0, then copy the identical view to every scenario."""
    summary: dict[str, dict] = {}
    for split in EVAL_SPLITS:
        names = balanced_names(roots["B0"], split, ratio, seed)
        summary[split] = {"images": len(names), "names_sha256": hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest()}
        for scenario, root in roots.items():
            out_images = root / "balanced_diagnostic" / split / "images"
            out_labels = root / "balanced_diagnostic" / split / "labels"
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            for name in names:
                src_image = root / split / "images" / name
                src_label = root / split / "labels" / Path(name).with_suffix(".txt").name
                if not src_image.exists() or not src_label.exists():
                    raise RuntimeError(f"Scenario {scenario} lacks balanced member {name}")
                shutil.copy2(src_image, out_images / name)
                shutil.copy2(src_label, out_labels / src_label.name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/tiled")
    parser.add_argument("--output-root", default="data/processed/scenarios")
    parser.add_argument("--config", default="configs/method_b.yaml")
    parser.add_argument("--report-dir", default="reports/method_b")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    random.seed(seed)
    input_root = Path(args.input)
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    archived_to = archive_existing_scenarios(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    roots = {scenario: output_root / scenario for scenario in SCENARIOS}
    copy_dataset(input_root, roots["B0"])
    copy_dataset(roots["B0"], roots["B1"])

    aug_cfg = cfg.get("augmentation", {})
    geo_copies = int(aug_cfg.get("geometric_copies_per_image", 1))
    fda_copies = int(aug_cfg.get("fda_copies_per_image", 1))
    if geo_copies != 1 or fda_copies != 1:
        raise RuntimeError(
            "The B0/B1/B2 design requires exactly one geometric and one FDA copy "
            f"(got geo={geo_copies}, fda={fda_copies})"
        )
    transform = build_geometric_transform(cfg)

    b1_images = roots["B1"] / "train" / "images"
    b1_labels = roots["B1"] / "train" / "labels"
    originals = list(iter_images(b1_images))
    augmentation_manifest: list[dict] = []
    for image_path in originals:
        label_path = matching_label(image_path, b1_images, b1_labels)
        rows = read_yolo_labels(label_path)
        out_name = f"{image_path.stem}__GEO0{image_path.suffix.lower()}"
        flags = save_augmented(
            read_image(image_path),
            rows,
            transform,
            b1_images / out_name,
            b1_labels / Path(out_name).with_suffix(".txt").name,
        )
        augmentation_manifest.append(
            {
                "scenario": "B1",
                "kind": "GEO",
                "content": image_path.name,
                "content_source": source_prefix(image_path),
                "donor": None,
                "donor_source": None,
                **flags,
            }
        )

    # B2 inherits originals and the exact B1 geometric copies byte-for-byte.
    copy_dataset(roots["B1"], roots["B2"])
    b2_images = roots["B2"] / "train" / "images"
    b2_labels = roots["B2"] / "train" / "labels"
    fda_cfg = cfg.get("fda", {})
    if not bool(fda_cfg.get("enabled", True)):
        raise RuntimeError("B2 requires fda.enabled=true")
    beta = float(fda_cfg.get("beta", 0.01))
    lambda_mix = float(fda_cfg.get("lambda_mix", 0.35))
    bidirectional = bool(fda_cfg.get("bidirectional", True))
    donor_resize = bool(fda_cfg.get("donor_resize", True))
    one_way_content = str(fda_cfg.get("one_way_content_source", "")).upper()
    one_way_donor = str(fda_cfg.get("one_way_donor_source", "")).upper()
    if not bidirectional:
        if not one_way_content or not one_way_donor or one_way_content == one_way_donor:
            raise RuntimeError(
                "bidirectional=false requires distinct one_way_content_source and "
                "one_way_donor_source"
            )

    original_b2 = [b2_images / path.name for path in originals]
    by_source: dict[str, list[Path]] = defaultdict(list)
    for path in original_b2:
        by_source[source_prefix(path)].append(path)
    for paths in by_source.values():
        paths.sort(key=lambda path: path.name)

    fda_created = 0
    for image_path in original_b2:
        content_source = source_prefix(image_path)
        if not bidirectional and content_source != one_way_content:
            continue
        donor_sources = (
            sorted(source for source in by_source if source != content_source)
            if bidirectional
            else [one_way_donor]
        )
        donor_pool = sorted(
            [path for source in donor_sources for path in by_source.get(source, [])],
            key=lambda path: path.name,
        )
        if not donor_pool:
            raise RuntimeError(
                f"No FDA donor for content source {content_source}; requested {donor_sources}"
            )
        donor_path = donor_pool[
            stable_index(seed, f"FDA|{image_path.name}|0", len(donor_pool))
        ]
        image = read_image(image_path)
        donor = read_image(donor_path)
        sizes_differ = donor.shape[:2] != image.shape[:2]
        mixed = fourier_amplitude_mix(
            image,
            donor,
            beta=beta,
            lambda_mix=lambda_mix,
            donor_resize=donor_resize,
        )
        rows = read_yolo_labels(matching_label(image_path, b2_images, b2_labels))
        out_name = (
            f"{image_path.stem}__FDA_{source_prefix(donor_path)}_0"
            f"{image_path.suffix.lower()}"
        )
        flags = save_augmented(
            mixed,
            rows,
            transform,
            b2_images / out_name,
            b2_labels / Path(out_name).with_suffix(".txt").name,
        )
        augmentation_manifest.append(
            {
                "scenario": "B2",
                "kind": "FDA",
                "content": image_path.name,
                "content_source": content_source,
                "donor": donor_path.name,
                "donor_source": source_prefix(donor_path),
                "donor_resized": sizes_differ,
                **flags,
            }
        )
        fda_created += 1

    balanced_cfg = cfg.get("evaluation", {}).get("balanced_diagnostic", {})
    balanced_ratio = float(
        balanced_cfg.get("max_background_to_positive_ratio", 1.0)
    )
    if balanced_ratio < 0:
        raise ValueError("balanced background/positive ratio must be non-negative")
    balanced = build_balanced_views(roots, balanced_ratio, seed)
    for root in roots.values():
        write_data_yaml(root, balanced=False)
        write_data_yaml(root, balanced=True)

    counts = {
        scenario: len(list(iter_images(root / "train" / "images")))
        for scenario, root in roots.items()
    }
    summary = {
        "seed": seed,
        "archive_created": str(archived_to) if archived_to else None,
        "scenario_design": {
            "B0": "original training data only",
            "B1": "B0 plus exactly one H/V geometric copy",
            "B2": "B1 plus exactly one geometrically transformed FDA copy per eligible original",
            "interpretation": {
                "B1_minus_B0": "H/V geometric augmentation contribution",
                "B2_minus_B1": "incremental FDA contribution on the same geometric pipeline",
                "warning": "B2-B0 is not a pure FDA effect",
            },
        },
        "train_counts": counts,
        "balanced_diagnostic": {
            "max_background_to_positive_ratio": balanced_ratio,
            "splits": balanced,
        },
        "fda_policy": {
            "enabled": True,
            "bidirectional": bidirectional,
            "donor_resize": donor_resize,
            "one_way_content_source": one_way_content if not bidirectional else None,
            "one_way_donor_source": one_way_donor if not bidirectional else None,
            "beta": beta,
            "lambda_mix": lambda_mix,
            "eligible_originals": len(original_b2) if bidirectional else sum(
                source_prefix(path) == one_way_content for path in original_b2
            ),
            "fda_created": fda_created,
        },
        "augmentation_manifest": augmentation_manifest,
    }
    save_json(summary, report_dir / "scenario_build_summary.json")
    (report_dir / "scenario_build_summary.pretty.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print("METHOD B B0/B1/B2 SCENARIO BUILD")
    print("=" * 88)
    for scenario in SCENARIOS:
        print(f"{scenario} train: {counts[scenario]}")
    print(
        "[FDA POLICY] "
        f"bidirectional={bidirectional}, donor_resize={donor_resize}, "
        f"one_way={one_way_content}->{one_way_donor if not bidirectional else 'N/A'}"
    )
    print(f"[BALANCED] ratio={balanced_ratio}, counts={balanced}")
    print(f"[REPORT] {report_dir / 'scenario_build_summary.json'}")


if __name__ == "__main__":
    main()
