from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import iter_images, load_yaml, read_image, seed_everything
from src.fda import fourier_amplitude_mix


def prefix(path: Path) -> str:
    return path.name.split("__", 1)[0] if "__" in path.name else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/tiled")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", default="reports/fda_preview")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    fda_cfg = cfg["fda"]

    paths = list(iter_images(Path(args.dataset) / "train" / "images"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.count, len(paths))):
        content_path = random.choice(paths)
        donors = [p for p in paths if prefix(p) != prefix(content_path)]
        if not donors:
            break
        donor_path = random.choice(donors)
        content = read_image(content_path)
        donor = read_image(donor_path)
        mixed = fourier_amplitude_mix(
            content,
            donor,
            beta=float(fda_cfg.get("beta", 0.01)),
            lambda_mix=float(fda_cfg.get("lambda_mix", 0.35)),
        )

        donor_resized = cv2.resize(donor, (content.shape[1], content.shape[0]))
        panel = cv2.hconcat([content, donor_resized, mixed])
        cv2.imwrite(str(output / f"preview_{i:03d}.jpg"), panel)

    print(f"[OK] Preview tersimpan di {output}")


if __name__ == "__main__":
    main()
