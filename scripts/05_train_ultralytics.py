from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--project", default="runs/ultralytics")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    t = cfg.get("training", {})
    model = YOLO(args.model)

    model.train(
        data=args.dataset,
        epochs=int(t.get("epochs", 100)),
        imgsz=int(t.get("imgsz", 640)),
        batch=int(t.get("batch", 8)),
        patience=int(t.get("patience", 20)),
        device=t.get("device", "0"),
        workers=int(t.get("workers", 4)),
        seed=int(t.get("seed", 42)),
        project=args.project,
        name=args.name,
        exist_ok=False,
        # Dataset B1 sudah dibuat offline agar adil lintas framework.
        # Matikan augmentasi Ultralytics sebisa mungkin.
        augment=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
    )


if __name__ == "__main__":
    main()
