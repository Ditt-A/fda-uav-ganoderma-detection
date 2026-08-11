from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_yaml


def build_model(size: str):
    try:
        from rfdetr import (
            RFDETRNano,
            RFDETRSmall,
            RFDETRMedium,
            RFDETRLarge,
        )
    except ImportError as exc:
        raise SystemExit(
            'RF-DETR belum terpasang. Jalankan: pip install "rfdetr[train,loggers]"'
        ) from exc

    mapping = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    if size not in mapping:
        raise ValueError(f"Ukuran tidak didukung: {size}. Pilih {list(mapping)}")
    return mapping[size]()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    t = cfg.get("training", {})
    output_dir = Path("runs/rfdetr") / args.name
    output_dir.mkdir(parents=True, exist_ok=False)

    model = build_model(args.size)
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=str(output_dir),
        epochs=int(t.get("epochs", 100)),
        batch_size=int(t.get("batch", 8)),
        grad_accum_steps=1,
        lr=1e-4,
        device=t.get("device", "cuda"),
        num_workers=int(t.get("workers", 4)),
        early_stopping=True,
        early_stopping_patience=int(t.get("patience", 20)),
    )


if __name__ == "__main__":
    main()
