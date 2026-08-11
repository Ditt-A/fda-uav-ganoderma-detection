#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-external/yolov13}"

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/iMoonLab/yolov13 "$REPO_DIR"
fi

echo
echo "YOLOv13 repository tersedia di: $REPO_DIR"
echo "Ikuti environment resmi repository karena FlashAttention/CUDA harus cocok."
echo
echo "Dataset:"
echo "  B0: $(pwd)/data/processed/scenarios/B0/data.yaml"
echo "  B1: $(pwd)/data/processed/scenarios/B1/data.yaml"
echo
echo "Catatan: perintah training upstream dapat berubah. Gunakan README versi commit yang Anda checkout,"
echo "catat commit hash, environment CUDA, dan pretrained checkpoint dalam technical report."
