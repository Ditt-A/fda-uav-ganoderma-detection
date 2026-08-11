#!/usr/bin/env bash

set -euo pipefail

echo
echo "=============================================="
echo "STEP 00 - Download raw Roboflow datasets"
echo "=============================================="

python scripts/00_download_roboflow.py \
  --config configs/datasets.yaml


echo
echo "=============================================="
echo "STEP 00b - Recover augmented Roboflow sources"
echo "=============================================="

python scripts/00b_recover_roboflow_versions.py \
  --config configs/datasets.yaml


echo
echo "=============================================="
echo "STEP 01 - Audit effective datasets"
echo "=============================================="

python scripts/01_audit_sources.py \
  --config configs/datasets.yaml


echo
echo "=============================================="
echo "CLEAN OLD PROCESSED OUTPUTS"
echo "=============================================="

rm -rf data/processed/harmonized
rm -rf data/processed/tiled
rm -rf data/processed/scenarios
rm -rf reports/fda_preview


echo
echo "=============================================="
echo "STEP 02 - Harmonize classes"
echo "=============================================="

python scripts/02_harmonize_sources.py \
  --config configs/datasets.yaml \
  --output data/processed/harmonized


echo
echo "=============================================="
echo "STEP 03 - Tile dataset"
echo "=============================================="

python scripts/03_tile_dataset.py \
  --input data/processed/harmonized \
  --output data/processed/tiled \
  --config configs/experiment.yaml


echo
echo "=============================================="
echo "STEP 07 - Preview FDA"
echo "=============================================="

python scripts/07_preview_fda.py \
  --dataset data/processed/tiled \
  --config configs/experiment.yaml


echo
echo "=============================================="
echo "STEP 04 - Build B0 and B1 scenarios"
echo "=============================================="

python scripts/04_build_scenarios.py \
  --input data/processed/tiled \
  --output-root data/processed/scenarios \
  --config configs/experiment.yaml


echo
echo "=============================================="
echo "PIPELINE DATA SELESAI"
echo "=============================================="
echo
echo "Periksa:"
echo "  reports/recovery/source_a/recovery_summary.json"
echo "  reports/recovery/source_a/recovery_report.csv"
echo "  reports/recovery/source_a/review_sheets/"
echo "  reports/fda_preview/"
echo
echo "Dataset efektif:"
echo "  Source A: data/raw/source_a_recovered"
echo "  Source B: data/raw/source_b"
echo "  B0      : data/processed/scenarios/B0"
echo "  B1      : data/processed/scenarios/B1"