$ErrorActionPreference = "Stop"

python scripts/00_download_roboflow.py --config configs/datasets.yaml
python scripts/01_audit_sources.py --config configs/datasets.yaml
python scripts/02_harmonize_sources.py --config configs/datasets.yaml --output data/processed/harmonized
python scripts/03_tile_dataset.py --input data/processed/harmonized --output data/processed/tiled --config configs/experiment.yaml
python scripts/07_preview_fda.py --dataset data/processed/tiled --config configs/experiment.yaml
python scripts/04_build_scenarios.py --input data/processed/tiled --output-root data/processed/scenarios --config configs/experiment.yaml

Write-Host "Pipeline data selesai. Periksa reports/ dan sampel FDA sebelum training."
