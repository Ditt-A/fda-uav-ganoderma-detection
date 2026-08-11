# Gemastik: Cross-Source Ganoderma Detection from UAV Imagery

This repository contains the reproducible source code, configuration, audited metadata, evaluation outputs, figures, and technical documentation for a single-class basal stem rot (BSR/Ganoderma) detector benchmark using two public UAV datasets.

The full experiment explanation is available in [reports/EXPERIMENT_TECHNICAL_REPORT.md](reports/EXPERIMENT_TECHNICAL_REPORT.md).

## Experiment status

- Dataset recovery, harmonization, group-aware splitting, tiling, and leakage audits: complete.
- YOLOv10-S, YOLO11-S, YOLOv12-S, and YOLOv13-S on B0/B1/B2: complete.
- RF-DETR-S: B0 is consolidated; B1/B2 are still running or pending in the current result snapshot.
- TinyFormer-S-PBM and DEIMv2-S: preflight validation passed; full B0/B1/B2 results are pending.

The checked-in `reports/benchmark/final_metrics.csv` is therefore a reproducible partial snapshot, not the final seven-model matrix.

## Dataset design

Two Roboflow sources are harmonized into one foreground class, `BSR`. Healthy and Non-BSR examples remain as background images.

- Source A: `m-rizky-pribadi/ganoderma-detection-oil-palm-afp9b`, configured version 6.
- Source B: `adnans-object-detection/oil-palm-bsr-disease-detection`, configured version 1.

The deterministic group-aware split is 70% train, 15% validation, and 15% test. The final audit reports zero group leakage, zero exact-image leakage, and zero invalid labels.

| Scenario | Training images | Definition |
|---|---:|---|
| B0 | 1,179 | Prepared original training data |
| B1 | 2,358 | B0 plus one offline H/V geometric copy per image |
| B2 | 3,537 | B1 plus one cross-source FDA copy per original image |

Validation and test images are byte-identical across all three scenarios. `B1 - B0` measures the geometric augmentation contribution, while `B2 - B1` measures the incremental FDA contribution.

## Repository contents

- `configs/`: acquisition, preprocessing, scenario, and model registry settings.
- `src/`: shared image, label, augmentation, and FDA utilities.
- `scripts/`: dataset reconstruction, auditing, training adapters, common COCO evaluation, and figure generation.
- `reports/`: technical report, dataset audits, environment manifests, consolidated metrics, and report-ready figures.
- `vendor_overrides/`: the small set of modified TinyFormer and DEIMv2 files/configurations used by the benchmark, stored separately from their upstream repositories.

Datasets, model weights, checkpoints, external repository clones, environments, archives, and raw training logs are intentionally excluded. They are large, reproducible/generated, or may have separate distribution terms.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

On Windows, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

RF-DETR additionally requires:

```bash
pip install -r requirements-rfdetr.txt
```

Create a private `.env` from `.env.example`; never commit API keys or access tokens.

## Rebuild and audit the dataset

Download the configured public datasets:

```bash
python scripts/00_download_roboflow.py --config configs/datasets.yaml
```

Run the complete Method B reconstruction and fail-loudly audit:

```bash
python scripts/rebuild_method_b.py
```

The active construction parameters are in `configs/method_b.yaml`. Audit evidence is written to `reports/method_b/`.

## External detector repositories

Official model implementations are not vendored in full. Clone the required upstream repositories under `external/`, then overlay the audited project-specific files where applicable.

TinyFormer:

```bash
git clone https://github.com/mmpmmpmmpjosh/TinyFormer.git external/TinyFormer
cp -a vendor_overrides/TinyFormer/. external/TinyFormer/
```

DEIMv2:

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git external/DEIMv2
cp -a vendor_overrides/DEIMv2/. external/DEIMv2/
```

The overlay directories include the respective upstream license files. Upstream commit identifiers and Python environment manifests are recorded in `reports/environments/`.

YOLOv10, YOLOv12, and YOLOv13 use their official external repositories. The setup/orchestration scripts record the expected paths and environment behaviour.

## Run the benchmark

The complete configured matrix is seven detector families by three scenarios:

```bash
python scripts/run_benchmark_auto.py \
  --mode full \
  --models yolov10s,yolo11s,yolov12s,yolov13s,rfdetrs,tinyformers,deimv2s \
  --scenarios B0,B1,B2 \
  --epochs 100 \
  --batch 32 \
  --patience 10 \
  --device 0 \
  --workers 2 \
  --rf-batch 32 \
  --rf-grad-accum 1 \
  --budget-mode epoch_matched
```

The main comparison is `primary_full / COMBINED / mAP50-95`. The `balanced_diagnostic` view retains all positive images while capping excess background-only images and should be used only as supporting diagnostic evidence.

## Generate report figures

```bash
python scripts/09_generate_report_figures.py
python scripts/11_generate_detector_figures.py
```

Generated publication figures are stored in `reports/figures/` as PNG and PDF.

## Current preliminary result

Across the four completed YOLO families, mean combined primary-full mAP50-95 increases from 0.2627 in B0 to 0.3244 in B1 and 0.3411 in B2. Geometric augmentation improves every YOLO family. The incremental FDA effect is architecture-dependent: it improves YOLOv10-S and YOLO11-S but reduces YOLOv12-S and YOLOv13-S relative to B1.

RF-DETR-S B0 currently has the strongest individual result at 0.4875 mAP50-95, but the overall detector and augmentation conclusions remain preliminary until all transformer runs finish.

## Security and artifact policy

- Do not commit `.env`, Roboflow keys, Hugging Face tokens, or other credentials.
- Do not commit raw/processed datasets, checkpoints, downloaded weights, external repository clones, or archived runs.
- Consolidated metrics, small audit manifests, environment manifests, technical documentation, and report figures are intended to be versioned.
