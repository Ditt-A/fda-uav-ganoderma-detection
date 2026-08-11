#!/usr/bin/env bash
set -euo pipefail
cd /home/nafisnaufal1426/adit/gemastik/ganoderma_uav_research_starter
echo "$$" > reports/benchmark/remaining_models_active.pid
/home/nafisnaufal1426/miniforge3/envs/soilcpu/bin/python scripts/run_benchmark_auto.py   --mode full   --models rfdetrs,tinyformers,deimv2s   --scenarios B0,B1,B2   --base-environment soilcpu   --temporary-environment ganoderma_bench_tmp   --epochs 100   --batch 32   --patience 10   --device 0   --workers 2   --rf-batch 32   --rf-grad-accum 1   --keep-environment   --budget-mode epoch_matched   --preflight-batch   --reuse-existing-temporary-environment
/opt/conda/bin/python scripts/09_generate_report_figures.py
/opt/conda/bin/python scripts/11_generate_detector_figures.py
touch reports/benchmark/remaining_models.complete
