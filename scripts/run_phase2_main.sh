#!/bin/bash
# =============================================================
# Phase 2 Main: 3 models x 3 datasets x 3 seeds = 27 conditions
# Estimated: ~25-35 hours on RTX 4090
# =============================================================

set -e
cd /workspace/Masters_degree

export PYTHONUNBUFFERED=1
export WANDB_PROJECT=medical-vqa-vlm

mkdir -p results/phase2_finetune

echo "============================================================"
echo "Starting Phase 2 Main at $(date)"
echo "============================================================"

python -u -m src.finetune.run_phase2 \
  --config_dir configs/models \
  --finetune_config configs/finetune/base_qlora.yaml \
  --output_dir results/phase2_finetune \
  --seeds 42 123 456 \
  --data_dir data \
  2>&1 | tee results/phase2_finetune/run_phase2.log

echo ""
echo "============================================================"
echo "Phase 2 Main finished at $(date)"
echo "============================================================"
