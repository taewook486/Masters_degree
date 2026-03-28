#!/bin/bash
# =============================================================
# Phase 2 Ablation: A (data%), B (rank), C (targets)
# Estimated: ~40-60 hours on RTX 4090
#
# NOTE: --best_model_config should be the best model from Phase 2 Main.
#       Default: qwen3-vl-2b (best in Phase 1). Update if needed.
# =============================================================

set -e
cd /workspace/Masters_degree

export PYTHONUNBUFFERED=1
export WANDB_PROJECT=medical-vqa-vlm

# --- UPDATE THIS after Phase 2 Main results ---
BEST_MODEL_CONFIG="configs/models/qwen3_vl_2b.yaml"
# ------------------------------------------------

echo "============================================================"
echo "Starting Phase 2 Ablation at $(date)"
echo "Best model config: ${BEST_MODEL_CONFIG}"
echo "============================================================"

python -u -m src.finetune.run_phase2 \
  --config_dir configs/models \
  --finetune_config configs/finetune/base_qlora.yaml \
  --output_dir results/phase2_finetune \
  --seeds 42 123 456 \
  --data_dir data \
  --best_model_config "${BEST_MODEL_CONFIG}" \
  --ablation all \
  2>&1 | tee -a results/phase2_finetune/run_phase2.log

echo ""
echo "============================================================"
echo "Phase 2 Ablation finished at $(date)"
echo "============================================================"
