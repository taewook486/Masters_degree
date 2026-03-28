#!/bin/bash
# =============================================================
# Phase 3: HPO Strategy Comparison (4 strategies x 5 repeats)
# Estimated: ~50-80 hours on RTX 4090
#
# IMPORTANT: Set ANTHROPIC_API_KEY for autoresearch strategy.
#            Update --model_config with the best model from Phase 2.
# =============================================================

set -e
cd /workspace/Masters_degree

export PYTHONUNBUFFERED=1
export WANDB_PROJECT=medical-vqa-vlm

# --- REQUIRED: Set your Anthropic API key ---
if [ -z "${ANTHROPIC_API_KEY}" ]; then
    echo "WARNING: ANTHROPIC_API_KEY not set."
    echo "Autoresearch strategy will fail without it."
    echo "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
    echo ""
fi

# --- UPDATE THIS after Phase 2 results ---
MODEL_CONFIG="configs/models/qwen3_vl_2b.yaml"
# -------------------------------------------

mkdir -p results/phase3_autoresearch

echo "============================================================"
echo "Starting Phase 3 HPO at $(date)"
echo "Model: ${MODEL_CONFIG}"
echo "============================================================"

python -u -m src.autoresearch.run_phase3 \
  --model_config "${MODEL_CONFIG}" \
  --finetune_config configs/finetune/base_qlora.yaml \
  --output_dir results/phase3_autoresearch \
  --strategies manual random optuna autoresearch \
  --repeats 5 \
  --trials_per_repeat 40 \
  --seed 42 \
  --data_dir data \
  --time_budget_min 15 \
  2>&1 | tee results/phase3_autoresearch/run_phase3.log

echo ""
echo "============================================================"
echo "Phase 3 HPO finished at $(date)"
echo "============================================================"
