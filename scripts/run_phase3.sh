#!/bin/bash
# =============================================================
# Phase 3: HPO Strategy Comparison (4 strategies x 10 repeats)
# THESIS v0.5 Section 4.5: 10 independent repeats per strategy
# (statistical power ~0.6-0.7, Mann-Whitney U test)
# Estimated: ~100-160 hours on RTX 4090
#
# IMPORTANT: Set ANTHROPIC_API_KEY for autoresearch strategy.
#            Update --model_config with the best model from Phase 2.
# =============================================================

set -e
cd /workspace/Masters_degree

export PYTHONUNBUFFERED=1
export WANDB_PROJECT=medical-vqa-vlm

# --- REQUIRED: Set your Anthropic API key ---
# Preflight: if 'autoresearch' is among the strategies, verify the API works.
# Without this, a bad key silently degrades autoresearch to random search,
# invalidating the core HPO comparison (strategies.py fallback).
STRATEGIES="manual random optuna autoresearch"
if echo "${STRATEGIES}" | grep -qw autoresearch; then
    if [ -z "${ANTHROPIC_API_KEY}" ]; then
        echo "ABORT: autoresearch requested but ANTHROPIC_API_KEY is not set."
        echo "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
        exit 1
    fi
    echo "[preflight] Verifying Anthropic API (autoresearch)..."
    python -c "
import anthropic
c = anthropic.Anthropic()
c.messages.create(model='claude-sonnet-4-6', max_tokens=8,
                  messages=[{'role': 'user', 'content': 'ping'}])
print('[preflight] Anthropic API OK')
" || { echo "ABORT: Anthropic API preflight failed. Fix the key/quota before running autoresearch (else it silently falls back to random)."; exit 1; }
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
  --repeats 10 \
  --trials_per_repeat 40 \
  --seed 42 \
  --data_dir data \
  --time_budget_min 15 \
  2>&1 | tee results/phase3_autoresearch/run_phase3.log

echo ""
echo "============================================================"
echo "Phase 3 HPO finished at $(date)"
echo "============================================================"
