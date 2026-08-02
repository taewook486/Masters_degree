#!/bin/bash
# =============================================================
# Phase 3: HPO Strategy Comparison (4 strategies x 10 repeats)
# THESIS v0.5 Section 4.5: 10 independent repeats per strategy
# (statistical power ~0.6-0.7, Mann-Whitney U test)
#
# v0.11: max_steps is now a fixed 200 for every trial (was previously drawn
# from {100,200,400,800} — see src/autoresearch/strategies.py
# PHASE3_FIXED_MAX_STEPS). --time_budget_min is now 90 (was 15) purely as a
# wall-clock safety cutoff, not the controlled variable. The old "~100-160h"
# estimate assumed the (buggy) 15-min-per-trial ceiling; using Phase 2 main's
# measured throughput (~13s/step at effective_batch=8, results/phase2_finetune
# run logs) as a rough baseline, a 200-step trial now needs ~35-45min at
# effective_batch=8 and proportionally more at higher effective_batch (up to
# 64 in the search space) -- so the real total is unverified and likely
# substantially higher than the old estimate. Run a short smoke trial first
# (1-2 trials per strategy) and re-derive the total from measured wall-clock
# before committing to the full 1,210-trial run.
#
# IMPORTANT: Set ANTHROPIC_API_KEY for autoresearch strategy.
#            Update --model_config with the best model from Phase 2.
# =============================================================

set -e
cd /workspace/Masters_degree

export PYTHONUNBUFFERED=1
export WANDB_PROJECT=medical-vqa-vlm
# run_phase2_main.sh/run_phase2_ablation.sh와 동일하게 캐시를 분산 배치 (누락 시
# 이 컨테이너는 $HOME=/workspace라 HF_HOME 기본값이 /workspace/.cache로 새어
# 볼륨 quota를 채우고 Disk quota exceeded로 죽는다 — Ablation에서 실제로 겪은 버그,
# 2026-07-22 커밋 2db2361에서 run_phase2_ablation.sh에 반영, 여기도 동일 적용)
export HF_HOME=/hf_cache
export MOAI_CHAT_CACHE_DIR=/hf_cache/chat_cache

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

# GPU 개수만큼 (strategy, repeat) job을 동시 배정 (run_phase3.py --max_parallel,
# 로컬 run_phase3.bat과 동일한 자동 감지 아이디어를 pod의 실제 GPU 수에 맞춰 적용)
MAX_PARALLEL=$(python -c "import torch; print(torch.cuda.device_count())")

echo "============================================================"
echo "Starting Phase 3 HPO at $(date)"
echo "Model: ${MODEL_CONFIG}"
echo "GPU count (max_parallel): ${MAX_PARALLEL}"
echo "============================================================"

python -u -m src.autoresearch.run_phase3 \
  --model_config "${MODEL_CONFIG}" \
  --finetune_config configs/finetune/base_qlora.yaml \
  --output_dir results/phase3_autoresearch \
  --strategies manual random optuna autoresearch \
  --repeats 10 \
  --trials_per_repeat 20 \
  --seed 42 \
  --data_dir data \
  --time_budget_min 90 \
  --max_test_samples 500 \
  --max_parallel "${MAX_PARALLEL}" \
  2>&1 | tee results/phase3_autoresearch/run_phase3.log

echo ""
echo "============================================================"
echo "Phase 3 HPO finished at $(date)"
echo "============================================================"
