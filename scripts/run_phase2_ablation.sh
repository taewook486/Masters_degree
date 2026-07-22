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
export WANDB_MODE=offline  # 온라인 동기화 시도로 멈추지 않도록 (다일 실행 안정성)
export PYTORCH_ALLOC_CONF=expandable_segments:True  # GPU 메모리 파편화 완화 (다조건 반복)
# run_phase2_main.sh와 동일하게 캐시를 분산 배치 (누락 시 HF_HOME 기본값인
# $HOME/.cache/huggingface로 저장되는데 이 컨테이너는 $HOME=/workspace라
# /workspace 볼륨 quota를 채워 'Disk quota exceeded'로 이어짐 — 2026-07-21 실제 발생).
# chat_cache 경로는 Main과 동일하게 둬서 이미 빌드된 캐시(pathvqa 등)를 재사용한다.
export HF_HOME=/hf_cache
export MOAI_CHAT_CACHE_DIR=/workspace/hf_cache/chat_cache

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
  --max_eval_samples 500 \
  2>&1 | tee -a results/phase2_finetune/run_phase2.log

echo ""
echo "============================================================"
echo "Phase 2 Ablation finished at $(date)"
echo "============================================================"
