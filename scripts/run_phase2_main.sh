#!/bin/bash
# =============================================================
# Phase 2 Main: 4 models x 3 datasets x 3 seeds = 36 conditions
# Models (THESIS v0.5): Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B
# (configs/models/*.yaml 중 enabled:false 제외 → qwen25-vl-7b, _template 제외됨)
# Estimated: ~35-50 hours on RTX 4090
# =============================================================

set -e
cd /workspace/Masters_degree

export PYTHONUNBUFFERED=1
export WANDB_PROJECT=medical-vqa-vlm
export WANDB_MODE=offline  # 온라인 동기화 시도로 멈추지 않도록 (다일 실행 안정성)
export PYTORCH_ALLOC_CONF=expandable_segments:True  # GPU 메모리 파편화 완화 (다조건 반복)

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
  --max_eval_samples 500 \
  2>&1 | tee results/phase2_finetune/run_phase2.log

echo ""
echo "============================================================"
echo "Phase 2 Main finished at $(date)"
echo "============================================================"
