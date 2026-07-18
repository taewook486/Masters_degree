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
# [2026-07-14] 당시엔 /workspace 볼륨이 50GB뿐이라 컨테이너 디스크(/hf_cache)로 뒀었으나,
# [2026-07-18] 실제 36조건 Main 완주 시 HF 모델(23G)+chat 캐시(21G)=44G가 컨테이너 디스크
# (보통 80GB)를 가득 채워 'No space left on device'로 30/36조건이 연쇄 실패했다(디스크
# 여유 있게 잡아도 컨테이너 쪽은 OS/베이스 이미지 자체가 이미 상당량을 차지해 여유가 작음).
# /workspace 볼륨은 보통 100GB+로 주문하고 실사용은 venv+data+results 합쳐 ~20GB뿐이라
# 훨씬 여유롭다 — 캐시를 여기로 옮긴다. 볼륨을 정말 작게(50GB 미만) 잡은 경우에만
# HF_HOME=/hf_cache로 되돌려야 한다.
export HF_HOME=/workspace/hf_cache
export MOAI_CHAT_CACHE_DIR=/workspace/hf_cache/chat_cache

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
