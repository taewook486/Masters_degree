#!/bin/bash
# =============================================================
# RunPod Phase 1: 범용 단일 모델 Zero-Shot 평가
# =============================================================
# Template: RunPod PyTorch 2.4+ (CUDA 12.x)
# GPU: RTX 4090 (24GB) 권장
# Disk: 50GB+ Container (모델 + 데이터셋 + 패키지)
#
# 사용법:
#   1. RunPod에서 Pod 생성 (PyTorch 2.4+ 템플릿, RTX 4090)
#   2. 터미널 접속 후:
#      git clone https://github.com/taewook486/Masters_degree.git
#      cd Masters_degree
#
#      # 특정 모델 평가
#      bash scripts/runpod_phase1.sh --config configs/models/gemma4_e2b.yaml
#
#      # 배치 크기 변경 (기본: 8)
#      bash scripts/runpod_phase1.sh --config configs/models/gemma4_e2b.yaml --batch_size 4
#
#      # 기존 결과 건너뛰기
#      bash scripts/runpod_phase1.sh --config configs/models/gemma4_e2b.yaml --skip_existing
#
#   3. 완료 후 결과 다운로드:
#      scp 또는 RunPod UI에서 results/phase1_baseline/*.json
# =============================================================

set -e

# --- 인자 파싱 ---
CONFIG=""
BATCH_SIZE=8
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --skip_existing)
            EXTRA_ARGS="$EXTRA_ARGS --skip_existing"
            shift
            ;;
        --max_samples)
            EXTRA_ARGS="$EXTRA_ARGS --max_samples $2"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "사용법: bash scripts/runpod_phase1.sh --config <model_config.yaml>"
    echo ""
    echo "사용 가능한 모델 config:"
    ls -1 configs/models/*.yaml 2>/dev/null || echo "  (없음)"
    exit 1
fi

MODEL_NAME=$(python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('$CONFIG')
print(c.model_name)
" 2>/dev/null || basename "$CONFIG" .yaml)

echo "=========================================="
echo " RunPod Phase 1: $MODEL_NAME"
echo "=========================================="

# 0. HF_TOKEN 확인
if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN 미설정. 다운로드 속도가 느릴 수 있습니다."
    echo "  설정: export HF_TOKEN=hf_xxxxx"
    echo ""
fi

# 1. 의존성 설치
echo "=========================================="
echo " Step 1: 의존성 설치"
echo "=========================================="
pip install -e . 2>&1 | tail -5

# 2. 환경 확인
echo ""
echo "=========================================="
echo " Step 2: 환경 확인"
echo "=========================================="
python -c "
import torch, transformers

print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'GPU:          {torch.cuda.get_device_name(0)}')
print(f'VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'Transformers: {transformers.__version__}')
"

# 3. 호환성 사전 검증 (추론 제외)
echo ""
echo "=========================================="
echo " Step 3: 호환성 사전 검증"
echo "=========================================="
python scripts/test_model_compat.py "$CONFIG" --skip-inference

# 4. Phase 1 평가 실행
echo ""
echo "=========================================="
echo " Step 4: Phase 1 평가 실행"
echo "=========================================="
echo "  모델: $MODEL_NAME"
echo "  조건: 3 데이터셋 x 3 시드 = 9개"
echo "  배치: $BATCH_SIZE"
echo ""

python -u scripts/run_phase1_single.py \
    --config "$CONFIG" \
    --output_dir results/phase1_baseline \
    --data_dir data \
    --batch_size "$BATCH_SIZE" \
    $EXTRA_ARGS

# 5. 결과 요약
echo ""
echo "=========================================="
echo " 결과 요약"
echo "=========================================="
python -c "
import json, glob
files = sorted(glob.glob('results/phase1_baseline/${MODEL_NAME}_*.json'))
if not files:
    print('  결과 파일 없음!')
else:
    print(f'  {\"Model\":<15} {\"Dataset\":<10} {\"Seed\":<6} {\"Closed\":<8} {\"Open\":<8} {\"Overall\":<8} {\"VRAM\":<8}')
    print('  ' + '-' * 63)
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        m, s = d['metadata'], d['summary']
        print(f'  {m[\"model_name\"]:<15} {m[\"dataset\"]:<10} {m[\"seed\"]:<6} {s[\"closed_accuracy\"]:<8.4f} {s[\"open_accuracy\"]:<8.4f} {s[\"overall_accuracy\"]:<8.4f} {s[\"peak_vram_mb\"]:<.0f}MB')
"
echo ""
echo "로컬로 결과 복사:"
echo "  scp runpod:/workspace/Masters_degree/results/phase1_baseline/${MODEL_NAME}_*.json ./results/phase1_baseline/"
