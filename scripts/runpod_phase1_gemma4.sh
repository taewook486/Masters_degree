#!/bin/bash
# =============================================================
# RunPod Phase 1: Gemma 4 E2B Zero-Shot Evaluation
# =============================================================
# Template: RunPod PyTorch 2.4+ (CUDA 12.x)
# GPU: RTX 4090 (24GB) 권장 — Gemma4-E2B peak VRAM ~10.3GB
# Disk: 50GB+ Container (모델 ~10GB + 데이터셋 + 패키지)
# 예상 소요: ~5-6시간 (4090 기준)
# 비용: ~$2.5 (RTX 4090 $0.44/hr × 6hr)
# =============================================================
#
# 사용법:
#   1. RunPod에서 Pod 생성 (PyTorch 2.4+ 템플릿, RTX 4090)
#   2. 터미널 접속 후:
#      git clone https://github.com/taewook486/Masters_degree.git
#      cd Masters_degree
#      bash scripts/runpod_phase1_gemma4.sh
#   3. 완료 후 결과 다운로드:
#      scp 또는 RunPod UI에서 results/phase1_baseline/gemma4-e2b_*.json
# =============================================================

set -e

echo "=========================================="
echo " RunPod Phase 1: Gemma 4 E2B Evaluation"
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

# 2. transformers 버전 확인 (Gemma 4 지원에 5.5.0+ 필요)
echo ""
echo "=========================================="
echo " Step 2: 환경 확인"
echo "=========================================="
python -c "
import torch, transformers

print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'GPU:          {torch.cuda.get_device_name(0)}')
print(f'VRAM:         {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
print(f'Transformers: {transformers.__version__}')

if not hasattr(transformers, 'Gemma4ForConditionalGeneration'):
    print('Gemma4 미지원 -> transformers 업그레이드 필요')
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'transformers>=5.5.0'])
    print('업그레이드 완료')
else:
    print('Gemma4 지원: OK')
"

# 3. Phase 1 평가 실행 (모델 1회 로드, 9개 조건 순차 실행)
echo ""
echo "=========================================="
echo " Step 3: Phase 1 평가 실행"
echo "=========================================="
echo "  모델: Gemma4-E2B (모델 1회 로드 최적화)"
echo "  조건: 3 데이터셋 x 3 시드 = 9개"
echo "  배치: 8 (RunPod VRAM 여유 활용)"
echo ""

python -u scripts/run_phase1_gemma4.py \
    --output_dir results/phase1_baseline \
    --data_dir data \
    --batch_size 8

# 4. 결과 요약
echo ""
echo "=========================================="
echo " 결과 요약"
echo "=========================================="
python -c "
import json, glob
files = sorted(glob.glob('results/phase1_baseline/gemma4-e2b_*.json'))
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
echo "  scp runpod:/workspace/Masters_degree/results/phase1_baseline/gemma4-e2b_*.json ./results/phase1_baseline/"
