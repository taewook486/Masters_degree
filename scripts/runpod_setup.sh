#!/bin/bash
# =============================================================
# RunPod Phase 2 & 3 Setup Script
# =============================================================
# Template: CUDA 12.8 드라이버 + Python 3.12 (예: RunPod PyTorch 2.8/cu128)
#   설치는 uv.lock을 재현하므로 템플릿 torch 버전과 무관하게
#   검증 스택(transformers 5.5.0 + torch 2.10.0+cu128)이 그대로 깔린다.
# GPU: RTX 4090 (24GB) recommended
# Disk: 80GB+ Container, 50GB+ Volume
# =============================================================

set -e

echo "=========================================="
echo " RunPod Environment Setup"
echo "=========================================="

# 1. Clone repository
cd /workspace
if [ ! -d "Masters_degree" ]; then
    git clone https://github.com/taewook486/Masters_degree.git
    cd Masters_degree
else
    cd Masters_degree
    git pull
fi

# 2. Install dependencies via uv (uv.lock 재현: transformers 5.5.0 + torch 2.10.0+cu128)
#    [중요] pip install -e . 는 pyproject 하한(transformers>=4.45.0)만 보고
#    4.57.2를 잘못 설치해 Gemma4(Gemma4ForConditionalGeneration, 5.5.0 전용)가 로드 실패한다.
#    uv sync 는 uv.lock을 그대로 재현하므로 검증된 스택이 정확히 깔린다.
echo "Installing uv..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "Syncing dependencies from uv.lock..."
# 일부 RunPod 리전은 PyPI 대용량 wheel(torch/CUDA 계열, 수백MB) 다운로드가
# 기본 30초 타임아웃보다 느려 'Failed to download distribution due to network
# timeout' 로 실패한다. 넉넉히 늘려 재시도 없이 통과하도록 한다.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
uv sync --extra unsloth

# 이후 명령과 이후 터미널이 프로젝트 venv를 쓰도록 activate + bashrc 등록
source .venv/bin/activate
if ! grep -q 'Masters_degree/.venv/bin/activate' ~/.bashrc 2>/dev/null; then
    echo 'source /workspace/Masters_degree/.venv/bin/activate' >> ~/.bashrc
fi

# 3. Verify GPU + 버전 확인
echo ""
echo "=========================================="
echo " GPU Check"
echo "=========================================="
python -c "
import torch, transformers
print(f'PyTorch:      {torch.__version__}')      # 2.10.0+cu128 이어야 함
print(f'Transformers: {transformers.__version__}')  # 5.5.0 이어야 함 (Gemma4 필수)
print(f'CUDA:         {torch.version.cuda}')
print(f'GPU:          {torch.cuda.get_device_name(0)}')
print(f'VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 4. Download datasets (first run only)
echo ""
echo "=========================================="
echo " Dataset Check"
echo "=========================================="
python -m src.data.download

# 5. Download VQAv2 subset for CF measurement
python -c "
from src.data.general_vqa import download_vqav2_subset
download_vqav2_subset(save_dir='data', n_samples=2000, seed=42)
print('VQAv2 subset ready')
"

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "Run experiments with:"
echo "  Phase 2 Main:     bash scripts/run_phase2_main.sh"
echo "  Phase 2 Ablation: bash scripts/run_phase2_ablation.sh"
echo "  Phase 3 HPO:      bash scripts/run_phase3.sh"
echo ""
