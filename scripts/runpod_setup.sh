#!/bin/bash
# =============================================================
# RunPod Phase 2 & 3 Setup Script
# =============================================================
# Template: RunPod PyTorch 2.4+ (CUDA 12.x)
# GPU: RTX 4090 (24GB) recommended (required for 7B QLoRA)
# Disk: 80GB+ Container, 50GB+ Volume (7B model ~15GB)
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

# 2. Install dependencies
echo "Installing dependencies..."
pip install -e ".[unsloth]" 2>&1 | tail -5
pip install flash-attn --no-build-isolation 2>&1 | tail -3

# 3. Verify GPU
echo ""
echo "=========================================="
echo " GPU Check"
echo "=========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.version.cuda}')
print(f'GPU:     {torch.cuda.get_device_name(0)}')
print(f'VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

# 4. Download datasets (first run only)
echo ""
echo "=========================================="
echo " Dataset Check"
echo "=========================================="
python -c "
from src.data.medical_datasets import load_medical_vqa
for ds in ['pathvqa', 'slake', 'vqa_rad']:
    samples = load_medical_vqa(ds, split='train', data_dir='data')
    print(f'{ds} train: {len(samples)} samples')
"

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
