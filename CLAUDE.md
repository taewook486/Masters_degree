# Medical VQA VLM Fine-Tuning Research Project

## Project Overview

Master's thesis research: "Domain Adaptation of Lightweight Vision-Language Models for Medical Visual Question Answering: QLoRA Fine-Tuning with Autonomous Hyperparameter Optimization"

- University: Konkuk University Graduate School (건국대학교 정보통신대학원)
- Department: Convergence Information Technology, AI Major
- Target submission: September 2026
- Language: Korean thesis, bilingual data (Korean + English)

## Hardware Constraints

- GPU: NVIDIA RTX 5060 Ti (16GB GDDR7 VRAM)
- CPU: AMD Ryzen 5 5600X
- RAM: 32GB
- All models and experiments MUST fit within 16GB VRAM budget
- Use QLoRA (4-bit NF4 quantization) for all fine-tuning

## Research Phases

1. **Phase 1 - Baseline**: Zero-shot evaluation of 4 lightweight VLMs on medical VQA datasets
2. **Phase 2 - Fine-tuning**: QLoRA fine-tuning with systematic hyperparameter analysis
3. **Phase 3 - AutoResearch**: Autonomous hyperparameter optimization loop

## Target Models

- Qwen3-VL-2B (Alibaba, 2B)
- Qwen2.5-VL-3B (Alibaba, 3B)
- Florence-2-large (Microsoft, 0.77B)
- SmolVLM-2.2B (HuggingFace, 2.2B)

## Datasets

- PathVQA: Pathology images, 5K images / 33K QA pairs
- SLAKE: Bilingual (EN+CN), 642 images / 14K QA pairs
- VQA-RAD: Radiology, 315 images / 2.2K QA pairs

## Key Technical Stack

- Python 3.11+
- PyTorch >= 2.1.0
- HuggingFace Transformers + PEFT + TRL
- BitsAndBytes (4-bit quantization)
- Unsloth (optional, for faster fine-tuning)
- Optuna (Bayesian HPO baseline)
- WandB (experiment tracking)

## Project Structure

```
configs/models/   - Model-specific YAML configs (4 models)
configs/finetune/ - QLoRA training configs + ablation
configs/autoresearch/ - HPO search configs
src/baseline/     - Zero-shot evaluation (model_loader, evaluate_zero_shot, run_all)
src/data/         - Dataset download and unified loading
src/finetune/     - QLoRA fine-tuning pipeline
src/autoresearch/ - Autonomous HPO loop
src/evaluate/     - VQA metrics (closed/open/overall accuracy)
src/utils/        - Shared utilities (seed, VRAM monitor)
results/          - Experiment outputs by phase
data/             - Dataset storage (gitignored)
thesis/           - Thesis LaTeX/document files
notebooks/        - Exploratory Jupyter notebooks
```

## Coding Conventions

- Language: Python with type hints
- Formatter: ruff (line-length 88)
- Config format: YAML (OmegaConf)
- Reproducibility: Always use `src/utils/seed.py` with seeds [42, 123, 456]
- VRAM monitoring: Use `src/utils/vram_monitor.py` for peak VRAM tracking
- All experiments must log to WandB

## Statistical Validation

- 3 seeds per experiment for reproducibility
- ANOVA + Tukey HSD for multi-group comparison
- Paired t-test + Cohen's d for before/after comparison
- Bootstrap 95% CI for robustness
- Significance level: alpha = 0.05
