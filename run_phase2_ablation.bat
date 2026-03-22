@echo off
cd /d "C:\project\Masters_degree"
set PYTHONUNBUFFERED=1
set WANDB_PROJECT=medical-vqa-vlm

REM ============================================================
REM Phase 2 Ablation Studies (A, B, C)
REM Run AFTER Phase 2 main + Phase 1 analysis to determine best model.
REM
REM IMPORTANT: Replace the --best_model_config path below with
REM the best-performing model from Phase 1/Phase 2 main results.
REM Example: configs/models/qwen25_vl_3b.yaml
REM ============================================================

echo Starting Phase 2 Ablation at %DATE% %TIME% >> results\phase2_finetune\run_phase2.log

.venv\Scripts\python.exe -u -m src.finetune.run_phase2 ^
  --config_dir configs/models ^
  --finetune_config configs/finetune/base_qlora.yaml ^
  --output_dir results/phase2_finetune ^
  --seeds 42 123 456 ^
  --data_dir data ^
  --best_model_config configs/models/qwen25_vl_3b.yaml ^
  --ablation all ^
  >> results\phase2_finetune\run_phase2.log 2>&1

echo Finished Phase 2 Ablation at %DATE% %TIME% >> results\phase2_finetune\run_phase2.log
