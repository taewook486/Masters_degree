@echo off
cd /d "C:\project\Masters_degree"
set PYTHONUNBUFFERED=1
set WANDB_PROJECT=medical-vqa-vlm

echo ============================================================ >> results\phase2_finetune\run_phase2.log
echo Starting Phase 2 QLoRA fine-tuning at %DATE% %TIME% >> results\phase2_finetune\run_phase2.log
echo ============================================================ >> results\phase2_finetune\run_phase2.log

REM Phase 2 Main: 3 models x 3 datasets x 3 seeds = 27 conditions
REM Then Ablation A/B/C on best model (specify --best_model_config after Phase 1)
.venv\Scripts\python.exe -u -m src.finetune.run_phase2 ^
  --config_dir configs/models ^
  --finetune_config configs/finetune/base_qlora.yaml ^
  --output_dir results/phase2_finetune ^
  --seeds 42 123 456 ^
  --data_dir data ^
  >> results\phase2_finetune\run_phase2.log 2>&1

echo Finished Phase 2 main conditions at %DATE% %TIME% >> results\phase2_finetune\run_phase2.log
echo. >> results\phase2_finetune\run_phase2.log
