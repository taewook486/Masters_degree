@echo off
cd /d "C:\project\Masters_degree"
set PYTHONUNBUFFERED=1
set WANDB_PROJECT=medical-vqa-vlm

REM ============================================================
REM Phase 3: Autonomous HPO (4 strategies x 5 repeats)
REM
REM IMPORTANT: Set ANTHROPIC_API_KEY for autoresearch strategy.
REM Replace --model_config with the best model from Phase 2.
REM ============================================================

if "%ANTHROPIC_API_KEY%"=="" (
    echo WARNING: ANTHROPIC_API_KEY not set. Autoresearch strategy will fall back to random.
    echo Set it with: set ANTHROPIC_API_KEY=sk-ant-...
    echo.
)

echo Starting Phase 3 HPO at %DATE% %TIME% >> results\phase3_autoresearch\run_phase3.log

.venv\Scripts\python.exe -u -m src.autoresearch.run_phase3 ^
  --model_config configs/models/qwen25_vl_3b.yaml ^
  --finetune_config configs/finetune/base_qlora.yaml ^
  --output_dir results/phase3_autoresearch ^
  --strategies manual random optuna autoresearch ^
  --repeats 5 ^
  --trials_per_repeat 40 ^
  --seed 42 ^
  --data_dir data ^
  --time_budget_min 15 ^
  >> results\phase3_autoresearch\run_phase3.log 2>&1

echo Finished Phase 3 HPO at %DATE% %TIME% >> results\phase3_autoresearch\run_phase3.log
