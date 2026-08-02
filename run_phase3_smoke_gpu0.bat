@echo off
cd /d "D:\project\Masters_degree"
set PYTHONUNBUFFERED=1
set WANDB_PROJECT=medical-vqa-vlm
set WANDB_MODE=disabled
set CUDA_VISIBLE_DEVICES=0
set HF_DATASETS_CACHE=D:\cache\huggingface_datasets_gpu0

REM ============================================================
REM Phase 3 local measured smoke test - GPU0
REM
REM Purpose: before deciding trial count for the dual-GPU main run
REM          [repeats=10], run 1 trial per strategy [4 total] on this
REM          card and measure actual wall-clock time. Do not estimate -
REM          use only the measured train_time_min from results.tsv to
REM          size the main run.
REM
REM IMPORTANT: before running, use nvidia-smi to confirm which card
REM            [5060 Ti or 4060] CUDA_VISIBLE_DEVICES=0 actually maps to.
REM time_budget_min 60->90 (2026-08-02): 2026-08-01 smoke on this same
REM script measured the worst-case combo at 66.9min, already above 60min.
REM IMPORTANT: Set ANTHROPIC_API_KEY for autoresearch strategy.
REM IMPORTANT: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID for phone alerts.
REM            Sends a Telegram alert on completion or failure. If the
REM            window is force-closed or the process is killed, the
REM            alert code never runs - only normal exit or error exit
REM            is detected.
REM ============================================================

if "%ANTHROPIC_API_KEY%"=="" (
    echo WARNING: ANTHROPIC_API_KEY not set. Autoresearch strategy will fall back to random.
    echo Set it with: set ANTHROPIC_API_KEY=sk-ant-...
    echo.
)

if "%TELEGRAM_BOT_TOKEN%"=="" (
    echo WARNING: TELEGRAM_BOT_TOKEN not set. Phone alerts will be skipped.
    echo Set it with: set TELEGRAM_BOT_TOKEN=... ^&^& set TELEGRAM_CHAT_ID=...
    echo.
)

if not exist results\phase3_local_smoke_gpu0 mkdir results\phase3_local_smoke_gpu0

echo Starting Phase 3 local smoke (GPU0) at %DATE% %TIME% >> results\phase3_local_smoke_gpu0\run_phase3.log

.venv\Scripts\python.exe -u -m src.autoresearch.run_phase3 ^
  --model_config configs/models/qwen3_vl_2b.yaml ^
  --finetune_config configs/finetune/base_qlora.yaml ^
  --output_dir results/phase3_local_smoke_gpu0 ^
  --strategies manual random optuna autoresearch ^
  --repeats 1 ^
  --trials_per_repeat 1 ^
  --seed 42 ^
  --data_dir data ^
  --time_budget_min 90 ^
  >> results\phase3_local_smoke_gpu0\run_phase3.log 2>&1

set PHASE3_EXIT_CODE=%ERRORLEVEL%
echo Finished Phase 3 local smoke (GPU0) at %DATE% %TIME% (exit=%PHASE3_EXIT_CODE%) >> results\phase3_local_smoke_gpu0\run_phase3.log

if not "%TELEGRAM_BOT_TOKEN%"=="" (
    if %PHASE3_EXIT_CODE% EQU 0 (
        curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d text="[Phase3 GPU0] Smoke test complete. Check results/phase3_local_smoke_gpu0/results.tsv" >nul
    ) else (
        curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d text="[Phase3 GPU0] Smoke test failed or aborted (exit=%PHASE3_EXIT_CODE%). Check results/phase3_local_smoke_gpu0/run_phase3.log" >nul
    )
)
