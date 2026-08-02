@echo off
cd /d "D:\project\Masters_degree"
set PYTHONUNBUFFERED=1
set WANDB_PROJECT=medical-vqa-vlm
set WANDB_MODE=disabled

REM ============================================================
REM Phase 3: Autonomous HPO - 4 strategies x 10 repeats
REM
REM repeats MUST stay at 10 -- THESIS_FINAL_v2.0.md:225,322 fixes this at
REM 10 independent run-level observations per strategy for the Kruskal-
REM Wallis / Mann-Whitney U tests (v0.4 changed it 5->10 specifically for
REM statistical power ~0.6-0.7; do not lower it again).
REM max_test_samples 500: caps only the final per-trial test-set eval
REM (was ~1680 samples/trial), not train/val -- same cost lever the
REM thesis docs cite as safe (does not touch repeats/trials_per_repeat).
REM trials_per_repeat 40->20 (2026-08-01): halves total trials 1210->610
REM and wall-clock ~25d->~12.8d on 2 GPUs (--max_parallel 2). Trades off
REM per-run search breadth only -- repeats(=10) is untouched, so the
REM run-level statistical design/power is unaffected.
REM
REM --max_parallel 2: run_phase3.py launches 2 (strategy, repeat)
REM combinations at once, each in its own subprocess pinned to one
REM physical GPU (CUDA_VISIBLE_DEVICES=0/1) with its own HF_DATASETS_CACHE
REM (D:\cache\huggingface_datasets_gpu0/gpu1) -- both assigned automatically
REM per subprocess, so no manual GPU env vars needed here. Uses whatever
REM 2 GPUs are visible to this process; if only 1 GPU is present, drop
REM --max_parallel 2 (or set it to 1) to fall back to single-GPU sequential.
REM
REM time_budget_min 60->90 (2026-08-02): 2026-08-01 local dual-GPU smoke
REM measured the worst-case combo (rank=64, effective_batch=64) at 66.9min
REM (results/phase3_local_smoke_gpu1/random_repeat0/trial_0010), already
REM above the old 60min cap. Raised to 90min so max_steps=200 finishes
REM before the wall-clock safety net fires on every combo in the search
REM space, matching the decision recorded in THESIS_PROPOSAL_FINAL_v0.11.md.
REM IMPORTANT: Set ANTHROPIC_API_KEY for autoresearch strategy.
REM IMPORTANT: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID for phone alerts.
REM            Sends a Telegram alert on completion or failure (same pattern
REM            as run_phase3_smoke_gpu0/1.bat). If the window is force-closed
REM            or the process is killed, the alert code never runs - only
REM            normal exit or error exit is detected.
REM Replace --model_config with the best model from Phase 2.
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

if not exist results\phase3_autoresearch mkdir results\phase3_autoresearch

echo Starting Phase 3 HPO at %DATE% %TIME% >> results\phase3_autoresearch\run_phase3.log

.venv\Scripts\python.exe -u -m src.autoresearch.run_phase3 ^
  --model_config configs/models/qwen3_vl_2b.yaml ^
  --finetune_config configs/finetune/base_qlora.yaml ^
  --output_dir results/phase3_autoresearch ^
  --strategies manual random optuna autoresearch ^
  --repeats 10 ^
  --trials_per_repeat 20 ^
  --seed 42 ^
  --data_dir data ^
  --time_budget_min 90 ^
  --max_test_samples 500 ^
  --max_parallel 2 ^
  >> results\phase3_autoresearch\run_phase3.log 2>&1

set PHASE3_EXIT_CODE=%ERRORLEVEL%
echo Finished Phase 3 HPO at %DATE% %TIME% (exit=%PHASE3_EXIT_CODE%) >> results\phase3_autoresearch\run_phase3.log

if not "%TELEGRAM_BOT_TOKEN%"=="" (
    if %PHASE3_EXIT_CODE% EQU 0 (
        curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d text="[Phase3 Main] HPO run complete. Check results/phase3_autoresearch/results.tsv" >nul
    ) else (
        curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d text="[Phase3 Main] HPO run failed or aborted (exit=%PHASE3_EXIT_CODE%). Check results/phase3_autoresearch/run_phase3.log" >nul
    )
)
