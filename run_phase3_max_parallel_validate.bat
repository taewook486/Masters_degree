@echo off
cd /d "D:\project\Masters_degree"
set PYTHONUNBUFFERED=1
set WANDB_PROJECT=medical-vqa-vlm
set WANDB_MODE=disabled

REM ============================================================
REM Phase 3: --max_parallel 2 real-GPU validation (short, disposable)
REM
REM Purpose: run_phase3.bat's dual-GPU dispatch path
REM (ThreadPoolExecutor + per-subprocess CUDA_VISIBLE_DEVICES +
REM per-GPU HF_DATASETS_CACHE, added 2026-08-01) has only been
REM verified at the logic/mock level -- never end-to-end on real
REM GPUs. The 2026-08-01 smoke instead used two SEPARATE single-GPU
REM scripts (run_phase3_smoke_gpu0.bat / gpu1.bat), which does NOT
REM exercise this dispatch code.
REM
REM This script runs the SMALLEST possible real case (repeats=1,
REM trials_per_repeat=1 -> 4 trials total, one per strategy) through
REM the real --max_parallel 2 path, writing to a throwaway
REM output_dir so it never touches results/phase3_autoresearch/
REM (the main run's results.tsv / trial_id sequence).
REM
REM What to check after this finishes:
REM   1) results\phase3_maxparallel_validate\results.tsv has 4 rows,
REM      status=completed (or a real error unrelated to GPU dispatch)
REM   2) No WinError5 / cache-path collision between the two GPU
REM      subprocesses (this exact bug hit the OLD single-GPU chat
REM      cache path -- confirm it does not resurface here)
REM   3) nvidia-smi during the run shows BOTH GPUs active
REM
REM If this passes, run_phase3.bat is clear to launch for real.
REM If it fails, do NOT launch run_phase3.bat -- fix the dispatch
REM bug first (the real ~12.8-day run is not the place to debug it).
REM ============================================================

if "%ANTHROPIC_API_KEY%"=="" (
    echo WARNING: ANTHROPIC_API_KEY not set. Autoresearch strategy will fall back to random.
    echo.
)

if not exist results\phase3_maxparallel_validate mkdir results\phase3_maxparallel_validate

echo Starting Phase3 max_parallel validation at %DATE% %TIME% >> results\phase3_maxparallel_validate\run_validate.log

.venv\Scripts\python.exe -u -m src.autoresearch.run_phase3 ^
  --model_config configs/models/qwen3_vl_2b.yaml ^
  --finetune_config configs/finetune/base_qlora.yaml ^
  --output_dir results/phase3_maxparallel_validate ^
  --strategies manual random optuna autoresearch ^
  --repeats 1 ^
  --trials_per_repeat 1 ^
  --seed 42 ^
  --data_dir data ^
  --time_budget_min 90 ^
  --max_test_samples 500 ^
  --max_parallel 2 ^
  >> results\phase3_maxparallel_validate\run_validate.log 2>&1

set VALIDATE_EXIT_CODE=%ERRORLEVEL%
echo Finished Phase3 max_parallel validation at %DATE% %TIME% (exit=%VALIDATE_EXIT_CODE%) >> results\phase3_maxparallel_validate\run_validate.log

echo.
echo Done. exit=%VALIDATE_EXIT_CODE%
echo Check results\phase3_maxparallel_validate\results.tsv and run_validate.log
