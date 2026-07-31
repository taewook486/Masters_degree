@echo off
cd /d "D:\project\Masters_degree"
set PYTHONUNBUFFERED=1
set WANDB_PROJECT=medical-vqa-vlm
set CUDA_VISIBLE_DEVICES=0

REM ============================================================
REM Phase 3 로컬 실측 스모크 테스트 - GPU0
REM
REM 목적: 듀얼 GPU 본 실행(repeats=10) trial 수를 정하기 전에,
REM       이 카드에서 전략당 1 trial(총 4개)만 돌려 실제 소요 시간을 잰다.
REM       추정치가 아니라 이 results.tsv의 train_time_min 실측값으로만
REM       본 실행 규모를 결정할 것.
REM
REM IMPORTANT: 실행 전 nvidia-smi로 CUDA_VISIBLE_DEVICES=0이 실제로
REM            어느 카드(5060 Ti/4060)인지 확인할 것.
REM IMPORTANT: Set ANTHROPIC_API_KEY for autoresearch strategy.
REM IMPORTANT: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID for phone alerts.
REM            완료/실패 시 텔레그램으로 알림을 보냄. 창을 강제로 닫거나
REM            프로세스를 강제종료(kill)한 경우는 알림 코드가 실행될 기회가
REM            없어 못 잡음 — 정상 종료/에러 종료만 감지 가능.
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
  --time_budget_min 60 ^
  >> results\phase3_local_smoke_gpu0\run_phase3.log 2>&1

set PHASE3_EXIT_CODE=%ERRORLEVEL%
echo Finished Phase 3 local smoke (GPU0) at %DATE% %TIME% (exit=%PHASE3_EXIT_CODE%) >> results\phase3_local_smoke_gpu0\run_phase3.log

if not "%TELEGRAM_BOT_TOKEN%"=="" (
    if %PHASE3_EXIT_CODE% EQU 0 (
        curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d text="[Phase3 GPU0] 스모크 테스트 완료. results/phase3_local_smoke_gpu0/results.tsv 확인." >nul
    ) else (
        curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d text="[Phase3 GPU0] 스모크 테스트 실패/중단 (exit=%PHASE3_EXIT_CODE%). results/phase3_local_smoke_gpu0/run_phase3.log 확인." >nul
    )
)
