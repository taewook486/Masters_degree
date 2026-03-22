@echo off
cd /d "C:\project\Masters_degree"
set PYTHONUNBUFFERED=1
echo Starting Phase 1 evaluation at %DATE% %TIME% >> results\phase1_baseline\run_all.log
.venv\Scripts\python.exe -u -m src.baseline.run_all ^
  --config_dir configs/models ^
  --output_dir results/phase1_baseline ^
  --seeds 42 123 456 ^
  --data_dir data ^
  --batch_size 4 ^
  >> results\phase1_baseline\run_all.log 2>&1
echo Finished Phase 1 evaluation at %DATE% %TIME% >> results\phase1_baseline\run_all.log
