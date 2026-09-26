"""Quick script to check Phase 1 evaluation progress."""
import json
from pathlib import Path

results_dir = Path("results/phase1_baseline")
json_files = sorted(results_dir.glob("*.json"))

completed = []
for f in json_files:
    with open(f) as fp:
        d = json.load(fp)
    meta = d.get("metadata", {})
    summary = d.get("summary", {})
    n = meta.get("num_samples", 0)
    vram = summary.get("peak_vram_mb", 0)
    gpu_run = vram > 0
    status = "GPU" if gpu_run else "CPU"
    full = "FULL" if n > 10 else f"debug({n})"
    print(f"  [{status}/{full}] {f.stem}: overall={summary.get('overall_accuracy', 0):.3f}")
    if gpu_run and n > 10:
        completed.append(f.stem)

print(f"\nCompleted (GPU, full): {len(completed)}/27")
print(f"Remaining: {27 - len(completed)}")

# Check if phase1_summary.csv exists
csv_path = results_dir / "phase1_summary.csv"
if csv_path.exists():
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"\n=== Phase 1 Summary ===")
    print(df.to_string(index=False))
