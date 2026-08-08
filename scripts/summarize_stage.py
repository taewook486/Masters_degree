import sys
import pandas as pd

stages = sys.argv[1:] if len(sys.argv) > 1 else ["optuna"]
tsv_path = "/workspace/Masters_degree/results/phase3_autoresearch/results.tsv"
df = pd.read_csv(tsv_path, sep="\t", on_bad_lines="skip")

def summarize(stage):
    sub = df[df["strategy"] == stage].copy()
    completed = sub[sub["status"] == "completed"]
    failed = sub[sub["status"] != "completed"]

    lines = []
    lines.append(f"=== {stage} stage summary ===")
    lines.append(f"total rows: {len(sub)}  completed: {len(completed)}  failed/other: {len(failed)}")

    if len(completed) > 0:
        best = completed.loc[completed["val_accuracy"].idxmax()]
        lines.append("")
        lines.append(f"best trial: trial_id={int(best['trial_id'])} repeat={int(best['repeat_id'])} val_accuracy={best['val_accuracy']:.4f}")
        lines.append(f"  lora_rank={best['lora_rank']} lora_alpha={best['lora_alpha']} lr={best['learning_rate']:.6g} "
                     f"batch_size={best['batch_size']} grad_accum={best['grad_accum_steps']} "
                     f"warmup_ratio={best['warmup_ratio']} weight_decay={best['weight_decay']} lora_targets={best['lora_targets']}")
        lines.append(f"  val_closed_acc={best['val_closed_acc']:.4f} val_open_acc={best['val_open_acc']:.4f} train_loss={best['train_loss']:.4f}")
        lines.append("")
        lines.append(f"val_accuracy: mean={completed['val_accuracy'].mean():.4f} std={completed['val_accuracy'].std():.4f} "
                     f"min={completed['val_accuracy'].min():.4f} max={completed['val_accuracy'].max():.4f}")
        lines.append(f"train_time_min: mean={completed['train_time_min'].mean():.1f} total={completed['train_time_min'].sum():.1f}")
        lines.append("")
        lines.append("per-repeat completed count:")
        for rid, cnt in completed.groupby("repeat_id").size().items():
            lines.append(f"  repeat {rid}: {cnt}")

    if len(failed) > 0:
        lines.append("")
        lines.append(f"failed/other rows ({len(failed)}):")
        for _, row in failed.iterrows():
            lines.append(f"  trial_id={row['trial_id']} status={row['status']}")

    return "\n".join(lines)

sections = [summarize(s) for s in stages]
out = ("\n\n").join(sections)
print(out)

if len(stages) == 1:
    out_name = f"{stages[0]}_summary.txt"
else:
    out_name = "phase3_summary.txt"
out_path = f"/workspace/Masters_degree/results/phase3_autoresearch/{out_name}"
with open(out_path, "w") as f:
    f.write(out + "\n")
print(f"\n[saved to {out_path}]")
