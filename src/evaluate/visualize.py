"""논문용 시각화 모듈.

Phase 1 히트맵, Phase 2 비교 차트, Phase 3 HPO 궤적 등
논문 품질의 그래프를 생성한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 논문 품질 기본 설정
matplotlib.rcParams.update({
    "font.size": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_model_accuracy_heatmap(summary_df: pd.DataFrame, output_dir: str) -> str:
    """히트맵: 모델(행) x 데이터셋(열), 셀 = closed_acc.

    Args:
        summary_df: 'model', 'dataset', 'closed_acc' 컬럼을 포함하는 DataFrame.
        output_dir: 출력 디렉토리 경로.

    Returns:
        생성된 PNG 파일 경로.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pivot = summary_df.pivot(index="model", columns="dataset", values="closed_acc")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Model Accuracy Heatmap (Closed-ended)")
    ax.set_ylabel("Model")
    ax.set_xlabel("Dataset")

    out_path = str(Path(output_dir) / "phase1_heatmap")
    fig.savefig(out_path + ".png")
    fig.savefig(out_path + ".pdf")
    plt.close(fig)

    return out_path + ".png"


def plot_closed_vs_open(summary_df: pd.DataFrame, output_dir: str) -> str:
    """Grouped bar chart: closed vs open accuracy per model-dataset.

    Args:
        summary_df: 'model', 'dataset', 'closed_acc', 'open_acc'
            컬럼을 포함하는 DataFrame.
        output_dir: 출력 디렉토리 경로.

    Returns:
        생성된 PNG 파일 경로.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_melted = summary_df.melt(
        id_vars=["model", "dataset"],
        value_vars=["closed_acc", "open_acc"],
        var_name="metric",
        value_name="accuracy",
    )
    df_melted["label"] = df_melted["model"] + " / " + df_melted["dataset"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df_melted, x="label", y="accuracy", hue="metric", ax=ax)
    ax.set_title("Closed vs Open Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Metric")

    out_path = str(Path(output_dir) / "phase1_closed_vs_open")
    fig.savefig(out_path + ".png")
    fig.savefig(out_path + ".pdf")
    plt.close(fig)

    return out_path + ".png"


def plot_base_vs_finetuned(comparison_df: pd.DataFrame, output_dir: str) -> str:
    """Side-by-side bar: base vs fine-tuned accuracy per condition.

    Args:
        comparison_df: 'condition', 'base_acc', 'finetuned_acc'
            컬럼을 포함하는 DataFrame.
        output_dir: 출력 디렉토리 경로.

    Returns:
        생성된 PNG 파일 경로.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_melted = comparison_df.melt(
        id_vars=["condition"],
        value_vars=["base_acc", "finetuned_acc"],
        var_name="type",
        value_name="accuracy",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x="condition", y="accuracy", hue="type", ax=ax)
    ax.set_title("Base vs Fine-tuned Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Condition")
    ax.legend(title="Model Type")

    out_path = str(Path(output_dir) / "phase2_comparison")
    fig.savefig(out_path + ".png")
    fig.savefig(out_path + ".pdf")
    plt.close(fig)

    return out_path + ".png"


def plot_hpo_trajectories(trials_df: pd.DataFrame, output_dir: str) -> str:
    """Line chart: trial_number vs accuracy per HPO strategy.

    Args:
        trials_df: 'trial_number', 'accuracy', 'strategy' 컬럼을 포함하는 DataFrame.
        output_dir: 출력 디렉토리 경로.

    Returns:
        생성된 PNG 파일 경로.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy, group in trials_df.groupby("strategy"):
        ax.plot(group["trial_number"], group["accuracy"], marker="o", label=strategy)

    ax.set_title("HPO Strategy Trajectories")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Strategy")
    ax.grid(True, alpha=0.3)

    out_path = str(Path(output_dir) / "phase3_trajectories")
    fig.savefig(out_path + ".png")
    fig.savefig(out_path + ".pdf")
    plt.close(fig)

    return out_path + ".png"
