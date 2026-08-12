"""논문용 시각화 모듈.

Phase 1 히트맵, Phase 2 비교 차트, Phase 3 HPO 궤적 등
논문 품질의 그래프를 생성한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
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


def plot_anytime_performance(
    curve_df: pd.DataFrame,
    output_dir: str,
    filename: str = "phase3_anytime",
) -> str:
    """Anytime performance 곡선: trial 진행에 따른 누적 최고 val_accuracy.

    `plot_hpo_trajectories`와의 차이: 저 함수는 trial별 '원본' 정확도를
    그대로 그리지만, 이 함수는 (1) 각 run의 '누적 최고'를 그리고
    (2) 독립 반복 10회를 중앙값 + IQR 밴드로 집계한다. 논문 §4.3.3의
    탐색 효율 비교용 그림이다.

    평균이 아니라 중앙값 + IQR을 쓰는 이유: 본 연구의 run-level 검정이
    비모수 검정(Kruskal-Wallis / Mann-Whitney)이므로 시각화도 같은
    분포 가정을 따르게 맞춘 것이다.

    Args:
        curve_df: 'strategy', 'trial_index', 'median', 'q1', 'q3', 'n'
            컬럼을 포함하는 DataFrame. `scripts/plot_phase3_anytime.py`가 생성한다.
        output_dir: 출력 디렉토리 경로.
        filename: 확장자를 제외한 출력 파일명.

    Returns:
        생성된 PNG 파일 경로.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("colorblind", n_colors=curve_df["strategy"].nunique())
    x_max = int(curve_df["trial_index"].max())

    for color, (strategy, group) in zip(
        palette, curve_df.groupby("strategy", sort=True)
    ):
        group = group.sort_values("trial_index")

        # Manual처럼 반복당 trial이 1개뿐인 전략은 곡선이 점 하나로만 찍혀
        # 비교가 안 된다. 전 구간에 걸친 수평 기준선으로 그린다.
        if len(group) == 1:
            baseline = float(group["median"].iloc[0])
            ax.axhline(
                baseline,
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"{strategy} (baseline, n={int(group['n'].iloc[0])})",
            )
            continue

        ax.plot(
            group["trial_index"],
            group["median"],
            marker="o",
            markersize=4,
            color=color,
            label=f"{strategy} (median of {int(group['n'].max())} runs)",
        )
        ax.fill_between(
            group["trial_index"], group["q1"], group["q3"], color=color, alpha=0.15
        )

    # 일부 반복이 아직 해당 trial 수에 도달하지 못한 구간을 표시한다.
    # (전 반복이 완주한 뒤에는 이 선이 그려지지 않는다.)
    multi = curve_df[curve_df.groupby("strategy")["trial_index"].transform("max") > 1]
    if not multi.empty:
        full_n = multi.groupby("strategy")["n"].transform("max")
        incomplete = multi[multi["n"] < full_n]
        if not incomplete.empty:
            cutoff = int(incomplete["trial_index"].min())
            ax.axvline(cutoff - 0.5, color="grey", linestyle=":", linewidth=1.5)
            # 그림 라벨은 기존 Phase 1/2 그림과 동일하게 영문으로 통일한다
            # (한글 글리프가 matplotlib 기본 폰트에 없어 PDF에서 깨짐).
            # x는 데이터 좌표, y는 축 비율(0~1)로 잡아 축 하단에 잘리지 않게 띄운다
            ax.text(
                cutoff - 0.4,
                0.04,
                " incomplete runs beyond this point",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
                color="grey",
                va="bottom",
            )

    ax.set_title("Anytime Performance by HPO Strategy")
    ax.set_xlabel("Trial Index (within a repeat)")
    ax.set_ylabel("Cumulative Best val_accuracy")
    ax.set_xlim(0.5, x_max + 0.5)
    # trial 번호는 정수이므로 눈금도 정수로만 찍는다 (기본값은 2.5, 7.5처럼 소수로 찍힘)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # 곡선이 좌하 -> 우상으로 상승하므로 좌상단이 비어 있다.
    # 하단 미완료 주석과 겹치지 않도록 범례를 그쪽에 둔다.
    ax.legend(title="Strategy", loc="upper left")
    ax.grid(True, alpha=0.3)

    out_path = str(Path(output_dir) / filename)
    fig.savefig(out_path + ".png")
    fig.savefig(out_path + ".pdf")
    plt.close(fig)

    return out_path + ".png"
