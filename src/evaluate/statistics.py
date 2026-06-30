"""논문 통계 분석 모듈.

Phase 1 (모델 비교), Phase 2 (Base vs Fine-tuned), Phase 3 (HPO 전략 비교)에
필요한 통계 검정 함수를 제공한다.
"""

from __future__ import annotations

import math

from scipy import stats


def run_anova_models(results: dict[str, list[float]]) -> dict:
    """One-way ANOVA + Tukey HSD post-hoc 검정.

    Args:
        results: {"model_name": [acc_seed42, acc_seed123, acc_seed456], ...}

    Returns:
        dict with: f_stat, p_value, tukey_pairs
        (list of dicts with group1, group2, meandiff, p_adj, reject)
    """
    groups = list(results.values())
    group_names = list(results.keys())

    f_stat, p_value = stats.f_oneway(*groups)

    # Tukey HSD post-hoc
    tukey_pairs: list[dict] = []
    try:
        import numpy as np
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        all_data = []
        all_labels = []
        for name, vals in results.items():
            all_data.extend(vals)
            all_labels.extend([name] * len(vals))

        tukey = pairwise_tukeyhsd(np.array(all_data), np.array(all_labels), alpha=0.05)
        for i in range(len(tukey.summary().data) - 1):  # skip header row
            row = tukey.summary().data[i + 1]
            tukey_pairs.append({
                "group1": str(row[0]),
                "group2": str(row[1]),
                "meandiff": float(row[2]),
                "p_adj": float(row[3]),
                "reject": bool(
                    row[4] if isinstance(row[4], bool)
                    else str(row[4]) == "True"
                ),
            })
    except ImportError:
        # statsmodels 미설치 시 Bonferroni 보정 사용
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                t_stat_pair, p_val_pair = stats.ttest_ind(
                    results[group_names[i]], results[group_names[j]]
                )
                p_adj = min(p_val_pair * n_comparisons, 1.0)
                mean_i = sum(results[group_names[i]]) / len(results[group_names[i]])
                mean_j = sum(results[group_names[j]]) / len(results[group_names[j]])
                tukey_pairs.append({
                    "group1": group_names[i],
                    "group2": group_names[j],
                    "meandiff": mean_j - mean_i,
                    "p_adj": p_adj,
                    "reject": p_adj < 0.05,
                })

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "tukey_pairs": tukey_pairs,
    }


def run_paired_ttest(before: list[float], after: list[float]) -> dict:
    """Paired t-test + Cohen's d (Phase 2: Base vs Fine-tuned).

    Args:
        before: base 모델 정확도 리스트.
        after: fine-tuned 모델 정확도 리스트.

    Returns:
        dict with: t_stat, p_value, cohens_d, significant (alpha=0.05)
    """
    t_stat, p_value = stats.ttest_rel(before, after)

    # Cohen's d for paired samples
    diffs = [a - b for a, b in zip(after, before)]
    mean_diff = sum(diffs) / len(diffs)
    sd_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1))
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": bool(p_value < 0.05),
    }


def run_wilcoxon(before: list[float], after: list[float]) -> dict:
    """Wilcoxon signed-rank test (비모수 Phase 2 검정).

    Args:
        before: base 모델 정확도 리스트.
        after: fine-tuned 모델 정확도 리스트.

    Returns:
        dict with: stat, p_value, significant
    """
    result = stats.wilcoxon(before, after)

    return {
        "stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
    }


def run_kruskal_wallis(groups: dict[str, list[float]]) -> dict:
    """Kruskal-Wallis H-test (Phase 3: HPO 전략 비교).

    Args:
        groups: {"strategy_name": [acc_trial1, acc_trial2, ...], ...}

    Returns:
        dict with: h_stat, p_value, significant
    """
    samples = list(groups.values())
    h_stat, p_value = stats.kruskal(*samples)

    return {
        "h_stat": float(h_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
