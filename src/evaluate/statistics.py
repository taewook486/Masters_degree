"""논문 통계 분석 모듈.

Phase 1 (모델 비교), Phase 2 (Base vs Fine-tuned), Phase 3 (HPO 전략 비교)에
필요한 통계 검정 함수를 제공한다.
"""

from __future__ import annotations

import math

from scipy import stats


def bootstrap_accuracy_ci(
    correctness: list[int] | list[float],
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """정확도(0/1 정오 벡터 평균)의 퍼센타일 부트스트랩 신뢰구간.

    zero-shot greedy 평가는 결정적이므로 시드 재실행으로는 분산이 생기지 않는다.
    결정적 평가의 올바른 불확실성 정량화는 '테스트셋 표본 변동'이며, 이를
    per-sample 정오 벡터의 부트스트랩 CI로 보고한다.

    Args:
        correctness: 샘플별 정오 (1=정답, 0=오답).
        confidence_level: 신뢰수준 (기본 0.95).
        n_resamples: 부트스트랩 재표본 수.
        seed: 재현성을 위한 난수 시드.

    Returns:
        (ci_low, ci_high) 튜플. 표본이 비었으면 (0.0, 0.0),
        전부 동일(정답 전부/오답 전부)하면 (point, point).
    """
    import numpy as np

    arr = np.asarray(correctness, dtype=float)
    n = arr.size
    if n == 0:
        return (0.0, 0.0)
    point = float(arr.mean())
    # 축퇴 케이스: 전부 0 또는 전부 1이면 부트스트랩 분산이 0 → 점추정 반환
    if float(arr.min()) == float(arr.max()):
        return (round(point, 4), round(point, 4))

    res = stats.bootstrap(
        (arr,),
        np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method="percentile",
        random_state=seed,
    )
    return (
        round(float(res.confidence_interval.low), 4),
        round(float(res.confidence_interval.high), 4),
    )


def run_cochran_q(model_correctness: dict[str, list[int]]) -> dict:
    """Cochran's Q 검정: 동일 테스트셋에서 k개 모델의 이진 정오 비교 (RQ1).

    결정적 zero-shot 평가에서는 모든 모델이 '같은 샘플'로 평가되므로,
    시드-분산 ANOVA가 아니라 반복측정 짝지은 검정(Cochran's Q)이 적절하다.
    H0: 모든 모델의 정확도가 동일하다.

    Args:
        model_correctness: {"model": [샘플별 0/1 정오], ...}.
            모든 모델의 리스트 길이와 샘플 순서가 동일해야 한다.

    Returns:
        dict: q_stat, p_value, df, k, significant (alpha=0.05)
    """
    import numpy as np

    names = list(model_correctness.keys())
    mat = np.asarray([model_correctness[n] for n in names], dtype=int).T  # (n_samples, k)
    k = mat.shape[1]

    try:
        from statsmodels.stats.contingency_tables import cochrans_q

        res = cochrans_q(mat)
        q_stat, p_value = float(res.statistic), float(res.pvalue)
    except ImportError:
        # statsmodels 미설치 시 수동 계산 + chi2 근사
        col = mat.sum(axis=0)  # 모델별 정답 수
        row = mat.sum(axis=1)  # 샘플별 정답 모델 수
        n_total = int(mat.sum())
        denom = k * n_total - int((row ** 2).sum())
        q_stat = (
            (k - 1) * (k * int((col ** 2).sum()) - n_total ** 2) / denom
            if denom != 0 else 0.0
        )
        p_value = float(1 - stats.chi2.cdf(q_stat, k - 1))

    return {
        "q_stat": float(q_stat),
        "p_value": float(p_value),
        "df": k - 1,
        "k": k,
        "significant": bool(p_value < 0.05),
    }


def run_mcnemar(correct_a: list[int], correct_b: list[int]) -> dict:
    """McNemar 검정: 동일 테스트셋에서 두 모델의 짝지은 이진 정오 비교.

    Cochran's Q가 유의할 때 쌍별 post-hoc으로 사용한다.
    불일치 샘플 수가 작으면(<25) 정확검정(binomial), 크면 chi2 근사.

    Args:
        correct_a: 모델 A의 샘플별 0/1 정오.
        correct_b: 모델 B의 샘플별 0/1 정오 (A와 같은 순서).

    Returns:
        dict: stat, p_value, n_discordant, b01, b10, significant
    """
    import numpy as np

    a = np.asarray(correct_a, dtype=int)
    b = np.asarray(correct_b, dtype=int)
    b01 = int(((a == 1) & (b == 0)).sum())  # A만 정답
    b10 = int(((a == 0) & (b == 1)).sum())  # B만 정답
    n_disc = b01 + b10

    try:
        from statsmodels.stats.contingency_tables import mcnemar

        table = [
            [int(((a == 1) & (b == 1)).sum()), b01],
            [b10, int(((a == 0) & (b == 0)).sum())],
        ]
        res = mcnemar(table, exact=n_disc < 25)
        stat, p_value = float(res.statistic), float(res.pvalue)
    except ImportError:
        if n_disc == 0:
            stat, p_value = 0.0, 1.0
        else:
            stat = float(min(b01, b10))
            p_value = float(stats.binomtest(min(b01, b10), n_disc, 0.5).pvalue)

    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "n_discordant": n_disc,
        "b01": b01,
        "b10": b10,
        "significant": bool(p_value < 0.05),
    }


def run_anova_models(results: dict[str, list[float]]) -> dict:
    """One-way ANOVA + Tukey HSD post-hoc 검정.

    NOTE (v0.6): Phase 1 zero-shot은 결정적이라 시드-분산이 0이므로 이 함수 대신
    run_cochran_q / run_mcnemar(공유 테스트셋 짝지은 검정)를 사용한다.
    이 함수는 시드-분산이 실재하는 맥락(예: 학습 기반 다중 시드)에서만 유효하다.

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


def run_mann_whitney(x: list[float], y: list[float]) -> dict:
    """Mann-Whitney U 검정 (2-독립표본, 양측) + rank-biserial 효과 크기.

    Phase 3: run-level 비교 (예: Autoresearch vs Optuna, n=10 vs n=10)에 사용하는
    비모수 2-독립표본 순위합 검정. REQ-EM-006, REQ-EM-007, REQ-EM-008, REQ-EM-009.

    scipy의 정규근사(동점 보정)를 그대로 사용하므로 동순위(tied ranks)가 있어도
    예외 없이 [0, 1] 범위의 유효한 p-value를 반환한다 (REQ-EM-009).

    부호 규약 (REQ-EM-007): rank_biserial_r > 0 은 첫 표본(x)이 둘째 표본(y)보다
    확률적으로 큰 경향임을 의미한다. scipy.stats.mannwhitneyu가 반환하는 U는
    "x 기준" 통계량 — 즉 x_i > y_j 인 쌍의 수(동점은 0.5로 계산, 경험적으로 검증:
    x=[5,6,7], y=[1,2,3] -> U=9=n1*n2)이므로, x가 y를 완전히 압도하면
    U -> n1*n2 이 된다. 따라서 r = 2*U/(n1*n2) - 1 로 정의하면 이 경우 r -> +1이
    되어 요구된 부호 규약을 만족한다. 인자 순서를 (y, x)로 반전하면 항상
    U_y = n1*n2 - U_x (동점 여부와 무관하게 U_x + U_y = n1*n2)이므로 r의 부호도
    반전한다.

    Args:
        x: 첫 번째 표본 (예: Autoresearch run-level accuracy, n=10).
        y: 두 번째 표본 (예: Optuna run-level accuracy, n=10).

    Returns:
        dict with: u_stat, p_value, n1, n2, rank_biserial_r, significant (alpha=0.05).
    """
    n1 = len(x)
    n2 = len(y)
    result = stats.mannwhitneyu(x, y, alternative="two-sided")
    u_stat = float(result.statistic)
    p_value = float(result.pvalue)
    rank_biserial_r = (2 * u_stat) / (n1 * n2) - 1

    return {
        "u_stat": u_stat,
        "p_value": p_value,
        "n1": n1,
        "n2": n2,
        "rank_biserial_r": float(rank_biserial_r),
        "significant": bool(p_value < 0.05),
    }
