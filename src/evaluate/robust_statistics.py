"""Robust 통계 분석: BCa Bootstrap + Mixed-Effects Model.

동료 심사 II번 지적("Phase 2 n=9 paired t-test의 Cohen's d 추정 한계")에 대응.

3가지 방법을 병행하여 결과의 robustness를 검증:
1. Paired t-test + Cohen's d (관행 비교)
2. BCa Bootstrap 95% CI for Cohen's d (n=9에 robust)
3. Mixed-Effects Model (seed, dataset random effect 처리)

참조:
- Efron & Tibshirani, "An Introduction to the Bootstrap" (1993)
- Bates et al., "Fitting Linear Mixed-Effects Models Using lme4" (JSS 2015)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RobustStatResult:
    """3가지 통계 방법 통합 결과."""
    # Primary: paired t-test
    t_statistic: float
    t_pvalue: float
    cohens_d: float

    # Robust: BCa Bootstrap
    cohens_d_ci_lower: float
    cohens_d_ci_upper: float
    bootstrap_iterations: int

    # Non-parametric: Wilcoxon signed-rank
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    wilcoxon_effect_size_r: float

    # Sample size
    n: int


def compute_cohens_d(differences: Sequence[float]) -> float:
    """Paired Cohen's d 계산.

    Args:
        differences: 쌍별 차이 (예: fine-tuned - base).

    Returns:
        Cohen's d 효과 크기.
    """
    arr = np.asarray(differences, dtype=float)
    if len(arr) < 2:
        return 0.0
    mean_diff = float(arr.mean())
    std_diff = float(arr.std(ddof=1))
    if std_diff == 0:
        return 0.0
    return mean_diff / std_diff


def bootstrap_cohens_d_ci(
    differences: Sequence[float],
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """BCa Bootstrap 95% CI for Cohen's d.

    scipy.stats.bootstrap 사용. n=9 등 작은 표본에서 paired Cohen's d의
    분포가 비대칭일 때 BCa(Bias-Corrected and accelerated) 보정이 t-test보다
    robust한 추정을 제공한다.

    Args:
        differences: 쌍별 차이 데이터.
        n_resamples: Bootstrap resample 횟수 (기본 10,000).
        confidence_level: 신뢰구간 수준 (기본 0.95).
        random_state: 재현성 시드.

    Returns:
        (lower, upper) 신뢰구간 튜플.
    """
    arr = np.asarray(differences, dtype=float)
    if len(arr) < 2:
        return 0.0, 0.0

    rng = np.random.default_rng(random_state)
    result = stats.bootstrap(
        (arr,),
        statistic=lambda data, axis: _vectorized_cohens_d(data, axis),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="BCa",
        random_state=rng,
        vectorized=True,
    )
    return float(result.confidence_interval.low), float(
        result.confidence_interval.high
    )


def _vectorized_cohens_d(data: np.ndarray, axis: int) -> np.ndarray:
    """Cohen's d 벡터화 버전 (scipy bootstrap 호환).

    Args:
        data: shape (..., n_samples) 형태의 배열.
        axis: bootstrap 축.

    Returns:
        해당 축이 축약된 Cohen's d 배열.
    """
    mean_d = np.mean(data, axis=axis)
    std_d = np.std(data, axis=axis, ddof=1)
    # 0으로 나누기 방지
    return np.where(std_d > 0, mean_d / std_d, 0.0)


def wilcoxon_with_effect_size(
    base_values: Sequence[float],
    finetuned_values: Sequence[float],
) -> tuple[float, float, float]:
    """Wilcoxon signed-rank test + 효과 크기 r (z / √n).

    Args:
        base_values: 파인튜닝 전 측정값.
        finetuned_values: 파인튜닝 후 측정값.

    Returns:
        (statistic, p_value, effect_size_r) 튜플.
    """
    base = np.asarray(base_values, dtype=float)
    ft = np.asarray(finetuned_values, dtype=float)

    if len(base) != len(ft) or len(base) < 1:
        return 0.0, 1.0, 0.0

    try:
        result = stats.wilcoxon(ft, base, zero_method="wilcox", correction=False)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

        # 효과 크기 r = |z| / √n (Rosenthal 1991)
        # scipy wilcoxon은 z 값을 직접 반환하지 않으므로 정규 근사 기반 계산
        n = len(base)
        # z-score 근사: 부호화된 평균 / 표준오차
        diffs = ft - base
        non_zero = diffs[diffs != 0]
        if len(non_zero) == 0:
            return statistic, p_value, 0.0
        # p_value → 양측 z-score 환산
        z_abs = float(abs(stats.norm.ppf(p_value / 2))) if p_value > 0 else 0.0
        effect_r = z_abs / np.sqrt(n) if n > 0 else 0.0
        return statistic, p_value, round(effect_r, 4)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Wilcoxon 계산 실패: {e}")
        return 0.0, 1.0, 0.0


def analyze_paired_robust(
    base_values: Sequence[float],
    finetuned_values: Sequence[float],
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> RobustStatResult:
    """3가지 통계 방법 통합 분석.

    Args:
        base_values: 파인튜닝 전 측정값 (예: 9개).
        finetuned_values: 파인튜닝 후 측정값 (동일 길이).
        n_bootstrap: Bootstrap resample 횟수.
        random_state: 재현성 시드.

    Returns:
        RobustStatResult dataclass.
    """
    base = np.asarray(base_values, dtype=float)
    ft = np.asarray(finetuned_values, dtype=float)
    differences = ft - base
    n = len(differences)

    # 1. Paired t-test + Cohen's d
    if n >= 2:
        t_result = stats.ttest_rel(ft, base)
        t_statistic = float(t_result.statistic)
        t_pvalue = float(t_result.pvalue)
    else:
        t_statistic, t_pvalue = 0.0, 1.0
    cohens_d = compute_cohens_d(differences)

    # 2. BCa Bootstrap CI
    if n >= 2:
        ci_lower, ci_upper = bootstrap_cohens_d_ci(
            differences,
            n_resamples=n_bootstrap,
            random_state=random_state,
        )
    else:
        ci_lower, ci_upper = 0.0, 0.0

    # 3. Wilcoxon + r
    wilcoxon_stat, wilcoxon_p, wilcoxon_r = wilcoxon_with_effect_size(base, ft)

    return RobustStatResult(
        t_statistic=round(t_statistic, 4),
        t_pvalue=round(t_pvalue, 4),
        cohens_d=round(cohens_d, 4),
        cohens_d_ci_lower=round(ci_lower, 4),
        cohens_d_ci_upper=round(ci_upper, 4),
        bootstrap_iterations=n_bootstrap,
        wilcoxon_statistic=wilcoxon_stat,
        wilcoxon_pvalue=wilcoxon_p,
        wilcoxon_effect_size_r=wilcoxon_r,
        n=n,
    )


def mixed_effects_analysis(
    accuracies: Sequence[float],
    conditions: Sequence[str],
    seeds: Sequence[int],
    datasets: Sequence[str],
) -> dict:
    """Mixed-Effects Model: accuracy ~ condition + (1|seed) + (1|dataset).

    Args:
        accuracies: 정확도 값 배열.
        conditions: 조건 라벨 ("base" 또는 "finetuned").
        seeds: 시드 ID.
        datasets: 데이터셋 이름.

    Returns:
        {fixed_effect_coef, p_value, icc_seed, icc_dataset, residual_var, n} 딕셔너리.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        logger.warning(
            "statsmodels/pandas 미설치. mixed-effects 분석 생략. "
            "uv pip install statsmodels pandas"
        )
        return {
            "fixed_effect_coef": 0.0,
            "p_value": 1.0,
            "icc_seed": 0.0,
            "icc_dataset": 0.0,
            "residual_var": 0.0,
            "n": len(accuracies),
            "skipped": True,
        }

    df = pd.DataFrame({
        "accuracy": list(accuracies),
        "condition": list(conditions),
        "seed": [str(s) for s in seeds],
        "dataset": list(datasets),
    })

    if len(df) < 4:
        return {
            "fixed_effect_coef": 0.0,
            "p_value": 1.0,
            "icc_seed": 0.0,
            "icc_dataset": 0.0,
            "residual_var": 0.0,
            "n": len(df),
            "skipped": True,
        }

    try:
        # 단순 random intercept (seed만, dataset은 fixed로 처리)
        # statsmodels는 다중 random effect 제한적이므로 seed 단일 group 사용
        model = smf.mixedlm(
            "accuracy ~ condition + dataset",
            data=df,
            groups=df["seed"],
        )
        result = model.fit(method="lbfgs", reml=True)

        # condition[finetuned] 계수 (base가 reference)
        coef_keys = [k for k in result.params.index if "condition" in k.lower()]
        coef = float(result.params[coef_keys[0]]) if coef_keys else 0.0
        pval = float(result.pvalues[coef_keys[0]]) if coef_keys else 1.0

        # ICC = 그룹 분산 / (그룹 분산 + 잔차 분산)
        group_var = float(result.cov_re.iloc[0, 0]) if not result.cov_re.empty else 0.0
        residual_var = float(result.scale)
        total_var = group_var + residual_var
        icc_seed = group_var / total_var if total_var > 0 else 0.0

        return {
            "fixed_effect_coef": round(coef, 4),
            "p_value": round(pval, 4),
            "icc_seed": round(icc_seed, 4),
            "icc_dataset": 0.0,  # dataset은 fixed effect로 처리됨
            "residual_var": round(residual_var, 6),
            "n": len(df),
            "skipped": False,
        }
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        logger.warning(f"Mixed-effects 적합 실패: {e}")
        return {
            "fixed_effect_coef": 0.0,
            "p_value": 1.0,
            "icc_seed": 0.0,
            "icc_dataset": 0.0,
            "residual_var": 0.0,
            "n": len(df),
            "skipped": True,
        }
