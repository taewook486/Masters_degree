"""statistics.py의 run_mann_whitney 테스트 (SPEC-EVAL-METRICS-001, 영역 2).

REQ-EM-006 (2-독립표본 순위합 검정 제공), REQ-EM-007 (효과 크기 + 부호 규약),
REQ-EM-008 (유의성 판정 규약 일관성), REQ-EM-009 (동순위 처리)를 검증한다.

scipy 실호출(경량, 모델 다운로드 불필요)로 수행한다 — mock 불필요.
범위 제한(SPEC Exclusions #2, #3): Phase 3 분석 스크립트 배선 및 기존 5개
무테스트 statistics.py 함수의 백필은 본 파일의 범위 밖이다.
"""

from __future__ import annotations

from src.evaluate.statistics import run_mann_whitney


def test_run_mann_whitney_basic_return_structure():
    """AC-2-1, REQ-EM-006, REQ-EM-008: 기본 양측 검정 반환 구조.

    n1=10, n2=10 두 독립표본에 대해 u_stat, [0,1] 범위의 p_value, n1/n2,
    rank_biserial_r, 그리고 significant(=p_value<0.05의 bool)를 반환해야 한다.
    """
    x = [0.70, 0.72, 0.68, 0.75, 0.71, 0.69, 0.73, 0.74, 0.70, 0.72]
    y = [0.60, 0.62, 0.58, 0.65, 0.61, 0.59, 0.63, 0.64, 0.60, 0.62]

    result = run_mann_whitney(x, y)

    assert "u_stat" in result
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n1"] == 10
    assert result["n2"] == 10
    assert "rank_biserial_r" in result
    assert isinstance(result["significant"], bool)
    assert result["significant"] == (result["p_value"] < 0.05)


def test_run_mann_whitney_tied_ranks_no_exception():
    """AC-2-2, REQ-EM-009: 동순위(tied ranks) 처리 — 엣지케이스.

    두 표본에 동일 값이 다수 포함되어도 예외 없이 [0,1] 범위의 유효한
    p_value를 반환해야 한다 (표준 정규근사 동점 보정).
    """
    x = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    y = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4]

    result = run_mann_whitney(x, y)

    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n1"] == 10
    assert result["n2"] == 10


def test_run_mann_whitney_effect_size_extreme_positive():
    """AC-2-3a, REQ-EM-007: 효과 크기 극단값 (순방향).

    첫 표본(x)이 둘째 표본(y)보다 확률적으로 큰 완전 분리 상태(x의 모든 값 >
    y의 모든 값)이면, 양의 극단(+1 근처)의 rank_biserial_r와
    significant=True를 반환해야 한다.
    """
    x = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    y = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]

    result = run_mann_whitney(x, y)

    assert result["rank_biserial_r"] > 0.9
    assert result["significant"] is True


def test_run_mann_whitney_effect_size_sign_flip_on_argument_reversal():
    """AC-2-3b, REQ-EM-007: 효과 크기 부호 반전 (인자 반전).

    동일 표본을 인자 순서를 반전하여(y, x) 호출하면, 부호가 반전된(음수)
    rank_biserial_r를 반환해야 한다.
    """
    x = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    y = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]

    forward = run_mann_whitney(x, y)
    reversed_ = run_mann_whitney(y, x)

    assert forward["rank_biserial_r"] > 0
    assert reversed_["rank_biserial_r"] < 0
    assert forward["rank_biserial_r"] == -reversed_["rank_biserial_r"]


def test_run_mann_whitney_schema_consistency_with_siblings():
    """AC-2-4, REQ-EM-008: 유의성 판정 규약 일관성.

    significant 플래그는 형제 순위검정 함수(run_wilcoxon, run_kruskal_wallis)와
    동일한 규약, 즉 p_value < 0.05의 bool로 보고되어야 한다.
    """
    x = [0.70, 0.72, 0.68, 0.75, 0.71, 0.69, 0.73, 0.74, 0.70, 0.72]
    y = [0.71, 0.73, 0.69, 0.74, 0.70, 0.68, 0.72, 0.75, 0.71, 0.73]

    result = run_mann_whitney(x, y)

    assert result["significant"] == (result["p_value"] < 0.05)
    assert set(result.keys()) == {
        "u_stat",
        "p_value",
        "n1",
        "n2",
        "rank_biserial_r",
        "significant",
    }
