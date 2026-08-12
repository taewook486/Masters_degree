# Phase 3 RQ3 분석 — HPO 전략 비교 (run-level)

- 분석 단위: run-level (전략 x 독립 반복 -> 반복별 최고 val_accuracy)
- trial-level 데이터는 탐색 궤적 시각화 전용이며 본 비교 검정에는 사용하지 않음 (설계서 §4.5 v0.5 변경 — sequential optimization의 trial 간 의존성)

## 전략별 run-level 성능 (Bootstrap 95% CI)

| 전략 | n | mean | 95% CI |
|------|:-:|:----:|:------:|
| manual | 10 | 0.3776 | [0.3760, 0.3794] |
| random | 10 | 0.4186 | [0.4106, 0.4274] |
| optuna | 10 | 0.4490 | [0.4368, 0.4594] |
| autoresearch | 10 | 0.4184 | [0.4064, 0.4328] |

## Kruskal-Wallis (전략 간 비교)

- H = 27.9161, p = 3.782e-06 -> 유의함 (p < 0.05)

## Mann-Whitney U — Autoresearch vs Optuna (RQ3 핵심 쌍별 비교)

- U = 16.0000, p = 0.01115, rank-biserial r = -0.6800 -> 유의함 (p < 0.05)
- r > 0: Autoresearch가 Optuna보다 확률적으로 우세하는 경향 (부호 규약: `src/evaluate/statistics.py` run_mann_whitney 참조)
