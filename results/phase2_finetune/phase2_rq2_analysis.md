# Phase 2 RQ2 분석 — QLoRA 파인튜닝 효과

- base(zero-shot) seed: 42  ·  조건 수: 36
- 모델: gemma4-e2b, qwen25-vl-3b, qwen3-vl-2b, smolvlm2-2b
- 검정: paired t-test + BCa Bootstrap 95% CI(Cohen's d) + Wilcoxon signed-rank
- 파인튜닝 효과 = finetuned − base (overall_accuracy)

## 모델별 파인튜닝 효과

| 모델 | n | base | finetuned | Cohen's d | d 95% CI | t p | Wilcoxon p |
|------|:-:|:----:|:---------:|:---------:|:--------:|:---:|:----------:|
| gemma4-e2b | 9 | 0.3285 | 0.2288 | -0.652 | [-1.599, 0.032] | 0.0864 | 0.1289 |
| qwen25-vl-3b | 9 | 0.4582 | 0.5749 | 2.646 | [1.953, 4.723] | 0 | 0.003906 |
| qwen3-vl-2b | 9 | 0.4725 | 0.5845 | 1.620 | [0.932, 3.153] | 0.0013 | 0.003906 |
| smolvlm2-2b | 9 | 0.4253 | 0.4036 | -2.284 | [-3.160, -1.552] | 0.0001 | 0.003906 |

## Mixed-Effects Model (accuracy ~ condition + dataset, group=seed)

- condition[finetuned] 고정효과 계수: **0.0268** (p = 0.3629)
- ICC(seed): 0.0  ·  잔차분산: 0.015635  ·  n = 72
