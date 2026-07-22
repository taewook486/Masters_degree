# Phase 2 RQ2 분석 — QLoRA 파인튜닝 효과

- base(zero-shot) seed: 42  ·  조건 수: 0
- 모델: 
- 검정: paired t-test + BCa Bootstrap 95% CI(Cohen's d) + Wilcoxon signed-rank
- 파인튜닝 효과 = finetuned − base (overall_accuracy)

> **주의**: base 또는 eval 누락으로 제외된 조건 36개: gemma4-e2b/pathvqa/seed123, gemma4-e2b/pathvqa/seed42, gemma4-e2b/pathvqa/seed456, gemma4-e2b/slake/seed123, gemma4-e2b/slake/seed42, gemma4-e2b/slake/seed456, gemma4-e2b/vqa_rad/seed123, gemma4-e2b/vqa_rad/seed42, gemma4-e2b/vqa_rad/seed456, qwen25-vl-3b/pathvqa/seed123, qwen25-vl-3b/pathvqa/seed42, qwen25-vl-3b/pathvqa/seed456, qwen25-vl-3b/slake/seed123, qwen25-vl-3b/slake/seed42, qwen25-vl-3b/slake/seed456, qwen25-vl-3b/vqa_rad/seed123, qwen25-vl-3b/vqa_rad/seed42, qwen25-vl-3b/vqa_rad/seed456, qwen3-vl-2b/pathvqa/seed123, qwen3-vl-2b/pathvqa/seed42, qwen3-vl-2b/pathvqa/seed456, qwen3-vl-2b/slake/seed123, qwen3-vl-2b/slake/seed42, qwen3-vl-2b/slake/seed456, qwen3-vl-2b/vqa_rad/seed123, qwen3-vl-2b/vqa_rad/seed42, qwen3-vl-2b/vqa_rad/seed456, smolvlm2-2b/pathvqa/seed123, smolvlm2-2b/pathvqa/seed42, smolvlm2-2b/pathvqa/seed456, smolvlm2-2b/slake/seed123, smolvlm2-2b/slake/seed42, smolvlm2-2b/slake/seed456, smolvlm2-2b/vqa_rad/seed123, smolvlm2-2b/vqa_rad/seed42, smolvlm2-2b/vqa_rad/seed456

## 모델별 파인튜닝 효과

| 모델 | n | base | finetuned | Cohen's d | d 95% CI | t p | Wilcoxon p |
|------|:-:|:----:|:---------:|:---------:|:--------:|:---:|:----------:|

## Mixed-Effects Model (accuracy ~ condition + dataset, group=seed)

> statsmodels/pandas 미설치 또는 표본 부족으로 생략 (`uv pip install statsmodels pandas`).
