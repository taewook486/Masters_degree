# 구현 계획: SPEC-EVAL-METRICS-001

## 구현 접근 개요

두 개의 독립적 코드 산출물을 추가한다. 서로 파일 의존이 없어 병렬 진행 가능하다.

- **item 1** (`metrics.py` 중심): BERTScore 이중 모델 계산을 **opt-in 파라미터 스레딩**으로 도입. 기본 경로(Phase 1)는 완전 불변, Phase 2 호출부에서만 BioBERT를 명시적으로 활성화.
- **item 2** (`statistics.py`): `run_mann_whitney` 신규 함수 + 단위 테스트. 의존성 없는 독립 완결 단위.

방법론: `.moai/config/sections/quality.yaml`의 `development_mode`에 따름. 두 항목 모두 순수 함수 + 기존 mock 관행(`tests/test_metrics.py`)이 확립되어 있어 **TDD(RED-GREEN-REFACTOR)** 에 적합.

---

## 수정 / 신규 대상 파일

| 파일 | 변경 유형 | 상세 |
|------|-----------|------|
| `src/evaluate/metrics.py` | 수정 | `compute_open_bertscore(..., num_layers: int \| None = None)` 추가; `compute_overall_accuracy(..., bertscore_models: list[str] \| None = None)` 추가; 이중 모델 반복 호출 + secondary 키 + 상관 산출 |
| `src/baseline/evaluate_zero_shot.py` | 수정 | `evaluate_with_loaded_model(..., bertscore_models: list[str] \| None = None)` 추가, L96 `compute_overall_accuracy` 호출로 전달 |
| `src/finetune/train_qlora.py` | 수정 | L665 부근 `evaluate_with_loaded_model` 호출에 `bertscore_models=["roberta-large", "dmis-lab/biobert-v1.1"]` 전달 |
| `src/evaluate/statistics.py` | 신규 함수 | `run_mann_whitney(x, y)` |
| `tests/test_metrics.py` | 테스트 추가 | 이중 모델 / 상관 / num_layers 전달 (bert_score_fn mock) |
| `tests/test_statistics.py` | 신규 파일 | `run_mann_whitney` 단위 테스트 (docstring에 REQ-ID 참조) |

---

## item 1: opt-in 스레딩 설계 (핵심)

```
[Phase 2] train_qlora.py
   evaluate_with_loaded_model(bertscore_models=["roberta-large","dmis-lab/biobert-v1.1"])
      -> compute_overall_accuracy(bertscore_models=[...])
            -> compute_open_bertscore(model_type="roberta-large")            # primary
            -> compute_open_bertscore(model_type="dmis-lab/biobert-v1.1",
                                      num_layers=9)                          # secondary
            -> Spearman/Pearson(per-sample F1_roberta, per-sample F1_biobert)

[Phase 1] evaluate_zero_shot.py main() / run_all.py
   evaluate_with_loaded_model(bertscore_models=None)   # 미지정
      -> compute_overall_accuracy(bertscore_models=None)
            -> compute_open_bertscore(model_type="roberta-large")            # 기존 동작 그대로
```

**하위 호환 불변식**: `bertscore_models=None` → roberta-large 단일 → 기존 결과 키(`open_bertscore_f1`, `open_bertscore_accuracy`)만 생성 → Phase 1 호출부·`phase1_summary.csv` 스키마 불변.

### 결과 키 스키마

| 키 | 조건 | 의미 |
|----|------|------|
| `open_bertscore_f1` | 항상(open 존재 시) | roberta-large F1 mean — **primary** |
| `open_bertscore_accuracy` | 항상(open 존재 시) | roberta-large @ 0.7 정확도 — **primary/결정** |
| `open_bertscore_f1_biobert` | BioBERT 요청 시 | BioBERT F1 mean — secondary |
| `open_bertscore_accuracy_biobert` | BioBERT 요청 시 | BioBERT @ 0.7 정확도 — informational (임계값 미재보정) |
| `open_bertscore_spearman` | 이중 모델 시 | per-sample F1 Spearman 상관 — **주 보고값** |
| `open_bertscore_pearson` | 이중 모델 시 | per-sample F1 Pearson 상관 — 병기 |

상관 계산 전제: 두 모델의 per-sample F1 리스트 길이가 동일(같은 open 샘플 집합). 표본 < 2이면 상관 미정의 → `None` 또는 계산 생략(테스트로 고정).

### 상관계수 설계 (REQ-EM-003)

- Spearman: `scipy.stats.spearmanr(f1_roberta, f1_biobert).correlation` — headline.
- Pearson: `scipy.stats.pearsonr(f1_roberta, f1_biobert)[0]` — 병기.
- `scipy`는 이미 의존성(`>=1.14.0`).

### BioBERT num_layers 기술 리스크 + 결정 (REQ-EM-005)

- bert-score 0.3.13은 `model2layers` 레지스트리로 모델별 기본 임베딩 레이어를 결정한다. `dmis-lab/biobert-v1.1`은 미등록 가능성이 높아, `num_layers` 미지정 호출은 `KeyError` 예상.
- 결정: `compute_open_bertscore`가 BioBERT 계산 시 `bert_score.score(..., num_layers=9)`를 전달. 근거: BioBERT-v1.1 = BERT-base(12층), BERTScore 원논문의 bert-base 경험적 최적 레이어 = 9. 문서화된 근거 있는 기본값.
- 구현 노트: `num_layers`를 `compute_open_bertscore` 시그니처에 노출하고, 이중 모델 경로에서 roberta-large는 `num_layers=None`(레지스트리 기본), BioBERT는 `num_layers=9`를 넘긴다. bert-score 0.3.13 `score()`는 `num_layers` 파라미터를 지원한다.

### primary 규칙 (REQ-EM-004)

- `compute_overall_accuracy`의 correctness / `overall_accuracy` / `open_bertscore_accuracy`는 roberta-large 경로에만 근거. BioBERT 값은 별도 secondary 키로만 노출되며 어떤 정오 판정에도 관여하지 않는다.

---

## item 2: run_mann_whitney 설계

- 시그니처: `run_mann_whitney(x: list[float], y: list[float]) -> dict`
- 본체: `U, p = scipy.stats.mannwhitneyu(x, y, alternative="two-sided")`
- 효과 크기: `rank_biserial_r = 1 - (2*U) / (n1*n2)` (U는 x 기준; 양수 = x가 y보다 큰 경향 — docstring 명시)
- 반환: `{"u_stat": float(U), "p_value": float(p), "n1": len(x), "n2": len(y), "rank_biserial_r": float(r), "significant": bool(p < 0.05)}`
- 동순위: scipy 기본 정규근사(동점 보정)를 그대로 사용 (별도 처리 불필요, 테스트로 검증).
- 형제 함수(`run_wilcoxon`, `run_kruskal_wallis`)와 동일한 dict 반환 규약 준수.

---

## 마일스톤 (우선순위 기반, 시간 추정 없음)

1. **(High)** `run_mann_whitney` 구현 + `tests/test_statistics.py` — 의존성 없음, 독립 완결. 먼저 착수.
2. **(High)** `metrics.py`: `num_layers` 노출 + `bertscore_models` 이중 모델 반복 + secondary 키 + Spearman/Pearson.
3. **(High)** `evaluate_zero_shot.py` opt-in 배선(기본 None) → `train_qlora.py`에서 BioBERT 활성화.
4. **(Medium)** `tests/test_metrics.py` 이중 모델/상관/num_layers 테스트 확장.

권장 순서: 1 → 2 → 3 → 4 (2는 3의 선행, 1은 병렬 가능).

---

## 테스트 전략

- bert-score 실모델 다운로드 금지. `tests/test_metrics.py`의 기존 `bert_score_fn` / `bert_score` 모듈 mock 패턴을 그대로 따른다.
- num_layers 검증은 mock의 호출 인자(call_args)를 검사하여 BioBERT 계산 시 `num_layers=9`가 전달되는지 확인.
- `run_mann_whitney` 테스트는 scipy 실호출(경량, 모델 불필요)로 수행.

---

## 리스크

- **(범위 외 관찰)** `statistics.py`의 기존 5개 함수는 현재 무테스트 상태이나, 본 SPEC은 이를 백필하지 않는다(Exclusions #3). 향후 별도 SPEC 권장.
- **BioBERT 초기 다운로드**: 실제 Phase 2 실행 시 BioBERT 가중치 최초 다운로드 + VRAM 추가 점유 발생. opt-in이므로 Phase 1은 영향 없음. Phase 2 GPU 예산에서 고려 필요(테스트에는 영향 없음 — mock).
- **num_layers=9 가정**: BioBERT-v1.1의 최적 레이어가 원논문 bert-base와 다를 수 있으나, BioBERT는 secondary 지표이고 값은 문서화된 근거 기반이므로 논문 방어 가능.
