# §4.3 Phase 3 결과 — 작성용 초안 (숫자 미확정)

> **이 파일의 성격**: `docs/THESIS_FINAL_v2.0.md`의 424행 `*(다음 작업: 4.3 ...)*` 자리에 들어갈 §4.3의 **뼈대 초안**이다.
> 2026-08-12 작성 시점에 Phase 3 autoresearch가 172/200 진행 중이라 **최종 수치가 없다.**
> 숫자가 들어갈 자리는 전부 `[TBD-n]`으로 비워 두고, 아래 §0에 각 자리를 채우는 출처 명령을 적어 두었다.
> **`[TBD-n]`이 하나라도 남아 있는 상태로 본문에 병합하지 말 것.**

---

## §0. 채우는 절차 (작성 시작 시 이 순서대로)

### 0-1. 선행 확인 (표 작성 전 반드시)

```bash
# (a) autoresearch 200/200 완료 확인 — 200 미만이면 §4.3 작성 자체를 시작하지 말 것
awk -F'\t' '$2=="autoresearch" && $19=="completed"' results/phase3_autoresearch/results.tsv | wc -l

# (b) 전략별 completed 행수 (기대: manual 10 / random 200 / optuna 200 / autoresearch 200)
awk -F'\t' 'NR>1 && $2!=""{print $2, $19}' results/phase3_autoresearch/results.tsv | sort | uniq -c

# (c) run-level 표본 수 확인 — 전 전략 repeat 0~9가 각각 1개 이상 있어야 n=10
awk -F'\t' '$19=="completed"{print $2, $3}' results/phase3_autoresearch/results.tsv | sort -u | awk '{print $1}' | uniq -c
```

### 0-2. 분석 실행

```bash
python3 scripts/analyze_phase3.py --results_dir results/phase3_autoresearch
#   → results/phase3_autoresearch/phase3_rq3_analysis.md  (사람이 읽는 리포트)
#   → results/phase3_autoresearch/phase3_rq3_analysis.json (기계 판독 — 표 수치는 여기서 인용)

python3 scripts/summarize_stage.py manual random optuna autoresearch
#   → results/phase3_autoresearch/phase3_summary.txt (전략별 best trial 하이퍼파라미터)
```

### 0-3. `[TBD-n]` 대응표

| 자리 | 출처 | 비고 |
|------|------|------|
| `[TBD-1]` ~ `[TBD-4]` | `phase3_rq3_analysis.json` → `bootstrap_ci.<전략>.mean` | 소수 4자리 |
| `[TBD-5]` ~ `[TBD-8]` | 같은 파일 → `bootstrap_ci.<전략>.ci_low` / `ci_high` | `[low, high]` 형식 |
| `[TBD-9]` `[TBD-10]` `[TBD-11]` | 같은 파일 → `kruskal_wallis.h_stat` / `p_value` / `significant` | |
| `[TBD-12]` ~ `[TBD-15]` | 같은 파일 → `mann_whitney_autoresearch_vs_optuna.u_stat` / `p_value` / `rank_biserial_r` / `significant` | 부호 규약은 `src/evaluate/statistics.py` 확인 |
| `[TBD-16]` ~ `[TBD-31]` | `phase3_summary.txt` → 전략별 `best trial` 블록 | 하이퍼파라미터 8종 |
| `[TBD-32]` ~ `[TBD-35]` | `phase3_summary.txt` → 전략별 `val_accuracy: mean/std` | 참고용 trial-level |
| `[TBD-36]` ~ | 4.3.3 탐색 궤적 — 아래 §4.3.3 주석 참조 | 별도 집계 필요 |

---

## §4.3 Phase 3: 자율 하이퍼파라미터 최적화 결과  *(← 본문 병합 대상)*

Phase 3은 Phase 2에서 최고 성능을 보인 Qwen3-VL-2B를 PathVQA에 고정하고, 하이퍼파라미터 탐색 전략 4종(Manual·Random Search·Optuna(TPE)·Autoresearch(LLM 에이전트))을 동일 조건에서 비교하여 RQ3("LLM 에이전트의 자율 탐색이 기존 HPO 기법 대비 경쟁력 있는 성능에 도달하는가")에 답한다. 전 trial 공통으로 `max_steps=200`으로 학습량을 통제했으며, 최종 실행 규모는 Manual 10 + Random 200 + Optuna 200 + Autoresearch 200 = 총 610 trial(전략당 20 trial × 10회 독립 반복, Manual은 반복당 1 trial)이다.

§3.7에서 기술한 대로 **통계 검정은 trial-level이 아닌 run-level에서만 수행**한다. Optuna와 Autoresearch는 순차 최적화 특성상 동일 run 내 trial 간 의존성이 있어 개별 trial을 독립 관측치로 취급할 수 없기 때문이다. 따라서 검정 단위는 각 전략의 10회 독립 반복에서 산출한 **반복별 최고 val_accuracy 10개**이며(Manual은 반복당 trial이 1개이므로 그 값 자체가 run-level 값이 된다), trial-level 데이터는 4.3.3의 탐색 궤적 분석에만 사용한다.

### 4.3.1 전략별 최종 성능 비교 (run-level)

**Table 4.3. HPO 전략별 run-level 성능 (n=10 = 독립 반복 10회의 반복별 최고 val_accuracy)**

| 전략 | n | 평균 val_accuracy | 95% CI (Bootstrap) |
|------|:-:|:-----------------:|:------------------:|
| Manual | 10 | [TBD-1] | [TBD-5] |
| Random Search | 10 | [TBD-2] | [TBD-6] |
| Optuna (TPE) | 10 | [TBD-3] | [TBD-7] |
| **Autoresearch (LLM)** | 10 | **[TBD-4]** | [TBD-8] |

> run-level 값 = 각 독립 반복에서 완료된 trial 중 최고 val_accuracy. 실패(status ≠ completed) trial은 집계에서 제외한다. Manual 전략에서는 반복 6~9에서 총 10건의 실패 후 재시도 기록이 있으나, 전 반복이 최종적으로 1건씩 정상 완료되어 run-level 표본 수 10은 유지된다.

4개 전략 간 run-level 성능 차이에 대한 **Kruskal-Wallis 검정** 결과는 H = [TBD-9], p = [TBD-10]으로 [TBD-11: 유의함 / 유의하지 않음]이었다.

RQ3의 핵심 쌍별 비교인 **Autoresearch vs Optuna**에 대한 Mann-Whitney U 검정은 U = [TBD-12], p = [TBD-13], rank-biserial r = [TBD-14]로 [TBD-15: 유의함 / 유의하지 않음]이었다.

<!-- 해석 분기 가이드 — 결과 확인 후 해당하는 쪽만 남기고 나머지는 삭제할 것

  [분기 A] KW 유의 + MW 유의 + Autoresearch 우세:
    → "LLM 에이전트의 자율 탐색이 확립된 베이지안 최적화(TPE) 대비 통계적으로 우월한 성능에 도달했다"로 서술.
       단 n=10, 단일 모델·단일 데이터셋 조건임을 즉시 병기하고 일반화 주장은 하지 말 것.

  [분기 B] KW 유의 + MW 비유의 (예: 둘 다 Manual보다 높지만 서로는 구분 안 됨):
    → "전략 간 차이는 존재하나 Autoresearch와 Optuna는 통계적으로 구분되지 않는 수준으로 근접" 
       = RQ3의 답으로는 "우월"이 아니라 "경쟁력 있음(comparable)"이 정확한 표현.
       4.1.1에서 Qwen3-VL-2B vs Qwen2.5-VL-3B를 다룬 서술 방식을 그대로 차용할 것.

  [분기 C] KW 비유의:
    → "탐색 전략 선택이 최종 성능에 미치는 영향은 본 실험 조건(max_steps=200, 20 trial 예산)에서 
       통계적으로 검출되지 않았다"로 서술. 억지로 순위를 주장하지 말 것.
       이 경우 원인 후보(탐색 예산 20 trial의 부족, max_steps=200에 의한 성능 상한 압축,
       탐색 공간 자체가 좁아 무작위 탐색으로도 충분)를 5.3 한계점과 연결해 논의할 것.

  ※ 어느 분기든, trial-level 최고값(예: optuna trial 304의 0.4700)을 결론 근거로 인용하지 말 것.
     그 값은 run-level 검정 단위가 아니며, 본문에 인용하면 §3.7에서 선언한 분석 단위와 모순된다.
-->

### 4.3.2 전략별 탐색 결과 하이퍼파라미터

**Table 4.3a. 전략별 최고 성능 trial의 하이퍼파라미터 구성**

| 전략 | rank | alpha | learning_rate | batch | grad_accum | warmup | wd | targets | val_acc |
|------|:----:|:-----:|:-------------:|:-----:|:----------:|:------:|:--:|:-------:|:-------:|
| Manual | [TBD-16] | [TBD-17] | [TBD-18] | [TBD-19] | | | | [TBD-20] | [TBD-21] |
| Random | [TBD-22] | | | | | | | | [TBD-23] |
| Optuna | [TBD-24] | | | | | | | | [TBD-25] |
| Autoresearch | [TBD-26] | | | | | | | | [TBD-27] |

<!-- 작성 지침:
  - 위 표는 "각 전략이 도달한 최선의 설정"이며 run-level 검정과는 별개의 기술 통계다.
    본문에서 이 표를 근거로 전략 우열을 주장하지 말 것 (표본 1개짜리 비교임).
  - 확인 포인트: 네 전략이 수렴한 설정이 서로 비슷한가, 다른가?
      · 비슷하다면 → "탐색 공간 내에 뚜렷한 최적 영역이 존재하며 전략과 무관하게 수렴" 서술 가능
      · 다르다면  → "서로 다른 국소 최적에 도달" + 그 차이가 성능 차이로 이어졌는지 4.3.1과 대조
  - Phase 2 Ablation(4.2.3/4.2.4)에서 확정한 rank=64 / target=full과 비교할 것.
    Phase 3의 자율 탐색이 같은 결론에 독립적으로 도달했는지가 흥미로운 논점이다.
-->

### 4.3.3 탐색 궤적 분석 (trial-level, 시각화 전용)

전략별 탐색 효율을 비교하기 위해 trial 진행에 따른 누적 최고 성능(anytime performance) 곡선을 제시한다. §3.7·4.3 서두에서 밝힌 대로 이 절의 trial-level 데이터는 **탐색 과정의 기술적 묘사에만 사용하며 통계 검정의 근거로 삼지 않는다.**

**생성 명령** (실행 완료 후 재생성할 것):

```bash
python3 scripts/plot_phase3_anytime.py --results_dir results/phase3_autoresearch
#   → phase3_anytime.png / .pdf        그림 (그림 4.x 후보)
#   → phase3_anytime_curve.csv         곡선 원본 수치 (trial index별 median/q1/q3/n)
#   → phase3_anytime_summary.md        최고 성능 도달 trial 요약표
```

**Table 4.3b. 전략별 최고 성능 도달 시점** *(phase3_anytime_summary.md에서 인용)*

| 전략 | 반복 수 | 최종 최고(중앙값) | 최고 도달 trial(중앙값) | 도달 trial IQR |
|------|:------:|:----------------:|:---------------------:|:-------------:|
| Manual | 10 | [TBD-36] | 1.0 | — |
| Random Search | 10 | [TBD-37] | [TBD-38] | [TBD-39] |
| Optuna (TPE) | 10 | [TBD-40] | [TBD-41] | [TBD-42] |
| Autoresearch | 10 | [TBD-43] | [TBD-44] | [TBD-45] |

<!-- 작성 지침:
  - 그림(phase3_anytime.png)과 위 표는 같은 내용을 다르게 보여준다.
    지면이 빠듯하면 표만, 여유가 있으면 그림을 본문에 + 표를 부록으로 돌릴 것.
  - 그림 사양: x축 = repeat 내 trial index(1~20), y축 = 그 시점까지의 누적 최고 val_accuracy,
    선 = 10개 반복의 중앙값, 밴드 = IQR(25~75%), Manual은 반복당 trial이 1개라 수평 기준선.
    평균 대신 중앙값+IQR을 쓴 이유는 run-level 검정이 비모수 검정이라 시각화도 맞춘 것 —
    본문에 한 줄 각주로 밝힐 것.

  관찰 포인트: Optuna·Autoresearch는 순차 최적화이므로 후반 trial에서 개선이 나타나야 정상.
  만약 곡선이 초반에 평평해진다면 → 탐색 예산 20 trial이 이미 충분했다는 근거,
  끝까지 상승 중이라면 → 예산 부족(설계 축소의 트레이드오프)이 실제로 성능을 제약했다는 근거.
  어느 쪽이든 §3.7의 "40→20 축소" 결정에 대한 사후 평가로 5.3에 연결할 것.
-->

### 4.3.4 Autoresearch 에이전트의 탐색 행태 (정성 분석)

Autoresearch 전략은 매 trial마다 이전 결과를 읽고 다음 설정을 **자연어 근거(rationale)와 함께** 제안한다. 이 rationale 로그는 다른 세 전략에는 존재하지 않는 Autoresearch 고유의 산출물로, 에이전트가 어떤 가설을 세우고 탐색 방향을 조정했는지를 관찰할 수 있다.

<!-- 작성 지침:
  - 원본: results/phase3_autoresearch/results.tsv 의 agent_reasoning 컬럼(21번째),
    그리고 각 trial 폴더의 rationale.md.
  - 부록 D에 실제 로그 발췌를 싣기로 이미 합의돼 있으므로(NEXT_SESSION.md),
    본 절은 "패턴 요약 + 대표 사례 2~3건 인용", 전문은 부록 D로 넘길 것.
  - 볼 것: (1) 에이전트가 rank/target을 키우는 방향으로 일관되게 움직였는가
           (2) 실패한 설정에서 실제로 학습했는가(같은 실수 반복 여부)
           (3) 탐색이 조기에 한 지점으로 고착(premature convergence)됐는가
  - 주의: 이 절은 정성 분석이므로 "에이전트가 ~을 이해했다" 같은 내적 상태 주장은 피하고,
    "제안 로그에서 ~한 패턴이 관찰된다"는 관찰 진술로 한정할 것.
-->

### 4.3.5 종합

<!-- 4.3.1~4.3.4를 종합해 RQ3에 대한 답을 1~2문단으로 정리.
     반드시 포함할 것:
       - RQ3에 대한 직접적 답 (4.3.1의 분기 A/B/C 중 실제 결과에 맞는 표현)
       - 본 결과의 조건 한정: 단일 모델(Qwen3-VL-2B) · 단일 데이터셋(PathVQA) ·
         전략당 20 trial 예산 · max_steps=200 통제 조건에서의 결과임
       - Autoresearch의 비용 측면 언급(LLM API 호출 비용이 추가로 발생) — 성능이 동등하다면
         비용까지 고려한 실용적 판단이 필요하다는 논의
     다음 절 연결: 4.4 종합 분석에서 Phase 1~3을 관통하는 논의로 이어짐 -->

---

## §5. 작성 전 확인이 필요한 열린 항목

아래는 초안 작성 중 발견했으나 **아직 검증하지 않은** 사항이다. 수치를 채우기 전에 확인할 것.

1. **`results.tsv`의 다중행 필드 문제 (일부 해소, 대조 필요)** — `agent_reasoning` 컬럼에 줄바꿈이 포함된 인용 문자열이 들어 있어, 행 단위 집계(`wc -l`)와 컬럼 기준 집계(`awk -F'\t' '$2=="..."'`)의 결과가 다르다. 실제로 2번째 컬럼이 빈 문자열인 행이 611개 관측된다.
   - **해소된 부분**: `analyze_phase3.py`와 신규 `plot_phase3_anytime.py`는 csv 모듈 기반의 `ExperimentTracker`로 읽으므로 인용된 줄바꿈을 정상 처리한다(2026-08-12 pod 실행에서 manual 10 / random 10×20 / optuna 10×20 반복이 정확히 집계됨을 확인).
   - **남은 확인**: `summarize_stage.py`만 `pandas.read_csv(..., on_bad_lines="skip")`를 쓴다. **이 옵션이 유효한 trial을 조용히 누락시키지 않는지** 대조가 필요하다. 확인법: `summarize_stage.py`가 보고하는 전략별 completed 수가 위 §0-1(b)의 awk 집계 및 `phase3_anytime_summary.md`의 반복 수와 일치하는지 비교.

2. **`train_time_min` 신뢰성 (§4.2.2 버그와 연결)** — §4.2.2에서 확인된 대로 `train_time_min`은 전처리 캐시를 처음 만드는 조건에서 wall-clock이 부풀려진다. Phase 3 결과 파일에는 `train_runtime_sec` 컬럼이 **없고** `train_time_min`만 있다. 다만 Phase 3는 전 trial이 동일 (모델, 데이터셋) 조합이라 캐시 생성 비용은 최초 소수 trial에만 실릴 것으로 예상된다 — **비용/효율을 §4.3에서 논할 경우 초반 trial의 이상치 여부를 먼저 확인**하고, 필요하면 해당 trial을 제외하거나 중앙값을 쓸 것.

3. **4.3.3 anytime performance 곡선 (스크립트 작성 완료)** — `scripts/plot_phase3_anytime.py` + `src/evaluate/visualize.py::plot_anytime_performance`를 신규 작성했고, 2026-08-12 pod에서 부분 데이터로 정상 동작을 확인했다(ruff 통과, 그림·CSV·요약표 3종 생성). **남은 결정**: 본문에 그림을 넣을지 표(Table 4.3b)만 넣을지 — 지도교수 확인 권장.
   - 주의: 검증 실행 시점의 수치는 autoresearch 반복 8·9가 미완료(7/20)인 상태라 **결론 근거로 쓸 수 없다.** 200/200 완료 후 반드시 재생성할 것.

---

*(초안 작성: 2026-08-12, Phase 3 autoresearch 172/200 진행 시점 — 수치 미확정 상태)*
