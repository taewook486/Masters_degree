# 논문 설계서 개정 비교: 교수님 제출본 vs v0.5 최신본

> **작성일**: 2026-05-16
> **비교 대상**:
> - **교수님 제출본**: `정보통신대학원_황태욱_석사학위논문설계서.pdf` (2026-03-22, v0.1 추정)
> - **최신본**: [THESIS_PROPOSAL_FINAL_v0.5.md](THESIS_PROPOSAL_FINAL_v0.5.md) (2026-05-16)

---

## 1. 거시적 차이 요약

| 항목 | 교수님 제출본 (PDF) | v0.5 최신본 | 변경 단계 |
|------|---------|----------|---------|
| **버전** | v0.1 추정 (2026-03-22) | v0.5 (2026-05-16) | 4회 개정 |
| **대상 모델** | **3개** | **4개** (Gemma4-E2B 추가) | v0.3 |
| **실험 환경** | RTX 5060 Ti 16GB 단독 | 로컬 + RunPod 4090 24GB 이원화 | v0.4 |
| **RQ3 비교 대상** | **Random Search** | **Optuna(TPE)** (격상) | v0.4 |
| **Phase 3 시간 통제** | 15분 고정 시간 예산 | **max_steps 고정** (confounding 해소) | v0.4 |
| **Phase 3 반복** | 5회 | **10회** (검정력 ~0.7) | v0.4 |
| **통계 분석 단위** | trial-level | **run-level** (독립성 가정 위반 해소) | v0.5 |

---

## 2. 평가 지표 비교

| 지표 | 교수님 제출본 | v0.5 최신본 |
|------|------------|----------|
| Closed-ended | Yes/No 정확도 | 동일 |
| Open-ended | **정답 토큰 매칭만** | **EM + BERTScore F1 이중 보고** (roberta-large + BioBERT) |
| 응답 시간 | ms/question | 동일 |
| VRAM | peak MB | 동일 |
| **WCA (Weighted Clinical Accuracy)** | ❌ 없음 | ✅ **신설** (PathVQA 7개 질문 유형 × 임상 가중치) |
| **ECE (Expected Calibration Error)** | ❌ 없음 | ✅ **신설** (10 bins, 모델 confidence vs 실제 정확도) |
| 훈련 시간/파라미터 비율 | 있음 | 동일 |
| **CF 측정** | "범용 성능 감소율" 단순 언급 | ✅ **이중 측정**: (A) VQAv2 2,000 subset + (B) cross-dataset 24회 추가 |

---

## 3. 통계 검증 비교

| 검증 항목 | 교수님 제출본 | v0.5 최신본 |
|---------|------------|----------|
| Phase 1 모델 비교 | ANOVA + Tukey HSD (3개 모델) | ANOVA + Tukey HSD (**4개 모델**) |
| Phase 2 Base vs FT | Paired t-test, Cohen's d, Wilcoxon | + **BCa Bootstrap 10,000 resamples** + **Mixed-Effects Model** + **Wilcoxon r** (3중 검증) |
| Phase 3 전략 비교 | Kruskal-Wallis (5회 반복, trial-level) | KW (**10회 반복, run-level**) + **Mann-Whitney U** (Autoresearch vs Optuna 쌍별) |
| Bootstrap CI | 언급만 | 명시적 적용 (BCa method) |
| n 한계 인지 | 없음 | 5.3 한계점에 명시 |

---

## 4. 데이터 오염 통제 (완전 신설)

| 측면 | 교수님 제출본 | v0.5 최신본 |
|------|------------|----------|
| 데이터 오염 인식 | ❌ 언급 자체 없음 | ✅ §4.2.1 신설 |
| 통제 방법 | - | **Min-K% Probability Attack** (Shi et al., NAACL 2024) |
| 측정 대상 | - | 4 모델 × 3 데이터셋 = 12조합, 모든 test sample |
| Sub-analysis | - | Contamination 의심 sample 제거 후 결과 재계산 |
| 코드 | - | `scripts/measure_contamination.py` (273줄) |

---

## 5. 선행 연구 비교 (Table 4.4 신설)

| 비교 모델 | 교수님 제출본 | v0.5 최신본 |
|---------|------------|----------|
| 의료 특화 VLM 직접 비교 | ❌ 없음 | ⚠️ 범위 외 (한계 명시) |
| 의료 특화 VLM 간접 비교 | ❌ 없음 | ✅ **Table 4.4 신설** |
| 비교 대상 모델 | - | LLaVA-Med-7B, Med-Flamingo, CheXagent, BioViL-T |
| 비교 metric | - | PathVQA Open/Closed, SLAKE Open, VQA-RAD Open |

---

## 6. Phase 3 자율 HPO 상세 변경

| 항목 | 교수님 제출본 | v0.5 최신본 |
|------|------------|----------|
| 학습 단위 | **epochs** {1, 2, 3, 5} | **max_steps** {100, 200, 400, 800} |
| 시간 통제 | 15분 고정 시간 예산 | max_steps 고정 (data throughput 차이는 별도 보고) |
| 보고 추가 항목 | - | effective_batch_size, total_samples_seen, wall_clock_time |
| 자율 탐색 로그 | "변경 근거 기록" 단순 언급 | rationale.md + 자연어 탐색 근거 + Git commit 추적 |
| LLM 비결정성 통제 | ❌ 없음 | temperature=0, top_p=1, 모델 ID + 스냅샷 날짜 고정, API 응답 로깅 |
| RQ3 차별점 | "효율적인가?" | "**경쟁적 성능 + 해석 가능한 탐색 근거**" |
| 총 trial 수 | ~120 (3전략×40) | **1,210** (4전략 × 10반복 × 40 + Manual 10) |

---

## 7. 논문 구조 변경

| 절 | 교수님 제출본 | v0.5 최신본 |
|----|------------|----------|
| 2.3.3 기존 연구 성과 | 일반 언급 | **LLaVA-Med, Med-Flamingo, CheXagent 등 의료 특화 VLM 명시** |
| 2.4.2 LLM 에이전트 HPO | 일반 언급 | **이론적 차별점: cross-domain transfer, 탐색 공간 구조 이해, Hyperband 인용** |
| 3.8 평가 지표 | 단일 절 | **3.8.1~3.8.5 5개 서브 절** (BERTScore 이중, CF 이중, WCA+ECE, Robust 통계, Min-K%) |
| 4.4.5 임상적 의미 분석 | ❌ 없음 | ✅ **신설** |
| 4.3.5 Autoresearch 해석 가능성 분석 | ❌ 없음 | ✅ **신설** |
| 5.3 한계점 | 일반 언급 | **7개 항목 명시** (데이터 오염, 의료 특화 비교, confounding, LLM 비결정성, Ablation 일반화, 통계 검정력, WCA 임시 가중치) |
| 부록 D | 없음 | ✅ **Autoresearch 탐색 근거 로그 (자연어 rationale)** |

---

## 8. 재현성 보장 항목 비교

| 항목 | 교수님 제출본 | v0.5 최신본 |
|------|------------|----------|
| 코드 관리 | GitHub | 동일 |
| 환경 재현 | pyproject.toml + CUDA | 동일 |
| 실험 추적 | WandB + results.tsv | 동일 |
| 랜덤 시드 | 42, 123, 456 | 동일 |
| Git 커밋 | 각 실험 설정 추적 | 동일 |
| **LLM 비결정성 통제** | ❌ 없음 | ✅ **신설** (temp=0, 모델 버전 고정) |
| **API 응답 로깅** | ❌ 없음 | ✅ **신설** (실험별 JSON 저장) |
| **변동성 흡수** | 3회 반복 | **Phase 3는 10회로 확장** |

---

## 9. 실험 규모 변화

| 항목 | 교수님 제출본 | v0.5 최신본 |
|------|------------|----------|
| **Phase 1 조건 수** | 27 (3 모델 × 3 데이터셋 × 3 시드) | **36** (4 모델 × 3 × 3) |
| **Phase 2 조건 수** | 27 | **36** + cross-dataset CF 24회 추가 |
| **Phase 3 trial 수** | ~120 (3 전략 × 40) | **1,210** (4 전략 × 10반복 × 40 + Manual) |
| **Phase 1.5 (신설)** | - | **12** (4 모델 × 3 데이터셋 Min-K%) |
| **총 GPU 시간 추정** | 미산정 | **~250시간** ([RUNPOD_COST_ESTIMATE.md](RUNPOD_COST_ESTIMATE.md) 참조) |

---

## 10. 변경 트리거 (왜 바뀌었나?)

| 단계 | 트리거 | 주요 변경 |
|------|-------|---------|
| v0.1→v0.2 | 1차 리뷰 피드백 | Florence-2 탈락 (3개 모델), BERTScore 추가, max_steps 명시 |
| v0.2→v0.3 | Gemma 4 E2B 출시 | 4번째 모델 추가, transformers 5.5.0 |
| v0.3→v0.4 | **동료 심사 의견서 (Major Revision)** | RQ3 Optuna 격상, Phase 3 반복 10회, 하드웨어 이원화, BioBERT, cross-dataset CF, max_steps 고정 |
| v0.4→v0.5 | **잔여 치명적 4건 처리** | Run-level 통계, BCa Bootstrap, Min-K%, WCA+ECE, 의료 특화 간접 비교 |

---

## 11. 한 줄 요약

> 교수님께 보낸 설계서는 **"3개 모델로 9개 조건 + 단순 정확도 + Random Search 비교"**의 기본 설계였다면, v0.5는 **"4개 모델로 36개 조건 + BERTScore/WCA/ECE 다층 평가 + Optuna 격상 비교 + Min-K% 오염 통제 + 3중 통계 검증"**으로 **방법론적 엄밀성이 크게 향상**되었습니다.

---

## 12. 교수님 보고 시 강조 포인트 (권장)

지도교수 면담 시 v0.1 → v0.5의 발전 과정을 다음 순서로 설명하는 것을 권장합니다:

1. **모델 추가 (v0.3)**: "Gemma 4 E2B가 4월 초 출시되어 4번째 평가 모델로 추가했습니다."
2. **동료 심사 반영 (v0.4)**: "동료 심사에서 9가지 지적을 받았고, 그중 5개를 v0.4에서 처리했습니다."
3. **잔여 4건 처리 (v0.5)**: "남은 통계·오염·임상 의미 관련 지적을 v0.5에서 모두 처리하여 코드 모듈도 구현 완료했습니다."
4. **실험 비용 산정 (v0.5)**: "RunPod 기준 단계별 비용을 산정하여 첫 단계 ~$5로 핵심 결과를 확보 가능합니다."

이로써 단순한 설계 변경이 아니라 **체계적 피드백 반영 과정**으로 보일 수 있습니다.

---

## 참조 문서

- [THESIS_PROPOSAL_FINAL_v0.5.md](THESIS_PROPOSAL_FINAL_v0.5.md): v0.5 최신 설계서
- [동료_심사_의견서.md](동료_심사_의견서.md): 9개 지적사항 원본
- [REVIEW_FEEDBACK.md](REVIEW_FEEDBACK.md): 비판적 검토
- [RUNPOD_COST_ESTIMATE.md](RUNPOD_COST_ESTIMATE.md): 실행 비용 산정

---

*최종 업데이트: 2026-05-16*
