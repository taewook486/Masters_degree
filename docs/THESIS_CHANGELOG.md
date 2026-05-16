# 논문 설계서 변경 이력 (Changelog)

> **연구 주제**: 경량 멀티모달 모델의 의료 영상 VQA 도메인 적응
> **저자**: 황태욱 (건국대학교 정보통신대학원 융합정보기술학과 인공지능전공)

본 문서는 논문 설계서의 모든 버전 변경 사항을 추적합니다. 본문 PDF에는 포함되지 않으며, 연구자 내부 작업 추적 및 향후 review 대응 자료로 사용됩니다.

---

## 버전 관리 체계 (Dual Versioning)

| 구분 | 형식 | 용도 |
|------|------|------|
| **Internal** | v0.X | git 커밋 단위 작업 추적, 마크다운 파일 |
| **External** | v1.X, v2.X | 교수님/심사위원 제출 단위, PDF |

### 버전 매핑

| Internal | External | 시점 | 트리거 |
|---------|----------|------|--------|
| v0.1 | **v1.0** | 2026-03-22 | 첫 교수 제출 (완료) |
| v0.2 | (보고 안 함) | 2026-03-24 | 동료 심사 1차 반영 (BERTScore, max_steps) |
| v0.3 | (보고 안 함) | 2026-04-05 | Gemma 4 E2B 추가 |
| v0.4 | (보고 안 함) | 2026-05-15 | 동료 심사 9건 중 5건 처리 |
| v0.5 | **v1.1** | 2026-05-16 | 잔여 치명적 4건 처리 |
| (예정) | **v2.0** | 2026-07 예상 | Phase 1/2 실험 결과 포함 본 심사용 |

### 시맨틱 버전 규칙 (External)

- **MAJOR (v1, v2, ...)**: 연구 방향, RQ, 또는 큰 결과 추가 (예: Phase 1 실험 결과 포함)
- **MINOR (v1.1, v1.2)**: 방법론 추가/개선 (예: 동료 심사 반영, 모델 추가)
- **PATCH (v1.1.1)**: 오타, 표 형식 수정

---

## v1.1 (Internal v0.5) — 2026-05-16

**잔여 치명적 지적사항 5가지 처리** (동료 심사 의견서의 미해결 항목).

### 추가 / 강화

#### 통계 분석
- **Run-level 분석 도입** — Sequential optimization(Autoresearch/Optuna)의 trial 간 의존성을 인정하여, 각 전략 × 10회 독립 반복을 독립 관측치로 사용. Kruskal-Wallis와 Mann-Whitney U 검정을 **run-level**에만 적용
- **BCa Bootstrap 95% CI** — Phase 2 paired Cohen's d의 robust 추정 (n=9 한계 보완)
- **Mixed-Effects Model** — `accuracy ~ condition + (1|seed) + (1|dataset)` 모형으로 seed/dataset random effect 처리
- **Wilcoxon r 효과 크기** — z / √n 추가 보고

#### 데이터 오염 통제
- **Min-K% Probability Attack** (Shi et al., NAACL 2024) 능동적 측정 (§4.2.1)
- 4 모델 × 3 데이터셋 = 12조합 contamination score 산출
- Suspected sample 제거 후 결과 재계산 sub-analysis 추가

#### 임상적 의미
- **WCA (Weighted Clinical Accuracy)** — PathVQA 7개 질문 유형(Where/What/Why/How/...) × 임상 중요도 가중치 (§4.4.5)
- **ECE (Expected Calibration Error)** — 모델 confidence vs 실제 정확도 (10 bins)

#### 선행 연구 비교
- **Table 4.4 신설** — LLaVA-Med-7B, Med-Flamingo, CheXagent, BioViL-T 등 의료 특화 VLM 수치와 간접 비교

### 코드 산출물

- `src/evaluate/robust_statistics.py` (316줄) — BCa Bootstrap + Mixed-Effects + Wilcoxon r
- `src/evaluate/clinical_significance.py` (237줄) — WCA + ECE
- `scripts/measure_contamination.py` (273줄) — Min-K% Probability 측정

### 한계점 추가 (§5.3)

- 통계적 검정력의 한계 (Phase 2 n=9, Phase 3 run-level n=10)
- WCA 가중치의 임시성 (임상 의사 검증 필요)
- Min-K% Probability의 간접 indicator 한계

---

## v0.4 — 2026-05-15

**동료 심사 의견서 1차 반영** (9건 중 5건).

### 변경

- **RQ3 귀무가설 격상**: `H0: Autoresearch = Random Search` → `H0: Autoresearch = Optuna(TPE)`
  - LLM HPO의 차별점을 (1) 자연어 탐색 근거, (2) cross-domain transfer, (3) 추적 가능성으로 재정의
  - Random Search는 하한선(lower bound)으로만 유지
- **Phase 3 반복 횟수**: 5회 → **10회** (통계 검정력 ~0.6-0.7 확보)
- **Phase 3 시간 통제**: 15분 고정 시간 예산 → **max_steps 고정** (200 steps)
  - confounding 해소: effective_batch_size, total_samples_seen, wall_clock_time 별도 보고
- **하드웨어 이원화**: 로컬 RTX 5060 Ti 단독 → 로컬 + RunPod RTX 4090 (재현성 검증용 + 주 실험용)
- **BERTScore 이중 보고**: roberta-large (범용) + **BioBERT** (의료 특화) 병기
- **CF 측정 이중 구조**: (A) VQAv2 subset + **(B) cross-dataset 일반화** (24회 추가)
- **LLM 비결정성 통제**: temperature=0, top_p=1, 모델 ID + 스냅샷 날짜 고정, API 응답 로깅

### 한계점 명시 (§5.3)

- 데이터 오염 가능성 (능동적 통제는 v0.5에서 추가)
- 의료 특화 VLM 직접 비교 부재
- Phase 3 confounding
- LLM 비결정성

---

## v0.3 — 2026-04-05

**Gemma 4 E2B 모델 추가**.

### 변경

- **대상 모델 3개 → 4개**: Gemma4-E2B 추가
  - PLE(Per-Layer Embeddings) 아키텍처, 2.3B active / 5.1B total
  - Apache 2.0 라이선스
  - 제로샷 VRAM ~10.3GB 측정
- **실험 조건 수**: 9 → **12** (4 모델 × 3 데이터셋)
- **transformers 5.5.0 업그레이드** (Gemma4ForConditionalGeneration 지원)

---

## v0.2 — 2026-03-24

**1차 리뷰 피드백 반영**.

### 변경

- **Florence-2-large 탈락** (3개 모델로 축소)
  - transformers 5.x에서 SA causal mask 캐시 비정상 동작
- **Phase 2 평가 지표 정교화**:
  - Open-ended accuracy에 **BERTScore F1** 추가 (roberta-large, threshold >= 0.7)
  - "cancer" vs "malignancy" 등 의미 동등 응답 평가
- **CF 대조군 구체화**: "MMMU 등" → **VQAv2 validation subset (2,000 샘플)**
- **Phase 3 탐색 공간**: epochs → **max_steps**로 전환
- **Phase 3 시간 예산 정의 명확화**: 15분 = 순수 GPU 학습 시간 (모델 로드/평가 제외)

### 신규 위험 추가

- BERTScore 모델 VRAM 충돌 가능성
- VQAv2 데이터셋 추가 준비 부담
- Phase 3 max_steps 기반 비교 공정성

---

## v0.1 — 2026-03-22

**교수 제출용 초안 (External v1.0)**.

### 초기 설계

- **3개 모델**: Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B
- **3개 데이터셋**: PathVQA, SLAKE, VQA-RAD
- **3개 시드**: 42, 123, 456
- **Phase 1** (RQ1): 제로샷 베이스라인 평가 (9개 조건)
- **Phase 2** (RQ2): QLoRA 파인튜닝 + 3종 Ablation Study
- **Phase 3** (RQ3): 자율 HPO 4전략 비교 (Manual / Random / Optuna / Autoresearch)
- **실험 환경**: RTX 5060 Ti 16GB 단독
- **평가 지표**: Closed/Open Accuracy, 응답 시간, VRAM
- **통계 검증**: ANOVA + Tukey HSD, Paired t-test + Cohen's d, Kruskal-Wallis

---

## 향후 계획 (v2.0 예정)

Phase 1/2/3 실험 완료 후 본 심사용 버전(v2.0) 작성 예정:

- Phase 1 결과 (BERTScore, WCA, ECE 포함)
- Phase 1.5 Min-K% Probability contamination 분석
- Phase 2 결과 (BCa Bootstrap CI, Mixed-Effects, cross-dataset CF)
- Phase 3 결과 (4 HPO 전략 × 10 run-level 비교)
- Table 4.4 의료 특화 VLM 간접 비교 (실측값 채움)
- 본 연구의 한계 및 향후 연구 방향 (실증 결과 기반)

---

## 참조

- 본문 PDF (current): [v1.1](submitted/황태욱_석사학위논문설계서_v1.1_2026-05-16.pdf)
- 본문 마크다운 (current): [THESIS_PROPOSAL_FINAL_v0.5.md](THESIS_PROPOSAL_FINAL_v0.5.md)
- 동료 심사 의견서: [동료_심사_의견서.md](동료_심사_의견서.md)
- 비판적 검토: [REVIEW_FEEDBACK.md](REVIEW_FEEDBACK.md)
- 변경사항 상세 비교: [THESIS_REVISION_COMPARISON.md](THESIS_REVISION_COMPARISON.md)

---

*최종 업데이트: 2026-05-16*
