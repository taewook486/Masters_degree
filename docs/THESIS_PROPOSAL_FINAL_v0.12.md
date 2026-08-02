# 석사학위 논문 설계서

<!-- pdf:strip-meta -->
> 본문 버전: Internal v0.12 (2026-08-02, Phase 3 trials_per_repeat 40→20 축소 + 실행 규모 재산출 반영) / 직전 External v1.1
> 변경 이력은 [THESIS_CHANGELOG.md](THESIS_CHANGELOG.md) 참조
<!-- /pdf:strip-meta -->

## 논문 정보

- **대학교**: 건국대학교 정보통신대학원 융합정보기술학과 인공지능전공
- **제출 목표**: 2026년 9월
- **실험 환경**:
  - **로컬**: RTX 5060 Ti (16GB VRAM), Ryzen 5 5600X, RAM 32GB — 16GB 재현성 검증용
  - **클라우드**: RunPod RTX 4090 (24GB VRAM) — 주 실험 환경
  - **클라우드(대안)**: 24GB 단일 GPU 자원 확보가 어려운 시점에는 16GB GPU 2장(예: 4080 Super ×2) 멀티-GPU pod로 대체 진행. Phase 2는 조건(model×dataset×seed)을 GPU 개수만큼 동시 배정해 GPU 1장씩 독립 실행하는 방식(`--max_parallel`, `src/finetune/run_phase2.py`)으로 검증됨 — 상세는 `docs/RUNPOD_GUIDE.md` §4.0 참조

---

## 1. 제목

**한국어**: 경량 멀티모달 모델의 의료 영상 VQA 도메인 적응: QLoRA 파인튜닝과 자율 하이퍼파라미터 최적화

**영문**: Domain Adaptation of Lightweight Vision-Language Models for Medical Visual Question Answering: QLoRA Fine-Tuning with Autonomous Hyperparameter Optimization

---

## 2. 연구 배경 및 목적

### 2.1 연구 배경

- Vision-Language Model(VLM)의 급속한 발전 (GPT-4V, Gemini 등)
- 의료 영상 분석에서 VLM 활용 가능성 증대
- 범용 VLM의 도메인 특화 성능 한계 (의학 용어, 전문 지식 부족)
- 대규모 GPU 없이도 QLoRA를 통한 효율적 파인튜닝 가능
- 하이퍼파라미터 선택의 자동화는 VLM PEFT의 미해결 과제 (NeurIPS 2024 서베이)

### 2.2 연구 목적

1. 소비자 GPU(16-24GB) 환경에서 경량 VLM의 의료 VQA 도메인 적응 가능성 실증
2. QLoRA 파인튜닝의 하이퍼파라미터가 성능에 미치는 영향 체계적 분석
3. autoresearch 스타일 자율 실험 루프를 통한 하이퍼파라미터 최적화 방법론 제안

---

## 3. 연구 질문

| # | 연구 질문 | 귀무가설 |
|---|----------|----------|
| RQ1 | 경량 VLM(2-3B)의 의료 VQA 제로샷 성능은 모델별로 유의미한 차이가 있는가? | H0: 모델 간 VQA 정확도 차이 없음 |
| RQ2 | QLoRA 파인튜닝이 의료 VQA 성능을 유의미하게 향상시키는가? | H0: Base = Fine-tuned 성능 |
| RQ3 | LLM 에이전트 기반 자율 하이퍼파라미터 탐색이 베이지안 최적화(Optuna TPE)와 경쟁적 성능을 달성하면서 해석 가능한 탐색 근거를 제공하는가? | H0: Autoresearch = Optuna (TPE) |

> **v0.4 변경 (RQ3)**: 귀무가설을 `= Random Search`에서 `= Optuna(TPE)`로 격상. LLM 기반 HPO의 차별점은 성능 경쟁력뿐 아니라, (1) 자연어 기반 탐색 근거 설명 가능성, (2) 사전 지식(cross-domain transfer)을 활용한 탐색 공간 구조 이해, (3) 설정 변경 이유의 추적 가능성에 있다. Random Search는 하한선(lower bound)으로만 유지한다.

---

## 4. 실험 설계

### 4.1 대상 모델 (4개)

| 모델 | 파라미터 | 아키텍처 특징 | 예상 QLoRA VRAM |
|------|---------|-------------|:---:|
| Qwen3-VL-2B | 2B | Thinking mode, DeepStack | ~8-10 GB |
| Qwen2.5-VL-3B | 3B | Dynamic Resolution, 19개 언어 OCR | ~8-10 GB |
| SmolVLM-2.2B | 2.2B | HuggingFace 경량 VLM | ~8-10 GB |
| Gemma4-E2B | 2.3B (active) / 5.1B (total) | PLE(Per-Layer Embeddings), Apache 2.0 | ~12-14 GB |

> **v0.3 변경**: Gemma 4 E2B 추가. PLE 기술로 2.3B active 파라미터만으로 5.1B급 표현력 제공. 제로샷 VRAM ~10.3GB 확인.

**선정 기준**:
- 16GB VRAM에서 QLoRA 파인튜닝 가능
- Apache 2.0 또는 MIT 라이선스 (연구 활용 자유)
- 충분한 커뮤니티/프레임워크 지원

### 4.2 데이터셋 (3개)

| 데이터셋 | 이미지 수 | QA 쌍 | 언어 | 도메인 | 질문 유형 |
|----------|:---:|:---:|:---:|:---:|:---:|
| PathVQA | 4,998 | 32,799 | 영어 | 병리학 | Open+Closed (7종) |
| SLAKE | 642 | 14,028 | 영어+중국어 | 방사선/CT | Open+Closed |
| VQA-RAD | 315 | 2,248 | 영어 | 방사선 | Open+Closed |

**데이터 분할**: 각 데이터셋의 공식 train/val/test split 사용

### 4.2.1 데이터 오염 통제 절차 (v0.5 신설)

PathVQA(2018), SLAKE(2021), VQA-RAD(2018)는 본 연구의 대상 모델(Qwen3-VL 2025, Qwen2.5-VL 2025, SmolVLM2 2025, Gemma4 2026) 사전훈련 시점 이전에 공개되었으므로, **사전훈련 데이터 오염 가능성을 배제할 수 없다**. 본 연구는 이를 능동적으로 측정한다.

**Min-K% Probability Attack (Shi et al., ICLR 2024)**:

각 sample의 정답 텍스트에 대해 모델의 token-level log-probability를 계산하고, 하위 K%(K=20) token의 평균 log-probability를 contamination indicator로 사용한다.

수식: `MinK_score(x) = mean({log P(x_i | x_<i) : x_i in bottom-K% by log-prob})`

이론: 사전훈련 데이터에 포함된 sample은 비훈련 sample 대비 평균적으로 높은 token 확률을 보임 → MinK_score가 높음.

**실험 절차**:
1. 4개 모델 × 3개 데이터셋의 모든 test sample에 MinK_score 계산
2. Calibration set(공개 데이터셋 무작위 샘플 1,000개, 본 데이터셋과 유사한 도메인)으로 threshold T 결정
3. MinK_score > T인 sample을 "contamination 의심" 분류
4. Sub-analysis: 의심 sample 제거 후 Phase 1/2 결과 재계산

**해석 시나리오**:
- Clean subset vs 전체 결과 차이 < 1%p: 본 연구 결론 robust
- 차이 1-5%p: 한계점에 명시, clean subset 기준 결과를 보조 보고
- 차이 > 5%p: 결론 재검토, 의심 sample 제거 결과를 primary로 보고

**구현**: `scripts/measure_contamination.py` (별도 모듈, Phase 1 결과 산출 후 실행)

> **v0.5 변경**: 동료 심사 IV번 지적(데이터 오염 완전 무시)에 대응. "전후 비교 상쇄" 가정에 의존하지 않고, Min-K% Probability로 contamination 정도를 정량화하여 본 연구 결론의 robustness를 검증한다.

### 4.3 Phase 1: 베이스라인 평가 (RQ1)

**목적**: 파인튜닝 전 각 모델의 의료 VQA 제로샷 성능 측정

**실험 조건**:
- 4개 모델 x 3개 데이터셋 = 12개 조건
- 단일 결정적 평가 (seed 42, greedy 디코딩)
- 동일 프롬프트 템플릿 사용

> **v0.6 변경 (방법론 정정)**: zero-shot은 greedy 디코딩이라 **결정적**이므로 시드를 바꿔도 결과가 동일하다(seed-std ≡ 0, ANOVA 그룹내 분산=0으로 degenerate). 따라서 (1) 3시드 반복 → **단일 시드(42)**, (2) 불확실성은 각 조건 per-sample 정오의 **부트스트랩 95% CI**로 보고, (3) 모델 비교(RQ1)는 시드-분산 ANOVA → **공유 테스트셋 짝지은 검정(Cochran's Q + McNemar)**으로 대체한다. 4개 모델이 동일 샘플로 평가되므로 짝지은 검정이 더 적절하고 검정력도 높다. Phase 2/3은 학습이 확률적이므로 다중 시드/반복을 유지한다.

**측정 지표**:
- Closed-ended accuracy (Yes/No, 선택형)
- Open-ended accuracy (정답 토큰 매칭) + BERTScore F1 (roberta-large)
- 각 정확도 지표의 **부트스트랩 95% CI** (테스트셋 표본 변동 기반)
- 응답 시간 (ms/question), VRAM 사용량 (peak MB)

**통계 검증 (v0.6 정정)**:
- Cochran's Q (4개 모델 공유 테스트셋 이진 정오, H0: 정확도 동일)
- 사후 검증: McNemar 쌍별 검정 (Bonferroni 보정)
- 효과 추정: 정확도 및 모델 간 차이의 부트스트랩 95% CI
- 유의수준: alpha = 0.05

### 4.4 Phase 2: QLoRA 파인튜닝 (RQ2)

**목적**: 도메인 특화 파인튜닝의 효과 측정

**기본 QLoRA 설정**:

| 파라미터 | 값 |
|----------|-----|
| Quantization | NF4 (4-bit NormalFloat) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, v_proj |
| Learning Rate | 2e-4 |
| Batch Size | 1 (gradient accumulation 8, effective batch = 8) |
| 학습 예산 | 목표 3 epochs, 단 RunPod RTX 4090 시간·비용 제약으로 `max_steps=500` **cap** 적용 (조건당 samples_seen = 500 × 8 = 4,000 고정). 데이터셋 크기와 무관하게 학습량이 고정되므로, 소형 VQA-RAD는 약 2 epoch 이상, 중형 SLAKE·대형 PathVQA는 1 epoch 미만만 학습된다. 조건별 실효 `samples_seen`·`num_train_epochs`(환산값)·`wall_clock`을 결과에 투명 보고. 상세 한계는 §5.3 참조 |
| Optimizer | paged_adamw_8bit |

**실험 조건**:
- 4개 모델 x 3개 데이터셋 = 12개 조건
- 각 조건 3회 반복 (seed 변경: 42, 123, 456)

**Ablation Study A - 데이터 크기 영향**:
- PathVQA 기준, 최적 모델 1개 선택
- 훈련 데이터 비율: 5%, 10%, 25%, 50%, 100%
- 학습 곡선(learning curve) 분석

**Ablation Study B - LoRA Rank 영향**:
- PathVQA 기준, 최적 모델 1개 선택
- LoRA rank: 4, 8, 16, 32, 64
- 성능 vs 파라미터 효율성 분석

**Ablation Study C - Target Module 영향**:
- [q_proj, v_proj] vs [q_proj, k_proj, v_proj, o_proj] vs [all_linear]
- 성능 vs 학습 시간 트레이드오프

**측정 지표** (Phase 1과 동일 + 추가):
- Open-ended accuracy: Exact Match + BERTScore F1 이중 보고
  - 범용 기준: roberta-large (threshold >= 0.7) — 선행 연구 비교용
  - 의료 특화 기준: BioBERT (dmis-lab/biobert-v1.1) — 의료 용어 민감도 향상
  - 두 지표 간 상관관계 분석 보고
  - **primary 지표 규칙**: 정확도·통계 검정의 유일 결정 지표는 범용 모델(roberta-large @ 0.7)이며, BioBERT F1은 보조 지표로만 병기하고 정오 판정에 관여하지 않는다(이중 게이팅 없음).
- 훈련 시간 (총 분)
- 훈련 가능 파라미터 비율 (%)
- Peak VRAM 사용량 (MB)
- Catastrophic Forgetting 측정 (아래 상세)

**Catastrophic Forgetting 측정 방법**:

(A) 범용 VQA 성능 변화:
- 대조군: VQAv2 validation subset (2,000 샘플, 균형 샘플링)
- 측정 시점: 파인튜닝 전(Base) 1회 + 파인튜닝 후(QLoRA) 1회
- 지표: 범용 VQA 정확도 감소율 = (Base - Fine-tuned) / Base x 100%
- 적용 범위: 12개 조건(4모델 x 3데이터셋) 각각에 대해 측정

(B) 의료 도메인 내 cross-dataset 일반화:
- 훈련 데이터셋 ≠ 평가 데이터셋 조합으로 일반화 능력 측정
- 예: PathVQA로 훈련 → SLAKE, VQA-RAD로 평가 (각 조합 측정)
- 지표: cross-dataset 정확도 변화율
- 적용 범위: 12개 조건 x 2개 cross-dataset = 24회 추가 평가

> **v0.4 변경**: CF 측정을 (A) 범용 VQA + (B) 의료 cross-dataset 이중 구조로 확장. 의료 도메인 내 일반화 능력을 직접 측정하여 CF의 실질적 영향을 파악한다.

**통계 검증 (v0.5 강화)**:

n=9 (3 seeds × 3 datasets)의 한계를 인지하므로, 3가지 방법을 병행하여 robustness를 검증한다:

- **Primary**: Paired t-test + Cohen's d (관행 비교)
- **Robust**: BCa Bootstrap (10,000 resamples) → 95% CI for Cohen's d
- **Random effects**: Mixed-Effects Model
  - 모델: `accuracy ~ condition + (1|seed) + (1|dataset)`
  - Library: statsmodels.formula.api.mixedlm 또는 R lme4
  - Fixed effect의 t-value, p-value, ICC 보고
- **Non-parametric**: Wilcoxon signed-rank test + 효과 크기 r (z / √n)

해석 가이드: t-test와 mixed-effects의 결과가 일치하면 robust한 결론. 불일치 시 BCa CI를 기준으로 판단.

> **v0.5 변경**: BCa Bootstrap CI 및 Mixed-Effects Model 추가. 동료 심사 II-3 지적(n=9 Cohen's d 추정 한계)에 대응하여 3가지 통계 방법 병행.

### 4.4.5 임상적 의미 분석 (v0.5 신설)

단순 accuracy로는 의료 AI의 가치를 포착할 수 없다. 본 연구는 두 가지 보조 지표를 추가 보고한다.

**(1) Weighted Clinical Accuracy (WCA)**

PathVQA는 7개 질문 유형(Where, What, Why, How, How much/many, When, Yes/no) 라벨을 제공한다. 임상 중요도에 따라 가중치를 부여하여 WCA를 산출한다:

| 유형 | 임상 중요도 | 가중치 | 근거 |
|------|----------|------|------|
| Diagnosis (Why, What is wrong) | High | 1.0 | 진단 오류는 치료 방향 결정에 직접 영향 |
| Location (Where) | Medium-High | 0.8 | 병변 위치 식별은 후속 검사 결정 |
| Measurement (How much/many) | Medium | 0.7 | 정량 측정은 보조적이나 중요 |
| Description (What, How) | Medium | 0.6 | 소견 기술은 상세 정보 제공 |
| Temporal (When) | Low-Medium | 0.5 | 시간 정보는 부수적 |
| Yes/No | Baseline | 0.5 | 이진 판단의 정보량 제한 |

수식: `WCA = Σ (유형별 정확도 × 가중치) / Σ 가중치`

**(2) Expected Calibration Error (ECE)**

의료 AI에서는 "모델의 자신감이 실제 정확도와 일치하는지"가 accuracy보다 중요한 경우가 많다 (Guo et al., ICML 2017).

수식: `ECE = Σ (|B_m|/n) × |acc(B_m) - conf(B_m)|`

- M = 10 bins (confidence 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
- B_m: bin m에 속하는 sample 집합
- 낮을수록 calibration이 좋음 (이상적 값: 0)

**임상적 유의미성 임계점**:
- 선행 연구 인용: FDA AI/ML SaMD guidance (2021), Topol (Nature Medicine 2019), Liu et al. (Lancet Digital Health 2019)
- 단순 비교: 5%p accuracy 향상 → 1,000명 진단 시 50명 추가 정확 분류
- WCA 기준: high-criticality 유형에서의 향상이 핵심 기여
- ECE 기준: 0.05 이하 권장 (well-calibrated)

> **v0.5 변경**: 동료 심사 III 지적(평가 지표 부적절, 임상적 의미 부재)에 대응하여 WCA와 ECE를 보조 지표로 추가. PathVQA의 7개 질문 유형 라벨을 활용하여 임상 중요도 가중 정확도 산출.

### 4.5 Phase 3: 자율 하이퍼파라미터 최적화 (RQ3)

**목적**: autoresearch 패턴 기반 자율 HPO의 효과 검증 — Optuna(TPE) 대비 경쟁적 성능 및 해석 가능성 평가

**탐색 공간**:

| 파라미터 | 탐색 범위 | 타입 |
|----------|----------|------|
| lora_rank | {4, 8, 16, 32, 64} | 이산 |
| lora_alpha | rank x {1, 2, 4} | 이산 |
| learning_rate | [1e-5, 5e-4] | 연속 (로그스케일) |
| batch_size | {1, 2, 4} | 이산 |
| grad_accum_steps | {4, 8, 16} | 이산 |
| warmup_ratio | [0.0, 0.1] | 연속 |
| weight_decay | [0.0, 0.1] | 연속 |
| lora_targets | {minimal, medium, full} | 범주형 |

**비교 대상 (4가지 HPO 전략)**:

| 전략 | 설명 | trial 수 |
|------|------|:---:|
| Manual | 연구자가 직접 설정한 기본값 | 1 |
| Random Search | 탐색 공간에서 무작위 샘플링 | ~20 |
| Optuna (TPE) | 베이지안 최적화 (Tree-structured Parzen Estimator) | ~20 |
| Autoresearch | LLM 에이전트 기반 자율 탐색 | ~20 |

**자율 탐색 루프 (autoresearch 스타일)**:
1. 에이전트가 이전 실험 결과(results.tsv) 읽기
2. 다음 실험 설정 제안 + 자연어로 변경 근거 기록 (config.yaml + rationale.md)
3. Git commit
4. 고정 max_steps 학습 실행
5. 검증 세트 평가 -> VQA accuracy 기록
6. 성능 향상 시 keep, 그렇지 않으면 discard
7. 반복 (20회 또는 하룻밤)

**고정 조건**:
- 동일 모델 (Phase 2에서 최적 모델 1개)
- 동일 데이터셋 (PathVQA)
- 고정 학습량: max_steps 고정 (200 steps, 전 trial 공통) — trial 간 동일 학습 step 수 보장. 구현 상 학습 콜백에 안전장치용 wall-clock 상한(`time_budget_min`)도 함께 걸려 있으나, 이는 비정상적으로 느린 조합에 대한 안전장치일 뿐 실험 통제 변수가 아니며, max_steps=200이 모든 하이퍼파라미터 조합에서 안전장치보다 먼저 도달하도록 충분히 넉넉하게(90분) 설정한다.
- 동일 검증 세트

**예상 실험 규모 (v0.12 재산출)**:
- 총 trial 수: Manual 10회 + RS 200회(20x10) + Optuna 200회(20x10) + Autoresearch 200회(20x10) = 610회 (원안 1,210회에서 축소)
- trial당 소요 실측: 2026-08-01 로컬 듀얼 GPU(RTX 5060 Ti + RTX 4060) 스모크(전략당 1trial)로 재검증 완료. train_time_min 기준 약 25-65분(전략·GPU별 편차)이며, 여기에 validation eval(~12-13분) + 최종 test eval(~29-40분, `--max_test_samples 500` 적용 전 1,680샘플 기준)이 wall-clock에 추가로 더해짐 — 초기 추정치(~10분)는 eval 시간을 누락한 과소추정이었음을 확인.
- 총 소요시간(wall-clock) 재산출: repeats=10(run-level 통계 검정 단위, 불변) + trials_per_repeat=20(원안 40에서 축소) + `--max_test_samples 500` 적용 + 로컬 듀얼 GPU 병렬(`--max_parallel 2`) 기준 **약 12.8일** 소요로 재확정됨 — 원안의 "RunPod RTX 4090 단일 GPU, 약 8-9일" 추정치를 대체한다. trials_per_repeat 축소는 repeats(§4.5 통계 검증 참조, 통계적 검정력의 기준 단위)를 훼손하지 않는 범위에서 남은 유일한 안전 축소 레버로 적용했으며, 전략당 탐색 가능한 하이퍼파라미터 조합 수가 절반으로 줄어 최적값을 못 찾을 위험이 있다는 트레이드오프가 있음(§5.3 한계점에 반영).

> **v0.4 변경**: "15분 고정 시간 예산"에서 "max_steps 고정"으로 변경. batch_size × grad_accum 조합에 따라 동일 시간 내 처리 데이터량이 달라지는 confounding을 해소. effective_batch_size(= batch × grad_accum)와 total_samples_seen(= steps × effective_batch), wall_clock_time을 별도 보고하여 데이터 처리량 차이를 투명하게 공개한다.

**측정 지표**:
- 최종 최적 VQA accuracy
- 최적 도달까지 실험 횟수
- 총 소요 시간 (wall-clock)
- 탐색 효율성 (최적 성능 / 총 실험 수)
- 탐색 궤적 시각화 (실험 번호 vs accuracy 그래프)
- effective_batch_size, total_samples_seen 보고
- Autoresearch: 각 trial별 자연어 탐색 근거 로그

**통계 검증 (Run-level 분석)**:

본 연구는 "trial-level"이 아닌 **"run-level"**에서만 비교 통계 검정을 수행한다. Autoresearch와 Optuna는 sequential optimization 특성상 동일 run 내 trial 간 의존성이 존재하므로(trial t의 결과가 trial t+1의 제안에 영향), 40개 trial을 독립 관측치로 간주하는 것은 통계적 가정 위반이다.

**분석 단위**:
- **Run-level (검정 단위)**: 각 전략 × 10회 독립 반복 → 10개 final best accuracy 값 (서로 다른 seed, 서로 다른 시작 config)
- **Trial-level (시각화 전용)**: 학습 궤적, anytime performance curve, 탐색 효율성 시각화

**적용 검정**:
- Kruskal-Wallis test: 4그룹 × n=10 (자유도 3, 검정력 ~0.7)
- Mann-Whitney U test: Autoresearch vs Optuna 쌍별 비교 (n=10 vs n=10)
- BCa Bootstrap 95% CI: 각 전략의 최종 성능 분포
- Anytime Performance Curve (보조): X축 trial 번호, Y축 best-so-far accuracy, 평균 + 95% bootstrap CI 밴드

> **v0.4 변경**: 반복 횟수 5회 → 10회. 통계적 검정력(statistical power)을 ~0.6-0.7 수준으로 확보. Autoresearch vs Optuna 쌍별 비교를 위한 Mann-Whitney U test 추가.
> **v0.5 변경**: 통계 분석 단위를 **run-level**로 명시. Sequential optimization의 내부 trial 의존성을 인정하고, 독립 가정이 성립하는 run 단위에서만 비교 검정을 수행. Trial-level 데이터는 시각화 및 anytime performance curve에만 사용한다. 이로써 동료 심사 II-2 지적(독립성 가정 위반) 해소.

---

## 5. 논문 구조 (장별 구성)

### 제1장. 서론 (약 5p)
- 1.1 연구 배경
- 1.2 연구 목적
- 1.3 연구 범위 및 제한
- 1.4 논문 구성

### 제2장. 이론적 배경 (약 15p)
- 2.1 Vision-Language Model 개요
  - 2.1.1 멀티모달 학습의 발전
  - 2.1.2 경량 VLM 아키텍처 (Qwen-VL, SmolVLM 등)
- 2.2 Parameter-Efficient Fine-Tuning
  - 2.2.1 LoRA (Low-Rank Adaptation)
  - 2.2.2 QLoRA (Quantized LoRA)
  - 2.2.3 기타 PEFT 기법 비교
- 2.3 의료 영상 Visual Question Answering
  - 2.3.1 Medical VQA 과제 정의
  - 2.3.2 주요 벤치마크 데이터셋
  - 2.3.3 기존 연구 성과 (LLaVA-Med, Med-Flamingo, CheXagent 등 의료 특화 VLM 포함)
- 2.4 자율 하이퍼파라미터 최적화
  - 2.4.1 전통적 HPO (Grid, Random, Bayesian)
  - 2.4.2 LLM 에이전트 기반 최적화 (autoresearch)
    - 베이지안 최적화 대비 LLM HPO의 이론적 차별점: cross-domain transfer, 탐색 공간 구조 이해, 자연어 기반 설명 가능성
    - 조기 학습 신호 기반 HPO의 이론적 정당성: Hyperband(Li et al., 2018)의 successive halving, 초반 학습 곡선과 최종 수렴 성능 간 순위 상관관계 (Phase 2 Ablation A 학습 곡선 결과로 본 연구 내 자체 실증)
- 2.5 선행 연구 요약 및 본 연구의 차별점

> **v0.4 변경**: 2.3.3에 의료 특화 VLM(LLaVA-Med, Med-Flamingo, CheXagent 등) 선행 연구 비교 추가. 2.4.2에 LLM HPO의 이론적 근거 및 Optuna(TPE) 대비 차별점 논의 추가.

### 제3장. 연구 방법 (약 15p)
- 3.1 연구 설계 개요
- 3.2 실험 환경 및 도구
  - 3.2.1 하드웨어 사양 (로컬 RTX 5060 Ti + 클라우드 RTX 4090 이원 구성)
  - 3.2.2 소프트웨어 스택 (HuggingFace, Optuna 등)
- 3.3 대상 모델 및 선정 기준
- 3.4 데이터셋 및 전처리
- 3.5 실험 1: 제로샷 베이스라인 평가
- 3.6 실험 2: QLoRA 파인튜닝
- 3.7 실험 3: 자율 하이퍼파라미터 최적화
- 3.8 평가 지표 및 통계 분석 방법
  - 3.8.1 BERTScore 이중 보고 (roberta-large + BioBERT)
  - 3.8.2 Catastrophic Forgetting 이중 측정 (VQAv2 + cross-dataset)
  - 3.8.3 임상적 의미 분석 (WCA + ECE) — v0.5 신설
  - 3.8.4 Robust 통계 (Bootstrap + Mixed-Effects) — v0.5 신설
  - 3.8.5 데이터 오염 통제 (Min-K% Probability) — v0.5 신설

### 제4장. 실험 결과 및 분석 (약 20p)
- 4.1 Phase 1: 제로샷 베이스라인 결과
  - 4.1.1 모델별 성능 비교
  - 4.1.2 데이터셋별 난이도 분석
  - 4.1.3 오류 유형 분석
- 4.2 Phase 2: QLoRA 파인튜닝 결과
  - 4.2.1 Base vs Fine-tuned 성능 향상
  - 4.2.2 데이터 크기 영향 (Ablation A)
  - 4.2.3 LoRA Rank 영향 (Ablation B)
  - 4.2.4 Target Module 영향 (Ablation C)
  - 4.2.5 Catastrophic Forgetting 분석 (VQAv2 + cross-dataset)
- 4.3 Phase 3: 자율 HPO 결과
  - 4.3.1 각 전략의 최적 성능 비교 (Autoresearch vs Optuna 중심)
  - 4.3.2 탐색 효율성 비교
  - 4.3.3 탐색 궤적 분석
  - 4.3.4 발견된 최적 하이퍼파라미터 조합
  - 4.3.5 Autoresearch 탐색 근거 해석 가능성 분석
- 4.4 종합 분석 및 논의

### 제5장. 결론 (약 5p)
- 5.1 연구 요약
- 5.2 연구 기여
- 5.3 한계점
  - **데이터 오염 능동적 통제의 한계**: Min-K% Probability(Shi et al., ICLR 2024)로 contamination 정도를 정량화하나, 이는 간접 indicator이며 완전한 통제는 불가. Clean subset 결과를 보조 보고하여 결론의 robustness 검증.
  - **의료 특화 VLM 직접 비교 부재**: LLaVA-Med, Med-Flamingo 등과 동일 환경에서의 직접 실험 비교는 본 연구 범위 외. 동일 데이터셋·평가 프로토콜의 선행 연구 수치와 간접 비교 (Table 4.4).
  - **Phase 3 confounding**: max_steps 고정에도 effective_batch_size에 따른 total_samples_seen 차이 존재. 모든 trial의 effective_batch, samples_seen, wall_clock_time을 투명하게 보고.
  - **LLM 비결정성**: Autoresearch의 API 비결정성으로 완전 재현 불가. 구현은 temperature=0 고정이 아니라 **temperature 스케줄링(trial 진행률에 따라 1.0→0.3, 초반 탐색/후반 활용 균형)**을 사용하며, top_p는 API 기본값을 사용한다(별도 고정 없음). 모델 ID(`claude-sonnet-4-6`)만 고정하고 스냅샷 날짜 접미사는 별도 지정하지 않는다. API 응답 로깅, 10회 반복으로 변동성 흡수.
  - **Ablation Study 일반화**: LoRA Rank/Target Module/데이터 크기 Ablation은 PathVQA + 최적 모델 1개 기준 수행. **(v0.11 정정)** "SLAKE rank=8,16,32 보조 검증으로 일관성만 확인"은 2026-07-25 기준 git 이력·결과 파일·auto-memory 어디에도 실행 근거가 없어 아직 수행되지 않은 계획이었음을 확인했다 — `src/finetune/run_phase2.py`의 Ablation B(`run_ablation_b`)도 현재 `ABLATION_DATASET="pathvqa"`로 고정돼 있어 SLAKE를 바로 지정할 수 없다(코드 수정 필요). 이 보조 검증은 실행 여부를 확정하지 않은 채 완료된 것처럼 서술되어 있었으므로, 실제로 수행하거나(코드에 dataset 파라미터화 추가 후 실행) 또는 계획을 철회하고 "전체 cross-dataset 확장은 향후 연구"로만 남기는 결정이 필요하다. 전체 cross-dataset 확장은 향후 연구.
  - **통계적 검정력의 한계**: Phase 2 paired t-test n=9, Phase 3 run-level KW n=10. BCa Bootstrap + Mixed-Effects + Wilcoxon 등 3중 검증으로 robust한 결론 유도하나, n 자체의 한계로 효과 크기 추정 구간은 넓을 수 있음.
  - **WCA(Weighted Clinical Accuracy) 임시 가중치 한계**: 본 연구의 WCA 가중치(Diagnosis 1.0, Location 0.8, ...)는 임상 문헌이나 Delphi 기법 등 외부 검증 없이 연구자가 임의로 부여한 척도다. 이 가중치로 산출한 수치는 절대적 임상 중요도의 척도로 해석될 수 없으며, primary 지표(정확도, BERTScore)를 보완하는 참고용 보조 지표로만 제한적으로 사용한다. 임상의 설문 또는 Delphi 합의를 통한 가중치 검증은 후속 연구 과제로 남긴다.
  - **다중 비교 보정 부재**: 본 연구는 Phase 1(Cochran's Q + McNemar), Phase 2(paired t-test, Wilcoxon, Bootstrap, Mixed-Effects 병행), Phase 3(Kruskal-Wallis, Mann-Whitney 쌍별 비교)에 걸쳐 총 20회 이상의 통계 검정을 수행한다. 유의수준 0.05를 각 검정에 독립 적용할 경우 family-wise error rate가 누적되어 우연에 의한 유의 결과(제1종 오류)의 위험이 커진다. 본 연구는 Phase 1의 McNemar 사후검정에만 Bonferroni 보정을 적용했으며, Phase 2·3을 포함한 전체 파이프라인에 걸친 통합 다중비교 보정은 적용하지 않았다. 개별 p-value는 이 점을 감안하여 해석되어야 하며, 전체 파이프라인 수준의 FDR 보정 적용은 향후 분석 과제로 남긴다.
  - **Cross-dataset CF 개념 재정의**: PathVQA(병리 조직 영상)와 SLAKE/VQA-RAD(방사선 영상)는 이미지 도메인 자체가 상이하므로, (B) cross-dataset 성능 변화는 엄밀한 의미의 Catastrophic Forgetting(파인튜닝 이전에 가능했던 것을 파인튜닝 이후 수행하지 못하게 되는 현상)이라기보다, 도메인 특화에 따라 예측 가능한 도메인 일반화 격차(domain generalization gap)에 가깝다. 본 논문은 (B) 결과를 'cross-dataset 일반화 능력' 지표로 재명명하여 보고하며, CF의 엄밀한 판정은 (A) VQAv2 지표에 한정하여 해석한다.
  - **Gemma4-E2B MoE 공정성**: 평가 대상 모델 중 Gemma4-E2B는 Mixture-of-Experts(MoE) 구조로 추론 시 2.3B 파라미터만 활성화되나 전체 저장 파라미터는 5.1B에 달한다. 반면 Qwen3-VL-2B, SmolVLM-2.2B는 밀집(dense) 아키텍처로 활성/전체 파라미터가 동일하다. 본 연구의 '경량 VLM' 선정 기준은 활성 파라미터(추론 시 연산량 및 VRAM 사용량) 기준이며, 이는 소비자 GPU 환경에서의 실질적 구동 가능성이라는 연구 목적에 부합한다. 다만 Gemma4-E2B의 표현력이 저장 파라미터 5.1B에 기인할 가능성이 있어, 순수 파라미터 규모 기준 비교로 확대 해석해서는 안 된다는 한계를 명시한다.
  - **ECE(Expected Calibration Error) 산출 불가 (v0.11 신설)**: §4.4.5는 ECE를 WCA와 함께 임상적 의미 분석의 보조 지표로 제시하나, 현재 평가 파이프라인은 per-sample confidence(모델 예측 확률)를 저장하지 않는다. `scripts/analyze_clinical.py`/`src/evaluate/clinical_significance.py`에는 ECE 계산 로직 자체는 구현돼 있지만 입력값이 없어 항상 "N/A(미저장)"으로 보고된다. ECE를 실제로 산출하려면 평가 스크립트가 정답 여부뿐 아니라 예측 confidence까지 저장하도록 확장해야 하며, 이는 Phase 1/2 재실행을 요구하는 후속 작업이므로 본 연구에서는 ECE를 결과에 포함하지 못하고 WCA만 보조 지표로 보고한다.
  - **학습 예산(max_steps cap)의 한계**: Phase 2 QLoRA 파인튜닝은 목표 3 epochs였으나 RunPod RTX 4090의 시간·비용 제약으로 `max_steps=500`(조건당 samples_seen = 4,000 고정) 상한을 적용했다. 이로 인해 데이터셋 크기에 따라 실효 학습량이 크게 달라진다 — 소형 VQA-RAD(train ~1.8K)는 약 2 epoch 이상 학습되나, 중형 SLAKE(~11K)와 대형 PathVQA(~26K)는 각각 1 epoch 미만(약 0.4·0.15 epoch 수준)만 학습된다. 따라서 대형 데이터셋의 파인튜닝 성능은 수렴 이전의 과소학습(under-training) 상태일 수 있으며, 데이터셋 간 성능 비교는 '동일 학습 step 예산 하의 학습 효율' 관점으로 해석해야 하고 '완전 수렴 성능'으로 확대 해석해서는 안 된다. 모든 조건의 실효 `samples_seen`·환산 epoch·`wall_clock`을 결과에 투명 보고하며, 데이터셋별 full-epoch 재학습(특히 PathVQA)은 후속 연구 과제로 남긴다.
  - **Phase 3 trials_per_repeat 축소의 한계 (v0.12 신설)**: §4.5의 GPU 시간 재산출 결과 원안(전략당 40 trial)은 로컬 듀얼 GPU 환경에서 20 trial 기준(약 12.8일)의 약 2배에 해당하는 소요시간이 요구되어, run-level 통계 검정 단위인 `repeats=10`(§4.5 통계 검증 참조, run-level 독립성 가정의 기준)을 유지하는 대신 `trials_per_repeat`를 40에서 20으로 축소했다. 이로 인해 전략당(특히 Optuna/Autoresearch의 sequential exploration) 탐색 가능한 하이퍼파라미터 조합 수가 절반으로 줄어, 각 전략이 도달하는 최적 성능이 원안(40 trial) 대비 과소평가될 위험이 있다. Run-level 비교 검정(Kruskal-Wallis, Mann-Whitney U) 자체의 타당성은 repeats=10 유지로 영향받지 않으나, 절대적 "최적 성능" 수치는 trial 예산 제약 하의 결과로 해석해야 한다.
- 5.4 향후 연구 방향

> **v0.4 변경**: 5.3 한계점에 데이터 오염, 의료 특화 모델 비교 부재, confounding, LLM 비결정성을 명시적으로 서술.
> **v0.5 변경**: 5.3 한계점에 통계적 검정력의 한계와 WCA의 임시 가중치 한계를 추가 명시. 데이터 오염은 "Min-K%로 능동 통제하나 indicator 한계"로, 의료 특화 모델 비교는 "Table 4.4로 간접 비교"로 격상.
> **v0.7 변경**: 동료 심사 v0.5 잔여 지적(WCA 근거 부족, 다중비교 보정 누락, cross-dataset CF 개념 오용, Gemma4 MoE 공정성) 4건을 5.3 한계점에 명시적으로 반영. WCA 항목은 "절대적 임상 중요도 척도로 해석 불가"로 표현을 강화하고, 다중비교 보정 부재·cross-dataset CF 재정의·Gemma4 MoE 공정성 3건을 신규 추가. 4.4에 SPEC-EVAL-METRICS-001(REQ-EM-004)의 primary 지표 규칙(roberta-large 단일 결정, BioBERT 비-게이팅)을 명시.
> **v0.8 변경**: 4.4 QLoRA 표의 "Epochs 3"을 실제 구현(`max_steps=500` cap, samples_seen=4,000 고정)에 맞춰 "학습 예산" 행으로 수정하고, 5.3에 **학습 예산(max_steps cap)의 한계**를 신규 추가(대형 데이터셋 과소학습 가능성, 데이터셋 간 비교는 '동일 step 예산 하 학습 효율' 관점 해석). RUNPOD_GUIDE.md 실행 절차와의 정합성 확보(구현 ↔ 설계 일치).
> **v0.9 변경**: 실험 환경에 16GB GPU 2장(4080 Super ×2) 멀티-GPU pod 대안 환경을 명시(24GB 단일 GPU 확보가 어려웠던 시점의 실사용 환경). `run_phase2.py`의 조건별 병렬 실행(`--max_parallel`, GPU 1장당 1조건 고정)으로 model-parallel/DataParallel 충돌 없이 검증됨을 반영 — 상세는 `docs/RUNPOD_GUIDE.md` §4.0.
> **v0.10 변경**: Phase 3 LLM 비결정성 통제 서술(5.3, 6)을 실제 구현(`src/autoresearch/agent.py`)에 맞춰 정정 — "temperature=0, top_p=1 고정"이 아니라 **temperature 스케줄링(1.0→0.3)** 을 사용하며 top_p는 API 기본값임을 명시. 코드 검토 중 `anthropic` 패키지가 `pyproject.toml`/`uv.lock`에 누락되어 있던 것도 함께 발견·수정(Phase 3 `autoresearch` 전략이 새 pod `uv sync` 환경에서 즉시 실패할 수 있었던 문제).
> **v0.11 변경**: §4.5 탐색 공간 표와 고정 조건 문단이 max_steps를 두고 서로 모순됐던 것(표=탐색 가능 {100,200,400,800} vs 고정 조건=200 고정)을 발견·해소. `src/autoresearch/strategies.py`가 실제로는 탐색 가능한 버전으로 구현돼 있어 RandomSearch/Optuna가 trial마다 다른 max_steps를 뽑고 있었음을 확인, 코드를 고정 조건에 맞춰 수정(`PHASE3_FIXED_MAX_STEPS=200`, `agent.py`의 LLM 제안값 강제 덮어쓰기 포함). 탐색 공간 표에서 max_steps 행 제거(9→8개 파라미터). 추가로 학습 콜백에 걸린 wall-clock 안전장치(`time_budget_min`, 기존 15분)가 max_steps보다 먼저 trial을 끊어 사실상의 숨은 시간 예산으로 작동하고 있었음을 발견 — Phase 2 main 실측 처리량(스텝당 약 13초) 기준 200스텝 완주에는 최소 35-45분이 필요해 기존 15분 설정으로는 거의 모든 trial이 조기 절단됐을 가능성이 높음. 90분으로 상향해 안전장치가 실험 통제 변수를 침범하지 않도록 정정. 이에 따라 "예상 실험 규모"의 trial당 소요·총 GPU 시간 추정치도 재검증 필요로 표시(§4.5 본문 참조). 추가로 5.3에 두 항목을 반영: (1) ECE가 per-sample confidence 미저장으로 항상 N/A 산출됨을 명시(신규 한계점), (2) "SLAKE rank=8,16,32 보조 검증으로 일관성만 확인" 문장이 실제로는 실행 근거 없이 완료된 것처럼 서술돼 있었음을 발견·정정(Ablation Study 일반화 항목).
> **v0.12 변경**: §4.5 "예상 실험 규모"를 2026-08-01 로컬 듀얼 GPU(RTX 5060 Ti + RTX 4060) 스모크 실측치 기준으로 재산출. `trials_per_repeat`를 원안 40에서 **20**으로 축소(repeats=10은 run-level 통계 검정 단위이므로 불변 유지) — 총 trial 수 1,210회→**610회**로 재확정하고, `run_phase3.bat` 최종 실행 인자(`--repeats 10 --trials_per_repeat 20 --max_test_samples 500 --max_parallel 2`)를 문서에 명시. 총 소요시간 추정치를 "RunPod RTX 4090 단일 GPU 약 8-9일"에서 "로컬 듀얼 GPU 병렬 약 12.8일"로 정정(초기 추정치가 validation/test eval 시간을 누락했던 과소추정이었음을 실측으로 확인). §비교 대상 표·자율 탐색 루프 step 7·Table 4.3의 trial 수(40→20)도 동일하게 정정. 5.3 한계점에 trials_per_repeat 축소로 인한 탐색 공간 커버리지 저하 트레이드오프를 반영.

### 참고문헌
### 부록
- A. 상세 실험 결과 표
- B. autoresearch 에이전트 program.md
- C. 실험 재현 가이드
- D. Autoresearch 탐색 근거 로그 (자연어 rationale 전문)

---

## 6. 재현성 보장 계획

| 항목 | 방법 |
|------|------|
| 코드 관리 | GitHub 리포지토리 (실험 코드 + 설정 파일) |
| 환경 재현 | pyproject.toml + CUDA 버전 명시 |
| 실험 추적 | Weights & Biases + results.tsv |
| 랜덤 시드 | Phase 1(결정적): seed 42 / Phase 2·3(확률적): 42, 123, 456 고정 |
| 반복 실험 | Phase 1: 단일 결정적 평가 + 부트스트랩 95% CI / Phase 2·3: 3회 반복 -> 평균 +/- 표준편차 |
| 데이터 버전 | 데이터셋 버전 및 다운로드 URL 명시 |
| Git 커밋 | 각 실험 설정을 git commit으로 추적 |
| LLM 비결정성 통제 | temperature 스케줄링(1.0→0.3, 탐색/활용 균형), top_p는 API 기본값, 모델 ID(`claude-sonnet-4-6`) 고정 |
| API 응답 로깅 | 전체 API 요청/응답 JSON을 실험별 로그 파일로 저장 |
| 변동성 흡수 | Phase 3 각 전략 10회 반복으로 분포 보고 |

> **v0.4 변경**: LLM 비결정성 통제, API 응답 로깅, 변동성 흡수를 위한 10회 반복 항목 추가.

---

## 7. 예상 결과 테이블 (형식 예시)

### Table 4.1: 제로샷 베이스라인 결과 (VQA Accuracy %)

| 모델 | PathVQA (Open) | PathVQA (Closed) | SLAKE (Open) | SLAKE (Closed) | VQA-RAD (Open) | VQA-RAD (Closed) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | ?.?? +/- ?.?? | ?.?? +/- ?.?? | ... | ... | ... | ... |
| Qwen2.5-VL-3B | ... | ... | ... | ... | ... | ... |
| SmolVLM-2.2B | ... | ... | ... | ... | ... | ... |
| Gemma4-E2B | ... | ... | ... | ... | ... | ... |

### Table 4.2: QLoRA 파인튜닝 전후 비교

| 모델 | 조건 | PathVQA (EM) | PathVQA (BERTScore-roberta) | PathVQA (BERTScore-BioBERT) | SLAKE Acc. | VQA-RAD Acc. | 향상율 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | Zero-shot | ... | ... | ... | ... | ... | - |
| Qwen3-VL-2B | QLoRA | ... | ... | ... | ... | ... | +?.?% |
| ... | ... | ... | ... | ... | ... | ... | ... |

### Table 4.2b: Catastrophic Forgetting 측정

**(A) VQAv2 subset (범용 VQA)**

| 모델 | 학습 데이터셋 | Base Acc. | Fine-tuned Acc. | 감소율 (%) |
|------|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | PathVQA | ?.?? | ?.?? | ?.?% |
| Qwen3-VL-2B | SLAKE | ?.?? | ?.?? | ?.?% |
| ... | ... | ... | ... | ... |

**(B) 의료 도메인 내 cross-dataset 일반화**

| 모델 | 훈련 | 평가 | Base Acc. | Fine-tuned Acc. | 변화율 (%) |
|------|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | PathVQA | SLAKE | ?.?? | ?.?? | ?.?% |
| Qwen3-VL-2B | PathVQA | VQA-RAD | ?.?? | ?.?? | ?.?% |
| ... | ... | ... | ... | ... | ... |

### Table 4.3: HPO 전략 비교 (PathVQA, 최적 모델)

| HPO 전략 | 최적 Accuracy | 실험 횟수 | 총 시간(h) | 효율성 | effective_batch 범위 | total_samples 범위 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Manual (기본값) | ?.?? | 1 | ?.? | - | 8 | ?.?K |
| Random Search | ?.?? +/- ?.?? | 20 | ~?? | ?.?? | 4-64 | ?.?K-?.?K |
| Optuna (TPE) | ?.?? +/- ?.?? | 20 | ~?? | ?.?? | 4-64 | ?.?K-?.?K |
| Autoresearch | ?.?? +/- ?.?? | 20 | ~?? | ?.?? | 4-64 | ?.?K-?.?K |

### Table 4.4: 의료 특화 VLM vs 본 연구 경량 VLM (간접 비교, v0.5 신설)

본 연구의 경량 범용 VLM(2-3B)이 의료 특화 모델 대비 어느 수준에 도달하는지 간접 비교한다. 모델 크기와 학습 데이터가 다르므로 직접 공정 비교는 어려우나, 동일 데이터셋·동일 평가 프로토콜의 선행 연구 수치와 대조하여 실용적 가치를 평가한다.

| 모델 | 파라미터 | 학습 데이터 | PathVQA (Open) | PathVQA (Closed) | SLAKE (Open) | VQA-RAD (Open) | 출처 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LLaVA-Med-7B** | 7B | PMC-15M + LLaVA-Med Inst. | 39.6% | 91.2% | 38.8% | 31.2% | Li et al., NeurIPS 2023 |
| **Med-Flamingo** | 8.3B (active) | MTB + PMC-OA | 35.8% | 88.0% | - | - | Moor et al., 2023 |
| **CheXagent** | 7B (Mistral-7B 기반) | CheXinstruct + PadChest | - | - | - | 45.1% (chest) | Chen et al., 2024 |
| **BioViL-T** | 0.4B | MIMIC-CXR | - | - | - | 28.4% (chest) | Bannur et al., CVPR 2023 |
| **Qwen3-VL-2B** (제로샷, 본 연구) | 2B | 일반 멀티모달 | ?.?% | ?.?% | ?.?% | ?.?% | - |
| **Qwen3-VL-2B** (QLoRA, 본 연구) | 2B | 일반 + 의료 fine-tune | ?.?% | ?.?% | ?.?% | ?.?% | - |
| **Gemma4-E2B** (제로샷, 본 연구) | 2.3B/5.1B | 일반 멀티모달 | ?.?% | ?.?% | ?.?% | ?.?% | - |
| **Gemma4-E2B** (QLoRA, 본 연구) | 2.3B/5.1B | 일반 + 의료 fine-tune | ?.?% | ?.?% | ?.?% | ?.?% | - |

**해석 가이드**:
- 본 연구 모델은 의료 특화 학습 데이터의 1/100 이하 규모로 학습 시도
- 본 연구의 가치: (1) 의료 특화 모델 대비 어느 정도 격차, (2) 소비자 GPU에서 운용 가능한 경량 솔루션, (3) 도메인 적응 효율성 분석
- 의료 특화 모델 대비 격차가 클 경우: future work에서 "본 연구 방법론 + 의료 특화 데이터"의 결합 가능성 제시
- 격차가 적을 경우: 경량 범용 VLM + QLoRA가 실용적 대안임을 실증

> **v0.5 변경**: 동료 심사 V번 지적(의료 특화 모델 비교 부재)에 대응. 직접 실험 비교는 환경 차이로 불공정할 수 있어, 동일 데이터셋·동일 평가 프로토콜의 선행 연구 수치와 간접 비교한다. 결과 해석 시 모델 크기와 학습 데이터 규모의 차이를 함께 고려한다.
