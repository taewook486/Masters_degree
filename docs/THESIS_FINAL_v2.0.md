# 석사학위 논문 (심사용 초안)

<!-- pdf:strip-meta -->
> 본문 버전: v2.0-draft (2026-07-31 시작, 2026-08-13 전체 초안 완성)
> 기반 설계서: [THESIS_PROPOSAL_FINAL_v0.12.md](THESIS_PROPOSAL_FINAL_v0.12.md)
> 진행 상태: **제1-5장 + 참고문헌 + 부록 A-D 전체 초안 완성.** 3단계 실험(Phase 1 제로샷 12조건 / Phase 2 QLoRA 75조건 / Phase 3 자율 HPO 610 trial) 모두 완료되어 실측값으로 작성됨. 2장 인용은 WebSearch 검증 완료(설계서의 Min-K% 인용 오류 NAACL→ICLR 발견·정정), 참고문헌 15건은 원문 대조 완료.
> 남은 작업: 인용 표기 형식의 학과 지정 양식 통일, 지도교수 피드백 반영.
<!-- /pdf:strip-meta -->

## 논문 정보

- **대학교**: 건국대학교 정보통신대학원 융합정보기술학과 인공지능전공
- **제출 목표**: 2026년 9월
- **제목(한국어)**: 경량 멀티모달 모델의 의료 영상 VQA 도메인 적응: QLoRA 파인튜닝과 자율 하이퍼파라미터 최적화
- **제목(영문)**: Domain Adaptation of Lightweight Vision-Language Models for Medical Visual Question Answering: QLoRA Fine-Tuning with Autonomous Hyperparameter Optimization

---

## 목차 (설계서 §5 기준)

- 제1장. 서론 ✅
- 제2장. 이론적 배경 ✅ (인용 WebSearch 검증 완료)
- 제3장. 연구 방법 ✅
- 제4장. 실험 결과 및 분석 ✅
  - 4.1 Phase 1: 제로샷 베이스라인 결과 (RQ1)
  - 4.2 Phase 2: QLoRA 파인튜닝 결과 (RQ2)
  - 4.3 Phase 3: 자율 하이퍼파라미터 최적화 결과 (RQ3)
  - 4.4 종합 분석 및 논의 (4.4.6 선행 연구 간접 비교 포함)
- 제5장. 결론 ✅ (5.1 요약 / 5.2 기여 / 5.3 한계점 / 5.4 향후 연구)
- 참고문헌 ✅ (15건, 원문 대조 완료)
- 부록 A~D ✅ (A 결과파일 경로 / B 에이전트 시스템 프롬프트 / C 재현 가이드 / D 제안 근거 로그)

---

## 제1장. 서론

### 1.1 연구 배경

거대언어모델(LLM)의 성공에 힘입어 이미지와 텍스트를 함께 이해하는 Vision-Language Model(VLM)이 GPT-4V, Gemini 등을 중심으로 빠르게 발전하고 있다. 이러한 범용 VLM은 자연 이미지에 대한 질의응답에서는 높은 성능을 보이지만, 의료 영상과 같이 전문 지식이 요구되는 도메인에서는 의학 용어와 병리·방사선학적 소견에 대한 이해 부족으로 성능이 제한되는 경향이 있다. 반면 GPU 자원이 제한된 연구·임상 환경에서 대규모 모델을 처음부터 재학습하는 것은 현실적이지 않으며, 이에 QLoRA(Quantized Low-Rank Adaptation)와 같은 Parameter-Efficient Fine-Tuning(PEFT) 기법이 소비자급 GPU(16-24GB VRAM) 환경에서도 도메인 특화 파인튜닝을 가능케 하는 대안으로 주목받고 있다.

한편 QLoRA와 같은 PEFT 기법을 실제로 적용하려면 LoRA rank, target module, 학습률 등 다수의 하이퍼파라미터를 선택해야 하는데, 이 선택이 최종 성능에 미치는 영향은 체계적으로 규명되지 않은 경우가 많다. 하이퍼파라미터 최적화(HPO)의 자동화는 2024년 NeurIPS 서베이에서도 지적된 VLM PEFT 분야의 미해결 과제이며, 전통적인 Grid/Random Search나 베이지안 최적화(Optuna TPE)를 넘어 LLM 에이전트가 이전 실험 결과를 스스로 해석하고 다음 설정을 제안하는 자율 탐색(autoresearch 스타일) 방식이 새로운 대안으로 제시되고 있다.

### 1.2 연구 목적

본 연구는 소비자 GPU 환경에서 경량 Vision-Language Model을 의료 영상 VQA(Visual Question Answering) 도메인에 적응시키는 전 과정을 실증적으로 검증하는 것을 목적으로 하며, 구체적으로 다음 세 가지를 달성하고자 한다.

1. 16-24GB급 소비자 GPU 환경에서 경량 VLM의 의료 VQA 도메인 적응 가능성을 실증한다.
2. QLoRA 파인튜닝의 주요 하이퍼파라미터(데이터 규모, LoRA rank, target module 범위)가 성능에 미치는 영향을 체계적으로 분석한다.
3. autoresearch 스타일의 LLM 에이전트 기반 자율 하이퍼파라미터 탐색이 베이지안 최적화 대비 경쟁력 있는 성능과 해석 가능한 탐색 근거를 제공하는지 검증한다.

이 세 목적은 각각 아래의 연구 질문(RQ)으로 구체화된다.

| # | 연구 질문 | 귀무가설 |
|---|----------|----------|
| RQ1 | 경량 VLM(2-3B)의 의료 VQA 제로샷 성능은 모델별로 유의미한 차이가 있는가? | H0: 모델 간 VQA 정확도 차이 없음 |
| RQ2 | QLoRA 파인튜닝이 의료 VQA 성능을 유의미하게 향상시키는가? | H0: Base = Fine-tuned 성능 |
| RQ3 | LLM 에이전트 기반 자율 하이퍼파라미터 탐색이 베이지안 최적화(Optuna TPE)와 경쟁적 성능을 달성하면서 해석 가능한 탐색 근거를 제공하는가? | H0: Autoresearch = Optuna (TPE) |

RQ3의 귀무가설을 단순 Random Search가 아닌 Optuna(TPE)로 설정한 것은, LLM 기반 HPO가 단지 "무작위보다 나은" 수준을 넘어 이미 실무 표준으로 자리잡은 베이지안 최적화와 경쟁할 수 있어야 그 도입 근거(자연어 기반 탐색 근거 설명, 사전 지식을 활용한 탐색 공간 구조 이해, 설정 변경 이유의 추적 가능성)가 성립한다고 판단했기 때문이다. Random Search는 하한선(lower bound) 비교 대상으로만 유지한다.

RQ1·RQ2·RQ3는 각각 제4장 4.1·4.2·4.3에서 실측 데이터로 검증했으며, 세 결과를 관통하는 논의는 4.4에서 종합한다.

### 1.3 연구 범위 및 제한

본 연구의 실험 범위는 4개 경량 VLM(Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B)과 3개 공개 의료 VQA 데이터셋(PathVQA, SLAKE, VQA-RAD)으로 한정한다. 모델 선정은 16GB급 VRAM에서 QLoRA 파인튜닝이 가능하고 재배포가 자유로운 라이선스(Apache 2.0/MIT)를 갖춘 모델을 기준으로 했으며, 데이터셋은 공개적으로 접근 가능하고 임상 질문 유형이 라벨링된 벤치마크로 한정했다.

연구의 주요 제한점은 다음과 같으며, 상세한 근거와 완화 조치는 제5장 5.3에서 논의한다: (1) 대상 데이터셋이 모델의 사전훈련 시점 이전에 공개되어 사전훈련 데이터 오염 가능성이 존재하므로, 본 연구는 이를 Min-K% Probability 기법으로 능동 측정하고 결론의 강건성을 별도 검증한다(제4장 4.1.1 오염 강건성 검증 참조). (2) LLaVA-Med, Med-Flamingo 등 기존 의료 특화 VLM과의 직접 실험 비교는 본 연구 범위 밖이며 선행 연구 수치와의 간접 비교로 대체한다. (3) GPU 시간·비용 제약으로 QLoRA 학습에 `max_steps` 상한을 적용했으며, 이로 인해 데이터셋 크기에 따라 실효 학습량(epoch 환산)이 달라진다.

### 1.4 논문 구성

본 논문은 다음과 같이 구성된다. 제2장에서는 Vision-Language Model과 Parameter-Efficient Fine-Tuning, 의료 VQA, 자율 하이퍼파라미터 최적화에 관한 이론적 배경과 선행 연구를 검토한다. 제3장에서는 실험 환경, 대상 모델·데이터셋, 3단계 실험(Phase 1 제로샷 베이스라인, Phase 2 QLoRA 파인튜닝, Phase 3 자율 HPO)의 구체적 설계와 평가·통계 분석 방법을 기술한다. 제4장에서는 각 Phase의 실험 결과를 RQ1-RQ3에 따라 제시하고 종합 논의한다. 제5장에서는 연구 결과를 요약하고 기여점과 한계, 향후 연구 방향을 제시한다.

---

## 제2장. 이론적 배경

### 2.1 Vision-Language Model 개요

#### 2.1.1 멀티모달 학습의 발전

Vision-Language Model(VLM)은 이미지 인코더와 대규모 언어모델(LLM)을 결합하여 시각 정보와 언어 정보를 함께 이해·생성하는 모델이다. 초기 멀티모달 학습은 이미지-텍스트 쌍에 대한 대조 학습(contrastive learning)으로 공유 임베딩 공간을 학습하는 방식이 중심이었으며, 이후 LLM의 instruction-following 능력이 발전하면서 이미지 특징을 LLM의 입력 토큰 공간에 투영(projection)하고 instruction-tuning으로 시각 질의응답·설명 생성 능력을 학습시키는 방식이 주류가 되었다. Liu 등의 LLaVA(Visual Instruction Tuning, arXiv:2304.08485)는 GPT-4로 생성한 시각 instruction 데이터로 오픈소스 LLM을 파인튜닝해 상용 모델에 근접한 시각 대화 능력을 보인 대표적 사례로, 이후 다수의 VLM 연구가 이 instruction-tuning 패러다임을 계승했다. GPT-4V, Gemini 등 상용 대규모 VLM은 방대한 파라미터와 학습 데이터로 범용 시각 이해 성능을 확보했으나, 동시에 추론 비용과 배포 제약이 커 소비자급 하드웨어에서의 활용이 어렵다는 한계가 있다.

#### 2.1.2 경량 VLM 아키텍처

본 연구가 대상으로 삼는 4개 모델은 모두 2025-2026년에 공개된 2-3B급 경량 VLM으로, 각기 다른 방식으로 파라미터 효율성을 추구한다. Qwen2.5-VL(Qwen Team, Alibaba, Technical Report arXiv:2502.13923)은 이미지 크기에 따라 시각 토큰 수를 동적으로 조절하는 dynamic resolution 처리와 시간 정보를 절대 시간으로 정렬하는 MRoPE 확장을 특징으로 하며, 문서 파싱과 다국어 OCR에 강점을 보인다. 후속 모델인 Qwen3-VL은 dense(2B/4B/8B/32B)와 Mixture-of-Experts(30B-A3B/235B-A22B) 계열로 확장되었고 "thinking mode"와 DeepStack 방식의 다단계 시각 특징 결합을 도입했다. SmolVLM2(HuggingFace, "SmolVLM: Redefining small and efficient multimodal models", arXiv:2504.05299)는 SigLIP 이미지 인코더와 SmolLM2 언어모델을 결합해 극단적으로 낮은 메모리 사용량(2.2B 모델 기준 영상 추론 시 5.2GB)을 달성하는 데 초점을 맞춘 온디바이스 지향 아키텍처다. Gemma4-E2B(Google)는 Per-Layer Embeddings(PLE) 기법으로 추론 시 2.3B 파라미터만 활성화하면서 5.1B급 총 파라미터의 표현력을 활용하는 Mixture-of-Experts 계열 구조로, 활성 파라미터 기준으로는 경량이나 저장 파라미터 규모는 다른 세 모델보다 크다는 아키텍처적 차이가 있다(이 차이가 결과 해석에 갖는 함의는 제5장 5.3에서 논의한다).

### 2.2 Parameter-Efficient Fine-Tuning

#### 2.2.1 LoRA (Low-Rank Adaptation)

LoRA(Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685, ICLR 2022)는 사전학습된 가중치 행렬 $W_0$을 고정한 채, 그 변화량 $\Delta W$를 두 개의 저랭크(low-rank) 행렬 $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$의 곱 $BA$($r \ll \min(d,k)$)로 근사하여 학습하는 기법이다. 순전파는 $h = W_0 x + BAx$로 계산되며, 학습 대상은 $A$, $B$뿐이므로 전체 파라미터 대비 학습 파라미터 비율을 극적으로 낮출 수 있다(본 연구의 Ablation B에서 rank=64 기준 학습 파라미터 비율은 전체의 약 0.2-1.6% 수준, 제4장 4.2.3 참조). rank $r$은 표현력과 파라미터 효율성 사이의 트레이드오프를 결정하는 핵심 하이퍼파라미터이며, alpha는 $\Delta W$의 스케일을 조정하는 계수다($BAx$에 $\alpha/r$을 곱함).

#### 2.2.2 QLoRA (Quantized LoRA)

QLoRA(Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314, NeurIPS 2023)는 LoRA에 4-bit 양자화를 결합하여 VRAM 사용량을 추가로 절감한다. 핵심 구성 요소는 (1) 정규분포를 따르는 가중치에 최적화된 4-bit NormalFloat(NF4) 양자화, (2) 양자화 상수 자체를 다시 양자화하는 이중 양자화(double quantization), (3) GPU 메모리 스파이크를 CPU로 흘려보내는 paged optimizer이다. 기반 모델은 4-bit로 양자화된 상태로 고정하고 그 위에 LoRA 어댑터만 16-bit 정밀도로 학습하므로, 65B급 모델을 단일 48GB GPU에서 파인튜닝하면서도 완전 16-bit 파인튜닝에 근접한 성능을 유지함을 원 논문에서 보였다. 본 연구는 이 QLoRA 방식(NF4 양자화 + LoRA + paged AdamW 8bit)을 4개 경량 VLM 전체에 적용하여, 16GB급 소비자 GPU에서의 의료 도메인 파인튜닝 가능성을 검증한다(제3장 3.6).

#### 2.2.3 기타 PEFT 기법 비교

LoRA/QLoRA 외에도 일부 레이어만 학습하는 partial fine-tuning, 입력 앞단에 학습 가능한 prefix 벡터를 추가하는 prefix-tuning, 트랜스포머 레이어 사이에 소규모 어댑터 모듈을 삽입하는 adapter 방식 등 다양한 PEFT 기법이 제안되어 왔다. 이들과 비교했을 때 LoRA 계열의 장점은 (1) 추론 시 $BA$를 $W_0$에 합산(merge)할 수 있어 추가 지연시간이 없고, (2) 어댑터만 별도로 저장·교체할 수 있어 여러 태스크에 대해 하나의 기반 모델을 재사용할 수 있다는 점이다. 본 연구가 LoRA/QLoRA를 채택한 것은 이 두 장점이 소비자 GPU 환경에서의 반복적인 하이퍼파라미터 탐색(제3장 3.7, Phase 3)에 특히 유리하기 때문이다.

### 2.3 의료 영상 Visual Question Answering

#### 2.3.1 Medical VQA 과제 정의

Medical VQA는 의료 영상(병리 조직 슬라이드, 방사선 영상 등)과 자연어 질문이 주어졌을 때 올바른 답변을 생성하는 과제로, 일반 도메인 VQA와 달리 전문 의학 용어에 대한 이해와 영상 소견에 대한 세밀한 시각적 근거가 함께 요구된다. 질문 유형은 크게 (1) 예/아니오나 선택지 중에서 고르는 closed-ended 질문과, (2) 자유 서술형 답변을 요구하는 open-ended 질문으로 나뉘며, 두 유형은 채점 방식과 난이도가 크게 다르다(본 연구 제4장 4.1.2에서 실측 확인).

#### 2.3.2 주요 벤치마크 데이터셋

본 연구가 사용하는 세 데이터셋은 각기 다른 의료 영상 하위 도메인을 대표한다. PathVQA(He et al., "PathVQA: 30000+ Questions for Medical Visual Question Answering", arXiv:2003.10286, 2020)는 병리학 교과서와 PEIR 디지털 라이브러리에서 추출한 4,998개 병리 조직 영상에 대해 32,799개의 질문-답변 쌍을 제공하며, 미국병리위원회(ABP) 전문의 자격시험 형식을 참고해 7개 질문 유형으로 구성했다. SLAKE(Liu et al., "SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering", ISBI 2021)는 642개 방사선/CT 영상에 대한 14,028개의 영어-중국어 이중언어 질문-답변 쌍과, 의학 지식 그래프(5,232개 지식 triplet)를 결합한 데이터셋이다. VQA-RAD(Lau et al., "A dataset of clinically generated visual questions and answers about radiology images", Scientific Data 5, 2018)는 임상의가 실제로 방사선 영상(CT/MRI/X-ray)을 보고 자연스럽게 제기한 질문을 수집한 최초의 데이터셋으로, 315개 영상에 대해 약 3,500여 개의 질문-답변 쌍을 담고 있다.

#### 2.3.3 기존 연구 성과

범용 VLM을 의료 도메인에 특화시키려는 시도는 크게 대규모 재학습형과 어댑테이션형으로 나뉜다. LLaVA-Med(Li et al., "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day", arXiv:2306.00890, NeurIPS 2023 Datasets and Benchmarks Track)는 PubMed Central의 대규모 의생명 그림-캡션 데이터로 GPT-4 기반 instruction 데이터를 생성하고, 커리큘럼 학습으로 범용 LLaVA를 의생명 도메인에 적응시켰다. Med-Flamingo(Moor et al., "Med-Flamingo: a Multimodal Medical Few-shot Learner", arXiv:2307.15189, 2023)는 OpenFlamingo-9B를 기반으로 의학 논문·교과서의 이미지-텍스트 데이터로 계속 사전학습(continued pretraining)하여, 소수 예시(few-shot)만으로 의료 VQA에 적응하는 능력을 확보했다. CheXagent(Chen et al., "CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation", arXiv:2401.12208, 2024)는 흉부 X-ray 판독에 특화된 임상 LLM·시각 인코더·교차 모달 브리지 네트워크로 구성된 foundation model로, 특정 영상 하위 도메인(흉부 X-ray)에 깊이 특화된 접근을 대표한다.

이들 선행 연구는 공통적으로 수십억~수백억 파라미터 규모의 모델을 대상으로 하며, 대규모 의생명 코퍼스에 대한 계속 사전학습이나 대규모 instruction 데이터 생성을 전제로 한다. 이와 달리 본 연구는 2-3B급 경량 모델에 QLoRA로 소규모 도메인 데이터(수천-수만 샘플)만을 사용해 파인튜닝하는, 계산 자원이 제한된 환경에서의 실용적 적응 가능성에 초점을 둔다는 점에서 접근 방식이 다르다. 이들 선행 연구와의 직접적인 실험 비교는 본 연구 범위 밖이다. 다만 LLaVA-Med는 본 연구와 동일한 세 데이터셋의 표준 test split에 대해 수치를 보고하므로, 채점 기준이 일치하는 closed-ended 지표에 한정해 제4장 4.4.6(Table 4.4)에서 간접 비교한다. Open-ended 지표는 양측의 채점 방식이 근본적으로 달라 비교 대상에서 제외하며, 그 근거와 남은 제약은 제5장 5.3(2)에서 논의한다.

### 2.4 자율 하이퍼파라미터 최적화

#### 2.4.1 전통적 HPO (Grid, Random, Bayesian)

하이퍼파라미터 최적화(HPO)의 가장 단순한 방법은 사전에 정의한 격자(grid) 위의 모든 조합을 평가하는 grid search이나, 탐색 공간의 차원이 늘어날수록 필요한 평가 횟수가 지수적으로 증가한다. Bergstra와 Bengio("Random Search for Hyper-Parameter Optimization", JMLR 13, 2012)는 무작위로 조합을 샘플링하는 random search가 동일한 계산 예산 내에서 grid search보다 실질적으로 더 나은(또는 동등한) 성능을 낼 수 있음을 이론적·경험적으로 보였으며, 이는 실제로 성능에 영향을 미치는 하이퍼파라미터의 유효 차원(effective dimensionality)이 낮은 경우가 많기 때문이다. 베이지안 최적화는 이전 평가 결과로 목적함수의 확률적 대리 모델(surrogate model)을 구성하고, 이를 바탕으로 다음 평가 지점을 선택함으로써 grid/random search보다 적은 평가 횟수로 더 나은 해를 찾는 것을 목표로 한다. 본 연구가 대조군으로 사용하는 Optuna(Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework", KDD 2019)는 Tree-structured Parzen Estimator(TPE) 알고리즘을 채택한 베이지안 최적화 프레임워크로, 탐색 공간을 실행 중 동적으로 정의할 수 있는 define-by-run API와 효율적인 pruning(조기 중단) 전략을 제공한다.

조기 중단 기반 방법 중 Hyperband(Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 18, 2018)는 successive halving을 반복 적용해, 유망하지 않은 설정에는 적은 자원(학습 스텝 등)만 할당하고 유망한 설정에는 점진적으로 더 많은 자원을 배분하는 순수 탐색형 밴딧(bandit) 문제로 HPO를 정식화한다. 이 방법이 성립하려면 조기 학습 곡선(초반 몇 스텝의 성능)과 최종 수렴 성능 사이에 유의한 순위 상관관계가 있어야 하는데, 본 연구의 Phase 2 Ablation A(데이터 비율별 학습 곡선, 제4장 4.2.2)는 이와 유사한 조기 신호-최종 성능 관계를 자체 실험 맥락에서 관찰할 수 있는 근거를 제공한다.

#### 2.4.2 LLM 에이전트 기반 최적화 (autoresearch)

베이지안 최적화가 수치적 대리 모델에 의존하는 것과 달리, 최근에는 LLM 에이전트가 이전 실험 로그를 직접 해석하고 다음 설정을 자연어 추론으로 제안하는 방식이 하이퍼파라미터 탐색의 새로운 대안으로 논의되고 있다. 이 접근의 이론적 차별점은 세 가지로 요약된다: (1) LLM이 사전학습 과정에서 습득한 도메인 지식을 활용해 서로 다른 태스크·아키텍처 간 지식을 전이(cross-domain transfer)할 수 있다는 점, (2) 개별 하이퍼파라미터를 독립 변수로 다루는 베이지안 최적화와 달리 하이퍼파라미터 간 상호작용을 사전 지식에 기반해 구조적으로 이해할 수 있다는 점, (3) 설정을 변경한 이유를 자연어로 명시적으로 기록하여 탐색 과정의 해석 가능성과 추적 가능성을 제공한다는 점이다. 본 연구가 채택한 autoresearch 스타일 루프(이전 결과 읽기 → 자연어 근거와 함께 다음 설정 제안 → 실행 → 결과 기록, 제3장 3.7)는 이 세 차별점이 실제로 베이지안 최적화(Optuna/TPE) 대비 경쟁력 있는 성능과 결합될 수 있는지를 RQ3으로 검증하기 위한 실험 장치다. 다만 이는 아직 정립된 표준 방법론이라기보다 LLM 에이전트의 과학적 실험 자동화 활용이라는 새로운 연구 흐름의 한 사례이며, 본 연구는 이 방식의 일반적 우수성을 주장하기보다 의료 VQA QLoRA 튜닝이라는 구체적 과제에서의 경쟁력을 실증적으로 검증하는 데 초점을 둔다.

### 2.5 선행 연구 요약 및 본 연구의 차별점

앞서 검토한 선행 연구를 종합하면, 의료 특화 VLM 연구(2.3.3)는 대체로 대규모 모델과 대규모 도메인 데이터를 전제로 하며, PEFT 방법론 연구(2.2)는 범용 도메인에서의 파라미터 효율성 검증에 집중되어 있고, HPO 자동화 연구(2.4)는 아직 LLM 에이전트 기반 방식과 확립된 베이지안 최적화를 동일 과제에서 직접 비교한 사례가 드물다. 본 연구는 이 세 흐름의 교차점 — **소비자 GPU 환경 · 경량 VLM · QLoRA 도메인 적응 · LLM 기반 자율 HPO** — 에서, (1) 경량 VLM의 의료 VQA 제로샷·파인튜닝 성능을 통계적으로 엄밀하게 실증하고(RQ1, RQ2), (2) autoresearch 스타일 자율 탐색을 업계 표준인 Optuna(TPE)와 직접 비교하여(RQ3) 그 실용적 가치를 검증한다는 점에서 선행 연구와 차별화된다.

---

## 제3장. 연구 방법

### 3.1 연구 설계 개요

본 연구는 경량 VLM의 의료 VQA 도메인 적응을 3단계 실험(Phase)으로 순차 검증하는 실증 연구로 설계했다. Phase 1은 파인튜닝 이전 4개 모델의 제로샷 성능을 측정하여 RQ1(모델 간 성능 차이)에 답하고, Phase 2는 QLoRA 파인튜닝을 적용해 RQ2(파인튜닝 효과)를 검증하며, 동시에 최적 QLoRA 설정(데이터 규모·LoRA rank·target module)을 Ablation Study로 탐색한다. Phase 3은 Phase 2에서 확인된 최적 설정을 출발점으로, Manual·Random Search·Optuna(TPE)·Autoresearch 4개 HPO 전략을 비교하여 RQ3(자율 탐색의 경쟁력)을 검증한다. 각 Phase는 이전 Phase의 결과(최적 모델, 최적 QLoRA 설정)를 다음 Phase의 고정 조건으로 사용하는 순차적 구조를 갖는다.

### 3.2 실험 환경 및 도구

#### 3.2.1 하드웨어 사양

Phase 1·2는 클라우드 GPU(RunPod, RTX 4090 24GB)를 주 실험 환경으로 사용했으며, 로컬 RTX 5060 Ti(16GB VRAM, Ryzen 5 5600X, RAM 32GB)에서 일부 조건을 재현하여 소비자 GPU(16GB) 환경에서의 재현 가능성을 함께 확인했다. Phase 3은 소속 기관의 비용 지원이 불가하고 클라우드 예산 확보가 어려워진 관계로 RunPod 사용을 중단하고, 로컬 RTX 5060 Ti(16GB) 단독 — 하드웨어 여건에 따라 4060(8GB)과의 듀얼 GPU 구성 — 환경으로 전환하여 진행한다. 이러한 환경 전환의 경위와 영향은 제5장 5.3 한계점에서 상세히 논의한다.

#### 3.2.2 소프트웨어 스택

모델 로딩·QLoRA 파인튜닝은 HuggingFace `transformers`와 `unsloth`(4-bit 양자화 학습 가속) 백엔드를 사용했으며, `peft` 라이브러리로 LoRA 어댑터를 구성했다. Phase 3의 베이지안 최적화 대조군은 `Optuna`(TPE sampler)를, 실험 관리·로깅은 `wandb`를 사용했다. Open-ended 응답 채점의 BERTScore 계산에는 `bert-score` 라이브러리(roberta-large / BioBERT 백본)를, 통계 분석에는 `scipy`·`statsmodels`(Mixed-Effects Model)를 사용했다. Autoresearch 전략의 LLM 에이전트는 Anthropic Claude API를 호출한다.

### 3.3 대상 모델 및 선정 기준

**Table 3.1. 대상 모델**

| 모델 | 파라미터 | 아키텍처 특징 | 예상 QLoRA VRAM |
|------|---------|-------------|:---:|
| Qwen3-VL-2B | 2B | Thinking mode, DeepStack | ~8-10 GB |
| Qwen2.5-VL-3B | 3B | Dynamic Resolution, 19개 언어 OCR | ~8-10 GB |
| SmolVLM2-2.2B | 2.2B | HuggingFace 경량 VLM | ~8-10 GB |
| Gemma4-E2B | 2.3B(활성)/5.1B(전체) | PLE(Per-Layer Embeddings), Apache 2.0 | ~12-14 GB |

선정 기준은 (1) 16GB VRAM에서 QLoRA 파인튜닝이 가능할 것, (2) Apache 2.0 또는 MIT 라이선스로 연구 활용이 자유로울 것, (3) 충분한 커뮤니티·프레임워크 지원을 갖출 것의 세 가지다. Gemma4-E2B는 Mixture-of-Experts 계열의 PLE 기술로 추론 시 2.3B 파라미터만 활성화되면서도 5.1B급 표현력을 제공한다는 점에서 포함했으며, 이 아키텍처 특성이 결과 해석에 미치는 영향은 제5장 5.3에서 별도로 논의한다.

### 3.4 데이터셋 및 전처리

**Table 3.2. 대상 데이터셋**

| 데이터셋 | 이미지 수 | QA 쌍 | 언어 | 도메인 | 질문 유형 |
|----------|:---:|:---:|:---:|:---:|:---:|
| PathVQA | 4,998 | 32,799 | 영어 | 병리학 | Open+Closed (7종) |
| SLAKE | 642 | 14,028 | 영어+중국어 | 방사선/CT | Open+Closed |
| VQA-RAD | 315 | 2,248 | 영어 | 방사선 | Open+Closed |

각 데이터셋은 공식 train/val/test split을 그대로 사용했다.

**데이터 오염 통제**: PathVQA(2018)·SLAKE(2021)·VQA-RAD(2018)는 모두 대상 모델들의 사전훈련 시점(2025-2026) 이전에 공개되어, 사전훈련 데이터 오염(pretraining data contamination) 가능성을 배제할 수 없다. 본 연구는 Min-K% Probability Attack(Shi et al., "Detecting Pretraining Data from Large Language Models", arXiv:2310.16789, ICLR 2024)으로 이를 능동 측정한다. 각 샘플 정답 텍스트의 token-level log-probability 하위 K%(K=20) 평균을 contamination indicator로 사용하며(사전훈련에 노출된 샘플일수록 평균 확률이 높다는 이론에 기반), 데이터셋 내 상위 5% 이상치를 오염 의심 샘플로 분류한 뒤 이를 제거한 축소 샘플셋으로 주요 결론(RQ1)을 재검증한다(제4장 4.1.1 오염 강건성 검증). 절차 및 해석 기준(원본-축소 결과 차이 1%p 미만은 강건, 1-5%p는 한계점 명시, 5%p 초과는 결론 재검토)은 `scripts/measure_contamination.py`로 구현했다.

### 3.5 실험 1: 제로샷 베이스라인 평가 (Phase 1)

4개 모델 × 3개 데이터셋 = 12개 조건에 대해 파인튜닝 이전 제로샷 성능을 측정했다. 평가는 greedy 디코딩을 사용하므로 결정적(deterministic)이다 — 시드를 바꿔도 결과가 동일하여 반복 시행이 무의미하므로, 단일 시드(42)로 평가하고 불확실성은 각 조건 per-sample 정오 판정에 대한 부트스트랩 95% 신뢰구간으로 보고한다. 모델 간 비교는 4개 모델이 동일한 테스트셋으로 평가되는 짝지은(paired) 구조이므로, 독립표본을 가정하는 ANOVA 대신 Cochran's Q 검정(공유 테스트셋 이진 정오, H0: 정확도 동일)과 McNemar 쌍별 사후검정(Bonferroni 보정)을 사용한다.

측정 지표는 Closed-ended accuracy(선택형), Open-ended accuracy(정답 토큰 매칭) 및 BERTScore F1(roberta-large), 각 정확도의 부트스트랩 95% CI, 응답 시간(ms/문항), Peak VRAM(MB)이다.

### 3.6 실험 2: QLoRA 파인튜닝 (Phase 2)

**Table 3.3. 기본 QLoRA 설정**

| 파라미터 | 값 |
|----------|-----|
| Quantization | NF4 (4-bit NormalFloat) |
| LoRA Rank | 64 (Ablation B로 확정) |
| LoRA Alpha | 128 |
| LoRA Dropout | 0.05 |
| Target Modules | all-linear (Ablation C로 확정) |
| Learning Rate | 2e-4 |
| Batch Size | 1 (gradient accumulation 8, effective batch = 8) |
| Optimizer | paged_adamw_8bit |

학습 예산은 목표 3 epoch였으나 클라우드 GPU 시간·비용 제약으로 `max_steps=500`(조건당 samples_seen = 4,000 고정) 상한을 적용했다. 데이터셋 크기와 무관하게 학습량이 고정되므로 소형 VQA-RAD는 약 2 epoch 이상, 중형 SLAKE·대형 PathVQA는 1 epoch 미만만 학습되며, 이 비대칭이 결과 해석에 미치는 영향은 제5장 5.3에서 논의한다.

**실험 조건**: 4개 모델 × 3개 데이터셋 = 12개 조건, 각 조건 3회 반복(seed 42/123/456). LoRA rank=64·target=all-linear는 아래 세 Ablation Study의 결과로 확정한 값이며 상세 결과는 제4장 4.2.2-4.2.4에서 다룬다.

- **Ablation A (데이터 크기 영향)**: PathVQA·Qwen3-VL-2B 고정, 학습 데이터 비율 5/10/25/50/100%
- **Ablation B (LoRA Rank 영향)**: PathVQA·Qwen3-VL-2B 고정, rank ∈ {4, 8, 16, 32, 64}
- **Ablation C (Target Module 영향)**: PathVQA·Qwen3-VL-2B 고정, {q/v_proj} vs {q/k/v/o_proj} vs {all-linear}

**Catastrophic Forgetting 측정**은 두 가지로 이중 측정한다. (A) 범용 능력 변화: VQAv2 validation subset(2,000샘플)에서 파인튜닝 전/후 정확도 감소율을 12개 조건 전체에서 측정. (B) 의료 도메인 내 cross-dataset 일반화: 훈련 데이터셋과 다른 데이터셋으로 평가(예: PathVQA 학습 → SLAKE/VQA-RAD 평가)하여 12개 조건 × 2개 cross-dataset = 24회 추가 평가를 수행한다. PathVQA(병리)와 SLAKE/VQA-RAD(방사선)는 이미지 도메인 자체가 다르므로, (B)의 결과는 엄밀한 CF보다는 도메인 일반화 격차로 해석한다(제4장 4.2.5).

### 3.7 실험 3: 자율 하이퍼파라미터 최적화 (Phase 3)

**Table 3.4. Phase 3 탐색 공간**

| 파라미터 | 탐색 범위 | 타입 |
|----------|----------|------|
| lora_rank | {4, 8, 16, 32, 64} | 이산 |
| lora_alpha | rank × {1, 2, 4} | 이산 |
| learning_rate | [1e-5, 5e-4] | 연속(로그스케일) |
| batch_size | {1, 2, 4} | 이산 |
| grad_accum_steps | {4, 8, 16} | 이산 |
| warmup_ratio | [0.0, 0.1] | 연속 |
| weight_decay | [0.0, 0.1] | 연속 |
| lora_targets | {minimal, medium, full} | 범주형 |

**비교 대상 4개 전략**: Manual(연구자 기본값, 1회) / Random Search(무작위 샘플링) / Optuna·TPE(베이지안 최적화) / Autoresearch(LLM 에이전트 자율 탐색). Autoresearch는 (1) 이전 실험 결과(results.tsv)를 읽고 (2) 다음 설정을 자연어 근거와 함께 제안(config.yaml + rationale.md)한 뒤 (3) git commit, (4) 고정 학습 실행, (5) 검증셋 평가, (6) 성능 개선 시 유지·아니면 폐기하는 루프를 반복한다.

전 trial 공통으로 동일 모델(Phase 2 최적 모델)·동일 데이터셋(PathVQA)·고정 `max_steps=200`(안전장치용 wall-clock 상한 `time_budget_min`은 실험 통제 변수가 아닌 이상 조합 방지용)을 사용해 학습량을 통제한다. 당초 설계는 Manual 10 + Random Search 400 + Optuna 400 + Autoresearch 400 = 총 1,210 trial(각 전략 40 trial × 10회 독립 반복)이었으나, 로컬 듀얼 GPU 스모크 테스트 실측 결과 학습(train)뿐 아니라 검증·최종 테스트 평가 시간까지 합산한 wall-clock 기준으로 GPU 2장 병렬 실행 시에도 원안 규모는 약 24~25일이 소요될 것으로 재추정되었다. 이에 통계 검정 단위인 반복 횟수(10회, run-level 검정력의 근거)는 그대로 유지하고, 대신 전략당 탐색 trial 수를 40에서 20으로 축소하여 총 소요 시간을 약 12.8일로 절반 단축했다. **최종 실행 규모는 Manual 10 + Random Search 200 + Optuna 200 + Autoresearch 200 = 총 610 trial(각 전략 20 trial × 10회 독립 반복)이다.** 이 축소는 전략당 탐색하는 하이퍼파라미터 조합의 다양성을 절반으로 낮추는 트레이드오프가 있으나, run-level 통계 검정(10회 독립 반복)의 타당성 자체에는 영향을 주지 않는다.

**통계 검증은 trial-level이 아닌 run-level에서만 수행**한다. Autoresearch와 Optuna는 순차 최적화 특성상 동일 run 내 trial 간 의존성이 있어(trial t의 결과가 t+1의 제안에 영향), 독립 관측치 가정이 위반되기 때문이다. 검정 단위는 각 전략의 10회 독립 반복에서 나온 10개 최종 성능값이며, Kruskal-Wallis test(4그룹 비교)와 Mann-Whitney U test(Autoresearch vs Optuna 쌍별), BCa Bootstrap 95% CI를 사용한다. Trial-level 데이터는 anytime performance curve 등 시각화에만 사용한다.

### 3.8 평가 지표 및 통계 분석 방법

#### 3.8.1 BERTScore 이중 보고

Open-ended 응답은 Exact Match와 BERTScore F1을 함께 보고한다. 범용 기준(roberta-large, threshold ≥ 0.7)을 정확도·통계 검정의 유일한 결정 지표(primary)로 삼고, 의료 특화 기준(BioBERT, dmis-lab/biobert-v1.1)은 보조 지표로만 병기하여 이중 게이팅(두 지표 모두 통과해야 정답 처리)을 하지 않는다.

#### 3.8.2 Catastrophic Forgetting 이중 측정

3.6에서 기술한 (A) VQAv2 기준 범용 능력 변화와 (B) cross-dataset 일반화 격차를 함께 보고하여, 단일 지표로는 포착하기 어려운 파인튜닝의 부작용을 다각도로 측정한다.

#### 3.8.3 임상적 의미 분석 (WCA + ECE)

단순 accuracy는 의료 AI의 임상적 가치를 온전히 포착하지 못한다는 문제의식에서, PathVQA의 7개 질문 유형(diagnosis, location, measurement, description, temporal, yes_no, unknown) 라벨을 이용해 Weighted Clinical Accuracy(WCA)를 산출한다.

`WCA = Σ(유형별 정확도 × 가중치) / Σ가중치`

가중치는 임상 중요도에 따라 diagnosis=1.0(진단 오류는 치료 방향에 직접 영향) > location=0.8 > measurement=0.7 > description=0.6 > temporal=yes_no=0.5(정보량이 제한적인 이진 판단) 순으로 부여했다. 다만 이 가중치는 외부 임상 문헌이나 Delphi 합의 없이 연구자가 임의로 설정한 것으로, 절대적 임상 중요도의 척도로 해석할 수 없으며 primary 지표(정확도·BERTScore)를 보완하는 참고용 보조 지표로만 제한적으로 사용한다.

Expected Calibration Error(ECE, Guo et al., ICML 2017)는 모델의 예측 확신도(confidence)가 실제 정확도와 얼마나 일치하는지를 측정하는 지표로 함께 제시할 계획이었으나, 현재 평가 파이프라인이 per-sample confidence를 저장하지 않아 산출하지 못했다(제5장 5.3 한계점).

#### 3.8.4 Robust 통계 (Bootstrap + Mixed-Effects)

Phase 2의 파인튜닝 효과 검정은 n=9(3 데이터셋 × 3 시드)의 표본 한계를 고려하여 세 가지 방법을 병행한다: (1) paired t-test + Cohen's d(관행적 비교), (2) BCa Bootstrap(10,000 resample)으로 산출한 Cohen's d의 95% CI(robust 추정), (3) Wilcoxon signed-rank test(비모수 검정). 추가로 4개 모델을 통합한 Mixed-Effects Model(`accuracy ~ condition + dataset`, group=seed)을 보조적으로 적용하되, 모델 간 이질적 효과가 pooled 추정에서 상쇄될 수 있음을 감안해 모델별 3중 검증 결과를 1차 근거로, MEM 결과는 보조 설명으로만 사용한다(제4장 4.2.1에서 실제로 이 현상이 관측됨).

#### 3.8.5 데이터 오염 통제 (Min-K% Probability)

방법론은 3.4에서 기술한 바와 같다. 강건성 검증 결과는 제4장 4.1.1에서 다룬다.

---

## 제4장. 실험 결과 및 분석

### 4.1 Phase 1: 제로샷 베이스라인 결과

Phase 1은 파인튜닝 이전 4개 경량 VLM(Gemma4-E2B, Qwen2.5-VL-3B, Qwen3-VL-2B, SmolVLM2-2.2B)의 제로샷 성능을 3개 의료 VQA 데이터셋(PathVQA, SLAKE, VQA-RAD)에서 측정하여 RQ1("파인튜닝 없이도 경량 VLM이 의료 VQA에서 실용적 성능을 보이는가, 모델 간 차이는 유의한가")에 답한다. 평가는 시드 42, 데이터셋별 전체 test split(PathVQA 6,719 / SLAKE 1,061 / VQA-RAD 451문항)에서 수행했다.

#### 4.1.1 모델별 성능 비교

**Table 4.1. 제로샷 베이스라인 결과 (모델 × 데이터셋)**

| 모델 | 데이터셋 | Closed Acc | Open Acc | Overall Acc | BERTScore F1 | 응답시간(ms) | Peak VRAM(MB) |
|------|:--------:|:----------:|:--------:|:-----------:|:-------------:|:------------:|:-------------:|
| Gemma4-E2B | PathVQA | 0.1633 | 0.0477 | **0.1055** | 0.8069 | 892.6 | 13,932.7 |
| Gemma4-E2B | SLAKE | 0.6394 | 0.4178 | **0.4920** | 0.8679 | 689.8 | 13,927.3 |
| Gemma4-E2B | VQA-RAD | 0.4502 | 0.3100 | **0.3880** | 0.8373 | 721.5 | 13,929.1 |
| Qwen2.5-VL-3B | PathVQA | 0.6130 | 0.0354 | **0.3245** | 0.8613 | 483.4 | 7,581.3 |
| Qwen2.5-VL-3B | SLAKE | 0.7465 | 0.4632 | **0.5580** | 0.9359 | 254.8 | 7,561.9 |
| Qwen2.5-VL-3B | VQA-RAD | 0.6614 | 0.2800 | **0.4922** | 0.8886 | 310.6 | 7,580.5 |
| Qwen3-VL-2B | PathVQA | 0.6336 | 0.0605 | **0.3472** | 0.8487 | 419.1 | 4,527.3 |
| Qwen3-VL-2B | SLAKE | 0.7915 | 0.4575 | **0.5693** | 0.9081 | 344.9 | 4,412.9 |
| Qwen3-VL-2B | VQA-RAD | 0.7211 | 0.2250 | **0.5011** | 0.8894 | 245.7 | 4,428.5 |
| SmolVLM2-2.2B | PathVQA | 0.5892 | 0.0274 | **0.3085** | 0.8557 | 666.0 | 6,021.3 |
| SmolVLM2-2.2B | SLAKE | 0.6648 | 0.3598 | **0.4618** | 0.9130 | 774.7 | 5,991.2 |
| SmolVLM2-2.2B | VQA-RAD | 0.6574 | 0.3150 | **0.5055** | 0.9029 | 756.7 | 5,996.2 |

> Overall Acc = Closed(객관식형)과 Open(주관식형) 문항을 합산한 정확도. Open 문항은 BERTScore(roberta-large 기반, 임계값 방식) 채점, Closed 문항은 정답 문자열 일치 채점(설계서 §4.3 참조).

전체 데이터셋을 합산(pooled)한 정확도 기준으로 4개 모델의 성능은 다음과 같다(n=8,231, 3개 데이터셋 합산 문항 수).

**Table 4.1b. Pooled 정확도 및 통계 검정**

| 순위 | 모델 | Pooled Overall Acc | 95% CI |
|:---:|------|:-------------------:|:------:|
| 1 | **Qwen3-VL-2B** | **0.3843** | [0.3740, 0.3947] |
| 2 | Qwen2.5-VL-3B | 0.3637 | [0.3533, 0.3740] |
| 3 | SmolVLM2-2.2B | 0.3391 | [0.3289, 0.3495] |
| 4 | Gemma4-E2B | 0.1708 | [0.1627, 0.1790] |

Cochran's Q 검정 결과 4개 모델 간 정오답 패턴은 pooled 기준으로 통계적으로 유의하게 다르다(Q = 1904.28, df = 3, p < .001). 데이터셋별로 개별 검정해도 세 데이터셋 모두 유의했다(PathVQA: Q = 2067.08, p < .001 / SLAKE: Q = 71.34, p < .001 / VQA-RAD: Q = 27.18, p < .001).

McNemar 쌍별 사후검정(Bonferroni 보정)에서, **Gemma4-E2B는 pooled 기준 및 PathVQA·VQA-RAD 개별 데이터셋에서 나머지 세 모델 전부와 유의하게 낮은 성능**을 보였다(해당 비교 전부 p(adj) < .005). 다만 SLAKE에서는 예외적으로 Gemma4-E2B와 SmolVLM2-2.2B 간 차이가 유의하지 않았다(0.492 vs 0.462, p(adj) = 0.326) — SLAKE에서 Gemma4-E2B의 상대적 성능이 다른 두 데이터셋보다 나은 편이라 이 한 쌍만 통계적으로 구분되지 않는다. 최상위권인 Qwen2.5-VL-3B·Qwen3-VL-2B·SmolVLM2-2.2B 세 모델 사이에서도 데이터셋에 따라 유의성이 갈렸다 — pooled 기준으로는 세 모델 모두 서로 유의하게 달랐으나(p(adj) < .001), SLAKE·VQA-RAD 개별 데이터셋에서는 Qwen2.5-VL-3B와 Qwen3-VL-2B 간 차이가 유의하지 않았다(각각 p(adj) = 1, p(adj) = 1). 즉 **Qwen3-VL-2B가 pooled 최고 성능(best model)이지만, Qwen2.5-VL-3B와는 데이터셋에 따라 통계적으로 구분되지 않는 수준으로 근접**한다.

**오염 강건성 검증**: Phase 1.5 Min-K% Probability 분석(K=20%, 데이터셋 내 상위 5% 이상치)으로 식별한 사전훈련 노출 의심 샘플(PathVQA 1,020개·SLAKE 233개·VQA-RAD 73개, 4모델 합집합)을 제거하고 동일 검정을 재수행한 결과, 세 데이터셋 모두 Cochran's Q가 여전히 유의했고 **pooled 모델 순위와 best model(Qwen3-VL-2B)이 그대로 유지**되었다(원본 0.3849 → 축소셋 0.3041, 절대 정확도는 하락하나 순위 불변). 따라서 본 절의 결론은 잠재적 데이터 오염에 대해 강건하다(상세: `results/phase1_baseline/phase1_robustness.md`).

#### 4.1.2 데이터셋별 난이도 분석

세 데이터셋의 난이도는 모델과 무관하게 뚜렷한 순서를 보인다. 4개 모델의 Overall Acc 평균을 데이터셋별로 비교하면 **PathVQA(평균 0.271) < VQA-RAD(평균 0.472) < SLAKE(평균 0.520)** 순으로, PathVQA가 나머지 두 데이터셋보다 확연히 어렵다.

이 격차는 특히 **Open(주관식) 문항**에서 두드러진다. 세 데이터셋 모두 Closed 문항 정확도(0.45~0.79)에 비해 Open 문항 정확도(0.03~0.46)가 크게 낮지만, 그 낙폭이 PathVQA에서 가장 심하다 — PathVQA의 Open Acc는 전 모델에서 0.027~0.061 수준으로, SLAKE(0.360~0.463)·VQA-RAD(0.225~0.315)와 비교해 한 자릿수 이상 낮다. PathVQA는 병리 조직 영상에 대한 세부 소견을 자유 서술로 요구하는 문항 비중이 높아, 객관식형 문항이 상대적으로 많은 SLAKE·VQA-RAD보다 주관식 생성형 응답의 난이도가 큰 것으로 해석된다.

응답 시간·VRAM은 모델 크기에 비례하며 데이터셋 간 차이는 미미하다(동일 모델 내 데이터셋 간 VRAM 변동 < 1%). 다만 응답 시간은 데이터셋별 문항 특성(설명 요구 길이 등)에 따라 모델별로 어느 정도 편차가 있다(예: Gemma4-E2B는 PathVQA에서 892.6ms로 SLAKE/VQA-RAD보다 오래 걸림).

#### 4.1.3 오류 유형 분석

WCA(Weighted Clinical Accuracy) 분석을 위해 PathVQA(seed 42) 문항을 7개 임상 질문 유형(diagnosis·location·measurement·description·temporal·yes_no·unknown)으로 분류하고 유형별 정확도를 산출했다(가중치는 5.3에서 논의하는 대로 임상 문헌 검증 없는 임시 척도이며, 여기서는 참고용 오류 패턴 식별에만 사용한다).

**Table 4.1c. PathVQA 질문 유형별 정확도**

| 유형 | 샘플 수 | Gemma4-E2B | Qwen2.5-VL-3B | Qwen3-VL-2B | SmolVLM2-2.2B |
|------|:-------:|:----------:|:--------------:|:-----------:|:--------------:|
| diagnosis | 23 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| location | 433 | 0.0647 | 0.1062 | 0.1478 | 0.0762 |
| measurement | 33 | 0.2121 | 0.0909 | 0.1515 | 0.0909 |
| description | 2,729 | 0.0425 | 0.0224 | 0.0447 | 0.0198 |
| temporal | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| yes_no | 3,362 | 0.1633 | 0.6130 | 0.6336 | 0.5892 |
| unknown | 130 | 0.0692 | 0.0692 | 0.0923 | 0.0154 |

가장 두드러진 패턴은 **diagnosis(진단) 및 temporal(시간 경과) 유형 문항에서 4개 모델 전부 0.0000의 정확도**를 보인다는 점이다. diagnosis는 WCA 가중치가 가장 높게 설정된(1.0) 임상적으로 중요한 유형임에도, 제로샷 상태의 모델들은 병리 소견으로부터 진단명을 도출하는 과제를 전혀 수행하지 못했다. description(자유 서술) 유형도 표본 수가 가장 많음(2,729개, 전체의 41%)에도 정확도가 0.02~0.04에 그쳐, PathVQA 전체 정확도를 끌어내리는 주된 요인으로 확인된다.

반대로 **yes_no(예/아니오) 유형은 상대적으로 높은 정확도**를 보이며(Qwen3-VL-2B 0.6336 등), 모델 간 순위도 4.1.1의 전체 순위(Qwen3-VL-2B > Qwen2.5-VL-3B > SmolVLM2-2.2B ≫ Gemma4-E2B)와 대체로 일치한다. 다만 Gemma4-E2B는 yes_no 유형에서도(0.1633) 나머지 세 모델(0.59~0.63)과 큰 격차를 보여, 4.1.1의 전반적 열위가 특정 유형에 국한되지 않고 전 유형에 걸친 것임을 확인할 수 있다.

종합하면, 제로샷 경량 VLM은 **이진 판별형(yes_no) 과제에는 어느 정도 대응 가능하나, 진단명 도출·자유 서술형 소견 생성처럼 임상적으로 중요도가 높은 개방형 과제에는 실질적으로 대응하지 못한다.** 이는 RQ2(파인튜닝 효과)의 필요성을 뒷받침하는 근거이며, 4.2에서 파인튜닝 이후 이 유형별 격차가 어떻게 변화하는지(4.2.5 임상적 의미 분석)와 연결하여 논의한다.

---

### 4.2 Phase 2: QLoRA 파인튜닝 결과

Phase 2는 4개 모델을 3개 데이터셋에 대해 QLoRA(rank=64, alpha=128, target=all-linear, `max_steps=500` 상한)로 개별 파인튜닝하고, RQ2("도메인 특화 파인튜닝이 zero-shot 대비 성능을 유의하게 향상시키는가")를 검증한다. Main 36조건(4모델×3데이터셋×3시드) 외에, 최적 QLoRA 설정을 찾기 위한 Ablation A(데이터 비율)·B(LoRA rank)·C(target module) 39조건을 Qwen3-VL-2B·PathVQA 고정 조건에서 추가로 수행했다(총 75조건).

#### 4.2.1 Base vs Fine-tuned 성능 향상

**Table 4.2. 모델별 파인튜닝 효과 (paired, n=9 = 3데이터셋×3시드)**

| 모델 | Base Acc | FT Acc | Cohen's d | d 95% CI (BCa) | paired t-test p | Wilcoxon p |
|------|:--------:|:------:|:---------:|:---------------:|:----------------:|:----------:|
| Qwen2.5-VL-3B | 0.4582 | 0.5749 | **+2.646** | [1.953, 4.723] | < .001 | .0039 |
| Qwen3-VL-2B | 0.4725 | 0.5845 | **+1.620** | [0.932, 3.153] | .0013 | .0039 |
| SmolVLM2-2.2B | 0.4253 | 0.4036 | **-2.284** | [-3.160, -1.552] | < .001 | .0039 |
| Gemma4-E2B | 0.3285 | 0.2288 | -0.652 | [-1.599, 0.032] | .0864 (n.s.) | .1289 (n.s.) |

파인튜닝 효과는 모델별로 이질적(heterogeneous)이다. **Qwen2.5-VL-3B·Qwen3-VL-2B는 파인튜닝으로 유의하게(p < .01, 큰 효과크기) 향상**되었으나, **SmolVLM2-2.2B는 오히려 유의하게 악화**되었고, Gemma4-E2B는 부정적 방향이나 통계적으로 유의하지 않았다. 세 검정(paired t-test·BCa Bootstrap Cohen's d·Wilcoxon)이 모델별로 일관된 결론을 내어 결과의 강건성을 뒷받침한다.

4개 모델을 구분하지 않고 합쳐 추정한 Mixed-Effects Model(`accuracy ~ condition + dataset`, group=seed)은 고정효과가 유의하지 않았다(계수 = 0.0268, p = .3629, ICC(seed) = 0.0, n = 72). 이는 계산 오류가 아니라, **모델 간 이질적 효과가 pooled 평균에서 상쇄**되기 때문이다 — 위 모델별 3중 검증 결과를 RQ2의 1차 근거로 삼고, MEM pooled 결과는 "모델 구분 없는 전체 효과는 이질성으로 인해 유의하지 않다"는 보조 설명으로만 인용한다(설계서 §5.3 한계점 반영).

#### 4.2.2 데이터 크기 영향 (Ablation A)

**Table 4.2a. 학습 데이터 비율별 성능 (Qwen3-VL-2B, PathVQA, 3시드 평균)**

| subset_ratio | 학습 샘플 수 | Overall Acc (평균) |
|:---:|:---:|:---:|
| 0.05 | 982 | 0.4150 |
| 0.10 | 1,965 | 0.4309 |
| 0.25 | 4,913 | 0.4357 |
| 0.50 | 9,827 | 0.4628 |
| 1.00 | 19,654 | **0.5019** |

학습 데이터 비율이 클수록 정확도가 단조 증가하며(0.05→1.0 구간에서 아직 성능 한계(ceiling)에 도달한 징후가 없음), 전 구간 실험 범위 내에서는 **전체 데이터(ratio=1.0) 사용이 최적**이다.

> **`train_time_min` 필드 버그(원인 확인됨)**: 위 표에는 포함하지 않았으나, `phase2_summary.csv`의 `train_time_min` 컬럼은 동일 조건에서 seed=42만 유독 크게 나온다(예: ratio=1.0에서 seed42=369.6분 vs seed123/456=약 28분). 각 조건의 `train_result.json`을 직접 대조한 결과, Trainer가 내부적으로 측정하는 `train_runtime_sec`는 전 시드에서 27~29분으로 **일관되게 정상**이었고, 문제는 감싸는 스크립트가 wall-clock으로 재는 `train_time_min`에만 있었다. 이 버그는 (모델, 데이터셋) 조합의 전처리 캐시를 **처음 만드는 조건**에서만 나타나며(캐시 생성 1회성 비용이 wall-clock 시간에 합산됨), Main 36조건 전수 대조에서도 동일 패턴이 확인됐다(예: `qwen25-vl-3b/pathvqa/seed42` 395.4분 vs 실제 44.9분). **정확도·loss 등 학습 결과 지표는 이 버그와 무관하며, 본 절의 정확도 기반 결론에는 영향이 없다.** 시간 비교가 필요한 향후 분석(예: Phase 3 비용 산정)에서는 `train_time_min` 대신 `train_runtime_sec`를 사용해야 한다.

#### 4.2.3 LoRA Rank 영향 (Ablation B)

**Table 4.2c. LoRA rank별 성능 (Qwen3-VL-2B, PathVQA, 3시드 평균)**

| LoRA Rank | Peak VRAM (MB) | Overall Acc (평균) |
|:---:|:---:|:---:|
| 4 | 3,870.6 | 0.4733 |
| 8 | 3,875.2 | 0.4907 |
| 16 | 3,884.4 | 0.5020 |
| 32 | 3,902.8 | 0.5172 |
| 64 | 3,918.8 | **0.5210** |

Rank가 클수록 성능이 단조 증가하나 32→64 구간에서 증가폭이 둔화되고(16→32구간 +0.0152 대비 32→64구간 +0.0038), VRAM 증가는 rank 4→64 전 구간에서 1.3%(3,870.6→3,918.8MB) 수준으로 미미하다. VRAM 비용 대비 성능 이득이 여전히 양(+)이므로 **rank=64를 채택**한다(Phase 2 main 실험에 반영된 설정).

#### 4.2.4 Target Module 영향 (Ablation C)

**Table 4.2d. Target module 범위별 성능 (Qwen3-VL-2B, PathVQA, 3시드 평균)**

| 설정 | Target Modules | 학습 파라미터 비율 | Overall Acc (평균) |
|------|----------------|:---:|:---:|
| minimal | q_proj, v_proj | 0.21% | 0.5015 |
| medium | q/k/v/o_proj | 0.43% | 0.5155 |
| **full** | all-linear | 1.55% | **0.5400** |

Target module 범위가 넓을수록(더 많은 linear layer에 LoRA 적용) 성능이 단조 증가한다. **full(all-linear)이 세 설정 중 최고 성능**으로, Phase 2 main 실험의 기본 설정(rank=64, alpha=128, **target=all-linear**)으로 채택했다. 다만 이 3개 축(비율·rank·target)은 각각 나머지 두 축을 고정한 채 독립적으로만 검증했으며, 세 최적값을 동시 적용한 조합 자체를 별도 검증하지는 않았다(설계서 §5.3 한계점).

#### 4.2.5 Catastrophic Forgetting 분석 (VQAv2 + cross-dataset)

**(A) 범용 능력 상실 — VQAv2 validation subset(2,000샘플) 기준**

**Table 4.2b. 모델별 VQAv2 성능 저하율 (n=9 = 3데이터셋×3시드 평균)**

| 모델 | 평균 저하율(%) | 표준편차 | 범위 |
|------|:---:|:---:|:---:|
| **Gemma4-E2B** | **51.50** | 4.74 | 43.50 ~ 57.49 |
| Qwen3-VL-2B | 7.23 | 3.11 | 3.29 ~ 10.47 |
| Qwen2.5-VL-3B | 4.42 | 3.60 | -0.34 ~ 8.30 |
| SmolVLM2-2.2B | 0.49 | 0.29 | 0.15 ~ 0.96 |

VQAv2 기준 일반 능력 상실 정도는 모델별로 극명하게 갈린다. **Gemma4-E2B는 파인튜닝 후 VQAv2 성능이 평균 51.5% 하락**하여 뚜렷한 catastrophic forgetting을 보이는 반면, **SmolVLM2-2.2B는 사실상 저하가 없다(평균 0.49%)**. 흥미롭게도 이 순서는 4.2.1의 도메인 성능 향상 순위와 정반대 방향으로 겹친다 — 도메인 향상이 가장 컸던 Qwen 계열은 일반 능력도 어느 정도(4~7%) 내어주는 반면, Gemma4-E2B는 도메인 성능(4.2.1, 유의한 향상 없음)도 일반 능력(51.5% 손실)도 모두 잃는 이중 손실을 보이고, SmolVLM2-2.2B는 일반 능력은 지키지만 도메인 성능이 유의하게 악화된다. 이 상관관계는 관측된 패턴이며, 본 연구가 인과관계를 별도로 검증한 것은 아니다.

**(B) Cross-dataset 일반화 격차 — 학습 도메인과 다른 데이터셋 평가 시 변화율**

**Table 4.2b-B. 모델별 cross-dataset 성능 변화율 (n=18 = 2평가셋×3학습셋×3시드)**

| 모델 | 평균 변화율(%) | 표준편차 | 양(+)의 방향 비율 |
|------|:---:|:---:|:---:|
| Gemma4-E2B | +73.94 | 79.97 | 15/18 |
| Qwen2.5-VL-3B | +0.38 | 4.98 | 9/18 |
| Qwen3-VL-2B | -9.46 | 9.23 | 3/18 |
| SmolVLM2-2.2B | **-31.53** | 17.49 | 0/18 |

설계서 §5.3에서 이미 정의한 대로, PathVQA(병리 조직)와 SLAKE/VQA-RAD(방사선 영상)는 이미지 도메인 자체가 다르므로 (B)는 엄밀한 CF가 아니라 **도메인 일반화 격차**로 해석한다. SmolVLM2-2.2B는 한 데이터셋에 특화 학습할수록 다른 데이터셋 성능이 뚜렷하게 하락하는(평균 -31.5%, 18개 조건 전부 음의 방향) 반면, Gemma4-E2B는 오히려 평균적으로 크게 상승한다(+73.9%, 다만 표준편차 79.97로 변동성이 극심함 — 최저 -8.06%에서 최고 +205.56%까지 분포). Gemma4-E2B의 이 큰 양의 평균은 4.1에서 확인한 낮은 zero-shot 베이스라인(특히 PathVQA 0.10 수준)에서 기인한 바닥효과(floor effect)일 가능성이 있다 — base 정확도가 매우 낮은 상태에서는 어느 방향의 파인튜닝이든 상대적 변화율이 과장되기 쉽다. 전체 상세 72개 조건은 `results/phase2_finetune/cross_dataset_cf_summary.md` 참조.

---

### 4.3 Phase 3: 자율 하이퍼파라미터 최적화 결과

Phase 3은 Phase 2에서 최고 성능을 보인 Qwen3-VL-2B를 PathVQA에 고정하고, 하이퍼파라미터 탐색 전략 4종(Manual·Random Search·Optuna(TPE)·Autoresearch(LLM 에이전트))을 동일 조건에서 비교하여 RQ3("LLM 에이전트의 자율 탐색이 기존 HPO 기법 대비 경쟁력 있는 성능에 도달하는가")에 답한다. 전 trial 공통으로 `max_steps=200`으로 학습량을 통제했으며, 최종 실행 규모는 Manual 10 + Random 200 + Optuna 200 + Autoresearch 200 = 총 610 trial(전략당 20 trial × 10회 독립 반복, Manual은 반복당 1 trial)이다.

§3.7에서 기술한 대로 **통계 검정은 trial-level이 아닌 run-level에서만 수행**한다. Optuna와 Autoresearch는 순차 최적화 특성상 동일 run 내 trial 간 의존성이 있어 개별 trial을 독립 관측치로 취급할 수 없기 때문이다. 검정 단위는 각 전략의 10회 독립 반복에서 산출한 **반복별 최고 val_accuracy 10개**이며(Manual은 반복당 trial이 1개이므로 그 값 자체가 run-level 값이 된다), trial-level 데이터는 4.3.3~4.3.4의 탐색 과정 묘사에만 사용한다.

#### 4.3.1 전략별 최종 성능 비교 (run-level)

**Table 4.3. HPO 전략별 run-level 성능 (n=10 = 독립 반복 10회의 반복별 최고 val_accuracy)**

| 전략 | n | 평균 val_accuracy | 95% CI (Bootstrap) |
|------|:-:|:-----------------:|:------------------:|
| **Optuna (TPE)** | 10 | **0.4490** | [0.4368, 0.4594] |
| Random Search | 10 | 0.4186 | [0.4106, 0.4274] |
| Autoresearch (LLM) | 10 | 0.4184 | [0.4064, 0.4328] |
| Manual | 10 | 0.3776 | [0.3760, 0.3794] |

> run-level 값 = 각 독립 반복에서 완료된 trial 중 최고 val_accuracy. 실패(status ≠ completed) trial은 집계에서 제외한다. Manual 전략에서는 반복 6~9에서 총 10건의 실패 후 재시도 기록이 있으나 전 반복이 최종적으로 1건씩 정상 완료되어 run-level 표본 수 10은 유지된다.

4개 전략 간 run-level 성능 차이는 **Kruskal-Wallis 검정에서 통계적으로 유의**했다(H = 27.92, df = 3, p < .001).

RQ3의 핵심 쌍별 비교인 **Autoresearch vs Optuna**의 Mann-Whitney U 검정 결과는 U = 16.00, p = .0112, rank-biserial r = **-0.68**로 유의했다. 본 연구의 부호 규약(`src/evaluate/statistics.py`의 `run_mann_whitney`)에서 r > 0은 첫 표본(Autoresearch)이 둘째 표본(Optuna)보다 확률적으로 우세함을 의미하므로, **음의 r은 Optuna가 Autoresearch보다 유의하게 우수함**을 뜻한다.

이 방향성은 검정과 독립적으로 신뢰구간에서도 확인된다 — Optuna의 95% CI 하한(0.4368)이 Autoresearch의 상한(0.4328)보다 높아 **두 구간이 겹치지 않는다.** 반면 Autoresearch(0.4184)와 Random Search(0.4186)는 평균이 사실상 동일하고 신뢰구간이 대부분 겹쳐, 두 전략은 통계적으로 구분되지 않는다.

종합하면 본 실험 조건에서 전략 간 서열은 **Optuna > (Random ≈ Autoresearch) > Manual**이다. 즉 **LLM 에이전트의 자율 탐색은 확립된 베이지안 최적화(TPE)에 유의하게 미치지 못했으며, 무작위 탐색 대비로도 이점을 보이지 못했다.** 다만 Autoresearch를 포함한 세 자동 탐색 전략은 모두 연구자 수동 설정(Manual, 0.3776)을 상회하여, 자동 하이퍼파라미터 탐색 자체의 유효성은 확인되었다.

#### 4.3.2 전략별 탐색 결과 하이퍼파라미터

**Table 4.3a. 전략별 최고 성능 trial의 하이퍼파라미터 구성**

| 전략 | rank | alpha | learning_rate | batch | grad_accum | warmup | weight_decay | targets | val_acc | closed | open |
|------|:----:|:-----:|:-------------:|:-----:|:----------:|:------:|:------------:|:-------:|:-------:|:------:|:----:|
| Manual | 16 | 32 | 2.00e-4 | 1 | 8 | 0.030 | 0.010 | minimal | 0.3840 | 0.7213 | 0.0625 |
| Random | 64 | 256 | 2.08e-4 | 2 | 16 | 0.072 | 0.013 | full | 0.4440 | 0.7951 | 0.1094 |
| **Optuna** | 32 | 128 | 4.91e-4 | 4 | 16 | 0.089 | 0.054 | full | **0.4700** | 0.8115 | 0.1445 |
| Autoresearch | 64 | 256 | 2.00e-4 | 4 | 16 | 0.050 | 0.010 | full | 0.4640 | 0.7992 | 0.1445 |

> 본 표는 각 전략이 도달한 최선의 단일 설정을 기술한 것으로 표본 수가 1이며, 전략 간 우열의 근거로는 사용하지 않는다(우열 판단은 4.3.1의 run-level 검정에 따른다). closed/open은 해당 trial의 `val_closed_acc` / `val_open_acc`이다.

네 전략 전부에서 **closed 정확도(0.72~0.81)와 open 정확도(0.06~0.14)의 격차가 유지**된다. 탐색으로 얻은 전체 정확도 향상(Manual 0.3840 → Optuna 0.4700)이 주로 closed 문항에서 발생했으며, open 정확도는 최고 설정에서도 0.1445에 그친다. 이 격차는 Phase 1에서 관측된 개방형 응답의 취약성(4.1.2, 4.1.3)이 하이퍼파라미터 최적화로 해소되지 않았음을 보여준다(4.4.4에서 세 단계에 걸쳐 종합한다).

탐색 공간을 자유롭게 탐색한 세 전략(Random·Optuna·Autoresearch)은 모두 **`lora_targets=full`(all-linear)과 rank 32~64 영역으로 수렴**했다. 이는 Phase 2 Ablation B·C에서 rank=64와 target=all-linear가 최적이라고 확인한 결과(4.2.3, 4.2.4)와 독립적으로 일치하는 것으로, 서로 다른 실험 설계(고정 축 ablation vs 자유 탐색)가 같은 영역을 지목했다는 점에서 Phase 2 결론을 보강한다. 한편 최고 성능을 낸 Optuna의 설정은 rank=32에 상대적으로 높은 학습률(4.91e-4)과 weight decay(0.054)를 결합한 조합으로, 나머지 전략이 도달하지 못한 영역이었다.

#### 4.3.3 탐색 궤적 분석 (trial-level, 기술적 묘사 전용)

**Table 4.3b. 전략별 최고 성능 도달 시점 (10회 반복 기준)**

| 전략 | 최종 최고(중앙값) | 최고 도달 trial(중앙값) | 도달 trial IQR |
|------|:----------------:|:---------------------:|:-------------:|
| Manual | 0.3770 | 1.0 | — |
| Random Search | 0.4150 | 15.0 | [12.2, 17.0] |
| Optuna (TPE) | 0.4550 | 15.0 | [13.5, 18.0] |
| Autoresearch | 0.4060 | 17.5 | [11.5, 20.0] |

Autoresearch는 최고 성능 도달 trial의 중앙값이 17.5로 가장 늦고, **IQR 상한이 탐색 예산의 한계값인 20.0에 걸쳐 있다.** 이는 절반 가까운 run이 예산이 소진되는 시점까지도 여전히 성능을 개선하는 중이었음을 의미하며, 20 trial이라는 예산이 이 전략에는 부족했을 가능성을 시사한다(§3.7의 40→20 축소와 직결되며 5.3에서 한계점으로 논의한다). 다만 이는 관측된 정황이며, 예산을 늘렸을 때 실제로 Optuna를 따라잡는지는 본 연구가 검증하지 않았다.

**Table 4.3c. 전략별 trial-level 성능 분포 (200 trial 전수, 기술 통계)**

| 전략 | trial-level 평균 | 표준편차 | 총 학습시간(분) |
|------|:---------------:|:-------:|:--------------:|
| Manual | 0.3776 | 0.0031 | 228.3 |
| Random Search | 0.3672 | 0.0321 | 5,078.6 |
| Optuna (TPE) | 0.3905 | 0.0374 | 5,967.3 |
| Autoresearch | 0.3980 | **0.0145** | 5,450.0 |

주목할 점은 **Autoresearch가 trial-level 평균은 가장 높으면서(0.3980) 표준편차는 가장 낮다(0.0145)**는 것이다. 즉 이 전략은 개별 제안의 평균 품질은 우수하나 분산이 작아 극단적으로 좋은 설정에 도달하는 빈도가 낮았다. 반대로 Optuna는 표준편차가 가장 크다(0.0374). run-level 지표가 "20 trial 중 최고값"으로 정의되는 이상, 분산이 큰 탐색이 구조적으로 유리하다 — 4.3.1에서 관측된 두 전략의 역전(trial 평균은 Autoresearch가 높으나 run-level 최고값은 Optuna가 높음)은 이 분산 차이로 설명된다.

anytime performance 곡선(trial 진행에 따른 누적 최고 성능의 중앙값 및 IQR)은 `results/phase3_autoresearch/phase3_anytime.png`, 곡선의 원본 수치는 `phase3_anytime_curve.csv`에 있다.

#### 4.3.4 Autoresearch 에이전트의 탐색 행태

4.3.3에서 관측된 낮은 분산의 원인을 확인하기 위해, 각 반복에서 에이전트가 제안한 **고유 하이퍼파라미터 조합의 수**를 세었다.

**Table 4.3d. 반복당 고유 하이퍼파라미터 조합 수 (20 trial 중)**

| 전략 | 반복당 고유 조합 수 (평균) | 범위 |
|------|:------------------------:|:----:|
| Random Search | 20.0 / 20 | 20 ~ 20 |
| Optuna (TPE) | 20.0 / 20 | 20 ~ 20 |
| **Autoresearch** | **12.5 / 20** | 10 ~ 17 |

Random·Optuna는 10회 반복 전부에서 20개 trial이 모두 서로 다른 설정이었던 반면, **Autoresearch는 평균 12.5개의 고유 설정만 시도했다.** 즉 탐색 예산의 약 37%가 이미 시도한 설정의 재실행에 소비되었으며, 이 현상은 특정 반복에 국한되지 않고 10회 반복 전체에서 일관되게 관측되었다(최소 10개, 최대 17개).

이 패턴은 개별 반복의 제안 궤적에서 더 분명하게 드러난다. 예컨대 반복 8에서 에이전트는 초반 6개 trial 동안 rank를 16→64→32→8→32→64로 바꾸며 탐색하다가, 7번째 trial 이후 `rank=64, alpha=256, lr=2.0e-4, batch=2, targets=full` 조합에 고착되어 이를 11회 연속 제안했다. 주목할 점은 **동일한 설정이 반복 실행되는 동안 val_accuracy가 0.388~0.444 범위에서 변동**했다는 것이다 — 이 변동은 설정 차이가 아니라 학습·평가 과정의 확률적 변동이며, 에이전트가 이 노이즈를 성능 신호로 해석했을 가능성을 시사한다. 이 반복에서 에이전트가 고착을 벗어나 batch_size를 4로 바꾼 것은 마지막 trial이었고, 그 설정이 해당 반복의 최고 성능(0.4640)을 기록했다.

정리하면 Autoresearch의 낮은 성능은 제안하는 설정의 품질이 나빠서가 아니라(trial 평균은 오히려 가장 높다), **동일 설정의 반복 제안으로 실효 탐색 예산이 축소된 데 기인**하는 것으로 보인다. 다만 본 절은 결과 로그에 대한 사후 관찰이며, 에이전트의 내부 판단 근거를 직접 검증한 것은 아니다. 제안 근거(rationale) 원문은 부록 D에 수록한다.

#### 4.3.5 종합

RQ3에 대한 본 연구의 답은 **부정적**이다. LLM 에이전트의 자율 탐색(Autoresearch)은 동일한 탐색 공간과 동일한 20 trial 예산에서 **Optuna(TPE)에 유의하게 미치지 못했고**(p = .0112, r = -0.68), **무작위 탐색과도 통계적으로 구분되지 않았다**(0.4184 vs 0.4186). 다만 세 자동 탐색 전략이 모두 수동 설정을 상회했다는 점에서, 자동 탐색 자체의 유효성과 그 안에서 기존 기법(TPE)의 우위가 함께 확인되었다.

이 결과의 기저 원인으로 4.3.4는 **중복 제안에 의한 실효 탐색 예산 축소**(고유 설정 12.5/20)를 지목한다. 순차적 자기 개선을 전제로 설계된 에이전트가 오히려 조기 고착에 빠졌고, 동일 설정의 반복 실행에서 나타나는 확률적 변동을 개선 신호와 구분하지 못한 정황이 관측되었다.

본 결론은 다음 조건에 한정된다: 단일 모델(Qwen3-VL-2B) · 단일 데이터셋(PathVQA) · 전략당 20 trial 예산 · `max_steps=200` 통제 조건. 특히 4.3.3에서 확인된 대로 Autoresearch는 예산 소진 시점까지 개선 중이던 run이 상당수였으므로, 더 큰 탐색 예산에서의 결과는 달라질 수 있다(5.3).

실용적 관점에서는 비용도 함께 고려해야 한다. 총 학습시간은 Autoresearch 5,450분으로 Optuna 5,967분보다 다소 적었으나, Autoresearch는 매 trial마다 LLM API 호출 비용이 추가로 발생한다. 성능이 더 낮으면서 추가 비용이 드는 구성이므로, 본 실험 조건에서 Autoresearch를 Optuna 대신 선택할 근거는 확인되지 않았다.

---

### 4.4 종합 분석 및 논의

4.1~4.3은 각 실험 단계의 결과를 개별적으로 보고했다. 본 절은 세 단계를 관통하는 논점을 종합한다.

#### 4.4.1 단계 간 비교의 제약

본격적인 논의에 앞서 비교 가능성의 범위를 명확히 한다. 세 단계는 평가 조건이 서로 다르다.

| 단계 | 학습량 | 평가 대상 | 평가 표본 |
|------|--------|-----------|-----------|
| Phase 1 | 없음(zero-shot) | 3개 데이터셋 | 전체 test split (8,231문항) |
| Phase 2 | `max_steps=500` | 3개 데이터셋 × 3시드 | 전체 test split |
| Phase 3 | `max_steps=200` | PathVQA 단일 | validation, 최대 500샘플 |

따라서 **단계 간 정확도의 절대값을 직접 비교하는 것은 성립하지 않는다.** 예컨대 Phase 2의 파인튜닝 정확도(Qwen3-VL-2B 0.5845)와 Phase 3의 최고 val_accuracy(0.4700)의 차이는 성능 저하가 아니라 학습량(500 vs 200 steps)·평가 데이터셋(3개 평균 vs PathVQA 단독)·평가 split의 차이에서 비롯된 것이다. 이하의 논의는 절대 수치가 아니라 **단계 내부에서 관측된 순위·패턴·격차의 방향성**만을 교차 참조한다.

#### 4.4.2 추론 자원 소비와 성능의 비단조 관계

Phase 1의 가장 실용적인 함의는 **추론 시 자원 소비와 성능이 단조 관계를 이루지 않는다**는 점이다.

| 모델 | Pooled Acc (순위) | 활성 / 전체 파라미터 | Peak VRAM |
|------|:---------------:|:---:|:---------:|
| Qwen3-VL-2B | 0.3843 (1위) | 2B / 2B | 약 4,500 MB (최소) |
| Qwen2.5-VL-3B | 0.3637 (2위) | 3B / 3B | 약 7,580 MB |
| SmolVLM2-2.2B | 0.3391 (3위) | 2.2B / 2.2B | 약 6,000 MB |
| Gemma4-E2B | 0.1708 (4위) | 2.3B / **5.1B (MoE)** | 약 13,930 MB (최대) |

**최고 성능 모델(Qwen3-VL-2B)이 동시에 메모리를 가장 적게 쓰고 응답도 가장 빠르다**(4.1.1, 4.1.2). 반대로 메모리를 가장 많이 쓰는 Gemma4-E2B가 최하위이며, 그 격차는 나머지 세 모델과 통계적으로 뚜렷하다(4.1.1의 McNemar 사후검정). 즉 이 규모대에서는 활성 파라미터 수나 메모리 소비보다 아키텍처와 사전학습 구성이 의료 VQA 성능을 지배하며, 자원 제약 환경에서 성능과 효율을 동시에 만족하는 선택이 존재한다.

다만 **해석의 축을 명확히 해야 한다.** 위 네 모델 중 Gemma4-E2B만 Mixture-of-Experts(MoE) 구조로 활성 파라미터(2.3B)와 저장 파라미터(5.1B)가 다르다(3.3). 본 연구의 "경량 VLM" 선정은 소비자 GPU 구동 가능성이라는 목적에 따라 **활성 파라미터를 기준**으로 삼았고, 위 논의도 그 기준에서 성립한다. 반면 **저장 파라미터를 기준으로 보면 Gemma4-E2B가 가장 큰 모델**이므로, 그 최하위 성능은 "작은 모델이 큰 모델을 이겼다"가 아니라 "가장 큰 저장 규모의 모델이 최하위였다"로도 읽힌다. 따라서 본 절의 결론은 *추론 시 자원 소비 대비 성능*에 관한 것이며, 순수한 파라미터 규모와 성능의 관계로 확대 해석해서는 안 된다(5.3(11)).

이 결론은 Phase 1.5의 오염 강건성 검증(4.1.1)에서 사전훈련 노출 의심 샘플을 제거한 뒤에도 순위가 유지되었으므로, 데이터 오염으로 설명되지 않는다.

#### 4.4.3 파인튜닝 효과의 이질성과 모델 선택의 우선성

Phase 2에서 확인된 가장 중요한 사실은 **"파인튜닝하면 성능이 오른다"는 명제가 모델과 무관하게 성립하지 않는다**는 것이다(4.2.1). 주목할 점은 그 방향이 Phase 1의 zero-shot 순위와 겹친다는 것이다.

| 모델 | Phase 1 순위 | Phase 2 파인튜닝 효과 (Cohen's d) |
|------|:-----------:|:--------------------------------:|
| Qwen3-VL-2B | 1위 | +1.620 (유의) |
| Qwen2.5-VL-3B | 2위 | +2.646 (유의) |
| SmolVLM2-2.2B | 3위 | -2.284 (유의, 악화) |
| Gemma4-E2B | 4위 | -0.652 (비유의) |

zero-shot 상위 2개 모델은 파인튜닝으로 유의하게 향상된 반면, **하위 2개 모델은 개선되지 않거나 오히려 악화**되었다. 이는 파인튜닝이 부진한 기반 모델을 끌어올리는 수단으로 기능하지 못했음을 뜻하며, 실무적으로는 **기반 모델 선택이 파인튜닝 설계보다 선행하는 결정**임을 시사한다. 다만 본 연구는 4개 모델만을 다루었으므로 이 대응 관계가 일반 법칙인지는 확인되지 않았다.

여기에 Catastrophic Forgetting 분석(4.2.5)을 겹치면 상충 관계가 드러난다. 도메인 성능을 크게 얻은 Qwen 계열은 VQAv2 기준 범용 능력을 4~7% 내주었고, 범용 능력을 거의 지킨 SmolVLM2-2.2B는 도메인 성능이 유의하게 악화되었으며, Gemma4-E2B는 도메인 이득 없이 범용 능력만 51.5% 잃는 이중 손실을 보였다. **도메인 특화와 범용 능력 보존은 본 실험 범위에서 동시에 달성되지 않았다.** 4.2.5에서 이미 밝힌 대로 이는 관측된 상관이며 인과관계를 검증한 것은 아니다.

#### 4.4.4 Closed–Open 격차: 세 단계에 걸쳐 해소되지 않은 제약

본 연구가 확인한 가장 실질적인 한계는 **개방형(open-ended) 응답 성능**이다. 이 격차는 세 단계 전부에서 일관되게 관측된다.

- **Phase 1(zero-shot)**: PathVQA의 Open Acc는 전 모델 0.027~0.061로, Closed Acc(0.45~0.79)와 한 자릿수 이상 차이났다. 질문 유형별로는 **diagnosis(진단)와 temporal 유형에서 4개 모델 전부 정확도 0.0000**이었다(4.1.3).
- **Phase 3(파인튜닝 + HPO 최적 설정)**: 610 trial 중 최고 성능 설정에서도 val_closed_acc 0.8115 대비 **val_open_acc는 0.1445**에 머물렀다.

4.4.1에서 밝힌 대로 두 단계의 절대값은 직접 비교할 수 없다. 그러나 **Open이 Closed의 20% 미만이라는 비율 관계는 어느 단계에서도 동일**하다(Phase 1의 Qwen3-VL-2B PathVQA 기준 약 9.5%, Phase 3 최고 설정 기준 약 17.8%). 파인튜닝과 하이퍼파라미터 최적화를 모두 동원했을 때 Closed 정확도는 0.63에서 0.81 수준까지 올라간 반면, Open은 여전히 0.15를 넘지 못했다.

이 사실이 중요한 이유는 **임상적 중요도가 높은 질문 유형이 개방형에 집중**되어 있기 때문이다. 4.1.3에서 확인했듯 WCA 가중치가 가장 높은 diagnosis 유형과 표본이 가장 많은 description 유형(전체의 41%)이 모두 자유 서술형이며, 제로샷 상태에서 각각 0.0000과 0.02~0.04였다. 반면 상대적으로 높은 정확도를 보인 yes_no 유형은 정보량이 제한된 이진 판단이다. 즉 **본 연구가 달성한 정확도 향상의 상당 부분은 임상적 가치가 상대적으로 낮은 유형에서 발생**했을 가능성이 있다. 다만 Phase 3은 PathVQA validation 단일 조건이고 유형별 분해를 수행하지 않았으므로, 이 해석의 정밀한 검증은 향후 과제로 남는다(5.3, 5.4).

#### 4.4.5 자동 탐색의 도달점과 자율 에이전트의 한계

Phase 2와 Phase 3은 하이퍼파라미터 탐색을 서로 다른 방식으로 수행했고, **독립적으로 같은 영역에 도달**했다.

- **Phase 2(수동 ablation)**: 나머지 축을 고정한 채 한 축씩 독립 검증 → rank=64, target=all-linear, ratio=1.0 (4.2.2~4.2.4)
- **Phase 3(자유 탐색)**: Random·Optuna·Autoresearch 세 전략의 최고 설정이 **모두 `lora_targets=full`과 rank 32~64 영역**으로 수렴 (4.3.2)

설계가 전혀 다른 두 접근이 같은 영역을 지목했다는 점은 Phase 2 결론의 신뢰도를 높인다. 특히 Phase 2 ablation은 세 축을 각각 독립적으로만 검증했고 세 최적값의 동시 조합 자체는 검증하지 못한다는 한계가 있었는데(4.2.4), Phase 3의 자유 탐색이 그 조합 영역을 실제로 탐색하고도 같은 결론에 도달함으로써 이 한계를 부분적으로 보완한다.

그러나 자동화의 수준에는 뚜렷한 경계가 있었다. **탐색 알고리즘의 자동화(TPE)는 수동 설정을 큰 폭으로 능가**했지만(0.4490 vs 0.3776), **LLM 에이전트의 자율적 판단은 그 알고리즘에 유의하게 미치지 못했다**(4.3.1). 4.3.4가 지목한 원인은 성능 예측 능력의 부재가 아니라 **탐색 행태의 문제**다 — 에이전트가 제안한 설정의 trial-level 평균 품질은 오히려 가장 높았으나(0.3980), 동일 설정의 반복 제안으로 20 trial 중 평균 12.5개의 고유 조합만 시도해 실효 예산이 축소되었다. 요컨대 본 연구의 범위에서 **"탐색의 자동화"는 성숙한 반면 "자율적 연구 판단"은 아직 기존 최적화 알고리즘을 대체할 수준이 아니다.**

#### 4.4.6 선행 연구와의 간접 비교

본 연구는 의료 특화 VLM과의 직접 실험 비교를 수행하지 않았다(2.5). 다만 LLaVA-Med(Li et al., NeurIPS 2023 D&B)는 본 연구와 **동일한 세 데이터셋의 표준 test split**에 대해 수치를 보고하므로, 그 범위에서 간접 비교가 가능하다. 비교에 앞서 **어느 축이 비교 가능하고 어느 축이 불가능한지**를 먼저 규정한다.

- **Closed-ended는 비교 가능하다.** 양측 모두 정답 문자열 일치 기반의 단순 정확도를 보고하며, 평가 표본도 SLAKE 1,061문항·VQA-RAD 451문항으로 동일하다(PathVQA만 6,761 vs 6,719로 42문항 차이).
- **Open-ended는 비교할 수 없다.** LLaVA-Med는 *"생성된 응답에 정답 토큰이 포함된 비율(recall)"* 을 open 지표로 사용하는 반면, 본 연구는 *BERTScore F1 ≥ 0.7 임계값 통과 여부*를 사용한다(3.8.1). 전자가 구조적으로 훨씬 관대한 척도이므로 두 수치를 나란히 두는 것 자체가 오도의 소지가 있다. 아래 표에는 참고를 위해 병기하되 **비교 대상이 아님을 명시**한다.

**Table 4.4. 선행 연구 보고 수치와의 간접 비교 (Closed-ended 기준)**

| 모델 | 규모 | 학습 방식 | PathVQA | SLAKE | VQA-RAD |
|------|:----:|-----------|:-------:|:-----:|:-------:|
| LLaVA (범용) | 7B | 도메인 학습 없음 | 63.20 | 63.22 | 65.07 |
| LLaVA-Med | 7B | 의생명 코퍼스 사전학습 + 데이터셋별 파인튜닝 | **91.21** | 85.34 | **84.19** |
| **본 연구 (Qwen3-VL-2B)** | **2B** | **QLoRA 파인튜닝만** | 83.12 | **85.26** | 72.91 |

> LLaVA·LLaVA-Med 수치는 원논문 Table 4(a)에서 인용했다. 본 연구 수치는 Phase 2 main 조건(3시드 평균)의 `closed_acc`다. **Open-ended 수치는 척도가 달라 표에서 제외**했다(참고: LLaVA-Med의 open은 PathVQA 37.95 / SLAKE 83.08 / VQA-RAD 61.52이며 토큰 recall 기준, 본 연구는 각각 17.15 / 66.95 / 26.00이며 BERTScore 임계값 기준).

가장 주목할 결과는 **SLAKE에서 2B 모델이 7B 의료 특화 모델과 사실상 동일한 closed 정확도에 도달했다는 점**이다(85.26 vs 85.34). LLaVA-Med는 PubMed Central 규모의 의생명 코퍼스로 사전학습한 뒤 데이터셋별 파인튜닝까지 거친 모델인 반면, 본 연구는 대규모 도메인 사전학습 없이 소비자급 GPU에서 QLoRA 파인튜닝만 수행했다. 세 모델 모두 도메인 학습이 없는 범용 LLaVA(63.22)를 크게 상회한다는 점에서, **SLAKE 수준의 과제에서는 대규모 도메인 사전학습의 이점이 QLoRA 적응만으로 상당 부분 상쇄될 수 있음**을 시사한다.

반면 PathVQA(83.12 vs 91.21)와 VQA-RAD(72.91 vs 84.19)에서는 8~11%p의 격차가 남는다. 두 데이터셋의 성격을 고려하면 이 격차는 자연스럽다 — PathVQA는 병리 조직이라는 고도로 전문화된 영상 도메인이고, VQA-RAD는 문항 수가 451개로 적어 소규모 파인튜닝이 불리하다. 즉 **대규모 도메인 사전학습의 이점은 과제의 전문성이 높을수록 뚜렷하게 남는다.**

이 비교에는 다음 제약이 있으므로 결론을 확대 해석해서는 안 된다. (1) 본 연구 수치는 3시드 평균이나 LLaVA-Med는 단일 실행 보고값이다. (2) 학습 예산이 크게 다르다(본 연구는 `max_steps=500` 상한). (3) 동일 코드·동일 환경에서 재현한 것이 아니라 논문 보고값을 인용한 것이므로, 전처리·프롬프트 등 보고되지 않은 차이가 존재할 수 있다. 동일 프로토콜 하의 직접 재현 비교는 향후 과제로 남는다(5.3(2)).

#### 4.4.7 방법론적 논의: 집계 단위가 결론을 바꾼다

본 연구에서 두 차례, 서로 다른 국면에서 **집계 단위 선택이 결론 자체를 뒤집는** 상황이 발생했다. 이는 결과 해석에 부수적인 사항이 아니라 독립적으로 기록할 가치가 있는 관찰이다.

**(1) 모델을 합칠 것인가 — Phase 2.** 4개 모델을 통합한 Mixed-Effects Model은 파인튜닝 효과가 유의하지 않다고 보고했다(p = .3629). 그러나 이는 효과가 없어서가 아니라 **모델별로 정반대 방향의 효과가 pooled 평균에서 상쇄**되었기 때문이다(4.2.1). 모델별로 분해하면 네 모델 중 셋에서 유의한 효과가, 그중 둘은 음의 방향으로 나타난다. 이질적 효과가 예상되는 상황에서 pooled 추정만 보고하면 "효과 없음"이라는 잘못된 결론에 이른다.

**(2) trial을 독립 관측치로 볼 것인가 — Phase 3.** 순차 최적화 전략은 동일 run 내 trial 간 의존성이 있어 trial-level 검정이 성립하지 않는다(3.7). 실제로 두 단위는 상반된 그림을 보여준다 — trial-level 평균은 Autoresearch(0.3980)가 Optuna(0.3905)보다 높지만, run-level 최고값은 Optuna(0.4490)가 Autoresearch(0.4184)보다 유의하게 높다(4.3.3). 만약 trial-level로 검정했다면 정반대 결론에 도달했을 것이다.

여기에 더해 4.3.4의 관측은 **반복 설계의 필요성**을 직접적으로 뒷받침한다. 동일한 하이퍼파라미터 설정이 반복 실행되었을 때 val_accuracy는 0.388~0.444 범위에서 변동했는데, 이 변동폭(약 0.056)은 **본 연구가 검출한 전략 간 평균 차이(Optuna 0.4490 - Autoresearch 0.4184 = 0.031)보다 크다.** 즉 단일 trial 결과를 근거로 전략 우열을 판단하는 것은 원리적으로 불가능하며, 10회 독립 반복을 검정 단위로 삼은 본 연구의 설계(3.7)는 이 노이즈 규모에 비추어 최소한의 요건이었다고 볼 수 있다.

---

## 제5장. 결론

### 5.1 연구 요약

본 연구는 소비자급 GPU 환경에서 경량 Vision-Language Model(2-3B)을 의료 영상 VQA 도메인에 적응시키는 전 과정을 3단계 실험으로 검증했다. 4개 모델(Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B)과 3개 공개 데이터셋(PathVQA, SLAKE, VQA-RAD)을 대상으로 제로샷 베이스라인(Phase 1), QLoRA 파인튜닝 75조건(Phase 2), 하이퍼파라미터 탐색 전략 비교 610 trial(Phase 3)을 수행했다. 연구 질문별 결과는 다음과 같다.

**RQ1 — 경량 VLM의 제로샷 성능은 모델별로 유의미한 차이가 있는가?**
귀무가설은 기각되었다. 4개 모델의 정오답 패턴은 pooled 기준(n=8,231) 및 세 데이터셋 개별 검정 모두에서 통계적으로 유의하게 달랐다(Cochran's Q = 1904.28, df = 3, p < .001). Qwen3-VL-2B가 pooled 정확도 0.3843으로 최고 성능을 보였으나 Qwen2.5-VL-3B와는 데이터셋에 따라 통계적으로 구분되지 않을 만큼 근접했고, Gemma4-E2B는 나머지 세 모델 전부와 유의하게 낮았다. 이 순위는 Min-K% Probability로 식별한 사전훈련 노출 의심 샘플을 제거한 뒤에도 유지되어 데이터 오염에 강건했다(4.1.1).

**RQ2 — QLoRA 파인튜닝이 성능을 유의미하게 향상시키는가?**
귀무가설의 기각 여부는 **모델에 따라 달랐다.** Qwen2.5-VL-3B(d = +2.646)와 Qwen3-VL-2B(d = +1.620)에서는 유의하게 향상되었으나, SmolVLM2-2.2B(d = -2.284)는 유의하게 악화되었고 Gemma4-E2B(d = -0.652)는 유의하지 않았다. 4개 모델을 통합한 Mixed-Effects Model은 효과가 유의하지 않다고 보고했으나(p = .3629), 이는 효과의 부재가 아니라 **상반된 방향의 모델별 효과가 pooled 평균에서 상쇄된 결과**다(4.2.1). 따라서 "QLoRA 파인튜닝은 의료 VQA 성능을 향상시킨다"는 명제는 모델과 무관하게 성립하지 않는다.

**RQ3 — LLM 에이전트의 자율 탐색이 베이지안 최적화와 경쟁적 성능을 달성하며 해석 가능한 탐색 근거를 제공하는가?**
귀무가설(Autoresearch = Optuna)은 기각되었으나, **그 방향은 가설이 기대한 것과 반대였다.** run-level 비교에서 Optuna(0.4490)가 Autoresearch(0.4184)보다 유의하게 우수했고(Mann-Whitney U = 16.00, p = .0112, r = -0.68), Autoresearch는 하한선 비교 대상인 Random Search(0.4186)와도 통계적으로 구분되지 않았다. 다만 세 자동 탐색 전략이 모두 수동 설정(0.3776)을 상회하여 자동 탐색 자체의 유효성은 확인되었다.

RQ3의 두 번째 요건인 **해석 가능한 탐색 근거**는 **본 실험 설계에서 충분히 검증되지 못했다.** 에이전트에게 주어진 시스템 프롬프트(부록 B)가 "설명이나 다른 텍스트 없이 JSON 객체만 응답하라"고 명시적으로 지시했기 때문에, 자연어 근거의 산출은 애초에 요구되지 않았다. 실제로 200개 trial 중 **147개(73.5%)는 하이퍼파라미터 JSON만을 반환했고, 자연어 서술이 포함된 것은 53개(26.5%)에 그쳤다.** 따라서 "LLM 에이전트가 해석 가능한 탐색 근거를 제공하는가"라는 질문에 대해 본 연구는 긍정도 부정도 할 수 없으며, 이는 결과가 아니라 **설계상의 결함**이다(5.3(8)).

한편 탐색 품질 자체는 저조했다. 에이전트는 20 trial 중 평균 12.5개의 고유 설정만 시도했고(Random·Optuna는 20/20), 동일 설정을 반복 제안하며 조기 고착하는 패턴이 10회 반복 전체에서 관측되었다(4.3.4).

### 5.2 연구 기여

본 연구의 기여는 다음 네 가지다.

**첫째, 경량 VLM의 의료 도메인 적응에 대한 3단계 실증 데이터를 제공한다.** 제로샷 12조건, 파인튜닝 75조건, 하이퍼파라미터 탐색 610 trial의 전체 결과를 동일한 평가 프로토콜 하에서 측정하고 공개했다. 특히 성능과 자원 소비가 단조 관계를 이루지 않는다는 점(최고 성능 모델이 동시에 최소 VRAM·최속 응답, 4.4.2)은 자원 제약 환경의 모델 선택에 직접적인 실무 근거가 된다. 나아가 4.4.6의 간접 비교는 **SLAKE closed-ended 기준으로 2B 모델의 QLoRA 적응이 7B 의료 특화 모델과 동등한 수준(85.26 vs 85.34)에 도달할 수 있음**을 보였다 — 다만 PathVQA·VQA-RAD에서는 8~11%p의 격차가 남아, 대규모 도메인 사전학습의 이점이 과제의 전문성에 따라 달라짐을 함께 확인했다.

**둘째, 파인튜닝 효과의 모델별 이질성을 정량화하고 pooled 분석의 함정을 실증했다.** 동일한 파인튜닝 절차가 모델에 따라 큰 폭의 향상과 유의한 악화를 동시에 낳는다는 사실, 그리고 이를 합쳐 추정하면 "효과 없음"으로 오독된다는 사실을 모델별 3중 검증으로 보였다(4.2.1, 4.4.7).

**셋째, LLM 기반 자율 HPO에 대한 부정적 결과와 그 실패 메커니즘을 함께 보고한다.** 자율 에이전트가 기존 기법에 미치지 못한다는 결과 자체보다, 그 원인이 제안 품질의 열세가 아니라 **중복 제안에 의한 실효 탐색 예산 축소**(고유 설정 12.5/20)라는 진단이 후속 설계에 활용될 수 있다. 에이전트의 trial-level 평균 성능은 오히려 네 전략 중 가장 높았다(4.3.3, 4.3.4).

**넷째, 집계 단위 선택이 결론을 뒤집는 두 사례와 그 판단 근거를 기록했다.** 모델을 합칠 것인가(Phase 2), trial을 독립 관측치로 볼 것인가(Phase 3)의 두 국면에서 집계 단위에 따라 상반된 결론이 도출됨을 보였다. 특히 동일 설정의 반복 실행에서 관측된 변동폭(약 0.056)이 본 연구가 검출한 전략 간 평균 차이(0.031)보다 크다는 사실은, 단일 실행 결과로 기법을 비교하는 관행에 대한 정량적 반례가 된다(4.4.7).

### 5.3 한계점

**(1) 데이터 오염 통제의 한계.** 대상 데이터셋은 모델의 사전훈련 시점 이전에 공개되었으므로 사전훈련 데이터 오염 가능성이 존재한다. 본 연구는 Min-K% Probability(Shi et al., ICLR 2024)로 노출 의심 샘플을 식별하고 이를 제거한 축소셋에서 결론의 강건성을 확인했으나(4.1.1), Min-K%는 어디까지나 간접 지표이며 오염의 완전한 통제나 정확한 정량화는 불가능하다.

**(2) 의료 특화 VLM과의 직접 비교 부재.** LLaVA-Med, Med-Flamingo, CheXagent 등 의료 특화 VLM을 본 연구의 환경에서 직접 재현해 비교하지는 않았다(2.5). 4.4.6의 Table 4.4는 LLaVA-Med가 **동일한 세 데이터셋의 표준 test split**에 대해 보고한 수치를 인용한 간접 비교이며, 다음 세 가지 제약을 안는다. 첫째, **채점 기준이 일치하는 closed-ended 지표로만 비교가 성립**한다 — open-ended는 LLaVA-Med가 정답 토큰 포함 비율(recall)을, 본 연구는 BERTScore F1 임계값을 쓰므로 척도가 근본적으로 다르다. 둘째, 본 연구 수치는 3시드 평균이나 인용한 수치는 단일 실행 보고값이며 학습 예산도 크게 다르다. 셋째, 동일 코드·환경에서 재현한 것이 아니므로 전처리·프롬프트 등 보고되지 않은 차이가 남아 있다. Med-Flamingo·CheXagent는 이 세 데이터셋에 대해 동일 프로토콜의 수치를 제공하지 않아 표에서 제외했다. 동일 프로토콜 하의 직접 재현 비교는 향후 과제로 남는다.

**(3) Phase 3의 실효 학습량 교란.** Phase 3은 `max_steps=200`을 전 trial에 고정하여 학습량을 통제하고자 했으나, **step 수의 고정이 실제 학습 샘플 수의 고정으로 이어지지는 않았다.** 탐색 공간에 `batch_size`(1/2/4)와 `grad_accum_steps`(4/8/16)가 포함되어 있어, 실효 학습 샘플 수(= batch × grad_accum × max_steps)는 최소 800에서 최대 12,800까지 **약 16배 차이**가 났다. 따라서 4.3의 전략 간 비교에는 하이퍼파라미터 품질 외에 실효 학습량의 차이가 일부 섞여 있다. 특히 고정 설정을 사용한 Manual(batch 1 × grad_accum 8 → 1,600 샘플)의 열세에는 학습량 열세가 포함되어 있을 수 있다. 다만 Optuna와 Autoresearch의 최고 설정은 모두 batch 4 × grad_accum 16(12,800 샘플)으로 동일하여, RQ3의 핵심 비교인 두 전략 간 차이가 이 교란으로 설명되지는 않는다. `results.tsv`는 batch_size·grad_accum_steps·max_steps를 전 trial 기록하므로 실효 학습량은 사후 산출이 가능하나, 해당 값 자체를 컬럼으로 보고하지는 않았다.

**(4) LLM 에이전트의 비결정성.** Autoresearch는 외부 LLM API에 의존하므로 완전한 재현이 보장되지 않는다. 본 연구는 temperature = 0으로 고정하고(전 trial 기록 확인) 10회 독립 반복으로 변동성을 흡수했으나, API 측 모델 갱신에 따른 결과 변화 가능성은 통제 범위 밖이다.

**(5) Ablation 결과의 일반화 제약.** Phase 2의 Ablation A·B·C는 모두 PathVQA와 Qwen3-VL-2B 단일 조건에서 수행되었다(4.2.2~4.2.4). 설계 단계에서 계획했던 SLAKE 기반 rank 보조 검증은 GPU 시간 제약으로 **수행하지 못했다.** 따라서 rank=64·target=all-linear·ratio=1.0이라는 결론이 다른 데이터셋·모델로 확장되는지는 검증되지 않았다. 또한 세 축은 각각 나머지를 고정한 채 독립적으로만 검증했으며 세 최적값의 동시 조합 자체는 별도 검증하지 않았다(4.2.4). 다만 Phase 3의 자유 탐색이 독립적으로 같은 영역에 수렴한 것은 이 한계를 부분적으로 보완한다(4.4.5).

**(6) 통계적 검정력의 한계.** Phase 2의 파인튜닝 효과 검정은 n = 9(3 데이터셋 × 3 시드), Phase 3의 run-level 검정은 전략당 n = 10이다. BCa Bootstrap·Mixed-Effects·Wilcoxon 3중 검증과 run-level 반복 설계로 강건성을 확보했으나, 표본 수 자체의 한계로 효과 크기의 신뢰구간은 넓다(예: 4.2.1의 Cohen's d 95% CI가 [0.932, 3.153]에 이르는 사례).

**(7) 임상적 의미 평가의 간접성.** Weighted Clinical Accuracy(WCA)의 질문 유형별 가중치는 외부 임상 문헌이나 전문가 합의 없이 연구자가 부여한 임시 척도이며(3.8.3), 절대적 임상 중요도의 척도로 해석할 수 없다. Expected Calibration Error(ECE)는 현재 평가 파이프라인이 per-sample confidence를 저장하지 않아 산출하지 못했다. 또한 Phase 3은 질문 유형별 분해를 수행하지 않았으므로, 4.4.4에서 제기한 "정확도 향상이 임상적 가치가 낮은 유형에 편중되었을 가능성"은 검증되지 않은 해석에 머문다.

**(8) 자율 에이전트 설정의 내적 불일치.** 본 연구가 사후에 확인한 가장 중대한 한계로, Autoresearch 조건은 다음 세 가지 설정 불일치를 포함한다. 이는 4.3의 부정적 결과를 "LLM 에이전트의 본질적 한계"로 일반화할 수 없게 만드는 요인이므로 명시한다.

- **탐색 근거 산출을 프롬프트가 금지했다.** 시스템 프롬프트(부록 B)는 "설명·마크다운·기타 텍스트 없이 JSON 객체만 응답하라"고 지시한다. 그 결과 200 trial 중 73.5%가 JSON만 반환했다. RQ3가 요구한 "해석 가능한 탐색 근거"는 측정 대상이 되기 전에 설계에 의해 배제되었다.
- **탐색 일정이 실제 예산과 어긋난다.** 프롬프트는 탐색 단계를 절대 trial 번호로 규정한다 — 초기 탐색 0~5, 중기 착취 5~20("최고 설정을 가져와 1~2개 파라미터만 변경"), 후기 정밀화 20+. 그러나 실제 예산은 반복당 20 trial이므로 **후기 정밀화 단계는 한 번도 발동하지 않았고, 예산의 약 75%가 "최고 설정 주변만 변형"하도록 지시된 구간에서 소비**되었다. 4.3.4가 관측한 중복 제안(고유 설정 12.5/20)은 에이전트의 판단 실패라기보다 이 지시를 충실히 따른 결과일 가능성이 있다. 코드 측 단계 전환 로직(`src/autoresearch/agent.py`)은 진행률 비율(0.25/0.75) 기준이어서 예산에 맞게 조정되나, 프롬프트 텍스트는 그렇지 않아 양자가 어긋난다.
- **무효 파라미터를 탐색 대상으로 제시했다.** 프롬프트는 `epochs`를 탐색 공간에 포함하고 "데이터가 제한적일 때 더 많은 epoch(3-5)이 도움이 된다"는 지침까지 제공하지만, 구현(`src/autoresearch/agent.py`)은 제안된 `epochs`를 폐기하고 `max_steps=200`으로 고정한다. 실제 로그에서 에이전트가 "모든 trial이 200 step뿐이라 학습 부족으로 보인다", "epochs가 변경되지 않았다"고 진단하는 사례가 관측되는데(부록 D), 이는 정확한 진단이었으나 해당 조정 수단은 애초에 작동하지 않았다.

이 세 항목은 모두 Autoresearch 조건에만 적용되며 Random·Optuna 조건에는 해당하지 않는다. 따라서 4.3.1의 비교는 "동일 탐색 공간에서의 알고리즘 비교"라기보다 **"이 프롬프트 구성으로 운용된 에이전트와 기존 알고리즘의 비교"**로 한정해 해석해야 한다.

**(9) 다중 비교 보정의 부재.** 본 연구는 Phase 1(Cochran's Q + McNemar), Phase 2(paired t-test·Wilcoxon·Bootstrap·Mixed-Effects 병행), Phase 3(Kruskal-Wallis + Mann-Whitney)에 걸쳐 총 20회 이상의 통계 검정을 수행했다. **Bonferroni 보정은 Phase 1의 McNemar 사후검정에만 적용했으며, 전체 파이프라인을 아우르는 통합 다중비교 보정은 적용하지 않았다.** 유의수준 0.05를 각 검정에 독립 적용하면 family-wise error rate가 누적되어 우연에 의한 유의 결과(제1종 오류)의 위험이 커지므로, 개별 p값은 이 점을 감안해 해석해야 한다. 다만 본 연구의 주요 결론은 단일 p값이 아니라 3중 검증(Phase 2) 또는 신뢰구간 비겹침(Phase 3, 4.3.1)으로 뒷받침되므로 보정 부재에 따른 결론 반전 가능성은 제한적이다. 파이프라인 수준의 FDR 보정은 향후 분석 과제로 남긴다.

**(10) Cross-dataset 결과의 성격.** 4.2.5(B)의 cross-dataset 성능 변화는 엄밀한 의미의 Catastrophic Forgetting(파인튜닝 이전에 가능했던 것을 이후 수행하지 못하게 되는 현상)이 아니다. PathVQA(병리 조직)와 SLAKE·VQA-RAD(방사선)는 이미지 도메인 자체가 상이하므로, 이 지표는 **도메인 특화에 따라 예측 가능한 도메인 일반화 격차(domain generalization gap)**에 가깝다. 본 논문은 이를 4.2.5에서 명시했으며, CF의 엄밀한 판정은 (A) VQAv2 지표에 한정해 해석한다.

**(11) Gemma4-E2B의 아키텍처 이질성.** 평가 대상 4개 모델 중 **Gemma4-E2B만 Mixture-of-Experts(MoE) 구조**로, 추론 시 2.3B만 활성화되나 저장 파라미터는 5.1B에 달한다(3.3). 나머지 세 모델은 활성 파라미터와 전체 파라미터가 같은 밀집(dense) 구조다. 본 연구의 "경량 VLM" 선정 기준은 소비자 GPU 구동 가능성이라는 연구 목적에 따라 **활성 파라미터**를 기준으로 삼았으므로 선정 자체는 일관되나, 4.4.2의 "규모와 성능의 비단조 관계" 논의를 **저장 파라미터 기준의 규모 비교로 확대 해석해서는 안 된다** — 저장 파라미터 기준으로는 Gemma4-E2B가 가장 큰 모델이며, 그 최하위 성능은 "작은 모델이 큰 모델을 이겼다"가 아니라 "가장 큰 저장 규모가 최하위였다"로도 읽힌다. 4.4.2의 결론(활성 파라미터·VRAM 기준의 비단조성)은 유효하나 해석 축을 명시해야 한다.

**(12) 학습 예산 상한(`max_steps` cap)의 구조적 제약.** Phase 2는 GPU 시간 제약으로 `max_steps=500` 상한을 적용했고(3.6), 이로 인해 **데이터셋 크기와 무관하게 학습량이 고정**된다. 결과적으로 소형 VQA-RAD는 약 2 epoch 이상 학습되는 반면 대형 PathVQA는 1 epoch에 미치지 못한다. 4.2와 4.4.6에서 관측된 데이터셋별 성능 격차에는 이 실효 학습량 차이가 일부 섞여 있을 수 있으며, 특히 4.4.6에서 PathVQA·VQA-RAD의 격차를 "과제 전문성"으로 해석한 부분은 학습 예산 차이라는 대안 설명을 배제하지 못한다. 데이터셋별 full-epoch 재학습은 후속 과제로 남긴다.

**(13) Phase 3 탐색 예산 축소의 영향.** 3.7에서 기술한 대로 전략당 탐색 trial 수를 원안 40에서 20으로 축소했다. run-level 검정 단위인 반복 횟수(10회)는 유지했으므로 통계 검정의 타당성에는 영향이 없으나, **각 전략이 탐색할 수 있는 하이퍼파라미터 조합 수가 절반으로 줄어 도달 성능이 원안 대비 과소평가**되었을 수 있다. 이 영향은 순차 최적화 전략(Optuna·Autoresearch)에서 더 클 것으로 예상되며, 실제로 4.3.3에서 Autoresearch는 예산 소진 시점까지 개선 중이던 run이 상당수였다.

### 5.4 향후 연구 방향

**첫째, 탐색 예산을 확대한 자율 HPO 재검증이 필요하다.** Autoresearch는 최고 성능 도달 trial의 IQR 상한이 예산 한계값(20)에 걸쳐 있어, 절반 가까운 run이 예산 소진 시점까지 개선 중이었다(4.3.3). 40~100 trial 규모에서 Optuna와의 격차가 유지되는지, 아니면 역전되는지는 본 연구가 답하지 못한 질문이다.

**둘째, 설정 불일치를 제거한 상태에서 자율 HPO를 재평가해야 한다.** 5.3(8)이 지적한 세 가지 불일치 — 탐색 근거 산출 금지, 절대 trial 번호 기준의 탐색 일정, 무효 파라미터(`epochs`) 노출 — 는 모두 프롬프트와 구현의 정합성 문제이므로 수정 가능하다. 이를 바로잡은 뒤에야 "LLM 에이전트가 베이지안 최적화와 경쟁 가능한가"라는 질문에 대한 공정한 답을 얻을 수 있다. 아울러 이미 시도한 설정을 회피하는 명시적 제약과, 반복 실행 간 확률적 변동을 성능 개선과 구분하는 판단 기준(4.4.7의 노이즈 규모 참조)을 도입하는 방안도 함께 검토할 필요가 있다. 본 연구는 이러한 개선안을 제안할 뿐 검증하지는 않았다.

**셋째, 개방형 응답 성능의 개선이 가장 시급한 과제다.** 파인튜닝과 하이퍼파라미터 최적화를 모두 동원했음에도 open 정확도는 closed의 20% 수준을 넘지 못했으며(4.4.4), 임상적 중요도가 높은 diagnosis·description 유형이 여기에 집중되어 있다. 생성형 응답 자체를 겨냥한 학습 목표나 평가 지표의 재설계가 요구된다.

**넷째, 실효 학습량을 통제한 재실험이 필요하다.** 5.3(3)에서 지적한 16배 차이를 제거하려면 `max_steps` 대신 총 학습 샘플 수를 고정하거나, 실효 학습량을 공변량으로 포함한 분석이 요구된다.

**다섯째, 동일 프로토콜 하의 의료 특화 VLM 비교와 임상 검증된 평가 체계의 확립이 필요하다.** 5.3(2)의 비교 부재와 5.3(7)의 WCA 임시 가중치·ECE 미산출은 모두 후속 연구에서 해소되어야 할 항목이다.

---

## 참고문헌

본 목록은 본문에 실제로 인용된 문헌만 수록한다. 각 항목의 저자·게재처·권호·페이지·DOI는 2026년 8월 원문(arXiv 초록 페이지, 출판사 공식 페이지, PubMed 서지 레코드)을 직접 대조해 확인했다.

**Parameter-Efficient Fine-Tuning**

1. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. "LoRA: Low-Rank Adaptation of Large Language Models." *International Conference on Learning Representations (ICLR)*, 2022. arXiv:2106.09685. (OpenReview: nZeVKeeFYf9)
2. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. "QLoRA: Efficient Finetuning of Quantized LLMs." *Advances in Neural Information Processing Systems (NeurIPS)*, 2023 (Oral). arXiv:2305.14314.

**Vision-Language Model**

3. Liu, H., Li, C., Wu, Q., & Lee, Y. J. "Visual Instruction Tuning." *Advances in Neural Information Processing Systems (NeurIPS)*, 2023 (Oral). arXiv:2304.08485.
4. Bai, S., Chen, K., et al. (Qwen Team, Alibaba Group). "Qwen2.5-VL Technical Report." arXiv:2502.13923, 2025. (총 27인 공저, 기술 보고서 — 학술대회 게재본 없음)
5. Marafioti, A., Zohar, O., Farré, M., Noyan, M., Bakouch, E., et al. "SmolVLM: Redefining small and efficient multimodal models." arXiv:2504.05299, 2025. (총 17인 공저, 프리프린트)

**의료 특화 VLM**

6. Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T., Poon, H., & Gao, J. "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day." *NeurIPS 2023 Datasets and Benchmarks Track*, 2023 (Spotlight). arXiv:2306.00890.
7. Moor, M., Huang, Q., Wu, S., Yasunaga, M., Dalmia, Y., Leskovec, J., Zakka, C., Reis, E. P., & Rajpurkar, P. "Med-Flamingo: a Multimodal Medical Few-shot Learner." *Proceedings of the 3rd Machine Learning for Health Symposium (ML4H)*, PMLR 225:353-367, 2023. arXiv:2307.15189.
8. Chen, Z., Varma, M., Xu, J., Paschali, M., Van Veen, D., et al. "A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation." arXiv:2401.12208, 2024. (총 23인 공저. 초판 제목은 "CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation"이었으나 2024년 12월 개정판에서 현재 제목으로 변경됨. 본문 2.5의 CheXagent가 이 문헌이다.)

**의료 VQA 데이터셋**

9. He, X., Zhang, Y., Mou, L., Xing, E., & Xie, P. "PathVQA: 30000+ Questions for Medical Visual Question Answering." arXiv:2003.10286, 2020. (프리프린트 — 학술대회 게재본 없음)
10. Liu, B., Zhan, L.-M., Xu, L., Ma, L., Yang, Y., & Wu, X.-M. "SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering." *2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)*, pp. 1650-1654, 2021. DOI: 10.1109/ISBI48211.2021.9434010.
11. Lau, J. J., Gayen, S., Ben Abacha, A., & Demner-Fushman, D. "A dataset of clinically generated visual questions and answers about radiology images." *Scientific Data*, 5, 180251, 2018. DOI: 10.1038/sdata.2018.251.

**하이퍼파라미터 최적화**

12. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. "Optuna: A Next-generation Hyperparameter Optimization Framework." *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD)*, pp. 2623-2631, 2019. DOI: 10.1145/3292500.3330701.
13. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization." *Journal of Machine Learning Research*, 18(185), pp. 1-52, 2018.

**평가 방법론**

14. Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D., & Zettlemoyer, L. "Detecting Pretraining Data from Large Language Models." *The Twelfth International Conference on Learning Representations (ICLR)*, 2024. arXiv:2310.16789. (OpenReview: zWqr3MQuNs. 본문의 Min-K% Probability 기법)
15. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. "On Calibration of Modern Neural Networks." *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pp. 1321-1330, 2017.

> **표기 원칙**: 저자가 20인을 초과하는 문헌(4·5·8번)은 제1저자부터 5인까지 기재하고 총 인원을 병기했다. 3·7·10·11·12·13·15번은 원문에 표시된 전체 공저자를 모두 기재했다. 1·2·6·14번은 arXiv 및 출판사 페이지에서 확인한 공저자 목록을 기재했다.

---

## 부록

### 부록 A. 실험 결과 파일 경로

본 연구의 모든 수치는 아래 파일에서 산출되었다. 논문 본문의 표·검정 결과는 모두 이 파일들로 역추적할 수 있다.

**Phase 1 — 제로샷 베이스라인**

| 파일 | 내용 | 본문 위치 |
|------|------|-----------|
| `results/phase1_baseline/` | 12조건(4모델×3데이터셋, seed 42) 원본 결과 | Table 4.1 |
| `results/phase1_baseline/phase1_robustness.md` | Min-K% 오염 강건성 재검정 | 4.1.1 |

**Phase 2 — QLoRA 파인튜닝**

| 파일 | 내용 | 본문 위치 |
|------|------|-----------|
| `results/phase2_finetune/phase2_summary.csv` | 75조건(Main 36 + Ablation 39) 요약 | Table 4.2, 4.2a, 4.2c, 4.2d |
| `results/phase2_finetune/phase2_rq2_analysis.md` | RQ2 3중 검증 + Mixed-Effects Model | 4.2.1 |
| `results/phase2_finetune/cross_dataset_cf_summary.md` | cross-dataset CF 72조건 | Table 4.2b-B |

**Phase 3 — 자율 하이퍼파라미터 최적화**

| 파일 | 내용 | 본문 위치 |
|------|------|-----------|
| `results/phase3_autoresearch/results.tsv` | 610 trial 전수 기록(하이퍼파라미터·성능·근거) | 4.3 전반 |
| `results/phase3_autoresearch/phase3_rq3_analysis.md` / `.json` | run-level Kruskal-Wallis·Mann-Whitney·Bootstrap CI | Table 4.3, 4.3.1 |
| `results/phase3_autoresearch/phase3_summary.txt` | 전략별 최고 trial 및 분포 통계 | Table 4.3a, 4.3c |
| `results/phase3_autoresearch/phase3_anytime_curve.csv` | anytime 곡선 원본 수치(trial별 중앙값·IQR·n) | 4.3.3 |
| `results/phase3_autoresearch/phase3_anytime_summary.md` | 최고 성능 도달 시점 요약 | Table 4.3b |
| `results/phase3_autoresearch/phase3_anytime.png` / `.pdf` | anytime performance 곡선 그림 | 4.3.3 |

> `results.tsv`의 `agent_reasoning` 컬럼은 줄바꿈이 포함된 인용 문자열을 담고 있어 행 단위 도구(`wc -l` 등)로는 정확히 집계되지 않는다. 집계·분석 시에는 `src/autoresearch/tracker.py`의 `ExperimentTracker`(csv 모듈 기반)를 사용해야 한다.

### 부록 B. Autoresearch 에이전트 시스템 프롬프트

아래는 `configs/autoresearch/program.md`의 전문이다. 5.3(8)에서 지적한 세 가지 설정 불일치의 근거이므로 원문 그대로 수록한다. **밑줄 친 부분이 문제가 된 지점이다.**

```markdown
# Autonomous HPO Agent - System Prompt

You are an autonomous hyperparameter optimization agent for medical VQA
fine-tuning research.

## Task
Given the history of previous QLoRA fine-tuning experiments, suggest the NEXT
hyperparameter configuration that is most likely to improve validation accuracy
on the PathVQA medical VQA dataset.

## Search Space
| Parameter        | Range                       | Type                     |
|------------------|-----------------------------|--------------------------|
| lora_rank        | {4, 8, 16, 32, 64}          | discrete                 |
| lora_alpha       | rank × {1, 2, 4}            | discrete                 |
| learning_rate    | [1e-5, 5e-4]                | continuous (log-scale)   |
| batch_size       | {1, 2, 4}                   | discrete                 |
| grad_accum_steps | {4, 8, 16}                  | discrete                 |
| warmup_ratio     | [0.0, 0.1]                  | continuous               |
| weight_decay     | [0.0, 0.1]                  | continuous               |
| lora_targets     | {"minimal","medium","full"} | categorical              |
| epochs           | {1, 2, 3, 5}                | discrete   ← (i) 무효    |

Where: minimal = [q_proj, v_proj] / medium = [q_proj, k_proj, v_proj, o_proj]
       full = all linear layers

## Strategy Guidelines
1. Early exploration (trials 0-5): Try diverse configurations to map the
   landscape. Vary multiple parameters at once.
2. Mid exploitation (trials 5-20): Focus on promising regions. Take the best
   configuration and vary 1-2 parameters at a time.        ← (ii) 예산 불일치
3. Late refinement (trials 20+): Fine-tune around the best configuration with
   small perturbations.                                    ← (ii) 미발동 구간

## Key Insights for Medical VQA
- Medical images benefit from higher LoRA ranks (16-64).
- Learning rate is often the most sensitive parameter.
- `medium` or `full` target modules often outperform `minimal`.
- Effective batch size = batch_size × grad_accum_steps. Keep in 4-16 range.
- More epochs (3-5) help when training data is limited.    ← (i) 무효 파라미터
- Warmup ratio 0.03-0.06 is generally safe.

## Response Format
Respond with ONLY a valid JSON object. No explanation, no markdown fences,
no other text.                                             ← (iii) 근거 산출 금지
```

**(i) 무효 파라미터**: `epochs`는 탐색 공간과 지침에 모두 등장하나, 구현(`src/autoresearch/agent.py`)이 제안값을 폐기하고 `max_steps=200`으로 고정한다.
**(ii) 예산 불일치**: 실제 예산은 반복당 20 trial이므로 "Late refinement (trials 20+)"는 발동하지 않으며, 예산의 약 75%가 "최고 설정을 1~2개 파라미터만 변경" 구간에 해당한다.
**(iii) 근거 산출 금지**: RQ3가 요구한 "해석 가능한 탐색 근거"를 프롬프트가 명시적으로 배제한다.

**에이전트 실행 설정**: 모델 `claude-sonnet-4-6`, `max_tokens=512`, `temperature=0`(전 trial 기록으로 확인). 온도 고정과 10회 독립 반복으로 API 비결정성을 완화했다(5.3(4)).

### 부록 C. 재현 가이드

아래 명령은 본 연구가 실제로 실행한 것이며, 저장소의 스크립트 인자와 일치한다.

**환경 준비**

```bash
uv sync --extra unsloth        # unsloth 백엔드 필수 (Qwen 계열 데이터 포맷)
uv run python -c "import unsloth"   # 설치 확인
export HF_HOME=/hf_cache                          # 캐시 분산 (디스크 quota 대비)
export MOAI_CHAT_CACHE_DIR=/workspace/hf_cache/chat_cache
export WANDB_API_KEY=...       # 학습 로깅
export ANTHROPIC_API_KEY=...   # Phase 3 autoresearch 전략에만 필요
```

> `python3`가 시스템 파이썬을 가리켜 `ModuleNotFoundError`가 발생하는 사례가 반복 관측되었으므로, 모든 실행은 `uv run python` 형태로 통일한다.

**Phase 1 — 제로샷 베이스라인**

```bash
bash scripts/runpod_phase1.sh    # 12조건(4모델 × 3데이터셋), seed 42
```

**Phase 2 — QLoRA 파인튜닝**

```bash
uv run python -u -m src.finetune.run_phase2 \
  --config_dir configs/models \
  --finetune_config configs/finetune/base_qlora.yaml \
  --output_dir results/phase2_finetune \
  --seeds 42 123 456 \
  --data_dir data \
  --max_eval_samples 500

bash scripts/run_phase2_ablation.sh   # Ablation A/B/C (PathVQA, Qwen3-VL-2B 고정)
```

**Phase 3 — 자율 하이퍼파라미터 최적화**

```bash
uv run python -u -m src.autoresearch.run_phase3 \
  --model_config configs/models/qwen3_vl_2b.yaml \
  --finetune_config configs/finetune/base_qlora.yaml \
  --output_dir results/phase3_autoresearch \
  --strategies manual random optuna autoresearch \
  --repeats 10 \
  --trials_per_repeat 20 \
  --seed 42 \
  --data_dir data \
  --time_budget_min 90 \
  --max_test_samples 500 \
  --max_parallel 2
```

> `--time_budget_min 90`은 실험 통제 변수가 아니라 이상 조합으로 인한 무한 학습을 막는 안전장치다. 학습량 통제는 `max_steps=200`(코드 상수)이 담당한다. `--max_parallel 2`는 GPU 2장에 (전략, 반복) 단위 작업을 배분한다.

**분석 및 그림 생성**

```bash
uv run python scripts/analyze_phase3.py --results_dir results/phase3_autoresearch
uv run python scripts/summarize_stage.py manual random optuna autoresearch
uv run python scripts/plot_phase3_anytime.py --results_dir results/phase3_autoresearch
```

### 부록 D. Autoresearch 제안 근거 로그 발췌

`results.tsv`의 `agent_reasoning` 컬럼 원문에서 발췌했다. 200개 완료 trial 중 **147개(73.5%)는 아래 (1)과 같이 JSON만 반환**했고, 자연어 서술이 포함된 것은 53개(26.5%)였다(5.1, 5.3(8)).

**(1) 전형적 응답 — 근거 없이 설정만 반환 (전체의 73.5%)**

```json
{"lora_rank": 64, "lora_alpha": 256, "learning_rate": 2.0e-4, "batch_size": 2,
 "grad_accum_steps": 8, "warmup_ratio": 0.05, "weight_decay": 0.01,
 "lora_targets": "full", "epochs": 3}
```

프롬프트가 설명을 금지했으므로(부록 B (iii)) 이 형태가 지시에 부합하는 응답이다.

**(2) 자연어 서술이 포함된 사례 — 정확한 진단, 그러나 작동하지 않는 수단**

```
Looking at the results:
1. `full` targets consistently outperform `medium`
2. LR ~3e-4 seems best (trials 435, 449, 451 all at 0.402)
3. Rank 32 with alpha 64 at 3e-4 appears to be a local optimum
4. All trials use bs=2, ga=8, and only 200 steps - likely under-trained
5. Epochs haven't been varied
```

주목할 점은 4·5번 항목이다. 에이전트는 **학습량 부족을 정확히 진단**하고 `epochs`가 변경되지 않았음을 지적했으나, 구현은 제안된 `epochs`를 폐기하고 `max_steps=200`으로 고정하므로(부록 B (i)) 이 진단을 실행에 옮길 수단이 없었다. 1~3번 항목은 4.3.2에서 확인된 전략 공통의 수렴 영역(`full` targets, rank 32~64)과 일치하여, 에이전트의 결과 해석 자체는 타당했음을 보여준다.

**(3) 조기 고착 사례 — 반복 8**

반복 8에서 에이전트는 trial 602 이후 아래 설정을 **11회 연속 제안**했다(4.3.4).

```json
{"lora_rank": 64, "lora_alpha": 256, "learning_rate": 2.0e-4, "batch_size": 2,
 "grad_accum_steps": 8, "warmup_ratio": 0.05, "weight_decay": 0.01,
 "lora_targets": "full", "epochs": 3}
```

동일 설정임에도 측정된 val_accuracy는 0.388~0.444 범위에서 변동했다. 이 변동은 설정 차이가 아니라 학습·평가의 확률적 변동이며, 그 폭(약 0.056)은 본 연구가 검출한 전략 간 평균 차이(0.031)보다 크다(4.4.7). 프롬프트가 trial 5~20 구간에서 "최고 설정을 가져와 1~2개 파라미터만 변경"하도록 지시한 점(부록 B (ii))을 함께 고려하면, 이 고착은 에이전트의 판단 실패보다 지시에 대한 충실한 이행에 가깝다.

> 전체 로그는 `results/phase3_autoresearch/results.tsv`의 `agent_reasoning` 컬럼 및 각 trial 디렉터리의 `rationale.md`에 있다.

---

*(참고문헌 15건은 2026년 8월 원문 대조 완료. 남은 작업: 인용 표기 형식을 학과 지정 양식으로 통일)*
