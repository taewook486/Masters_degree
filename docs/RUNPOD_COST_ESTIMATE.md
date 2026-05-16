# RunPod 실행 비용 및 시간 추정

> **작성일**: 2026-05-16
> **기준 GPU**: RTX 4090 (24GB VRAM)
> **기준 가격**: RunPod Community Cloud $0.34/hr, Secure Cloud $0.69/hr (2026년 시점)

---

## 1. 단계별 요약

| 단계 | 작업 | 조건 수 | 예상 시간 | Community ($0.34/h) | Secure ($0.69/h) |
|------|------|:---:|:---:|:---:|:---:|
| **Phase 1** | 제로샷 베이스라인 + BERTScore | 36 (4×3×3) | ~10-12h | **$3.4-4.1** | $6.9-8.3 |
| **Phase 1.5** | 데이터 오염 측정 (Min-K%) | 12 (4×3) | ~2-3h | **$0.7-1.0** | $1.4-2.1 |
| **Phase 2 main** | QLoRA 파인튜닝 | 36 (4×3×3) | ~25-30h | $8.5-10.2 | $17.3-20.7 |
| **Phase 2 ablation** | Data/Rank/Target 3종 | ~15 trials | ~10h | $3.4 | $6.9 |
| **Phase 3** | HPO 4전략 × 10반복 × 40 trial | 1,210 trials | ~200h | $68 | $138 |

---

## 2. 권장 실행 순서

### 1단계: 즉시 실행 (Phase 1 + 1.5)

| 항목 | 값 |
|------|-----|
| 총 시간 | ~12-15시간 |
| Community Cloud | **$4-5** |
| Secure Cloud | $8-10 |
| 산출물 | phase1_summary.csv, contamination 분석 |
| 의사 결정 시점 | best baseline 모델 확정, contamination 영향 평가 |

### 2단계: 지도교수 승인 후 (Phase 2 main)

| 항목 | 값 |
|------|-----|
| 총 시간 | ~25-30시간 |
| Community Cloud | **$8-10** |
| Secure Cloud | $17-21 |
| 산출물 | phase2_summary.csv, CF 측정 결과 (VQAv2 + cross-dataset), 최적 fine-tuned 모델 |
| 의사 결정 시점 | Phase 3 진행 여부 |

### 3단계: 시간 여유 있을 때 (Phase 2 ablation + Phase 3)

| 항목 | 값 |
|------|-----|
| 총 시간 | ~210시간 (8-9일 연속) |
| Community Cloud | **$70-80** |
| Secure Cloud | $140-160 |
| 산출물 | ablation 결과, HPO 4전략 비교, autoresearch rationale 로그 |

---

## 3. 세부 시간 분해

### 3.1 Phase 1 (모델당 ~2.5-3시간)

| 데이터셋 | 테스트 샘플 | seed당 시간 | 3 seeds 합계 |
|---------|:---:|:---:|:---:|
| PathVQA | ~6,719 | ~35-40min | ~2h |
| SLAKE | ~1,044 | ~5-7min | ~20min |
| VQA-RAD | ~451 | ~2-3min | ~8min |
| BERTScore 후처리 | (전체) | +5-10min | +30min |

**모델당 합계**: ~3h × 4 모델 = **~12시간**

### 3.2 Phase 1.5 Min-K% (모델당 ~30분)

- Forward pass만 필요 (generate 없음) → Phase 1의 ~25% 시간
- 4 모델 × ~30분 = **~2시간**

### 3.3 Phase 2 main (조건당 ~40-50분)

- QLoRA 학습: 3 epochs × max_steps 가변
- LoRA r=16, batch=1, grad_accum=8 기준
- 12조건(4모델×3데이터셋) × 3 seeds = 36
- 36 × ~40분 = **~24시간**

### 3.4 Phase 2 ablation (~10시간)

- Ablation A (data %): 5 trials × ~30min = 2.5h
- Ablation B (LoRA rank): 5 trials × ~40min = 3.3h (rank 4~64)
- Ablation C (target module): 3 trials × ~50min = 2.5h
- 합계 + Overhead: ~10h

### 3.5 Phase 3 (가장 비쌈)

| 항목 | 값 |
|------|-----|
| 전략 수 | 4 (Manual, Random, Optuna, Autoresearch) |
| 반복 횟수 | 10 (run-level 통계 확보용) |
| 전략당 trial | 40 (Manual은 1) |
| 총 trial | 1 + 400 + 400 + 400 = 1,201 |
| trial당 시간 | ~10분 (max_steps=200) |
| **총 GPU 시간** | **~200시간** |
| 연속 실행 기간 | 8-9일 (24h 가동) |

---

## 4. 비용 절감 옵션

| 전략 | 절감 효과 | 위험 |
|------|---------|------|
| Community Cloud 사용 | Secure 대비 50% 절감 | 가용성 변동 |
| Spot 인스턴스 | 추가 30-50% 절감 | 중단 위험 |
| Phase 분할 (Pod 종료) | idle 시간 차단 | 환경 재구성 필요 |
| `single_seed_first` 플래그 | Phase 1을 1/3 시간으로 단축 | best model 확정 후 추가 실행 필요 |
| GPU 변경 (A40 48GB) | 7B 모델 시 안정성 향상 | 비용 비슷, OOM 위험 ↓ |
| Phase 3 trial 수 축소 | 절반(20 trial)로 줄이면 50% 절감 | 통계 검정력 감소 |

---

## 5. 가정 및 한계

### 5.1 시간 추정 가정

- **Phase 1**: batch_size=8 기준, RTX 4090 throughput ~50-80 samples/s
- **Phase 2**: gradient_accumulation=8, paged_adamw_8bit 기준
- **Phase 3**: max_steps=200, LoRA r=16 기준 (실제 rank 변동 시 시간 변동)
- **BERTScore**: roberta-large 및 BioBERT 모델 로드 시간 포함

### 5.2 가격 가정

- RunPod Community Cloud RTX 4090 평균 $0.34/hr (2026년 시점)
- Secure Cloud $0.69/hr
- 실제 가격은 RunPod 정책 및 지역에 따라 변동
- Spot 인스턴스는 가용성에 따라 $0.20/hr까지 하락 가능

### 5.3 실제 시간이 더 걸릴 수 있는 요인

- HuggingFace 모델/데이터셋 최초 다운로드 (~10-30분)
- BERTScore 모델 다운로드 (roberta-large ~1.5GB, BioBERT ~400MB)
- 환경 세팅 시간 (`runpod_setup.sh` ~5-10분)
- Container 재시작/Pod 재배포 시 모델 재다운로드

### 5.4 추가 비용 요인

- **Storage**: Container Disk 80GB + Volume Disk 50GB 기준 $0.10/hr 추가 가능
- **Bandwidth**: 결과 다운로드 시 outbound 비용 (보통 무료 한도 내)
- **LLM API**: Phase 3 Autoresearch는 Anthropic API 호출 필요 (Claude Sonnet 기준 trial당 ~$0.01-0.05 추정)

Phase 3 LLM API 추가 비용:
- 400 trials × ~$0.02 = ~$8 추가
- 누적 input/output token 고려 (history 누적)

---

## 6. 실제 실행 시 확인 사항

### 6.1 출발 전 체크리스트

- [ ] HuggingFace 토큰 환경 변수 설정 (`HF_TOKEN`)
- [ ] Anthropic API 키 (`ANTHROPIC_API_KEY`) - Phase 3만
- [ ] WandB 토큰 (`WANDB_API_KEY`) - 선택
- [ ] GitHub 푸시 권한 확인 (결과 자동 push 옵션 사용 시)
- [ ] Storage 80GB+ 확보 확인

### 6.2 진행 중 모니터링

```bash
# GPU 사용률 + VRAM
nvidia-smi -l 5

# 진행 로그 (tail)
tail -f results/phase1_baseline/run_all*.log

# 결과 파일 수 확인 (Phase 1 진행률)
ls results/phase1_baseline/*.json | wc -l   # 목표: 36개
```

### 6.3 중단/재개

- Phase 1: `--no_skip_existing` 미사용 시 자동 skip (기본 동작)
- Phase 3: `checkpoint.py`로 마지막 완료 trial부터 자동 재개

---

## 7. 권장 첫 단계 (저비용 검증)

```bash
# 1. RunPod Pod 생성 (RTX 4090, PyTorch 2.4+ 템플릿)
# 2. 클론 및 환경 세팅
git clone https://github.com/taewook486/Masters_degree.git
cd Masters_degree
bash scripts/runpod_setup.sh   # ~10분

# 3. Phase 1 실행 (4개 모델 순차)
bash scripts/runpod_phase1.sh --config configs/models/qwen3_vl_2b.yaml
bash scripts/runpod_phase1.sh --config configs/models/qwen25_vl_3b.yaml
bash scripts/runpod_phase1.sh --config configs/models/smolvlm2_2b.yaml
bash scripts/runpod_phase1.sh --config configs/models/gemma4_e2b.yaml

# 4. Phase 1.5 Min-K% 측정
for model in qwen3_vl_2b qwen25_vl_3b smolvlm2_2b gemma4_e2b; do
  for dataset in pathvqa slake vqa_rad; do
    python scripts/measure_contamination.py \
      --config configs/models/${model}.yaml --dataset ${dataset} \
      --output_dir results/contamination --k_percent 20
  done
done

# 5. 결과 다운로드 후 Pod 종료 (비용 차단)
```

**예상 1단계 총비용: $4-5 (Community Cloud)**

---

## 참고 자료

- [docs/RUNPOD_GUIDE.md](RUNPOD_GUIDE.md): 상세 실행 가이드
- [docs/THESIS_PROPOSAL_FINAL_v0.5.md](THESIS_PROPOSAL_FINAL_v0.5.md): 논문 설계서 (실험 규모 산출 근거)
- [docs/phase1_work_log.md](phase1_work_log.md): 작업 로그 및 학습 사항

---

*최종 업데이트: 2026-05-16*
