# RunPod 실험 실행 가이드

Medical VQA VLM 석사 연구 — Phase 1/2/3 전체 파이프라인 실행 매뉴얼

---

## 목차

1. [RunPod 인스턴스 세팅](#1-runpod-인스턴스-세팅)
2. [환경 세팅 (최초 1회)](#2-환경-세팅-최초-1회)
3. [Phase 1: Zero-shot 베이스라인](#3-phase-1-zero-shot-베이스라인)
4. [Phase 2: QLoRA 파인튜닝](#4-phase-2-qlora-파인튜닝)
5. [Phase 2 Ablation](#5-phase-2-ablation)
6. [Phase 3: Autonomous HPO](#6-phase-3-autonomous-hpo)
7. [결과 다운로드](#7-결과-다운로드)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. RunPod 인스턴스 세팅

### 권장 사양

| 항목 | 권장 |
|------|------|
| GPU | RTX 4090 24GB |
| Template | CUDA 12.8 드라이버 + Python 3.12 |
| Container Disk | 80GB 이상 |
| Volume Disk | 50GB 이상 (`/workspace` 마운트) |

> **[중요] 설치는 반드시 `uv sync`로 — 검증 스택은 transformers 5.5.0 + torch 2.10.0+cu128**: 이 프로젝트의 `uv.lock`은 **transformers 5.5.0 / torch 2.10.0+cu128**로 고정돼 있고, 실제 실험(phase1_work_log)도 이 스택에서 검증됐다. **`pip install -e .` 는 pyproject 하한(`transformers>=4.45.0`)만 보고 4.57.2를 잘못 설치**하는데, 그러면 대상 모델 **Gemma4-E2B(`Gemma4ForConditionalGeneration`, 5.5.0 전용)가 로드 실패**한다. `uv sync`는 `uv.lock`을 그대로 재현하므로 템플릿 torch 버전과 무관하게 검증 스택이 정확히 깔린다. 템플릿은 CUDA 12.8 드라이버 + Python 3.12만 만족하면 된다. (Container 60GB / Volume 40GB 조합으로도 배포 확인됨)

> Gemma4-E2B(~10.3GB) 또는 선택 모델 Qwen2.5-VL-7B 사용 시 RTX 4090(24GB) 권장. 2B/3B 모델만 사용 시 A5000(24GB)도 가능.

> **대안: 16GB GPU 2장 멀티-GPU pod (예: 4080 Super ×2)** — 24GB 단일 GPU를 구하기 어려울 때의 대안 환경으로 실제 검증됨. Phase 2는 `src/finetune/run_phase2.py`가 조건(model×dataset×seed)을 GPU 개수만큼 동시 배정해(`--max_parallel`, 기본 자동 감지) 각 조건을 GPU 1장에 고정 실행하므로, model-parallel 분산 없이도 에러 없이 돌아가고 오히려 GPU 2장 몫의 처리량을 낸다. 상세는 §4.0 참조.

---

## 2. 환경 세팅 (최초 1회)

RunPod 터미널에서 실행:

```bash
cd /workspace
git clone https://github.com/taewook486/Masters_degree.git
cd Masters_degree

bash scripts/runpod_setup.sh
```

`runpod_setup.sh`가 자동으로 처리하는 항목:
- **uv 설치 + `uv sync --extra unsloth`** — `uv.lock` 재현(transformers 5.5.0 + torch 2.10.0+cu128)
- 프로젝트 venv(`.venv`) activate + `~/.bashrc` 등록 (새 터미널·tmux에서도 자동 활성)
- GPU + 버전 확인 (transformers 5.5.0 / torch 2.10.0+cu128 인지 출력)
- 의료 VQA 데이터셋 자동 다운로드 (PathVQA, SLAKE, VQA-RAD)
- VQAv2 subset 다운로드 (CF 측정용)

> **[중요] 새 터미널·tmux 창에서는 venv 활성 확인**: setup이 `~/.bashrc`에 `.venv` activate를 등록하므로 새 창은 자동 활성된다. 혹시 `python`이 5.5.0을 못 잡으면 `source /workspace/Masters_degree/.venv/bin/activate` 실행.

> **데이터 로컬 업로드 불필요** — 모든 데이터셋이 RunPod에서 자동 다운로드됩니다.

### ANTHROPIC_API_KEY 설정 (Phase 3 필수)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# 세션 재시작 대비 영구 설정
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
```

---

## 3. Phase 1: Zero-shot 베이스라인

**목표**: 4개 모델 × 3개 데이터셋 × **1개 시드(42)** 평가 (greedy 결정적 → 부트스트랩 95% CI로 불확실성 보고, BERTScore 포함)

**평가 모델 (4개, 논문 대상)**: Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B

**선택 모델**: Qwen2.5-VL-7B (성능 비교용, 논문 대상 아님)

### 전체 모델 일괄 실행 (권장 — 요약 CSV 자동 생성)

`run_all.py`는 대상 4개 모델(`enabled: false`인 7B·`_template` 자동 제외)을 각 1회씩 로드해 3 데이터셋 × **1 시드(42)**를 평가하고, best model 선택에 필요한 **`phase1_summary.csv`를 생성**한다.

```bash
python -m src.baseline.run_all \
  --output_dir results/phase1_baseline \
  --data_dir data \
  --batch_size 8
```

> **[방법론] zero-shot은 1시드 + 부트스트랩 95% CI**: zero-shot은 greedy 디코딩이라 결정적이므로 시드를 바꿔도 결과가 동일하다(seed-std ≡ 0). 따라서 3시드 반복 대신 **1시드(42)**로 평가하고, 불확실성은 각 조건의 per-sample 정오에 대한 **부트스트랩 95% CI**로 보고한다. 요약 CSV에 `overall_acc_ci_low/high`, `closed_acc_ci_low/high`, `open_acc_ci_low/high` 열이 포함된다. (Phase 2/3은 학습이 확률적이므로 다중 시드/반복 유지 — 변경 없음)

> **[중요] 요약 CSV는 `run_all.py`만 생성한다.** 아래 모델별 개별 실행(`runpod_phase1.sh`)은 개별 JSON만 만들고 `phase1_summary.csv`는 만들지 않는다. 개별 실행 방식을 택했다면, **마지막에 위 `run_all` 을 한 번 더 실행**해 요약을 생성한다(기존 JSON 결과는 skip되고 집계만 수행).

### 모델별 개별 실행 (OOM 디버깅·모델 격리용)

```bash
# Qwen3-VL-2B
bash scripts/runpod_phase1.sh --config configs/models/qwen3_vl_2b.yaml

# Qwen2.5-VL-3B
bash scripts/runpod_phase1.sh --config configs/models/qwen25_vl_3b.yaml

# SmolVLM2-2.2B
bash scripts/runpod_phase1.sh --config configs/models/smolvlm2_2b.yaml

# Gemma4-E2B (PLE, ~10.3GB VRAM)
bash scripts/runpod_phase1.sh --config configs/models/gemma4_e2b.yaml

# (선택) Qwen2.5-VL-7B — 24GB VRAM 필수, 논문 대상 외
bash scripts/runpod_phase1.sh --config configs/models/qwen25_vl_7b.yaml
```

> Gemma 4 전용 스크립트(`runpod_phase1_gemma4.sh`)는 deprecated. 위 범용 스크립트를 사용하세요.

### 기존 결과 덮어쓰기 (BERTScore 재계산)

```bash
bash scripts/runpod_phase1.sh --config configs/models/qwen25_vl_3b.yaml --no_skip_existing
```

### 완료 후 확인

```bash
# 결과 파일 수 확인
ls results/phase1_baseline/*.json | wc -l

# summary 확인
cat results/phase1_baseline/phase1_summary.csv
```

### Phase 1 전체 재실행 (BERTScore 포함 / 결과 덮어쓰기)

기존 결과가 BERTScore 없이 집계됐을 경우 전체 재실행한다. **권장 경로는 `run_all.py`(1시드)** — 요약 CSV까지 한 번에 생성된다.

> **[중요] 설계서 v0.6 정합성**: zero-shot은 greedy 결정적이라 시드를 바꿔도 결과가 동일하므로 **1시드(42)만 사용**한다(설계서 §4.3). 따라서 최종 산출물은 **12개 JSON(4모델 × 3데이터셋 × 1시드)** 이고 **seed-STD = 0.0 이 정상**이다(버그 아님). 3시드로 도는 `runpod_phase1.sh`(모델당 9조건)는 OOM 디버깅·모델 격리용 개별 실행에만 쓰고, 논문 산출물 집계는 아래 `run_all.py` 경로를 따른다.

`--no_skip_existing` 플래그는 `src/baseline/run_all.py`에 구현되어 있으므로 아래 명령어를 그대로 사용하면 된다.

```bash
# summary/intermediate 파일 초기화 (JSON 결과는 실행 중 덮어씌워짐)
rm -f results/phase1_baseline/phase1_summary.csv
rm -f results/phase1_baseline/phase1_intermediate.json

# 4개 모델 × 3데이터셋 × 1시드(42) 일괄 재실행 (BERTScore 포함, 기존 결과 무시)
python -m src.baseline.run_all \
  --output_dir results/phase1_baseline \
  --data_dir data \
  --batch_size 8 \
  --no_skip_existing
```

완료 후 확인:

```bash
ls results/phase1_baseline/*.json | wc -l   # 12개여야 함 (4모델 × 3데이터셋 × 1시드)
cat results/phase1_baseline/phase1_summary.csv  # seed-STD = 0.0 이 정상 (greedy 결정적), 불확실성은 *_ci_low/high 열로 확인
```

> **주의**: 기존 `results/phase1_baseline_pre_bertscore/` 폴더는 건드리지 않는다. 재실행 결과는 `results/phase1_baseline/`에 저장된다.

### Best Model 선택

`phase1_summary.csv`의 `overall_acc_mean` 기준으로 최고 성능 모델 선택 후 메모.

### Phase 1 통계 분석 (RQ1 — 모델 간 성능 차이 검정)

Phase 1 결과 산출 후, 모델 간 zero-shot 성능 차이의 통계적 유의성을 검정한다(설계서 §4.3). zero-shot은 결정적 평가이므로 시드-분산 ANOVA 대신 **공유 테스트셋 짝지은 검정**(Cochran's Q + McNemar, Bonferroni 보정)을 사용한다.

```bash
python scripts/analyze_phase1.py \
  --results_dir results/phase1_baseline \
  --seed 42
```

**산출물**:
- `results/phase1_baseline/phase1_rq1_analysis.md` — 사람이 읽는 리포트 (데이터셋별 + pooled Cochran's Q, McNemar 쌍별, 모델별 부트스트랩 95% CI)
- `results/phase1_baseline/phase1_rq1_analysis.json` — 기계 판독용

> 데이터 오염 강건성까지 확인하려면 아래 Phase 1.5 실행 후 `python scripts/robustness_phase1.py`로 의심 샘플 제거 재계산을 수행한다(설계서 §4.2.1).

### 임상적 의미 분석 (WCA — 보조 지표, PathVQA)

정확도 향상이 임상에서 어떤 의미인지 보완 설명하기 위한 보조 지표(설계서 §4.4.5). PathVQA 질문을 7개 임상 유형으로 분류해 중요도 가중 정확도(WCA)를 산출한다. 저장된 per-sample `correct` 플래그를 사용하므로 **모델 재실행·재채점이 필요 없다**.

```bash
python scripts/analyze_clinical.py \
  --results_dir results/phase1_baseline \
  --dataset pathvqa \
  --seed 42
```

**산출물**: `results/phase1_baseline/clinical_analysis_pathvqa.{md,json}` (모델별 WCA + 질문 유형별 정확도)

> **주의**: WCA 가중치는 외부 검증 없는 임시 척도이므로 참고용 보조 지표로만 사용한다(설계서 §5.3). ECE는 per-sample confidence 미저장으로 현재 산출 불가(리포트에 N/A 표기). Phase 2 PathVQA 결과에도 `--results_dir results/phase2_finetune/<조건 디렉터리>` 로 동일 적용 가능하다.

### Phase 1.5: 데이터 오염 측정 (v0.5 신설)

Phase 1 결과 산출 후, 사전훈련 데이터 오염 가능성을 능동적으로 측정합니다.

```bash
# 4개 모델 × 3개 데이터셋 (선택 모델 Qwen2.5-VL-7B 제외)
for model in qwen3_vl_2b qwen25_vl_3b smolvlm2_2b gemma4_e2b; do
  for dataset in pathvqa slake vqa_rad; do
    python scripts/measure_contamination.py \
      --config configs/models/${model}.yaml \
      --dataset ${dataset} \
      --output_dir results/contamination \
      --k_percent 20
  done
done
```

**예상 소요**: ~4시간 (RTX 4090, 12개 조건 × forward pass)

**결과 분석**:
- `results/contamination/<model>_<dataset>_minK20.json` 파일 12개
- summary.mean_minK 값이 calibration set보다 유의미하게 높으면 contamination 의심
- 의심 sample 제거 후 Phase 1 결과 재계산 필요 (논문 §4.2.1 절차 참조)

---

## 4. Phase 2: QLoRA 파인튜닝

**목표**: 4개 모델 × 3개 데이터셋 × 3개 시드 = 36개 조건

**대상 4개 모델 (THESIS v0.5)**: Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B

> `run_phase2_main.sh`는 `configs/models/*.yaml`을 전부 글롭하되 `enabled: false`는 건너뛴다. 논문 비대상인 `qwen25_vl_7b`와 `_template`은 `enabled: false`로 지정돼 자동 제외되므로, 위 4개 모델만 36개 조건으로 실행된다. (Florence-2는 v0.2에서 탈락 → `_excluded/` 유지)

### 4.0 실행 준비 상태 (2026-07-14, 커밋 ab3fa64 기준) — 필독

Phase 2 학습 코드는 라이브러리 스택 호환 이슈와 Main(full 데이터) 인프라 이슈를 모두 해결했다. **새 pod에서는 반드시 `git pull` 후 `git log --oneline -1`로 ab3fa64 이상인지 확인하고 시작한다.**

**모델별 검증 상태 (2026-07-14 새 pod 스모크로 확정):**
- ✅ **qwen3-vl-2b(best), qwen25-vl-3b**: 학습+평가 완주 확인.
- ✅ **smolvlm2-2b**: 학습+평가 완주 확인 (collator → 병합 dtype → 평가 dtype 3개 벽 해결).
- ✅ **gemma4-e2b**: 학습+평가 완주 확인 (standard backend + 텍스트 전용 LoRA 타깃).
- **4모델 12조건(스모크) 전부 통과** — 라이브러리 스택 이슈는 완전히 해소됨.

**해결된 라이브러리 스택 이슈 (7-12 ~ 7-13):**
- **unsloth 전역 SFTTrainer 패치** → `import unsloth`가 trl.SFTTrainer를 전역 몽키패치해 standard 모델(smolvlm2/gemma4)을 오염시킴. **해결: 조건마다 독립 프로세스 격리(`src/finetune/train_one.py`)** — standard 모델은 `MOAI_SKIP_UNSLOTH=1`로 unsloth 미로드, qwen은 로드. 부수효과: 조건마다 GPU/RAM 완전 해제.
- **standard backend**: trl 0.24 native VLM(`DataCollatorForVisionLanguageModeling`)로 재작성.
- **SmolVLM2 bf16 이미지 병합 dtype 버그**: `get_image_features` 출력(ModelOutput 또는 텐서)을 모델 dtype으로 캐스트.
- **gemma4 ClippableLinear 거부(peft#3129)**: LoRA 타깃을 실제 `nn.Linear`(텍스트 모델)로 한정해 vision/audio 타워의 ClippableLinear 자동 제외.
- **평가 dtype 충돌**: `merge_and_unload` 후 lm_head가 fp32로 남아 generation에서 bf16 hidden state와 충돌 → 평가·CF generation을 `torch.autocast(model dtype)`로 감쌈.

**해결된 Main(full 데이터) 인프라 이슈 (7-14, 스모크에선 안 보이던 것들):**
- **CF baseline OOM**: 메인 프로세스가 CF baseline 모델을 GPU에 올린 채 유지 → 학습 서브프로세스와 겹쳐 OOM. `unload_model` 호출자 쪽 참조까지 끊고 `gc + synchronize + empty_cache`로 확실히 해제.
- **HF 모델 캐시 디스크 quota**: `HF_HOME` 미설정 시 `/workspace` 볼륨(quota 있음)에 모델 4개(~27GB)가 쌓여 `Disk quota exceeded`. → `run_phase2_main.sh`가 `HF_HOME=/hf_cache`(컨테이너)로 고정.
- **데이터 로딩 30분/조건**: full pathvqa 19,654개 이미지를 조건마다 디코딩→재인코딩(`Dataset.from_list`, CPU 바운드)해 36조건이면 몇 주 소요. → `prepare_data.py`가 준비된 데이터셋을 `(dataset,split,format,samples,ratio)` 키로 디스크 캐시(최초 1회 빌드, 이후 `load_from_disk` mmap + 배치별 lazy 디코딩). 36회 재빌드 → 6회(3데이터셋×2포맷)로 축소.
- **캐시 디스크 quota**: 이미지 캐시가 다시 `/workspace` volume을 채움 → `MOAI_CHAT_CACHE_DIR=/hf_cache/chat_cache`(컨테이너)로 재지정.
- **학습 시간(full 3에폭 ≈ 2주)**: `base_qlora.yaml`에 **`max_steps=500`** cap 적용(QLoRA 표준, 조건당 samples_seen=4,000 고정, ~1.8h/조건). `train_qlora`가 max_steps>0이면 eval/save를 끝에서 1회만 수행. **논문 v0.8에 이 학습 예산 변경과 한계(데이터셋 크기별 실효 에폭 차이, PathVQA는 ~0.15epoch 과소학습 가능)를 이미 반영함** — §4.4 표 + §5.3 참조.

**해결된 16GB×2 멀티-GPU pod 이슈 + 조건별 병렬 최적화 (7-15~16, 24GB 단일 GPU 대신 16GB 2장으로 진행한 환경에서 발견):**
- **gemma4 kbit-training OOM (`Tried to allocate 8.75 GiB`)**: `peft.prepare_model_for_kbit_training`이 4bit 아닌 모든 파라미터를 fp32로 블랑켓 업캐스트하는데, Gemma4의 거대 vocab 임베딩(`embed_tokens`/`embed_tokens_per_layer`, frozen이라 LoRA 대상도 아님)까지 걸려 단일 텐서가 ~8.75GiB를 요구 → 16GB 카드에서 OOM(24GB 단일 GPU에서도 peak~16.17GB로 아슬아슬했던 문제). → 해당 함수 호출 **전**에 frozen 임베딩을 CPU로 옮겨 업캐스트가 CPU RAM에서 일어나게 하고, 끝나면 원래 device·dtype으로 복원.
- **DataParallel 재래핑 충돌 (`module must have its parameters ... found on cuda:1`)**: `device_map="auto"`로 모델이 2-GPU에 분산(model-parallel)됐는데 HF Trainer가 이를 인식 못 해 `nn.DataParallel`로 재래핑 시도 → 모델 로드 직후 `model.is_parallelizable=True` + `model.model_parallel=True` 설정(QLoRA 멀티GPU 튜토리얼 표준 해결책)으로 Trainer가 재래핑을 건너뛰게 함.
- **조건별 병렬 실행 최적화(신규)**: 위 두 수정으로 "에러 없이"는 해결됐으나 모델 1개를 2-GPU에 분산하는 구조는 조건을 여전히 순차 실행 — GPU 2장을 써도 속도 이득이 없었다(오히려 GPU간 통신 오버헤드로 손해 가능). `run_phase2.py`에 조건(model×dataset×seed)을 GPU 개수만큼 동시 배정하는 `--max_parallel` 플래그를 추가(기본: `torch.cuda.device_count()` 자동 감지). 각 조건은 `CUDA_VISIBLE_DEVICES`로 GPU 1장에 고정되어 완전히 독립적으로 학습되므로 model-parallel/DataParallel 충돌 자체가 생기지 않고, GPU 2장이면 실질적으로 처리량이 거의 2배가 된다. `--max_parallel 1`이면 기존과 완전히 동일한 순차 실행(회귀 없음). `run_phase2_main.sh`는 플래그를 안 주므로 자동 감지값이 적용된다.

**Main 실행 전 검증 절차 (선택, 새 pod 환경 확인용 — 이미 4모델 검증됐으므로 생략 가능):**
```bash
export WANDB_MODE=offline
python -m src.finetune.run_phase2 --config_dir configs/models \
  --finetune_config configs/finetune/base_qlora.yaml \
  --output_dir results/_phase2_smoke --seeds 42 --data_dir data \
  --max_train_samples 20 --max_eval_samples 20 --max_test_samples 20 --no_cf
ls results/_phase2_smoke/*/train_result.json | wc -l   # 12 기대 (4모델×3데이터셋)
```

**Main 실행 (tmux, 캐시·CF·max_steps·디스크 경로 전부 스크립트에 자동 반영됨):**
```bash
tmux new -s p2
cd /workspace/Masters_degree && source .venv/bin/activate
bash scripts/run_phase2_main.sh
```
첫 조건(알파벳순 gemma4/pathvqa)은 pathvqa std 포맷 캐시를 새로 빌드하므로 **~30분** 걸린다(1회성). 이후 같은 (dataset, format) 조합은 캐시를 즉시 로드해 바로 학습 시작. gemma4/pathvqa 첫 조건이 CF baseline → 학습(500스텝, ~1.8h) → 평가까지 `OutOfMemory`/`Disk quota` 없이 넘어가면 detach(`Ctrl+B, D`)하고 두면 된다.

**환경 주의:**
- RTX 3090 기준 **컨테이너 RAM ≥100GB 필요**(31GB는 OOM).
- **`/workspace` 볼륨은 100GB 이상 권장** (50GB는 venv+data+wandb+결과로 꽉 참 → 7-14 pod에서 실제로 quota 이슈 발생). HF 모델 캐시·chat 캐시는 컨테이너 디스크(`/hf_cache`)로 자동 분리되므로 볼륨은 결과·데이터·checkpoint 용도.
- `uv sync`로 transformers 5.5.0 / torch 2.10.0+cu128 / trl 0.24 / peft 0.18.1 고정. `WANDB_MODE=offline` 필수(스크립트에 내장됨).

> Phase 1 완료 및 best model 확인 후 실행. 4모델 12조건 스모크는 이미 검증됐으므로 바로 Main 실행 가능.

로그 확인:

```bash
tail -f results/phase2_finetune/run_phase2.log
grep -iE "OutOfMemory|Disk quota|FAILED|chat-cache|Summary:" results/phase2_finetune/run_phase2.log | tail
```

### Phase 2 통계 분석 (RQ2 — 파인튜닝 효과 검정)

36개 조건 학습 완료 후, zero-shot(base) 대비 파인튜닝 효과를 3중 검증한다(설계서 §4.4). base는 Phase 1 seed42 결과를, finetuned는 Phase 2 `eval_summary`를 사용한다.

```bash
# Mixed-Effects Model에는 statsmodels/pandas 필요 (미설치 시 해당 부분만 생략됨)
uv pip install statsmodels pandas

python scripts/analyze_phase2.py \
  --phase1_dir results/phase1_baseline \
  --phase2_dir results/phase2_finetune \
  --base_seed 42
```

**산출물**:
- `results/phase2_finetune/phase2_rq2_analysis.md` — 모델별 paired t-test + BCa Bootstrap 95% CI(Cohen's d) + Wilcoxon, 전체 Mixed-Effects Model
- `results/phase2_finetune/phase2_rq2_analysis.json` — 기계 판독용

> Catastrophic Forgetting은 각 조건의 `train_result.json` → `catastrophic_forgetting` 필드(VQAv2 기준 degradation)에 저장된다. WCA 임상 분석은 PathVQA 조건 디렉터리에 `scripts/analyze_clinical.py --results_dir results/phase2_finetune/<조건>` 로 적용한다.

---

## 5. Phase 2 Ablation

**목표**: Best model에 대해 Ablation A/B/C 실험

> Phase 2 main 완료 후 실행

### 실행 전: best model 수정

```bash
nano scripts/run_phase2_ablation.sh
# BEST_MODEL_CONFIG 변수를 Phase 1/2 best model로 수정
```

```bash
bash scripts/run_phase2_ablation.sh
```

---

## 6. Phase 3: Autonomous HPO

**목표**: 4개 전략(Manual/RS/Optuna/Autoresearch) × 10회 반복 × 40 trial = HPO 비교 실험

**예상 규모**: ~1,210 trials, ~200 GPU-hours (RTX 4090 기준 약 8-9일), 비용 ~$78-107 (Community Cloud)

> Phase 2 완료 및 ANTHROPIC_API_KEY 설정 후 실행

### 실행 전: model_config 수정

```bash
nano scripts/run_phase3.sh
# MODEL_CONFIG를 Phase 2 best model로 수정
```

```bash
bash scripts/run_phase3.sh
```

### 체크포인트 재개 (중단 시)

Phase 3는 체크포인트를 자동 저장합니다. 동일 명령어 재실행 시 마지막 완료 trial부터 자동 재개됩니다.

```bash
# 체크포인트 상태 확인
cat results/phase3_autoresearch/checkpoints/hpo_checkpoint.json
```

---

## 7. 결과 다운로드

### 방법 A: scp

```bash
# 로컬(집 컴퓨터) 터미널에서
scp -P <ssh-port> root@<runpod-ip>:/workspace/Masters_degree/results/phase1_baseline/*.json \
    "D:/project/Masters_degree/results/phase1_baseline/"
```

### 방법 B: RunPod UI

RunPod 콘솔 → Files 탭 → `/workspace/Masters_degree/results/` 에서 직접 다운로드

### 방법 C: git push

```bash
# RunPod에서 — results/ 폴더가 .gitignore에서 제외된 경우
git add results/
git commit -m "data: Phase 1 results with BERTScore"
git push
```

---

## 전체 실행 흐름 요약

```
bash scripts/runpod_setup.sh        # 최초 1회 — 의존성 + 데이터 자동 설치
         │
         ▼
bash scripts/runpod_phase1.sh ...   # 모델별 실행 (BERTScore 포함)
         │
         ▼ phase1_summary.csv에서 best model 선택
bash scripts/run_phase2_main.sh
         │
         ▼ run_phase2_ablation.sh 에서 BEST_MODEL_CONFIG 수정
bash scripts/run_phase2_ablation.sh
         │
         ▼ run_phase3.sh 에서 MODEL_CONFIG 수정
export ANTHROPIC_API_KEY=sk-ant-...
bash scripts/run_phase3.sh
```

**수동 개입 포인트**:
- Phase 1 완료 → `phase1_summary.csv` 분석 → best model 결정
- Phase 2 완료 → `run_phase2_ablation.sh`, `run_phase3.sh`에서 모델 변수 수정

---

## 8. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `CUDA out of memory` | VRAM 부족 | `--batch_size 4` (기본 8) 또는 더 작은 모델 |
| `ModuleNotFoundError` | 의존성 미설치 | `pip install -e ".[unsloth]"` 재실행 |
| BERTScore hang | roberta-large 최초 다운로드 중 | 잠시 대기 (자동 다운로드) |
| Phase 3 random fallback | API key 없음 | `export ANTHROPIC_API_KEY=sk-ant-...` |
| 데이터셋 로드 실패 | `runpod_setup.sh` 미실행 | `bash scripts/runpod_setup.sh` 재실행 |
| `wandb` 오류 | WANDB 미설정 | `export WANDB_MODE=offline` |
| `Gemma4ForConditionalGeneration` 로드 실패 | transformers가 4.57.2로 잘못 설치됨 (pip 하한 해석). Gemma4는 5.5.0 전용 | `uv sync`로 재설치(`uv.lock` = 5.5.0 재현). `pip install -e .` 금지 |
| `AutoProcessor`/`Qwen3VLForConditionalGeneration` import 실패 | pip 설치 시 transformers 버전 오설치 또는 unsloth/torch 불일치로 transformers 손상 | `uv sync --extra unsloth`로 검증 스택(transformers 5.5.0 + torch 2.10.0+cu128) 재현. `python -c "import transformers; print(transformers.__version__)"`가 5.5.0인지 확인 |
| `python`이 5.5.0을 못 잡음 (버전이 다르게 나옴) | venv 미활성 (시스템 python 사용 중) | `source /workspace/Masters_degree/.venv/bin/activate` |

---

## 예상 실험 시간 및 비용 (RTX 4090 기준)

| Phase | GPU-hours | 일수 (24h) | 비용 (Community) |
|-------|:---------:|:---------:|:---------------:|
| Phase 1 | ~9h | 0.4일 | ~$4 |
| Phase 2 | ~65h | 2.7일 | ~$25 |
| Phase 3 | ~200h | 8.3일 | ~$78 |
| **합계** | **~274h** | **~11.4일** | **~$107** |

> RTX 4090 2대 병렬 시 기간 약 절반으로 단축 (비용 동일)

---

*최종 업데이트: 2026-05-18 (v0.5 데이터 오염 측정 절차 + v0.4 예상 시간/비용 테이블 통합)*
