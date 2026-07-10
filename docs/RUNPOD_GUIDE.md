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

**목표**: 4개 모델 × 3개 데이터셋 × 3개 시드 평가 (BERTScore 포함)

**평가 모델 (4개, 논문 대상)**: Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B

**선택 모델**: Qwen2.5-VL-7B (성능 비교용, 논문 대상 아님)

### 전체 모델 일괄 실행 (권장 — 요약 CSV 자동 생성)

`run_all.py`는 대상 4개 모델(`enabled: false`인 7B·`_template` 자동 제외)을 각 1회씩 로드해 3 데이터셋 × 3 시드를 평가하고, best model 선택에 필요한 **`phase1_summary.csv`를 생성**한다.

```bash
python -m src.baseline.run_all \
  --output_dir results/phase1_baseline \
  --data_dir data \
  --batch_size 8
```

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

기존 결과가 BERTScore 없이 집계됐거나 STD=0.0 버그가 의심될 경우 전체 재실행한다.

`--no_skip_existing` 플래그는 `runpod_phase1.sh`와 `src/baseline/run_all.py` 모두에 구현되어 있으므로 아래 명령어를 그대로 사용하면 된다.

```bash
# summary/intermediate 파일 초기화 (JSON 결과는 실행 중 덮어씌워짐)
rm -f results/phase1_baseline/phase1_summary.csv
rm -f results/phase1_baseline/phase1_intermediate.json

# 4개 모델 순차 실행 (BERTScore 포함, 기존 결과 무시)
bash scripts/runpod_phase1.sh --config configs/models/qwen3_vl_2b.yaml  --no_skip_existing
bash scripts/runpod_phase1.sh --config configs/models/qwen25_vl_3b.yaml --no_skip_existing
bash scripts/runpod_phase1.sh --config configs/models/smolvlm2_2b.yaml  --no_skip_existing
bash scripts/runpod_phase1.sh --config configs/models/gemma4_e2b.yaml   --no_skip_existing
```

완료 후 확인:

```bash
ls results/phase1_baseline/*.json | wc -l   # 36개여야 함
cat results/phase1_baseline/phase1_summary.csv  # STD != 0.0 확인
```

> **주의**: 기존 `results/phase1_baseline_pre_bertscore/` 폴더는 건드리지 않는다. 재실행 결과는 `results/phase1_baseline/`에 저장된다.

### Best Model 선택

`phase1_summary.csv`의 `overall_acc_mean` 기준으로 최고 성능 모델 선택 후 메모.

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

> Phase 1 완료 및 best model 확인 후 실행

```bash
bash scripts/run_phase2_main.sh
```

로그 확인:

```bash
tail -f results/phase2_finetune/run_phase2.log
```

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
