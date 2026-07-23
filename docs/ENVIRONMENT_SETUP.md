# 새 컴퓨터/새 Pod 환경 설정 체크리스트

Medical VQA VLM 석사 연구 — RunPod 외 다른 컴퓨터(새 RunPod pod, KISTI 뉴론 등 다른 GPU 클러스터 포함)에서
이 프로젝트를 실행할 때 필요한 환경을 정리한 문서. `docs/RUNPOD_GUIDE.md`(실행 절차)와 `docs/NEXT_SESSION.md`
(작업 재개 지점)를 보완한다 — 실행 명령어는 `RUNPOD_GUIDE.md`, "지금 뭘 해야 하는지"는 `NEXT_SESSION.md`,
"환경이 갖춰져 있는지"는 이 문서.

---

## 1. 하드웨어

| 항목 | 요구사항 |
|------|---------|
| GPU | RTX 4090(24GB) 권장, 최소 RTX 3090(24GB) — 현재 pod는 3090 |
| VRAM | 24GB (2B/3B 모델만 쓸 경우 A5000 24GB도 가능) |
| Container Disk | 80GB 이상 (모델 hub 캐시 전용 — 2026-07-22 실제로 90%까지 찬 이력 있음, 여유 있게) |
| Volume/영구 Disk | 100GB 이상 (데이터셋 chat_cache + 결과물 저장) |
| 멀티 GPU 대안 | 16GB×2 (예: RTX 4080 Super×2) 조합도 실증됨 — `--max_parallel`로 조건별 GPU 1장씩 병렬 배정 |

## 2. OS / 컨테이너 전제조건

- Linux, Python 3.12, CUDA 12.8 드라이버(예: RunPod PyTorch 2.8/cu128 템플릿)
- **`$HOME` 환경변수를 가장 먼저 확인할 것.** 2026-07-22 disk quota 장애의 근본 원인이 이것이었다.
  현재 pod는 `$HOME=/workspace`라, 캐시 관련 환경변수(4번 항목)를 빠뜨리면 조용히 영구 볼륨 quota를
  채우고 몇 시간 뒤에야 `Disk quota exceeded`로 드러난다. 새 컴퓨터에서는 `echo $HOME`으로 먼저
  확인하고, 그 값을 기준으로 캐시 경로를 다시 설계해야 한다.

## 3. 리포지토리 설정

```bash
cd /workspace   # 또는 영구 디스크가 마운트된 경로
git clone https://github.com/taewook486/Masters_degree.git
cd Masters_degree
bash scripts/runpod_setup.sh
```

`runpod_setup.sh`가 자동 처리하는 항목:
- `uv` 설치 + `uv sync --extra unsloth` (**반드시 `uv sync`** — `pip install -e .`는
  `pyproject.toml` 하한만 보고 transformers 4.57.2를 잘못 설치해 Gemma4 로드 실패로 이어짐)
- 검증된 스택 고정: **transformers 5.5.0 / torch 2.10.0+cu128 / trl 0.24 / peft 0.18.1**
  (`uv.lock`에 정확히 고정돼 있음)
- PathVQA/SLAKE/VQA-RAD + VQAv2 subset(CF 측정용) 자동 다운로드 — 로컬에서 데이터 업로드 불필요

## 4. 필수 환경변수 (2026-07-22 핵심 교훈)

캐시 디스크 분산 배치는 **공용 설정이 아니라 스크립트마다 개별 export**되어 있다. 새 컴퓨터에서
실행 스크립트를 그대로 옮겨왔다면 아래가 다 들어있는지 확인할 것.

```bash
export HF_HOME=/hf_cache                                    # 모델 가중치 → 컨테이너 디스크
export MOAI_CHAT_CACHE_DIR=/workspace/hf_cache/chat_cache   # 데이터셋 캐시 → 영구 볼륨
export WANDB_MODE=offline                                    # 온라인 동기화 시도로 멈추는 것 방지
export PYTORCH_ALLOC_CONF=expandable_segments:True            # GPU 메모리 파편화 완화
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=medical-vqa-vlm
```

- `.sh` 래퍼(`run_phase2_main.sh`, `run_phase2_ablation.sh`, `run_phase3.sh`, `runpod_phase1.sh`)는
  이미 내장돼 있음(2026-07-22 전수 점검 후 누락분 추가 완료).
- `.sh` 없이 `python -m ...`을 직접 호출하는 경우(`measure_cross_dataset_cf.py`, `run_all.py` 등)는
  **셸에서 직접 export**해야 함 — 스크립트가 대신 잡아주지 않는다.
- `runpod_phase1_gemma4.sh`는 deprecated, 미수정 상태이니 사용하지 말 것.
- 경로 값(`/hf_cache`, `/workspace`)은 현재 pod 구조 기준이다. 완전히 다른 컴퓨터(다른 pod, KISTI
  뉴론 등)라면 컨테이너 디스크/영구 볼륨의 실제 마운트 경로에 맞게 값을 다시 잡아야 한다.

## 5. API 키 / 시크릿

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Phase 3 HPO 필수 (없으면 random fallback으로 저하됨)

# 세션 재시작 대비 영구 설정
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
```

- SSH 개인키(`runpod.ppk`) 등은 `.gitignore`로 보호돼 있어 git에 없다 — 새 컴퓨터에서 별도 준비 필요.

## 6. 디스크 관리 체크리스트

- `df -h`는 네트워크 볼륨(MooseFS 등)에서 리전 전체 풀 용량을 보여줘 오해를 줄 수 있다 — 실제 pod
  quota는 `du -h --max-depth=1 <path>`로 확인 (`-s`와 `--max-depth`는 동시 사용 불가, 같이 쓰면
  `du: warning`만 뜨고 결과가 안 나온다).
- 컨테이너 디스크(`/hf_cache/hub`)엔 현재 실험에 필요한 모델만 남길 것 — 안 쓰는 모델 스냅샷이
  모델당 수 GB~10GB대라 금방 쌓인다.
- `MOAI_CHAT_CACHE_DIR`의 비율별(subset ratio) 데이터셋 캐시는 해당 조건들이 전부 끝나면 삭제해도
  안전하다(다음에 필요하면 재빌드만 하면 됨).
- 학습 완료 후 raw 체크포인트(`checkpoints/`, 옵티마이저 상태 포함)는 최종 가중치가 `adapter/`로
  이미 추출된 뒤라 자동 삭제되도록 코드에 반영돼 있다(`src/finetune/train_qlora.py`).

## 7. 검증 방법

```bash
python -c "
import torch, transformers
print(f'PyTorch: {torch.__version__}')              # 2.10.0+cu128 이어야 함
print(f'Transformers: {transformers.__version__}')  # 5.5.0 이어야 함 (Gemma4 필수)
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

## 8. 다른 GPU 클러스터(KISTI 뉴론 등)로 옮길 경우 추가 고려사항

- SLURM 등 잡 스케줄러 구조가 RunPod의 즉시 실행형 컨테이너와 다름 — 스크립트를 배치 작업(sbatch 등)
  형태로 감싸야 할 수 있음.
- `$HOME`/스토리지 마운트 구조가 다를 가능성이 높음 — 4번 환경변수 경로를 그 환경의 실제 컨테이너
  디스크·영구 스토리지 마운트 지점에 맞게 다시 설계해야 함.
- `WANDB_MODE=offline`이 여기서도 여전히 필요한지(외부 네트워크 접근 제한 여부) 확인 필요.
- GPU 종류·VRAM이 다르면(예: A100) `configs/finetune/base_qlora.yaml`의 batch size/gradient
  accumulation 설정을 재조정할 여지가 있음(현재 값은 RTX 3090/4090 24GB 기준).

---

Version: 1.0.0
최초 작성: 2026-07-22
근거: `docs/RUNPOD_GUIDE.md`, `docs/NEXT_SESSION.md`, `scripts/runpod_setup.sh`,
2026-07-22 세션의 Phase 2 Ablation disk quota 디버깅 과정에서 확인된 사실들
