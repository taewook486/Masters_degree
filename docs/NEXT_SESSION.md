# 다음 세션 시작점 (마지막 갱신: 2026-08-09)

이 파일은 컴퓨터가 바뀌어도(로컬 `~/.claude` 메모리는 컴퓨터별로 따로 저장되어 동기화되지 않음)
`git pull` 한 번이면 항상 최신 상태로 받아지도록, 다음에 할 일을 저장소에 직접 남겨둔 것입니다.

## 현재 상태

- **2026-08-09 세션 — RunPod Phase3 본실행 진행상황 확인 (RunPod로 재전환된 상태, 아래 "다음 세션 최우선 작업"의 로컬 GPU 계획은 이후 상황과 다름)**
  - **[중요] 07-31 세션에서 "RunPod 접고 로컬 듀얼 GPU로 전환" 결정했었는데, 08-05~08-06 세션부터 RunPod(3090)로 다시 전환되어 Phase3 본실행이 계속 진행 중임** — 상세 경위는 로컬 auto-memory `2026-08-06-phase3-runpod-main-run-status` / `2026-08-07-phase3-runpod-monitoring-handoff` / `2026-08-08-phase3-notify-system-session-wrap` 참고(다른 컴퓨터엔 없을 수 있음).
  - **SSH 접속 정보(재사용)**: `ssh -i runpod_openssh.pem -p 40127 root@213.192.2.86`, pod 작업경로 `/workspace/Masters_degree`. 키 원본은 저장소 루트 `runpod_openssh.pem`(반드시 gitignore 확인 — 절대 커밋 금지).
  - **2026-08-09 08:03 UTC 기준 진행률**: manual 200/200 완료, random 200/200 완료, optuna 125/200 진행중(repeat 6~7 학습 중, trial당 평균 30분, 남은 75개 약 37~40시간 예상), autoresearch 0/200 대기(optuna 끝나야 시작). 최고 성능은 optuna trial 304(repeat 3) val_accuracy=0.4700.
  - **자동 완료 알림 구축돼 있음**: pod에 `scripts/notify_optuna_done.sh`를 nohup+disown으로 상시 실행 중(PID는 세션마다 다름, `ps aux | grep notify_optuna`로 확인) — optuna 200/200, autoresearch 200/200 각 시점에 텔레그램으로 자동 알림 발송. **SSH 세션이나 Claude Code 세션과 무관하게 pod 자체에서 독립 동작**하므로, 완료 전까지는 다른 컴퓨터에서도 별도 조치 불필요 — 궁금하면 위 SSH로 접속해 `python3 scripts/summarize_stage.py optuna`(또는 `optuna autoresearch`)로 최신 요약만 재생성해서 확인.
  - **[보안, 확인 필요] 텔레그램 봇 토큰이 이번 세션 대화 로그에 평문 노출됨**(`scripts/notify_optuna_done.sh` 안에 하드코딩된 `TELEGRAM_BOT_TOKEN` 값을 cat으로 출력함). 새로 유출된 건 아니고 원래 스크립트에 하드코딩돼 있던 값이지만, 민감하다고 판단되면 봇 토큰 재발급 고려할 것(BotFather에서 `/revoke`).
- **2026-07-31 세션 — Claude Code SessionStart 훅 에러 원인 진단(수정은 보류, 이 컴퓨터 포맷 예정)**
  - 증상: 세션 시작 시 SessionStart 훅이 에러남.
  - **원인 확정**: 프로젝트 설정 `.claude/settings.json`의 `env.PATH` 값이 Windows 스타일(`C:\Users\taewo\...`을 세미콜론 `;`으로 구분)로 되어 있어, WSL(Linux) bash에서는 통째로 하나의 잘못된 경로 문자열로 인식됨 → `ls`/`cat`/`head`/`which`/`git`/`grep` 등 기본 명령어조차 전부 `command not found`로 실패. 이 때문에 SessionStart를 포함한 `.claude/hooks/moai/*.sh` 훅 전체가 (`command -v moai` 같은 PATH 의존 호출이 깨지며) 정상 동작 못 하는 상태였음.
  - **대조**: 사용자 레벨 `~/.claude/settings.json`에는 이미 올바른 형식(콜론 `:` 구분, WSL 마운트 경로 `/mnt/c/...`)의 PATH가 들어 있어 정상 동작함 — 예: `/home/taewook/.local/bin:/home/taewook/go/bin:/usr/local/bin:...:/mnt/c/WINDOWS/system32:...`. 즉 문제는 프로젝트 설정이 이 정상 PATH를 깨진 값으로 덮어쓰는 것.
  - git 히스토리 확인 결과, 프로젝트 `.claude/settings.json`의 PATH 값이 "정상(콜론+`/c/...`)"과 "깨짐(세미콜론+`C:\...`)" 사이를 여러 차례 왔다갔다 함 — 수동 편집이라기보다 어떤 자동 갱신(예: `moai update`, ConfigChange 훅, 다른 도구 등)이 반복적으로 깨진 형식을 재도입하는 것으로 추정(원인 도구까지는 특정 못 함).
  - moai 바이너리 위치 확인됨: `/home/taewook/.local/bin/moai`, `/home/taewook/.local/bin/moai-adk`(WSL 네이티브 `$HOME/.local/bin`) — `$HOME/go/bin`은 없음.
  - **조치는 하지 않음** — 이 컴퓨터를 포맷할 예정이라 오늘은 진단만 하고 정리함. 새 컴퓨터/재설치 후 아래 "알아둘 것"의 PATH 관련 항목부터 확인할 것.
- **2026-07-31 세션 — RunPod pod 최종 정리(백업 확인 후 terminate)**
  - 세션 시작 시 pod가 또 마이그레이션됨(`troubled_cyan_shrimp-migration` → `troubled_cyan_shrimp-migration-migration`, RTX 3090). 07-27 마이그레이션 때와 동일 패턴 — 데이터 유실 없음, 진행률 표시만 신뢰 불가.
  - **`results/` 폴더 전체를 로컬로 백업 완료** (`D:\project\Masters_degree\results_pod_backup\`) — scp 사용. Windows PowerShell에서 겪은 이슈들: (1) `\` 줄바꿈 미지원 → 한 줄 명령으로 수정, (2) `.ppk`(PuTTY 형식) 키를 OpenSSH `scp`가 못 읽음 → PuTTYgen "Export OpenSSH key"로 변환 필요, (3) 변환한 키 파일 권한이 너무 열려있어 거부됨 → `icacls <키파일> /inheritance:r` + `icacls <키파일> /grant:r "<사용자명>:(R)"`로 해결.
  - **백업 무결성 3중 검증 완료**: ① `adapter_model.safetensors` 86개 전부 일치(손상 파일 0개), ② `git status --short`의 미추적 파일 138개 전부 로컬에 존재(0개 누락), ③ 바이트 단위 총합 비교 — pod 3,402,012,129 bytes vs 로컬 3,402,012,403 bytes (차이 274바이트, 오차 수준). **데이터 유실 없이 완전히 백업됨 확정.**
  - `results/` 안에 어댑터 가중치·체크포인트가 전부 포함되어 있음을 코드로 확인(`train_qlora.py`/`run_phase2.py`/`run_phase3.py` 전부 `output_dir` 기본값이 `results/phase2_finetune`, `results/phase3_autoresearch*`) — 앞으로 pod 백업 시 `results/` 폴더만 통째로 받으면 충분함.
  - **두 pod(`troubled_cyan_shrimp-migration`, `troubled_cyan_shrimp-migration-migration`) 모두 terminate 결정.** RunPod은 07-28에 이미 접기로 확정됐던 상태라(학과 비용지원 불가, 지도교수 회신 없음 → 로컬 듀얼 GPU로 전환), 백업만 확인되면 pod를 유지할 이유가 없음.
- **2026-07-28 세션 — Phase 3 스모크 테스트 마무리 + 인프라 정리**
  - Phase 3 HPO 스모크 테스트 결과 백업 완료(7/7 trial, 어댑터 가중치·토크나이저·옵티마이저 상태는 제외) — 커밋 `612191a`.
  - Phase 3 재개(resume) 시 이미 완료된 전략의 남은 trial이 0번 도는 버그 수정 — 커밋 `54397e5`.
  - Phase 3 최종 평가 샘플 수 제한 옵션(`--max_test_samples`) 추가 — 커밋 `667e177`.
  - README + CHANGELOG를 실제 진행 상황(Phase 1 1시드, Phase 2 완료, Phase 3 스모크)에 맞춰 동기화 — 커밋 `988e709`. 이후 origin과 merge 완료(`849e083`).
  - **Phase 3 본 실행(전체 1,210 trial) 비용 지원 — 학과 사무실 답변 옴: 지원 불가.** 지도교수에게 메일 문의했으나 바빠서 회신 없음 → **회신 대기 중단, RunPod 포기하고 로컬 5060 Ti + 4060 듀얼 GPU 조합으로 마무리하기로 확정.**
  - **[정정, 조치 불필요] `results/phase3_autoresearch_smoke/random_repeat0/trial_0016`은 미완료가 아니라 고아(orphan) 폴더로 확인됨**: 처음엔 미완료 trial로 의심했으나, `results.tsv`를 직접 확인한 결과 스모크 테스트는 이미 7/7 완료(manual 1, random 2, optuna 2, autoresearch 2)로 정상 종료된 상태였음. 이 폴더는 `results.tsv`의 어떤 행과도 매칭 안 되는 번호 중복 폴더(`optuna_repeat0`에도 별도로 완료된 `trial_0016`이 존재)로, 재개 버그(`54397e5`로 수정됨)가 있던 시점에 잘못 재실행되다 만 흔적으로 추정 — 이어서 돌릴 필요 없음, 나중에 정리 삭제해도 무방.
  - **RunPod pod는 이번 세션 종료 시 중지(stop)시킴.**
  - **로컬 듀얼 GPU(5060 Ti + 4060) 전환 — 다른 claude.ai 대화(테스트 계획 상담)를 검증하며 버그 2건 발견·수정**:
    - `run_phase3.bat`의 `--time_budget_min 15`는 위험함 확인(pod 스모크 실측 22.3~49.6분 — 15분보다 이미 김) → **60분으로 수정**.
    - `docs/RUNPOD_COST_ESTIMATE.md`의 총 trial 계산 오류(`1 + 400 + 400 + 400 = 1,201`, manual을 반복 횟수 무시하고 1로 계산) → **`10 + 400 + 400 + 400 = 1,210`으로 정정**(코드상 manual도 repeat마다 1trial씩 돌아 실제로는 10trial).
    - `--model_config qwen3_vl_2b.yaml`(현재값)이 Phase 2 실측(`phase2_summary.csv`, PathVQA overall_acc 0.502 vs qwen25-vl-3b 0.480, 속도도 40% 빠름) 기준으로 이미 최선의 선택임을 확인 — **바꿀 필요 없음**(다른 대화는 qwen25-vl-3b로 바꾸자고 제안했었는데 근거가 부족했음).
  - **반복 횟수는 10회 유지로 확정** (5회로 줄이면 설계서(v0.5)가 명시한 통계적 검정력 목표를 못 재현함 — Kruskal-Wallis/Mann-Whitney U 검정 단위인 run-level 10개를 지켜야 함).
  - **로컬 실측 스모크 스크립트 2개 신규 작성, 아직 미실행**: `run_phase3_smoke_gpu0.bat`(`CUDA_VISIBLE_DEVICES=0`), `run_phase3_smoke_gpu1.bat`(`CUDA_VISIBLE_DEVICES=1`) — 각 카드에서 전략당 1trial(총 4개)만 돌려 실측 시간을 잰 뒤, 그 값으로 repeats=10 본 실행의 전략당 trial 수를 정할 예정. GPU 분배는 "전략별"이 아니라 **"repeat 번호로 절반씩"**(예: repeat 0~4=GPU0, 5~9=GPU1) — Optuna TPE는 같은 repeat 안에서만 순차적이고 repeat끼리는 독립이라 이 분배가 더 공평함. 4060(8GB)도 이 모델 기준 peak VRAM 3.9GB라 배치 크기 걱정 없음(Phase 2 실측 확인).
- **2026-07-27 세션 — pod GPU 마이그레이션 사고 + 복구, Phase 1 폴더 정정, Phase 3 스모크 테스트 진행 중(디스크 위기 미해결 상태로 세션 종료)**
  - **pod GPU 마이그레이션**: 세션 시작 시 옛 pod(`troubled_cyan_shrimp`)가 "GPU no longer available"로 재시작 불가 → RunPod "Automatically migrate" 진행, 첫 확인 땐 `/workspace/Masters_degree`가 비어 보여 데이터 유실처럼 보였으나 **실제로는 대시보드 마이그레이션 진행바가 끝나기 전에 너무 일찍 확인한 것**이었음(진행률 %가 왔다갔다 하다 "migration completed successfully" 토스트가 뜨고 나서야 실제 완료). 완료 후 `adapter_model.safetensors` 77개(Phase 2 75조건+검증 2) + phase2_finetune 조건 폴더 75개 전부 확인, git 히스토리도 정상 — **데이터 유실 없음 확정**. 새 pod 이름은 `troubled_cyan_shrimp-migration`. 옛 pod는 이 확인 후 **terminate 완료**.
    - 교훈(auto-memory `project-runpod-workflow`에도 기록): 마이그레이션 진행바를 너무 일찍 믿지 말 것 — "migration complete" 알림이 뜨기 전엔 빈 폴더로 보여도 패닉하지 말 것. `git checkout -- .`로 git 추적 파일(코드/JSON/CSV/MD)은 복원 가능하지만, 어댑터 가중치(`.safetensors`)처럼 git 미추적 대용량 파일은 마이그레이션 자체가 끝나야만 살아있음 — `git status --short`에서 `D`(진짜 유실)와 `??`(추적 안 됨이라 정상)를 구분해서 판단할 것.
  - **Phase 1 결과 폴더 이름 오류 수정 — 커밋 완료(`d1bbc3f`, push 완료)**: 설계서 §4.3/`RUNPOD_GUIDE.md`는 Phase 1을 "1시드(42)만, 12조건(4모델×3데이터셋), `run_all.py`로 생성"이라 명시하는데, 실제 `results/phase1_baseline/`(메인 폴더로 오인되기 쉬운 이름)에는 구형 3시드 디버깅 스크립트(`runpod_phase1.sh`) 산출물(27개, gemma4 빠짐)이 들어있었고, 설계 그대로인 12조건+RQ1+임상분석 결과는 `results/phase1_baseline_rescored/`(마치 임시 백업처럼 보이는 이름)에 있었음. 이름을 서로 바꿔서 바로잡음: `phase1_baseline`(구, 3시드 27개) → `phase1_baseline_3seed_debug`로, `phase1_baseline_rescored`(구, 1시드 12조건 정본) → `phase1_baseline`으로 rename(`git mv`). `scripts/rescore_phase1.py`/`scripts/robustness_phase1.py`의 기본 경로도 같이 갱신.
  - **Phase 3 HPO 스모크 테스트 — 여러 차례 실패 끝에 겨우 학습 진입, 디스크 위기로 세션 종료 시점에 결과 미확인**:
    1. 첫 시도: `python3`가 시스템 파이썬을 가리켜 `ModuleNotFoundError: anthropic` — `uv run python`으로 우회(기존에 알려진 패턴, 2026-07-27에도 재발함 — venv activate가 자꾸 풀리는 것으로 보임).
    2. 데이터셋(PathVQA train, 19,654개) 캐시 굽는 데 ~2시간 소요(간헐적으로 손상된 TIFF 파일 만나 느려짐) — 일회성 비용이라 그대로 진행.
    3. wandb API 키 없이 첫 실행 → 전 trial `Trial N FAILED: No API key configured` — `WANDB_API_KEY` 발급 후 export.
    4. 두 번째 실행 → 전 trial `Trial N FAILED: 'images'` — 원인: `qwen3_vl_2b.yaml`은 항상 Qwen 전용 데이터 포맷(`prepare_qwen_chat_dataset`, `images` 컬럼 없음)을 쓰는데, **unsloth가 설치 안 돼 있어서** standard/PEFT 백엔드로 자동 전환됐고 이 백엔드는 `images` 컬럼을 요구함 → 포맷 불일치로 즉시 실패. `unsloth`가 없어진 원인은 마이그레이션 후 `uv run python -c "import anthropic..."`(`--extra unsloth` 없이)를 실행했을 때 암묵적 재동기화로 빠진 것으로 추정.
    5. `uv sync --extra unsloth`로 복구(부작용: `statsmodels`/`pandas`가 같이 빠짐 — **다음에 RQ2 재분석 돌릴 일 있으면 `uv pip install statsmodels pandas` 먼저 할 것**).
    6. 세 번째 실행 → wandb 정상 연결(`wandb.ai/taewook486-konkuk-university/medical-vqa-vlm`), 학습 프로세스가 실제로 CPU 600%+ 쓰며 진행 중인 것까지 확인. **그런데 이 시점에 디스크가 다시 96G→99.8G/100G로 치솟음**(원인: 새 tmux 창에서 `HF_HOME` export 없이 실행돼 `/workspace/.cache/huggingface`로 모델 캐시가 샌 것으로 추정 — 4개 모델 8.4G, 이후 안정화됨). `du`(96G 합산)와 RunPod 대시보드(99.8G) 사이 3.8G 차이가 안 맞았는데, `lsof +L1`로 확인해도 유의미한 삭제-후-미해제 파일 없었음 — MooseFS(네트워크 볼륨) 휴지통 지연 반영으로 추정. **몇 분 뒤 재확인 시 디스크가 96G로 자동으로 다시 내려감 — 추정이 맞았던 것으로 보임(별도 조치 없이 해결).**
    7. **학습이 실제로 정상 진행됨을 확인** — unsloth 백엔드로 로드, `Num examples = 19,654 | Total steps = 200`, loss가 3.23→0.65로 정상 수렴, 평가(1,565~1,680 샘플)까지 완주. **세션 종료 시점까지 `results/phase3_autoresearch_smoke/results.tsv`에 2/8 trial 완료**(`grep completed | wc -l` = 2): manual(minimal targets) `train_time_min=28.1`, random(full targets) `train_time_min=22.3` — 평균 ~25분/trial.
    - **[중요 발견] 기존 비용 추정치가 크게 낙관적이었을 가능성**: 표본 2개뿐이지만, 평균 ~25분/trial을 그대로 전체(1,210trial)에 대입하면 **~500시간(~21일)** 규모로, `RUNPOD_GUIDE.md`의 기존 추정(~200시간, ~$78-107)을 훨씬 초과함. 다음 세션에서 8/8(또는 그에 가깝게) 완료 후 평균을 다시 내서 확정할 것 — 전체 실행 여부(`scripts/run_phase3.sh`) 결정에 직결되는 핵심 수치임.
    - **세션 종료 시점 상태**: tmux 세션 `p3smoke` 안에서 스모크 테스트가 계속 실행 중인 채로 세션 마무리(사용자가 자리를 비움, 원격에서 이어볼 예정). 다음 세션 시작 시 `tmux attach -t p3smoke`로 이어보고 진행 상황부터 확인할 것.
- **2026-07-26 세션 완료 — 우선순위 ①②③(Cross-dataset CF → Phase 1 재실행 → Phase 2 재분석) 전부 끝남** ✅
  - Ablation-C 컨파운드 6조건 재실행 완료 + target modules 최적 조합 확정: **rank=64, alpha=128, target=full(all-linear)** → `configs/finetune/base_qlora.yaml`에 반영, push 완료(커밋 `f52080b`). ratio=1.0/rank=64/target=full은 각각 축 하나씩만 바꿔가며 독립 검증한 것으로, 세 값을 동시에 적용한 조합 자체는 미검증(한계점으로 기록해둘 것).
  - Cross-dataset CF 72/72 조건 측정 완료(커밋 `20be0eb`). 도중 `transformers==5.5.0`의 `TokenizersBackend` 리팩터링으로 `bert-score`가 쓰는 `build_inputs_with_special_tokens`가 빠진 버그 발견·수정(커밋 `121af8e`, `src/evaluate/metrics.py`) — roberta-large/biobert 둘 다 실측 검증됨.
  - Phase 1 재실행을 "36개 전부"가 아니라 **백업 복원(27개, `_pre_bertscore/`) + gemma4-e2b 9→3개만 신규 실행**으로 대체해 GPU 시간 절감(커밋 `25f9f9f`). `scripts/rescore_phase1.py`가 원본에 bertscore 없는 조건에서 빈 값을 남기던 버그도 수정(커밋 `71bb476`). RQ1(McNemar/Cochran's Q), 임상분석(WCA, pathvqa), RQ2(Mixed-Effects Model) 전부 완료·백업(커밋 `8ec9fed`~`ad850b3`).
  - **RQ2 해석 주의**: 아래 참고.
- **RQ2 Mixed-Effects Model 해석 주의 (2026-07-26, 논문 작성 시 반드시 반영)** — `results/phase2_finetune/phase2_rq2_analysis.md`의 MEM(`accuracy ~ condition + dataset`, group=seed) 고정효과는 **p=0.3629로 유의하지 않고 ICC(seed)=0.0**으로 나오는데, 이건 계산 오류가 아니라 **MEM이 4개 모델을 구분 없이 합쳐서 추정**하기 때문임. 실제로는 모델별 3중 검증(paired t-test + BCa Bootstrap Cohen's d + Wilcoxon)에서 효과가 **모델마다 정반대**로 나타남 — qwen25-vl-3b(d=+2.65, 유의) / qwen3-vl-2b(d=+1.62, 유의)는 파인튜닝이 확실히 도움됐고, smolvlm2-2b(d=-2.28, 유의)는 오히려 나빠졌고, gemma4-e2b(d=-0.65, 비유의)도 부정 방향임. 이질적(heterogeneous) 효과가 pooled 평균에서 상쇄되며 MEM이 "효과 없음"으로 보이는 것 — **RQ2 결론은 모델별 3중 검증 결과를 1차 근거로 쓰고, MEM pooled 결과는 "모델 구분 없는 전체 효과는 유의하지 않음(모델간 이질성 때문)"이라는 보조 설명으로만 인용할 것.** 코드는 수정하지 않기로 결정함(2026-07-26 확정) — MEM 수식에 model 고정효과/상호작용을 추가하는 개선은 보류.
- **RunPod pod 중지 예정 (2026-07-25 자정)** — 사용자가 오늘 세션을 마무리하며 pod를 중지(stop)함. 재시작 후 아래 "다음 세션 최우선 작업"부터 진행.
- **[주의] GitHub PAT 토큰 노출됨, 무효화 여부 미확인** — 이 세션 중 `git remote -v` 실행 결과가 그대로 대화에 노출되어 토큰이 평문으로 남았음. 다음 세션 시작 시 GitHub → Settings → Developer settings → Personal access tokens에서 해당 토큰이 아직 살아있는지 확인하고, 살아있다면 즉시 revoke 후 재발급할 것.
- **Phase 2 Main+Ablation 전체(75조건) 완료, git 백업+push 완료** ✅ — Main 36 + Ablation A(비율 5종×3시드=15) + B(rank 5종×3시드=15) + C(target 3종×3시드=9) = 75조건. 결과(JSON/CSV/MD만, 어댑터 가중치 제외) 커밋 `8230e64`로 push 완료.
  - **확정**: ablation_a → **ratio=1.0**이 최적(단조 증가, 아직 한계치 안 보임). ablation_b → **rank=64**가 최적(단조 증가, 시간·VRAM 비용 거의 안 늘어남). 재실행 불필요.
- **[긴급] Ablation-C 6개 조건(minimal 3시드 + medium 3시드) 재실행 필요** — 2026-07-25 밤 재검증으로 발견: 이전에 "target_medium/seed123 조건 하나만" 문제라고 진단했었으나(아래 옛 기록), 각 조건 `train_result.json`의 `metadata.timestamp`를 직접 대조한 결과 **minimal 3시드 전부(평균 390분) + medium 3시드 전부(평균 300분), 총 6개 조건**이 `max_steps=500` 캡(커밋 `64a3fe9`) 적용 전에 이미 실행 완료되어 있었던 것으로 확인됨(캡 적용 후 정상 실행된 full은 평균 36.8분). `skip_existing=True` 기본값 때문에 캡 반영 후 재실행해도 이 6개는 스킵되어 옛 결과가 그대로 남아있음. 상세 경위는 로컬 auto-memory `phase2-ablation-c-confound` 참고. **git pull 후 아래로 재실행**:
  ```bash
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_minimal_seed42
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_minimal_seed123
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_minimal_seed456
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_medium_seed42
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_medium_seed123
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_medium_seed456
  bash scripts/run_phase2_ablation.sh   # skip_existing=True라 나머지 조건은 자동 스킵, 삭제한 6개만 재학습(~37분/조건, 총 3~4시간 예상)
  ```
  재실행 후 `wc -l results/phase2_finetune/phase2_summary.csv`로 중복 행 없이 76줄(헤더+75) 유지되는지 확인할 것 — 검증 안 된 부분.
  (BEST_MODEL_CONFIG가 `qwen3_vl_2b.yaml`이 아니면 디렉터리명의 `qwen3-vl-2b` 부분을 실제 `model_name`으로 바꿀 것.)
  - *(옛 기록, 참고용 — 위 재검증으로 대체됨)*: 2026-07-25 최초 발견 시엔 `target_medium/seed123` 조건 하나만 문제라고 판단해 그 조건만 삭제 후 재실행을 계획했었으나, 표본 하나만 우연히 걸려 나온 불완전한 진단이었음.
- **Phase 2 Main: 36/36 전 조건 완료** ✅ (결과 git 백업 완료, 커밋 d6c6cd5)
- **Phase 2 Ablation: 원인 수정 후 재시작, 07-22 01:33 UTC부터 재실행 중 (오후에 진행 상황 재확인 예정)**
  - **수정 검증됨**: 재시작 후 이전 실패 지점(`unsloth_compile_transformers`의 `Conv1d` AttributeError)을 정상 통과, 모델 로딩·LoRA 패칭 완료, train split 생성까지 에러 없이 진행 확인(01:35 UTC 기준). 캐시 경로 수정이 root cause를 잡은 것으로 사실상 확정.
  - 재확인 시 우선 체크: `ls results/phase2_finetune/ablation_*/train_result.json | wc -l`, `tail -n 30 results/phase2_finetune/run_phase2.log`, `nvidia-smi`
  - 목표: Ablation A(데이터비율 5종)+B(rank 5종)+C(타겟모듈 3종) × 3 seed = 39개 조건, PathVQA로만, full test set 평가.
  - 2026-07-21 21:39 `ratio=1.0 seed=456` 조건에서 `AttributeError: module 'Conv1d' has no attribute 'forward'`(unsloth 컴파일 캐시 쓰기 실패로 추정) 실패 후, 곧이어 Ablation B 준비 중 `_write_ablation_config`가 작은 yaml 파일 하나 쓰다 `OSError: Disk quota exceeded`로 `run_phase2.py` 전체가 죽으며 파이프라인 중단.
  - **Root cause 확정**: `scripts/run_phase2_ablation.sh`에 `run_phase2_main.sh`와 달리 `HF_HOME`/`MOAI_CHAT_CACHE_DIR` 환경변수가 빠져 있었음. 이 컨테이너는 `$HOME=/workspace`라 HF 캐시가 기본값(`$HOME/.cache/huggingface`)으로 저장되며 `/workspace/.cache`(28G)를 채워 quota 초과. 커밋 `2db2361`에서 수정(Main과 동일 캐시 경로로 통일 — 이미 빌드된 chat_cache도 재사용 가능).
  - 조치 완료: 완료된 조건들의 결과(JSON/CSV/MD, 어댑터 가중치 제외)를 git 백업(`d6c6cd5`), stray 캐시 정리(`/workspace/.cache/huggingface`, `/workspace/.cache/uv` 삭제로 `/workspace` 89G→69G).
  - **재확인 시**: 여전히 정상 진행 중이면 그대로 완료까지 방치. 만약 다시 죽어있으면 `tail -n 50 run_phase2.log`로 원인부터 확인(재발 시 disk quota 재발이 아니라 unsloth_zoo 자체 컴파일 캐시 손상 가능성 있음 — `~/.cache/unsloth_compiled_cache` 등 삭제 후 재시도) 후, `bash scripts/run_phase2_ablation.sh`로 재실행하면 `skip_existing=True`가 기본값이라 완료된 조건은 자동 스킵하고 이어감.
  - 예상 소요 ~55-85시간(RTX 3090 기준), 비용 ~$25-40 추가 예상.
- 오늘(07-18~19) 세션에서 disk quota/캐시 손상 버그 4건 + eval-split 회귀 1건 수정 (커밋 981bd6d~52b318a)
- Table 4.2b(B) cross-dataset CF 신규 구현·푸시 완료 (커밋 335c808) — **아직 pod에서 실행 안 함, 미검증**
- SSH 개인키(`runpod.ppk`) gitignore 보호 완료 (커밋 c570c64)

## 다음 세션 최우선 작업

**[2026-08-09 갱신] 아래 "RunPod은 접었고..." 이하 블록은 07-31 시점 결정 기준이라 최신 상황과 다름 — 실제로는 RunPod로 재전환되어 Phase3 본실행이 진행 중임(위 "현재 상태"의 2026-08-09 항목 참고). 지금 당장 할 일은 없고, optuna/autoresearch가 200/200 도달하면 pod가 텔레그램으로 자동 알려줌. 궁금하면 SSH로 접속해 진행률만 확인하면 됨(명령은 위 항목 참고). 아래는 이전 결정 기록이라 참고용으로만 남겨둠.**

~~RunPod은 접었고, 데스크탑(5060 Ti + 4060 듀얼 GPU)에서 Phase 3를 마무리하는 걸로 확정.~~ pod는 재사용 안 함. *(→ 08-05 이후 번복됨, 위 참고)*

```bash
git pull   # run_phase3.bat 수정 + run_phase3_smoke_gpu0/1.bat 신규 + NEXT_SESSION 갱신 확인
```

0. **[최우선] 로컬 실측 스모크 실행** — 데스크탑에서:
   1) `nvidia-smi`로 GPU0/GPU1이 실제로 5060 Ti/4060 중 뭔지 확인 (스크립트는 0=5060Ti 가정하고 작성됨 — 다르면 스크립트 안 `CUDA_VISIBLE_DEVICES` 값 서로 바꿀 것)
   2) `set ANTHROPIC_API_KEY=sk-ant-...` 설정
   3) cmd 창 2개에서 `run_phase3_smoke_gpu0.bat`, `run_phase3_smoke_gpu1.bat` 동시 실행(전략당 1trial, 총 4개씩)
   4) 완료 후 `type results\phase3_local_smoke_gpu0\results.tsv`, `type results\phase3_local_smoke_gpu1\results.tsv`로 `train_time_min` 실측값 확인 — **이 값으로만** repeats=10 본 실행의 전략당 trial 수를 정할 것(추측 금지, 이전 세션 feedback 기록됨).
1. **[여전히 미확인, 4회 이상 이월됨] GitHub PAT 토큰 무효화 여부** — 2026-07-25 세션 중 `git remote -v` 결과가 노출됐던 건. GitHub → Settings → Developer settings → Personal access tokens에서 살아있는지 확인, 살아있으면 즉시 revoke 후 재발급.
2. **[미확인, 2회째 이월] `ANTHROPIC_API_KEY` 노출됨** — 2026-07-27 밤 세션 중 스크린샷에 pod 터미널의 `export ANTHROPIC_API_KEY=sk-ant-...` 값이 평문으로 그대로 노출됨. Anthropic Console → API Keys에서 해당 키가 살아있는지 확인 후 즉시 revoke·재발급할 것.
3. **하드웨어 물리 확인(미확인)** — 5060 Ti 실제 설치 여부, PSU 용량, 케이스 내부 슬롯 간격/길이 여유. 스모크 실행 전 반드시 확인.

## 알아둘 것

- **(신규 컴퓨터/WSL 설정 시 최우선 확인) `.claude/settings.json`의 `env.PATH` 형식**: 프로젝트 설정의 PATH가 Windows 스타일(`C:\Users\...`를 세미콜론 `;`으로 구분)로 들어가면 WSL bash에서 `ls`/`cat`/`git` 등 전체 명령어가 깨지고 Claude Code 훅(SessionStart 포함) 전부가 조용히 실패함. 사용자 레벨 `~/.claude/settings.json`의 콜론 `:` 구분 + `/mnt/c/...` 형식이 정상 동작 확인됨(2026-07-31) — 프로젝트 설정에서 PATH를 별도 재정의하지 않거나, 재정의한다면 반드시 이 형식(콜론 구분, `/mnt/c/...`)을 따를 것.
- **캐시 디스크 분산 배치는 스크립트마다 개별 설정해야 함**: `HF_HOME=/hf_cache`, `MOAI_CHAT_CACHE_DIR=/workspace/hf_cache/chat_cache`는 공용 함수가 아니라 각 실행 스크립트에 개별적으로 export돼 있음. 2026-07-22 전수 점검 결과 `run_phase2_main.sh`/`run_phase2_ablation.sh`는 있었지만 `run_phase3.sh`/`runpod_phase1.sh`엔 빠져있어서 추가함(커밋 확인은 위 git pull 안내 참고). `runpod_phase1_gemma4.sh`(deprecated, 미수정)와 `.sh` 래퍼 없이 직접 실행하는 `measure_cross_dataset_cf.py`/`run_all.py` 같은 bare python 명령은 **셸에서 직접 export해야 함**(위 최우선 작업 블록에 이미 추가함). 새 실행 스크립트를 추가하거나 복사해서 만들 때 이 export 누락 여부를 반드시 확인할 것 — 누락되면 `$HOME=/workspace`인 이 컨테이너에서는 조용히 `/workspace/.cache`로 캐시가 새며, 몇 시간 뒤 quota 초과로 전체 파이프라인이 죽는 형태로만 드러남(초기 증상은 무관해 보이는 `AttributeError` 등으로 나타날 수 있어 진단이 어려움).
- **`df -h /workspace`는 신뢰 불가**: `/workspace`는 `mfs#eu-cz-1.runpod.net` 네트워크 볼륨이라 `df -h`가 리전 전체 풀 용량(851T)을 보여줌. 실제 이 pod의 quota 확인은 `du -h --max-depth=1 /workspace`로 해야 함(`-s`와 `--max-depth`는 동시 사용 불가, `du: warning`만 뜨고 결과 없음).
- **unsloth 어댑터 호환성 — 검증 완료(2026-07-26)**: `measure_cross_dataset_cf.py`의 `PeftModel.from_pretrained` 어댑터 로드, 72/72 조건 전부 정상 동작 확인됨. 더 이상 미검증 아님.
- **비용 민감**: 이미 $40+ 사용. RunPod 대시보드에서 지출 한도(spending limit) 설정 권장.
- **로컬↔pod 작업 방식**: Claude Code(로컬)는 이 노트북/PC에만 직접 접근 가능. RunPod pod는 SSH 직접 접속 없이 사용자가 웹 터미널에서 명령 실행 후 결과를 복사해서 붙여넣는 방식.
- **`python3`가 venv를 안 가리킬 수 있음(2026-07-26 실증)**: 세션 중간에 venv activate가 풀리면 `python3`가 `/usr/bin/python3`(시스템, 패키지 없음)로 잡히면서 멀쩡한 패키지가 `ModuleNotFoundError`로 보이는 헛다리짚기가 발생함. `which python3`로 `.venv` 경로인지 먼저 확인하거나, 아예 **`uv run python ...`으로 통일**하면 셸 상태와 무관하게 항상 올바른 프로젝트 환경을 씀 — 앞으로는 이걸 기본으로 쓸 것.
- **`uv sync`의 하드링크가 이 네트워크 볼륨(`/workspace`)에서 가끔 깨짐(2026-07-26 실증)**: `nvidia-cusparselt-cu12`, `nvidia-nvshmem-cu12`, `scipy`가 각각 "설치됨"으로 기록돼 있는데 실제 파일은 로드 안 되는 증상이 반복됨(패키지 하나씩 `uv sync --reinstall-package <pkg>`로 개별 복구 가능하지만 계속 재발할 수 있음). 근본적으로는 `UV_LINK_MODE=copy uv sync --reinstall`로 하드링크 대신 실제 복사를 강제하는 게 더 안정적임 — 다음에 또 이런 `ModuleNotFoundError`/`ImportError: lib*.so`류가 나오면 이걸 먼저 시도할 것.
- **bert-score + `transformers==5.5.0` 호환성 버그(2026-07-26 수정)**: `transformers` 5.5.0에서 토크나이저가 새 `TokenizersBackend`로 리팩터링되며 `build_inputs_with_special_tokens`가 빠짐 — `bert_score` 0.3.13이 이 메서드를 직접 호출해 `AttributeError`로 죽음. `src/evaluate/metrics.py`의 `_patch_tokenizers_backend_special_tokens`(커밋 `121af8e`)로 공유 베이스 클래스에 호환 shim을 패치해뒀음. 만약 다른 bert-score 계열 스크립트에서 비슷한 에러가 또 나면, 이미 고쳐져 있는지부터(`git log -- src/evaluate/metrics.py`) 확인.
- **`git` index.lock이 이 저장소(`/mnt/d/...` WSL 마운트)에서 종종 stale하게 남음**: 느린 DrvFs 때문에 `git status`류가 오래 걸리다 index.lock을 남기고, 다음 git 명령이 "Another git process seems to be running"로 막히는 경우가 반복됨. `ps aux | grep git`+`lsof <lockfile>`로 실제 홀더가 없는 걸 확인한 뒤에만 `rm -f .git/index.lock`으로 지울 것(무작정 지우지 말 것).
- **`uv run python -c ...`(옵션 없이)가 `unsloth`를 조용히 지울 수 있음(2026-07-27 실증)**: 마이그레이션 후 `--extra unsloth` 없이 `uv run python`을 한 번만 실행해도 암묵적 재동기화로 unsloth가 빠짐. 증상은 unsloth 관련 에러가 아니라 한참 뒤 Qwen 모델 학습에서 `KeyError: 'images'`로 나타나 진단이 어려움. Phase 2/3 학습 전엔 `uv run python -c "import unsloth"`로 먼저 확인할 것. 복구(`uv sync --extra unsloth`)는 `statsmodels`/`pandas`를 같이 지울 수 있음 — RQ2 재분석 전엔 재설치 필요.
- **tmux 창마다 export한 환경변수가 독립적임**: 한 창에서 `export HF_HOME=...`을 해도 다른 창/새로 연 창에는 안 먹음. 새 tmux 창을 열 때마다 `HF_HOME`/`MOAI_CHAT_CACHE_DIR`/`WANDB_API_KEY`/`ANTHROPIC_API_KEY`를 다시 export해야 함 — 안 하면 캐시가 `/workspace/.cache`로 새거나 wandb가 오프라인/에러로 돎.
- **Phase 3 스모크 결과 파일은 `results/<output_dir>/results.tsv`임 (Phase 2의 `train_result.json` 아님)**: Phase 2와 Phase 3는 결과 저장 방식이 다름 — Phase 3는 `ExperimentTracker`가 trial마다 `results.tsv`에 한 줄씩 append. `train_time_min` 열로 실측 시간 확인.
- 상세 이력은 `docs/RUNPOD_GUIDE.md`와 (로컬 `.claude` 메모리가 있는 컴퓨터에서는) auto-memory `runpod-experiment-status.md` 참고.
