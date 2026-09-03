# 다음 세션 시작점 (마지막 갱신: 2026-09-03)

> 이 파일이 **유일한 인계 문서**다. 최신 세션 엔트리와 상시 참조 항목만 둔다.
> 파일이 180KB까지 불어나 2026-09-03에 지난 달 로그를 월별 아카이브로 옮겼다.

| 기간 | 파일 |
|---|---|
| 2026-08 | [`next-session/2026-08.md`](next-session/2026-08.md) |
| 2026-07 | [`next-session/2026-07.md`](next-session/2026-07.md) |

> 루트에 있던 `NEXT_SESSION.md`는 2026-09-03에 삭제하고 이 파일로 합쳤다.
> 그 파일의 8/24~8/30 중간판은 git 히스토리에만 있다 — `git log -p -- NEXT_SESSION.md`.

## 🟢 2026-09-03 — 지도교수 회신 대기 + 개발 환경 설정 정리

**논문 내용은 9/1 이후 바뀐 것이 없다.** 지도교수 회신 대기 상태 그대로다.
9/3 세션은 전부 개발 환경 설정 정리였고 원고·제출본은 건드리지 않았다.

### 1. 지적 4건 반영 (8/31, 커밋 `f63f102`)

| # | 지적 사항 | 반영 |
|---|---|---|
| 1 | §2.5 선행연구 빈약 | 3문단 재작성 + Table 2.1 신설, 문헌 6편 추가(참고문헌 16→23건) |
| 2 | WCA 가중치가 임의적 | 원리를 ACR RADPEER로 근거화 + 가중 체계 5종 민감도 검증(Table 4.4a 신설) |
| 3 | 학습 시간 버그 | 표 4.2a에 `train_runtime_sec` 실측 열 추가, 버그 각주를 측정 기준 설명으로 교체 |
| 4 | LLaVA-Med 동일 기준 비교 | `calculate_f1score` recall 재현해 저장된 예측 재채점(GPU 재실행 없음), Table 4.4b + 부록 C |

가중치는 **역순으로 뒤집어도 WCA 증가가 유지**되므로 §4.4.4 결론이 특정 가중치에
의존하지 않음을 확인했다. 다만 평가 표본 부족(diagnosis 3건, measurement 4건,
temporal 0건)은 어떤 가중 체계로도 해소되지 않는다.

부수 정정으로 §3.8.1의 지표 서술을 실제 코드에 맞췄다. BERTScore F1 ≥ 0.7을
"유일한 결정 지표"로 적어 두었으나 실제 판정은 정규화 후 완전 일치 또는 포함이고
BERTScore는 병기 지표다. BERTScore 통과율은 Phase 2 main 36조건 전부에서 99.91%
이상으로 포화돼 변별력이 없다.

### 2. "재빌드 금지"는 폐기됐다 (8/31, 커밋 `be2d06d`)

직전 원고로 재빌드해 제출본과 대조한 결과:

- **본문 텍스트는 완전히 재현된다.** 차이 25건 중 23건이 Word가 채우는 표 목차
  필드였고 나머지 2건은 같은 문장의 문단 분할 차이뿐이었다.
- **유실되는 것은 표 서식뿐이다.** 30개 중 27개가 달라졌는데 전부 열 폭 균등화와
  `tblHeader` 소실이었다.

그래서 `scripts/restore_table_formatting.py`를 만들었다. 재빌드 후 기존 제출본에서
열 폭·셀 폭·머리행 반복을 옮겨온다. 표는 캡션의 표 번호로 짝짓고, 캡션 없는
표(표지·인준서·목차)는 열 수와 행 수로 짝짓는다. 번호가 바뀐 표는 별칭표로 잇되
**별칭은 구본에만** 적용한다(신본에 적용하면 새 표 4.4a가 이름 바뀐 기존 표와 키 충돌).

이식 결과는 국문 34개 중 30개, 영문 33개 중 29개다. 짝이 없는 신규 표 4개
(Table 2.1, 4.4a, C.1, 부록 C 기준 대조표)와 열이 늘어난 표 4.2a(3→5열)는 머리행
반복만 적용됐으므로 **열 폭은 Word에서 조정해야 한다.**

### 3. §3.8.1 코드 경로 제거 (9/1, 커밋 `5865333`)

주관식 판정 기준 문장에 병기돼 있던 `(src/evaluate/metrics.py의
compute_open_accuracy)`를 국·영문 모두에서 뺐다. 기준 서술 자체는 그대로다.

### 4. 설정 정리 (9/3, 커밋 `2a49ece` · `533446f` · `8854428`)

`.claude/settings.local.json`이 `.gitignore`에 등재돼 있는데도 **공개 저장소에
3커밋째 올라가 있었다.** gitignore는 이미 추적이 시작된 파일에는 효력이 없다.
노출된 것은 종료된 팟의 IP(`213.192.2.86`)와 키 파일 경로뿐이고 키·토큰 내용은
아니었다(`*.pem`과 `.env`는 처음부터 추적된 적 없음). `git rm --cached`로 추적만
끊었고 히스토리는 재작성하지 않았다.

`settings.json`에서는 allow 규칙 17건을 정리했다. 삭제·덮어쓰기·업로드가 무인
승인되던 9건(`curl` `find` `rsync` `mv` `sed` `git stash·checkout·switch·merge`)은
`ask`로 옮겼고, 이 저장소에 근거가 없는 8건(`npm`/`npx` 계열, `make`, `moai-adk`)은
지웠다. 하드코딩된 `env.PATH`도 뺐다.

### 9/3 이후 달라진 것 — 다음 세션이 먼저 읽을 것

| 항목 | 이전 | 지금 |
|---|---|---|
| RunPod SSH 키 | `/mnt/d/.../runpod_openssh.pem`을 `/tmp`로 복사 후 `chmod 600` | `~/keys/runpod_openssh.pem` (600) — `ssh -i` 한 줄로 끝 |
| `curl` `find` `rsync` `mv` `sed` | 자동 승인 | **확인 프롬프트 1회** |
| `git checkout·switch·merge·stash` | 자동 승인 | **확인 프롬프트 1회** |
| `.claude/settings.local.json` | git 추적 | 추적 해제 (로컬 전용) |

`/mnt/d`의 pem 원본은 **지우지 않았다.** git에도 없으므로 그것이 유일한 장기
보관본이다. WSL을 초기화하면 `~/keys`는 사라지니 그때 다시 복사하고 `chmod 600`만
주면 된다. `/mnt/d`는 `metadata` 마운트 옵션이 없어 chmod가 반영되지 않고, 그래서
거기서 직접 `ssh -i`를 쓰면 `not a key file`로 거부된다(키 손상이 아니다).

### 현재 상태

| 항목 | 상태 |
|---|---|
| `docs/THESIS_FINAL_v2.0.md` / `_EN.md` | 최신 (9/1 수정 포함) |
| `황태욱_석사학위논문_국문.pdf` / `.docx` | **96쪽**, 8/31 빌드본 |
| `석사학위논문_영문.pdf` / `.docx` | **117쪽**, 8/31 빌드본 |
| 원고 ↔ 제출본 | **§3.8.1 한 문장만큼 어긋남** (9/1 수정 미반영) |
| 지도교수 검토 | 지적 4건 반영분 송부 완료, 회신 대기 |
| `arxiv/` 원고 | 완성·빌드 검증됨, 제출 보류 |

**제출본 재빌드는 다음 수정과 묶어서 한 번에 하기로 했다.** 표 서식 이식이 매번
육안 확인을 요구하므로 한 문장 때문에 그 과정을 반복할 이유가 없다.

### 논문 텍스트를 고쳐야 할 때의 경로

1. `docs/THESIS_FINAL_v2.0.md` / `_EN.md` 수정
2. `python scripts/build_thesis_docx.py` 로 재빌드
   — `--md` 기본값이 국문 고정이라 **영문은 `--lang en`을 반드시 줄 것**
3. `python scripts/restore_table_formatting.py` 로 구 제출본에서 표 서식 이식
4. `powershell.exe -File scripts/word_to_pdf.ps1 -InPath <abs> -OutPath <abs> -SaveDocx`
   로 PDF 재생성 (목차 필드 갱신 + 재페이지네이션)
5. 신규 표·열 늘어난 표의 열 폭은 Word에서 손으로 확인

영문 빌드는 학교 양식 템플릿을 Windows 절대경로(`C:\Users\taewo\Downloads\붙임4_...`)로
하드코딩해 둬서 WSL에서는 실행이 실패한다(`build_thesis_docx.py:44`). 영문 재빌드 전에
이 경로를 먼저 봐야 한다.

pre-commit ruff 게이트는 `SKIP_MOAI_PRECOMMIT=1`로만 우회된다
(`MOAI_SYNC_GATE_BLOCKING=0`은 듣지 않는다).

### 다음에 할 일

1. **지도교수 회신 대기** — 회신 오면 지적사항 반영
2. 회신 반영 시 위 재빌드 경로를 따를 것 (§3.8.1 수정도 이때 함께 반영됨)
3. (선택) `build_thesis_docx.py`에 열 폭 비례 배분·머리행 반복을 직접 이식하면
   `restore_table_formatting.py` 단계를 없앨 수 있음
4. (선택) 영문 템플릿 경로 플랫폼 중립화
5. 지도교수 검토 완료 후에야 arXiv 제출 재개
   (재개 조건: ① 검토 완료 ② 공저자 등재 동의 ③ 학위 취득 — v1은 철회 불가)

### 미해결로 남긴 것

- `arxiv/sections/03-method.tex`에 8/25의 표 3.2 배포본 통일(32,632 / 2,244)과
  8/31 수정분이 **아직 반영되지 않았다.** 제출 재개 시 학위논문과 대조할 것
- `results/phase1_baseline/phase1_robustness.json` 재생성 — 하지 않기로 결정
  (커밋된 산출물을 덮어쓰므로). 대신 §4.1.1에 각주로 사유를 명시했다
- `backup/석사학위논문_국문_사본.docx`(8/25)는 사용자 백업본이다. 9/3에 `backup/`을
  `.gitignore`에 넣어 이제 `git status`에 뜨지 않는다(기존 `*_backup_*/` 규칙은
  맨 이름 `backup/`을 잡지 못했다)
- `/install-github-app`이 만든 `add-claude-github-actions-*` 브랜치가 원격에 있다
- `moai update`를 돌리면 사용자 설정이 조용히 템플릿 기본값으로 초기화된다.
  9/3에 확인된 범위는 `.moai/` 밖까지다 — 업데이트 직후 아래를 확인할 것:

  ```bash
  git diff .moai/config/sections/language.yaml .moai/config/sections/user.yaml .claude/settings.json
  ```

  | 파일 | 되돌아간 값 | 정상값 |
  |---|---|---|
  | `language.yaml` | `conversation_language: en` | `ko` / `Korean` |
  | `user.yaml` | `name: ""` | `taewook` |
  | `.claude/settings.json` | `model: sonnet` | `opus` |

  `model`은 9/3에 키 자체를 삭제해 재발 지점을 없앴다(`settings.local.json`의 `opus`가
  어차피 이기므로 프로젝트 파일의 값은 무의미했다). 같은 요령이 다른 회귀 키에도 쓸
  만하다 — local이 이기는 키는 프로젝트 파일에서 지우면 update가 되돌릴 대상 자체가
  사라진다. `language.yaml`은 여전히 매번 확인이 필요하다

## 알아둘 것

- **커밋이 ruff 부채 265건에 막힐 때 우회는 `SKIP_MOAI_PRECOMMIT=1 git commit ...`** (2026-08-16 실증). 막는 주체는 **git pre-commit 훅**이고, 훅이 저장소 전체 `moai gate`(ruff 포함)를 돌린 뒤 실패 메시지 마지막 줄에 이 변수를 직접 안내한다. **안 통하는 것들**: `--no-verify`(08-13 확인), `MOAI_SYNC_GATE_BLOCKING=0`(08-16 확인 — 3분 넘게 돌다 실패). 08-13에 썼던 `quality.yaml`의 `enforce_quality`를 임시로 내렸다 복구하는 방법도 되지만 설정 파일을 건드리므로 위 변수를 쓰는 게 낫다. **우회 전에 자기가 고친 파일만 `ruff check <파일>`로 통과하는지는 확인할 것** — 265건은 선존재 부채이지 면죄부가 아니다.
- **`env.PATH`는 사용자 레벨(`~/.claude/settings.json`) 한 곳만 소유한다 — 프로젝트 설정에 다시 넣지 말 것.** 2026-09-03에 `.claude/settings.json`의 `env.PATH`를 삭제했다(커밋 `2a49ece`). 지우기 전에는 `/home/taewook`·CUDA v13.1·Nsight 2025.4.0 같은 이 컴퓨터 고유 경로가 박혀 있었고, 그게 **공개 저장소에 커밋된 상태**라 다른 PC에서 그대로 깨졌다. PATH는 셸에서 상속되므로 프로젝트 설정이 재정의할 이유가 없다.
  - 삭제 후 실측(9/3): 프로젝트 `env`에는 `CLAUDE_CODE_*` 5개와 `MOAI_CONFIG_SOURCE`만 남았고, 사용자 레벨이 콜론 `:` 구분으로 PATH를 단독 정의한다. `ls`/`git`/`python3`/`moai` 전부 정상 해석 확인.
  - **원래의 위험은 그대로 유효하다**: 프로젝트 설정에 PATH가 Windows 스타일(`C:\Users\...`를 세미콜론 `;`으로 구분)로 들어가면 WSL bash에서 `ls`/`cat`/`git` 등 **전체 명령어가 깨지고 Claude Code 훅(SessionStart 포함)이 전부 조용히 실패한다**(2026-07-31 실증). 그래서 사용자 레벨에만 두고, 부득이 프로젝트에 넣어야 한다면 반드시 콜론 구분 + `/mnt/c/...` 형식일 것.
  - 참고로 `moai update`는 사용자 설정을 템플릿 기본값으로 되돌리는 이력이 있다(위 9/3 엔트리 「미해결로 남긴 것」 참조). 업데이트 후 명령어가 갑자기 안 잡히면 이 항목을 먼저 의심할 것.
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
