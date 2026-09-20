# 다음 세션 시작점 (마지막 갱신: 2026-09-20)

> 이 파일이 **유일한 인계 문서**다. 최신 세션 엔트리와 상시 참조 항목만 둔다.
> 파일이 180KB까지 불어나 2026-09-03에 지난 달 로그를 월별 아카이브로 옮겼다.

| 기간 | 파일 |
|---|---|
| 2026-08 | [`next-session/2026-08.md`](next-session/2026-08.md) |
| 2026-07 | [`next-session/2026-07.md`](next-session/2026-07.md) |

> 루트에 있던 `NEXT_SESSION.md`는 2026-09-03에 삭제하고 이 파일로 합쳤다.
> 그 파일의 8/24~8/30 중간판은 git 히스토리에만 있다 — `git log -p -- NEXT_SESSION.md`.

## 🟢 2026-09-20 — 보완 요청 반영·재빌드·회신 작성

**9/18 보완 요청을 반영해 국·영문 제출본을 다시 만들고 회신까지 써 두었다(HEAD `645ef73`).**
반영하지 못한 것은 목차 4.4.4의 점선 리더 하나뿐이고, 제목을 줄이지 않기로 해서 그대로 둔다.
**회신은 아직 보내지 않았다.**

| 절 | 내용 |
|---|---|
| §1~§4 | 저장소·설정 정리 (동기화, lock, language.yaml, 모델) |
| §5~§8 | 보완 요청 반영과 제출본 재빌드 |
| §9~§10 | 회신 작성과 윤문 |
| §11 | 중간보고서 (기한 9/28~10/2) |
| §12 | 다음 세션 시작점 |

### 1. 이 PC가 9/3 시점에 멈춰 있었다

세션 시작 시 HEAD가 `0250560`(9/3)이라, 9/18에 다른 PC에서 올린 커밋 2개
(`549146e` .gitattributes, `d84416a` 9/18 인계)가 빠져 있었다. 로컬이 `0 ahead`라
fast-forward로 맞췄고 잃은 커밋은 없다.

`git diff --stat`이 240파일·121만 줄로 뜨지만 내용 변경이 아니다. 삽입과 삭제가
거의 같은 수(1,213,946 / 1,213,857)인 건 `.gitattributes`의 `* text=auto eol=lf`가
일으킨 CRLF→LF 일괄 정규화 서명이다. 이제 이 PC 작업 트리에도 LF가 적용됐다.

**두 PC를 오가므로 세션 시작 때 원격 격차부터 볼 것.** 뒤처진 줄 모르고 그 위에
커밋하면 같은 파일을 양쪽에서 고친 상태가 된다.

```
git fetch origin master
git rev-list --count --left-right origin/master...HEAD   # "0 0"이어야 정상
```

### 2. lock 파일 gitignore 등재 (`f6b968f`)

statusline이 설정 파일을 쓸 때 남기는 `.claude/settings.local.json.lock`이 기존
`.claude/settings.local.*.json` 패턴에 걸리지 않아 매 세션 미추적 항목으로 떴다.
`.claude/*.lock`으로 등재하고 잔재를 지웠다.

### 3. language.yaml이 또 되돌아가 있었다

`conversation_language`가 `en`으로 초기화돼 있어 `ko` / `Korean`으로 복구했다.
아래 9/3 항목 「미해결로 남긴 것」의 경고가 그대로 재현된 것이다. HEAD 값과 같아져
커밋 대상은 아니었다. `model`처럼 키를 지워 재발 지점을 없앨 수 있는 종류가 아니라
**매 세션 확인이 계속 필요하다.**

### 4. 모델은 이미 Opus 5 / 1M이다

`/model`에서 Opus 5를 고르면 1M 컨텍스트가 기본이라 따로 고를 "1M판"이 없다.
확인은 `.moai/state/context-usage.json`의 `context_window_size`(= 1000000)로 한다.
statusline CW% 게이지도 같은 값을 쓴다.

### 5. 재빌드 선행 조건은 이 PC에서 풀린다

9/18에 멈춘 이유가 회사 PC에 양식과 실행 환경이 없어서였는데, 이 PC에는 둘 다 있다.
`THESIS_TEMPLATE_DIR`을 따로 지정할 필요도 없다 — 스크립트의 기본 경로가 실제 위치와
같다.

```
C:\Users\taewo\Downloads\붙임4_학위별학위논문작성양석(hwp,word)\
  4-5_Degree Paper Writing Form(Korean)_Master & Doctor.docx
  4-6_Degree Paper Writing Form (English_Word)_Master.docx
```

실행은 Windows `uv.exe`로 한다. WSL `python3`에는 python-docx가 없다.

```
/mnt/c/Users/taewo/.local/bin/uv.exe run --no-project --with python-docx python <스크립트>
```

### 6. 꼭지 1은 원고가 아니라 조판 문제였다

교수님이 지적한 6건 중 **4건은 원고에 존재하지 않는다**(`8.3c`, `rank-biserial r`의 `lr`
병기, 표목차 `검정` 누락, `-(H = 27.92` 하이픈). 원고·docx·PDF 어디에도 없어 회신에서
설명할 항목이다.

나머지 2건도 원고는 멀쩡했다. `warmup`은 원고에 `warmup`으로 정확히 적혀 있고, docx로
빌드될 때 갈라진 것이다. **원인은 1twip 차이였다.**

| 항목 | 수치 |
|---|---|
| Word 기본 셀 여백 | 좌우 108tw → 12열이면 2,592tw (본문 가용폭 7,482의 35%) |
| `warmup` 열 글자 자리 | 587tw − 216tw = **371tw** |
| 굵은 9pt 휴먼명조 6글자 필요폭 | **372tw** |

`scripts/fix_submission_typography.py`를 새로 만들어 셀 여백을 28tw로 줄이고 남는 폭을
열별 필요량에 비례 재분배했다. 지금은 457tw 자리에 372tw가 들어간다(여유 +85tw).
`learning_rate`(−29tw)와 `weight_decay`(−104tw)도 원래 모자랐던 것이 함께 해소됐다.
국문 본문 구역의 `pgNumType`도 `decimalFullWidth` → `decimal`로 바꿔 전각 쪽번호를
없앴다 — **학교 양식에서 벗어나는 변경이므로 회신에 반드시 적을 것.**

이 스크립트는 재빌드할 때마다 돌려야 한다. 순서는 아래 §8을 따른다.

### 7. 미해결 1건 — 목차 4.4.4 점선 리더

```
4.4.1 단계 간 비교의 제약 ..................................... 46
4.4.4 Closed–Open 격차: 세 단계에 걸쳐 해소되지 않은 제약 49
4.4.5 자동 탐색의 도달점과 자율 에이전트의 한계 .......... 52
```

전각은 풀려 숫자가 `49`로 붙었지만, 제목이 길어 점선 리더가 들어갈 자리가 없다. 목차
항목 중 이 줄만 그렇다. 제목을 네댓 글자 줄이면 해소되나 **줄이지 않기로 결정**했다.

### 8. 재빌드 절차 (이 순서를 지킬 것)

```
1) build_thesis_docx.py --lang ko                    → rebuilt_ko.docx
   build_thesis_docx.py --lang en --md docs/THESIS_FINAL_v2.0_EN.md
2) restore_table_formatting.py --old <기존 제출본> --new rebuilt --out final
3) fix_submission_typography.py --in final --lang ko|en
4) word_to_pdf.ps1 -InPath <docx> -OutPath <pdf> -SaveDocx
```

빠뜨리기 쉬운 것들:

- **2단계 전에 기존 제출본을 백업할 것.** `restore_table_formatting.py`가 서식을 가져오는
  원본이 기존 제출본이라, 덮어쓰면 이식할 대상이 사라진다.
- **2단계는 기존 제출본의 결함까지 가져온다.** 8/31 제출본의 Table 4.3a 열 폭이 바로
  `war nup`을 만든 그 폭이다. 그래서 3단계가 2단계 뒤에 와야 한다.
- **`--lang en`에는 `--md`를 반드시 붙일 것.** (지금은 기본값이 언어별로 갈리도록
  고쳐져 있으나, 과거 이걸 빠뜨려 본문이 통째로 날아간 적이 있다.)
- 빌드 직후 docx의 목차는 `[여기서 F9를 눌러 목차를 갱신하세요]` 자리표시자다. 4단계의
  `word_to_pdf.ps1`이 목차 필드를 갱신한 뒤 PDF로 내보내므로 따로 F9를 누를 필요는 없다.
- 변환 후 `tasklist.exe | grep -i winword`로 잔존 프로세스를 확인할 것. 남아 있으면
  다음 빌드에서 파일이 잠긴다.

검증은 PDF에서 직접 한다. 구본·신본을 같은 방식으로 훑어 대조하면 확실하다.

```
uv.exe run --no-project --with pypdf python  # 쪽별 extract_text()로 대조
```

**주의**: PDF 추출기가 하이픈 주위에 공백을 넣는다(`LLaVA -Med`, `open -ended`). 정확
일치로 검사하면 멀쩡한 문장이 "없음"으로 나온다 — 느슨한 정규식을 쓸 것.

### 9. 회신 작성 완료 (`a0dc233`) — 아직 보내지 않았다

회신은 **두 벌**이다. 담긴 사실은 같고 형식만 다르다.

| 파일 | 형태 | 용도 |
|---|---|---|
| `reports/보완요청_반영결과_황태욱.pdf` | 보고서 4쪽 | 메일 첨부 |
| `reports/thesis-feedback-response-20260920.md` | 위 PDF의 원본 | 고칠 때 |
| `docs/지도교수_회신_2026-09-20.md` | 편지, 표 없음 | 메일 본문 |

양식은 8/31 회신(`reports/thesis-feedback-response-20260831.md`)을 그대로 따랐고, PDF는
`scripts/md_to_pdf.py`로 만든다. 디렉터리를 통째로 훑는 스크립트라 임시 폴더에 md 하나만
복사해 돌린 뒤 결과를 옮긴다 — `reports/`에 직접 돌리면 8/31 md들까지 PDF로 찍어낸다.

**구성은 요청서를 그대로 따라간다.** 꼭지를 상위에 두고 그 안에 항목을 배치했으며 번호도
요청서 순서(1-1~1-5, 2-1~2-2, 3-1)를 썼다. 꼭지 1의 네 번째 지적은 목차와 표목차 두 곳을
담고 있는데 결과가 갈려 `1-4a`·`1-4b`로 나눴다 — 목차는 있었고 표목차는 없었다. 요청서
항목 8개가 확인 결과 9건이 되는 이유다.

**회신에 반드시 남아 있어야 할 네 가지** (고쳐 쓸 때 빠뜨리지 말 것):

- 재현 안 된 4건(1-1·1-3·1-4b·1-5)은 원고에 없음을 설명하고 재확인을 청한다
- 전각 쪽번호를 일반 숫자로 바꾼 것은 **학교 양식을 벗어나는 변경**임을 고지한다 (1-4a 안에 있다)
- 꼭지 3을 SLAKE 한정 판으로 쓴 이유 — 교수님 원문대로면 5.2절의 8~11%p 격차와 어긋난다
- 목차 4.4.4 점선은 제목 축약을 하지 않기로 해 남겨둔 것임을 밝힌다

**두 판은 같이 고쳐야 한다.** 한쪽만 고치면 첨부와 본문이 어긋난다. 실제로 편지판에서
`0.348`이 빠져 있던 것을 뒤늦게 발견해 맞췄다.

### 10. 회신 윤문 (`645ef73`) — humanize-korean light 경로

두 판 모두 `/humanize-korean`으로 다듬었다. 두 문서 다 어휘형 AI 티는 **0건**이었고
(무생물 주어·`-에 의해` 피동·이중피동·결산 어휘 전부 없음), 실제로 걸린 건 연결어미
(`-며`/`-고`/`-나`/`-어`) 뒤 쉼표 하나였다.

| 문서 | 글자수 | `ending_comma_rate` z |
|---|---|---|
| 편지판 | 2,943 → 2,936 (−7자) | +3.22 → +0.23 |
| 보고서판 | 3,794 → 3,787 (−7자) | +2.87 → −0.52 |

변경은 쉼표 삭제뿐이고 어휘·어순·어미는 건드리지 않았다. `verify_gates.py` 4축 모두 수렴.

알아둘 것:

- **`--run-dir`은 절대 경로로 줄 것.** 상대 경로를 주면 shim이 플러그인 설치 경로 기준으로
  해석해 엉뚱한 곳을 찾는다. 스크립트는 플러그인 루트 `scripts/`에 있고, 레퍼런스는
  `.claude/skills/humanize-korean/references/`에 있다 — 두 경로가 다르다.
- **수치·표·코드블록은 보존 목록으로 명시해 넘길 것.** 회신은 수치 하나가 틀어지면 곤란한
  문서다. 보고서판은 요약표 9행·비교표 U+2212 3개·목차 코드펜스·인용블록 2개를 따로
  못 박았다.
- 쉼표를 전량 제거하지는 않았다. 수치가 연달아 나오는 최장문과 110자가 넘는 대조문에서는
  쉼표가 절 경계를 지탱한다. 다 지우면 리듬이 균일해져 그 자체로 또 티가 난다.
- `_workspace/`는 gitignore 대상이라 커밋에 섞이지 않는다.

### 11. 중간보고서 (2026학년도 2학기)

제출 기한 **9/28(월) ~ 10/2(금)**. gsit@konkuk.ac.kr로 스캔본을 보내거나 행정실(공학관 A동
302호) 방문. **본인 + 지도교수 서명 필수이고 사진 파일은 반려된다 — 스캔만 인정.**

양식이 요구하는 것은 사실상 논문제목(한/영)과 논문구성 계획 둘뿐이다. 기재용 초안은
`docs/중간보고서_2026-09.md`에 있고, 작성된 실물은 git에 없다:

```
C:\Users\taewo\Downloads\학위논문 중간보고서-융합정보기술-황태욱.hwp
D:\project\Masters_degree\학위논문중간보고서-황태욱.pdf
```

학번·서명란이 들어간 행정 서류라 `.gitignore`에 `학위논문*중간보고서*`로 막아 뒀다.
**git이 보관하지 않으므로 다른 PC에서는 보이지 않는다.**

hwp 읽기는 `uv.exe run --no-project --with olefile python`으로 OLE 스트림을 열면 된다.
`PrvText` 스트림이 UTF-16LE 평문이라 가장 간단하다.

### 12. 이어서 시작할 때

```text
✂──── 여기부터 복사 ────✂

ultrathink. 학위논문 회신 발송 후속 진입.
docs/NEXT_SESSION.md의 2026-09-20 항목을 먼저 읽을 것.

전제 검증:
1) git rev-list --count --left-right origin/master...HEAD → "0 0"
2) reports/보완요청_반영결과_황태욱.pdf 4쪽 존재
3) 중간보고서 제출 여부 확인 (기한 9/28~10/2)

실행: 교수님 회신을 받았으면 그 내용부터 원고와 대조할 것.
  · 꼭지 3을 원안으로 되돌리라 하시면 원고 5곳 + 회신 2판을 같이 고친다
  · 목차 4.4.4를 다시 지적하시면 제목 축약을 검토한다
  · 재현 안 된 4건을 다시 짚으시면 PDF 생성 환경을 바꿔 재빌드한다

후속: 심사 일정 확인 · 중간보고서 제출 마무리

✂──── 여기까지 복사 ────✂
```

## 🟢 2026-09-18 — 지도교수 최종 보완 요청 접수 (9/20에 반영 완료)

> 이 항목의 「4. 이어서 시작할 때 붙여 넣을 문장」은 9/20에 소진됐다. 반영 결과는
> 위 9/20 항목을 볼 것. 아래는 요청 접수 당시의 검증 기록으로 보존한다.

**원고·제출본은 아직 수정하지 않았다(HEAD `549146e`).** 회신을 8/31 빌드본
(`황태욱_석사학위논문_국문.pdf`, 커밋 `be2d06d`)과 대조해 반영 범위를 확정한 뒤,
재빌드 환경이 없어서 멈췄다. 템플릿이 있는 PC에서 아래 순서대로 이어간다.

### 1. 지적 사항 검증 결과

PDF 쪽을 이미지로 렌더링해 직접 확인했다(PDF 쪽 = 인쇄 쪽 + 11).

| 꼭지 | 지적 | 판정 | 조치 |
|---|---|---|---|
| 1 | Table 4.3a `war nup` (38쪽) | 실제 문제 — 12열이라 머리글·수치가 전부 줄바꿈됨(`war`/`mup`, `2.00e-`/`4`) | 빌드 후 열 폭·글꼴 조정 |
| 1 | 목차 4.4.4 `제약 4 9` (ii쪽) | 실제 문제 — 아래 전각 쪽 번호가 원인 | 쪽 번호 형식 변경 |
| 1 | Table 4.2d `8.3c` (34쪽) | 재현 안 됨 — PDF에 `-0.34 ~ 8.30` | 수정 없음, 회신에서 설명 |
| 1 | Table 4.3g `lr` (43쪽) | 재현 안 됨 — `rank-biserial r`만 있음 | 수정 없음, 회신에서 설명 |
| 1 | 표목차 4.3g `검정` 누락 (iv쪽) | 재현 안 됨 — `검정` 있음 | 수정 없음, 회신에서 설명 |
| 1 | `-(H = 27.92` 하이픈 (37쪽) | 재현 안 됨 — `유의했다(H = 27.92` | 수정 없음, 회신에서 설명 |
| 2 | §3.2.2 BERTScore 서술 | 실제 문제 | 국문 `:190`, 영문 `:174` |
| 2 | §4.1.1 "BERTScore 재스코어링 이전" | 실제 문제 | 국문 `:363`, 영문 `:347` |
| 3 | LLaVA-Med 비교 결론 누락 | 실제 문제 | 국문초록·5.1·영문 Abstract (영문판은 대응 위치 + `:1095` 국문 초록) |

재현 안 된 4건은 원고·docx·PDF 어디에도 없다. 교수님 쪽 PDF 텍스트 추출에서 생긴
착오로 보이나 추정이다.

**전각 쪽 번호의 원인**: 국문 학교 양식의 본문 구역이 `<w:pgNumType w:fmt="decimalFullWidth">`
로 되어 있어 본문 하단 쪽 번호와 목차 쪽수가 전부 `３４`, `４９`처럼 전각으로 찍힌다.
영문 양식은 일반 숫자다. 빌드에서 `decimal`로 바꾸기로 했다(양식에서 벗어나는
변경이므로 회신에 적을 것).

### 2. 확정된 결정

- 꼭지 2 문구
  - §3.2.2: "Open-ended 응답 채점의 BERTScore 계산" → "Open-ended 응답의 보조 평가 지표인 BERTScore 계산"
  - §4.1.1: "BERTScore 재스코어링 이전 시점" → "주관식 판정 기준 정비 이전 시점"
  - 원고 전체에서 이런 잔재는 이 두 곳뿐이다.
- 꼭지 3 문장 — 교수님 제안에서 **SLAKE 한정을 분명히 한 판**을 쓴다. 교수님 문장대로
  "폐쇄형에서 대등"이라고만 쓰면 5.2절(`:852`)의 PathVQA·VQA-RAD 8~11%p 격차와 어긋난다.

  > 선행 7B 의료 특화 모델(LLaVA-Med)과의 간접 비교에서 경량 2B 모델의 QLoRA 적응은
  > SLAKE 폐쇄형 질의에서 대등한 정확도(85.26% vs 85.34%)에 도달했으나, PathVQA·VQA-RAD
  > 폐쇄형에서는 8~11%p, 동일 토큰 재현율 기준 개방형 질의에서는 10~27%p의 격차를 보여
  > 도메인 적응의 실질적 한계 지점을 규명했다.

  수치 근거: SLAKE 85.26/85.34(`:800`), 폐쇄형 격차 8.09/11.28%p, 개방형 recall 격차
  10.21~27.20%p(부록 C, Table C.1).
- 국문·영문 원고 모두 반영하고 PDF도 둘 다 재빌드한다. 9/2 §3.8.1 수정(`5865333`)도
  이 재빌드에 함께 실린다.

### 3. 재빌드 전에 풀어야 할 것

- **학교 양식 템플릿**: `4-5_Degree Paper Writing Form(Korean)_Master & Doctor.docx`,
  `4-6_Degree Paper Writing Form (English_Word)_Master.docx`가 회사 PC에는 없었다.
  있는 곳을 `THESIS_TEMPLATE_DIR`로 지정한다.
- **파이썬 환경**: `.venv/Scripts/python.exe`는 기반 Python311이 지워져 깨졌다. WSL
  `python3`에는 python-docx·pip·uv가 없다. Windows `uv.exe`로 프로젝트 의존성을 건드리지
  않고 실행한다.

  ```
  uv.exe run --no-project --with python-docx python scripts/build_thesis_docx.py --lang ko
  uv.exe run --no-project --with python-docx python scripts/build_thesis_docx.py --lang en --md docs/THESIS_FINAL_v2.0_EN.md
  ```

- 이후 절차는 아래 9/3 항목의 "논문 텍스트를 고쳐야 할 때의 경로"를 따른다. 표 4.3a는
  서식 이식 후에도 열 폭·글꼴을 손으로 맞춰야 할 가능성이 크다.
- 이 PC에는 poppler가 없어서 PDF 확인은 PowerShell의 `Windows.Data.Pdf`로 쪽을
  PNG로 렌더링해서 했다.

### 4. 이어서 시작할 때 붙여 넣을 문장

```text
ultrathink. 학위논문 교수 최종 피드백(9/18) 반영 진입.
docs/NEXT_SESSION.md의 2026-09-18 항목을 먼저 읽을 것.

전제 검증:
1) git log --oneline -3 → 9/18 인계 커밋 이후 원고 미수정 확인
2) 학교 양식 템플릿 4-5/4-6 docx 위치 확인 → THESIS_TEMPLATE_DIR 지정
3) uv run --no-project --with python-docx python -c "import docx" → 성공

실행: 9/18 항목의 결정대로 국문·영문 원고 수정 후 재빌드
후속: 새 PDF에서 해당 쪽 재확인 후 교수님 회신(재현 안 된 4건 포함) 작성
```

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

### 5. 인계 문서 정리 (9/3, 커밋 `26b82de` · `ac83d84` · `0d44993`)

같은 이름의 인계 문서가 루트와 `docs/`에 각각 있었고 서로 다른 시점을 가리켰다
(루트 9/3판 4KB, `docs/` 8/23판 173KB). 루트본을 지우고 `docs/NEXT_SESSION.md`로
합쳤다. 메모리에 기록된 규칙은 원래부터 `docs/` 쪽을 정본으로 지정하고 있었으므로,
규칙에서 벗어나 있던 건 루트 파일이었다.

합친 뒤 183KB / 1,481행이 되어 읽기 어려워져 월별로 나눴다. 라이브 파일에는
최신 엔트리와 `## 알아둘 것`만 남기고(16KB) 지난 달 로그는
`docs/next-session/YYYY-MM.md`로 옮겼다. 옮길 때 원본을 git HEAD에서 꺼내
6개 구간을 바이트 대조해 누락 0을 확인했다.

`## 알아둘 것`의 `env.PATH` 항목은 권고에서 확정 상태로 고쳐 썼다 — 9/3에 실제로
프로젝트 설정에서 PATH를 지웠으므로 이제 사용자 레벨 `~/.claude/settings.json`이
단독 소유한다. 세미콜론 구분 Windows PATH가 전 명령어와 훅을 조용히 죽이는
7/31 실증은 여전히 유효하므로 하위 항목으로 보존했다.

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
  - **훅은 `moai gate`를 인자 없이 부른다 — 스테이징 범위로 좁혀지지 않는다**(2026-09-20 실증). `--staged`/`--file` 없이 호출되므로 `.gitignore` 두 줄짜리 커밋도 `results/` 아래까지 저장소 전체를 훑고, 이 저장소에서는 5분 가까이 걸린다. 느리다고 고장 난 게 아니니 죽이기 전에 `ps -o etime,cmd -p $(pgrep -f "moai gate")`로 돌고 있는지부터 볼 것. 문서·설정만 건드린 커밋이면 위 변수로 우회하는 게 합리적이다.
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
