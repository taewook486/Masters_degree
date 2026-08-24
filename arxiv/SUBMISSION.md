# arXiv 제출 체크리스트

## 1. 제출 패키지

`arxiv-submission.tar.gz` (52 KB) — 이 파일 하나를 arXiv 업로드 화면에 올리면 됩니다.

```
main.tex
main.bbl                     ← arXiv는 BibTeX을 돌리지 않으므로 .bbl 포함 필수
sections/00-abstract.tex
sections/01-introduction.tex
sections/02-related-work.tex
sections/03-method.tex
sections/04-results.tex
sections/05-discussion.tex
sections/06-conclusion.tex
sections/A-appendix.tex
figures/phase3_anytime.pdf
```

`refs.bib`는 일부러 뺐습니다 — `.bbl`이 있으면 필요 없고, 둘 다 넣으면 arXiv가 어느 쪽을
쓸지 모호해집니다. 참고문헌을 고치려면 `refs.bib`를 고친 뒤 로컬에서 재빌드해
`main.bbl`을 갱신하고 패키지를 다시 만들어야 합니다.

`.aux` / `.log` / `.out` / `main.pdf`는 넣지 않았습니다. arXiv가 소스에서 직접 컴파일합니다.

## 2. 제출 폼에 넣을 메타데이터

| 항목 | 값 |
|---|---|
| Title | Domain Adaptation of Lightweight Vision-Language Models for Medical Visual Question Answering: QLoRA Fine-Tuning with Autonomous Hyperparameter Optimization |
| Authors | Taewook Hwang, Dugki Min |
| Abstract | `abstract-plain.txt` 내용 복사 (1,882자 — 상한 1,920 이내) |
| Primary category | cs.CV (Computer Vision and Pattern Recognition) |
| Cross-list | cs.CL, cs.LG |
| Comments | 27 pages, 1 figure, 16 tables. Condensed from the author's master's thesis, Konkuk University. |
| Journal ref | (없음) |
| License | 아래 3번 참고 |

초록은 반드시 `abstract-plain.txt`를 쓰십시오. `.tex` 쪽 원문을 그대로 붙이면 LaTeX
명령이 그대로 노출됩니다.

## 3. 라이선스 — 확정: arXiv perpetual non-exclusive

업로드 화면의 라이선스 선택에서 다음을 고르십시오.

> **arXiv.org perpetual, non-exclusive license to distribute this article**

arXiv에 배포권만 주고 저작권은 저자가 그대로 보유하는 형태입니다. 이후 저널이
독점 저작권 양도를 요구해도 마찰이 가장 적습니다.

**선택 후 변경 불가입니다.** CC 계열로 바꾸려면 논문을 새로 올려야 합니다.

## 4. 제출 전 확정해야 할 것

- [x] **지도교수 공저자** — `Taewook Hwang, Dugki Min` 2인으로 확정
- [x] **라이선스** — arXiv perpetual non-exclusive (3번)
- [ ] **로마자 표기 확인** — 저자 `Taewook Hwang`, 지도교수 `Dugki Min`.
      건국대 학과 소개는 성을 앞에 둔 `Min, Dugki`로 표기하므로, 지도교수께서
      본인 논문에 쓰시는 형태가 어느 쪽인지 확인 필요
- [ ] **지도교수 동의** — 공저자 등재와 제출 시점에 대한 사전 동의. v1은 철회 불가
- [ ] **엔도스먼트** (5번)

## 5. 엔도스먼트 절차

지도교수가 공저자가 되면서 **엔도서를 따로 구할 필요가 없어졌습니다.** 공저자가 곧
엔도스 자격자이므로, 아래 4번 단계에서 코드를 전달할 상대가 이미 정해져 있습니다.
다만 자격 요건(6번)은 여전히 확인이 필요합니다.

내 상태를 미리 조회하는 페이지는 없습니다. 제출을 시도하면 시스템이 알려줍니다.

1. arxiv.org 계정 생성 — **`taewook486@konkuk.ac.kr`로 등록** (논문 표기 주소와 일치)
2. 첫 제출 시작 → 엔도스먼트 필요 여부가 화면에 표시됨
3. 필요하면 요청 메일이 자동 발송되고, 그 안에 **6자리 영숫자 코드**가 들어 있음
4. 지도교수께 코드 전달 → 교수님이 `arxiv.org/auth/endorse`에서 입력
5. 엔도서 자격: 해당 분야 논문 일정 편수 이상, **제출 시점 3개월~5년 이내** 논문만 계수
6. 카테고리별로 따로 필요 — cs.CV 하나만 받고 나머지는 교차 등재로 처리

2026년 1월 정책 개정으로 기관 이메일만으로는 자동 승인되지 않습니다. 승인에 며칠
걸릴 수 있으니 원고 완성과 병행하십시오.

## 6. 빌드 재현

```bash
cd arxiv
tectonic -X compile main.tex
```

tectonic은 `~/.local/bin/tectonic`에 설치돼 있습니다. TeX Live 표준 패키지만 쓰므로
arXiv 서버에서도 그대로 컴파일됩니다 (커스텀 `.sty` 없음).

## 7. 검증 기록

| 항목 | 결과 |
|---|---|
| 빌드 | exit 0, 27쪽 |
| 미해결 인용 | 0건 |
| 미해결 참조 | 0건 |
| 참고문헌 | 16건 전부 인용·해석됨 |
| 소수값 대조 | 원고 364개 전건 귀속 (원문 360 + 실측 4) |
| 실측 신규 값 | `MEASURED_VALUES.md`에 명령어 단위 근거 기록 |

**검증 범위의 한계**: 수치 대조기는 소수점 값만 봅니다. 정수(표본 수, trial 수 등)는
표본 8종만 수동 확인했으므로 전수 검증이 아닙니다.

**PDF 폰트 임베딩은 확인하지 않았습니다** — 객체 스트림 압축 때문에 검사가 무의미했고,
소스 제출이라 arXiv가 자체 컴파일하므로 해당 사항이 아닙니다. PDF를 직접 제출하는
경우에만 문제가 됩니다.
