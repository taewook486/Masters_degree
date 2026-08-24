# 다음 세션 시작점 (2026-08-24 기준)

## 이번 세션에 한 일

학위논문 영문판을 arXiv 프리프린트로 조판했습니다. `arxiv/`에 자족적 LaTeX 소스가
들어 있고, 빌드는 통과했습니다(27쪽, 미해결 인용·참조 0건).

조판보다 중요한 건 **논문 서술과 실제 데이터가 어긋나는 3건을 실측으로 잡아낸 것**입니다.
아래 "지도교수 검토 때 올릴 것"을 보십시오.

## 현재 상태

| 항목 | 상태 |
|---|---|
| `arxiv/` 원고 | 완성, 빌드 검증됨 |
| arXiv 계정 | 생성 완료 (`taewook.hwang`, `taewook486@konkuk.ac.kr`) |
| 제출 | **보류** — Start Submission 단계에서 중단 |

### 왜 보류했나

엔도스먼트 미보유로 화면이 막힌 시점에 멈췄지만, 진짜 이유는 순서입니다.
**지도교수가 논문을 아직 검토하지 않았습니다.** `docs/THESIS_FINAL_v2.0_EN.md`
머리말도 남은 작업으로 `supervisor review`를 명시하고 있습니다.
arXiv v1은 철회가 불가능하므로 검토 전 공개는 되돌릴 수 없습니다.

**재개 조건**: ① 지도교수 논문 검토 완료 ② 공저자 등재 동의 ③ 학위 취득

## 지도교수 검토 때 올릴 것 (arXiv와 무관, 학위논문 자체 문제)

근거는 전부 `arxiv/MEASURED_VALUES.md`에 명령어 단위로 기록돼 있습니다.

1. **SLAKE를 이중언어로 서술했지만 실제로는 영어 전용본을 썼습니다.**
   논문 표 3.2는 `14,028 / English + Chinese`인데, `src/data/download.py:19`는
   `mdwiratathya/SLAKE-vqa-english`를 받고 캐시 실측은 7,033건입니다.
   실험 자체는 일관되므로 결과에 영향 없고, 틀린 건 서술뿐입니다. **가장 중요합니다.**

2. **데이터셋 규모가 발표/배포/실평가 3중값입니다.**

   | 데이터셋 | 논문 표(발표) | 배포본 총량 | 실제 평가(test 전량) |
   |---|---:|---:|---:|
   | PathVQA | 32,799 | 32,632 | 6,719 |
   | SLAKE | 14,028 | 7,033 | 1,061 |
   | VQA-RAD | 2,248 | 2,244 | 451 |

3. **`results/phase1_baseline/phase1_robustness.json`이 stale이라
   논문의 `0.3849 → 0.3041`이 재현되지 않습니다.**
   현행 per_sample로 재계산한 정답은 `0.3843 → 0.3037`이고,
   이러면 논문 내부의 0.3843 vs 0.3849 불일치도 동시에 해소됩니다.
   원인은 PathVQA 한 곳(저장본 0.348 vs 실제 0.3472). 순위는 제거 전후 그대로 보존,
   편차 최대 0.0015라 **결론은 하나도 바뀌지 않습니다.**

**학위논문 수정본은 아직 만들지 않았습니다.** "일단 arXiv판만" 하기로 결정했기 때문입니다.
필요하면 국·영문 해당 절 수정안을 만들면 됩니다.

## 재개 시 할 일

`arxiv/SUBMISSION.md` 체크리스트를 따라가면 됩니다. 남은 미확정 항목:

- [ ] 지도교수 로마자 표기 확인 — 현재 `Dugki Min`(이름-성). 건국대 공식 표기는 `Min, Dugki`
- [ ] 지도교수 공저자 등재 동의
- [ ] 엔도스먼트 — 공저자인 민 교수님이 해주시면 되나,
      **최근 3개월~5년 내 cs 계열 arXiv 제출 이력**이 있어야 자격 성립.
      없으면 학과의 다른 활동 저자를 찾아야 함

확정된 것: 저자 `Taewook Hwang, Dugki Min` / 라이선스 `arXiv perpetual, non-exclusive`
/ 분류 `cs.CV` (교차 `cs.CL`, `cs.LG`) / 분량 27쪽

## 빌드 방법

```bash
cd arxiv
tectonic -X compile main.tex     # ~/.local/bin/tectonic (sudo 불필요)
tar czf arxiv-submission.tar.gz main.tex main.bbl sections/*.tex figures/phase3_anytime.pdf
```

TeX Live 표준 패키지만 사용하므로 arXiv 서버에서도 그대로 컴파일됩니다(커스텀 `.sty` 없음).
PDF·중간 산출물·패키지는 `.gitignore` 처리했습니다.

## 함정 기록

- **arXiv 비밀번호는 영문·숫자·밑줄만** 허용합니다. 특수문자 전면 금지라
  관리자 자동생성 암호는 거의 다 거부됩니다.
- **카테고리 드롭다운은 코드가 아니라 전체 이름으로 표시**됩니다.
  `cs.CV` → `Computer Vision and Pattern Recognition`.
- **수치 대조기는 소수점 값만 봅니다.** 정수(표본 수·trial 수)는 안 걸리므로
  위 3건은 전부 수동으로 세어서 잡은 것입니다.
- `moai update`가 `language.yaml`을 ko→en으로 되돌립니다. 이번에도 발생해 복원했습니다.
