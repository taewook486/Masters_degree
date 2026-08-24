# 학위논문 수정안 (2026-08-24)

arXiv 프리프린트 조판 과정에서 논문 서술과 실제 산출물을 대조하다 발견한 3건입니다.
전부 실측으로 확인했고, 근거 명령어는 `arxiv/MEASURED_VALUES.md`에 있습니다.

**세 건 모두 실험 결과와 결론에는 영향이 없습니다.** 틀린 것은 서술이거나, 오래된
중간 산출물에서 옮겨 적은 수치입니다.

| # | 사안 | 성격 | 우선순위 |
|---|---|---|---|
| 1 | SLAKE를 이중언어로 서술 | **사실 오류** | 높음 |
| 2 | 표 3.2 문항 수 열의 의미가 불명확 | 오해 소지 | 중간 |
| 3 | 오염 강건성 수치가 재현되지 않음 | **재현 불가** | 높음 |

수정 대상은 `docs/THESIS_FINAL_v2.0.md`(국문)와 `docs/THESIS_FINAL_v2.0_EN.md`(영문)
입니다. docx·PDF는 `scripts/build_thesis_docx.py`로 재생성하면 반영됩니다.

---

## 수정 1. SLAKE는 영어 전용본을 사용했다 (사실 오류)

### 근거

`src/data/download.py:19`가 내려받는 것:

```python
"slake": {
    "hf_id": "mdwiratathya/SLAKE-vqa-english",
    "description": "SLAKE - English Medical VQA (642 images, ~7K QA pairs)",
```

캐시 실측(`data/slake/*/dataset_info.json`): train 4,919 + validation 1,053 + test 1,061
= **7,033건**. 논문이 적은 14,028의 절반이며, 코드의 설명 문구(`~7K`)와 일치합니다.

즉 SLAKE 원 데이터셋은 이중언어가 맞지만, **본 연구가 실제로 사용한 것은 영어 절반**
입니다. 실험은 영어 전용으로 일관되게 수행되었으므로 결과에는 영향이 없습니다.

### 국문 121행

**현재**

> SLAKE [7]는 642개 방사선/CT 영상에 대한 14,028개의 영어-중국어 이중언어 질문-답변 쌍과 의학 지식 그래프(5,232개 지식 triplet)를 결합한 데이터셋이다.

**수정안**

> SLAKE [7]는 642개 방사선/CT 영상에 대한 14,028개의 영어-중국어 이중언어 질문-답변 쌍과 의학 지식 그래프(5,232개 지식 triplet)를 결합한 데이터셋이다. 본 연구는 이 가운데 영어 문항만을 담은 공개 배포본(`mdwiratathya/SLAKE-vqa-english`, 7,033문항)을 사용했다.

원문을 고치지 않고 한 문장을 덧붙이는 방식입니다. 원 데이터셋 설명은 그대로 참인
서술이므로 건드릴 필요가 없고, 무엇을 썼는지만 밝히면 됩니다.

### 국문 191행 (표 3.2)

**현재**

```
| SLAKE | 642 | 14,028 | 영어+중국어 | 방사선/CT | Open+Closed |
```

**수정안**

```
| SLAKE | 642 | 7,033 | 영어 | 방사선/CT | Open+Closed |
```

표는 "본 연구가 사용한 데이터셋"을 정리한 것이므로, 사용하지 않은 중국어 문항을
포함한 수치를 적는 것은 맞지 않습니다.

### 영문 105행

**현재**

> SLAKE [7] combines 14,028 bilingual English-Chinese question-answer pairs over 642 radiology/CT images with a medical knowledge graph of 5,232 knowledge triplets.

**수정안**

> SLAKE [7] combines 14,028 bilingual English-Chinese question-answer pairs over 642 radiology/CT images with a medical knowledge graph of 5,232 knowledge triplets. This study uses the English-only release of that dataset (`mdwiratathya/SLAKE-vqa-english`, 7,033 items).

### 영문 175행 (Table 3.2)

**현재**

```
| SLAKE | 642 | 14,028 | English + Chinese | Radiology / CT | Open + Closed |
```

**수정안**

```
| SLAKE | 642 | 7,033 | English | Radiology / CT | Open + Closed |
```

### 5.3 한계 절에 한 줄 추가 권장

확인해 보니 **5.3에는 언어 범위에 관한 언급이 전혀 없습니다.** 표 3.2를 영어 전용으로
고치고 나면 "왜 중국어는 안 썼는가"가 자연스러운 질문이 되므로, 미리 답을 적어 두는
편이 낫습니다.

> 본 연구는 SLAKE의 영어 배포본만을 사용했으므로, 중국어를 포함한 다국어 의료 VQA
> 로의 일반화는 검증 범위 밖이다.

---

## 수정 2. 표 3.2의 "문항 수"가 무엇을 세는지 밝힌다 (오해 소지)

### 상황

**논문은 이미 실제 평가 건수를 정확히 적고 있습니다.** 국문 290행:

> 평가는 시드 42, 데이터셋별 전체 test split(PathVQA 6,719 / SLAKE 1,061 / VQA-RAD 451문항)에서 수행했다.

이 값은 실측과 정확히 일치하고(결과 파일 12개 전부 `metadata.split=test`, 서브샘플링
없음), 775행의 `pooled 기준(n=8,231)`도 맞습니다. **따라서 이건 허위 서술이 아닙니다.**

문제는 표 3.2의 "문항 수" 열이 원논문 발표 수치인데 그 사실이 표에 안 적혀 있다는
점입니다. 표만 본 독자는 PathVQA를 32,799문항 평가했다고 읽을 수 있습니다.

| 데이터셋 | 표 3.2 (발표 수치) | 배포본 총량 | 실제 평가 (test) |
|---|---:|---:|---:|
| PathVQA | 32,799 | 32,632 | 6,719 |
| SLAKE | 14,028 | 7,033 | 1,061 |
| VQA-RAD | 2,248 | 2,244 | 451 |

### 수정안 — 표 각주 한 줄

국문 표 3.2 아래 (191행 표 직후):

> 문항 수는 데이터셋 배포본 기준이며, 실제 평가는 각 데이터셋의 test split 전량
> (PathVQA 6,719 / SLAKE 1,061 / VQA-RAD 451, 합계 8,231문항)에서 수행했다(4.1).

영문 Table 3.2 아래:

> The QA-pair counts are those of the distributed datasets; all reported accuracies
> are computed on the full test split of each dataset (PathVQA 6,719 / SLAKE 1,061 /
> VQA-RAD 451; 8,231 items in total). See Section 4.1.

### 선택 사항

VQA-RAD는 본문(영문 105행)이 "약 3,500여 개"라고 원논문 수치를 쓰고 표는 2,248을
써서 한 데이터셋에 두 값이 나옵니다. 실측 배포본은 2,244건(train 1,793 + test 451)
이라 표의 2,248과도 4건 차이가 납니다. 심사에서 지적될 만한 크기는 아니지만,
정리하시려면 본문 쪽을 "원논문 기준 약 3,500문항, 공개 배포본 2,244문항"으로
쓰시면 세 값이 정합합니다.

---

## 수정 3. 오염 강건성 수치가 재현되지 않는다 (재현 불가)

### 근거

논문의 `원본 0.3849 → 축소셋 0.3041`은
`results/phase1_baseline/phase1_robustness.json`에서 옮겨온 값인데, **그 파일이
근거로 삼는 per_sample 기록으로 재계산하면 그 값이 나오지 않습니다.**

`scripts/robustness_phase1.py`와 동일한 방식(모델 간 합집합 제거, 표본 가중 pooled)
으로 현행 데이터를 재계산한 결과:

| 모델 | full (재계산) | clean (재계산) | full (저장본) | clean (저장본) |
|---|---:|---:|---:|---:|
| **Qwen3-VL-2B** | **0.3843** | **0.3037** | 0.3849 | 0.3041 |
| Qwen2.5-VL-3B | 0.3637 | 0.2765 | 0.3638 | 0.2765 |
| SmolVLM2-2.2B | 0.3391 | 0.2660 | 0.3389 | 0.2662 |
| Gemma4-E2B | 0.1708 | 0.1076 | 0.1721 | 0.1091 |

제거 건수는 저장본과 정확히 일치하므로(PathVQA 1,020 / SLAKE 233 / VQA-RAD 73)
오염 판정 단계의 문제가 아닙니다. 차이는 **PathVQA 한 곳**에 몰려 있습니다 —
저장본은 `acc = 0.348`인데, 현행 `phase1_baseline`과 재스코어링 이전 사본
(`phase1_baseline_pre_bertscore`) 모두 `0.3472`입니다. SLAKE와 VQA-RAD는 정확히
재현됩니다. 저장본의 PathVQA 값이 재스코어링 이전 상태로 굳어 있는 것으로 보입니다.

### 부수 효과 — 논문 내부 불일치도 해소됩니다

현재 논문은 같은 모델의 pooled 정확도를 두 곳에서 다르게 적고 있습니다.

- 표 4.1a(319행) / 5.1(775행) / 초록(39행): **0.3843**
- 4.1.1 오염 검증(328행): **0.3849**

재계산값을 쓰면 양쪽이 0.3843으로 일치합니다. 즉 이 수정은 재현성 문제와 내부
불일치를 동시에 정리합니다.

### 국문 328행

**현재**

> …그대로 유지**되었다(원본 0.3849 → 축소셋 0.3041, 절대 정확도는 하락하나 순위 불변).

**수정안**

> …그대로 유지**되었다(원본 0.3843 → 축소셋 0.3037, 절대 정확도는 하락하나 순위 불변). 네 모델의 순위는 제거 전후로 자리까지 동일하다(제거 전 0.3843 / 0.3637 / 0.3391 / 0.1708, 제거 후 0.3037 / 0.2765 / 0.2660 / 0.1076).

### 영문 312행

**현재**

> …were preserved** (original 0.3849 → reduced set 0.3041: absolute accuracy falls but the ranking is unchanged).

**수정안**

> …were preserved** (original 0.3843 → reduced set 0.3037: absolute accuracy falls but the ranking is unchanged). The ordering holds position for position before and after removal (0.3843 / 0.3637 / 0.3391 / 0.1708 before, 0.3037 / 0.2765 / 0.2660 / 0.1076 after).

### 결론에 미치는 영향

**없습니다.** 편차는 최대 0.0015이고, 순위는 제거 전후 자리까지 그대로입니다.
"오염에 강건하다"는 4.1.1의 결론은 유지됩니다.

다만 절대 정확도가 약 8%p 떨어지는 것은 3.4에 명시한 해석 기준
("5%p 초과는 결론 재검토")에 걸립니다. 이건 이번 수정으로 생긴 문제가 아니라
원래부터 그랬던 것이므로(0.3849 → 0.3041도 8%p), **순위 보존을 근거로 강건하다고
한 것인지, 절대 정확도 기준으로는 재검토가 필요한 것인지**를 5.3 한계 절에서
한 번 정리해 두시면 심사에서 방어가 쉽습니다.

### 산출물 쪽 조치

`phase1_robustness.json`을 재생성하는 것이 가장 깔끔합니다. 스크립트는 인자 없이
동작합니다(기본값: `--phase1_dir results/phase1_baseline`,
`--contamination_json results/contamination/contamination_analysis.json`, `--seed 42`).

```bash
python -m scripts.robustness_phase1
```

**주의: 이 명령은 기존 `phase1_robustness.json`을 덮어씁니다.** 커밋된 산출물이므로,
돌리기 전에 현재 파일을 따로 남겨 두거나 커밋이 깨끗한 상태에서 실행하십시오.
저는 이번에 실행하지 않았습니다 — 원본 보존이 우선이고, 재생성 여부는 판단이
필요한 사안이라 남겨 두었습니다.

재생성이 여의치 않으면 논문 수치만 위 재계산값으로 고치고, 그 사실을 각주로
남기십시오.

---

## 적용 순서

1. 국·영문 md 3곳씩 수정 (수정 1·2·3)
2. `phase1_robustness.json` 재생성
3. `python scripts/build_thesis_docx.py`로 docx 재생성
   - **주의**: `--md` 기본값이 국문 고정이라 영문 빌드는 `--lang en`을 반드시 붙일 것
4. 표지·페이지 나누기 재확인 (국문 개행 8개 유지, 영문 r2 trHeight 5000)

각 항목이 실제로 반영됐는지는 아래로 확인할 수 있습니다.

```bash
grep -n "14,028\|7,033" docs/THESIS_FINAL_v2.0.md docs/THESIS_FINAL_v2.0_EN.md
grep -n "0.3849\|0.3843" docs/THESIS_FINAL_v2.0.md docs/THESIS_FINAL_v2.0_EN.md
```
