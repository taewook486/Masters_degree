# SPEC-IMPROVE-001: 구현 계획

## SPEC 참조
- SPEC ID: SPEC-IMPROVE-001
- 제목: Medical VQA VLM 프로젝트 코드 품질 및 실험 파이프라인 개선

---

## 마일스톤

### Milestone 1: 긴급 수정 (Primary Goal)

**목표**: Phase 2 진입 전 반드시 수정해야 할 버그 및 품질 문제 해결

#### Task 1.1: REQ-003 - CF 인터페이스 버그 수정
- `src/evaluate/catastrophic_forgetting.py:70` 수정
- `generate_answer(model, processor, image, prompt, config)` -> `generate_answer(model, processor, sample.image, sample.question, config)`
- CF 모듈의 프롬프트 구성 방식을 model_loader의 MEDICAL_PROMPT 기반으로 통일
- 영향 범위: `catastrophic_forgetting.py`만 수정

#### Task 1.2: REQ-005 - Phase 1 집계 버그 수정
- `results/phase1_baseline/phase1_summary.csv` 재생성
- `run_all.py`의 시드 집계 로직 디버깅: `_aggregate_seed_results()` 함수 검증
- SmolVLM-2.2B 누락 시드 결과 확인 및 재실행 필요 여부 판단
- 영향 범위: `run_all.py`, 결과 파일

#### Task 1.3: REQ-002 - MEDICAL_PROMPT 중복 제거
- `src/utils/prompts.py` 생성
- `MEDICAL_PROMPT` 상수를 이 파일로 이동
- `model_loader.py`, `prepare_data.py`에서 import 변경
- 영향 범위: 3개 파일

#### Task 1.4: REQ-001 - metrics.py 단위 테스트
- `tests/test_metrics.py` 생성
- 테스트 케이스 설계:
  - `preprocess_answer`: 공백 처리, 대소문자, 구두점 제거, 다중 공백
  - `_extract_yes_no`: yes/yeah/yep/correct/true, no/nope/nah/incorrect/false, 문장 시작
  - `compute_closed_accuracy`: 정상 케이스, 빈 리스트, 혼합 정답/오답
  - `compute_open_accuracy`: 정확 매칭, 포함 매칭, 불일치
  - `compute_overall_accuracy`: open/closed 혼합, BERTScore 활성화 테스트
- pytest 형식, parametrize 활용

#### Task 1.5: REQ-007 - BERTScore 기본 활성화
- `evaluate_zero_shot.py`의 `compute_overall_accuracy()` 호출에 `compute_bertscore=True` 추가
- 결과 JSON summary에 BERTScore 필드 포함 확인
- `_aggregate_seed_results()`에 BERTScore 관련 집계 추가
- Phase 1 결과 재실행 시 BERTScore 포함 여부 결정

---

### Milestone 2: 연구 기반 강화 (Secondary Goal)

**목표**: 논문 통계 분석 및 실험 추적 기반 구축

#### Task 2.1: REQ-004 - 통계 분석 모듈
- `src/evaluate/statistics.py` 생성
- 함수 구현:
  ```
  run_anova_models(results_dir: str) -> dict
  run_tukey_hsd(results_dir: str) -> dict
  run_paired_ttest(base_results: dict, ft_results: dict) -> dict
  run_wilcoxon(base_results: dict, ft_results: dict) -> dict
  run_kruskal_wallis(strategy_results: dict) -> dict
  compute_cohens_d(group1: list, group2: list) -> float
  ```
- `scipy.stats` 활용: `f_oneway`, `tukey_hsd`, `ttest_rel`, `wilcoxon`, `kruskal`
- 결과를 JSON으로 저장하는 `save_statistical_report()` 함수

#### Task 2.2: REQ-012 - 환경 정보 자동 기록
- `src/utils/environment.py` 생성
- `get_environment_info()` 구현:
  - `sys.version`, `torch.__version__`, `torch.version.cuda`
  - `torch.cuda.get_device_name()`, `torch.cuda.get_device_properties()`
  - `platform.platform()`, `transformers.__version__`, `peft.__version__`
- `evaluate_zero_shot.py`의 `result["metadata"]`에 `"environment"` 필드 추가
- `train_qlora.py`의 `result["metadata"]`에도 동일 적용

#### Task 2.3: REQ-008 - 구조화된 로깅
- `src/utils/logging_config.py` 생성
- `setup_logging(log_dir: str, experiment_name: str, level: str = "INFO")` 구현
- 핸들러: StreamHandler(INFO) + FileHandler(DEBUG)
- 포맷: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
- `run_all.py`, `train_qlora.py`, `run_phase3.py`의 `logging.basicConfig()` 교체

---

### Milestone 3: 논문 준비 (Final Goal)

**목표**: 논문 작성에 필요한 시각화 및 추가 테스트

#### Task 3.1: REQ-009 - 시각화 모듈
- `src/evaluate/visualize.py` 생성
- matplotlib rcParams 논문 스타일 설정
- 함수 구현:
  - `plot_phase1_heatmap(summary_csv: str, output_dir: str)`
  - `plot_closed_vs_open(summary_csv: str, output_dir: str)`
  - `plot_base_vs_finetuned(base_results, ft_results, output_dir)`
  - `plot_hpo_trajectory(tracker_path: str, output_dir: str)`
  - `plot_ablation_curves(results_dir: str, output_dir: str)`
- PNG(300 DPI) + PDF 동시 저장

#### Task 3.2: REQ-010 - 데이터 로더 테스트
- `tests/test_data_loader.py` 생성
- 각 데이터셋 로딩 테스트 (data/ 디렉토리 존재 시에만 실행: `@pytest.mark.skipif`)
- 반환 타입 및 필드 검증

#### Task 3.3: REQ-011 - 재현성 검증 테스트
- `tests/test_reproducibility.py` 생성
- `set_seed()` 후 난수 생성 결과 동일성 검증
- CPU 환경에서의 재현성 보장 테스트

---

### Milestone 4: 선택적 개선 (Optional Goal)

#### Task 4.1: REQ-006 - 통합 Python CLI
- `run_experiment.py` 생성 (argparse 서브커맨드)
- 기존 .bat 파일의 기능 통합
- .bat 파일은 `python run_experiment.py` 호출로 변경

---

## 기술 접근 방식

### 테스트 전략
- pytest + pytest-cov 활용
- `@pytest.mark.parametrize`로 다양한 입력 조합 테스트
- 데이터셋 의존 테스트는 `@pytest.mark.skipif` 적용
- GPU 의존 테스트는 `@pytest.mark.skipif(not torch.cuda.is_available())` 적용

### 코드 품질
- ruff 린터 통과 확인
- 타입 힌트 유지
- 기존 코드 스타일(snake_case, docstring) 일관성 유지

### 위험 요소 및 대응
| 위험 | 대응 |
|------|------|
| BERTScore 계산으로 Phase 1 실행 시간 증가 | roberta-large 모델 캐싱, 배치 처리 활용 |
| SmolVLM 시드 재실행에 GPU 시간 소요 | skip_existing 옵션 활용, 누락분만 실행 |
| 통계 분석 결과가 유의미하지 않을 수 있음 | 연구 설계서에 명시된 대로 결과 보고 (부정적 결과도 유효) |
| BERTScore 의존성(roberta-large)이 추가 VRAM 요구 | CPU에서 BERTScore 계산 or 추론 후 모델 언로드 후 실행 |
