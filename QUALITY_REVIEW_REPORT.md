# 코드 리뷰 리포트 — SPEC-RESEARCH-IMPROVE-001

## 종합 평가

**최종 판정: CRITICAL** (즉시 수정 필요 사항 3건)

---

## Critical Issues (반드시 수정)

### 1. [Breaking Change] src/evaluate/catastrophic_forgetting.py: 결과 스키마 변경으로 인한 호환성 파괴
**파일:줄번호**: `catastrophic_forgetting.py:189-202, 211`

**상세 설명**:
- REQ-RI-009에서 결과 구조를 다음과 같이 변경:
  - 기존: `{"model_name": "...", "catastrophic_forgetting": {degradation rates}}`
  - 신규: `{"metadata": {...}, "summary": {degradation rates}, "base_vqav2": {...}, "finetuned_vqav2": {...}}`
- 줄 211에서 `deg = result["summary"]["degradation_overall_accuracy_pct"]` 로 접근
- 이 변경의 **영향받는 모든 호출 지점을 검토하지 않음**
- 기존 결과 파일 또는 다른 스크립트에서 이전 스키마로 접근하면 KeyError 발생

**해결 방안**:
```python
# 마이그레이션 함수 추가
def normalize_cf_result(result: dict) -> dict:
    """기존 포맷을 새 포맷으로 마이그레이션"""
    if "summary" in result:
        return result  # 이미 new format
    # Old format migration
    if "catastrophic_forgetting" in result:
        return {
            "metadata": {"model_name": result.get("model_name")},
            "summary": result["catastrophic_forgetting"],
            "base_vqav2": result.get("base_vqav2", {}),
            "finetuned_vqav2": result.get("finetuned_vqav2", {})
        }
    return result

# 호출처에서:
result = load_cf_result(path)
result = normalize_cf_result(result)
deg = result["summary"]["degradation_overall_accuracy_pct"]
```

**영향도**: 높음 (Phase 2 완료 후 기존 결과를 재처리할 때 역호환성 필수)

---

### 2. [Logic Error] src/autoresearch/strategies.py:_is_duplicate() — Floating-point 비교 tolerance 문제
**파일:줄번호**: `strategies.py:281-283`

**상세 설명**:
```python
lr_match = abs(config.get("learning_rate", 0) - existing.get("learning_rate", 0)) < 1e-6
wu_match = abs(config.get("warmup_ratio", 0) - existing.get("warmup_ratio", 0)) < 1e-4
wd_match = abs(config.get("weight_decay", 0) - existing.get("weight_decay", 0)) < 1e-4
```

반면 agent.py의 반올림:
```python
"learning_rate": round(lr, 6)  # 6자리 반올림
"warmup_ratio": round(wu, 4)   # 4자리 반올림
"weight_decay": round(wd, 4)   # 4자리 반올림
```

**문제점**: 
- `round(x, 6)`으로 반올림된 값도 부동소수점 연산 오차로 인해 1e-6보다 클 수 있음
- 실제로는 다른 값인데도 의도치 않게 "중복"으로 간주되거나 반대의 경우 발생 가능
- 예: `0.0001` vs `0.00010001` → 부동소수점 표현 오차로 불일치로 검사될 수 있음

**현재 위험**: `_MAX_DUPLICATE_RETRIES = 3`으로 최대 3회 재시도하지만, 계속 실패하면 중복으로 진행

**해결 방안**:
```python
def _is_duplicate(self, config: dict, history: list[TrialResult]) -> bool:
    """이전 완료 trial과 동일한 하이퍼파라미터 설정인지 확인한다."""
    compare_keys = [
        "lora_rank", "lora_alpha", "batch_size",
        "grad_accum_steps", "lora_targets", "max_steps",
    ]
    for trial in history:
        if trial.status != "completed":
            continue
        existing = config_to_dict(trial)
        match = True
        for key in compare_keys:
            if config.get(key) != existing.get(key):
                match = False
                break
        if match:
            # 반올림값 기준으로 비교 (agent 로직과 일치)
            lr_match = (
                round(config.get("learning_rate", 0), 6) ==
                round(existing.get("learning_rate", 0), 6)
            )
            wu_match = (
                round(config.get("warmup_ratio", 0), 4) ==
                round(existing.get("warmup_ratio", 0), 4)
            )
            wd_match = (
                round(config.get("weight_decay", 0), 4) ==
                round(existing.get("weight_decay", 0), 4)
            )
            if lr_match and wu_match and wd_match:
                return True
    return False
```

**영향도**: 중간 (중복 감지 실패로 동일 설정 재실행 가능성)

---

### 3. [Race Condition] src/autoresearch/loop.py — 체크포인트 저장 타이밍
**파일:줄번호**: `loop.py:284-294`

**상세 설명**:
```python
tracker.append(trial)
results.append(trial)

# REQ-RI-008: 각 trial 후 체크포인트 저장
completed_so_far = sum(1 for t in tracker.load_by_strategy(strategy_name, repeat_id) 
                        if t.status == "completed")
save_checkpoint(checkpoint_dir, {
    "strategy": strategy_name,
    "repeat_id": repeat_id,
    "completed_trials": completed_so_far,
    "last_trial_id": trial.trial_id,
})
```

**문제점**:
1. `tracker.append()`으로 파일에 쓰고 > 체크포인트 저장 사이의 시간 윈도우에서 프로세스 크래시 가능
2. 체크포인트의 `completed_trials` 값이 tracker 파일과 불일치할 수 있음
3. 재시작 시 잘못된 start_index로 인해 trial이 중복 실행되거나 건너뛰어질 수 있음

**시나리오**:
```
Trial 10 완료 → tracker.append() 성공
completed_so_far = 10 계산
체크포인트 저장 직전 → OOM 크래시 (체크포인트는 저장 안 됨)
재시작 시: tracker에서 completed=10, 체크포인트에서 completed=9
start_index 결정 로직에서 불일치 발생
```

**해결 방안**:
```python
tracker.append(trial)
results.append(trial)

# trial.status가 "completed"인 경우에만 체크포인트 저장
if trial.status == "completed":
    # 필요하면 tracker에서 최신값 다시 읽기
    all_trials = tracker.load_by_strategy(strategy_name, repeat_id)
    completed_so_far = sum(1 for t in all_trials if t.status == "completed")
    save_checkpoint(checkpoint_dir, {
        "strategy": strategy_name,
        "repeat_id": repeat_id,
        "completed_trials": completed_so_far,
        "last_trial_id": trial.trial_id,
    })
```

또는 tracker에 체크포인트 저장을 통합:
```python
# ExperimentTracker에 메서드 추가
def save_state(self, strategy: str, repeat_id: int) -> dict:
    """현재 상태를 반환 (체크포인트 저장에 사용)"""
    trials = self.load_by_strategy(strategy, repeat_id)
    completed = [t for t in trials if t.status == "completed"]
    return {
        "strategy": strategy,
        "repeat_id": repeat_id,
        "completed_trials": len(completed),
        "last_trial_id": max((t.trial_id for t in trials), default=0),
    }
```

**영향도**: 중간 (오래 실행되는 HPO 루프에서 크래시 후 재개 시 불일치)

---

## Warnings (수정 권장)

### 4. [API Design] src/utils/logging_config.py:50 — 기존 핸들러 제거의 부작용
**파일:줄번호**: `logging_config.py:46-50`

**상세 설명**:
```python
logger = logging.getLogger(experiment_name)
logger.setLevel(level)
logger.handlers.clear()  # 모든 핸들러 제거
```

**문제점**:
- `logging.getLogger()`는 이름별 단일 인스턴스(싱글톤) 반환
- 실험 중에 `setup_logging()`을 여러 번 호출하면 (예: 병렬 처리):
  - 첫 번째: 파일 핸들러 추가
  - 두 번째: **기존 파일 핸들러 제거** → 이전 파일이 닫히지 않고 열린 상태
  - 메모리/파일 디스크립터 누수
- Windows에서 특히 파일 잠금 문제 발생 가능

**해결 방안**:
```python
def setup_logging(
    log_dir: str,
    experiment_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """구조화된 로깅을 설정하고 Logger를 반환한다."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{experiment_name}_{timestamp}.log"
    
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)
    
    # 기존 핸들러 제거 시 먼저 닫기
    for handler in logger.handlers[:]:  # 복사본으로 반복
        handler.close()
        logger.removeHandler(handler)
    
    # propagate 끄기 (상위 logger에 전파 방지)
    logger.propagate = False
    
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
```

**영향도**: 낮음 (현재는 run_all.py, evaluate_zero_shot.py에서만 단일 호출)

---

### 5. [Error Handling] src/utils/environment.py:34-44 — ImportError 무시로 인한 의도 불명확
**파일:줄번호**: `environment.py:34-44`

**상세 설명**:
```python
try:
    import torch
    info["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        # ...
except ImportError:
    pass  # 조용히 무시
```

**문제점**:
- torch 없이 실행되면 당연히 성공하지만, 이 후 코드(loop.py 등)에서 `torch.cuda.OutOfMemoryError` 처리는 torch 필수
- 결과에 `torch_version: "N/A"` 로 기록되는데, torch가 설치되지 않은 건지 단순 import 실패인지 구분 불가
- **재현성 문제**: 리뷰어가 환경 정보를 보고 이상하게 느낄 수 있음

**해결 방안**:
```python
def get_environment_info() -> dict:
    """실험 환경 정보를 수집하여 딕셔너리로 반환한다."""
    info: dict = {...}

    # torch는 필수 (이미 import되어 있다고 가정)
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
            )
    except ImportError:
        logger.warning(
            "torch not found - GPU info unavailable. "
            "torch should be installed for this experiment."
        )

    # transformers, peft는 선택적
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        pass

    return info
```

**영향도**: 낮음 (현재 코드는 torch가 필수이므로 영향 없음)

---

### 6. [Data Migration] src/autoresearch/tracker.py:113-116 — 구버전 TSV 호환성
**파일:줄번호**: `tracker.py:113-116`

**상세 설명**:
```python
status=row["status"],
notes=row.get("notes", ""),
agent_reasoning=row.get("agent_reasoning", ""),
phase=row.get("phase", ""),
temperature=float(row.get("temperature", 0.0)),
```

**문제점**:
- 새 필드(phase, temperature)를 추가했지만, 기존 TSV에는 없을 수 있음
- `row.get()` 사용으로 안전하게 처리되지만, `phase` 기본값이 `""`인 것이 의미적으로 모호
- 향후 phase를 기반으로 필터링할 때 기존 데이터와 섞이면 버그 가능성

**해결 방안**:
```python
# TrialResult dataclass 기본값 검토
@dataclass
class TrialResult:
    # ...
    phase: str = "unknown"  # "" 대신 "unknown" 사용
    temperature: float = 0.0

# 또는 마이그레이션 함수
def load_all(self) -> list[TrialResult]:
    """Load all trial results from TSV."""
    if not self.path.exists():
        return []

    results = []
    with open(self.path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            phase = row.get("phase", "").strip()
            if not phase:  # 기존 데이터 마이그레이션
                phase = "unknown"
            
            trial = TrialResult(
                # ... 다른 필드 ...
                phase=phase,
                temperature=float(row.get("temperature", 0.0)) or 0.0,
            )
            results.append(trial)
    return results
```

**영향도**: 낮음 (현재는 새 코드로 작성되지만, 기존 TSV와의 호환성 문제)

---

### 7. [Consistency] src/autoresearch/agent.py:154 — 프롬프트와 실제 코드 불일치
**파일:줄번호**: `agent.py:151-155`

**상세 설명**:
```python
f"Response Format\n"
f"Respond with ONLY a valid JSON object containing these keys: "
f"lora_rank, lora_alpha, learning_rate, batch_size, grad_accum_steps, "
f"warmup_ratio, weight_decay, lora_targets, epochs"  # ← epochs는 더 이상 사용 안 함
```

실제 코드는 `max_steps` 기반이지만, 프롬프트는 `epochs`를 언급.

**문제점**:
- agent가 `epochs` 키를 반환할 수 있음
- `_validate_config()`에서 마이그레이션 처리하지만, 혼동 유발 가능성

**해결 방안**:
```python
f"Response Format\n"
f"Respond with ONLY a valid JSON object containing these keys: "
f"lora_rank, lora_alpha, learning_rate, batch_size, grad_accum_steps, "
f"warmup_ratio, weight_decay, lora_targets, max_steps"
```

**영향도**: 매우 낮음 (마이그레이션 로직이 이미 있음)

---

## Suggestions (선택적 개선)

### 8. [Performance] src/autoresearch/loop.py:288 — 완료 trial 수 재계산의 비효율
**파일:줄번호**: `loop.py:288`

```python
completed_so_far = sum(1 for t in tracker.load_by_strategy(strategy_name, repeat_id) 
                        if t.status == "completed")
```

각 trial 후 모든 trial을 다시 읽고 완료 상태를 확인하는 O(n) 작업.

**제안**: 간단한 카운터 유지:
```python
if trial.status == "completed":
    completed_count = tracker.load_by_strategy(strategy_name, repeat_id)
    completed_so_far = sum(1 for t in completed_count if t.status == "completed")
    save_checkpoint(...)
```

또는:
```python
completed_so_far = 0
for i in range(start_index, max_trials):
    # ... run trial ...
    if trial.status == "completed":
        completed_so_far += 1
    save_checkpoint(checkpoint_dir, {
        "strategy": strategy_name,
        "repeat_id": repeat_id,
        "completed_trials": completed_so_far,
        "last_trial_id": trial.trial_id,
    })
```

**영향도**: 매우 낮음 (max_trials ≤ 40이므로 성능상 미미)

---

### 9. [Documentation] src/utils/checkpoint.py — 상태 스키마 명확화
**파일:줄번호**: `checkpoint.py:19-24`

**제안**:
```python
def save_checkpoint(checkpoint_dir: str, state: dict) -> None:
    """HPO 루프 상태를 체크포인트 파일에 저장한다.

    Args:
        checkpoint_dir: 체크포인트 파일을 저장할 디렉토리.
        state: 저장할 상태 딕셔너리 (권장 스키마):
            {
                "strategy": str,        # 전략 이름 (manual, random, optuna, autoresearch)
                "repeat_id": int,       # 반복 인덱스
                "completed_trials": int,  # 완료한 trial 수
                "last_trial_id": int,   # 마지막 trial ID
            }
    """
```

**영향도**: 매우 낮음 (문서화 개선)

---

### 10. [Code Style] 모든 신규 파일 — PEP 8 마지막 개행
각 파일 마지막에 개행 문자 추가 (PEP 8 컨벤션)

**영향도**: 매우 낮음 (스타일)

---

## 종합 평가

### Security: PASS
- 경로 주입, 명령 실행 등 보안 취약점 없음
- JSON `ensure_ascii=False`: 한국어 메시지 저장용으로 안전
- 파일 I/O: 경로 생성 및 검증 안전

### Performance: PASS
- 심각한 성능 문제 없음
- `load_by_strategy()` O(n) 호출이 매 iteration에 발생하지만, max_trials ≤ 40이므로 무시할 수준

### Quality: WARN
- 신규 코드의 전반적 품질 양호
- 하지만 breaking change와 logic error는 반드시 해결 필요

### Breaking Changes: CRITICAL
1. `catastrophic_forgetting.py` 결과 스키마 변경
2. `TrialResult` 필드 추가 (phase, temperature)

### TRUST 5 점수

| 항목 | 점수 | 설명 |
|------|------|------|
| Testable | 3/5 | 테스트 코드 미포함 |
| Readable | 4/5 | 명확한 명명, 다만 비교 로직 복잡 |
| Unified | 4/5 | 스타일 일관성 있음 |
| Secured | 4/5 | 보안 취약점 미발견 |
| Trackable | 4/5 | REQ 참조 명확 |

**종합 TRUST 5**: 3.8/5 (Warning 수준)

---

## 행동 계획

### 필수 수정 (Blocker)
1. catastrophic_forgetting 스키마 변경의 모든 호출 지점 확인 및 마이그레이션 로직 추가
2. _is_duplicate() 부동소수점 tolerance 수정 (round 기반 비교로 변경)
3. 체크포인트 race condition 개선 (failed trial에는 저장하지 않기)

### 권장 수정
4-7. Logging, Error Handling, Data Migration, Consistency 개선

### 선택적 개선
8-10. Performance, Documentation, Code Style

---

## 최종 결론

**현재 상태**: Critical issues 3건으로 인해 **수정 후 재검토 필수**

**다음 단계**:
1. Critical 3건 수정
2. 테스트 추가 (특히 checkpoint recovery와 duplicate detection)
3. 전체 HPO 루프 end-to-end 통합 테스트
