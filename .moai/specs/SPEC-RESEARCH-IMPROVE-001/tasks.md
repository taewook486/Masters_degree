# SPEC-RESEARCH-IMPROVE-001 — Task Decomposition

Generated: 2026-06-30
Methodology: TDD (Brownfield Enhancement)
Strategy: Ghost test replacement → real import/call tests → 80%+ coverage per module

## Task Summary

| Task ID | REQ | Target File | Dependencies | Complexity | Status |
|---------|-----|-------------|-------------|-----------|--------|
| T0 | All | All 11 test files | None | Low | completed |
| T1 | REQ-RI-007 | tests/test_logging_config.py | T0 | Low | completed |
| T2 | REQ-RI-003 | tests/test_environment.py | T0 | Low | completed |
| T3 | REQ-RI-008 | tests/test_checkpoint.py | T0 | Low | completed |
| T4 | REQ-RI-001 | tests/test_metrics.py | T1 | Medium | completed |
| T5 | REQ-RI-001,003 | tests/test_evaluate_zero_shot.py | T2,T4 | High | completed |
| T6 | REQ-RI-002 | tests/test_run_all.py | T5 | Medium | completed |
| T7 | REQ-RI-004 | tests/test_strategies.py | T0 | Medium | completed |
| T8 | REQ-RI-005,006 | tests/test_agent.py | T7 | High | completed |
| T9 | REQ-RI-008 | tests/test_loop.py | T8 | High | completed |
| T10 | REQ-RI-006,009 | tests/test_tracker.py | T9 | Medium | completed |
| T11 | REQ-RI-009 | tests/test_catastrophic_forgetting.py | T2 | Medium | completed |

## Execution Order (dependency-resolved)

```
T0 (baseline)
├── T1 → T4 → T5 → T6
├── T2 → T5, T11
├── T3
└── T7 → T8 → T9 → T10
```

## T0: Ghost Test Audit

**Goal**: Diagnose all 11 ghost test files — identify which ones don't actually import source modules.
**Action**: For each test file, check: does it import from `src.*`? Does it call actual functions?
**Files**: tests/test_*.py (all)
**Exit criterion**: Audit report listing ghost vs real tests per file

## T1: REQ-RI-007 — Structured Logging (logging_config.py)

**Goal**: Real tests for `src/utils/logging_config.py`
**Source file**: Already exists at 96% coverage — verify coverage, add edge cases if needed
**REQ**: When logging is configured, structured JSON output and log level filtering must work
**Exit criterion**: test_logging_config.py imports LoggingConfig, calls setup_logging(), verifies JSON format

## T2: REQ-RI-003 — Environment Info (environment.py)

**Goal**: Real tests for `src/utils/environment.py`
**Source file**: Already exists at 72% coverage — add missing 28%
**REQ**: System captures GPU/CPU/memory info at experiment start
**Exit criterion**: test_environment.py imports EnvironmentInfo, calls capture(), verifies hardware fields

## T3: REQ-RI-008 — HPO Checkpoint (checkpoint.py)

**Goal**: Real tests for `src/utils/checkpoint.py`
**Source file**: Already exists at 100% coverage — verify tests actually call source, not just pass
**REQ**: HPO trials save/resume checkpoint state
**Exit criterion**: test_checkpoint.py imports CheckpointManager, tests save/load/resume

## T4: REQ-RI-001 — BERTScore (metrics.py)

**Goal**: Real tests for BERTScore metric computation
**Source file**: src/evaluate/metrics.py (or equivalent)
**REQ**: BERTScore F1 computed, result aggregated with mean/std/median
**Exit criterion**: test_metrics.py imports compute_bert_score(), calls with sample predictions, verifies F1 > 0

## T5: REQ-RI-001,003 — Zero-shot Evaluator (evaluate_zero_shot.py)

**Goal**: Real tests for `src/baseline/evaluate_zero_shot.py`
**Source file**: Currently 0% — full test suite needed
**REQ**: Zero-shot baseline with BERTScore + environment info logging
**Exit criterion**: test_evaluate_zero_shot.py imports evaluate_zero_shot, mocks VLM call, verifies BERTScore + env log

## T6: REQ-RI-002 — Run Aggregation (run_all.py)

**Goal**: Real tests for `src/baseline/run_all.py`
**Source file**: Currently 0% — full test suite needed
**REQ**: Aggregates results from multiple runs with mean/std/median
**Exit criterion**: test_run_all.py imports aggregate_results(), verifies statistical output

## T7: REQ-RI-004 — Duplicate Detection (strategies.py)

**Goal**: Real tests for `src/autoresearch/strategies.py`
**Source file**: Currently 0% — full test suite needed
**REQ**: Duplicate hypothesis detection before LLM call
**Exit criterion**: test_strategies.py imports detect_duplicates(), verifies near-duplicate rejection

## T8: REQ-RI-005,006 — Validation + Exploration Logging (agent.py)

**Goal**: Real tests for `src/autoresearch/agent.py`
**Source file**: Currently 0% — full test suite needed
**REQ**: Validates configs before run, logs exploration decisions
**Exit criterion**: test_agent.py imports AutoresearchAgent, mocks Anthropic API, verifies validation + logging

## T9: REQ-RI-008 — HPO Loop (loop.py)

**Goal**: Real tests for `src/autoresearch/loop.py`
**Source file**: Currently 0% — full test suite needed
**REQ**: HPO trial loop with checkpoint save/resume
**Exit criterion**: test_loop.py imports HPOLoop, verifies checkpoint integration

## T10: REQ-RI-006,009 — Result Tracker (tracker.py)

**Goal**: Real tests for `src/autoresearch/tracker.py`
**Source file**: Currently 0% — full test suite needed
**REQ**: Exploration logs + unified result format
**Exit criterion**: test_tracker.py imports ResultTracker, verifies structured output

## T11: REQ-RI-009 — Format Unification (catastrophic_forgetting.py)

**Goal**: Real tests for `src/evaluate/catastrophic_forgetting.py`
**Source file**: Currently 0% — full test suite needed
**REQ**: Unified result format across all phases
**Exit criterion**: test_catastrophic_forgetting.py imports evaluator, verifies result schema matches baseline format

## Coverage Targets

| Module | Current | Target |
|--------|---------|--------|
| src/utils/logging_config.py | 96% | 96%+ |
| src/utils/environment.py | 72% | 85%+ |
| src/utils/checkpoint.py | 100% | 100% |
| src/baseline/evaluate_zero_shot.py | 0% | 80%+ |
| src/baseline/run_all.py | 0% | 80%+ |
| src/autoresearch/agent.py | 0% | 80%+ |
| src/autoresearch/strategies.py | 0% | 80%+ |
| src/autoresearch/loop.py | 0% | 80%+ |
| src/autoresearch/tracker.py | 0% | 80%+ |
| src/evaluate/catastrophic_forgetting.py | 0% | 80%+ |
| **Overall** | **7%** | **80%+** |
