"""setup_logging() 함수 테스트 (REQ-RI-007).

구조화된 로깅 시스템: 파일 + 콘솔 동시 출력, 일관된 포맷.
"""

from __future__ import annotations

import logging
import os
import tempfile

import pytest


def _cleanup_logger(logger: logging.Logger) -> None:
    """Windows에서 FileHandler 잠금 방지를 위해 핸들러를 명시적으로 닫는다."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def test_setup_logging_returns_logger():
    """setup_logging은 Logger 인스턴스를 반환해야 한다."""
    from src.utils.logging_config import setup_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logging(log_dir=tmpdir, experiment_name="test_ret")
        assert isinstance(logger, logging.Logger)
        _cleanup_logger(logger)


def test_setup_logging_creates_log_file():
    """setup_logging은 log_dir에 로그 파일을 생성해야 한다."""
    from src.utils.logging_config import setup_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logging(log_dir=tmpdir, experiment_name="my_experiment")
        log_files = [f for f in os.listdir(tmpdir) if f.endswith(".log")]
        assert len(log_files) >= 1
        assert any("my_experiment" in f for f in log_files)
        _cleanup_logger(logger)


def test_setup_logging_has_console_and_file_handlers():
    """setup_logging은 콘솔과 파일 핸들러를 모두 설정해야 한다."""
    from src.utils.logging_config import setup_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logging(log_dir=tmpdir, experiment_name="test_hdl")
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types
        assert "FileHandler" in handler_types
        _cleanup_logger(logger)


def test_setup_logging_writes_to_file():
    """로그 메시지가 파일에 기록되어야 한다."""
    from src.utils.logging_config import setup_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logging(log_dir=tmpdir, experiment_name="write_test")
        logger.info("테스트 로그 메시지")

        # flush 후 파일 확인
        for h in logger.handlers:
            h.flush()

        log_files = [f for f in os.listdir(tmpdir) if f.endswith(".log")]
        assert len(log_files) >= 1

        log_path = os.path.join(tmpdir, log_files[0])
        with open(log_path, encoding="utf-8") as f:
            content = f.read()
        assert "테스트 로그 메시지" in content
        _cleanup_logger(logger)


def test_setup_logging_creates_log_dir_if_not_exists():
    """log_dir가 존재하지 않으면 자동 생성해야 한다."""
    from src.utils.logging_config import setup_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = os.path.join(tmpdir, "nested", "logs")
        logger = setup_logging(log_dir=nested_dir, experiment_name="nested_test")
        assert os.path.isdir(nested_dir)
        assert isinstance(logger, logging.Logger)
        _cleanup_logger(logger)


def test_setup_logging_format_includes_timestamp():
    """로그 포맷에 타임스탬프가 포함되어야 한다."""
    from src.utils.logging_config import setup_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logging(log_dir=tmpdir, experiment_name="fmt_test")
        logger.info("format check")

        for h in logger.handlers:
            h.flush()

        log_files = [f for f in os.listdir(tmpdir) if f.endswith(".log")]
        log_path = os.path.join(tmpdir, log_files[0])
        with open(log_path, encoding="utf-8") as f:
            content = f.read()
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2}", content)
        _cleanup_logger(logger)
