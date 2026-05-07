"""구조화된 로깅 설정 모듈 (REQ-RI-007).

파일과 콘솔에 동시 출력하는 일관된 로깅 포맷을 제공한다.
장시간 GPU 실험(Phase 1: ~6시간, Phase 3: ~10시간)에서
로그 파일 보존이 필수적이므로, 파일 핸들러를 기본으로 포함한다.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

# 로그 포맷: 타임스탬프 + 레벨 + 모듈명 + 메시지
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: str,
    experiment_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """구조화된 로깅을 설정하고 Logger를 반환한다.

    파일 핸들러와 콘솔(StreamHandler)을 모두 설정한다.
    log_dir가 존재하지 않으면 자동으로 생성한다.

    Args:
        log_dir: 로그 파일을 저장할 디렉토리 경로.
        experiment_name: 실험 이름 (로그 파일명에 사용).
        level: 로깅 레벨 (기본: INFO).

    Returns:
        설정된 Logger 인스턴스.
    """
    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 로그 파일명: {experiment_name}_{timestamp}.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{experiment_name}_{timestamp}.log"

    # 로거 설정
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)

    # 기존 핸들러 제거 (중복 방지) — close() 먼저 호출하여 파일 핸들 누수 방지
    for handler in logger.handlers[:]:
        handler.close()
    logger.handlers.clear()

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
