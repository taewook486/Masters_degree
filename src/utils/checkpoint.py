"""HPO 루프 체크포인트 관리 모듈 (REQ-RI-008).

HPO 루프가 예기치 않게 중단되었을 때(OOM, 전원 차단, 프로세스 종료)
마지막으로 완료된 trial 이후부터 재개할 수 있도록 상태를 저장/복원한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "hpo_checkpoint.json"


def save_checkpoint(checkpoint_dir: str, state: dict) -> None:
    """HPO 루프 상태를 체크포인트 파일에 저장한다.

    Args:
        checkpoint_dir: 체크포인트 파일을 저장할 디렉토리.
        state: 저장할 상태 딕셔너리 (completed_trials, strategy, repeat_id 등).
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint_file = path / _CHECKPOINT_FILENAME
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    logger.debug(f"체크포인트 저장: {checkpoint_file}")


def load_checkpoint(checkpoint_dir: str) -> dict | None:
    """체크포인트 파일에서 HPO 루프 상태를 복원한다.

    Args:
        checkpoint_dir: 체크포인트 파일이 있는 디렉토리.

    Returns:
        저장된 상태 딕셔너리, 또는 체크포인트가 없으면 None.
    """
    checkpoint_file = Path(checkpoint_dir) / _CHECKPOINT_FILENAME
    if not checkpoint_file.exists():
        return None

    with open(checkpoint_file, encoding="utf-8") as f:
        state = json.load(f)

    logger.debug(f"체크포인트 복원: {checkpoint_file}")
    return state
