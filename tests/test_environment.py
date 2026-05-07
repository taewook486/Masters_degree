"""get_environment_info() 함수 테스트 (REQ-RI-003).

실험 환경 정보 자동 수집: Python, torch, CUDA, GPU, OS, transformers, peft.
"""

from __future__ import annotations


def test_get_environment_info_returns_dict():
    """get_environment_info는 딕셔너리를 반환해야 한다."""
    from src.utils.environment import get_environment_info

    info = get_environment_info()
    assert isinstance(info, dict)


def test_get_environment_info_has_required_keys():
    """반환된 딕셔너리에 필수 키가 모두 포함되어야 한다."""
    from src.utils.environment import get_environment_info

    info = get_environment_info()
    required_keys = [
        "python_version",
        "torch_version",
        "cuda_version",
        "gpu_name",
        "gpu_memory_mb",
        "os",
        "transformers_version",
        "peft_version",
    ]
    for key in required_keys:
        assert key in info, f"필수 키 '{key}'가 누락됨"


def test_get_environment_info_python_version():
    """Python 버전은 문자열이어야 한다."""
    from src.utils.environment import get_environment_info

    info = get_environment_info()
    assert isinstance(info["python_version"], str)
    assert len(info["python_version"]) > 0


def test_get_environment_info_os_info():
    """OS 정보는 비어있지 않은 문자열이어야 한다."""
    from src.utils.environment import get_environment_info

    info = get_environment_info()
    assert isinstance(info["os"], str)
    assert len(info["os"]) > 0


def test_get_environment_info_gpu_not_available_graceful():
    """GPU가 없는 환경에서도 에러 없이 동작해야 한다."""
    from src.utils.environment import get_environment_info

    # GPU가 없어도 함수 호출 자체는 실패하면 안 됨
    info = get_environment_info()
    # GPU 관련 값은 문자열 또는 숫자여야 한다
    assert isinstance(info["gpu_name"], str)
    assert isinstance(info["gpu_memory_mb"], (int, float))
