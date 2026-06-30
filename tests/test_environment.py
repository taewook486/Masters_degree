"""get_environment_info() 함수 테스트 (REQ-RI-003).

실험 환경 정보 자동 수집: Python, torch, CUDA, GPU, OS, transformers, peft.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch


def _call_get_env_info(**module_overrides):
    """torch/transformers/peft를 mock한 상태에서 get_environment_info를 호출한다."""
    defaults = {"torch": None, "transformers": None, "peft": None}
    defaults.update(module_overrides)
    with patch.dict("sys.modules", defaults):
        import src.utils.environment as env_mod
        importlib.reload(env_mod)
        return env_mod.get_environment_info()


def test_get_environment_info_returns_dict():
    """get_environment_info는 딕셔너리를 반환해야 한다."""
    info = _call_get_env_info()
    assert isinstance(info, dict)


def test_get_environment_info_has_required_keys():
    """반환된 딕셔너리에 필수 키가 모두 포함되어야 한다."""
    info = _call_get_env_info()
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
    info = _call_get_env_info()
    assert isinstance(info["python_version"], str)
    assert len(info["python_version"]) > 0


def test_get_environment_info_os_info():
    """OS 정보는 비어있지 않은 문자열이어야 한다."""
    info = _call_get_env_info()
    assert isinstance(info["os"], str)
    assert len(info["os"]) > 0


def test_get_environment_info_gpu_not_available_graceful():
    """GPU가 없는 환경에서도 에러 없이 동작해야 한다."""
    info = _call_get_env_info()
    # GPU 관련 값은 문자열 또는 숫자여야 한다
    assert isinstance(info["gpu_name"], str)
    assert isinstance(info["gpu_memory_mb"], (int, float))


def test_get_environment_info_with_cuda_available():
    """CUDA가 사용 가능한 환경을 시뮬레이션한다 (lines 37-43 커버)."""
    # torch 모듈 mock 생성
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.3.0"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.version.cuda = "12.1"
    mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
    # total_mem 속성 (bytes 단위)
    mock_props = MagicMock()
    mock_props.total_mem = 24 * 1024 * 1024 * 1024  # 24GB
    mock_torch.cuda.get_device_properties.return_value = mock_props

    with patch.dict("sys.modules", {"torch": mock_torch}):
        # 모듈을 다시 import하기 위해 캐시 제거
        import importlib
        import src.utils.environment as env_mod

        importlib.reload(env_mod)
        info = env_mod.get_environment_info()

    assert info["torch_version"] == "2.3.0"
    assert info["cuda_version"] == "12.1"
    assert info["gpu_name"] == "NVIDIA RTX 4090"
    assert info["gpu_memory_mb"] > 0


def test_get_environment_info_with_transformers():
    """transformers 패키지가 설치된 환경을 시뮬레이션한다 (line 49 커버)."""
    mock_transformers = MagicMock()
    mock_transformers.__version__ = "4.40.0"

    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        import importlib
        import src.utils.environment as env_mod

        importlib.reload(env_mod)
        info = env_mod.get_environment_info()

    assert info["transformers_version"] == "4.40.0"


def test_get_environment_info_with_peft():
    """peft 패키지가 설치된 환경을 시뮬레이션한다 (line 56 커버)."""
    mock_peft = MagicMock()
    mock_peft.__version__ = "0.11.0"

    with patch.dict("sys.modules", {"peft": mock_peft}):
        import importlib
        import src.utils.environment as env_mod

        importlib.reload(env_mod)
        info = env_mod.get_environment_info()

    assert info["peft_version"] == "0.11.0"


def test_get_environment_info_torch_import_fails():
    """torch가 없을 때 기본값을 사용한다."""
    info = _call_get_env_info(torch=None)
    assert info["torch_version"] == "N/A"
    assert info["cuda_version"] == "N/A"
