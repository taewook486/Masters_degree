"""실험 환경 정보 수집 모듈 (REQ-RI-003).

Python, torch, CUDA, GPU, OS, transformers, peft 버전 정보를
수집하여 결과 JSON의 metadata.environment 필드에 포함한다.
논문 재현성 요건 및 리뷰어 요청 대응에 필수적이다.
"""

from __future__ import annotations

import platform
import sys


def get_environment_info() -> dict:
    """실험 환경 정보를 수집하여 딕셔너리로 반환한다.

    GPU/CUDA가 없는 환경에서도 에러 없이 동작한다.

    Returns:
        Python, torch, CUDA, GPU, OS, transformers, peft 버전 정보.
    """
    info: dict = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": f"{platform.system()} {platform.release()}",
        "torch_version": "N/A",
        "cuda_version": "N/A",
        "gpu_name": "N/A",
        "gpu_memory_mb": 0,
        "transformers_version": "N/A",
        "peft_version": "N/A",
    }

    # torch 버전 및 CUDA/GPU 정보
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            )
    except ImportError:
        pass

    # transformers 버전
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    # peft 버전
    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        pass

    return info
