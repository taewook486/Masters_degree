"""GPU VRAM monitoring utilities."""

import torch


def get_vram_usage() -> dict:
    """Get current GPU VRAM usage in MB.

    Returns:
        Dictionary with allocated, reserved, and peak VRAM in MB.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}

    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def reset_peak_stats() -> None:
    """Reset peak VRAM tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def print_vram_status(label: str = "") -> None:
    """Print current VRAM status with optional label."""
    stats = get_vram_usage()
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}VRAM - "
        f"Allocated: {stats['allocated_mb']:.0f}MB | "
        f"Peak: {stats['peak_mb']:.0f}MB | "
        f"Reserved: {stats['reserved_mb']:.0f}MB"
    )
