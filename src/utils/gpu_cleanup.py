"""
GPU Cleanup Utilities — VRAM management between pipeline stages.

Implementation notes for Claude Code:
- On T4 (16GB VRAM), TTS and video generation CANNOT run simultaneously
- Preferred: run each stage as subprocess (OS reclaims VRAM on exit)
- Fallback: call cleanup_gpu() between stages if running in same process

Usage:
    from src.utils.gpu_cleanup import cleanup_gpu
    cleanup_gpu()  # Call between TTS and video stages
"""

import gc
import logging

logger = logging.getLogger(__name__)


def cleanup_gpu() -> None:
    """Release VRAM held by the current process via torch cache flush + GC.

    Only use when stages must share a process. Prefer subprocess isolation
    (OS releases VRAM unconditionally on subprocess exit).
    """
    try:
        import torch

        if torch.cuda.is_available():
            before_mb = torch.cuda.memory_allocated() / 1024**2
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            after_mb = torch.cuda.memory_allocated() / 1024**2
            freed_mb = before_mb - after_mb
            logger.info("VRAM cleared: %.0f MB freed (%.0f MB remaining)", freed_mb, after_mb)
    except ImportError:
        logger.debug("torch not available — skipping CUDA cleanup")

    gc.collect()


def vram_usage_mb() -> float:
    """Return currently allocated VRAM in MB, or 0.0 if CUDA unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
    except ImportError:
        pass
    return 0.0


def assert_vram_under(limit_gb: float) -> None:
    """Raise RuntimeError if VRAM usage exceeds *limit_gb* GB."""
    used_gb = vram_usage_mb() / 1024
    if used_gb > limit_gb:
        raise RuntimeError(
            f"VRAM usage {used_gb:.2f} GB exceeds limit {limit_gb:.2f} GB. "
            "Call cleanup_gpu() before starting the next stage."
        )
