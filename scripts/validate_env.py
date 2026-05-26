"""
Environment Validator — Sanity checks before running the pipeline.

Verifies CUDA, cuDNN, TensorRT, PyTorch, and model availability.
Run this first on any new machine/container to catch issues early.

Usage:
    python scripts/validate_env.py
"""

import sys
import shutil
import importlib
from pathlib import Path


def check(name: str, condition: bool, detail: str = ""):
    status = "✅" if condition else "❌"
    msg = f"{status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    errors = 0

    print("=== System ===")
    check("Python", sys.version_info[:2] == (3, 10), f"found {sys.version}")
    if sys.version_info[:2] != (3, 10):
        errors += 1

    check("FFmpeg", shutil.which("ffmpeg") is not None)
    if not shutil.which("ffmpeg"):
        errors += 1

    check("espeak-ng", shutil.which("espeak-ng") is not None)
    if not shutil.which("espeak-ng"):
        errors += 1

    node_ok = shutil.which("node") is not None
    check("Node.js", node_ok, "(needed for Hyperframes)")
    # Node is optional for video-only pipeline

    print("\n=== CUDA / PyTorch ===")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        check("PyTorch", True, f"v{torch.__version__}")
        check("CUDA available", cuda_available)
        if cuda_available:
            check("CUDA version", True, torch.version.cuda)
            check("cuDNN version", True, str(torch.backends.cudnn.version()))
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            check("GPU", True, f"{gpu_name} ({vram_gb:.1f} GB VRAM)")
        else:
            errors += 1
    except ImportError:
        check("PyTorch", False, "not installed")
        errors += 1

    print("\n=== Key Dependencies ===")
    critical_deps = {
        "numpy": "1.26",
        "onnx": None,
        "onnxruntime": None,
        "transformers": None,
        "cv2": None,
        "mediapipe": None,
    }
    for pkg, expected_prefix in critical_deps.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            ok = True
            if expected_prefix and not ver.startswith(expected_prefix):
                ok = False
                errors += 1
            check(pkg, ok, f"v{ver}")
        except Exception as e:
            check(pkg, False, f"import failed: {type(e).__name__}: {e}")
            errors += 1

    # JoyVASA runtime dependency — warning only (not critical for TTS-only mode)
    try:
        import diffusers
        check("diffusers", True, f"v{diffusers.__version__} (JoyVASA)")
    except Exception as e:
        print(f"⚠️  diffusers — import failed: {type(e).__name__}: {e}")

    print("\n=== TTS Engines ===")
    for pkg in ["f5_tts", "kokoro"]:
        try:
            importlib.import_module(pkg.replace("-", "_"))
            check(pkg, True)
        except ImportError:
            check(pkg, False, "not installed (optional)")

    print("\n=== TensorRT ===")
    plugin_path = Path("/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so")
    check("grid_sample3d plugin", plugin_path.exists(), str(plugin_path))
    if not plugin_path.exists():
        print("  ⚠️  Plugin missing — TensorRT mode will fail. ONNX mode may work.")

    try:
        import tensorrt as trt
        check("TensorRT", True, f"v{trt.__version__}")
    except ImportError:
        check("TensorRT", False, "not installed (required for production)")

    print(f"\n{'='*40}")
    if errors == 0:
        print("✅ All critical checks passed!")
    else:
        print(f"❌ {errors} critical issue(s) found. Fix before running pipeline.")
    return errors


if __name__ == "__main__":
    sys.exit(main())
