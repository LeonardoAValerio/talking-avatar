"""
FasterLivePortrait + JoyVASA Runner — Audio-driven video generation.

Implementation notes for Claude Code:
- MUST run as subprocess.run() to isolate VRAM from TTS stage
- Uses TensorRT engines pre-compiled during docker build
- JoyVASA provides audio→motion via Hubert acoustic features

CLI command reference:
    cd /app/FasterLivePortrait && python run.py \\
      --src_image <avatar_image.jpg> \\
      --dri_audio <voice.wav> \\
      --cfg configs/trt_infer.yaml \\
      --joyvasa \\
      --output <output.mp4>

Key YAML flags in configs/trt_infer.yaml:
    flag_relative_motion: true    # Keep true for lip-sync quality
    flag_pasteback: true          # REQUIRED for full-frame output
    flag_stitching: true          # Smooth face boundary
    flag_eye_retargeting: false   # Save VRAM
    flag_normalize_lip: true      # Commercial quality lip closure

VRAM usage: ~8-12 GB (requires T4 16GB or better)
Output: MP4 with video only (no audio track — mux separately via FFmpeg)
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_FLP_ROOT = "/app/FasterLivePortrait"
DEFAULT_CFG = "configs/trt_infer.yaml"


def run(
    src_image: str | Path,
    dri_audio: str | Path,
    output_mp4: str | Path,
    *,
    flp_root: str | Path = DEFAULT_FLP_ROOT,
    cfg_path: str | Path = DEFAULT_CFG,
    use_joyvasa: bool = True,
) -> Path:
    """Run FasterLivePortrait + JoyVASA as an isolated subprocess.

    The subprocess model guarantees VRAM is fully released after this function
    returns — the OS reclaims GPU memory on subprocess exit.

    Args:
        src_image: Avatar source image (JPG/PNG, face clearly visible).
        dri_audio: Driving audio WAV (24kHz mono from TTS stage).
        output_mp4: Destination path for the generated video-only MP4.
        flp_root: Root directory of the FasterLivePortrait repo.
        cfg_path: Path to TRT inference config, relative to flp_root.
        use_joyvasa: Enable JoyVASA audio-driven motion (required for lip-sync).

    Returns:
        Path to the generated video-only MP4.

    Raises:
        FileNotFoundError: If src_image or dri_audio do not exist.
        RuntimeError: If flp_root is not a valid FasterLivePortrait directory.
        subprocess.CalledProcessError: If run.py exits with a non-zero code.
    """
    src_image = Path(src_image)
    dri_audio = Path(dri_audio)
    output_mp4 = Path(output_mp4)
    flp_root = Path(flp_root)

    if not src_image.exists():
        raise FileNotFoundError(f"Avatar image not found: {src_image}")
    if not dri_audio.exists():
        raise FileNotFoundError(f"Driving audio not found: {dri_audio}")
    if not (flp_root / "run.py").exists():
        raise RuntimeError(
            f"FasterLivePortrait run.py not found in: {flp_root}\n"
            "Ensure the repo is cloned at that path (see Dockerfile)."
        )

    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "run.py",
        "--src_image", str(src_image),
        "--dri_audio", str(dri_audio),
        "--cfg", str(cfg_path),
        "--output", str(output_mp4),
    ]
    if use_joyvasa:
        cmd.append("--joyvasa")

    logger.info(
        "FasterLivePortrait starting: image=%s audio=%s joyvasa=%s",
        src_image.name, dri_audio.name, use_joyvasa,
    )
    logger.debug("Command: %s (cwd=%s)", " ".join(cmd), flp_root)

    result = subprocess.run(cmd, cwd=str(flp_root), capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FasterLivePortrait stderr:\n%s", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    if not output_mp4.exists():
        raise FileNotFoundError(
            f"FasterLivePortrait completed (rc=0) but output not found: {output_mp4}\n"
            f"stdout: {result.stdout}"
        )

    size_mb = output_mp4.stat().st_size / 1024**2
    logger.info("FasterLivePortrait complete: %s (%.1f MB)", output_mp4, size_mb)
    return output_mp4
