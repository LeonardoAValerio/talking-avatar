"""
FFmpeg Multiplexer — Combines video (from FasterLivePortrait) with audio (from TTS).

Implementation notes for Claude Code:
- FasterLivePortrait outputs video-only MP4
- TTS outputs 24kHz mono WAV
- This module combines them into a single MP4

FFmpeg command:
    ffmpeg -y -i <video.mp4> -i <audio.wav> \\
      -c:v copy -c:a aac -b:a 192k -ar 24000 -ac 1 \\
      -shortest <output.mp4>

CRITICAL: Use -ar 24000 -ac 1 to match F5-TTS output format.
The -shortest flag ensures the output matches the shorter of video/audio.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def mux_audio_video(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    audio_sample_rate: int = 24000,
    audio_channels: int = 1,
    audio_bitrate: str = "192k",
) -> Path:
    """Mux a video-only MP4 with a WAV audio track into a single MP4.

    Args:
        video_path: Path to video-only MP4 (FasterLivePortrait output).
        audio_path: Path to WAV audio file (TTS output, 24kHz mono).
        output_path: Destination MP4 path.
        audio_sample_rate: Target sample rate; must match TTS output (24000 Hz).
        audio_channels: Number of audio channels; F5-TTS outputs mono (1).
        audio_bitrate: AAC bitrate for the encoded audio track.

    Returns:
        Path to the muxed output file.

    Raises:
        subprocess.CalledProcessError: If FFmpeg exits with a non-zero code.
        FileNotFoundError: If input files do not exist.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", str(audio_sample_rate),
        "-ac", str(audio_channels),
        "-shortest",
        str(output_path),
    ]

    logger.info("Muxing audio+video → %s", output_path)
    logger.debug("FFmpeg cmd: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg stderr:\n%s", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    logger.info("Mux complete: %s (%.1f MB)", output_path, output_path.stat().st_size / 1024**2)
    return output_path


def audio_only_encode(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    audio_sample_rate: int = 24000,
    audio_channels: int = 1,
    audio_bitrate: str = "192k",
) -> Path:
    """Re-encode a WAV to AAC MP4 (no video stream). Used for QA/preview."""
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", str(audio_sample_rate),
        "-ac", str(audio_channels),
        str(output_path),
    ]

    logger.info("Encoding audio → %s", output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg stderr:\n%s", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    return output_path
