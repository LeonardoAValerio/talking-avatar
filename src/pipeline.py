"""
Talking Avatar Pipeline Orchestrator.

This is the main entry point. It runs the full pipeline:
    1. TTS: Text → WAV audio (F5-TTS or Kokoro)
    2. Video: Avatar image + WAV → raw MP4 (FasterLivePortrait + JoyVASA)
    3. Mux: Combine audio + video into single MP4
    4. Compose: Apply overlays via Hyperframes (optional)

Each stage runs as an isolated subprocess to guarantee VRAM cleanup.

Usage:
    # From CLI
    python -m src.pipeline \\
      --text "Olá, este é um vídeo de demonstração." \\
      --voice narrator_male_01 \\
      --avatar assets/avatars/presenter.jpg \\
      --output output/final.mp4

    # From AWS Batch (env vars)
    TEXT, VOICE_ID, AVATAR_S3_URI, OUTPUT_S3_URI set as env vars
    python -m src.pipeline --from-env

Implementation notes for Claude Code:
- Load config from config/pipeline.yaml
- Resolve voice profile from config/voices.yaml
- Create unique workdir per job: /tmp/talking-avatar/<uuid>/
- Each stage is a function that calls subprocess.run()
- After all stages complete, upload final MP4 to S3 (if AWS mode)
- Clean up workdir on success
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _REPO_ROOT / "config"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_pipeline_config() -> dict:
    return _load_yaml(_CONFIG_DIR / "pipeline.yaml")


def _load_voices_config() -> dict:
    return _load_yaml(_CONFIG_DIR / "voices.yaml")


def _resolve_voice(voice_id: str, voices: dict) -> dict:
    profile = voices.get("voices", {}).get(voice_id)
    if profile is None:
        available = list(voices.get("voices", {}).keys())
        raise ValueError(f"Unknown voice '{voice_id}'. Available: {available}")
    return profile


# ---------------------------------------------------------------------------
# Stage 1 — TTS
# ---------------------------------------------------------------------------

def run_tts_stage(
    text: str,
    voice_id: str,
    workdir: Path,
    cfg: dict,
    voices: dict,
) -> Path:
    """Synthesize speech and return the path to the generated WAV.

    Dispatches to F5-TTS or Kokoro based on the voice profile's engine field.
    F5-TTS runs in a subprocess to isolate VRAM; Kokoro runs in-process.
    """
    profile = _resolve_voice(voice_id, voices)
    engine = profile.get("engine", cfg["tts"]["engine"])
    output_wav = workdir / "voice.wav"

    logger.info("TTS stage: engine=%s voice=%s", engine, voice_id)

    if engine == "f5tts_ptbr":
        from src.tts.f5tts_synth import synthesize as f5_synthesize

        f5_cfg = cfg["tts"]["f5tts"]
        ref_audio = _REPO_ROOT / profile["ref_audio"]
        ref_text = profile["ref_text"]

        f5_synthesize(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            output_wav=output_wav,
            ckpt_file=f5_cfg["ckpt_file"],
            vocab_file=f5_cfg["vocab_file"],
            nfe_step=f5_cfg.get("nfe_step", 32),
        )

    elif engine == "kokoro":
        from src.tts.kokoro_synth import synthesize as kokoro_synthesize

        kokoro_synthesize(
            text=text,
            output_wav=output_wav,
            voice=profile.get("voice", cfg["tts"]["kokoro"]["default_voice"]),
            lang_code=profile.get("lang_code", cfg["tts"]["kokoro"]["lang_code"]),
        )

    else:
        raise ValueError(f"Unknown TTS engine: '{engine}'")

    if not output_wav.exists():
        raise RuntimeError(f"TTS stage produced no output at {output_wav}")

    logger.info("TTS stage complete: %s", output_wav)
    return output_wav


# ---------------------------------------------------------------------------
# Stage 2 — Video generation (FasterLivePortrait + JoyVASA)
# ---------------------------------------------------------------------------

def run_video_stage(
    wav_path: Path,
    avatar_path: Path,
    workdir: Path,
    cfg: dict,
) -> Path:
    """Generate audio-driven video and return the video-only MP4 path."""
    from src.video.liveportrait_runner import run as flp_run

    video_cfg = cfg["video"]
    output_raw = workdir / "raw.mp4"

    logger.info("Video stage: avatar=%s", avatar_path.name)

    flp_run(
        src_image=avatar_path,
        dri_audio=wav_path,
        output_mp4=output_raw,
        flp_root=video_cfg["flp_root"],
        cfg_path=video_cfg["config"],
        use_joyvasa=video_cfg.get("use_joyvasa", True),
    )

    logger.info("Video stage complete: %s", output_raw)
    return output_raw


# ---------------------------------------------------------------------------
# Stage 3 — Mux (audio + video → single MP4)
# ---------------------------------------------------------------------------

def run_mux_stage(
    raw_mp4: Path,
    wav_path: Path,
    workdir: Path,
    cfg: dict,
) -> Path:
    """Combine the video-only MP4 with the WAV audio track."""
    from src.utils.ffmpeg_mux import mux_audio_video

    mux_cfg = cfg["mux"]
    output_muxed = workdir / "with_audio.mp4"

    logger.info("Mux stage: %s + %s", raw_mp4.name, wav_path.name)

    mux_audio_video(
        video_path=raw_mp4,
        audio_path=wav_path,
        output_path=output_muxed,
        audio_sample_rate=mux_cfg["audio_sample_rate"],
        audio_channels=mux_cfg["audio_channels"],
        audio_bitrate=mux_cfg["audio_bitrate"],
    )

    logger.info("Mux stage complete: %s", output_muxed)
    return output_muxed


# ---------------------------------------------------------------------------
# Stage 4 — Compose (Hyperframes overlays — optional)
# ---------------------------------------------------------------------------

def run_compose_stage(
    muxed_mp4: Path,
    workdir: Path,
    cfg: dict,
    compose_params: dict | None = None,
) -> Path:
    """Apply Hyperframes overlays and return the final composed MP4."""
    from src.compose.hyperframes_render import render

    compose_cfg = cfg["compose"]
    output_final = workdir / "final.mp4"

    params = {**compose_cfg.get("params", {}), **(compose_params or {})}

    logger.info("Compose stage: template=%s", compose_cfg["template"])

    render(
        base_video=muxed_mp4,
        output_mp4=output_final,
        template_name=compose_cfg["template"],
        params=params,
        quality=compose_cfg.get("quality", "high"),
        workers=compose_cfg.get("workers", 1),
        job_workdir=workdir,
    )

    logger.info("Compose stage complete: %s", output_final)
    return output_final


# ---------------------------------------------------------------------------
# AWS Batch helpers
# ---------------------------------------------------------------------------

def _resolve_aws_inputs(workdir: Path, cfg: dict) -> tuple[str, str, Path, str | None]:
    """Read TEXT, VOICE_ID, AVATAR_S3_URI, OUTPUT_S3_URI from env vars.

    Returns:
        (text, voice_id, local_avatar_path, output_s3_uri)
    """
    from src.utils.s3_transfer import download

    text = os.environ["TEXT"]
    voice_id = os.environ.get("VOICE_ID", "kokoro_dora")
    avatar_s3 = os.environ.get("AVATAR_S3_URI")
    output_s3 = os.environ.get("OUTPUT_S3_URI")

    if avatar_s3:
        ext = Path(avatar_s3).suffix or ".jpg"
        local_avatar = workdir / f"avatar{ext}"
        download(avatar_s3, local_avatar)
    else:
        # Fallback to a default avatar baked into the image
        defaults = list((_REPO_ROOT / "assets" / "avatars").glob("*.jpg"))
        if not defaults:
            raise RuntimeError(
                "No AVATAR_S3_URI env var and no default avatar in assets/avatars/"
            )
        local_avatar = defaults[0]
        logger.warning("AVATAR_S3_URI not set, using default avatar: %s", local_avatar)

    return text, voice_id, local_avatar, output_s3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Talking Avatar full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--from-env",
        action="store_true",
        help="Read TEXT, VOICE_ID, AVATAR_S3_URI, OUTPUT_S3_URI from env (AWS Batch mode)",
    )

    p.add_argument("--text", help="Text to synthesize")
    p.add_argument("--voice", default="kokoro_dora", help="Voice profile ID from voices.yaml")
    p.add_argument("--avatar", help="Path to avatar image (JPG/PNG)")
    p.add_argument("--output", help="Output file path. WAV if --tts-only, MP4 otherwise.")

    p.add_argument(
        "--title", default="", help="Lower-third title text (Hyperframes compose stage)"
    )
    p.add_argument(
        "--subtitle", default="", help="Lower-third subtitle text (Hyperframes compose stage)"
    )

    p.add_argument(
        "--tts-only",
        action="store_true",
        help="Run only the TTS stage (outputs WAV, skips video/mux/compose)",
    )
    p.add_argument(
        "--no-compose",
        action="store_true",
        help="Skip the Hyperframes compose stage (output is muxed MP4 without overlays)",
    )
    p.add_argument(
        "--workdir",
        help="Override job working directory (default: /tmp/talking-avatar/<uuid>/)",
    )
    p.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Do not delete workdir after the job completes",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_pipeline_config()
    voices = _load_voices_config()

    # Set up per-job working directory
    if args.workdir:
        workdir = Path(args.workdir)
    else:
        base = Path(cfg["paths"]["workdir"])
        workdir = base / str(uuid.uuid4())
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("Job workdir: %s", workdir)

    try:
        # ── Resolve inputs ────────────────────────────────────────────────────
        if args.from_env:
            text, voice_id, avatar_path, output_s3_uri = _resolve_aws_inputs(workdir, cfg)
        else:
            if not args.text:
                logger.error("Provide --text or use --from-env")
                return 1
            text = args.text
            voice_id = args.voice
            avatar_path = Path(args.avatar) if args.avatar else None
            output_s3_uri = None

        # ── Stage 1: TTS ──────────────────────────────────────────────────────
        wav_path = run_tts_stage(text, voice_id, workdir, cfg, voices)

        if args.tts_only:
            final_path = wav_path
            if args.output:
                dest = Path(args.output)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wav_path, dest)
                final_path = dest
            print(str(final_path), flush=True)
            return 0

        # ── Stage 2: Video generation ─────────────────────────────────────────
        if avatar_path is None:
            logger.error("--avatar is required for video generation (or use --tts-only)")
            return 1
        if not avatar_path.exists():
            logger.error("Avatar image not found: %s", avatar_path)
            return 1

        raw_mp4 = run_video_stage(wav_path, avatar_path, workdir, cfg)

        # ── Stage 3: Mux ─────────────────────────────────────────────────────
        muxed_mp4 = run_mux_stage(raw_mp4, wav_path, workdir, cfg)

        # ── Stage 4: Compose (optional) ───────────────────────────────────────
        compose_enabled = cfg["compose"].get("enabled", True) and not args.no_compose
        if compose_enabled:
            compose_params = {}
            if args.title:
                compose_params["title"] = args.title
            if args.subtitle:
                compose_params["subtitle"] = args.subtitle
            final_path = run_compose_stage(muxed_mp4, workdir, cfg, compose_params)
        else:
            final_path = muxed_mp4

        # ── Copy to requested output path ─────────────────────────────────────
        if args.output:
            dest = Path(args.output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_path, dest)
            final_path = dest
            logger.info("Output saved to: %s", final_path)

        # ── Upload to S3 (AWS Batch mode) ─────────────────────────────────────
        if output_s3_uri:
            from src.utils.s3_transfer import upload
            upload(final_path, output_s3_uri)

        print(str(final_path), flush=True)

    except Exception:
        logger.exception("Pipeline failed")
        return 1
    finally:
        if not args.keep_workdir and not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)
            logger.debug("Workdir removed: %s", workdir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
