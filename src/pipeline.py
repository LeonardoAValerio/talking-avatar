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
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Talking Avatar pipeline (Phase 1: TTS only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--from-env",
        action="store_true",
        help="Read TEXT, VOICE_ID from environment variables (AWS Batch mode)",
    )

    p.add_argument("--text", help="Text to synthesize")
    p.add_argument("--voice", default="kokoro_dora", help="Voice profile ID from voices.yaml")
    p.add_argument(
        "--output-wav",
        help="Output WAV path (Phase 1 only). Defaults to /tmp/talking-avatar/<job>/voice.wav",
    )
    p.add_argument(
        "--workdir",
        help="Override job working directory. Defaults to /tmp/talking-avatar/<uuid>/",
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

    # Resolve inputs
    if args.from_env:
        text = os.environ["TEXT"]
        voice_id = os.environ.get("VOICE_ID", "kokoro_dora")
    else:
        if not args.text:
            logger.error("Provide --text or use --from-env")
            return 1
        text = args.text
        voice_id = args.voice

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
        wav_path = run_tts_stage(text, voice_id, workdir, cfg, voices)

        # If caller requested a specific output path, copy there
        if args.output_wav:
            dest = Path(args.output_wav)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, dest)
            logger.info("WAV copied to: %s", dest)
            wav_path = dest

        print(str(wav_path), flush=True)

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
