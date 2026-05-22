"""
Kokoro-82M Synthesizer — Lightweight TTS with limited pt-br support.

Implementation notes for Claude Code:
- Uses kokoro Python package directly (can run in-process, <2GB VRAM)
- pt-br lang_code: 'p'
- Available pt-br voices: pf_dora (female), pm_alex (male), pm_santa (male)
- No voice cloning — only built-in voices
- License: Apache 2.0 (safe for commercial use)
- Much faster than F5-TTS but lower quality for Portuguese

Usage pattern:
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='p')
    for chunk in pipeline(text, voice='pf_dora'):
        # chunk.audios contains numpy arrays at 24kHz
        pass
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
PT_BR_VOICES = ("pf_dora", "pm_alex", "pm_santa")
DEFAULT_VOICE = "pf_dora"
DEFAULT_LANG = "p"


def synthesize(
    text: str,
    output_wav: str | Path,
    *,
    voice: str = DEFAULT_VOICE,
    lang_code: str = DEFAULT_LANG,
) -> Path:
    """Synthesize text with Kokoro-82M and write a 24kHz mono WAV.

    Kokoro runs in-process (low VRAM, <2 GB), so no subprocess isolation is
    needed. Still call cleanup_gpu() afterwards if chaining with a heavy stage.

    Args:
        text: Portuguese text to synthesize.
        output_wav: Destination path for the generated WAV file.
        voice: Kokoro voice ID. pt-br options: pf_dora, pm_alex, pm_santa.
        lang_code: Kokoro language code. Use 'p' for Portuguese.

    Returns:
        Path to the generated WAV file.

    Raises:
        ValueError: If an unsupported pt-br voice is requested.
        ImportError: If the `kokoro` package is not installed.
    """
    if lang_code == DEFAULT_LANG and voice not in PT_BR_VOICES:
        raise ValueError(
            f"Voice '{voice}' is not a valid pt-br Kokoro voice. "
            f"Choose from: {', '.join(PT_BR_VOICES)}"
        )

    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise ImportError(
            "kokoro package not found. Install with: pip install kokoro"
        ) from exc

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Kokoro synthesis: voice=%s lang=%s", voice, lang_code)

    pipeline = KPipeline(lang_code=lang_code)

    audio_chunks: list[np.ndarray] = []
    for chunk in pipeline(text, voice=voice):
        if hasattr(chunk, "audios"):
            for arr in chunk.audios:
                audio_chunks.append(np.asarray(arr, dtype=np.float32))
        elif hasattr(chunk, "audio"):
            audio_chunks.append(np.asarray(chunk.audio, dtype=np.float32))

    if not audio_chunks:
        raise RuntimeError("Kokoro returned no audio chunks for the given text.")

    audio = np.concatenate(audio_chunks)

    # Ensure mono (Kokoro may return (N,) or (1, N))
    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    sf.write(str(output_wav), audio, SAMPLE_RATE, subtype="PCM_16")

    logger.info(
        "Kokoro synthesis complete: %s (%.2f s, %.1f MB)",
        output_wav,
        len(audio) / SAMPLE_RATE,
        output_wav.stat().st_size / 1024**2,
    )
    return output_wav


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kokoro-82M pt-br synthesizer")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--out", required=True, help="Output WAV path")
    p.add_argument("--voice", default=DEFAULT_VOICE, choices=PT_BR_VOICES, help="Kokoro voice ID")
    p.add_argument("--lang", default=DEFAULT_LANG, help="Language code (p = pt-br)")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    synthesize(text=args.text, output_wav=args.out, voice=args.voice, lang_code=args.lang)
    print(args.out, flush=True)
    sys.exit(0)
