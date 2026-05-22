"""
F5-TTS Synthesizer — Portuguese Brazilian voice cloning via F5-TTS.

Implementation notes for Claude Code:
- Uses f5-tts_infer-cli as subprocess (isolates VRAM)
- Checkpoint: checkpoints/F5-TTS-ptbr/Brazilian_Portuguese/model_2600000.pt
- Vocab: checkpoints/F5-TTS-ptbr/vocab.txt
- Output: 24kHz mono WAV
- VRAM usage: ~4-6 GB (fits in 6GB GPU for local dev)
- ref_audio + ref_text are required for voice cloning

CLI command reference:
    f5-tts_infer-cli \\
      --model F5-TTS \\
      --ckpt_file <ckpt_path> \\
      --vocab_file <vocab_path> \\
      --ref_audio <reference_wav> \\
      --ref_text "<transcription of ref audio>" \\
      --gen_text "<text to synthesize>" \\
      --output_dir <output_dir> \\
      --nfe_step 32
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths (override via voices.yaml or CLI args)
DEFAULT_CKPT = "checkpoints/F5-TTS-ptbr/Brazilian_Portuguese/model_2600000.pt"
DEFAULT_VOCAB = "checkpoints/F5-TTS-ptbr/vocab.txt"
DEFAULT_NFE_STEP = 32


def synthesize(
    text: str,
    ref_audio: str | Path,
    ref_text: str,
    output_wav: str | Path,
    *,
    ckpt_file: str | Path = DEFAULT_CKPT,
    vocab_file: str | Path = DEFAULT_VOCAB,
    nfe_step: int = DEFAULT_NFE_STEP,
) -> Path:
    """Run F5-TTS inference as an isolated subprocess.

    The subprocess model guarantees the GPU is fully released after this
    function returns — the OS reclaims VRAM on subprocess exit.

    Args:
        text: Portuguese text to synthesize.
        ref_audio: Path to reference WAV (3-10 s) for voice cloning.
        ref_text: Exact transcription of the reference audio.
        output_wav: Destination path for the generated 24kHz mono WAV.
        ckpt_file: Path to the F5-TTS pt-br checkpoint (.pt).
        vocab_file: Path to the vocabulary file (.txt).
        nfe_step: Diffusion sampling steps (32 = quality/speed balance).

    Returns:
        Path to the generated WAV file.

    Raises:
        FileNotFoundError: If ref_audio, ckpt_file or vocab_file is missing.
        subprocess.CalledProcessError: If f5-tts_infer-cli exits non-zero.
    """
    ref_audio = Path(ref_audio)
    output_wav = Path(output_wav)
    ckpt_file = Path(ckpt_file)
    vocab_file = Path(vocab_file)

    for p, label in [(ref_audio, "ref_audio"), (ckpt_file, "ckpt_file"), (vocab_file, "vocab_file")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_wav.parent

    cmd = [
        "f5-tts_infer-cli",
        "--model", "F5-TTS",
        "--ckpt_file", str(ckpt_file),
        "--vocab_file", str(vocab_file),
        "--ref_audio", str(ref_audio),
        "--ref_text", ref_text,
        "--gen_text", text,
        "--output_dir", str(output_dir),
        "--nfe_step", str(nfe_step),
    ]

    logger.info("F5-TTS synthesis starting (nfe_step=%d)", nfe_step)
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("F5-TTS stderr:\n%s", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    # f5-tts_infer-cli writes infer_cli_out.wav by default; rename to desired path
    generated = output_dir / "infer_cli_out.wav"
    if generated.exists() and generated != output_wav:
        generated.rename(output_wav)
    elif not output_wav.exists():
        # Fallback: look for any WAV produced in output_dir
        wavs = list(output_dir.glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(
                f"F5-TTS completed (rc=0) but no WAV found in {output_dir}.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        if wavs[0] != output_wav:
            wavs[0].rename(output_wav)

    logger.info(
        "F5-TTS synthesis complete: %s (%.1f MB)",
        output_wav,
        output_wav.stat().st_size / 1024**2,
    )
    return output_wav


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F5-TTS pt-br synthesizer")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--ref-audio", required=True, help="Reference WAV for voice cloning")
    p.add_argument("--ref-text", required=True, help="Transcription of the reference audio")
    p.add_argument("--out", required=True, help="Output WAV path")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Checkpoint .pt path")
    p.add_argument("--vocab", default=DEFAULT_VOCAB, help="Vocabulary .txt path")
    p.add_argument("--nfe-step", type=int, default=DEFAULT_NFE_STEP, help="Sampling steps")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    synthesize(
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_wav=args.out,
        ckpt_file=args.ckpt,
        vocab_file=args.vocab,
        nfe_step=args.nfe_step,
    )
    print(args.out, flush=True)
    sys.exit(0)
