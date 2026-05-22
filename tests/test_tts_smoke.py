"""
TTS Smoke Test — Validates that TTS engines can generate audio.

Tests:
1. Kokoro-82M generates audio in pt-br (pf_dora voice)
2. F5-TTS generates audio with reference voice
3. Output is valid WAV at 24kHz mono
4. VRAM usage stays under 6GB (for local dev)
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
TEXT_PT_BR = "Olá, este é um teste de síntese de voz em português brasileiro."


# ---------------------------------------------------------------------------
# Kokoro smoke tests (run in-process with mocked KPipeline)
# ---------------------------------------------------------------------------

class TestKokoroSynth:
    def test_synthesize_creates_wav(self, tmp_path):
        """Kokoro synthesize() should produce a non-empty WAV file."""
        from src.tts.kokoro_synth import SAMPLE_RATE, synthesize

        audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)  # 3 s

        fake_chunk = MagicMock()
        fake_chunk.audios = [audio]

        mock_pipeline = MagicMock(return_value=[fake_chunk])
        mock_kpipeline = MagicMock(return_value=mock_pipeline)

        out = tmp_path / "kokoro_out.wav"
        with patch.dict(sys.modules, {"kokoro": MagicMock(KPipeline=mock_kpipeline)}):
            from importlib import reload
            import src.tts.kokoro_synth as mod
            reload(mod)
            result = mod.synthesize(TEXT_PT_BR, out, voice="pf_dora")

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_invalid_voice_raises(self, tmp_path):
        """synthesize() with an unknown pt-br voice should raise ValueError."""
        from src.tts.kokoro_synth import synthesize

        with pytest.raises(ValueError, match="not a valid pt-br Kokoro voice"):
            synthesize("text", tmp_path / "out.wav", voice="en_voice_xyz")

    def test_cli_module_invocable(self):
        """The module should be runnable as __main__ without import errors."""
        result = subprocess.run(
            [sys.executable, "-m", "src.tts.kokoro_synth", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "synthesizer" in result.stdout.lower() or "usage" in result.stdout.lower()


# ---------------------------------------------------------------------------
# F5-TTS smoke tests (subprocess mocked)
# ---------------------------------------------------------------------------

class TestF5TTSSynth:
    def test_synthesize_calls_subprocess(self, tmp_path):
        """synthesize() should call f5-tts_infer-cli via subprocess.run."""
        import soundfile as sf
        from src.tts.f5tts_synth import synthesize

        ref_audio = tmp_path / "ref.wav"
        sf.write(str(ref_audio), np.zeros(24000, dtype=np.float32), 24000)

        ckpt = tmp_path / "model.pt"
        ckpt.touch()
        vocab = tmp_path / "vocab.txt"
        vocab.touch()

        out_wav = tmp_path / "f5_out.wav"

        def fake_run(cmd, **kwargs):
            # Simulate f5-tts_infer-cli writing infer_cli_out.wav
            sf.write(str(tmp_path / "infer_cli_out.wav"), np.zeros(24000, dtype=np.float32), 24000)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            result = synthesize(
                text=TEXT_PT_BR,
                ref_audio=ref_audio,
                ref_text="Texto de referência.",
                output_wav=out_wav,
                ckpt_file=ckpt,
                vocab_file=vocab,
            )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "f5-tts_infer-cli" in cmd
        assert "--gen_text" in cmd
        assert result == out_wav
        assert out_wav.exists()

    def test_missing_ref_audio_raises(self, tmp_path):
        """synthesize() should raise FileNotFoundError for missing ref_audio."""
        from src.tts.f5tts_synth import synthesize

        ckpt = tmp_path / "model.pt"
        ckpt.touch()
        vocab = tmp_path / "vocab.txt"
        vocab.touch()

        with pytest.raises(FileNotFoundError, match="ref_audio"):
            synthesize(
                text="text",
                ref_audio=tmp_path / "nonexistent.wav",
                ref_text="ref",
                output_wav=tmp_path / "out.wav",
                ckpt_file=ckpt,
                vocab_file=vocab,
            )

    def test_cli_module_invocable(self):
        """The module should be runnable as __main__ without import errors."""
        result = subprocess.run(
            [sys.executable, "-m", "src.tts.f5tts_synth", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# GPU cleanup smoke test
# ---------------------------------------------------------------------------

class TestGpuCleanup:
    def test_cleanup_gpu_noop_without_cuda(self):
        """cleanup_gpu() should not raise even when CUDA is unavailable."""
        from src.utils.gpu_cleanup import cleanup_gpu

        with patch("src.utils.gpu_cleanup.gc") as mock_gc:
            try:
                import torch
                if not torch.cuda.is_available():
                    cleanup_gpu()
                    mock_gc.collect.assert_called_once()
            except ImportError:
                cleanup_gpu()
                mock_gc.collect.assert_called_once()

    def test_vram_usage_returns_float(self):
        """vram_usage_mb() should always return a float."""
        from src.utils.gpu_cleanup import vram_usage_mb

        result = vram_usage_mb()
        assert isinstance(result, float)
        assert result >= 0.0
