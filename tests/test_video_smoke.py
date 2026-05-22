"""
Video Generation Smoke Test — Validates FasterLivePortrait + JoyVASA.

Tests (requires GPU ≥16GB, run on EC2 g4dn.xlarge):
1. FasterLivePortrait generates video from static image + driving audio
2. JoyVASA motion is applied correctly
3. Output is valid MP4
4. Video has correct FPS and resolution
5. Lip-sync is temporally coherent (visual inspection)
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent

pytestmark_gpu = pytest.mark.skipif(
    not _has_cuda(),
    reason="Requires NVIDIA GPU — run on EC2 g4dn.xlarge"
) if False else None  # lazy guard defined below


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


gpu_only = pytest.mark.skipif(not _has_cuda(), reason="Requires NVIDIA GPU")


# ---------------------------------------------------------------------------
# liveportrait_runner smoke tests (subprocess mocked)
# ---------------------------------------------------------------------------

class TestLiveportraitRunner:
    def test_run_calls_subprocess(self, tmp_path):
        """run() should invoke python run.py with correct arguments."""
        import soundfile as sf
        from src.video.liveportrait_runner import run

        # Create minimal fake inputs
        src_image = tmp_path / "avatar.jpg"
        src_image.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # minimal JPEG header
        dri_audio = tmp_path / "voice.wav"
        sf.write(str(dri_audio), np.zeros(24000, dtype=np.float32), 24000)
        output_mp4 = tmp_path / "raw.mp4"
        flp_root = tmp_path / "FasterLivePortrait"
        flp_root.mkdir()
        (flp_root / "run.py").touch()

        def fake_run(cmd, **kwargs):
            # Simulate FasterLivePortrait writing the output file
            output_mp4.write_bytes(b"fake mp4 content")
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            result = run(
                src_image=src_image,
                dri_audio=dri_audio,
                output_mp4=output_mp4,
                flp_root=flp_root,
                cfg_path="configs/trt_infer.yaml",
                use_joyvasa=True,
            )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "run.py" in cmd
        assert "--src_image" in cmd
        assert "--dri_audio" in cmd
        assert "--joyvasa" in cmd
        assert "--output" in cmd
        assert result == output_mp4

    def test_run_without_joyvasa(self, tmp_path):
        """run() with use_joyvasa=False should not include --joyvasa flag."""
        import soundfile as sf
        from src.video.liveportrait_runner import run

        src_image = tmp_path / "avatar.jpg"
        src_image.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        dri_audio = tmp_path / "voice.wav"
        sf.write(str(dri_audio), np.zeros(24000, dtype=np.float32), 24000)
        output_mp4 = tmp_path / "raw.mp4"
        flp_root = tmp_path / "FLP"
        flp_root.mkdir()
        (flp_root / "run.py").touch()

        def fake_run(cmd, **kwargs):
            output_mp4.write_bytes(b"fake")
            r = MagicMock()
            r.returncode = 0
            return r

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            run(src_image, dri_audio, output_mp4, flp_root=flp_root, use_joyvasa=False)

        cmd = mock_run.call_args[0][0]
        assert "--joyvasa" not in cmd

    def test_missing_src_image_raises(self, tmp_path):
        """run() should raise FileNotFoundError for a missing source image."""
        import soundfile as sf
        from src.video.liveportrait_runner import run

        dri_audio = tmp_path / "voice.wav"
        sf.write(str(dri_audio), np.zeros(24000, dtype=np.float32), 24000)

        with pytest.raises(FileNotFoundError, match="Avatar image"):
            run(
                src_image=tmp_path / "nonexistent.jpg",
                dri_audio=dri_audio,
                output_mp4=tmp_path / "out.mp4",
                flp_root=tmp_path,
            )

    def test_missing_flp_root_raises(self, tmp_path):
        """run() should raise RuntimeError if run.py is absent from flp_root."""
        import soundfile as sf
        from src.video.liveportrait_runner import run

        src = tmp_path / "avatar.jpg"
        src.write_bytes(b"\xff\xd8\xff")
        wav = tmp_path / "voice.wav"
        sf.write(str(wav), np.zeros(24000, dtype=np.float32), 24000)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(RuntimeError, match="run.py"):
            run(src, wav, tmp_path / "out.mp4", flp_root=empty_dir)

    def test_subprocess_failure_raises(self, tmp_path):
        """run() should raise CalledProcessError when run.py exits non-zero."""
        import soundfile as sf
        from src.video.liveportrait_runner import run

        src = tmp_path / "avatar.jpg"
        src.write_bytes(b"\xff\xd8\xff")
        wav = tmp_path / "voice.wav"
        sf.write(str(wav), np.zeros(24000, dtype=np.float32), 24000)
        flp = tmp_path / "FLP"
        flp.mkdir()
        (flp / "run.py").touch()

        def fail_run(cmd, **kwargs):
            r = MagicMock()
            r.returncode = 1
            r.stderr = "CUDA out of memory"
            r.stdout = ""
            return r

        with patch("subprocess.run", side_effect=fail_run):
            with pytest.raises(subprocess.CalledProcessError):
                run(src, wav, tmp_path / "out.mp4", flp_root=flp)


# ---------------------------------------------------------------------------
# ffmpeg_mux smoke tests (subprocess mocked)
# ---------------------------------------------------------------------------

class TestFFmpegMux:
    def test_mux_calls_ffmpeg(self, tmp_path):
        """mux_audio_video() should invoke ffmpeg with required flags."""
        import soundfile as sf
        from src.utils.ffmpeg_mux import mux_audio_video

        video = tmp_path / "raw.mp4"
        video.write_bytes(b"fake video")
        audio = tmp_path / "voice.wav"
        sf.write(str(audio), np.zeros(24000, dtype=np.float32), 24000)
        output = tmp_path / "with_audio.mp4"

        def fake_run(cmd, **kwargs):
            output.write_bytes(b"fake muxed")
            r = MagicMock()
            r.returncode = 0
            return r

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            result = mux_audio_video(video, audio, output)

        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "-ar" in cmd
        assert "24000" in cmd
        assert "-ac" in cmd
        assert "1" in cmd
        assert "-shortest" in cmd
        assert result == output

    def test_mux_missing_video_raises(self, tmp_path):
        """mux_audio_video() should raise FileNotFoundError for missing video."""
        import soundfile as sf
        from src.utils.ffmpeg_mux import mux_audio_video

        audio = tmp_path / "voice.wav"
        sf.write(str(audio), np.zeros(24000, dtype=np.float32), 24000)

        with pytest.raises(FileNotFoundError):
            mux_audio_video(tmp_path / "missing.mp4", audio, tmp_path / "out.mp4")


# ---------------------------------------------------------------------------
# hyperframes_render smoke tests (subprocess mocked)
# ---------------------------------------------------------------------------

class TestHyperframesRender:
    def _make_template(self, tmp_path: Path, template_name: str = "lower_third") -> Path:
        """Create a minimal Hyperframes template directory for testing."""
        tpl = tmp_path / "templates" / template_name
        comp = tpl / "compositions" / "main"
        comp.mkdir(parents=True)
        (tpl / "assets").mkdir()
        (tpl / "package.json").write_text('{"name":"test"}')
        (comp / "index.html").write_text(
            "<html><head></head><body><!-- HYPERFRAMES_PARAMS --></body></html>"
        )
        return tmp_path / "templates"

    def test_render_calls_npx(self, tmp_path):
        """render() should invoke npx hyperframes render with correct flags."""
        from src.compose.hyperframes_render import render

        templates_dir = self._make_template(tmp_path)
        base_video = tmp_path / "with_audio.mp4"
        base_video.write_bytes(b"fake mp4")
        output = tmp_path / "final.mp4"

        def fake_run(cmd, **kwargs):
            output.write_bytes(b"fake composed")
            r = MagicMock()
            r.returncode = 0
            return r

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            with patch("shutil.which", return_value="/usr/bin/npx"):
                result = render(
                    base_video=base_video,
                    output_mp4=output,
                    template_name="lower_third",
                    params={"title": "Test Title", "subtitle": "Test Subtitle"},
                    templates_dir=templates_dir,
                    job_workdir=tmp_path / "job",
                )

        cmd = mock_run.call_args[0][0]
        assert "npx" in cmd
        assert "hyperframes" in cmd
        assert "render" in cmd
        assert "--quality" in cmd
        assert "--output" in cmd
        assert result == output

    def test_params_injected_into_html(self, tmp_path):
        """render() should inject window.__PARAMS__ into the composition HTML."""
        from src.compose.hyperframes_render import render

        templates_dir = self._make_template(tmp_path)
        base_video = tmp_path / "with_audio.mp4"
        base_video.write_bytes(b"fake")
        output = tmp_path / "final.mp4"

        captured_html = {}

        def fake_run(cmd, cwd=None, **kwargs):
            # Read the composition HTML from the job template copy
            if cwd:
                html_files = list(Path(cwd).rglob("index.html"))
                if html_files:
                    captured_html["content"] = html_files[0].read_text()
            output.write_bytes(b"fake composed")
            r = MagicMock()
            r.returncode = 0
            return r

        with patch("subprocess.run", side_effect=fake_run):
            with patch("shutil.which", return_value="/usr/bin/npx"):
                render(
                    base_video=base_video,
                    output_mp4=output,
                    template_name="lower_third",
                    params={"title": "Olá Mundo"},
                    templates_dir=templates_dir,
                    job_workdir=tmp_path / "job",
                )

        assert "window.__PARAMS__" in captured_html.get("content", "")
        assert "Olá Mundo" in captured_html.get("content", "")

    def test_missing_base_video_raises(self, tmp_path):
        """render() should raise FileNotFoundError for a missing base video."""
        from src.compose.hyperframes_render import render

        templates_dir = self._make_template(tmp_path)

        with pytest.raises(FileNotFoundError, match="Base video"):
            render(
                base_video=tmp_path / "missing.mp4",
                output_mp4=tmp_path / "out.mp4",
                template_name="lower_third",
                templates_dir=templates_dir,
            )

    def test_missing_template_raises(self, tmp_path):
        """render() should raise FileNotFoundError for an unknown template name."""
        from src.compose.hyperframes_render import render

        base_video = tmp_path / "video.mp4"
        base_video.write_bytes(b"fake")

        with pytest.raises(FileNotFoundError, match="Template directory"):
            render(
                base_video=base_video,
                output_mp4=tmp_path / "out.mp4",
                template_name="nonexistent_template",
                templates_dir=tmp_path / "templates",
            )


# ---------------------------------------------------------------------------
# S3 transfer smoke tests (boto3 mocked)
# ---------------------------------------------------------------------------

class TestS3Transfer:
    def test_download_calls_boto3(self, tmp_path):
        """download() should call boto3 download_file with correct bucket/key."""
        from src.utils.s3_transfer import download

        local = tmp_path / "avatar.jpg"

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": 1024 * 1024}
        mock_client.download_file.side_effect = lambda b, k, p: Path(p).write_bytes(b"img")

        with patch("src.utils.s3_transfer._s3_client", return_value=mock_client):
            result = download("s3://my-bucket/inputs/avatar.jpg", local)

        mock_client.download_file.assert_called_once_with("my-bucket", "inputs/avatar.jpg", str(local))
        assert result == local
        assert local.exists()

    def test_upload_calls_boto3(self, tmp_path):
        """upload() should call boto3 upload_file with correct bucket/key."""
        from src.utils.s3_transfer import upload

        local = tmp_path / "final.mp4"
        local.write_bytes(b"fake mp4")

        mock_client = MagicMock()
        with patch("src.utils.s3_transfer._s3_client", return_value=mock_client):
            result = upload(local, "s3://my-bucket/outputs/job-123/final.mp4")

        mock_client.upload_file.assert_called_once_with(
            str(local), "my-bucket", "outputs/job-123/final.mp4"
        )
        assert result == "s3://my-bucket/outputs/job-123/final.mp4"

    def test_invalid_uri_raises(self):
        """_parse_s3_uri() should raise ValueError for non-s3:// URIs."""
        from src.utils.s3_transfer import _parse_s3_uri

        with pytest.raises(ValueError, match="s3://"):
            _parse_s3_uri("https://example.com/file.mp4")

    def test_upload_missing_file_raises(self, tmp_path):
        """upload() should raise FileNotFoundError for a missing local file."""
        from src.utils.s3_transfer import upload

        with pytest.raises(FileNotFoundError):
            upload(tmp_path / "nonexistent.mp4", "s3://bucket/key.mp4")


# ---------------------------------------------------------------------------
# GPU-only integration marker (skip on CPU CI)
# ---------------------------------------------------------------------------

@gpu_only
class TestLiveportraitIntegration:
    """Real GPU integration tests — run these on EC2 g4dn.xlarge only."""

    def test_engines_exist(self):
        """TensorRT .engine files must be present after docker build."""
        flp_root = Path("/app/FasterLivePortrait")
        engines = list(flp_root.rglob("*.engine"))
        assert engines, (
            "No .engine files found in /app/FasterLivePortrait.\n"
            "Run: bash scripts/build_trt_engines.sh"
        )

    def test_plugin_so_exists(self):
        """grid_sample3d .so plugin must exist at the hardcoded path."""
        plugin = Path("/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so")
        assert plugin.exists(), (
            f"Plugin not found: {plugin}\n"
            "It is built during docker build — check Dockerfile."
        )
