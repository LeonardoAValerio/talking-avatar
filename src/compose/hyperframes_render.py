"""
Hyperframes Renderer — HTML+GSAP overlays composited onto avatar video.

Implementation notes for Claude Code:
- Hyperframes renders HTML compositions into MP4 via Chrome headless + FFmpeg
- Requires Node.js 22+ and FFmpeg in PATH
- The base avatar video goes into the composition as <video src="../../assets/base.mp4">
- Overlays (lower-thirds, logos, captions) are defined in HTML+CSS+GSAP
- Templates live in src/compose/templates/<template_name>/

Workflow:
    1. Copy base video to template assets/ folder
    2. Inject dynamic params (title, subtitle, logo) into composition
    3. Run: npx hyperframes render --quality high --workers 1 --output <out.mp4>

Template structure:
    templates/<name>/
    ├── package.json          # Hyperframes project config
    ├── compositions/
    │   └── main/
    │       └── index.html    # Main composition with <video> + overlays
    └── assets/
        └── base.mp4          # Copied at runtime from pipeline

Skills for Claude Code:
    npx skills add heygen-com/hyperframes
    (Adds /hyperframes, /gsap, /hyperframes-cli slash commands)
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
_TEMPLATES_DIR = _REPO_ROOT / "src" / "compose" / "templates"

_PARAMS_MARKER = "<!-- HYPERFRAMES_PARAMS -->"


def render(
    base_video: str | Path,
    output_mp4: str | Path,
    template_name: str,
    *,
    params: dict | None = None,
    quality: str = "high",
    workers: int = 1,
    templates_dir: str | Path | None = None,
    job_workdir: str | Path | None = None,
) -> Path:
    """Composite an overlay onto a video using Hyperframes.

    Copies the template to a per-job directory (avoiding concurrent write
    conflicts), injects runtime params, and runs `npx hyperframes render`.

    Args:
        base_video: Path to the muxed MP4 (audio + video from FasterLivePortrait).
        output_mp4: Destination path for the final composed MP4.
        template_name: Name of the template directory under templates/.
        params: Dict of runtime params injected as window.__PARAMS__ in the HTML.
                Typical keys: title, subtitle, logo_url, bg_music, captions_srt.
        quality: Hyperframes render quality (high / medium / low).
        workers: Number of parallel render workers (keep 1 on T4 to avoid OOM).
        templates_dir: Override for the templates root directory.
        job_workdir: Override working directory for this job's template copy.

    Returns:
        Path to the final composed MP4.

    Raises:
        FileNotFoundError: If base_video or template directory is missing.
        RuntimeError: If npx or hyperframes is not found in PATH.
        subprocess.CalledProcessError: If hyperframes exits with a non-zero code.
    """
    base_video = Path(base_video)
    output_mp4 = Path(output_mp4)
    templates_root = Path(templates_dir) if templates_dir else _TEMPLATES_DIR
    template_src = templates_root / template_name

    if not base_video.exists():
        raise FileNotFoundError(f"Base video not found: {base_video}")
    if not template_src.is_dir():
        raise FileNotFoundError(
            f"Template directory not found: {template_src}\n"
            "Expected structure: templates/<name>/package.json + compositions/main/index.html"
        )

    # Copy template to job workdir for isolation between concurrent jobs
    if job_workdir:
        job_template = Path(job_workdir) / "templates" / template_name
    else:
        job_template = base_video.parent / "templates" / template_name

    if job_template.exists():
        shutil.rmtree(job_template)
    shutil.copytree(template_src, job_template)

    # Copy base video into template assets/
    assets_dir = job_template / "assets"
    assets_dir.mkdir(exist_ok=True)
    shutil.copy2(base_video, assets_dir / "base.mp4")
    logger.debug("Copied base video to %s/assets/base.mp4", job_template)

    # Inject runtime params into the main composition HTML
    composition_html = _find_composition_html(job_template)
    _inject_params(composition_html, params or {})

    # Run hyperframes render from inside the job template directory
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    _run_hyperframes(job_template, output_mp4, quality=quality, workers=workers)

    logger.info(
        "Hyperframes render complete: %s (%.1f MB)",
        output_mp4,
        output_mp4.stat().st_size / 1024**2,
    )
    return output_mp4


def _find_composition_html(template_dir: Path) -> Path:
    """Locate the main composition HTML file within a template directory."""
    # Standard Hyperframes structure: compositions/main/index.html
    standard = template_dir / "compositions" / "main" / "index.html"
    if standard.exists():
        return standard

    # Fallback: first HTML file found in compositions/
    compositions = template_dir / "compositions"
    if compositions.is_dir():
        found = sorted(compositions.rglob("*.html"))
        if found:
            return found[0]

    raise FileNotFoundError(
        f"No composition HTML found in {template_dir}.\n"
        "Expected: compositions/main/index.html"
    )


def _inject_params(html_path: Path, params: dict) -> None:
    """Replace the HYPERFRAMES_PARAMS marker with a window.__PARAMS__ script tag."""
    content = html_path.read_text(encoding="utf-8")

    params_json = json.dumps(params, ensure_ascii=False, indent=2)
    script_tag = f'<script>window.__PARAMS__ = {params_json};</script>'

    if _PARAMS_MARKER in content:
        content = content.replace(_PARAMS_MARKER, script_tag)
    else:
        # Fallback: inject just before </head>
        content = content.replace("</head>", f"{script_tag}\n</head>", 1)

    html_path.write_text(content, encoding="utf-8")
    logger.debug("Params injected into %s: %s", html_path.name, list(params.keys()))


def _run_hyperframes(template_dir: Path, output_mp4: Path, *, quality: str, workers: int) -> None:
    """Invoke npx hyperframes render from within the template directory."""
    if not shutil.which("npx"):
        raise RuntimeError(
            "npx not found in PATH. Install Node.js 22+ (see Dockerfile)."
        )

    cmd = [
        "npx", "hyperframes", "render",
        "--quality", quality,
        "--workers", str(workers),
        "--output", str(output_mp4),
    ]

    logger.info("Hyperframes render: quality=%s workers=%d", quality, workers)
    logger.debug("Command: %s (cwd=%s)", " ".join(cmd), template_dir)

    result = subprocess.run(cmd, cwd=str(template_dir), capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Hyperframes stderr:\n%s", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
