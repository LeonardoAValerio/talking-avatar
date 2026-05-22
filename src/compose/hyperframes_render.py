"""
Hyperframes Renderer — HTML+GSAP overlays composited onto avatar video.

Implementation notes for Claude Code:
- Hyperframes renders HTML compositions into MP4 via Chrome headless + FFmpeg
- Requires Node.js 22+ and FFmpeg in PATH
- The base avatar video goes into the composition as <video src="assets/base.mp4">
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
