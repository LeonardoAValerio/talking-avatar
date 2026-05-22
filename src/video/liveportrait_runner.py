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
