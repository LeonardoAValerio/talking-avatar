#!/bin/bash
# =============================================================================
# Download all required models from Hugging Face
# Run this once during docker build or first setup
# =============================================================================
set -euo pipefail

CHECKPOINTS_DIR="${1:-./checkpoints}"
mkdir -p "$CHECKPOINTS_DIR"

echo "=== Downloading FasterLivePortrait base models ==="
huggingface-cli download warmshao/FasterLivePortrait \
  --local-dir "$CHECKPOINTS_DIR/FasterLivePortrait"

echo "=== Downloading Kokoro-82M ==="
huggingface-cli download hexgrad/Kokoro-82M \
  --local-dir "$CHECKPOINTS_DIR/Kokoro-82M"

echo "=== Downloading chinese-hubert-base (for JoyVASA) ==="
huggingface-cli download TencentGameMate/chinese-hubert-base \
  --local-dir "$CHECKPOINTS_DIR/chinese-hubert-base"

echo "=== Downloading JoyVASA ==="
huggingface-cli download jdh-algo/JoyVASA \
  --local-dir "$CHECKPOINTS_DIR/JoyVASA"

echo "=== Downloading wav2vec2-base-960h ==="
huggingface-cli download facebook/wav2vec2-base-960h \
  --local-dir "$CHECKPOINTS_DIR/wav2vec2-base-960h"

echo "=== Downloading F5-TTS Brazilian Portuguese ==="
huggingface-cli download ModelsLab/F5-tts-brazilian \
  --local-dir "$CHECKPOINTS_DIR/F5-TTS-ptbr"

echo "=== All models downloaded to $CHECKPOINTS_DIR ==="
du -sh "$CHECKPOINTS_DIR"/*
