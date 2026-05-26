#!/bin/bash
set -euo pipefail

echo "=== Talking Avatar Pipeline ==="
echo "CUDA version: $(nvcc --version 2>/dev/null | tail -1 || echo 'not available')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not available')"
echo "Python: $(python --version)"
echo "TensorRT engines: $(ls /app/FasterLivePortrait/checkpoints/*.engine 2>/dev/null | wc -l) found"

# Validate environment before running (non-fatal — pipeline decides whether to continue)
python /app/scripts/validate_env.py || true

# Run the pipeline with all arguments passed through
exec python -m src.pipeline "$@"
