#!/bin/bash
# =============================================================================
# Build TensorRT engines from ONNX models
# MUST run on the same GPU architecture as production (T4 = compute 7.5)
# Engines are NOT portable between GPU architectures.
# =============================================================================
set -euo pipefail

FLP_DIR="${1:-/app/FasterLivePortrait}"

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo "=== Verifying grid_sample3d plugin ==="
PLUGIN_PATH="/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so"
if [ ! -f "$PLUGIN_PATH" ]; then
    echo "ERROR: grid_sample3d plugin not found at $PLUGIN_PATH"
    echo "Build it first (see Dockerfile)"
    exit 1
fi

echo "=== Converting ONNX → TensorRT engines ==="
cd "$FLP_DIR"

# Standard models (FP16 for speed)
if [ -f "scripts/all_onnx2trt.sh" ]; then
    bash scripts/all_onnx2trt.sh
else
    echo "WARNING: all_onnx2trt.sh not found. Running manual conversion..."
    # motion_extractor MUST be FP32 (per FasterLivePortrait README)
    python scripts/onnx2trt.py \
        -o ./checkpoints/liveportrait_onnx/motion_extractor.onnx -p fp32
    # All others use FP16
    for onnx_file in ./checkpoints/liveportrait_onnx/*.onnx; do
        basename=$(basename "$onnx_file" .onnx)
        if [ "$basename" != "motion_extractor" ]; then
            python scripts/onnx2trt.py -o "$onnx_file" -p fp16
        fi
    done
fi

# Animal models (optional, non-critical)
if [ -f "scripts/all_onnx2trt_animal.sh" ]; then
    bash scripts/all_onnx2trt_animal.sh || echo "WARNING: Animal model TRT build failed (non-critical)"
fi

echo "=== TRT engines built ==="
ls -lh "$FLP_DIR"/checkpoints/**/*.engine 2>/dev/null || echo "No .engine files found"
