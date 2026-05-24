# =============================================================================
# Talking Avatar — Production Dockerfile
# Base: shaoguo/faster_liveportrait:v3 (CUDA 11.8 + cuDNN 8.x + TRT 8.6.1)
# Target: AWS EC2 g4dn.xlarge (NVIDIA T4, 16GB VRAM)
# =============================================================================

FROM shaoguo/faster_liveportrait:v3 AS base

# TensorRT lives at /opt/TensorRT-8.6.1.6 inside the base image (not /opt/tensorrt)
# Python 3.10 + pip are already in /root/miniconda3 (conda base env of the base image)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH \
    TENSORRT_HOME=/opt/TensorRT-8.6.1.6

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git git-lfs ffmpeg espeak-ng \
        build-essential cmake \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# ── Clone source repos ───────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/warmshao/FasterLivePortrait /app/FasterLivePortrait \
    && git clone --depth 1 https://github.com/jdh-algo/JoyVASA /app/JoyVASA

# ── Python dependencies (PINNED — do not upgrade) ────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip==24.2 \
    && pip install --no-cache-dir -r /app/requirements.txt

# ── grid-sample3d TRT plugin ─────────────────────────────────────────────────
# The base image already has the repo at /opt/grid-sample3d-trt-plugin.
# We only recompile to add arch 75 (T4) which the upstream CMakeLists.txt omits.
RUN cd /opt/grid-sample3d-trt-plugin \
    && sed -i 's/"70;80;86;89"/"60;70;75;80;86"/' CMakeLists.txt \
    && sed -i '/add_subdirectory(test)/d; /enable_testing()/d' CMakeLists.txt \
    && mkdir -p build && cd build \
    && cmake .. -DTensorRT_ROOT=${TENSORRT_HOME} \
    && make -j$(nproc) \
    && test -f /opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so

# ── Download models (baked into image layer) ─────────────────────────────────
RUN cd /app/FasterLivePortrait \
    && huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints \
    && huggingface-cli download hexgrad/Kokoro-82M --local-dir ./checkpoints/Kokoro-82M \
    && huggingface-cli download TencentGameMate/chinese-hubert-base --local-dir ./checkpoints/chinese-hubert-base \
    && huggingface-cli download jdh-algo/JoyVASA --local-dir ./checkpoints/JoyVASA \
    && huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./checkpoints/wav2vec2-base-960h

# ── F5-TTS pt-br checkpoint ──────────────────────────────────────────────────
RUN mkdir -p /app/FasterLivePortrait/checkpoints/F5-TTS-ptbr \
    && huggingface-cli download ModelsLab/F5-tts-brazilian \
       --local-dir /app/FasterLivePortrait/checkpoints/F5-TTS-ptbr

# ── Pre-compile TRT engines (MUST run on T4 / same GPU arch as production) ───
# If the build machine is not a T4, skip this and run build_trt_engines.sh on EC2.
RUN cd /app/FasterLivePortrait \
    && bash scripts/all_onnx2trt.sh \
    && bash scripts/all_onnx2trt_animal.sh || true

# ── Application code ─────────────────────────────────────────────────────────
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY assets/ /app/assets/
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# ── Hyperframes (install globally) ───────────────────────────────────────────
RUN npm install -g hyperframes@latest

WORKDIR /app
ENTRYPOINT ["/app/docker/entrypoint.sh"]
