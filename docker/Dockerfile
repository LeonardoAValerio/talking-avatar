# =============================================================================
# Talking Avatar — Production Dockerfile
# Base: shaoguo/faster_liveportrait:v3 (CUDA 11.8 + cuDNN 8.x + TRT 8.6.1)
# Target: AWS EC2 g4dn.xlarge (NVIDIA T4, 16GB VRAM)
# =============================================================================

FROM shaoguo/faster_liveportrait:v3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/tensorrt/lib:$LD_LIBRARY_PATH \
    TENSORRT_HOME=/opt/tensorrt

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

# ── grid-sample3d TRT plugin (CRITICAL: add arch 75 for T4) ─────────────────
RUN git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin /opt/grid-sample3d-trt-plugin \
    && cd /opt/grid-sample3d-trt-plugin \
    && sed -i 's/"70;80;86;89"/"60;70;75;80;86"/' CMakeLists.txt \
    && sed -i '/add_subdirectory(test)/d; /enable_testing()/d' CMakeLists.txt \
    && mkdir build && cd build \
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
# This saves ~2 min per cold start. Engines are GPU-architecture-specific.
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
