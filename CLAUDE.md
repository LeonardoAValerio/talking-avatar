# Talking Avatar — Blueprint Técnico para Claude Code

## O que é este projeto
Sistema open-source de geração de vídeos com avatares falantes (talking heads), similar ao HeyGen.
Pipeline: Texto → TTS (F5-TTS/Kokoro) → Audio-to-Video (FasterLivePortrait+JoyVASA) → Pós-processamento (Hyperframes) → MP4 final.

## Decisões arquiteturais FIXAS (não altere)

### Stack de ML — versões EXATAS obrigatórias
Toda a cadeia depende de compatibilidade frágil entre CUDA/cuDNN/TensorRT. **NÃO atualize versões.**

| Componente | Versão | Motivo |
|---|---|---|
| Docker base | `shaoguo/faster_liveportrait:v3` | CUDA 11.8 + TRT 8.6.1 + grid_sample plugin pré-compilado |
| CUDA | **11.8** | 12.x quebra TRT 8.6 e grid_sample CUDA fork |
| cuDNN | **8.9.7** (linha 8.x) | 9.x incompatível com grid_sample plugin |
| TensorRT | **8.6.1.6** | ≥10.x não compatível com FasterLivePortrait |
| Python | **3.10** | Requerido por todos os modelos |
| PyTorch | **2.2.2+cu118** | JoyVASA requer; supera 2.0.1 do FLP |
| onnxruntime-gpu | **1.18.0** | Pin do JoyVASA |
| numpy | **1.26.4** | 2.x quebra onnx/opencv/transformers |

### Motores de TTS
- **F5-TTS** com checkpoint `ModelsLab/F5-tts-brazilian` → produção pt-br com voice cloning
- **Kokoro-82M** → fallback rápido, Apache 2.0, limitado em pt-br (3 vozes: pf_dora, pm_alex, pm_santa)

### Motor de vídeo
- **FasterLivePortrait** (warmshao/FasterLivePortrait) com TensorRT em modo headless
- **JoyVASA** (jdh-algo/JoyVASA) para audio-driven motion via Hubert
- **NÃO use modelos de Difusão** (Hallo, EchoMimic, DreamTalk) — estouram 16GB VRAM da T4

### Pós-processamento
- **Hyperframes** (heygen-com/hyperframes) → HTML+GSAP renderizado em MP4 via Chrome headless + FFmpeg
- Node.js 22+ obrigatório para Hyperframes

### Infraestrutura AWS (produção)
- EC2 g4dn.xlarge (1x NVIDIA T4, 16GB VRAM, 16GB RAM, 4 vCPUs)
- AWS Batch com Spot Instances (~US$ 0,23/h vs US$ 0,53/h On-Demand)
- Lambda como trigger → Step Functions para orquestração → Batch para execução GPU
- ECR para imagem Docker, S3 para input/output

## Regras para implementação

### Gerenciamento de VRAM (CRÍTICO)
- Cada estágio (TTS, Video) DEVE rodar como **subprocess separado** (`subprocess.run`)
- Quando o subprocess termina, o SO libera VRAM 100% (mais confiável que `torch.cuda.empty_cache()`)
- **NUNCA** carregue TTS e FasterLivePortrait na mesma instância Python simultaneamente
- Se precisar rodar no mesmo processo, chame `torch.cuda.empty_cache()` + `gc.collect()` entre estágios

### Execução headless
- **ZERO GUI**: sem Gradio, sem Flask, sem cv2.imshow, sem janelas
- Tudo via CLI e scripts Python com argumentos
- Configuração via YAML (`config/pipeline.yaml`)

### TensorRT engines
- Engines são GPU-específicos: compilar em T4 (compute 7.5), rodar em T4
- Plugin grid_sample3d DEVE ter `CMAKE_CUDA_ARCHITECTURES="60;70;75;80;86"` (75 = T4)
- Path hardcoded do .so: `/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so`
- Pré-compile engines no `docker build`, não em runtime

### Áudio F5-TTS
- Output é 24kHz mono WAV
- FFmpeg mux precisa de `-ar 24000 -ac 1` para evitar warnings/crashes

### Licenciamento
- InsightFace `buffalo_l` restringe uso comercial → use `--mp` (MediaPipe) em produção comercial
- F5-TTS base é CC-BY-NC (dataset Emilia) → checkpoint pt-br herda restrição
- Kokoro é Apache 2.0 → seguro para uso comercial

## Estrutura do projeto

```
talking-avatar/
├── CLAUDE.md                       # Este arquivo
├── docker/
│   ├── Dockerfile                  # Produção (base: shaoguo/faster_liveportrait:v3)
│   ├── Dockerfile.tts              # TTS-only para dev local (6GB GPU)
│   └── entrypoint.sh
├── src/
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── kokoro_synth.py         # Wrapper Kokoro-82M
│   │   └── f5tts_synth.py         # Wrapper F5-TTS pt-br
│   ├── video/
│   │   ├── __init__.py
│   │   └── liveportrait_runner.py  # Chama FasterLivePortrait via subprocess
│   ├── compose/
│   │   ├── __init__.py
│   │   ├── hyperframes_render.py   # Invoca npx hyperframes render
│   │   └── templates/              # HTML templates para overlays
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── ffmpeg_mux.py           # Mux áudio + vídeo
│   │   ├── gpu_cleanup.py          # VRAM cleanup utilities
│   │   └── s3_transfer.py          # Upload/download S3
│   └── pipeline.py                 # Orquestrador principal
├── config/
│   ├── pipeline.yaml               # Config principal do pipeline
│   └── voices.yaml                 # Mapping voice_id → ref_audio/ref_text
├── scripts/
│   ├── download_models.sh          # huggingface-cli batch download
│   ├── build_trt_engines.sh        # ONNX → .engine
│   └── validate_env.py             # Sanity check CUDA/cuDNN/TRT/torch
├── aws/
│   ├── lambda_trigger/
│   │   └── handler.py
│   ├── step_functions/
│   │   └── state_machine.json
│   ├── batch/
│   │   └── job_definition.json
│   └── cloudformation/
│       └── stack.yaml
├── tests/
│   ├── test_tts_smoke.py
│   ├── test_video_smoke.py
│   └── test_e2e.py
├── assets/voices/                  # Áudios de referência para clonagem
├── requirements.txt                # Pinned (produção completa)
├── requirements-tts.txt            # Subset TTS para dev local
├── .gitignore
└── README.md
```

## Comandos de referência

### TTS F5-TTS pt-br
```bash
f5-tts_infer-cli \
  --model F5-TTS \
  --ckpt_file checkpoints/F5-TTS-ptbr/Brazilian_Portuguese/model_2600000.pt \
  --vocab_file checkpoints/F5-TTS-ptbr/vocab.txt \
  --ref_audio assets/voices/narrator.wav \
  --ref_text "Texto de referência do áudio." \
  --gen_text "Texto a ser sintetizado." \
  --output_dir /tmp/job/ \
  --nfe_step 32
```

### TTS Kokoro (alternativa rápida)
```bash
python -m src.tts.kokoro_synth --lang p --voice pf_dora \
  --text "Texto em português" --out /tmp/job/voice.wav
```

### FasterLivePortrait + JoyVASA
```bash
cd /app/FasterLivePortrait && python run.py \
  --src_image /app/sources/avatar.jpg \
  --dri_audio /tmp/job/voice.wav \
  --cfg configs/trt_infer.yaml \
  --joyvasa \
  --output /tmp/job/raw.mp4
```

### Mux áudio + vídeo
```bash
ffmpeg -y -i /tmp/job/raw.mp4 -i /tmp/job/voice.wav \
  -c:v copy -c:a aac -b:a 192k -ar 24000 -ac 1 \
  -shortest /tmp/job/with_audio.mp4
```

### Hyperframes overlay
```bash
cd templates/lower_third && \
  cp /tmp/job/with_audio.mp4 ./assets/base.mp4 && \
  npx hyperframes render --quality high --workers 1 \
    --output /tmp/job/final.mp4
```

## Pitfalls conhecidos (leia antes de implementar)

1. **TensorRT 10.x quebra tudo** — Issue #91 do FasterLivePortrait. Trave em 8.6.1.6.
2. **cuDNN 9.x** quebra build do onnxruntime-gpu com grid_sample CUDA. Use 8.x.
3. **numpy 2.x** quebra onnx, opencv, transformers. Pin em 1.26.4.
4. **grid_sample3d CMakeLists.txt upstream** tem arch `"70;80;86;89"` — falta 75 (T4). Faça sed.
5. **Hardcoded path do .so** em `predictor.py`: `/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so`.
6. **TRT engines são GPU-específicos** — buildados em T4, só rodam em T4.
7. **CDI devices no Docker** (Issue #116): use `--gpus=all`, não CDI strings.
8. **Áudio F5-TTS é 24kHz mono** — FFmpeg precisa de `-ar 24000 -ac 1`.
9. **Spot Instances podem ser interrompidas** — Step Functions com retry obrigatório.
10. **Docker image é ~12GB** — primeiro pull demora. Use EBS persistente no dev.

## Fases de implementação

### Fase 1 — TTS local (sua máquina, 6GB GPU) — 1-2 dias
- Instalar Python 3.10 + requirements-tts.txt
- Validar Kokoro-82M e F5-TTS gerando áudio pt-br
- Checklist: ✅ WAV gerado, ✅ <30s para 10s de fala, ✅ VRAM peak <6GB

### Fase 2 — Pipeline completo em EC2 g4dn.xlarge Spot — 2-3 dias
- Lançar EC2 Spot manual com EBS 100GB persistente
- docker pull shaoguo/faster_liveportrait:v3, testar exemplo nativo
- Construir imagem custom, validar pipeline end-to-end
- Checklist: ✅ TRT engines OK, ✅ MP4 com lip-sync, ✅ Hyperframes overlay, ✅ <5× duração áudio

### Fase 3 — Deploy AWS produção — 3-5 dias
- Push imagem ECR, CloudFormation (VPC + S3 + IAM + ECR)
- AWS Batch: Compute Env g4dn.xlarge Spot + Job Queue + Job Definition
- Step Functions: ValidateInput → SubmitBatchJob(.sync) → PostProcess → SaveResult
- Lambda trigger via API Gateway
- Checklist: ✅ Cold-start <8min, ✅ Spot retry, ✅ CloudWatch logs, ✅ custo <US$0,03/vídeo
