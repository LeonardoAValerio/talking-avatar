# Talking Avatar — Open-Source Video Generation Pipeline

Sistema open-source de geração de vídeos com avatares falantes (talking heads), similar ao HeyGen.

## Pipeline

```
Texto → [F5-TTS / Kokoro] → WAV → [FasterLivePortrait + JoyVASA] → MP4 → [Hyperframes] → MP4 Final
```

## Stack Técnico

| Componente | Ferramenta | Licença |
|---|---|---|
| TTS (pt-br) | F5-TTS + checkpoint brasileiro | CC-BY-NC |
| TTS (rápido) | Kokoro-82M | Apache 2.0 |
| Audio-to-Video | FasterLivePortrait + JoyVASA | Apache 2.0 |
| Otimização GPU | NVIDIA TensorRT 8.6.1 | Proprietário |
| Pós-processamento | Hyperframes (HeyGen) | Apache 2.0 |
| Infraestrutura | AWS Batch + Step Functions + Lambda | — |

## Requisitos

- **GPU**: NVIDIA T4 16GB (AWS g4dn.xlarge) para pipeline completo
- **GPU local**: ≥6GB para TTS-only (dev)
- **Docker base**: `shaoguo/faster_liveportrait:v3` (CUDA 11.8)
- **Node.js**: 22+ (para Hyperframes)

## Quick Start

### Fase 1 — TTS local (6GB GPU)
```bash
python -m venv .venv --python=python3.10
source .venv/bin/activate
pip install -r requirements-tts.txt
python scripts/validate_env.py
python -m src.tts.f5tts_synth --text "Olá mundo" --out test.wav
```

### Fase 2 — Pipeline completo (EC2 g4dn.xlarge)
```bash
# Na EC2 com Docker + nvidia-container-toolkit
docker build -f docker/Dockerfile -t talking-avatar .
docker run --gpus all talking-avatar \
  --text "Olá, este é um vídeo de demonstração." \
  --voice narrator_male_01 \
  --avatar /app/assets/avatars/presenter.jpg
```

### Fase 3 — Deploy AWS
```bash
# Push para ECR + deploy CloudFormation
aws cloudformation deploy --template-file aws/cloudformation/stack.yaml \
  --stack-name talking-avatar --capabilities CAPABILITY_IAM
```

## Documentação técnica

Veja `CLAUDE.md` para o blueprint técnico completo com:
- Versões exatas de todas as dependências
- Decisões arquiteturais e justificativas
- Pitfalls conhecidos e soluções
- Comandos de referência para cada estágio
- Plano de implementação faseado

## Custos estimados (AWS, us-east-1)

| Volume | Custo mensal estimado |
|---|---|
| 100 vídeos/mês | ~US$ 5 |
| 1.000 vídeos/mês | ~US$ 25 |
| 10.000 vídeos/mês | ~US$ 215 |
