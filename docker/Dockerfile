FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONPATH=/app
ENV TTS_HOME=/app/tts-models
ENV HF_HOME=/app/hf-cache
ENV COQUI_TOS_AGREED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models into the image so no network volume is needed.
# Workers can then run in any datacenter. Only include models you actually use.
# XTTS v2 is ~2 GB; add/remove entries to match your production usage.
RUN python -c "\
from TTS.api import TTS; \
TTS('tts_models/multilingual/multi-dataset/xtts_v2'); \
TTS('tts_models/multilingual/multi-dataset/xtts_v1.1'); \
TTS('tts_models/en/ljspeech/vits'); \
TTS('tts_models/en/ljspeech/glow-tts'); \
TTS('tts_models/en/ljspeech/tacotron2-DDC'); \
"

COPY app /app/app
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
