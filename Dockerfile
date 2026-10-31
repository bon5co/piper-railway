FROM python:3.12-slim-bookworm

# ffmpeg encodes mp3/opus/flac for the OpenAI-style response_format values.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
       piper-tts==1.8.0 \
       fastapi==0.118.0 \
       "uvicorn[standard]==0.37.0"

# One default voice is baked in (LJSpeech, public domain) so a fresh deploy answers with no
# download. It lives outside the volume mount, which would hide it; the server copies it onto
# the volume at first boot.
RUN python -m piper.download_voices en_US-ljspeech-high --download-dir /opt/voices

COPY server.py NOTICE /app/

ENV PYTHONUNBUFFERED=1 \
    VOICES_DIR=/data/voices \
    SEED_VOICES_DIR=/opt/voices \
    DEFAULT_VOICE=en_US-ljspeech-high \
    PORT=8080

EXPOSE 8080

CMD ["python", "/app/server.py"]
