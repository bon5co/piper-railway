"""OpenAI-compatible speech API for Piper TTS (CPU, ONNX).

POST /v1/audio/speech   JSON {model, input, voice, response_format, speed}
GET  /v1/voices         voices on the volume (also GET /v1/audio/voices)
GET  /v1/models
GET  /healthz           200 once the default voice is loaded (no auth)

Bearer auth (API_KEY) is enforced on /v1/*. Voices are *.onnx files with their *.onnx.json
config in VOICES_DIR; extra ones are fetched at boot from EXTRA_VOICES (comma separated names
like en_GB-alan-medium) or can be copied onto the volume.
"""

import hmac
import io
import os
import shutil
import subprocess
import threading
import traceback
import wave

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

API_KEY = os.environ.get("API_KEY", "")
VOICES_DIR = os.environ.get("VOICES_DIR", "/data/voices")
SEED_DIR = os.environ.get("SEED_VOICES_DIR", "/opt/voices")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "en_US-ljspeech-high")
EXTRA_VOICES = [v.strip() for v in os.environ.get("EXTRA_VOICES", "").split(",") if v.strip()]
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "4000"))

FFMPEG_FORMATS = {
    "mp3": ("mp3", "audio/mpeg"),
    "opus": ("ogg", "audio/ogg", ["-c:a", "libopus", "-ar", "24000"]),
    "aac": ("adts", "audio/aac"),
    "flac": ("flac", "audio/flac"),
}


def cgroup_cpus() -> int:
    """Threads follow the container's CPU quota; onnxruntime would otherwise read the host's cores."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


THREADS = int(os.environ.get("PIPER_THREADS") or min(cgroup_cpus(), 16))

import onnxruntime  # noqa: E402

_SessionOptions = onnxruntime.SessionOptions


def _capped_session_options():
    opts = _SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    return opts


onnxruntime.SessionOptions = _capped_session_options  # piper builds its session from this name

from piper import PiperVoice, SynthesisConfig  # noqa: E402
from piper.download_voices import download_voice  # noqa: E402

state = {"status": "loading", "error": None}
voices: dict[str, PiperVoice] = {}
voices_lock = threading.Lock()


def seed_volume():
    os.makedirs(VOICES_DIR, exist_ok=True)
    if any(f.endswith(".onnx") for f in os.listdir(VOICES_DIR)):
        return
    for name in os.listdir(SEED_DIR):
        shutil.copy2(os.path.join(SEED_DIR, name), os.path.join(VOICES_DIR, name))


def list_voices() -> list[str]:
    try:
        return sorted(f[: -len(".onnx")] for f in os.listdir(VOICES_DIR) if f.endswith(".onnx"))
    except OSError:
        return []


def get_voice(name: str) -> PiperVoice:
    with voices_lock:
        voice = voices.get(name)
        if voice is None:
            path = os.path.join(VOICES_DIR, f"{name}.onnx")
            if not os.path.isfile(path):
                raise KeyError(name)
            voice = PiperVoice.load(path)
            voices[name] = voice
        return voice


def boot():
    try:
        seed_volume()
        get_voice(DEFAULT_VOICE)
        state["status"] = "ready"
        print(f"ready: default={DEFAULT_VOICE} threads={THREADS} voices={list_voices()}", flush=True)
    except Exception as exc:
        traceback.print_exc()
        state["status"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return
    for name in EXTRA_VOICES:  # after ready: a slow or failed download never blocks serving
        try:
            if not os.path.isfile(os.path.join(VOICES_DIR, f"{name}.onnx")):
                from pathlib import Path

                download_voice(name, Path(VOICES_DIR))
            print(f"voice available: {name}", flush=True)
        except Exception as exc:
            print(f"EXTRA_VOICES {name} failed: {type(exc).__name__}: {exc}", flush=True)


app = FastAPI(title="Piper TTS", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def bearer_auth(request: Request, call_next):
    if request.url.path.startswith("/v1/") and API_KEY:
        sent = request.headers.get("authorization", "")
        token = sent[7:] if sent.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(token.encode(), API_KEY.encode()):
            return JSONResponse({"error": {"message": "invalid or missing API key", "type": "invalid_request_error"}},
                                status_code=401, headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


class SpeechRequest(BaseModel):
    model: str = "piper"
    input: str
    voice: str = ""
    response_format: str = "wav"
    speed: float = Field(1.0, ge=0.25, le=4.0)


@app.get("/healthz")
def healthz():
    if state["status"] == "ready":
        return {"status": "ready", "voices": len(list_voices())}
    return JSONResponse({"status": state["status"], "error": state["error"]}, status_code=503)


@app.get("/")
def root():
    return {"service": "piper-tts", "status": state["status"], "docs": "POST /v1/audio/speech (Bearer API_KEY)"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "piper", "object": "model", "owned_by": "piper"}]}


@app.get("/v1/voices")
@app.get("/v1/audio/voices")
def voice_list():
    return {"default": DEFAULT_VOICE, "voices": list_voices()}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if state["status"] != "ready":
        raise HTTPException(503, f"not ready: {state['status']}")
    text = req.input.strip()
    if not text:
        raise HTTPException(400, "input is empty")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(413, f"input is {len(text)} characters; the limit is {MAX_INPUT_CHARS}")
    fmt = req.response_format.lower()
    if fmt not in ("wav", "pcm") and fmt not in FFMPEG_FORMATS:
        raise HTTPException(400, f"response_format must be one of wav, pcm, {', '.join(FFMPEG_FORMATS)}")
    # OpenAI voice names (alloy, nova, ...) and an empty voice fall back to the default voice.
    name = req.voice if req.voice in list_voices() else DEFAULT_VOICE
    voice = get_voice(name)
    cfg = SynthesisConfig(length_scale=1.0 / req.speed)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize_wav(text, wav, cfg)
    data = buf.getvalue()
    if fmt == "wav":
        return Response(data, media_type="audio/wav")
    if fmt == "pcm":
        return Response(data[44:], media_type="audio/pcm")
    codec, mime, *extra = FFMPEG_FORMATS[fmt]
    proc = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", *(extra[0] if extra else []), "-f", codec, "pipe:1"],
                          input=data, capture_output=True)
    if proc.returncode != 0:
        raise HTTPException(500, "audio encoding failed")
    return Response(proc.stdout, media_type=mime)


if __name__ == "__main__":
    threading.Thread(target=boot, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), log_level="info")
