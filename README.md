# piper-railway

Piper TTS (piper-tts 1.8.0) behind an OpenAI-compatible speech API, packaged for Railway.

- `POST /v1/audio/speech` with `{"input": "...", "response_format": "wav|pcm|mp3|opus|aac|flac", "voice": "...", "speed": 1.0}`
- `GET /v1/voices`, `GET /v1/models`, `GET /healthz` (no auth)
- Bearer auth from `API_KEY` on `/v1/*`
- Voices live in `VOICES_DIR` (default `/data/voices`, the volume). One public-domain voice
  (`en_US-ljspeech-high`) is baked in and copied there on first boot; add more with
  `EXTRA_VOICES=en_GB-alan-medium,...` or by copying `.onnx` + `.onnx.json` onto the volume.
- ONNX Runtime threads follow the container CPU quota, not the host core count.

Licences: see `NOTICE` (piper-tts is GPL-3.0-or-later; the wrapper is MIT).
