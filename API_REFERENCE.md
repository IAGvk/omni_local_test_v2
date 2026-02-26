# Omni v2 — API Reference

Base URL: `http://localhost:8000` (default)
Interactive docs (auto-generated): `http://localhost:8000/docs`

---

## REST Endpoints

### `GET /health`

Liveness and model readiness check. Safe to call at any time,
including before the model finishes loading.

**Response `200`**
```json
{
  "status":      "ok",
  "model_ready": true,
  "model_path":  "/path/to/models/Qwen2.5-Omni-3B",
  "profile":     "mac_mps"
}
```

`model_ready: false` means the server is up but model weights are
still being loaded into memory. Retry after a few seconds.

---

### `POST /infer` _(blocking)_

Upload an audio file and wait for the full inference result.

> **⚠️ Not recommended for the UI.** This call blocks for 500–700s on
> typical hardware. Use `POST /infer/submit` + `GET /infer/poll/{job_id}`
> instead. This endpoint remains for programmatic / scripted use.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `audio` | file | ✅ | WAV / FLAC / MP3. Mono or stereo, any sample rate. |
| `system_prompt` | string | ❌ | Override the default system prompt. See ⚠️ note in ARCHITECTURE.md before changing. |
| `return_audio` | bool | ❌ | Default `true`. Set `false` for text-only (faster). |

**Response `200`**
```json
{
  "text":       "The capital of France is Paris.",
  "audio_path": "response_1740000000.wav",
  "latency_s":  523.81,
  "model":      "Qwen2.5-Omni-3B"
}
```

`audio_path` is the **filename only** (not a full path). Fetch it via
`GET /audio/{filename}`. It is `null` when `return_audio=false` or when
the server config has `OMNI_RETURN_AUDIO=false`.

`latency_s` is the time inside `model.generate()` only. It does not
include upload, preprocessing, or disk write time.

**Errors**

| Code | Condition |
|---|---|
| `415` | Content-Type is not audio |
| `500` | Model inference failed |

---

### `POST /infer/submit` _(non-blocking)_

Submit an audio file for inference and return a `job_id` immediately.
The server runs inference in a background thread.

**Request** — same `multipart/form-data` fields as `POST /infer`

**Response `200`** _(returned in < 1s)_
```json
{
  "job_id": "a3f8c1d2e4b5",
  "status": "pending"
}
```

Use the `job_id` with `GET /infer/poll/{job_id}` to retrieve the result.

---

### `GET /infer/poll/{job_id}`

Check the status of a submitted inference job.
Poll this every 5–10 seconds until `status` is `"done"` or `"error"`.

**Path parameter:** `job_id` — returned by `POST /infer/submit`

**Response `200` — still running**
```json
{ "status": "pending" }
```

**Response `200` — completed**
```json
{
  "status": "done",
  "result": {
    "text":       "The capital of France is Paris.",
    "audio_path": "response_1740000000.wav",
    "latency_s":  523.81,
    "model":      "Qwen2.5-Omni-3B"
  }
}
```

**Response `200` — failed**
```json
{
  "status": "error",
  "error":  "CUDA out of memory"
}
```

**Response `404`** — `job_id` not found (server was restarted, or ID is wrong)

> **Note:** The job store is in-memory. Jobs are lost if the server
> restarts. There is no pagination or job history.

---

### `GET /audio/{filename}`

Serve a generated WAV file from the `audio/output/` directory.

**Path parameter:** `filename` — the value of `audio_path` from an
inference response (e.g. `response_1740000000.wav`)

**Response `200`** — WAV file (`audio/wav`)

**Response `404`** — file not found

---

## WebSocket Endpoint

### `WS /ws/stream`

Real-time audio streaming endpoint. **Current behaviour:** accepts
the full audio clip, runs a complete inference pass, returns the result
as a single final chunk. True token-level streaming will be enabled
when Qwen2.5-Omni streaming support is available in `transformers`.
See [ARCHITECTURE.md](ARCHITECTURE.md#streaming--current-status--upgrade-path).

#### Protocol

**Step 1 — Client sends JSON control frame (text)**

```json
{
  "action":        "infer",
  "system_prompt": null,
  "sample_rate":   16000,
  "return_audio":  true
}
```

All fields optional except `sample_rate` if your audio is not 16 kHz.

| Field | Type | Default | Description |
|---|---|---|---|
| `action` | string | `"infer"` | Reserved for future use |
| `system_prompt` | string\|null | `null` | Override default prompt |
| `sample_rate` | int | `16000` | Sample rate of the PCM bytes to follow |
| `return_audio` | bool | `true` | Whether to send audio response frame |

**Step 2 — Client sends raw PCM bytes (binary)**

- dtype: `float32` little-endian
- layout: mono (single channel)
- sample rate: as declared in step 1
- duration: any (the whole utterance at once)

**Step 3 — Server sends JSON text frame(s)**

```json
{ "text": "...", "is_final": false }
```
```json
{ "text": "The complete response text.", "is_final": true }
```

Today only a single `is_final: true` frame is sent. When streaming is
enabled, multiple `is_final: false` frames will be sent as tokens arrive,
followed by a final `is_final: true` frame with the complete text.

**Step 4 — Server sends binary frame** _(if `return_audio: true`)_

Raw PCM float32, mono, **24000 Hz**.
Sent immediately after the final JSON chunk.

Convert to numpy: `np.frombuffer(audio_bytes, dtype=np.float32)`

**Error frame** _(text, sent before close)_

```json
{ "error": "description of what went wrong" }
```

#### Python client example

```python
import asyncio, json
import numpy as np
import soundfile as sf
import websockets

async def main():
    audio, sr = sf.read("question.wav", dtype="float32")

    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        # Step 1: control frame
        await ws.send(json.dumps({
            "sample_rate":  sr,
            "return_audio": True,
        }))

        # Step 2: audio bytes
        await ws.send(audio.astype(np.float32).tobytes())

        # Step 3: receive text chunk(s)
        while True:
            msg = json.loads(await ws.recv())
            if "error" in msg:
                print(f"Error: {msg['error']}")
                return
            print(f"Text: {msg['text']}")
            if msg["is_final"]:
                break

        # Step 4: receive audio bytes
        audio_bytes  = await ws.recv()
        response_pcm = np.frombuffer(audio_bytes, dtype=np.float32)
        sf.write("response.wav", response_pcm, 24000)

asyncio.run(main())
```

---

## Schemas

All schemas are defined in [app/domain/schemas.py](app/domain/schemas.py).

### `InferResponse`
```python
class InferResponse(BaseModel):
    text:       str
    audio_path: Optional[str]  # filename only, or None
    latency_s:  float          # model.generate() time in seconds
    model:      str            # e.g. "Qwen2.5-Omni-3B"
```

### `HealthResponse`
```python
class HealthResponse(BaseModel):
    status:      str   # always "ok"
    model_ready: bool
    model_path:  str
    profile:     str   # "mac_mps" | "cuda_prod" | "cpu"
```

### `WSInferRequest`
```python
class WSInferRequest(BaseModel):
    action:        str           = "infer"
    system_prompt: Optional[str] = None
    sample_rate:   int           = 16000
    return_audio:  bool          = True
```

### `WSChunk`
```python
class WSChunk(BaseModel):
    text:     str  = ""
    is_final: bool = False
```

### `WSError`
```python
class WSError(BaseModel):
    error: str
```

---

## Audio Format Reference

| Direction | Format | Dtype | Channels | Sample Rate |
|---|---|---|---|---|
| Input (upload / WS) | WAV / FLAC / MP3 | any | mono or stereo | any (resampled to 16 kHz internally) |
| Output (disk) | WAV | float32 | mono | 24000 Hz |
| Output (WS binary frame) | raw PCM | float32 | mono | 24000 Hz |

The model natively outputs at **24000 Hz**. This is set by
`OMNI_OUTPUT_SAMPLE_RATE` (default `24000`) and should not be changed
without resampling the output.
