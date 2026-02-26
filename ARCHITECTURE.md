# Omni v2 — Architecture

## Overview

Omni v2 is a **hexagonal (ports & adapters) architecture** wrapping
Qwen2.5-Omni — an audio-in / text+audio-out multimodal model.

The core principle: the application logic never imports a concrete
implementation directly. Every external dependency (model, storage,
transport) is hidden behind an abstract **port**, and the concrete
**adapter** is injected at construction time. This makes every layer
independently testable and swappable.

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                              │
│   Browser UI (Streamlit)  ·  CLI  ·  REST  ·  WebSocket     │
└────────────┬──────────────────────────────┬────────────────-┘
             │ HTTP/WS                       │ subprocess
             ▼                              ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│      API Layer          │   │         CLI Layer           │
│  app/api/rest/router.py │   │  cli/main.py                │
│  app/api/ws/stream.py   │   │  (wires adapters directly,  │
│  app/api/rest/deps.py   │   │   no server needed)         │
└────────────┬────────────┘   └──────────────┬──────────────┘
             │                               │
             ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  app/services/inference.py  — InferenceService             │
│  app/services/registry.py   — ModelRegistry                │
│                                                             │
│  Depends on PORTS only. Never imports an adapter directly.  │
└────────────┬──────────────────────┬────────────────────────┘
             │                     │
             ▼                     ▼
┌─────────────────────┐  ┌───────────────────────────────────┐
│   ModelBackend port │  │         AudioIO port              │
│   (app/domain/      │  │         (app/domain/ports.py)     │
│    ports.py)        │  └───────────────┬───────────────────┘
└────────┬────────────┘                  │
         │                               │
         ▼                               ▼
┌─────────────────────┐  ┌───────────────────────────────────┐
│  QwenOmniBackend    │  │         LocalAudioIO              │
│  (adapters/model/   │  │         (adapters/audio/          │
│   qwen_omni.py)     │  │          local_io.py)             │
└─────────────────────┘  └───────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Qwen2.5-Omni       │
│  (transformers +    │
│   qwen-omni-utils)  │
└─────────────────────┘
```

---

## Directory Structure

```
v2/
├── app/
│   ├── main.py                  # FastAPI factory + lifespan (startup/shutdown)
│   ├── adapters/
│   │   ├── audio/
│   │   │   └── local_io.py      # AudioIO → filesystem
│   │   └── model/
│   │       └── qwen_omni.py     # ModelBackend → Qwen2.5-Omni
│   ├── api/
│   │   ├── rest/
│   │   │   ├── deps.py          # FastAPI dependency injection wiring
│   │   │   └── router.py        # REST endpoints (infer, submit, poll, health, audio)
│   │   └── ws/
│   │       └── stream.py        # WebSocket endpoint (ws://host/ws/stream)
│   ├── core/
│   │   ├── config.py            # All settings via pydantic-settings (OMNI_* env vars)
│   │   └── logging.py           # Structured logging setup
│   ├── domain/
│   │   ├── ports.py             # Abstract interfaces: ModelBackend, AudioIO
│   │   └── schemas.py           # Pydantic schemas for API boundary
│   ├── services/
│   │   ├── inference.py         # InferenceService: the core use case
│   │   └── registry.py          # ModelRegistry: singleton model lifecycle
│   └── utils/
│       └── device.py            # MPS/CUDA dtype helpers, cache flush
├── cli/
│   └── main.py                  # CLI entry point (no server needed)
├── scripts/
│   └── download_model.py        # Download weights (ModelScope → HF Hub → git-lfs)
├── ui/
│   ├── streamlit_app.py         # Browser UI: mic → model → audio playback
│   └── requirements.txt         # UI-only deps (streamlit, audio-recorder-streamlit)
├── audio/
│   ├── input/                   # Default location for input audio files
│   └── output/                  # Generated response WAV files land here
├── models/                      # Model weights directory (gitignored)
├── tests/
│   ├── conftest.py
│   ├── unit/                    # Pure unit tests, no model needed
│   └── integration/             # Smoke tests requiring a loaded model
├── .env.example                 # Copy to .env and fill in values
├── requirements.txt             # Python dependencies
└── Dockerfile
```

---

## Core Abstractions (Ports)

Both ports live in [app/domain/ports.py](app/domain/ports.py).

### `ModelBackend`

```python
class ModelBackend(ABC):
    def initialize(self) -> None: ...
    def process(self, audio, sample_rate, system_prompt) -> InferenceResult: ...
    def stream(self, audio, sample_rate, system_prompt) -> AsyncIterator[StreamChunk]: ...
    def cleanup(self) -> None: ...
    def is_ready(self) -> bool: ...
```

**Current implementation:** `QwenOmniBackend` (Qwen2.5-Omni, local weights)

**To add a new model:** create `app/adapters/model/your_model.py`, implement
`ModelBackend`, inject it into `ModelRegistry.load()` via a config flag.
Zero changes to `InferenceService`, the API layer, or the CLI.

### `AudioIO`

```python
class AudioIO(ABC):
    def read(self, source, target_sr) -> tuple[np.ndarray, int]: ...
    def write(self, audio, destination, sample_rate) -> Path: ...
```

**Current implementation:** `LocalAudioIO` (filesystem read/write)

**To add streaming audio:** create `app/adapters/audio/stream_io.py`,
implement `AudioIO` against WebSocket byte streams, inject it in the
WebSocket handler. Zero changes to `InferenceService`.

---

## Key Value Objects

```python
@dataclass
class InferenceResult:
    text: str
    audio: Optional[np.ndarray]  # float32 mono, 24000 Hz, or None
    sample_rate: int = 24000
    latency_s: float = 0.0       # time inside model.generate() only

@dataclass
class StreamChunk:
    text: str = ""
    audio: Optional[np.ndarray] = None
    is_final: bool = False
```

`latency_s` is the stopwatch **inside** `model.generate()` only.
Wall-clock time (upload + queue + inference + write + poll) is
measured by the UI and shown separately.

---

## Request Lifecycle — Non-Blocking REST

The blocking `POST /infer` was causing Streamlit's browser WebSocket
to drop during 500–700s inference. The solution:

```
POST /infer/submit
  ├── Upload saved to temp file (fast, < 1s)
  ├── job_id = uuid.uuid4().hex[:12]
  ├── _jobs[job_id] = {"status": "pending"}
  ├── asyncio.create_task(_run())          ← inference starts in background thread
  └── return {"job_id": ..., "status": "pending"}  ← returns immediately

asyncio.to_thread(_run_inference_job)      ← runs model.generate() without blocking
  ├── service.infer_from_file(...)
  ├── saves response_*.wav to audio/output/
  └── _jobs[job_id] = {"status": "done", "result": {...}}

GET /infer/poll/{job_id}                   ← UI polls every 5s
  ├── pending → {"status": "pending"}
  ├── done    → {"status": "done", "result": {...}}
  └── error   → {"status": "error", "error": "..."}

GET /audio/{filename}                      ← UI fetches WAV from disk (fast)
```

The `_jobs` dict is **in-memory only**. It does not survive a server
restart. This is acceptable for a single-user local deployment.
For production: replace with Redis or a proper job queue (Celery, ARQ).

---

## Model Loading — Startup Sequence

```
uvicorn starts
  └── lifespan() context manager runs
        ├── Settings validated
        ├── cfg.model_local_path.exists() checked → sys.exit(1) if missing
        ├── ModelRegistry(settings=cfg).load()
        │     └── QwenOmniBackend.initialize()
        │           ├── Qwen2_5OmniProcessor.from_pretrained(local_files_only=True)
        │           └── Qwen2_5OmniForConditionalGeneration.from_pretrained(...)
        │                 device_map: None (MPS) | "auto" (CUDA)
        │                 torch_dtype: float16 (MPS) | bfloat16 (CUDA)
        │                 attn_implementation: sdpa (MPS) | flash_attention_2 (CUDA)
        └── app.state.registry = registry
              ← server begins accepting requests
```

The model is loaded **once** and shared across all requests via
`app.state.registry`. It is never re-loaded between requests.

---

## Configuration

All settings are in `app/core/config.py`. Every field is overrideable
by an `OMNI_`-prefixed environment variable or a `.env` file.

| Setting | Default | Notes |
|---|---|---|
| `OMNI_PROFILE` | `mac_mps` | `mac_mps` / `cuda_prod` / `cpu` |
| `OMNI_MODELS_DIR` | `v2/models/` | Directory containing model subfolders |
| `OMNI_MODEL_SUBDIR` | _(profile default)_ | Exact folder name inside `OMNI_MODELS_DIR` |
| `OMNI_RETURN_AUDIO` | `true` | Set `false` for text-only (faster) |
| `OMNI_SPEAKER` | `Ethan` | `Ethan` or `Chelsie` |
| `OMNI_SYSTEM_PROMPT` | _(see below)_ | **Must keep "virtual human" wording for audio** |
| `OMNI_THINKER_MAX_NEW_TOKENS` | `512` | Caps reasoning length |
| `OMNI_API_HOST` | `0.0.0.0` | uvicorn bind address |
| `OMNI_API_PORT` | `8000` | uvicorn port |
| `OMNI_API_RELOAD` | `false` | Hot-reload (dev only) |

### ⚠️ System Prompt Warning

The default system prompt **must** contain the phrase
`"virtual human developed by the Qwen Team"`. Changing it can silently
disable audio generation — the model will return text only even when
`return_audio=True`. This is a known constraint of Qwen2.5-Omni.

---

## Streaming — Current Status & Upgrade Path

The WebSocket endpoint (`ws://host/ws/stream`) and its protocol are
fully implemented. The current limitation is the model layer:
`model.generate()` in `transformers==4.51.3` does not yield tokens
incrementally — it blocks until the full response is ready.

**When Qwen releases streaming support**, the upgrade is a single
function replacement:

1. In `app/adapters/model/qwen_omni.py`: override `stream()` with an
   async generator that yields `StreamChunk` objects as tokens arrive.
2. In `app/api/ws/stream.py`: replace the `_run_inference_ws()` call
   with an `async for chunk in backend.stream(...)` loop.

**Zero changes needed** to `InferenceService`, the REST endpoints,
the `ModelBackend` port interface, or the UI.

The `ModelBackend.stream()` default implementation already exists in
`ports.py` — it wraps `process()` and yields the full result as one
final chunk, which is what the WebSocket endpoint uses today.

---

## Dependency Injection — How Adapters Are Wired

```
# Server startup (app/main.py)
registry = ModelRegistry(settings=cfg)
registry.load()                    # loads QwenOmniBackend
app.state.registry = registry

# Per-request (app/api/rest/deps.py)
def get_inference_service(registry) -> InferenceService:
    return InferenceService(
        model    = registry.get(),   # shared QwenOmniBackend
        audio_io = LocalAudioIO(),   # new instance per request (stateless)
    )

# Tests
service = InferenceService(
    model    = MockModelBackend(),   # no GPU needed
    audio_io = MockAudioIO(),
)
```

---

## MPS-Specific Behaviour

Apple Silicon MPS does not support `bfloat16` or `device_map="auto"`.
The `mac_mps` profile automatically applies:

- `torch_dtype = float16`
- `attn_implementation = "sdpa"`
- `device_map = None` (sequential loading)
- `fix_mps_dtypes(inputs)` called before every `model.generate()` to
  cast any `bfloat16` input tensors to `float16`

---

## Testing

```bash
# Unit tests — no model, no GPU needed
pytest tests/unit/ -v

# Integration smoke tests — requires loaded model
pytest tests/integration/ -v

# All tests (Docker, no GPU)
docker run --rm omni-v2
```

Unit tests inject `MockModelBackend` and `MockAudioIO` directly into
`InferenceService`, so they run in milliseconds with no device memory.
