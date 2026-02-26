# Omni v2 — Runbook

Audience: anyone running Omni v2 locally — developers, end users,
and AI agents performing maintenance or feature work.

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.11.x | 3.12+ not yet tested |
| RAM | 8 GB | 16 GB recommended for the 3B model |
| Disk | 8 GB free | ~6 GB for model weights |
| macOS (Apple Silicon) | M1 or later | MPS acceleration |
| Linux + NVIDIA GPU | CUDA 12.x | `cuda_prod` profile |
| CPU-only | any | Slow — minutes per inference |

---

## 1. Install

```bash
# Clone and enter the repo
cd /path/to/omni_local_test/v2

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Install qwen-omni-utils (not on PyPI — install from source)
pip install git+https://github.com/QwenLM/qwen-omni-utils.git

# Install UI dependencies
pip install -r ui/requirements.txt
```

> **CUDA users:** install PyTorch with CUDA support **before** running
> `pip install -r requirements.txt`:
> ```bash
> pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 2. Configure

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` for your setup. The most important fields:

```dotenv
# Apple Silicon
OMNI_PROFILE=mac_mps

# NVIDIA GPU
# OMNI_PROFILE=cuda_prod

# Where your model weights live (default: v2/models/)
# OMNI_MODELS_DIR=/Volumes/ExternalSSD/models

# Which model subfolder to use (auto-selected from profile if unset)
# OMNI_MODEL_SUBDIR=Qwen2.5-Omni-3B
```

See [ARCHITECTURE.md](ARCHITECTURE.md#configuration) for the full
settings table.

---

## 3. Download Model Weights

Run once before first use. Downloads ~6 GB.

```bash
# From v2/
python scripts/download_model.py
```

The script tries three sources in order:
1. **ModelScope** (Alibaba CDN — fastest if accessible)
2. **HuggingFace Hub** — fallback
3. **git-lfs clone** — last resort

The weights land in `models/Qwen2.5-Omni-3B/` by default.
Override with `OMNI_MODELS_DIR` and `OMNI_MODEL_SUBDIR` in `.env`.

To verify the download:
```bash
python -m cli.main --env-check
```

Expected output:
```
Profile    : mac_mps
Model path : /path/to/models/Qwen2.5-Omni-3B
Weights    : ✅ found
```

---

## 4. Run the API Server

```bash
# From v2/
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or let the settings drive it:
```bash
python -m app.main
```

**Startup takes 30–90 seconds** — the model weights are loaded into
device memory before the server begins accepting requests.

Watch for this log line confirming readiness:
```
🚀  Omni API ready  profile=mac_mps  model=Qwen2.5-Omni-3B
```

Verify with:
```bash
curl http://localhost:8000/health
# {"status":"ok","model_ready":true,...}
```

---

## 5. Run the Streamlit UI

In a **second terminal** (server must already be running):

```bash
cd /path/to/omni_local_test/v2
source .venv/bin/activate
streamlit run ui/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### UI Walkthrough

1. **Click 🔄 Check backend** in the sidebar — confirm the model is ready
2. **Press 🎙️** to start recording; press again (or wait 2.5s of silence) to stop
3. Your recording is **auto-saved** to the "📁 Saved recordings" bank
4. Click **🚀 Send to Omni** — the job is submitted and a progress bar appears
5. The UI polls every 5s. When inference completes (~500–700s on MPS), the
   text response appears and the audio plays automatically
6. To re-send the **same audio with a different system prompt**: change the
   prompt in the sidebar, open "📁 Saved recordings", click **↩️ Load as active**,
   then click **🚀 Send to Omni** again — no re-recording needed

### Sidebar Options

| Option | Effect |
|---|---|
| Backend URL | Point at a remote server instead of localhost |
| System Prompt | Override the model persona (see ⚠️ in ARCHITECTURE.md before changing) |
| Return audio response | Disable for text-only responses (faster) |
| 🔄 Check backend | Ping `/health` and show model status |
| 🗑️ Clear history | Wipe conversation history and saved recordings from the session |

---

## 6. CLI Usage

For scripted or automated use without the server:

```bash
# Basic: audio file in, text + audio file out
python -m cli.main audio/input/my_question.wav

# Specify output path
python -m cli.main audio/input/my_question.wav --output audio/output/answer.wav

# Text-only (skip audio generation — faster)
python -m cli.main audio/input/my_question.wav --no-audio

# Override system prompt
python -m cli.main audio/input/my_question.wav --prompt "You are a compliance assistant."

# Use a specific device profile
python -m cli.main audio/input/my_question.wav --profile cuda_prod

# Quick smoke test — generates a 440 Hz test tone and runs it through the pipeline
python -m cli.main --test-tone

# Print environment summary (no model loaded)
python -m cli.main --env-check
```

CLI output:
```
🎤  Input  : audio/input/my_question.wav
🤖  Model  : Qwen2.5-Omni-3B

───────────────────────────────────────────────────────
📝  Text    : The answer to your question is...
⏱   Latency : 523.81s
🔊  Audio   : audio/output/response_1740000000.wav
───────────────────────────────────────────────────────
```

> The CLI creates and destroys its own model instance per invocation.
> It does not share the model with the API server.

---

## 7. Docker

```bash
# Build
docker build -t omni-v2 .

# Run smoke tests (no model, no GPU needed)
docker run --rm omni-v2

# Run the server — Apple Silicon note: MPS is not available inside Docker.
# Use cpu or cuda_prod profile.
docker run --rm -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/audio:/app/audio \
  -e OMNI_PROFILE=cpu \
  omni-v2 uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run with NVIDIA GPU
docker run --rm --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/audio:/app/audio \
  -e OMNI_PROFILE=cuda_prod \
  omni-v2 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 8. Run Tests

```bash
# Unit tests — no model, no GPU, runs in seconds
pytest tests/unit/ -v

# Integration smoke tests — requires model weights loaded
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ -v --tb=short
```

---

## 9. Troubleshooting

### Server won't start — "Model weights not found"
```
Model weights not found at: /path/to/models/Qwen2.5-Omni-3B
```
→ Run `python scripts/download_model.py`
→ Or set `OMNI_MODELS_DIR` and `OMNI_MODEL_SUBDIR` to match where your weights are

---

### UI shows "❌ Cannot reach backend"
→ Check that `uvicorn app.main:app --port 8000` is running in another terminal
→ The server log should show `🚀  Omni API ready` — if not, the model is still loading
→ Check the Backend URL in the sidebar matches the actual host/port

---

### Model returns text but no audio (`audio_path: null`)
Most likely causes, in order:

1. `OMNI_RETURN_AUDIO=false` is set in your `.env` → change to `true` and restart
2. The system prompt was changed away from the default wording →
   Qwen2.5-Omni silently disables audio if the prompt doesn't contain
   `"virtual human developed by the Qwen Team"`. Reset to the default.
3. The model ran out of memory generating audio → check for OOM errors
   in the server log

---

### UI shows "⚠️ Backend returned no audio path"
Same causes as above. The inference completed but the model skipped audio
generation. Check `OMNI_RETURN_AUDIO` and the system prompt first.

---

### "Read timed out" in the browser
This was the pre-fix symptom of using the blocking `POST /infer` with
Streamlit. The UI now uses `POST /infer/submit` + polling. If you still
see it:
→ Ensure you are running the latest `ui/streamlit_app.py` (the file was
  updated to use the non-blocking flow)
→ Restart Streamlit: `streamlit run ui/streamlit_app.py`

---

### MPS errors / `bfloat16` not supported
MPS on Apple Silicon does not support `bfloat16`.
The `mac_mps` profile automatically uses `float16` and calls
`fix_mps_dtypes()` before every inference pass.
If you see dtype errors, confirm `OMNI_PROFILE=mac_mps` in your `.env`.

---

### `qwen_omni_utils` not found
```
ModuleNotFoundError: No module named 'qwen_omni_utils'
```
→ `qwen-omni-utils` is not on PyPI. Install from source:
```bash
pip install git+https://github.com/QwenLM/qwen-omni-utils.git
```

---

### Audio output files accumulate on disk
Response WAV files are written to `audio/output/` and are never
automatically deleted. Clean up manually:
```bash
rm audio/output/response_*.wav
```
Or add a cron / scheduled task if running long-term.

---

## 10. Environment Variables — Quick Reference

```bash
# Device
OMNI_PROFILE=mac_mps          # mac_mps | cuda_prod | cpu

# Model location
OMNI_MODELS_DIR=./models
OMNI_MODEL_SUBDIR=Qwen2.5-Omni-3B

# Generation
OMNI_RETURN_AUDIO=true
OMNI_SPEAKER=Ethan             # Ethan | Chelsie
OMNI_THINKER_MAX_NEW_TOKENS=512

# API server
OMNI_API_HOST=0.0.0.0
OMNI_API_PORT=8000
OMNI_API_RELOAD=false          # true = hot reload (dev only)
```

All variables are also readable/settable from a `.env` file in the `v2/`
working directory. See `.env.example` for a fully annotated template.
