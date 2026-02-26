"""
streamlit_app.py — Omni Voice Assistant UI

Connects to the v2 FastAPI backend to:
  1. Capture microphone audio in the browser
  2. POST the recording to POST /infer
  3. Display the text response
  4. Play the audio response inline

Run (from v2/):
    streamlit run ui/streamlit_app.py

Backend must be running in a separate terminal:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

No changes are needed to any v2 backend code.
"""

from __future__ import annotations

import io
import time

import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Omni Voice Assistant",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────

if "history" not in st.session_state:
    # Each entry: {question_audio, text, audio_bytes, latency_s, model, wall_s, prompt}
    st.session_state.history: list[dict] = []

if "last_audio_key" not in st.session_state:
    # Used to detect when a new recording has arrived from the mic component
    st.session_state.last_audio_key: bytes | None = None

if "saved_recordings" not in st.session_state:
    # Bank of recordings kept for the whole session.
    # Each entry: {"label": str, "audio_bytes": bytes}
    st.session_state.saved_recordings: list[dict] = []

if "active_audio" not in st.session_state:
    # The audio currently queued to be sent — either a fresh mic recording
    # or one loaded from the saved bank.
    st.session_state.active_audio: bytes | None = None

if "active_audio_label" not in st.session_state:
    st.session_state.active_audio_label: str = ""

# ── Pending inference job ─────────────────────────────────────────────────────
# Populated when a job is submitted; cleared when the result is received.

if "job_id" not in st.session_state:
    st.session_state.job_id: str | None = None

if "job_start_time" not in st.session_state:
    st.session_state.job_start_time: float = 0.0

# Snapshot of the audio + prompt at submission time so we can save to history
if "job_audio" not in st.session_state:
    st.session_state.job_audio: bytes | None = None

if "job_audio_label" not in st.session_state:
    st.session_state.job_audio_label: str = ""

if "job_system_prompt" not in st.session_state:
    st.session_state.job_system_prompt: str = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️  Settings")

    backend_url: str = st.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="Where uvicorn is serving `app.main:app`.",
    )

    st.markdown("---")

    system_prompt: str = st.text_area(
        "System Prompt",
        value="",
        placeholder="Leave blank to use the server default.",
        help=(
            "Override the model's system prompt for this session. "
            "NOTE: Qwen2.5-Omni requires the default prompt wording "
            "to generate audio — only change this for text-only mode."
        ),
    )

    return_audio: bool = st.toggle(
        "Return audio response",
        value=True,
        help="Disable for faster text-only responses.",
    )

    st.markdown("---")

    # ── Backend health check ───────────────────────────────────────────────
    status_box = st.empty()

    if st.button("🔄  Check backend", use_container_width=True):
        try:
            r = requests.get(f"{backend_url}/health", timeout=3)
            r.raise_for_status()
            h = r.json()
            model_name = h["model_path"].split("/")[-1]
            if h.get("model_ready"):
                status_box.success(
                    f"✅  Model ready\n\n"
                    f"**{model_name}**\n\n"
                    f"Profile: `{h['profile']}`"
                )
            else:
                status_box.warning("⚠️  Backend is up but model is still loading…")
        except requests.exceptions.ConnectionError:
            status_box.error("❌  Cannot reach backend.")
        except Exception as exc:
            status_box.error(f"❌  {exc}")

    st.markdown("---")

    if st.button("🗑️  Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_audio_key = None
        st.rerun()

# ── Main layout ────────────────────────────────────────────────────────────────

st.title("🎙️  Omni Voice Assistant")
st.caption("Powered by Qwen2.5-Omni · Speak your question, hear the answer")

st.markdown("---")

# ── Step 1 · Record ────────────────────────────────────────────────────────────

st.subheader("1 · Record your question")
st.markdown(
    "Press **🎙️** to start recording. Press again (or wait for silence) to stop."
)

audio_bytes: bytes | None = audio_recorder(
    text="",
    recording_color="#e74c3c",
    neutral_color="#2196F3",
    icon_name="microphone",
    icon_size="3x",
    pause_threshold=2.5,   # auto-stop after 2.5 s of silence
    sample_rate=16_000,    # 16 kHz — matches Qwen2.5-Omni expected input
)

# ── Auto-save any brand-new mic recording to the session bank ─────────────────
if audio_bytes and audio_bytes != st.session_state.last_audio_key:
    st.session_state.last_audio_key = audio_bytes
    n     = len(st.session_state.saved_recordings) + 1
    label = f"Recording {n}"
    st.session_state.saved_recordings.append(
        {"label": label, "audio_bytes": audio_bytes}
    )
    # Make the new recording the active one automatically
    st.session_state.active_audio       = audio_bytes
    st.session_state.active_audio_label = f"🎙️ {label} (just recorded)"
    st.rerun()

# ── Saved recordings bank ─────────────────────────────────────────────────────
if st.session_state.saved_recordings:
    with st.expander(
        f"📁  Saved recordings ({len(st.session_state.saved_recordings)}) "
        "— load any to re-send with a different system prompt",
        expanded=False,
    ):
        options   = [r["label"] for r in st.session_state.saved_recordings]
        # Default selection: whichever is currently active, else the latest
        try:
            default_idx = next(
                i for i, r in enumerate(st.session_state.saved_recordings)
                if r["audio_bytes"] == st.session_state.active_audio
            )
        except StopIteration:
            default_idx = len(options) - 1

        selected_label = st.selectbox(
            "Choose a recording", options, index=default_idx, label_visibility="collapsed"
        )
        sel_idx   = options.index(selected_label)
        sel_audio = st.session_state.saved_recordings[sel_idx]["audio_bytes"]

        st.audio(sel_audio, format="audio/wav")

        col_load, col_del = st.columns([3, 1])
        with col_load:
            if st.button("↩️  Load as active", use_container_width=True, key="btn_load"):
                st.session_state.active_audio       = sel_audio
                st.session_state.active_audio_label = f"📁  {selected_label}"
                st.rerun()
        with col_del:
            if st.button("🗑️  Delete", use_container_width=True, key="btn_del_rec"):
                st.session_state.saved_recordings.pop(sel_idx)
                # Re-number remaining recordings so labels stay clean
                for i, rec in enumerate(st.session_state.saved_recordings, start=1):
                    rec["label"] = f"Recording {i}"
                # If the deleted one was active, clear active audio
                if st.session_state.active_audio == sel_audio:
                    st.session_state.active_audio       = None
                    st.session_state.active_audio_label = ""
                st.rerun()

# ── Active audio preview + send ───────────────────────────────────────────────
if st.session_state.active_audio:
    st.markdown(f"**Active:** {st.session_state.active_audio_label}")
    st.audio(st.session_state.active_audio, format="audio/wav")

    col_send, col_discard = st.columns([3, 1])

    with col_send:
        send_clicked = st.button(
            "🚀  Send to Omni",
            type="primary",
            use_container_width=True,
            disabled=bool(st.session_state.job_id),  # locked while job is running
        )

    with col_discard:
        if st.button("✖  Deselect", use_container_width=True):
            st.session_state.active_audio       = None
            st.session_state.active_audio_label = ""
            st.rerun()

    # ── Step 2 · Submit inference job (non-blocking) ───────────────────────
    if send_clicked:
        audio_to_send = st.session_state.active_audio
        files = {"audio": ("question.wav", io.BytesIO(audio_to_send), "audio/wav")}
        data: dict[str, str] = {"return_audio": "true" if return_audio else "false"}
        if system_prompt.strip():
            data["system_prompt"] = system_prompt.strip()

        try:
            resp = requests.post(
                f"{backend_url}/infer/submit",
                files=files,
                data=data,
                timeout=30,  # only the upload — inference runs in background
            )
            resp.raise_for_status()
            st.session_state.job_id           = resp.json()["job_id"]
            st.session_state.job_start_time   = time.perf_counter()
            st.session_state.job_audio        = audio_to_send
            st.session_state.job_audio_label  = st.session_state.active_audio_label
            st.session_state.job_system_prompt = system_prompt.strip() or "(server default)"
            st.rerun()
        except requests.exceptions.HTTPError as exc:
            st.error(f"Submit failed ({exc.response.status_code}): {exc.response.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌  Cannot reach the backend.\n\nMake sure `uvicorn app.main:app --port 8000` is running.")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

else:
    st.info("🎤  Press the microphone button above to record your question.")

st.markdown("---")

# ── Step 2 · Poll for inference result ────────────────────────────────────────
# This section re-renders every 5 s (via st.rerun) until the job finishes.
# Because the blocking work is in a FastAPI background thread, Streamlit's
# WebSocket to the browser stays alive the whole time.

if st.session_state.job_id:
    elapsed  = time.perf_counter() - st.session_state.job_start_time
    # Rough progress bar — model typically finishes in 500-700 s
    progress = min(elapsed / 650, 0.98)
    st.subheader("⏳  Inference in progress")
    st.progress(progress, text=f"{elapsed:.0f}s elapsed — model typically takes 500–700s")

    try:
        poll = requests.get(
            f"{backend_url}/infer/poll/{st.session_state.job_id}",
            timeout=10,
        )
        poll.raise_for_status()
        job = poll.json()

        if job["status"] == "pending":
            time.sleep(5)   # wait 5 s, then rerun to re-poll
            st.rerun()

        elif job["status"] == "done":
            result   = job["result"]
            wall_s   = time.perf_counter() - st.session_state.job_start_time

            # ── Fetch response audio from storage ─────────────────────────
            response_audio_bytes: bytes | None = None
            if not result.get("audio_path"):
                if return_audio:
                    st.warning(
                        "⚠️  Backend returned no audio path. "
                        "Check that `OMNI_RETURN_AUDIO` is not `false` on the server."
                    )
            elif return_audio:
                audio_url = f"{backend_url}/audio/{result['audio_path']}"
                ar = requests.get(audio_url, timeout=30)
                if ar.ok:
                    response_audio_bytes = ar.content
                else:
                    st.error(
                        f"❌  Audio fetch failed — HTTP {ar.status_code} "
                        f"for `{audio_url}`\n\n{ar.text[:300]}"
                    )

            # ── Save to history ────────────────────────────────────────────
            st.session_state.history.append({
                "question_audio": st.session_state.job_audio,
                "question_label": st.session_state.job_audio_label,
                "system_prompt":  st.session_state.job_system_prompt,
                "text":           result["text"],
                "audio_bytes":    response_audio_bytes,
                "latency_s":      result["latency_s"],
                "model":          result["model"],
                "wall_s":         wall_s,
            })

            # ── Clear job state ────────────────────────────────────────────
            st.session_state.job_id         = None
            st.session_state.job_audio      = None
            st.session_state.job_audio_label  = ""
            st.session_state.job_system_prompt = ""
            st.rerun()

        elif job["status"] == "error":
            st.error(f"❌  Inference failed: {job['error']}")
            st.session_state.job_id = None

    except requests.exceptions.ConnectionError:
        st.error("❌  Lost connection to backend while polling.")
        time.sleep(5)
        st.rerun()
    except Exception as exc:
        st.error(f"Polling error: {exc}")
        time.sleep(5)
        st.rerun()

st.markdown("---")

# ── Step 3 · Conversation history ─────────────────────────────────────────────

if st.session_state.history:
    st.subheader("💬  Conversation")

    # Display most-recent turn first
    for idx, turn in enumerate(reversed(st.session_state.history)):
        turn_number = len(st.session_state.history) - idx
        is_latest   = idx == 0

        with st.expander(f"Turn {turn_number}", expanded=is_latest):
            # ── Your question ──────────────────────────────────────────────
            q_label = turn.get("question_label", "")
            st.markdown(f"**🧑  You:** _{q_label}_")
            st.audio(turn["question_audio"], format="audio/wav")
            st.caption(f"System prompt: `{turn.get('system_prompt', '(server default)')}`")

            # ── Model text response ────────────────────────────────────────
            st.markdown(f"**🤖  Omni:** {turn['text']}")

            # ── Model audio response ───────────────────────────────────────
            if turn["audio_bytes"]:
                st.markdown("**🔊  Listen:**")
                st.audio(
                    turn["audio_bytes"],
                    format="audio/wav",
                    autoplay=is_latest,   # auto-play only the newest response
                )
            elif return_audio:
                st.caption("_(no audio returned for this turn)_")

            # ── Metadata ───────────────────────────────────────────────────
            st.caption(
                f"⏱️  Model: {turn['latency_s']:.2f}s · "
                f"Wall: {turn['wall_s']:.2f}s · "
                f"Model: `{turn['model']}`"
            )
