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
    # Each entry: {question_audio, text, audio_bytes, latency_s, model, wall_s}
    st.session_state.history: list[dict] = []

if "last_audio_key" not in st.session_state:
    # Used to detect when a new recording has been made
    st.session_state.last_audio_key: bytes | None = None

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

if audio_bytes:
    # Detect a fresh recording (audio_recorder re-returns same bytes on reruns)
    is_new_recording = audio_bytes != st.session_state.last_audio_key

    st.audio(audio_bytes, format="audio/wav")
    st.caption("✅  Recorded — review above, then send.")

    col_send, col_discard = st.columns([3, 1])

    with col_send:
        send_clicked = st.button(
            "🚀  Send to Omni",
            type="primary",
            use_container_width=True,
            disabled=not is_new_recording,
        )

    with col_discard:
        if st.button("✖  Discard", use_container_width=True):
            # Mark this audio as "seen" so the send button re-disables
            st.session_state.last_audio_key = audio_bytes
            st.rerun()

    # ── Step 2 · Inference ─────────────────────────────────────────────────
    if send_clicked and is_new_recording:
        st.session_state.last_audio_key = audio_bytes   # mark as consumed

        with st.spinner("Running inference — this may take a moment…"):
            try:
                files = {
                    "audio": ("question.wav", io.BytesIO(audio_bytes), "audio/wav")
                }
                data: dict[str, str] = {
                    "return_audio": "true" if return_audio else "false",
                }
                if system_prompt.strip():
                    data["system_prompt"] = system_prompt.strip()

                t0 = time.perf_counter()
                resp = requests.post(
                    f"{backend_url}/infer",
                    files=files,
                    data=data,
                    timeout=180,   # model inference can be slow on CPU / first run
                )
                resp.raise_for_status()
                result: dict = resp.json()
                wall_s = time.perf_counter() - t0

                # ── Fetch response audio ───────────────────────────────────
                response_audio_bytes: bytes | None = None
                if result.get("audio_path") and return_audio:
                    audio_url = f"{backend_url}/audio/{result['audio_path']}"
                    ar = requests.get(audio_url, timeout=30)
                    if ar.ok:
                        response_audio_bytes = ar.content

                # ── Persist in session history ─────────────────────────────
                st.session_state.history.append(
                    {
                        "question_audio": audio_bytes,
                        "text":           result["text"],
                        "audio_bytes":    response_audio_bytes,
                        "latency_s":      result["latency_s"],
                        "model":          result["model"],
                        "wall_s":         wall_s,
                    }
                )
                st.rerun()   # re-render so the history section shows the new turn

            except requests.exceptions.HTTPError as exc:
                st.error(
                    f"Backend error ({exc.response.status_code}): "
                    f"{exc.response.text}"
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌  Cannot reach the backend.\n\n"
                    "Make sure `uvicorn app.main:app --port 8000` is running."
                )
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

else:
    st.info("🎤  Press the microphone button above to record your question.")

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
            st.markdown("**🧑  You:**")
            st.audio(turn["question_audio"], format="audio/wav")

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
