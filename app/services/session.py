"""
session.py — ConversationSession manager for multi-turn inference.

Sessions live on the local filesystem under audio/sessions/{conversation_id}/:

    audio/sessions/
      {conversation_id}/
        session.json          ← metadata: turn list, timestamps
        turn_1_input.wav      ← saved input audio for turn 1
        turn_2_input.wav
        ...

Each conversation_id is a 16-hex-char UUID fragment, collision-resistant
for local single/small-team use.  There is no authentication — whoever
holds the conversation_id owns that session.

Lifecycle:
  1. create_session()              → conversation_id
  2. save_input_audio(id, bytes)   → (Path, turn_index, duration_s)
                                     called at job-submit time
  3. finalize_turn(id, idx, text, dur) → saved to session.json
                                     called when inference completes
  4. delete_turn(id, idx)          → user-initiated prune
  5. cleanup_expired_sessions()    → called at server startup

Multi-user isolation: every read/write is namespaced by conversation_id.
Two users with different IDs cannot see each other's files.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import soundfile as sf

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ── Value objects ──────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """
    A single completed turn stored in session.json.

    audio_duration_s is used for the context-budget meter in the UI.
    audio_tokens is estimated as  duration_s × audio_tokens_per_second.
    """
    turn_index:       int
    input_audio_path: str    # absolute path — must stay on disk for the session lifetime
    response_text:    str
    audio_duration_s: float
    audio_tokens:     int
    created_at:       float = field(default_factory=time.time)


@dataclass
class ConversationSession:
    conversation_id: str
    turns:           list[ConversationTurn] = field(default_factory=list)
    created_at:      float                  = field(default_factory=time.time)
    last_updated:    float                  = field(default_factory=time.time)


# ── SessionManager ─────────────────────────────────────────────────────────────

class SessionManager:
    """
    Manages per-conversation state for multi-turn inference.
    All state is stored on the local filesystem — no external services needed.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg = settings or get_settings()
        self._cfg.sessions_dir.mkdir(parents=True, exist_ok=True)

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def create_session(self) -> str:
        """Create a new empty session. Returns the conversation_id."""
        conversation_id = uuid.uuid4().hex[:16]
        session = ConversationSession(conversation_id=conversation_id)
        self._save_session(session)
        logger.info("New conversation session: %s", conversation_id)
        return conversation_id

    def get_session(self, conversation_id: str) -> Optional[ConversationSession]:
        """Load a session from disk. Returns None if not found."""
        meta_path = self._meta_path(conversation_id)
        if not meta_path.exists():
            return None
        try:
            data  = json.loads(meta_path.read_text())
            turns = [ConversationTurn(**t) for t in data.get("turns", [])]
            return ConversationSession(
                conversation_id = data["conversation_id"],
                turns           = turns,
                created_at      = data["created_at"],
                last_updated    = data["last_updated"],
            )
        except Exception as exc:
            logger.warning("Failed to load session %s: %s", conversation_id, exc)
            return None

    # ── Turn management ────────────────────────────────────────────────────────

    def save_input_audio(
        self,
        conversation_id: str,
        audio_bytes:     bytes,
    ) -> tuple[Path, int, float]:
        """
        Write the current-turn input audio to the session directory.

        Called at job-submit time so the path is available before inference
        starts.  The audio file must NOT be deleted by the inference job
        (unlike single-turn temp files).

        Returns:
            (audio_path, turn_index, audio_duration_s)
            turn_index is the pending turn number (completed turns + 1).
        """
        session = self.get_session(conversation_id)
        if session is None:
            raise ValueError(f"Session not found: {conversation_id}")

        turn_index = len(session.turns) + 1
        session_dir = self._session_dir(conversation_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        audio_path = session_dir / f"turn_{turn_index}_input.wav"
        audio_path.write_bytes(audio_bytes)

        # Measure duration using soundfile
        try:
            info       = sf.info(str(audio_path))
            duration_s = float(info.duration)
        except Exception:
            # Fallback: rough estimate from raw byte size (float32 mono, 16kHz)
            duration_s = len(audio_bytes) / (4 * 16000)

        logger.debug(
            "Session %s: saved turn %d input (%.2fs)",
            conversation_id, turn_index, duration_s,
        )
        return audio_path, turn_index, duration_s

    def finalize_turn(
        self,
        conversation_id: str,
        turn_index:      int,
        response_text:   str,
        audio_duration_s: float,
    ) -> ConversationTurn:
        """
        Record a completed inference turn in session.json.
        Called by the router after the background job succeeds.
        """
        session = self.get_session(conversation_id)
        if session is None:
            raise ValueError(f"Session not found: {conversation_id}")

        audio_tokens = int(audio_duration_s * self._cfg.audio_tokens_per_second)
        audio_path   = self._session_dir(conversation_id) / f"turn_{turn_index}_input.wav"

        turn = ConversationTurn(
            turn_index       = turn_index,
            input_audio_path = str(audio_path),
            response_text    = response_text,
            audio_duration_s = audio_duration_s,
            audio_tokens     = audio_tokens,
        )
        session.turns.append(turn)
        session.last_updated = time.time()
        self._save_session(session)

        logger.info(
            "Session %s: finalized turn %d (%d tokens)",
            conversation_id, turn_index, audio_tokens,
        )
        return turn

    def delete_turn(self, conversation_id: str, turn_index: int) -> bool:
        """
        Remove a turn from the session. Deletes the associated audio file.
        Returns True if the turn was found and removed, False otherwise.

        Note: turn indices of remaining turns are NOT renumbered — the
        session uses stable indices so the UI can reference them by ID.
        """
        session = self.get_session(conversation_id)
        if session is None:
            return False

        original_len = len(session.turns)
        session.turns = [t for t in session.turns if t.turn_index != turn_index]
        if len(session.turns) == original_len:
            return False   # turn not found

        # Remove audio file
        audio_path = self._session_dir(conversation_id) / f"turn_{turn_index}_input.wav"
        audio_path.unlink(missing_ok=True)

        session.last_updated = time.time()
        self._save_session(session)

        logger.info("Session %s: deleted turn %d", conversation_id, turn_index)
        return True

    def get_history(self, conversation_id: str) -> list:
        """
        Return the session's completed turns as HistoryTurn objects
        ready to be passed directly to ModelBackend.process(history=...).

        Returns [] if the session does not exist.
        """
        from app.domain.ports import HistoryTurn

        session = self.get_session(conversation_id)
        if session is None:
            return []

        return [
            HistoryTurn(
                input_audio_path = t.input_audio_path,
                response_text    = t.response_text,
            )
            for t in session.turns
            if Path(t.input_audio_path).exists()   # skip if file was deleted
        ]

    # ── Context budget ─────────────────────────────────────────────────────────

    def max_audio_tokens(self) -> int:
        """Token budget available for audio history (context window minus overhead)."""
        return (
            self._cfg.context_window_tokens
            - self._cfg.system_prompt_reserve_tokens
            - self._cfg.thinker_max_new_tokens
        )

    def max_audio_minutes(self) -> float:
        """Maximum audio history in minutes before the context window is full."""
        return self.max_audio_tokens() / (self._cfg.audio_tokens_per_second * 60)

    def context_stats(self, conversation_id: str) -> dict:
        """
        Return context-budget statistics for the UI meter.

        {
          "used_tokens":    int,
          "max_tokens":     int,
          "used_minutes":   float,
          "max_minutes":    float,
          "context_used_pct": float,   # 0.0 – 1.0
          "turns":          list[dict] — each with turn_index, audio_duration_s,
                                         audio_tokens, response_preview
        }
        """
        session = self.get_session(conversation_id)
        if session is None:
            return {
                "used_tokens": 0, "max_tokens": self.max_audio_tokens(),
                "used_minutes": 0.0, "max_minutes": self.max_audio_minutes(),
                "context_used_pct": 0.0, "turns": [],
            }

        used_tokens  = sum(t.audio_tokens     for t in session.turns)
        used_minutes = sum(t.audio_duration_s for t in session.turns) / 60
        max_tokens   = self.max_audio_tokens()
        max_minutes  = self.max_audio_minutes()

        return {
            "used_tokens":      used_tokens,
            "max_tokens":       max_tokens,
            "used_minutes":     round(used_minutes, 2),
            "max_minutes":      round(max_minutes, 2),
            "context_used_pct": round(min(used_tokens / max(max_tokens, 1), 1.0), 4),
            "turns": [
                {
                    "turn_index":       t.turn_index,
                    "audio_duration_s": round(t.audio_duration_s, 2),
                    "audio_tokens":     t.audio_tokens,
                    "response_preview": t.response_text[:120],
                }
                for t in session.turns
            ],
        }

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def cleanup_expired_sessions(self) -> int:
        """
        Delete session directories older than OMNI_SESSION_TTL_HOURS.
        Safe to call at server startup — does not affect active sessions.
        Returns the number of sessions deleted.
        """
        cutoff  = time.time() - (self._cfg.session_ttl_hours * 3600)
        deleted = 0

        if not self._cfg.sessions_dir.exists():
            return 0

        for session_dir in self._cfg.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            meta_path = session_dir / "session.json"

            if not meta_path.exists():
                # Orphaned directory — clean up
                shutil.rmtree(session_dir, ignore_errors=True)
                deleted += 1
                continue

            try:
                data = json.loads(meta_path.read_text())
                if data.get("last_updated", 0) < cutoff:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    deleted += 1
            except Exception:
                pass   # leave malformed sessions alone

        if deleted:
            logger.info("Session cleanup: removed %d expired sessions", deleted)
        return deleted

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _session_dir(self, conversation_id: str) -> Path:
        return self._cfg.sessions_dir / conversation_id

    def _meta_path(self, conversation_id: str) -> Path:
        return self._session_dir(conversation_id) / "session.json"

    def _save_session(self, session: ConversationSession) -> None:
        session_dir = self._session_dir(session.conversation_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "conversation_id": session.conversation_id,
            "turns":           [asdict(t) for t in session.turns],
            "created_at":      session.created_at,
            "last_updated":    session.last_updated,
        }
        (session_dir / "session.json").write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
