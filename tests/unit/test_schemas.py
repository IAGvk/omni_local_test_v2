"""
test_schemas.py — Unit tests for Pydantic schemas in app/domain/schemas.py

Tests serialisation, defaults, and field validation.
No model needed.
"""

from __future__ import annotations

import json

import pytest

from app.domain.schemas import (
    HealthResponse,
    InferResponse,
    WSChunk,
    WSError,
    WSInferRequest,
)


class TestInferResponse:
    def test_required_fields(self):
        r = InferResponse(text="hello", latency_s=1.5, model="Qwen2.5-Omni-3B")
        assert r.text == "hello"
        assert r.latency_s == 1.5
        assert r.model == "Qwen2.5-Omni-3B"

    def test_audio_path_defaults_none(self):
        r = InferResponse(text="x", latency_s=0.0, model="m")
        assert r.audio_path is None

    def test_audio_path_set(self):
        r = InferResponse(text="x", audio_path="response_123.wav", latency_s=1.0, model="m")
        assert r.audio_path == "response_123.wav"

    def test_json_serialisable(self):
        r    = InferResponse(text="hi", latency_s=0.5, model="m")
        data = json.loads(r.model_dump_json())
        assert data["text"] == "hi"
        assert data["audio_path"] is None


class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse(model_ready=True, model_path="/models/Qwen", profile="mac_mps")
        assert h.status == "ok"

    def test_not_ready(self):
        h = HealthResponse(model_ready=False, model_path="/models/Qwen", profile="mac_mps")
        assert h.model_ready is False


class TestWSInferRequest:
    def test_defaults(self):
        r = WSInferRequest()
        assert r.action      == "infer"
        assert r.sample_rate == 16000
        assert r.return_audio is True
        assert r.system_prompt is None

    def test_custom_prompt(self):
        r = WSInferRequest(system_prompt="You are a compliance bot.")
        assert "compliance" in r.system_prompt

    def test_parse_from_json(self):
        payload = '{"sample_rate": 8000, "return_audio": false}'
        r       = WSInferRequest(**json.loads(payload))
        assert r.sample_rate  == 8000
        assert r.return_audio is False


class TestWSChunk:
    def test_defaults(self):
        c = WSChunk()
        assert c.text     == ""
        assert c.is_final is False

    def test_final_chunk(self):
        c = WSChunk(text="Complete response.", is_final=True)
        assert c.is_final is True
        assert c.text == "Complete response."

    def test_json_round_trip(self):
        c    = WSChunk(text="hi", is_final=True)
        data = json.loads(c.model_dump_json())
        back = WSChunk(**data)
        assert back.text     == c.text
        assert back.is_final == c.is_final


class TestWSError:
    def test_error_field(self):
        e = WSError(error="Something went wrong")
        assert "wrong" in e.error

    def test_json_serialisable(self):
        e    = WSError(error="boom")
        data = json.loads(e.model_dump_json())
        assert data["error"] == "boom"
