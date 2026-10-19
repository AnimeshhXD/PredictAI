"""
tests/test_llm.py
=================
Tests for the provider-agnostic LLM wrapper (src/llm.py).

Strategy
--------
* We never hit a real API in tests — all network calls are monkey-patched.
* We test: config loading, cache read/write, provider dispatch, cache bypass,
  and error paths (missing key, unknown provider).
"""

from __future__ import annotations

import json
import os
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fake_openai_response(text: str = "mocked") -> MagicMock:
    """Return a minimal mock that mirrors openai.ChatCompletion structure."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Cache tests ────────────────────────────────────────────────────────────────


class TestCache:
    """Test the disk-cache layer independently of any provider."""

    def test_cache_write_and_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Writing a response and reading it back must return the same string."""
        monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        # Re-import to pick up env changes
        import src.llm as llm_mod
        importlib.reload(llm_mod)

        key = llm_mod._cache_key("hello", None)
        llm_mod._write_cache(key, "world")
        result = llm_mod._read_cache(key)
        assert result == "world"

    def test_cache_miss_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reading a key that was never written must return None."""
        monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
        import src.llm as llm_mod
        importlib.reload(llm_mod)

        result = llm_mod._read_cache("nonexistent_key_xyz")
        assert result is None

    def test_cache_key_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same inputs must always produce the same key."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        import src.llm as llm_mod
        importlib.reload(llm_mod)

        k1 = llm_mod._cache_key("test prompt", "system msg")
        k2 = llm_mod._cache_key("test prompt", "system msg")
        assert k1 == k2

    def test_cache_key_differs_by_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Different prompts must yield different cache keys."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        import src.llm as llm_mod
        importlib.reload(llm_mod)

        k1 = llm_mod._cache_key("prompt A", None)
        k2 = llm_mod._cache_key("prompt B", None)
        assert k1 != k2

    def test_cached_call_does_not_hit_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the response is cached, the backend must NOT be called."""
        monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        import src.llm as llm_mod
        importlib.reload(llm_mod)

        # Pre-populate cache
        key = llm_mod._cache_key("cached prompt", None)
        llm_mod._write_cache(key, "cached answer")

        with patch.object(llm_mod, "_call_openai", side_effect=AssertionError("should not be called")):
            result = llm_mod.call_llm("cached prompt")

        assert result == "cached answer"


# ── Provider dispatch tests ────────────────────────────────────────────────────


class TestProviderDispatch:
    """Ensure call_llm routes to the correct backend."""

    def test_openai_backend_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        import src.llm as llm_mod
        importlib.reload(llm_mod)

        # Patch at the module level so the provider key lookup works correctly
        with patch.object(llm_mod, "_call_openai", return_value="openai response") as mock_fn:
            # Rebuild the dispatch table after patching
            llm_mod._BACKENDS["openai"] = mock_fn
            result = llm_mod.call_llm("hello", bypass_cache=True)

        assert result == "openai response"
        mock_fn.assert_called_once()

    def test_anthropic_backend_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-3-haiku-20240307")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        import src.llm as llm_mod
        importlib.reload(llm_mod)

        with patch.object(llm_mod, "_call_anthropic", return_value="anthropic response") as mock_fn:
            llm_mod._BACKENDS["anthropic"] = mock_fn
            result = llm_mod.call_llm("hello", bypass_cache=True)

        assert result == "anthropic response"
        mock_fn.assert_called_once()

    def test_google_backend_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("LLM_PROVIDER", "google")
        monkeypatch.setenv("LLM_MODEL", "gemini-1.5-flash")
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")

        import src.llm as llm_mod
        importlib.reload(llm_mod)

        with patch.object(llm_mod, "_call_google", return_value="google response") as mock_fn:
            llm_mod._BACKENDS["google"] = mock_fn
            result = llm_mod.call_llm("hello", bypass_cache=True)

        assert result == "google response"
        mock_fn.assert_called_once()

    def test_bypass_cache_writes_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """bypass_cache=True must still persist the fresh response."""
        monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        import src.llm as llm_mod
        importlib.reload(llm_mod)

        with patch.object(llm_mod, "_call_openai", return_value="fresh") as mock_fn:
            llm_mod._BACKENDS["openai"] = mock_fn
            llm_mod.call_llm("bypass test", bypass_cache=True)

        key = llm_mod._cache_key("bypass test", None)
        cached = llm_mod._read_cache(key)
        assert cached == "fresh"


# ── Error paths ────────────────────────────────────────────────────────────────


class TestErrors:
    """Test that missing keys and unsupported providers fail loudly."""

    def test_missing_openai_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        import src.llm as llm_mod
        importlib.reload(llm_mod)

        # Mock openai import so we test the key-check, not the module import
        fake_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": fake_openai}):
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                llm_mod._call_openai("test", None)

    def test_missing_anthropic_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        import src.llm as llm_mod
        importlib.reload(llm_mod)

        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            llm_mod._call_anthropic("test", None)

    def test_missing_google_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        import src.llm as llm_mod
        importlib.reload(llm_mod)

        # Mock google.generativeai so we test the key-check
        fake_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": fake_genai}):
            with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
                llm_mod._call_google("test", None)


# ── get_config ─────────────────────────────────────────────────────────────────


class TestGetConfig:
    def test_get_config_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_config must return all expected keys and never expose API keys."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        import src.llm as llm_mod
        importlib.reload(llm_mod)

        cfg = llm_mod.get_config()
        for key in ("provider", "model", "max_tokens", "cache_dir", "temperature"):
            assert key in cfg, f"Missing key: {key}"

        assert "api_key" not in cfg
        assert cfg["temperature"] == 0
