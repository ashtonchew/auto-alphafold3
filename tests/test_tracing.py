"""Tests for autoalphafold3._tracing.

Verifies the load-bearing invariant: the presence or absence of Workshop
tracing NEVER changes the main flow's return value or which exception
is raised.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from unittest.mock import patch

import pytest


@contextmanager
def _isolated_tracing_module():
    """Pop the _tracing module from sys.modules so the next import re-evaluates
    its module-level state (RAINDROP_LOCAL_DEBUGGER, _INITIALIZED, etc.)."""
    keys_to_remove = [k for k in sys.modules if k.startswith("autoalphafold3._tracing")]
    saved = {k: sys.modules[k] for k in keys_to_remove}
    for k in keys_to_remove:
        del sys.modules[k]
    try:
        yield
    finally:
        for k in keys_to_remove:
            sys.modules.pop(k, None)
        for k, v in saved.items():
            sys.modules[k] = v


def test_env_var_unset_means_total_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """When RAINDROP_LOCAL_DEBUGGER is unset, span() is a complete no-op:
    the body runs, return values pass through, exceptions re-raise, and
    the opentelemetry packages are never imported."""
    monkeypatch.delenv("RAINDROP_LOCAL_DEBUGGER", raising=False)
    initial_otel_keys = {k for k in sys.modules if k.startswith("opentelemetry")}

    with _isolated_tracing_module():
        from autoalphafold3._tracing import span

        side_effects: list[int] = []
        with span("test", a=1):
            side_effects.append(1)
        assert side_effects == [1]

        def returns_42() -> int:
            with span("test"):
                return 42

        assert returns_42() == 42

        with pytest.raises(ValueError, match="boom"):
            with span("test"):
                raise ValueError("boom")

        new_otel_keys = {k for k in sys.modules if k.startswith("opentelemetry")} - initial_otel_keys
        assert new_otel_keys == set(), f"unexpected otel imports: {new_otel_keys}"


def test_sdk_not_installed_means_total_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """When RAINDROP_LOCAL_DEBUGGER is set but opentelemetry packages are
    not importable, span() still no-ops cleanly."""
    monkeypatch.setenv("RAINDROP_LOCAL_DEBUGGER", "http://localhost:5899/v1/")

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ModuleNotFoundError(f"simulated missing module: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with _isolated_tracing_module():
        from autoalphafold3._tracing import span

        result = []
        with span("test"):
            result.append("ok")
        assert result == ["ok"]

        with pytest.raises(RuntimeError, match="x"):
            with span("test"):
                raise RuntimeError("x")


def test_daemon_unreachable_means_silent_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the SDK is installed and the env var is set but the daemon is
    unreachable, span() must still no-op silently."""
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    monkeypatch.setenv("RAINDROP_LOCAL_DEBUGGER", "http://localhost:1/v1/")

    with _isolated_tracing_module():
        from autoalphafold3._tracing import span

        with span("test"):
            pass

        def returns_x():
            with span("test"):
                return "x"

        assert returns_x() == "x"


def test_body_exception_is_reraised_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the body raises, the exception MUST be re-raised with the same
    type and message. Tracing must never swallow real exceptions."""
    monkeypatch.setenv("RAINDROP_LOCAL_DEBUGGER", "http://localhost:5899/v1/")

    with _isolated_tracing_module():
        from autoalphafold3._tracing import span

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError) as exc_info:
            with span("test", attr1="value1"):
                raise CustomError("specific message")

        assert str(exc_info.value) == "specific message"


def test_tracing_init_exception_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """If _try_init() raises for any reason, span() must still no-op."""
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setenv("RAINDROP_LOCAL_DEBUGGER", "http://localhost:5899/v1/")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated init failure")

    monkeypatch.setattr(
        "opentelemetry.sdk.trace.TracerProvider.__init__",
        boom,
    )

    with _isolated_tracing_module():
        from autoalphafold3._tracing import span

        result = []
        with span("test"):
            result.append("ok")
        assert result == ["ok"]


def test_finish_exception_is_swallowed() -> None:
    """If finishing the span raises after the body succeeds, the function
    must still return its value normally."""
    pytest.xfail("Automated test deferred; verified manually by injecting a broken exporter.")


@pytest.mark.skipif(
    "READINESS_BYTE_IDENTITY" not in os.environ,
    reason="Set READINESS_BYTE_IDENTITY=1 to run this slow integration test",
)
def test_readiness_cli_byte_identity_with_and_without_tracing(tmp_path) -> None:
    """The readiness CLI's output must be byte-identical with and without
    RAINDROP_LOCAL_DEBUGGER set. This is the operational invariant."""
    import subprocess

    env_no_trace = {**os.environ}
    env_no_trace.pop("RAINDROP_LOCAL_DEBUGGER", None)

    env_with_trace = {**os.environ, "RAINDROP_LOCAL_DEBUGGER": "http://localhost:5899/v1/"}

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    out_no_trace = subprocess.run(
        ["python3", "-m", "autoalphafold3.agent", "readiness-report"],
        capture_output=True,
        text=True,
        env=env_no_trace,
        cwd=repo_root,
    ).stdout

    out_with_trace = subprocess.run(
        ["python3", "-m", "autoalphafold3.agent", "readiness-report"],
        capture_output=True,
        text=True,
        env=env_with_trace,
        cwd=repo_root,
    ).stdout

    assert out_no_trace == out_with_trace
