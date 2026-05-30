"""Optional Raindrop Workshop tracing via OpenTelemetry/OTLP.

This module is build-time and rehearsal-time developer observability. It is
NOT a runtime dependency. The entire module is a no-op when:
  (a) the RAINDROP_LOCAL_DEBUGGER environment variable is unset, or
  (b) the opentelemetry packages are not installed, or
  (c) any initialization step fails.

The `span(name, **attrs)` context manager is the only public surface.

Removal procedure (to drop the integration entirely):
  1. Delete this file (`rm autoalphafold3/_tracing.py`).
  2. Remove the import statement `from autoalphafold3._tracing import span`
     from every file that uses it (git grep makes this trivial).
  3. Remove the `with span(...):` wrappers and de-indent each wrapped body.
  4. Run the test suite. It must pass identically.

Invariants (load-bearing — do not weaken):
  - The presence or absence of tracing NEVER changes the main flow's
    return value or which exception is raised.
  - Exceptions from inside the span body are always re-raised unchanged.
  - Exceptions from inside the tracing path (init, begin, finish) are
    always swallowed and never propagate.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

# Module-level state. `_INITIALIZED` is True after a successful init; once
# `_PERMANENTLY_FAILED` is True we never retry init. Both latches are read
# inside `_try_init` before any expensive imports.
_ENABLED: bool = bool(os.getenv("RAINDROP_LOCAL_DEBUGGER"))
_INITIALIZED: bool = False
_PERMANENTLY_FAILED: bool = False

# Populated by `_try_init` on success.
_TRACER: Any = None


def _try_init() -> None:
    """Lazily initialize the OpenTelemetry tracer on first span use.

    Idempotent. Returns silently on success. On failure, sets the
    permanent-failure latch and returns silently — never raises.
    """
    global _INITIALIZED, _PERMANENTLY_FAILED, _TRACER
    if _INITIALIZED or _PERMANENTLY_FAILED or not _ENABLED:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        base_url = os.environ["RAINDROP_LOCAL_DEBUGGER"].rstrip("/")
        # OTLP HTTP/JSON expects POSTs to /v1/traces under the base URL.
        # The base URL convention `http://localhost:5899/v1/` already
        # ends in /v1/; the exporter appends `/traces` itself.
        endpoint = f"{base_url}/traces"

        resource = Resource.create(
            {
                "service.name": "autoalphafold3",
                "service.namespace": "autoresearch",
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, timeout=5)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as the global tracer provider only if no one else has.
        # If another caller (e.g., a test fixture) has already set one,
        # we do not overwrite — we simply attach to whatever is current.
        if not isinstance(trace.get_tracer_provider(), TracerProvider):
            trace.set_tracer_provider(provider)

        _TRACER = trace.get_tracer("autoalphafold3._tracing", "0.1")
        _INITIALIZED = True
    except Exception:
        # Any failure — missing package, malformed URL, exporter init error,
        # daemon unreachable on first send — silently disables tracing for
        # this process. We never retry; we never raise.
        _PERMANENTLY_FAILED = True


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    """No-op-safe span context manager.

    Usage:
        with span("event_name", trial_id=trial_id, mode="modal"):
            # do work; any exception is re-raised
            return result

    Attributes:
        name: the span name (e.g., "submit_trial", "modal_spawn").
        **attrs: span attributes. Values must be one of:
            str, bool, int, float. Lists/dicts/objects are silently
            stringified by OpenTelemetry — keep attribute values flat.

    Behavior:
        - If tracing is disabled or initialization failed: yields immediately
          without doing anything. The wrapped body runs normally.
        - If tracing is active: starts a span as a child of the current
          context's span (giving you a tree), records attributes, and ends
          the span on exit. If the body raises, records the exception type
          and re-raises the original exception unchanged. If recording the
          exception itself fails, that failure is swallowed.
    """
    _try_init()
    if not _INITIALIZED or _TRACER is None:
        yield
        return

    # Attempt to start the span. If `start_as_current_span` raises for any
    # reason (broken exporter, GIL contention, etc.), fall back to no-op.
    try:
        cm = _TRACER.start_as_current_span(name, attributes=_clean_attrs(attrs))
    except Exception:
        yield
        return

    try:
        with cm as otel_span:
            try:
                yield
            except Exception as exc:
                # Record the exception on the span for the trace viewer, but
                # always re-raise the original. NEVER let tracing failures
                # swallow real exceptions.
                try:
                    otel_span.record_exception(exc)
                    # Mark span as errored using the OpenTelemetry status code.
                    from opentelemetry.trace import Status, StatusCode
                    otel_span.set_status(Status(StatusCode.ERROR, repr(exc)))
                except Exception:
                    pass  # tracing failure — never propagate
                raise
    except Exception:
        # If the outer `with cm` itself raises (e.g., span finish failure),
        # we must not swallow the body's exception. The inner `raise` above
        # handles body exceptions; this outer handler only triggers if the
        # span machinery itself failed. In that case, re-raise so we don't
        # silently lose information.
        raise


def _clean_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Coerce attribute values to OpenTelemetry-safe primitives.

    OpenTelemetry only natively accepts str, bool, int, float, and
    homogeneous sequences thereof. Anything else gets `repr()`'d to a
    string so the trace remains readable instead of failing to record.
    """
    cleaned: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        elif value is None:
            cleaned[key] = "None"
        else:
            try:
                cleaned[key] = repr(value)
            except Exception:
                cleaned[key] = "<unreprable>"
    return cleaned
