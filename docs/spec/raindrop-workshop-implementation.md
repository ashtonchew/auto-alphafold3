# auto-AlphaFold3 — Raindrop Workshop Implementation Guide

**Audience:** the AI agent (Codex Goal Mode, Claude Code, or equivalent) that will build the Raindrop Workshop integration in one shot.

**Companion document:** [`docs/spec/raindrop-workshop.md`](raindrop-workshop.md) — read it first. It contains the design, non-goals, contracts, and boundaries. This document does not repeat them.

**Scope of this document:** every concrete artifact an implementing agent needs that the spec does not provide. Reference code for `autoalphafold3/_tracing.py`. Worked-out wrapper examples for three representative functions. One fully-written failure-mode test as a template for the other five. The exact patch-policy diff. Drafted text for AGENTS.md, README, and the canonical-spec cross-reference. A stepped implementation order with verification gates. A preflight checklist to run before writing any code. An explicit don't-do list.

If something is in the spec, this document does not restate it. If something is in this document, the spec describes its purpose at a higher level.

---

## 0. How to use this document

Read it top to bottom once before starting any edits. The order of sections matches the order of operations.

If you find a contradiction between this document and the spec, the **spec wins on design intent** and **this document wins on the implementation mechanics**. If a contradiction can't be reconciled, stop and report — do not improvise.

Do not skip the preflight checks in §1. They surface environment differences (SDK availability, project conventions, Workshop daemon health) that change the implementation. Skipping them is the single largest source of one-shot failures.

---

## 1. Preflight checks (run before writing any code)

Run each of these from the repo root. Record the output. If any fail or surprise you, **stop and resolve before proceeding** — do not adapt the implementation to work around an unexpected environment.

### 1.1 Workshop daemon

```bash
curl -fsS http://localhost:5899/health
# Expected: HTTP 200 with a JSON body indicating "ok" or similar
# If the daemon is not running, follow the spec §7.1 setup, then re-run this check
```

### 1.2 Python environment

```bash
python3 --version
# Expected: Python 3.11 or 3.12 (consistent with the project's existing modules)

python3 -c "import sys; print(sys.executable)"
# Note the venv path; all installs must go to this venv
```

### 1.3 OpenTelemetry packages (the transport this implementation uses)

```bash
pip show opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http 2>&1 | grep -E '^(Name|Version)' || echo "missing"
# If "missing", install:
pip install opentelemetry-api>=1.27 opentelemetry-sdk>=1.27 opentelemetry-exporter-otlp-proto-http>=1.27
```

This implementation uses OpenTelemetry (OTLP/HTTP/JSON) as the transport to Workshop, not the Raindrop Python SDK. Reason: Workshop's docs explicitly list OTLP at `/v1/traces` as a supported transport, and OpenTelemetry gives true nested span trees natively. The Raindrop SDK is "interaction-per-call" which doesn't model nested orchestrator → preflight → modal_spawn relationships cleanly.

The user-visible toggle remains the `RAINDROP_LOCAL_DEBUGGER` env var (per spec). Internally the value is parsed as the OTLP HTTP base URL, and traces are POSTed to `${RAINDROP_LOCAL_DEBUGGER}traces` (so if the env var is `http://localhost:5899/v1/`, traces go to `http://localhost:5899/v1/traces`).

### 1.4 Project conventions

```bash
# Confirm test framework
head -1 tests/test_falsification.py
# Expected: from __future__ import annotations OR a regular import

# Confirm Pydantic version (the project uses Pydantic models in schema.py)
python3 -c "import pydantic; print(pydantic.VERSION)"
# Should be Pydantic v2

# Confirm pytest invocation pattern
grep -E '^\[tool.pytest|^addopts' pyproject.toml pytest.ini 2>/dev/null
# Note any project-specific pytest config
```

### 1.5 Locked-surface inventory

```bash
cat autoalphafold3/editable_surface.md | sed -n '/## Locked During Search/,/^## /p'
# Read the full locked list. Memorize it. Your edits must not touch any file in this list
# except autoalphafold3/editable_surface.md itself (you will add _tracing.py to it).
```

### 1.6 Existing patch policy shape

```bash
grep -n 'def \|class ' autoalphafold3/patch_policy.py
# Note the existing function/class shapes — your extension must match the existing style
```

### 1.7 Branch state

```bash
git status
# Expected: clean tree on a feature branch (e.g., spec/raindrop-passive-observability or feat/raindrop-tracing)
# If on main, branch first: git checkout -b feat/raindrop-tracing main
```

If any preflight check fails or returns an unexpected value, stop. Resolve the discrepancy before writing any code. Do not adapt the implementation to compensate.

---

## 2. Step-by-step implementation order

Execute these steps **in order**. After each step, run the verification listed before proceeding. Do not batch — each step's verification protects the next step from corrupted state.

### Step 1 — Create `autoalphafold3/_tracing.py`

Use the reference implementation in §3 of this document **verbatim**. Do not modify it. If you think it needs modification, stop and report.

**Verify:**
```bash
python3 -c "from autoalphafold3._tracing import span; print('ok')"
# Expected: prints "ok" with no exceptions
```

### Step 2 — Create `tests/test_tracing.py`

Use the test scaffold in §5 of this document. It contains one fully-written failure-mode test plus the structure for the other five.

**Verify:**
```bash
python3 -m pytest -p no:cacheprovider tests/test_tracing.py -q
# Expected: all six failure-mode tests pass
```

### Step 3 — Add `_tracing.py` to `editable_surface.md`

Open `autoalphafold3/editable_surface.md`, find the `## Locked During Search` section, add this line at the end of the list (sorted lexicographically — alphabetize within the section):

```markdown
- `autoalphafold3/_tracing.py`
```

**Verify:**
```bash
grep -c '_tracing.py' autoalphafold3/editable_surface.md
# Expected: 1
```

### Step 4 — Extend `autoalphafold3/patch_policy.py`

Use the exact diff in §6 of this document. Do not paraphrase.

**Verify:**
```bash
python3 -m pytest -p no:cacheprovider tests/test_patch_policy.py -q
# Expected: all existing patch policy tests pass (you have not broken anything)
```

Then add a new patch policy test (template in §6) and confirm it passes.

### Step 5 — Wrap the CLI entry points (`autoalphafold3/agent.py`)

For each subcommand in `agent.py`, wrap the body with `with span("cli.<subcommand_name>", **attrs):` per the pattern shown in §4. The five subcommands are: `submit`, `poll`, `validate-manifest`, `audit-modal-assets`, `readiness`.

**Verify:**
```bash
python3 -m autoalphafold3.agent --help
# Expected: help text prints normally (your changes didn't break CLI parsing)

# Run the readiness subcommand once
unset RAINDROP_LOCAL_DEBUGGER
python3 -m autoalphafold3.agent readiness > /tmp/readiness_no_trace.json 2>&1 || true

export RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/
python3 -m autoalphafold3.agent readiness > /tmp/readiness_with_trace.json 2>&1 || true

diff /tmp/readiness_no_trace.json /tmp/readiness_with_trace.json
# Expected: no diff. The byte-identity invariant must hold.
```

### Step 6 — Wrap the orchestrator (`autoalphafold3/orchestrator.py`)

Wrap each of: `submit_trial`, `_submit_modal`, `poll_trial`, `_poll_modal`, `record_trial_status`, `decide_stage_one_result`, `record_stage_one_decision`, `cancel_trial`. Use the pattern in §4.

**Verify:**
```bash
python3 -m pytest -p no:cacheprovider tests/test_runner_and_locked_scorer.py -q
# Expected: existing tests pass
```

### Step 7 — Wrap the runner (`autoalphafold3/runner.py`)

Wrap: `run_fixed_budget_trial`, `run_final_validation`, `initialize_trial_directory`, `write_artifact_manifest_stub`, `write_prediction_artifact`.

**Verify:**
```bash
python3 -m pytest -p no:cacheprovider tests/test_runner_and_locked_scorer.py -q
# Expected: still passes
```

### Step 8 — Wrap the preflight gates (`autoalphafold3/preflight.py`)

Wrap the top-level `run_preflight` function as the parent span; wrap each individual gate (forbidden_files, config_schema, param_count, tiny_forward, one_batch_loss, scorer_dry_run) as a child span.

**Verify:**
```bash
python3 -m pytest -p no:cacheprovider tests/test_local_contracts.py -q
# Expected: still passes
```

### Step 9 — Wrap the falsification gate (`autoalphafold3/falsification.py`, `autoalphafold3/gate_wave.py`)

In `falsification.py`: wrap the verdict computation function with `span("falsification_verdict", ...)`.
In `gate_wave.py`: wrap the gate-wave runner with `span("gate_wave_run", ...)`, and each variant execution with `span("gate_wave_variant", ...)`.

**Verify:**
```bash
python3 -m pytest -p no:cacheprovider tests/test_falsification.py tests/test_gate_wave.py -q
# Expected: still passes
```

### Step 10 — Wrap the remaining modules

Wrap: `baseline_readiness.audit_baseline_readiness` → `span("baseline_audit", ...)`, `discovery_ledger.append_discovery` → `span("discovery_ledger_write", ...)`, `readiness.<top-level function>` → `span("readiness_run", ...)` with child spans per section, `modal_assets.audit_modal_assets` → `span("modal_asset_audit", ...)`.

**Verify:**
```bash
python3 -m pytest -p no:cacheprovider
# Expected: full suite passes
```

### Step 11 — Update `AGENTS.md` and `README.md`

Use the drafted text in §7 of this document. Add the optional-tracing section to README; add a one-line note to AGENTS.md.

**Verify:**
```bash
grep -c 'Raindrop' AGENTS.md README.md
# Expected: at least 2 (one in each file)
```

### Step 12 — Add canonical-spec cross-reference

Open `docs/spec/autoalphafold3-canonical (2).html`. Find an appropriate location (suggest §5.10 if it exists, or add a new sub-section §5.11 at the end of §5). Use the drafted HTML in §7 of this document.

**Verify:**
```bash
grep -c 'raindrop-workshop.md\|raindrop-workshop-implementation.md' "docs/spec/autoalphafold3-canonical (2).html"
# Expected: at least 1
```

### Step 13 — Final full verification

```bash
# Tests pass with tracing OFF
unset RAINDROP_LOCAL_DEBUGGER
python3 -m pytest -p no:cacheprovider
# Expected: all pass

# Tests pass with tracing ON
export RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/
python3 -m pytest -p no:cacheprovider
# Expected: all pass

# Tests pass with the SDK uninstalled (simulate)
# (Optional but recommended: temporarily install in a clean venv without otel packages)

# Byte-identity smoke test on readiness CLI
unset RAINDROP_LOCAL_DEBUGGER
python3 -m autoalphafold3.agent readiness > /tmp/a.json 2>&1
export RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/
python3 -m autoalphafold3.agent readiness > /tmp/b.json 2>&1
diff /tmp/a.json /tmp/b.json
# Expected: no diff

# Skill evals pass
python3 .claude/skill-evals/run_offline_evals.py
# Expected: pass
```

If any step's verification fails, fix it before proceeding. Do not batch failures.

---

## 3. Reference implementation: `autoalphafold3/_tracing.py`

Use this code **verbatim**. Do not change variable names, function shapes, error handling, or comment text. The exception contract is load-bearing.

```python
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
```

End of `_tracing.py` reference implementation. Do not modify.

---

## 4. Wrapped function examples

Three concrete examples covering the three common shapes. Use these as templates for the other wrapping sites.

### 4.1 Simple wrap (no return value, no computed attributes)

`autoalphafold3/agent.py` — `cli.submit` subcommand handler:

```python
# BEFORE
def main(argv: list[str] | None = None) -> int:
    # ... arg parsing ...
    if args.command == "submit":
        manifest_paths = _parse_manifest_args(args.manifest)
        call_id = submit_trial(
            args.trial_path,
            repo_root=args.repo_root,
            ledger_path=args.ledger_path,
            manifest_paths=manifest_paths,
            mode=args.mode,
        )
        print(json.dumps({"call_id": call_id}, sort_keys=True))
        return 0

# AFTER
from autoalphafold3._tracing import span

def main(argv: list[str] | None = None) -> int:
    # ... arg parsing ...
    if args.command == "submit":
        with span(
            "cli.submit",
            trial_path=str(args.trial_path),
            mode=args.mode,
            repo_root=str(args.repo_root),
        ):
            manifest_paths = _parse_manifest_args(args.manifest)
            call_id = submit_trial(
                args.trial_path,
                repo_root=args.repo_root,
                ledger_path=args.ledger_path,
                manifest_paths=manifest_paths,
                mode=args.mode,
            )
            print(json.dumps({"call_id": call_id}, sort_keys=True))
            return 0
```

Key points:
- The `with` block contains the entire branch body, including the `print` and `return`.
- The `return 0` inside the `with` block returns normally; the span ends as the function returns.
- Attribute values are strings/primitives — no objects, no full trial JSON.

### 4.2 Wrap with return value (function returns a meaningful object)

`autoalphafold3/orchestrator.py` — `_submit_modal`:

```python
# BEFORE
def _submit_modal(
    trial_payload: dict[str, object],
    *,
    repo_root: str | Path,
    ledger_path: str | Path,
) -> str:
    try:
        import modal
    except ModuleNotFoundError:
        result = AutoFoldResult(
            trial_id=str(trial_payload["trial_id"]),
            status=TrialStatus.INFRA_FAIL,
            # ... etc
        )
        append_ledger(result, ledger_path=Path(repo_root) / ledger_path, dedupe=True)
        return f"{MODAL_CALL_PREFIX}:INFRA_FAIL:{trial_payload['trial_id']}"
    fn = modal.Function.from_name(APP_NAME, "run_trial")
    call = fn.spawn(trial_payload)
    return f"{MODAL_CALL_PREFIX}:{call.object_id}"

# AFTER
from autoalphafold3._tracing import span

def _submit_modal(
    trial_payload: dict[str, object],
    *,
    repo_root: str | Path,
    ledger_path: str | Path,
) -> str:
    with span(
        "_submit_modal",
        trial_id=str(trial_payload["trial_id"]),
        app=APP_NAME,
        function="run_trial",
    ):
        try:
            import modal
        except ModuleNotFoundError:
            result = AutoFoldResult(
                trial_id=str(trial_payload["trial_id"]),
                status=TrialStatus.INFRA_FAIL,
                # ... etc
            )
            append_ledger(result, ledger_path=Path(repo_root) / ledger_path, dedupe=True)
            return f"{MODAL_CALL_PREFIX}:INFRA_FAIL:{trial_payload['trial_id']}"
        fn = modal.Function.from_name(APP_NAME, "run_trial")
        call = fn.spawn(trial_payload)
        return f"{MODAL_CALL_PREFIX}:{call.object_id}"
```

Key points:
- The entire function body sits inside `with span(...)`.
- `return` inside the `with` block returns normally — the span context manager ends the span as the function returns.
- The `call.object_id` is not added to the span as an attribute because it's only available after the body runs. This is acceptable — the trace still shows the trial_id and app/function name, which is enough for cross-referencing.
- The `try/except ModuleNotFoundError` is preserved exactly — tracing must not alter this control flow.

### 4.3 Wrap with parent + child spans (nested instrumentation)

`autoalphafold3/preflight.py` — `run_preflight` with child gate spans:

```python
# BEFORE
def run_preflight(
    trial_path: str | Path,
    *,
    repo_root: str | Path = ".",
    changed_paths: list[str] | None = None,
    manifest_paths: dict[str, str] | None = None,
    enforce_git_diff: bool = False,
) -> PreflightResult:
    trial = _load_trial(trial_path)
    _check_forbidden_files_unchanged(...)
    _check_config_schema(trial.config_path)
    _check_param_count_under_cap(trial)
    _run_tiny_forward_pass(trial)
    _run_one_batch_loss_check(trial)
    metrics = _run_scorer_dry_run_schema_check(trial)
    return PreflightResult(trial=trial, scorer_metrics=metrics)

# AFTER
from autoalphafold3._tracing import span

def run_preflight(
    trial_path: str | Path,
    *,
    repo_root: str | Path = ".",
    changed_paths: list[str] | None = None,
    manifest_paths: dict[str, str] | None = None,
    enforce_git_diff: bool = False,
) -> PreflightResult:
    with span(
        "preflight_run",
        trial_path=str(trial_path),
        enforce_git_diff=enforce_git_diff,
    ):
        trial = _load_trial(trial_path)
        with span("preflight_forbidden_files", trial_id=trial.trial_id):
            _check_forbidden_files_unchanged(...)
        with span("preflight_config_schema", trial_id=trial.trial_id, config_path=str(trial.config_path)):
            _check_config_schema(trial.config_path)
        with span("preflight_param_count", trial_id=trial.trial_id, max_params=trial.max_params):
            _check_param_count_under_cap(trial)
        with span("preflight_tiny_forward", trial_id=trial.trial_id):
            _run_tiny_forward_pass(trial)
        with span("preflight_one_batch_loss", trial_id=trial.trial_id):
            _run_one_batch_loss_check(trial)
        with span("preflight_scorer_dry_run", trial_id=trial.trial_id):
            metrics = _run_scorer_dry_run_schema_check(trial)
        return PreflightResult(trial=trial, scorer_metrics=metrics)
```

Key points:
- The outer `with span("preflight_run", ...)` is the parent.
- Each gate gets its own child `with span("preflight_<gate>", ...)`.
- The child spans automatically become children of the parent because `start_as_current_span` uses OpenTelemetry's current-context propagation.
- If any gate raises, the child span records the exception and re-raises; the parent span also records "error" status because the raise propagates through it. This is correct OpenTelemetry behavior.

### 4.4 Apply this pattern to all sites listed in spec §5

Use 4.1 for functions that just do work. Use 4.2 for functions that return values. Use 4.3 for functions with significant internal structure (preflight, gate_wave, readiness). The span names and attributes are listed in spec §5; use them exactly.

---

## 5. Test scaffold: `tests/test_tracing.py`

This is one fully-written failure-mode test plus the structure for the other five. Write the others by analogy.

```python
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

# Note: importing `span` here triggers module-level state. We always re-import
# inside individual tests after manipulating the environment, so tests start
# from a known state.


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
        # Restore prior module state to avoid leaking into other tests.
        for k in keys_to_remove:
            sys.modules.pop(k, None)
        for k, v in saved.items():
            sys.modules[k] = v


def test_env_var_unset_means_total_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """When RAINDROP_LOCAL_DEBUGGER is unset, span() is a complete no-op:
    the body runs, return values pass through, exceptions re-raise, and
    the opentelemetry packages are never imported."""
    monkeypatch.delenv("RAINDROP_LOCAL_DEBUGGER", raising=False)
    # Sanity: confirm no opentelemetry module is currently loaded (or if it is,
    # we'll verify our module didn't import it).
    initial_otel_keys = {k for k in sys.modules if k.startswith("opentelemetry")}

    with _isolated_tracing_module():
        from autoalphafold3._tracing import span

        # 1. Body runs.
        side_effects: list[int] = []
        with span("test", a=1):
            side_effects.append(1)
        assert side_effects == [1]

        # 2. Return value is preserved (via a helper function).
        def returns_42() -> int:
            with span("test"):
                return 42

        assert returns_42() == 42

        # 3. Exception is re-raised unchanged.
        with pytest.raises(ValueError, match="boom"):
            with span("test"):
                raise ValueError("boom")

        # 4. No opentelemetry module was imported by our module.
        new_otel_keys = {k for k in sys.modules if k.startswith("opentelemetry")} - initial_otel_keys
        assert new_otel_keys == set(), f"unexpected otel imports: {new_otel_keys}"


def test_sdk_not_installed_means_total_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """When RAINDROP_LOCAL_DEBUGGER is set but opentelemetry packages are
    not importable, span() still no-ops cleanly. We simulate the absence
    by patching the import to raise ModuleNotFoundError."""
    monkeypatch.setenv("RAINDROP_LOCAL_DEBUGGER", "http://localhost:5899/v1/")

    # Block opentelemetry imports inside our module's _try_init.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ModuleNotFoundError(f"simulated missing module: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with _isolated_tracing_module():
        from autoalphafold3._tracing import span

        # Body still runs; return value preserved; exception still re-raised.
        result = []
        with span("test"):
            result.append("ok")
        assert result == ["ok"]

        with pytest.raises(RuntimeError, match="x"):
            with span("test"):
                raise RuntimeError("x")


def test_daemon_unreachable_means_silent_failure() -> None:
    """When the SDK is installed and the env var is set but the daemon is
    unreachable, span() must still no-op silently. The exporter typically
    fails on first export attempt, not on init — so this test exercises
    the post-init failure path."""
    # Implementation note: this test is environment-dependent. Skip it if
    # opentelemetry is not installed in the test environment.
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    import os
    # Use a port that nothing should be listening on.
    os.environ["RAINDROP_LOCAL_DEBUGGER"] = "http://localhost:1/v1/"

    with _isolated_tracing_module():
        from autoalphafold3._tracing import span

        # Even with the daemon unreachable, this must not raise.
        with span("test"):
            pass

        # And return values still pass through.
        def returns_x():
            with span("test"):
                return "x"

        assert returns_x() == "x"


def test_body_exception_is_reraised_unchanged() -> None:
    """When the body raises, the exception MUST be re-raised with the same
    type and message. Tracing must never swallow real exceptions."""
    os.environ["RAINDROP_LOCAL_DEBUGGER"] = "http://localhost:5899/v1/"

    with _isolated_tracing_module():
        from autoalphafold3._tracing import span

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError) as exc_info:
            with span("test", attr1="value1"):
                raise CustomError("specific message")

        assert str(exc_info.value) == "specific message"


def test_tracing_init_exception_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """If _try_init() raises for any reason, span() must still no-op.
    Simulated by patching TracerProvider construction to raise."""
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

        # Init fails internally; span no-ops; body runs normally.
        result = []
        with span("test"):
            result.append("ok")
        assert result == ["ok"]


def test_finish_exception_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """If finishing the span raises after the body succeeds, the function
    must still return its value normally. Tracing failures during finish
    must never become visible to the caller."""
    # This is the hardest test to write because we need to inject a failure
    # specifically into the OpenTelemetry span-end path. For initial
    # implementation, document this as a manual smoke test rather than
    # automating it. Mark as xfail with a clear reason.
    pytest.xfail("Automated test deferred; verified manually by injecting a broken exporter.")


# Optional: byte-identity smoke test, run as a pytest invocation.
# This exists as a sanity check that wrapping doesn't change outputs.
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

    out_no_trace = subprocess.run(
        ["python3", "-m", "autoalphafold3.agent", "readiness"],
        capture_output=True,
        text=True,
        env=env_no_trace,
        cwd=tmp_path.parent,  # adjust to repo root
    ).stdout

    out_with_trace = subprocess.run(
        ["python3", "-m", "autoalphafold3.agent", "readiness"],
        capture_output=True,
        text=True,
        env=env_with_trace,
        cwd=tmp_path.parent,
    ).stdout

    assert out_no_trace == out_with_trace
```

Notes for the implementing agent:
- The first test (`test_env_var_unset_means_total_no_op`) is fully written. Use its style for the others.
- The `_isolated_tracing_module()` context manager is reusable — it ensures clean module state per test.
- The `test_finish_exception_is_swallowed` test is marked `xfail` because automating it cleanly requires monkey-patching deep OpenTelemetry internals. Manual verification is acceptable for this one; document the manual procedure in a comment.
- The byte-identity test is opt-in (`READINESS_BYTE_IDENTITY=1`) because it shells out and is slow. Run it manually before each rehearsal as part of the spec §9.3 smoke test.

---

## 6. Patch policy extension: `autoalphafold3/patch_policy.py`

Read the existing file first. Add this extension following the existing function-naming and exception-style conventions. The exact additions are below — adapt the naming to match the file's existing style if it differs.

### 6.1 New constants to add at the top of the file

```python
# Files that must not be modified by agent patches. _tracing.py is build-time
# developer tooling; if the agent could rewrite it, it could disable trace
# capture and mask its own behavior from human review.
TRACING_LOCKED_FILES = frozenset({
    "autoalphafold3/_tracing.py",
})

# Modules that may import the raindrop or opentelemetry SDK. Only _tracing.py
# is allowed; any other import path bypasses the locked tracing module and
# is rejected.
TRACING_SDK_IMPORTERS = frozenset({
    "autoalphafold3/_tracing.py",
})

TRACING_FORBIDDEN_IMPORTS = (
    "import raindrop",
    "from raindrop",
    "import opentelemetry",
    "from opentelemetry",
)
```

### 6.2 New validation function

```python
def validate_tracing_lockout(patch_paths: list[str], patch_diff_text: str) -> None:
    """Reject patches that touch _tracing.py or introduce raindrop/opentelemetry
    imports outside _tracing.py.

    Raises PatchPolicyError on violation.
    """
    # Block direct edits to the tracing module.
    for path in patch_paths:
        if path in TRACING_LOCKED_FILES:
            raise PatchPolicyError(
                f"patch touches locked tracing module: {path}. "
                f"_tracing.py is locked during search per editable_surface.md."
            )

    # Block raindrop/opentelemetry imports added outside _tracing.py.
    # We do this by scanning the patch text for added lines that introduce
    # the forbidden imports. A perfect implementation would parse the diff;
    # for the hackathon we accept the heuristic that any added line
    # starting with "+" and matching the forbidden imports is a violation.
    added_lines = [
        line[1:].lstrip()
        for line in patch_diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    for added in added_lines:
        for forbidden in TRACING_FORBIDDEN_IMPORTS:
            if added.startswith(forbidden):
                raise PatchPolicyError(
                    f"patch adds forbidden import outside _tracing.py: '{added}'. "
                    f"Only autoalphafold3/_tracing.py may import raindrop or opentelemetry."
                )
```

### 6.3 Wire into the existing scope validator

Find the existing `validate_patch_scope()` (or equivalent) function. Add a call to `validate_tracing_lockout(patch_paths, patch_diff_text)` at an appropriate point (after the file-list extraction, before returning). Match the existing call style.

### 6.4 New test for `tests/test_patch_policy.py`

```python
def test_patch_policy_rejects_edits_to_tracing_module() -> None:
    """Patches that modify autoalphafold3/_tracing.py must be rejected."""
    with pytest.raises(PatchPolicyError, match="locked tracing module"):
        validate_tracing_lockout(
            patch_paths=["autoalphafold3/_tracing.py"],
            patch_diff_text="",
        )


def test_patch_policy_rejects_raindrop_imports_outside_tracing() -> None:
    """Patches that introduce raindrop imports outside _tracing.py are rejected."""
    diff = """\
+++ b/autoalphafold3/runner.py
@@ -1,3 +1,4 @@
 from __future__ import annotations
+import raindrop.analytics
"""
    with pytest.raises(PatchPolicyError, match="forbidden import"):
        validate_tracing_lockout(
            patch_paths=["autoalphafold3/runner.py"],
            patch_diff_text=diff,
        )


def test_patch_policy_rejects_opentelemetry_imports_outside_tracing() -> None:
    """Patches that introduce opentelemetry imports outside _tracing.py are rejected."""
    diff = """\
+++ b/autoalphafold3/orchestrator.py
@@ -1,3 +1,4 @@
 from __future__ import annotations
+from opentelemetry import trace
"""
    with pytest.raises(PatchPolicyError, match="forbidden import"):
        validate_tracing_lockout(
            patch_paths=["autoalphafold3/orchestrator.py"],
            patch_diff_text=diff,
        )


def test_patch_policy_allows_unrelated_edits() -> None:
    """Patches that don't touch tracing should pass."""
    diff = """\
+++ b/autoalphafold3/runner.py
@@ -10,3 +10,4 @@
 def some_function():
+    return 42
"""
    # Should not raise.
    validate_tracing_lockout(
        patch_paths=["autoalphafold3/runner.py"],
        patch_diff_text=diff,
    )
```

---

## 7. Drafted text for AGENTS.md, README, and the canonical spec

### 7.1 `AGENTS.md` addition

Find an appropriate location (near the "Commands" section). Add:

```markdown
## Optional: Trace Observability

This project ships with optional Raindrop Workshop trace observability for
developers. To enable, install the local Workshop daemon and set
`RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/` before running the agent
CLI. The integration is a no-op without the env var; see
`docs/spec/raindrop-workshop.md` for details.
```

### 7.2 `README.md` addition

Add a new top-level section near the end (before any "License" section if one exists):

```markdown
## Optional: Trace Observability

The orchestrator and runner are instrumented for optional trace observability
via [Raindrop Workshop](https://github.com/raindrop-ai/workshop), a local
OpenTelemetry-compatible trace receiver and browser UI. Enabling tracing gives
you live visibility into the agent loop during build, rehearsals, and the live
event.

To enable:

```bash
# Install Workshop (one-time)
curl -fsSL https://raindrop.sh/install | bash

# Install OpenTelemetry packages (one-time)
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

# Start the Workshop daemon (each session)
raindrop workshop

# Enable tracing for the current shell (each session)
export RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/

# Run as normal; trace activity appears at http://localhost:5899
python -m autoalphafold3.agent submit trials/T001.json
```

The integration is opt-in: without the env var, every span is a no-op and the
project's behavior is unchanged. See `docs/spec/raindrop-workshop.md` for the
design and `docs/spec/raindrop-workshop-implementation.md` for the
implementation guide.
```

### 7.3 Canonical spec (`docs/spec/autoalphafold3-canonical (2).html`) cross-reference

Find an appropriate location — suggest adding a new short sub-section §5.11 at the end of §5 (Agent stack & tooling). If §5.10 already exists for Workshop, replace it with this version:

```html
  <h3>5.11 &nbsp;Optional trace observability via Raindrop Workshop</h3>
  <p>Build-time, rehearsal-time, and live-demo observability is provided by optional <a href="https://github.com/raindrop-ai/workshop">Raindrop Workshop</a> tracing. The integration is a single file (<code>autoalphafold3/_tracing.py</code>) wrapping the OpenTelemetry SDK with a no-op fallback: if the OpenTelemetry packages are not installed or <code>RAINDROP_LOCAL_DEBUGGER</code> is unset, every <code>with span(...)</code> block is a no-op and main behavior is unchanged. Tracing failures never propagate to the main loop. The instrumented surface is the orchestrator, runner, preflight gates, falsification gate, gate-wave adapter, baseline readiness, discovery ledger writer, readiness CLI, and Modal asset audit. Spans carry trial_id, status fields, and small structured summaries for cross-reference with the canonical ledger. No payloads, no locked-volume contents, no agent reasoning text are captured as span attributes.</p>
  <p>During the live demo, Workshop's web UI at <code>localhost:5899</code> is opened as a side window alongside the bespoke demo UI; the demo UI renders the science narrative while Workshop renders live trace activity. The two UIs are completely decoupled at the data layer and have zero runtime coupling. See <a href="raindrop-workshop.md"><code>docs/spec/raindrop-workshop.md</code></a> for the design and <a href="raindrop-workshop-implementation.md"><code>docs/spec/raindrop-workshop-implementation.md</code></a> for the implementation guide.</p>
```

---

## 8. Don't-do list

This is a hard list. If the implementation has done any of these, it has scope-crept and must be rolled back.

1. **Do NOT add `opentelemetry-*` or `raindrop-ai` to `requirements.txt` as mandatory dependencies.** They are optional dev-side installs. The project's required dependencies are unchanged.

2. **Do NOT import `opentelemetry` or `raindrop` anywhere except `autoalphafold3/_tracing.py`.** All other files use only the `span` symbol exported from `_tracing`. This is enforced by patch policy (§6); your implementation must not violate it.

3. **Do NOT add Raindrop or OpenTelemetry types to any function signature, return type, dataclass field, or Pydantic model.** If you ever find yourself writing `-> Span` or `Tracer` in a non-`_tracing.py` file, stop and remove it.

4. **Do NOT wrap any function inside a locked file.** Locked files per `editable_surface.md` include `autoalphafold3/scorer/**`, `autoalphafold3/locked_scorer.py`, `autoalphafold3/modal_app.py` (the spec §7 control plane). Wrappers go on the orchestrator-side callers of these functions, not inside the locked functions themselves.

5. **Do NOT change any existing function's signature, return type, or exception behavior.** The wrapping pattern is purely additive — body shifts inward by one indent level under the `with` block.

6. **Do NOT add `with span(...)` blocks inside Modal worker code paths** (anything that runs inside a Modal container). The Modal containers won't have `RAINDROP_LOCAL_DEBUGGER` set and the packages won't be installed; the spans would be no-ops there anyway, and adding them just creates dead code in the worker image.

7. **Do NOT capture full payloads as span attributes.** No full trial JSONs, no full patch diffs, no Arrow byte blobs, no full scorer metrics dicts. Attributes are short identifiers, status fields, and small numeric summaries. The "fits on one line of the Workshop UI" rule is operational.

8. **Do NOT capture validation labels, locked manifests, or scorer code as span attributes.** Only their hashes (which are already in metrics output).

9. **Do NOT capture free-form agent reasoning text** (hypothesis prose, postmortems, critique drafts). These live in the ledger; tracing does not duplicate them.

10. **Do NOT make `_tracing.py` retry init on failure.** Once `_PERMANENTLY_FAILED` is set, all subsequent calls no-op for the lifetime of the process. Retrying creates unbounded log spam on a flapping daemon.

11. **Do NOT use OpenTelemetry's auto-instrumentation packages** (`opentelemetry-instrumentation-requests`, etc.). They monkey-patch the Python runtime and can cause subtle interactions with other libraries. We only use manual `start_as_current_span` calls.

12. **Do NOT commit the `~/.raindrop/raindrop_workshop.db` file or any trace export files.** The trace store is local and developer-specific; it must not enter the repo.

13. **Do NOT modify the spec file (`docs/spec/raindrop-workshop.md`) during implementation.** If the spec is wrong, stop and report — do not silently rewrite it to match what you built.

---

## 9. Self-verification commands (run after each major step)

```bash
# After Step 1: _tracing.py exists and imports cleanly
python3 -c "from autoalphafold3._tracing import span; print('ok')"

# After Step 2: tests pass
python3 -m pytest -p no:cacheprovider tests/test_tracing.py -q

# After Step 4: patch policy still works
python3 -m pytest -p no:cacheprovider tests/test_patch_policy.py -q

# After Steps 5-10: full suite passes with and without tracing
unset RAINDROP_LOCAL_DEBUGGER
python3 -m pytest -p no:cacheprovider

export RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/
python3 -m pytest -p no:cacheprovider

# After Step 11-12: documentation links resolve
grep -c 'raindrop-workshop.md' AGENTS.md README.md "docs/spec/autoalphafold3-canonical (2).html"
# Expected: at least 1 per file, or document why not

# After Step 13: skill evals pass
python3 .claude/skill-evals/run_offline_evals.py
```

If any verification fails, **stop, diagnose, fix, re-verify before proceeding**. Do not batch fixes.

---

## 10. Common pitfalls

### 10.1 Forgetting to wrap return statements

```python
# WRONG — the wrap exits before the return
with span("foo"):
    result = compute()
return result  # OUTSIDE the with; span has ended

# RIGHT
with span("foo"):
    result = compute()
    return result  # INSIDE the with
```

The `with` statement ends at the dedent. If you put `return` outside, the span ends before the return value is computed (which is sometimes correct, but usually not — the span timing under-reports if the work happens before the return).

### 10.2 Spans inside `try/except` that swallow exceptions

If the existing function has a `try/except` that converts an exception to a non-exception result (e.g., `ModuleNotFoundError` → `INFRA_FAIL` return), the span will see "body completed normally" — which is correct. Don't add `raise` to the existing except blocks just to make the span record the exception. Preserve existing control flow exactly.

### 10.3 Forgetting the parent span

When wrapping nested functions (preflight gates inside `run_preflight`), make sure the parent `with span("preflight_run", ...)` is open when each child span starts. If you accidentally exit the parent before the children run (e.g., by misplacing a `return`), the children become top-level spans and the trace structure is broken.

### 10.4 Using locked attribute types

OpenTelemetry attributes must be primitives. The `_clean_attrs` helper handles common conversions, but if you pass a complex object as an attribute, it gets `repr()`'d to a string, which is usually unreadable. Prefer extracting the specific field you want:

```python
# WRONG — passes a whole dict
with span("foo", trial=trial.model_dump()):
    ...

# RIGHT — extract specific fields
with span("foo", trial_id=trial.trial_id, max_steps=trial.max_steps):
    ...
```

### 10.5 Running tests with leftover state

If a test sets `RAINDROP_LOCAL_DEBUGGER` and the next test doesn't unset it, you get state leakage. The `_isolated_tracing_module` fixture in §5 handles this; use it in every test that touches `_tracing` module-level state.

### 10.6 Modal worker confusion

If you accidentally add a `with span(...)` inside code that runs in a Modal container (the `run_trial` Function's body, for example), the span is harmless but it's dead code in production. The wrappers belong on the orchestrator's side of the Modal call boundary.

---

## 11. Final sign-off checklist

The integration is ready to ship when ALL of the following are true:

- [ ] `autoalphafold3/_tracing.py` matches the reference implementation in §3 verbatim
- [ ] All six failure-mode tests in `tests/test_tracing.py` pass (the xfail test is OK)
- [ ] All wrappers are placed per spec §5; no wrappers in locked files
- [ ] Patch policy extension in `autoalphafold3/patch_policy.py` is in place and tested
- [ ] `editable_surface.md` lists `autoalphafold3/_tracing.py` under "Locked During Search"
- [ ] Full test suite passes with `RAINDROP_LOCAL_DEBUGGER` set
- [ ] Full test suite passes with `RAINDROP_LOCAL_DEBUGGER` unset
- [ ] Byte-identity smoke test passes on the readiness CLI
- [ ] Skill evals pass (`python3 .claude/skill-evals/run_offline_evals.py`)
- [ ] `AGENTS.md`, `README.md`, and the canonical spec are updated with the drafted text
- [ ] No file in `requirements.txt` has `raindrop-ai` or `opentelemetry-*` as a mandatory entry
- [ ] No file outside `autoalphafold3/_tracing.py` contains `import raindrop` or `import opentelemetry`
- [ ] No function signature, return type, or dataclass field references Raindrop or OpenTelemetry types
- [ ] The git tree is clean apart from the intended changes
- [ ] A developer who runs `unset RAINDROP_LOCAL_DEBUGGER` followed by `python -m autoalphafold3.agent readiness` sees identical output to a developer with the env var set
- [ ] A developer following only `README.md` can install, enable, and see their first trace in under five minutes

If any item is unchecked, **do not declare the implementation complete**. Report what's blocking and either resolve it or escalate.

---

## 12. What to do if you get stuck

If any preflight check, verification step, or test fails in a way you can't explain in under five minutes:

1. **Stop.** Do not adapt the implementation to work around the failure.
2. **Capture the exact failure** — command run, expected output, actual output.
3. **Check whether the spec describes this case.** If yes, follow the spec. If no, this is a real gap.
4. **Report the gap.** Do not improvise a solution; an improvised solution to a gap in the spec is technical debt that will surface during the live event.

The single largest source of one-shot build failures is the agent silently working around an unexpected environment state. Don't do that.

---

That's the implementation guide. Together with [`docs/spec/raindrop-workshop.md`](raindrop-workshop.md), this is enough for a Claude Code or Codex agent in Goal Mode to one-shot the integration with high confidence.
