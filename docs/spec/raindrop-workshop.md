# Raindrop Workshop trace observability

Optional, opt-in trace logging for the agent loop and orchestrator. When enabled, every CLI invocation, orchestrator call, runner step, gate-wave verdict, baseline audit, discovery write, modal asset audit, and falsification computation streams as a span to the local Raindrop Workshop UI at `localhost:5899`. When disabled (the default), the integration is a hard no-op.

## What we built

- `autoalphafold3/_tracing.py` — a `span(name, **attrs)` context manager that emits OpenTelemetry spans to Workshop. Silently no-ops if the `RAINDROP_LOCAL_DEBUGGER` env var is unset or the OpenTelemetry packages are missing.
- `with span(...)` wrappers across the agent CLI, orchestrator, runner, falsification verdict, gate-wave runner, baseline readiness, discovery ledger, modal assets, and readiness report.
- `scripts/up.sh` — one-command setup that installs everything and runs your command with tracing enabled.

## How to use it

```bash
bash scripts/up.sh                                       # interactive shell with traces flowing
bash scripts/up.sh python -m autoalphafold3.agent readiness-report
```

Open `http://localhost:5899` to see live traces.

## Contract

- **Opt-in.** Without `RAINDROP_LOCAL_DEBUGGER` set, every span is a no-op. The project runs identically with and without tracing.
- **Silent failure.** Any tracing exception (SDK missing, daemon unreachable, malformed attribute) is swallowed. Main-flow exceptions always re-raise unchanged.
- **No mandatory dependencies.** The OpenTelemetry packages and the Raindrop daemon are optional installs handled by `scripts/up.sh`. They are not in `requirements.txt`.
- **Trial workers are not instrumented.** Wrappers are orchestrator-side only. Modal containers never reach out to localhost.
- **`_tracing.py` is locked during search** (per `editable_surface.md`). The patch policy rejects edits to `_tracing.py` and prevents `raindrop` / `opentelemetry` imports anywhere else in the codebase.

## What this is NOT

- No critique sub-agent
- No replay-with-mutation against current code
- No MCP-driven triage from coding agents
- No demo UI dependency (the demo UI reads canonical ledger files, never queries Workshop)
- No automated debugging or self-healing eval loop

These were considered and explicitly cut. The integration is passive observability only — logs streaming into a browser UI for humans to read.

## Demo relationship

Workshop's UI at `localhost:5899` runs as a side window during the live demo alongside the bespoke demo UI. The bespoke demo UI handles the science narrative panels (Hypothesis Card, Falsification Gate verdict tree, Discovery Ledger, Structure Overlay, Sampler Burst ensemble) reading from the canonical ledger. Workshop handles trace/timeline visibility for credibility ("this is real infrastructure, not a slideshow"). The two UIs are visually adjacent but completely decoupled at the data layer. If Workshop is unavailable during the demo, the demo UI continues to function normally.

## Maintenance

To add a new span:

```python
from autoalphafold3._tracing import span

def my_function(arg):
    with span("my_function", arg=str(arg)):
        # existing body
        return result
```

Attribute values must be primitives (str / bool / int / float). Anything else gets stringified by the `_clean_attrs` helper. Do not capture large payloads, validation labels, or agent reasoning text.

For long functions where re-indenting is awkward, use the delegation pattern:

```python
def my_function(arg):
    with span("my_function", arg=str(arg)):
        return _my_function_impl(arg)

def _my_function_impl(arg):
    # original body unchanged
    ...
```

To remove the integration entirely:

```bash
rm autoalphafold3/_tracing.py
git grep -l 'from autoalphafold3._tracing import span' | xargs sed -i '' '/from autoalphafold3._tracing import span/d'
# Then review remaining `with span(...)` blocks and either remove them or
# replace with `if True:` to preserve indentation while reviewing.
```

The patch policy locks `_tracing.py` from agent edits during search, so removal is a human-only action.

## Files

| File | Purpose |
|---|---|
| `autoalphafold3/_tracing.py` | The `span` context manager (locked) |
| `scripts/up.sh` | One-command setup |
| `tests/test_tracing.py` | Failure-mode tests (env unset, SDK missing, daemon down, body raises) |
| `tests/test_local_contracts.py` (additions) | Patch-policy tests for the tracing lockout |
