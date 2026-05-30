# auto-AlphaFold3 — Raindrop Workshop Passive Observability (spec)

**Status:** proposed addendum to the canonical spec. Build-time and rehearsal-time observability via local trace logging. Strictly opt-in, strictly modular, strictly passive.

**Selection criterion:** every span Raindrop captures must (a) be useful for a human reading the browser UI during build, rehearsals, or live event triage, AND (b) require zero behavioral change to the autoresearch loop if Raindrop is removed.

---

## 0. Document conventions

- "Workshop" refers to the local Raindrop Workshop daemon (https://github.com/raindrop-ai/workshop), an MIT-licensed local trace receiver and browser UI on `localhost:5899`.
- "Tracing" refers exclusively to the passive logging defined in this document. The critique sub-agent, replay-with-mutation, MCP-driven triage, and self-healing eval loop described in early Workshop materials are **explicitly out of scope** for this addendum.
- "The orchestrator" refers to `autoalphafold3/orchestrator.py` and the functions reachable from it. The "agent" refers to `autoalphafold3/agent.py` (the CLI entry point), not an in-process LLM agent. The autoresearch reasoning agent during the event is **Codex Goal Mode** running in a developer terminal, not a Python service.

---

## 1. Purpose

The autoresearch loop emits structured outcomes to `runs/ledger.jsonl`. The ledger captures *what happened* (status, metrics, gate verdicts, postmortems) but not *how it happened* (which Modal call returned which `object_id`, which preflight gate failed first, how long each polling attempt took, which inputs flowed into which function).

When a trial fails, when a gate produces an unexpected verdict, or when the orchestrator behaves in a way the developer did not predict, the ledger gives the verdict but not the path that led to it. Reconstructing the path from `runs/trials/T###/{stdout,stderr}.log` is slow and lossy.

Raindrop Workshop closes this gap. Every span the orchestrator and runner emit streams into the local browser UI as it happens. A developer with Workshop open can click into any trial and see the full call tree: inputs, outputs, exceptions, timings — in context.

This addendum scopes Workshop to **passive observability only**. The integration logs traces; humans read them. Nothing else.

---

## 2. Explicit non-goals

The following are NOT part of this addendum and must not be silently added without a separate spec change:

- **No automatic critique generation.** The Workshop integration does not spawn sub-agents to analyze failures or produce structured critiques.
- **No replay.** Raindrop's `setup-agent-replay` primitive is not used. (The autoresearch agent at runtime is Codex Goal Mode, not a Python LLM caller, so there is no instrumented LLM call to replay; the primitive does not apply to this architecture.)
- **No MCP-driven triage.** The remote Raindrop MCP server (`mcp.raindrop.ai`) is not registered with Claude Code or any other coding agent as part of this addendum. Trace queries happen by human eyeball in the browser UI, not by programmatic MCP calls.
- **No demo UI dependency on Workshop's runtime.** The demo UI's bespoke science panels (Hypothesis Card, Fold Cartographer Panel, Falsification Gate verdict tree, Discovery Ledger, Structure Overlay, Sampler Burst ensemble) read exclusively from the canonical ledger and Discovery Ledger files. The demo UI does not query Workshop, embed its API, or proxy its data. Workshop's web UI at `localhost:5899` IS opened as a side window during the live demo to render trace and timeline data the demo UI does not — but its absence does not break the demo's primary panels.
- **No required dependency.** The Raindrop SDK is not added to `requirements.txt` as a mandatory dependency. Developers who want tracing install it themselves. Trial workers running on Modal do not install Raindrop.
- **No remote ingestion.** Spans flow to `localhost:5899` only. No traces are sent to `app.raindrop.ai` or any cloud service. The integration is entirely local.
- **No locked-surface changes.** This addendum does not modify the scorer, the falsification gate logic, the gate-wave Modal adapter, baseline readiness, the discovery ledger, the schema, the preflight gates, or any other locked artifact. It only wraps existing functions with a no-op-safe span context manager.

---

## 3. Cumulative repo context

This addendum assumes all of the following are merged or about to be merged into `main`. The Workshop integration touches none of them directly; it wraps the outermost call sites and lets the existing code execute unchanged.

| Module | Purpose | Spans this addendum adds at its boundary |
|---|---|---|
| `autoalphafold3/agent.py` | CLI entry point | `cli.submit`, `cli.poll`, `cli.audit_modal_assets`, `cli.readiness` |
| `autoalphafold3/orchestrator.py` | Trial submission, polling, two-stage decisions | `submit_trial`, `_submit_modal`, `poll_trial`, `_poll_modal`, `record_trial_status`, `decide_stage_one_result`, `cancel_trial` |
| `autoalphafold3/runner.py` | Fixed-budget trial execution | `run_fixed_budget_trial`, `run_final_validation`, `initialize_trial_directory`, `write_artifact_manifest_stub`, `write_prediction_artifact` |
| `autoalphafold3/preflight.py` | Preflight gates | `preflight_run`, `preflight_forbidden_files`, `preflight_config_schema`, `preflight_param_count`, `preflight_tiny_forward`, `preflight_one_batch_loss`, `preflight_scorer_dry_run` |
| `autoalphafold3/falsification.py` | Pure verdict math | `falsification_verdict` (one span around the verdict computation) |
| `autoalphafold3/gate_wave.py` | Modal gate-wave adapter (fakeable) | `gate_wave_run`, with children per control variant |
| `autoalphafold3/baseline_readiness.py` | Baseline lock validator | `baseline_audit` |
| `autoalphafold3/discovery_ledger.py` | Confirmed-only Discovery Ledger writer | `discovery_ledger_write` |
| `autoalphafold3/readiness.py` | Pre-run readiness CLI | `readiness_run`, with child spans per section |
| `autoalphafold3/modal_assets.py` | Modal storage audit | `modal_asset_audit` |

The integration is purely additive. No function signature changes. No new fields in any schema. No new files except `autoalphafold3/_tracing.py` and tests.

---

## 4. Architectural shape

### 4.1 The single new module: `autoalphafold3/_tracing.py`

One file. ~80 lines including docstrings. Provides a `span(name, **attrs)` context manager and an `_init()` function that lazily initializes the Raindrop SDK on first use.

The module enforces four invariants:

1. **Opt-in.** Tracing is disabled unless the `RAINDROP_LOCAL_DEBUGGER` environment variable is set to a URL (conventionally `http://localhost:5899/v1/`).
2. **Silent failure.** Any exception inside the tracing path is swallowed. The main flow is never disrupted by a tracing problem.
3. **Idempotent init.** The Raindrop SDK is initialized at most once per process. If init fails, all subsequent spans are no-ops for the lifetime of the process.
4. **No type pollution.** The `span()` context manager yields `None`. No Raindrop types appear in business code function signatures, return types, or stored data.

### 4.2 Integration pattern (the only pattern)

Every wrapped function follows the identical pattern:

```
from autoalphafold3._tracing import span

def some_function(arg1, arg2, ...):
    with span("event_name", arg1=arg1, arg2_summary=summarize(arg2)):
        # existing function body, unchanged
        ...
```

No other pattern is permitted. The integration does not:
- Pass span handles around
- Conditionally branch on whether tracing is enabled
- Catch the span's exit to mutate behavior
- Use Raindrop SDK functions other than `init`, `begin`, `finish`

If a future integration is tempted to do any of the above, it requires a separate spec change.

### 4.3 The exception contract

Inside any `with span(...):` block:
- If the main flow raises an exception, the span records the exception type and message, then **re-raises the original exception unchanged**.
- If the span's `begin` or `finish` itself raises, the exception is swallowed inside `_tracing.py` and never reaches the caller.

This is the load-bearing invariant. **The presence or absence of Workshop must never change which exception the main code raises, or whether it raises at all.**

---

## 5. The span surface

Spans are added at the following call sites. Each is a single `with span(...)` wrapper around an existing function body. Attribute selection is conservative: identifiers, status fields, and small structured summaries — never full payloads.

### 5.1 CLI layer (`autoalphafold3/agent.py`)

| Span name | Attributes |
|---|---|
| `cli.submit` | `trial_path`, `mode` (dry_run\|modal), `repo_root` |
| `cli.poll` | `call_id`, `repo_root` |
| `cli.validate_manifest` | `manifest_count`, `repo_root` |
| `cli.audit_modal_assets` | `env`, `data_volume`, `locked_volume`, `search_ready` |
| `cli.readiness` | (none beyond the default span name) |

### 5.2 Orchestrator (`autoalphafold3/orchestrator.py`)

| Span name | Attributes | Child spans |
|---|---|---|
| `submit_trial` | `trial_path`, `mode`, `enforce_git_diff` | `preflight_run`, `_submit_modal` (if mode=modal), `ledger_write` |
| `_submit_modal` | `trial_id`, `app`, `function` (`run_trial`); on success: `call_object_id`; on failure: `error_type` | none |
| `poll_trial` | `call_id`, `repo_root` | `_poll_modal` (if modal-prefixed) |
| `_poll_modal` | `call_id`; per attempt: `attempt_index`, `status`, `elapsed_ms` | one child span per poll attempt |
| `record_trial_status` | `trial_id`, `status` | none |
| `decide_stage_one_result` | `trial_id`, `prior_best`, `delta`, `threshold`, `decision` (PROVISIONAL_KEEP \| DISCARD \| FAIL \| INFRA_FAIL) | none |
| `record_stage_one_decision` | `trial_id`, `decision` | none |
| `cancel_trial` | `call_id` | none |

### 5.3 Runner (`autoalphafold3/runner.py`)

| Span name | Attributes |
|---|---|
| `run_fixed_budget_trial` | `trial_id`, `max_steps`, `max_wall_minutes`, `budget`, `seed`; on success: `status`, `primary_metric_value`; on failure: `error_type` |
| `run_final_validation` | `trial_id`, `seed`, `n_seeds` |
| `initialize_trial_directory` | `trial_id`, `output_dir` |
| `write_artifact_manifest_stub` | `trial_id`, `path` |
| `write_prediction_artifact` | `trial_id`, `prediction_count`, `split` |

### 5.4 Preflight (`autoalphafold3/preflight.py`)

| Span name | Attributes |
|---|---|
| `preflight_run` | `trial_id`, `enforce_git_diff` |
| `preflight_forbidden_files` | `forbidden_count` |
| `preflight_config_schema` | `config_path` |
| `preflight_param_count` | `trial_id`, `max_params` |
| `preflight_tiny_forward` | `trial_id` |
| `preflight_one_batch_loss` | `trial_id` |
| `preflight_scorer_dry_run` | `split` |

Each preflight gate is a child of `preflight_run`. The parent span captures the overall outcome; children let the developer see exactly which gate failed first.

### 5.5 Falsification gate (`autoalphafold3/falsification.py`, `autoalphafold3/gate_wave.py`)

| Span name | Attributes | Child spans |
|---|---|---|
| `falsification_verdict` | `trial_id`, `gain_full`, `gain_knockout`, `gain_placebo`, `attributable_fraction`, `axis_held`, `verdict` | none |
| `gate_wave_run` | `trial_id`, `n_variants`, `n_seeds`, `timeout_seconds` | one per variant |
| `gate_wave_variant` | `variant_name` (knockout \| placebo \| seed_N), `status`, `elapsed_ms`; on failure: `error_type` | none |

`gate_wave_variant` spans capture the `starmap` execution per variant. When `return_exceptions=True` returns an exception object as a result, the span records it as a failed variant — without changing the gate-wave's own exception-to-evidence handling.

### 5.6 Baseline readiness, discovery ledger, readiness CLI

| Span name | Attributes |
|---|---|
| `baseline_audit` | `volume`, `manifest_count`, `pass_count`, `fail_count` |
| `discovery_ledger_write` | `trial_id`, `mechanism_kind`, `confirmed` |
| `readiness_run` | (none); children per section |
| `readiness_section` | `section_name`, `status` (PASS \| FAIL \| PENDING \| NOT_REQUESTED) |
| `modal_asset_audit` | `data_volume`, `locked_volume`, `status` |

### 5.7 Attribute selection rules

- **Identifiers are always captured** (trial_id, call_id, candidate_id, seed) — they let the developer correlate spans across files.
- **Outcome status fields are always captured** (status, verdict, decision) — they make the trace readable at a glance without expanding every span.
- **Small numeric quantities are captured** (gain_full, attributable_fraction, elapsed_ms, n_variants).
- **Large payloads are NEVER captured directly.** Full trial JSON, full patch diffs, full Arrow byte blobs, full scorer metrics objects do not become span attributes. If a developer needs the full payload, they read the canonical artifact file from disk.
- **Locked-volume contents are NEVER captured.** Validation labels, locked manifests, and scorer code never appear in span attributes. Only their hashes (which are already in metrics output and can be cross-referenced).
- **Free-form text from agent reasoning is NEVER captured by this addendum.** No hypothesis strings, no postmortems, no critique drafts. These exist in the ledger; tracing does not duplicate them.

The boundary between "small structured summary" and "large payload" is operationally defined as: a span attribute must fit in one line of the Workshop UI without truncation. If it does not, it is too large and must be summarized.

---

## 6. Modularity contract

### 6.1 Runtime disable

Unsetting `RAINDROP_LOCAL_DEBUGGER` disables tracing immediately for that process. Every `with span(...)` block becomes a no-op. No code change required.

### 6.2 SDK absence

If the Raindrop Python SDK is not installed in the current environment, `_tracing.py`'s `_init()` catches the `ImportError`, sets a permanent-failure latch, and treats all subsequent spans as no-ops. The integration imposes zero behavior change on environments without the SDK.

### 6.3 Full removal

To remove the integration entirely from the codebase:

```
rm autoalphafold3/_tracing.py
git grep -l '_tracing' autoalphafold3 tests | xargs sed -i '' '/from autoalphafold3._tracing import span/d'
# Then manually inspect each `with span(...)` block (git grep 'with span') and either:
#   (a) remove the with-line and de-indent the body, or
#   (b) replace with `if True:` to preserve the indentation for review
# Run the test suite. Done.
```

The removal is mechanical because the `with span(...)` pattern is the only integration shape used. There are no scattered Raindrop SDK calls, no decorator-based tracing, no monkey-patching, no MRO surprises.

The codebase's correctness, scoring behavior, falsification logic, and readiness contracts must all remain identical after removal. This invariant is verified by the test suite: every test that passes with the integration must pass after the integration is removed.

### 6.4 Trial-worker isolation

Trial workers running on Modal (the `run_trial` Function on the autoresearch app) do **not** receive Raindrop tracing. The Workshop integration is purely orchestrator-side. The runner module's spans fire only when `run_fixed_budget_trial` is invoked locally (e.g., during dry-run mode or in tests); when the runner is invoked inside a Modal container, the spans are no-ops because the env var is not set inside the container.

This avoids two failure modes:
- Modal containers attempting to reach `localhost:5899`, which they cannot
- The Raindrop SDK being a required dependency of the Modal training image

### 6.5 The locked-surface guarantee

`_tracing.py` is **not** in the agent's editable surface (per `autoalphafold3/editable_surface.md`). It is a developer-tooling module managed by humans during build and rehearsals. The agent must not modify it during search.

Conversely, `_tracing.py` is not a locked benchmark artifact either. It does not affect scoring, gate verdicts, or the discovery ledger. Removing it does not require a baseline relock.

Its appropriate classification is: **build-time developer tooling, agent-may-not-edit, baseline-independent**.

---

## 7. Setup

### 7.1 Per-developer one-time setup

```
# Install the Workshop daemon
curl -fsSL https://raindrop.sh/install | bash

# Configure and start the daemon (writes .env, starts daemon, opens browser)
raindrop workshop setup

# Install the Python SDK locally (NOT in requirements.txt for the project)
pip install raindrop-ai
```

### 7.2 Per-session activation

```
# In the terminal where you'll run python -m autoalphafold3.agent ...:
export RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/

# Verify the daemon is healthy:
curl -fsS http://localhost:5899/health
```

### 7.3 Verification

```
# Run any agent command:
python -m autoalphafold3.agent readiness

# Open the Workshop browser tab at localhost:5899. Confirm a trace appears.
```

If no trace appears, check:
1. `echo $RAINDROP_LOCAL_DEBUGGER` returns the expected URL
2. `pip show raindrop-ai` confirms the SDK is installed
3. `raindrop workshop status` reports healthy
4. Re-run the agent command — the first call after a fresh process initializes the SDK

If still no trace appears, the integration's silent-failure design has kicked in. Disable opportunistically with `unset RAINDROP_LOCAL_DEBUGGER` and proceed with normal development; tracing is not load-bearing.

---

## 8. Operational model

### 8.1 During build (this week and through pre-event prep)

Workshop is the default debugging tool for orchestrator-level and gate-wave-level issues. Every developer working on `autoalphafold3/orchestrator.py`, `runner.py`, `gate_wave.py`, `falsification.py`, or `readiness.py` should run with `RAINDROP_LOCAL_DEBUGGER` set so traces are available when a test or smoke run produces unexpected output.

Workshop is **not** the primary debugging tool for:
- Scorer math (read the test fixtures and `autoalphafold3/scorer/calpha_lddt.py` directly)
- Schema validation (Pydantic errors are explicit; no trace needed)
- Static type errors (use whatever type-checker is configured)

### 8.2 During rehearsals

The team runs end-to-end rehearsals with all components wired together. Workshop is open in a browser tab throughout. When a rehearsal produces unexpected behavior, the trace store contains everything the orchestrator did; reviewers click into the trace rather than grepping logs.

After each rehearsal, the trace database can be reset:

```
raindrop workshop reset
```

This avoids cross-rehearsal trace contamination but is optional; Workshop's SQLite store handles thousands of traces without performance issues.

### 8.3 During the live event

Workshop serves two roles during the live event: developer-side triage AND a visible credibility artifact during the demo itself.

**Developer-side triage (continuous):** Workshop runs in a side terminal/browser tab on the developer's machine throughout the event. When a wave produces an unexpected outcome — a trial that should have succeeded fails, a gate verdict that contradicts the trial's apparent metrics, a readiness check that flips PENDING to FAIL — the developer pulls up Workshop, finds the trace for the affected trial or wave, and diagnoses without re-running on Modal.

**Demo-time visibility (during the presentation):** Workshop's web UI is opened as a side window during the live demo presentation. The demo UI's bespoke science panels remain the primary surface; Workshop runs in parallel to render the trace/timeline data the demo UI does not. This serves the dual purpose of (a) reducing demo UI implementation scope (we do not build a Trial Timeline panel or a Modal Execution Panel ourselves) and (b) providing visible proof of real infrastructure activity during the presentation, directly supporting the project's "this is a real research loop, not a curated story" framing.

The two roles share the same daemon and trace store. The developer does not need to switch tools between triage and demo — Workshop is always running, always at `localhost:5899`, and always reflects the latest activity.

If the diagnosis identifies a code bug, the developer fixes it inline (the agent's "freeze tree per wave" rule per spec §7.4 ensures the fix lands at the next wave boundary). If the diagnosis identifies a hypothesis flaw, the developer notes it in their own reasoning and the next wave's hypothesis incorporates the lesson.

Note: this is human-driven debugging and human-curated demo support. The critique sub-agent and automated recovery loop described in earlier Workshop materials are **not part of this addendum**. They remain candidate v2 features.

### 8.4 Post-event

After the event, the SQLite trace store under `~/.raindrop/raindrop_workshop.db` contains a complete audit trail of every span the orchestrator emitted. This becomes useful for:
- Post-mortem analysis of failures or surprises
- Generating talk material (the trace store can produce screenshots of representative waves)
- Demonstrating the project's observability story in retrospectives

The trace store is not committed to the repo and is not backed up by default. If preservation matters, copy `~/.raindrop/raindrop_workshop.db` to `runs/postmortem/workshop_traces.db` after the event.

---

## 9. Failure modes and required invariants

### 9.1 The single load-bearing invariant

> **The presence or absence of Workshop must never change the autoresearch loop's behavior.**

Every other invariant in this document derives from this one.

### 9.2 Tested failure modes

A test in `tests/test_tracing.py` must verify each of the following:

| Failure | Required behavior |
|---|---|
| Raindrop SDK not installed | Every `with span(...)` is a no-op; no exception leaks |
| `RAINDROP_LOCAL_DEBUGGER` env var unset | Every `with span(...)` is a no-op; SDK is never imported |
| Raindrop SDK installed but daemon unreachable | First call's `_init` may attempt connection; failure latches; subsequent spans are no-ops; no exception leaks |
| Raindrop SDK installed, daemon reachable, but `begin()` raises | Span is a no-op for this invocation; main function body still executes; main exception (if any) is re-raised unchanged |
| Raindrop SDK installed, daemon reachable, `begin()` succeeds, body raises | Span records the exception; main exception is re-raised unchanged |
| Raindrop SDK installed, daemon reachable, `begin()` succeeds, body succeeds, `finish()` raises | `finish()` exception is swallowed; main function returns its normal value |

These six tests are mandatory before the integration is enabled in any rehearsal or event environment.

### 9.3 The "compare with and without" smoke test

Before each significant rehearsal:

```
# Run the readiness CLI with tracing OFF
unset RAINDROP_LOCAL_DEBUGGER
python -m autoalphafold3.agent readiness > /tmp/readiness_no_tracing.json

# Run with tracing ON
export RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/
python -m autoalphafold3.agent readiness > /tmp/readiness_with_tracing.json

# The two outputs MUST be byte-identical.
diff /tmp/readiness_no_tracing.json /tmp/readiness_with_tracing.json
```

Any difference is a regression and must be fixed before proceeding. This smoke test is the operational guarantee that the load-bearing invariant holds.

---

## 10. Demo UI relationship — division of labor

This section defines how Workshop's UI and the bespoke demo UI divide responsibility during the live presentation.

### 10.1 Two surfaces, both visible to judges

During the live demo, **both UIs are open simultaneously** on the presentation setup (either a dual-monitor configuration or a split screen on a single projector):

- **Primary surface — bespoke demo UI** (rendered from `docs/spec/demo-ui-plan.html`): the science panels that tell the project's narrative
- **Secondary surface — Workshop UI** (`localhost:5899`): the live trace/timeline activity that proves the infrastructure is real

The demo UI is the "what" — what hypothesis was registered, what the Falsification Gate verdict was, what the Discovery Ledger contains, what the predicted vs. true backbone overlay looks like. Workshop is the "how it ran" — the actual Modal spawn calls, polling sequence, gate-wave variant executions, preflight gate timing, all streaming live.

### 10.2 Division of UI responsibilities

The demo UI plan (`docs/spec/demo-ui-plan.html` §3) lists ten UI surfaces. With Workshop taking the trace/timeline load, the demo UI's scope is reduced:

| Demo UI surface | Built into demo UI? | Why |
|---|---|---|
| Run Overview | Yes — bespoke | Curated summary; not in Workshop's surface |
| Trial Trajectory (scatter) | Yes — bespoke | Domain visualization (lDDT vs trial number, color by family) |
| Trial Trajectory (per-trial timeline drill-down) | **No — substitute with Workshop** | Workshop renders per-trial spans natively |
| Hypothesis Card | Yes — bespoke | Pre-registration narrative; not in Workshop's surface |
| Fold Cartographer Panel | Yes — bespoke | Domain-specific four-axis diagnostic visualization |
| Falsification Gate Panel (verdict tree) | Yes — bespoke | Curated knockout/placebo/axis/seed verdict display |
| Falsification Gate Panel (control-wave timing) | **No — substitute with Workshop** | Workshop renders gate-wave variant spans natively |
| Discovery Ledger | Yes — bespoke | Curated, narrative |
| Modal Execution Panel | **No — substitute with Workshop** | Workshop renders Modal spawn/poll spans natively |
| Structure Overlay | Yes — bespoke | Completely custom 3D visualization |
| Sampler Burst (ensemble visual) | Yes — bespoke | Custom 3D ensemble visualization |
| Sampler Burst (live container scaling) | **No — substitute with Workshop** | Workshop renders the burst's parallel spans natively |

Net scope reduction: the demo UI does not build per-trial timeline rendering, container scaling visualization, or detailed Modal job-history rendering. Six bespoke science panels remain.

### 10.3 Data coupling between the two

Despite both being visible during the demo, the two UIs are **completely decoupled at the data layer**:

| Property | Workshop UI (`localhost:5899`) | Bespoke demo UI |
|---|---|---|
| Source of truth | Workshop's local SQLite (`~/.raindrop/raindrop_workshop.db`) | Canonical ledger (`runs/ledger.jsonl`) and Discovery Ledger |
| Data flow | Orchestrator → Raindrop SDK → Workshop daemon → SQLite | Orchestrator → ledger → demo UI renderer |
| Audience-visible together? | Yes (both shown during demo) | Yes |
| Code coupling | **Zero.** Demo UI does not query, embed, or proxy Workshop. | |
| Runtime coupling | **Zero.** If Workshop's daemon is down, demo UI's science panels remain fully functional from canonical ledger. Only the trace activity side-window is unavailable. | |

The demo UI must NOT iframe Workshop, NOT call Workshop's HTTP API, NOT depend on `~/.raindrop/raindrop_workshop.db` existing. Workshop is a parallel-running web app, not a data source for the demo UI.

### 10.4 The credibility framing this enables

A central concern in the canonical spec (§10 Risk assessment, §13 Winning criteria) is that judges will doubt the project's results — that the agent is hill-climbing a noisy validation score and narrating a story afterward. Showing Workshop live during the demo directly addresses this:

> *"On the left, you're seeing the science narrative — hypothesis, gate verdict, confirmed discoveries. On the right, this is our live observability layer. Every Modal call, every preflight check, every gate-wave variant — streaming as it happens, into a local trace store. This isn't a slideshow. The infrastructure on the right produced the science on the left."*

Workshop's presence is therefore not aesthetic noise — it is the project's strongest "this is real" signal. Treating it as a developer tool to be hidden was a missed framing opportunity.

### 10.5 What gets added to the demo UI plan

`docs/spec/demo-ui-plan.html` needs three changes:

1. **§3 UI Surfaces:** mark the per-trial timeline drill-down, Modal Execution Panel, and Sampler Burst live-scaling components as "Rendered by Workshop UI (`localhost:5899`), not built into the demo UI." Update the data-source notes on those panels accordingly.

2. **§4 Metrics Required:** unchanged; the demo UI's bespoke panels still need the same metrics from the canonical ledger.

3. **§9 Non-Goals:** add the following line:

   > Do not embed, iframe, or proxy Raindrop Workshop's UI inside the demo UI. Workshop runs in a parallel browser window during the demo (per `docs/spec/raindrop-workshop.md` §10) and renders the trace/timeline panels the demo UI does not. The two UIs are visually adjacent but completely independent at the code and data layer.

These three changes preserve the demo UI's narrative discipline while documenting the new scope reduction.

### 10.6 Setup for the demo

Before the demo presentation:

```
# Terminal 1: Workshop daemon (already running throughout the event)
raindrop workshop

# Terminal 2: the bespoke demo UI server (project-specific)
python -m autoalphafold3.demo  # or whatever the demo UI's entry point is

# Browser tab 1: bespoke demo UI (the primary surface)
http://localhost:8000  # or wherever the demo UI serves

# Browser tab 2: Workshop UI (the secondary surface)
http://localhost:5899
```

Position the two browser tabs in a split-screen or dual-monitor layout. Verify both render before the presentation begins.

### 10.7 Fallback if Workshop is unavailable during the demo

If the Workshop daemon crashes or refuses connections during the demo:

1. The bespoke demo UI continues to function normally — all science panels render from the canonical ledger.
2. The presenter shifts the spoken script to acknowledge: *"Our observability layer isn't available right now, but the canonical ledger on the left captures every outcome."*
3. The demo continues without the credibility-artifact framing. The science panels carry the full demo on their own.

Workshop being available is a "nice to have" for the demo, not a "must have." This must be true by construction: the bespoke demo UI must never have a hard dependency on Workshop being reachable.

---

## 11. Locked surface and editability

### 11.1 What this addendum locks

`autoalphafold3/_tracing.py` and the `with span(...)` integration points are not part of the agent's editable surface. The agent must not modify them during the search loop. Reasoning: tracing fidelity is a benchmark concern. An agent that can disable or alter tracing could mask its own behavior from human review.

Add to `autoalphafold3/editable_surface.md`:

> `autoalphafold3/_tracing.py` — locked during search. Not part of the agent's editable surface. Developer-managed observability module.

### 11.2 What this addendum does NOT lock

The canonical benchmark surface — scorer, falsification gate, gate-wave adapter, baseline readiness, discovery ledger, schema, preflight, modal_app — is unchanged. This addendum does not extend the locked surface beyond `_tracing.py` itself.

### 11.3 Patch policy

`autoalphafold3/patch_policy.py` should be extended to reject any patch in `autoalphafold3/patches/**` that touches `_tracing.py` or that adds `import raindrop` or `from raindrop` statements outside `_tracing.py`. This prevents agent-authored code from establishing alternate tracing paths that bypass the locked module.

---

## 12. Open questions and decisions

The following are intentionally left open for human decision after a calibration rehearsal:

1. **Should `cli.readiness` and `readiness_run` spans capture the full section-level PASS/FAIL/PENDING summary as attributes, or only the top-level status?**
   - Capturing the full summary is more useful in the browser UI but balloons span size.
   - Default: capture top-level status only; section-level statuses go in child `readiness_section` spans.

2. **Should the integration include automatic span emission for `modal_app.py`'s Cls lifecycle (`@modal.enter`, `@modal.method`)?**
   - These run inside Modal containers, which won't have `RAINDROP_LOCAL_DEBUGGER` set.
   - Default: no instrumentation inside Modal containers (per §6.4).
   - If a future need arises (e.g., wanting per-trial container start-up traces), it requires a separate spec addendum because it implies running the Raindrop SDK inside trial workers.

3. **Should `gate_wave_variant` spans capture the variant's output metrics as attributes?**
   - Capturing them lets the developer compare variants in the browser UI without round-tripping through the ledger.
   - But metrics objects are larger than the "fits on one line" rule.
   - Default: capture only `gain` (one float) and `verdict` (one enum). Other metrics stay in the ledger.

4. **Should preflight failures emit a higher-severity span attribute (`severity=error`) to make them more visible in the Workshop UI?**
   - Workshop honors a `severity` attribute for visual highlighting.
   - Default: yes for `preflight_*` spans that fail; no for normal pass-through.

5. **Should the trace store be reset between rehearsals or preserved across rehearsals?**
   - Preservation lets the team compare rehearsals; reset keeps the UI uncluttered.
   - Default: preserve unless explicitly reset; document the reset command in the rehearsal runbook.

These five questions can be answered after a single calibration rehearsal exercises the integration end-to-end. None of them are blocking.

---

## 13. Implementation surface (descriptive only)

For the actual build runbook — reference `_tracing.py` implementation, step-by-step order, wrapped-function examples, test scaffolds, patch policy diff, drafted prose, preflight checks, and don't-do list — see [`docs/spec/raindrop-workshop-implementation.md`](raindrop-workshop-implementation.md). This section describes what gets built and where; it does not give the build steps.


### 13.1 Files added

| File | Purpose | LOC estimate |
|---|---|---|
| `autoalphafold3/_tracing.py` | The no-op-safe `span()` context manager and SDK initializer | ~80 |
| `tests/test_tracing.py` | The six failure-mode tests from §9.2 plus the byte-identity smoke test scaffold | ~150 |
| `docs/spec/raindrop-workshop.md` | This document | (this file) |

### 13.2 Files modified

| File | Modification | LOC delta estimate |
|---|---|---|
| `autoalphafold3/agent.py` | 5 `with span(...)` wrappers around CLI subcommands | ~+10 |
| `autoalphafold3/orchestrator.py` | 8 `with span(...)` wrappers around public functions | ~+20 |
| `autoalphafold3/runner.py` | 5 `with span(...)` wrappers | ~+12 |
| `autoalphafold3/preflight.py` | 7 `with span(...)` wrappers (parent + 6 children) | ~+15 |
| `autoalphafold3/falsification.py` | 1 `with span(...)` wrapper around the verdict function | ~+3 |
| `autoalphafold3/gate_wave.py` | 1 parent + 1 child-per-variant `with span(...)` wrapper | ~+10 |
| `autoalphafold3/baseline_readiness.py` | 1 wrapper | ~+3 |
| `autoalphafold3/discovery_ledger.py` | 1 wrapper | ~+3 |
| `autoalphafold3/readiness.py` | 1 parent + N section children | ~+10 |
| `autoalphafold3/modal_assets.py` | 1 wrapper | ~+3 |
| `autoalphafold3/patch_policy.py` | Extend to reject `_tracing.py` edits and `import raindrop` outside `_tracing.py` | ~+15 |
| `autoalphafold3/editable_surface.md` | Add `_tracing.py` to the locked-during-search list | ~+2 |
| `docs/spec/demo-ui-plan.html` §3, §9 | (1) Mark per-trial timeline drill-down, Modal Execution Panel, and Sampler Burst live-scaling as "rendered by Workshop UI, not built into demo UI." (2) Add the §9 non-goal line from §10.5 of this document. (3) Document the demo's dual-window setup. | ~+25 |
| `AGENTS.md` | Add a one-line note about optional tracing | ~+2 |
| `README.md` | Add a one-paragraph "Optional: Trace Observability" section | ~+10 |
| Canonical spec (`docs/spec/autoalphafold3-canonical (2).html`) | Reference this Markdown from a new short §5.x subsection | ~+15 |

### 13.3 What is NOT modified

- The scorer (`autoalphafold3/scorer/**`) and its `locked_scorer.py` boundary
- The schema (`autoalphafold3/schema.py`) — no new fields, no extended dataclasses
- The locked manifests, validation labels, baseline ledger, or any file in `runs/baseline/**`
- The Modal app's GPU types, timeouts, `max_containers`, retry policy, or Volume mounts
- The falsification gate's verdict math, thresholds, or control construction (only the verdict function gets wrapped)
- The discovery ledger's confirmed-only contract (only the writer gets wrapped)
- The preflight gates' acceptance criteria (only the runner gets wrapped)

### 13.4 Test coverage

`tests/test_tracing.py` must include:
- All six failure-mode tests from §9.2
- The byte-identity smoke test from §9.3, run against the readiness CLI
- A test that asserts `_tracing.py` is not in the agent's editable surface (cross-reference with `editable_surface.md`)
- A test that the patch_policy rejects edits to `_tracing.py` and stray `import raindrop` statements
- A test that asserts no Raindrop-types appear in any function signature or return type in the affected modules (uses AST inspection)

Existing tests must continue to pass with both tracing enabled and disabled. The full test suite (`python3 -m pytest -p no:cacheprovider`) must pass under both conditions.

---

## 14. Implementation checklist

The integration is ready to ship when all of the following are true:

- [ ] `autoalphafold3/_tracing.py` exists and implements the no-op-safe `span()` context manager
- [ ] All six failure-mode tests in `tests/test_tracing.py` pass
- [ ] The byte-identity smoke test passes with the readiness CLI
- [ ] `_tracing.py` is added to `autoalphafold3/editable_surface.md` as locked-during-search
- [ ] `autoalphafold3/patch_policy.py` rejects edits to `_tracing.py` and unauthorized `import raindrop` statements
- [ ] All planned `with span(...)` wrappers are added to the orchestrator, runner, preflight, falsification, gate_wave, baseline_readiness, discovery_ledger, readiness, modal_assets, and agent modules
- [ ] The full test suite passes with `RAINDROP_LOCAL_DEBUGGER` set
- [ ] The full test suite passes with `RAINDROP_LOCAL_DEBUGGER` unset
- [ ] The full test suite passes with the Raindrop SDK uninstalled
- [ ] `docs/spec/demo-ui-plan.html` §9 includes the non-goal line from §10
- [ ] `AGENTS.md` mentions the optional tracing surface
- [ ] `README.md` has a one-paragraph setup section for developers who want to opt in
- [ ] Canonical spec (`docs/spec/autoalphafold3-canonical (2).html`) references this Markdown from a new short subsection
- [ ] A developer who has never touched this project can install, enable, and see their first trace in under five minutes following only `README.md`
- [ ] A developer can disable tracing for their session with `unset RAINDROP_LOCAL_DEBUGGER` and continue all normal workflows
- [ ] The integration removal procedure (§6.3) has been dry-run at least once on a throwaway branch to confirm it works mechanically

---

## 15. Cross-references

- Canonical spec: `docs/spec/autoalphafold3-canonical (2).html` (§4 schema; §5 agent stack; §5.5 preflight gates; §5.9 falsification gate; §7.4 freeze-tree-per-wave; §7.5 performance optimizations addendum)
- Demo UI plan: `docs/spec/demo-ui-plan.html` (§3 UI surfaces; §9 non-goals)
- Editable surface: `autoalphafold3/editable_surface.md`
- Program rules: `program.md`
- Foundation runbooks: `docs/runbooks/manifest_locking.md`, `docs/runbooks/modal_control_plane.md`, `docs/runbooks/nanofold_pin.md`
- Pre-run readiness handoff: `docs/handoffs/goal/autoalphafold3-prerun-readiness-goal-mode-prompt.md`

External references:
- Workshop repository: https://github.com/raindrop-ai/workshop
- Workshop docs: https://www.raindrop.ai/docs/workshop/overview/
- Raindrop SDK docs: https://raindrop.ai/docs/sdk/python

---

## 16. What this addendum is in one sentence

**A single new file (`autoalphafold3/_tracing.py`) that provides a no-op-safe `span(name, **attrs)` context manager, wrapped around the orchestrator's existing public functions in roughly 100 places, so that developers running with `RAINDROP_LOCAL_DEBUGGER` set get a live browser trace of every wave — used for build-time and rehearsal-time debugging, AND shown live during the demo as a parallel-window credibility artifact alongside the bespoke science panels — while remaining a hard no-op for anyone who hasn't installed the SDK or set the env var.**

If at any point this addendum starts describing more than that, it has scope-crept and the additions need a separate spec change.
