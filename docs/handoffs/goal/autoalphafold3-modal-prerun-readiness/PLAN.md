# AutoAlphaFold3 Modal Pre-Run Readiness Plan

## Snapshot

As of 2026-05-30, the Modal pre-run readiness implementation stack through PR
#31 is merged on `main`. Live Modal asset audit passes, but the readiness report
still blocks autonomous search because several remaining steps require real
live evidence or additional infrastructure commands.

## What Is Already Implemented

- Modal-hosted trusted orchestrator and worker-role contract surfaces.
- Harness/execution-plane secret-boundary checks.
- Falsification Gate schemas, verdict math, bounded control planning, and
  confirmed-only Discovery Ledger helpers.
- Baseline readiness reader and human-approved `lock-baseline` command.
- Readiness report with explicit `PENDING_HUMAN_LIVE_ACTION` classification.
- Live Modal asset handoff audit for helper Arrow files, feature fingerprints,
  scorer metadata, and public/locked Volume boundaries.

## Best-Practice Next Steps

### 1. Approved Local Fixture Materialization

Added a small PR path that lets local readiness gates consume an approved
local-only cached Arrow fixture without fabricating benchmark data.

Implemented shape:

- exact approval-token command:
  `python3 -m autoalphafold3.agent materialize-local-fixture --approve I_APPROVE_LOCAL_NANOFOLD_FIXTURE`,
- no writes to `runs/baseline/**`, canonical ledger, Discovery Ledger, or
  benchmark artifacts,
- deterministic local-only Arrow fixture with metadata provenance,
- `local_only=true`, `official_benchmark_result=false`, and `max_templates=0`,
- readiness gates can pass or fail on the fixture after materialization.

Still required:

- run the command only when the exact approval action is intended for this
  checkout.

### 2. Real Baseline Runner

Add a small PR for a human-approved baseline run command before using
`lock-baseline`.

Recommended shape:

- explicit command such as `run-baseline` with approval flags,
- uses the trusted Modal event path, not local smoke mode,
- writes trial-scoped source artifacts first,
- then `lock-baseline --dry-run` validates them,
- only `lock-baseline --approve` writes `runs/baseline/**`.

Why it cannot be done now:

- `runs/` has no real source artifacts,
- the existing CLI has `lock-baseline` but no command that produces the real
  scored baseline evidence that `lock-baseline` requires.

### 3. Modal Deployment Decision

Deploy the Modal app only after the cost/resource decision is explicit.

Known deploy command from the local plan:

```bash
.venv/bin/python -m modal deploy autoalphafold3/modal_app.py
```

Why it should not be treated as done now:

- `modal app list --env main` showed no deployed app,
- deployment is a live infrastructure action,
- the runbook calls out NanoFold pin, baseline readiness, and cost/resource tier
  review before deployment.

### 4. Falsification Gate Calibration Runner

Add a small PR for known-null and known-positive gate calibration.

Recommended shape:

- explicit command with dry-run/readiness mode first,
- read-only or smoke Modal controls unless separately approved,
- writes `runs/falsification_gate_calibration.json` only from real successful
  calibration evidence,
- readiness remains blocked if calibration is absent or incomplete.

Why it cannot be done now:

- there is no committed calibration command,
- creating the calibration evidence file manually would be fake gate evidence.

## Current Command Surface

The current `autoalphafold3.agent` commands are:

- `submit`
- `poll`
- `validate-manifest`
- `audit-modal-assets`
- `readiness-report`
- `lock-baseline`
- `materialize-local-fixture`

Not currently present:

- `run-baseline`
- `deploy`
- `calibrate-gate`

## Recommended Order

1. Implement approved local fixture materialization.
2. Implement real baseline runner.
3. Review Modal cost/resource tier and deploy the app.
4. Run the real baseline and lock it.
5. Implement and run Falsification Gate calibration.
6. Rerun readiness report.
7. Start autonomous search only after readiness is no longer blocked and a
   human explicitly approves the live search action.
