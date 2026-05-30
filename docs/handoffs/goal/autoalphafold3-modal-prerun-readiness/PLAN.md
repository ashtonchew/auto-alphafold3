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

Added a small PR path for a human-approved baseline run command before using
`lock-baseline`.

Implemented shape:

- `run-baseline --mode dry-run` plans the action without side effects,
- `run-baseline --mode modal --approve I_APPROVE_BASELINE_RUN` invokes the
  deployed Modal trial and scorer workers,
- writes only trial-scoped source artifacts under `runs/trials/T000/`,
- refuses scorer payloads that are not official, non-local, scorer-only,
  `status=SCORED`, and `max_templates=0`,
- then `lock-baseline --dry-run` validates the source artifacts,
- only `lock-baseline --approve I_APPROVE_BASELINE_LOCK` writes
  `runs/baseline/**`.

Still required:

- deploy/authenticate the Modal event authority before the modal run,
- run the command with explicit approval,
- provide approved feature fingerprints to `lock-baseline`.

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

Added a small PR path for known-null and known-positive gate calibration.

Implemented shape:

- `calibrate-gate --mode dry-run` prints the required evidence contract without
  side effects,
- `calibrate-gate --mode from-evidence --approve I_APPROVE_GATE_CALIBRATION`
  writes `runs/falsification_gate_calibration.json`,
- the command requires complete known-null and known-positive evidence records,
- known-null evidence must be killed by the gate,
- known-positive evidence must be `CONFIRMED`,
- synthetic fixture evidence is refused,
- readiness remains blocked if calibration is absent or incomplete.

Still required:

- produce the real known-null and known-positive evidence records after baseline
  and Modal authority are ready,
- run the command with explicit approval.

## Current Command Surface

The current `autoalphafold3.agent` commands are:

- `submit`
- `poll`
- `validate-manifest`
- `audit-modal-assets`
- `readiness-report`
- `lock-baseline`
- `run-baseline`
- `calibrate-gate`
- `materialize-local-fixture`

Not currently present:

- `deploy`

## Recommended Order

1. Implement approved local fixture materialization.
2. Implement real baseline runner.
3. Review Modal cost/resource tier and deploy the app.
4. Run the real baseline and lock it.
5. Implement and run Falsification Gate calibration.
6. Rerun readiness report.
7. Start autonomous search only after readiness is no longer blocked and a
   human explicitly approves the live search action.
