# AutoAlphaFold3 Modal Pre-Run Readiness Plan

## Snapshot

As of 2026-05-30, the Modal pre-run readiness implementation stack through PR
#41 is merged on `main`, the real Modal baseline is locked, and this follow-up
branch adds the remaining live readiness artifacts: Modal event authority proof
and known-null/known-positive Falsification Gate calibration evidence.

## What Is Already Implemented

- Modal-hosted trusted orchestrator and worker-role contract surfaces.
- Harness/execution-plane secret-boundary checks.
- Falsification Gate schemas, verdict math, bounded control planning, and
  confirmed-only Discovery Ledger helpers.
- Baseline readiness reader and human-approved `lock-baseline` command.
- Human-approved baseline runner and real locked baseline evidence under
  `runs/baseline/**`.
- Readiness report with explicit `PENDING_HUMAN_LIVE_ACTION` classification
  and live Modal authority proof support.
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

Status: complete for this checkout. Local gates pass on the approved local-only
fixture.

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

Status: complete for the baseline. The Modal baseline run completed and the
baseline lock was written through `lock-baseline`.

### 3. Modal Deployment Decision

Deploy the Modal app only after the cost/resource decision is explicit.

Known deploy command from the local plan:

```bash
.venv/bin/python -m modal deploy autoalphafold3/modal_app.py
```

Status: complete for the approved baseline run. This branch adds
`audit-modal-authority`, which records the deployed `TrustedOrchestrator`
authority proof in `runs/modal_event_authority.json` without starting search.

### 4. Falsification Gate Calibration Runner

Added a small PR path for known-null and known-positive gate calibration.

Implemented shape:

- `run-gate-calibration --mode dry-run` plans the evidence run without side
  effects,
- `run-gate-calibration --mode modal --approve I_APPROVE_GATE_CALIBRATION_RUN`
  uses the deployed Modal gate-control runner to write calibration-only
  evidence under `runs/gate_calibration/`,
- `calibrate-gate --mode dry-run` prints the required evidence contract without
  side effects,
- `calibrate-gate --mode from-evidence --approve I_APPROVE_GATE_CALIBRATION`
  writes `runs/falsification_gate_calibration.json`,
- the command requires complete known-null and known-positive evidence records,
- known-null evidence must be killed by the gate,
- known-positive evidence must be `CONFIRMED`,
- synthetic fixture evidence is refused,
- readiness remains blocked if calibration is absent or incomplete.

Status: complete for this checkout. The frozen calibration lives at
`runs/falsification_gate_calibration.json`.

## Current Command Surface

The current `autoalphafold3.agent` commands are:

- `submit`
- `poll`
- `validate-manifest`
- `audit-modal-assets`
- `readiness-report`
- `lock-baseline`
- `run-baseline`
- `run-gate-calibration`
- `calibrate-gate`
- `materialize-local-fixture`
- `audit-modal-authority`

Not currently present:

- `deploy`

## Recommended Order

1. Merge the readiness-live baseline lock PR.
2. Record Modal event authority proof with explicit approval.
3. Produce known-null and known-positive gate calibration evidence with
   explicit approval.
4. Run `calibrate-gate --mode from-evidence`.
5. Rerun readiness report.
6. Start autonomous search only after readiness is no longer blocked and a
   human explicitly approves the live search action.
