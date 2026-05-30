# AutoAlphaFold3 Modal Pre-Run Readiness Goal

## Objective

Bring the NanoFold-style AlphaFold3-lite system to honest pre-run readiness
after PR #21 and PR #22 by implementing only infrastructure that can certify
whether autonomous search is blocked, locally ready, Modal-smoke ready, or
waiting on explicit human-approved live action.

This goal does not start autonomous research search. It does not fabricate
benchmark results, Arrow files, Modal runs, baseline metrics, gate verdicts,
Discovery Ledger records, or validation labels.

## Current State: 2026-05-30

The original four Modal pre-run readiness implementation PRs are merged:

- PR #23 `feat/modal-prerun-contracts`
- PR #24 `feat/falsification-and-ledgers`
- PR #25 `feat/baseline-scorer-readiness`
- PR #26 `feat/prerun-readiness-report`

Follow-up readiness hardening PRs are also merged:

- PR #27 `fix: close PR 25 readiness gaps`
- PR #28 `fix: harden modal gate and ledger boundaries`
- PR #29 `fix: add human baseline lock command`
- PR #31 `fix: support live Modal asset handoff audit`
- PR #38 `feat: add approved gate calibration command`
- PR #40 `Add LLM phase policy`

The current readiness-live branch was rebased onto PR #40.

## Current State: Live Baseline Locked

The approved pre-search readiness actions made major progress:

- local NanoFold fixture materialization completed with approval token
  `I_APPROVE_LOCAL_NANOFOLD_FIXTURE`;
- local gates now pass on the approved local-only fixture:
  `parameter_count`, `tiny_forward`, and `finite_loss`;
- Modal asset audit passed with `--search-ready`;
- the Modal app deployed successfully after fixing deploy-time Volume mount,
  source packaging, scorer dependency, and locked manifest shape issues;
- the real Modal baseline ran with approval token `I_APPROVE_BASELINE_RUN`;
- the baseline lock dry-run passed;
- `lock-baseline` wrote the real locked baseline under `runs/baseline/**`.

Locked baseline evidence:

- trial id: `T000`;
- candidate id: `baseline_auto_tiny`;
- split: `public_val_small`;
- primary metric: `best_val_calpha_lddt`;
- score: `0.07941230438543605`;
- scorer version: `calpha_lddt_v1`;
- locked asset version: `event-small-bootstrap-2026-05-30`;
- `official_benchmark_result=true`;
- `local_only=false`;
- `max_templates=0`.

This baseline is no-LLM deterministic baseline evidence. It was produced by the
Modal trial/scorer path and locked through the approved baseline-lock procedure;
no autonomous search was started.

## Operational Check Results

Live Modal asset handoff audit now passes:

```bash
.venv/bin/python -m autoalphafold3.agent audit-modal-assets --search-ready
```

Observed result: `PASS`.

The audit verified readable helper Arrow files on the Modal Volume, valid
feature fingerprints, valid scorer metadata, and the locked/public Volume
boundary.

Readiness report still blocks autonomous search:

```bash
.venv/bin/python -m autoalphafold3.agent readiness-report --config-path configs/nanofold_dev_cpu_smoke.json
```

Observed result: `autonomous_search_ready: false`.

The report now passes baseline lock and local gates. It still blocks autonomous
search because real Falsification Gate calibration evidence is missing.

## Remaining Human/Live Blockers

1. Falsification Gate calibration evidence is missing.
   The readiness report correctly refuses to accept a missing
   `runs/falsification_gate_calibration.json`. The committed `calibrate-gate`
   command can validate and freeze real known-null and known-positive evidence,
   but no committed command currently produces that evidence. Hand-writing it
   would be fake gate evidence.

2. Readiness still reports Modal event authority as a pending live action in
   the offline report. The Modal app was deployed during the baseline run, but
   the readiness report does not currently persist a live deployment proof
   artifact. Autonomous search must remain blocked until the explicit search
   readiness action verifies event authority and gate calibration is complete.

## Done Standard

This goal is not complete until:

- all implementation PRs are merged,
- readiness blocks only on explicit human-approved live actions or passes,
- no fake data, fake metrics, fake gate verdicts, or fake Modal runs are
  created,
- the baseline is locked from real approved evidence,
- Falsification Gate calibration is real or readiness blocks search with the
  exact approval/command needed,
- final validation passes:

```bash
python3 -m pytest -p no:cacheprovider
python3 .claude/skill-evals/run_offline_evals.py
git diff --check
```
