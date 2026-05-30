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

Local `main` was fast-forwarded after PR #31. The worktree was clean after the
pull.

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

The report currently classifies one item as `PASS_MOCKED_MODAL` and four items
as `PENDING_HUMAN_LIVE_ACTION`.

## Remaining Human/Live Blockers

1. Real baseline lock is missing.
   `runs/` contains no real baseline source artifacts. The committed
   `lock-baseline` command can freeze already-produced evidence, but it does
   not run NanoFold, score a baseline, or create baseline metrics.

2. Modal event authority is not deployed.
   `modal app list --env main` showed no deployed app. The deploy plan points
   to `modal deploy autoalphafold3/modal_app.py`, but deployment is a real
   infrastructure/cost action and the runbook says cost/resource tiers and
   baseline readiness must be reviewed first.

3. Local tiny-forward and finite-loss gates still need an approved cached Arrow
   fixture path.
   Torch is installed and parameter counting passes. The remaining issue is
   that the local readiness gate cannot yet consume approved helper Arrow files
   from Modal or another approved cache.

4. Falsification Gate calibration evidence is missing.
   The readiness report correctly refuses to accept a missing
   `runs/falsification_gate_calibration.json`. No committed command currently
   runs known-null and known-positive calibration and writes real calibration
   evidence.

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
