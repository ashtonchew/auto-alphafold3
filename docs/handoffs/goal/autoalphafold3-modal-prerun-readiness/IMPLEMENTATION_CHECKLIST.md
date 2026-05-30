# AutoAlphaFold3 Modal Pre-Run Readiness Checklist

## Completed

- [x] PR #21 and PR #22 accepted before Modal pre-run readiness implementation.
- [x] PR #23 `feat/modal-prerun-contracts` merged.
- [x] PR #24 `feat/falsification-and-ledgers` merged.
- [x] PR #25 `feat/baseline-scorer-readiness` merged.
- [x] PR #26 `feat/prerun-readiness-report` merged.
- [x] PR #27 `fix: close PR 25 readiness gaps` merged.
- [x] PR #28 `fix: harden modal gate and ledger boundaries` merged.
- [x] PR #29 `fix: add human baseline lock command` merged.
- [x] PR #31 `fix: support live Modal asset handoff audit` merged.
- [x] Local `main` fast-forwarded after PR #31.
- [x] Live Modal asset handoff audit passed.
- [x] Readiness report run after PR #31.
- [x] Approved local fixture materialization command added for `tiny_forward`
      and `finite_loss` local gates.
- [x] Human-approved `run-baseline` command added to produce trial-scoped
      source artifacts for `lock-baseline`.
- [x] Human-approved `calibrate-gate` command added to write calibration only
      from known-null and known-positive real evidence records.
- [x] Approved local NanoFold fixture materialized for this checkout.
- [x] Local `parameter_count`, `tiny_forward`, and `finite_loss` readiness
      gates pass on approved local-only fixture evidence.
- [x] Modal app deployed for the approved baseline run without changing GPU
      type, timeouts, Volumes, max containers, min containers, warm pools, or
      cost caps.
- [x] Real Modal baseline run completed:
      `run-baseline --mode modal --approve I_APPROVE_BASELINE_RUN`.
- [x] Baseline lock dry-run passed.
- [x] Real baseline lock completed through `lock-baseline`; the only mutation
      under `runs/baseline/**` was made by that command.
- [x] Autonomous search remained blocked.
- [x] No fake baseline metrics, fake Modal runs, fake gate verdicts, fake
      discovery records, or fake benchmark artifacts were created.

## Pending Human Live Actions

- [ ] Produce real known-null and known-positive Falsification Gate calibration
      evidence. Current committed code validates/freezes evidence but does not
      produce it.
- [ ] Run real Falsification Gate calibration:
      `calibrate-gate --mode from-evidence --approve I_APPROVE_GATE_CALIBRATION`.
- [ ] Persist or verify Modal event authority for autonomous search readiness
      after calibration is complete.
- [ ] Approve autonomous search start only after readiness is no longer blocked.

## Validation To Rerun After Next PRs

```bash
python3 -m pytest -p no:cacheprovider
python3 .claude/skill-evals/run_offline_evals.py
git diff --check
.venv/bin/python -m autoalphafold3.agent audit-modal-assets --search-ready
.venv/bin/python -m autoalphafold3.agent readiness-report --config-path configs/nanofold_dev_cpu_smoke.json
```
