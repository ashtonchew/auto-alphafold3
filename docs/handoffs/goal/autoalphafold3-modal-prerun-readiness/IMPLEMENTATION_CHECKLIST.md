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
- [x] Autonomous search remained blocked.
- [x] No fake baseline metrics, fake Modal runs, fake gate verdicts, fake
      discovery records, or fake benchmark artifacts were created.

## Pending Infrastructure

## Pending Human Live Actions

- [ ] Review Modal deployment cost/resource tier and approve or defer deploy.
- [ ] Deploy Modal event authority after approval.
- [ ] Materialize the local-only NanoFold fixture only with the exact approval
      token if local gate evidence is needed in this checkout.
- [ ] Produce real baseline source artifacts through the approved baseline
      procedure: `run-baseline --mode modal --approve I_APPROVE_BASELINE_RUN`.
- [ ] Run `lock-baseline --dry-run` on the real baseline source artifacts.
- [ ] Run `lock-baseline --approve` only if dry-run passes and approval is
      explicit.
- [ ] Run real Falsification Gate calibration:
      `calibrate-gate --mode from-evidence --approve I_APPROVE_GATE_CALIBRATION`.
- [ ] Approve autonomous search start only after readiness is no longer blocked.

## Validation To Rerun After Next PRs

```bash
python3 -m pytest -p no:cacheprovider
python3 .claude/skill-evals/run_offline_evals.py
git diff --check
.venv/bin/python -m autoalphafold3.agent audit-modal-assets --search-ready
.venv/bin/python -m autoalphafold3.agent readiness-report --config-path configs/nanofold_dev_cpu_smoke.json
```
