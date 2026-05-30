# AutoAlphaFold3 Modal Pre-Run Readiness Experiments

No autonomous research experiments have been started.

This handoff tracks readiness and operational gate checks only. Any future
experiment or search trial must wait until readiness passes and a human
explicitly approves the live search action.

## Readiness Checks Run

### Live Modal Asset Handoff Audit

Command:

```bash
.venv/bin/python -m autoalphafold3.agent audit-modal-assets --search-ready
```

Observed result: `PASS`.

Meaning:

- helper Arrow files on the Modal Volume are readable,
- feature fingerprints are valid,
- scorer metadata is valid,
- public data and locked scorer boundaries are intact.

### Readiness Report

Command:

```bash
.venv/bin/python -m autoalphafold3.agent readiness-report --config-path configs/nanofold_dev_cpu_smoke.json
```

Observed result: `autonomous_search_ready: false`.

Meaning:

- readiness reporting is working,
- autonomous search remains blocked,
- remaining blockers are live/human or missing-command infrastructure items.

## Not Run

- No autonomous search loop.
- No baseline run.
- No baseline lock approval.
- No Falsification Gate calibration.
- No Discovery Ledger writes.
- No benchmark-result generation.
