# AutoAlphaFold3 Modal Pre-Run Readiness Notes

## PR #31 Root Cause

The live Modal audit initially failed on Arrow readability because the Modal CLI
`volume get ... -` appends a success footer to binary stdout. The bytes were
Arrow IPC bytes, but the footer corrupted the stream before parsing.

The best-practice fix was not to add a broad Arrow stream/file fallback. The
fix was to preserve the binary payload and strip only the exact Modal CLI
success footer variants.

PR #31 also accepted the current handoff metadata shape for helper assets:

- `feature_fingerprints.json` can describe `data_files`,
- scorer metadata can use the bootstrap scorer stamp.

After PR #31, live audit passed.

## Current Blocker Analysis

### Baseline Lock

The repo now has a `lock-baseline` command, but this command intentionally
freezes evidence; it does not create evidence.

Current blocker:

- no real `runs/trials/<id>/metrics.json`,
- no real `runs/trials/<id>/error_report.json`,
- no real `runs/trials/<id>/feature_fingerprints.json`,
- no real approved baseline provenance to lock.

Creating these by hand would be fake baseline evidence.

### Modal Event Authority

The Modal app is not deployed in the `main` environment. The local deploy plan
exists, but deployment is a real infrastructure action with cost/resource
implications.

Current blocker:

- no deployed Modal app found with `modal app list --env main`,
- deployment approval should include cost/resource tier review.

### Local Gate Fixtures

Torch is available and parameter-count readiness passes. The remaining local
gate issue is approved feature input materialization for `tiny_forward` and
`finite_loss`.

Current blocker:

- live Modal Volume has readable helper Arrow files,
- local readiness does not yet have a committed approved path to use those
  helper files.

### Falsification Gate Calibration

The readiness report correctly refuses missing calibration evidence.

Current blocker:

- no `calibrate-gate` command exists,
- no real `runs/falsification_gate_calibration.json` exists,
- manually creating that file would be fake gate evidence.

## Next Agent Guidance

Keep the next PRs small and operational:

1. approved local fixture materialization,
2. real baseline runner,
3. Falsification Gate calibration command.

Do not merge or mark readiness complete based on placeholders. Do not start
search until readiness passes and the human explicitly approves the live search
action.
