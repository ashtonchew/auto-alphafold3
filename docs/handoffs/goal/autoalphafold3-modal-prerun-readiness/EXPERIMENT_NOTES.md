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

## Live Baseline Lock: 2026-05-30

The readiness-live branch produced and locked real baseline evidence after PR
#40 was merged to `main`.

Commands run:

```bash
.venv/bin/python -m autoalphafold3.agent materialize-local-fixture --approve I_APPROVE_LOCAL_NANOFOLD_FIXTURE --overwrite
.venv/bin/python -m autoalphafold3.agent audit-modal-assets --search-ready
.venv/bin/python -m modal deploy --strategy recreate autoalphafold3/modal_app.py
.venv/bin/python -m autoalphafold3.agent run-baseline --mode modal --approve I_APPROVE_BASELINE_RUN
.venv/bin/python -m modal volume get autoalphafold3-data features/feature_fingerprints.json /private/tmp/feature_fingerprints.json
.venv/bin/python -m autoalphafold3.agent lock-baseline --source-dir runs/trials/T000 --feature-fingerprints /private/tmp/feature_fingerprints.json --approve I_APPROVE_BASELINE_LOCK --dry-run
.venv/bin/python -m autoalphafold3.agent lock-baseline --source-dir runs/trials/T000 --feature-fingerprints /private/tmp/feature_fingerprints.json --approve I_APPROVE_BASELINE_LOCK
```

Result:

- `runs/trials/T000/metrics.json` and `error_report.json` contain the
  trial-scoped source evidence;
- `runs/baseline/metrics.json`, `error_report.json`, and
  `feature_fingerprints.json` contain the locked baseline;
- baseline score is `0.07941230438543605` on `best_val_calpha_lddt`;
- scorer evidence is official and non-local:
  `official_benchmark_result=true`, `local_only=false`;
- official NanoFold-style AlphaFold3-lite baseline policy is preserved:
  `max_templates=0`;
- this was a deterministic no-LLM baseline path, not autonomous search.

Modal best-practice fixes applied during the run:

- mount the `autoalphafold3-data` Volume once per function and use subpaths
  inside the mount instead of mounting the same Volume multiple times;
- avoid read-only subpath creation failures by mounting the data Volume root
  read-only for scorer access;
- include scorer runtime dependencies in the Modal image;
- bake the local control-plane Python package into the image for deterministic
  readiness runs and deploy with `--strategy recreate` when stale warm
  containers would otherwise serve old code;
- keep `scorer_version=calpha_lddt_v1` as the scoring-code version and record
  the locked asset stamp separately.

## Current Blocker Analysis

### Baseline Lock

The repo now has a `lock-baseline` command, but this command intentionally
freezes evidence; it does not create evidence.

Status: resolved. Real source artifacts exist under `runs/trials/T000/` and
the locked baseline exists under `runs/baseline/`.

### Modal Event Authority

The Modal app is not deployed in the `main` environment. The local deploy plan
exists, but deployment is a real infrastructure action with cost/resource
implications.

Status: partially resolved for baseline. The app was deployed for the approved
baseline run. The offline readiness report still treats event authority as a
pending live action because no persistent deployment-proof artifact is recorded
for autonomous search readiness.

### Local Gate Fixtures

Torch is available and parameter-count readiness passes. The remaining local
gate issue is approved feature input materialization for `tiny_forward` and
`finite_loss`.

Status: resolved locally. The approved local-only NanoFold fixture was
materialized and local gates pass.

### Falsification Gate Calibration

The readiness report correctly refuses missing calibration evidence.

Current blocker:

- `calibrate-gate` exists and can validate/freeze evidence;
- no committed command currently produces real known-null and known-positive
  calibration evidence;
- no real `runs/falsification_gate_calibration.json` exists;
- manually creating that file would be fake gate evidence.

## Next Agent Guidance

Keep the next PRs small and operational:

1. merge the readiness-live baseline lock PR,
2. add or run a real known-null/known-positive calibration evidence producer,
3. run `calibrate-gate --mode from-evidence` only on real evidence,
4. rerun readiness and only then consider an explicit autonomous-search start.

Do not merge or mark readiness complete based on placeholders. Do not start
search until readiness passes and the human explicitly approves the live search
action.
