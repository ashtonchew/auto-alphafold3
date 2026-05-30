# Benchmark Contract

This benchmark is locked infrastructure. It defines what counts as a valid auto-AlphaFold3 AlphaFold3-lite trial.

## Locked Assets

Search-loop required assets:

- fixed train manifests in `data/manifests/`
- fixed public validation manifest in `data/manifests/`
- cached Arrow IPC feature files
- feature, label, and manifest SHA256 fingerprints
- locked validation labels outside the agent-editable training path
- scorer version stamp
- immutable baseline metrics and error report
- `max_templates=0` for every official baseline, trial, sampler run, and final validation

This first scaffold does not create real manifests, labels, Arrow files, baseline metrics, or Modal runs.

Modal storage follows the canonical spec:

- `autoalphafold3-data` mounted at `/mnt/autoalphafold3` for public data, reduced DBs, raw mmCIFs, Arrow features, run artifacts, checkpoints, logs, and renders.
- `autoalphafold3-locked` mounted at `/mnt/autoalphafold3-locked` only for scorer and final-validation code that must read locked manifests, labels, scorer code, or scorer metadata.
- Trial, sampler, and debug workers may read public features and write trial-scoped run artifacts, but they must not mount locked validation labels.

`autoalphafold3/modal_assets.py` audits the live Modal storage layout. The search-readiness gate requires `autoalphafold3-locked` to be present and populated.

## Trial Artifact Contract

`autoalphafold3/runner.py` defines the future `run_fixed_budget_trial(...)` interface used by Modal. In this local scaffold it can write a deterministic `artifact_manifest.json` only in explicit stub mode. The manifest always records `real_training_performed: false`; callers must not treat it as a NanoFold run, Modal job, checkpoint, Arrow feature load, or benchmark result.

The non-download runner lifecycle may also create local scaffold-only artifacts:

- `training_log.json` with `real_training_performed: false`
- empty `stdout.log` and `stderr.log`
- `patch.diff`
- `DONE` marker for local stub initialization only

These files prove directory layout and idempotency; they do not prove model execution.

Expected trial artifact paths:

- `artifact_manifest.json`
- `predictions.json`
- `training_log.json`
- `stdout.log`
- `stderr.log`
- `patch.diff`
- `checkpoint.pt`

## Scorer-Only Wrapper

`autoalphafold3/locked_scorer.py` represents the scorer-only boundary. It may read locked manifests and labels, and trial/sampler/debug workers must not call it during official search. The current implementation supports toy JSON prediction artifacts for local validation only and stamps outputs with `official_benchmark_result: false`.

The active scorer split is `public_val_small`. The local smoke split is allowed only when explicitly requested by tests or dry-run checks.

Local scorer outputs are written atomically to `metrics.json` and `error_report.json`. Toy prediction artifacts must use schema `autoaf3.predictions.v1`, must match the requested split, must not duplicate target IDs, and each `predicted_ca` must have shape `(L, 3)`.

## Scorer

Primary scorer: `calpha_lddt_v1`.

Rules:

- predicted C-alpha coordinates have shape `(L, 3)`
- target C-alpha coordinates have shape `(L, 3)`
- target resolved-label mask has shape `(L,)`
- eligible residue pairs require both target residues resolved and target distance less than 15 Angstrom
- thresholds are 0.5, 1.0, 2.0, and 4.0 Angstrom
- preservation is `abs(d_pred - d_true) < threshold`
- missing target residues are excluded from the denominator
- missing or NaN predictions count as non-preserved
- no structural superposition
- scorer uses float64 internally

## Primary Metric

`best_val_calpha_lddt` is the only objective metric.

Aggregate score is weighted by eligible pair count. Fold Cartographer routes only on `local_geometry_weak`, `long_range_topology_weak`, `distogram_good_lddt_flat`, and `stability_compute`; supporting losses and coarse length/MSA context do not replace the primary metric.

## Canonical Metrics Shape

Official trials must emit a `metrics.json` with:

- `schema_version`
- `scorer_version`
- `primary_metric`
- `status`
- `trial_id`
- `candidate_id`
- `seed`
- `split`
- manifest names and hashes
- aggregate C-alpha lDDT metrics
- quality gates
- Fold Cartographer summary
- artifact pointers

## Access Rules

- Training code may read training features and training labels.
- Training code may read public validation features for prediction.
- Only the locked scorer stage may read public validation labels.
- Workers write per-trial artifacts only.
- During event search, the Modal-hosted trusted orchestrator writes the
  canonical ledger and Discovery Ledger. Local ledger writes are scaffold
  smoke-test behavior only before deployment.
- Template tensors are empty placeholders; template DB provisioning/search is outside the official benchmark.

## Modal and Cost Rules

`autoalphafold3/modal_app.py` owns GPU type, timeouts, retry policy, Volume mounts, and `max_containers`. Agents must not edit those during search. GPU `min_containers` should remain zero unless a human explicitly approves a warm-pool cost decision.

`modal_deploy_plan()` exposes the intended deploy-once/call-many function contract for local validation without importing the Modal SDK. It is metadata only; it is not evidence of deployment or a benchmark run.

## Invalid Trial Conditions

A trial is invalid if it changes scorer math, validation membership, locked labels, manifest fingerprints, cached feature outputs, result parsing, baseline ledger, Modal resource caps, or ledger-writing authority.
