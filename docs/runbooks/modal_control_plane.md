# Modal Control Plane Runbook

This repo now encodes the Modal control-plane contract in `autoalphafold3/modal_app.py`, but local tests do not deploy or run GPU jobs.

## Static Local Check

```bash
python - <<'PY'
from autoalphafold3.modal_app import healthcheck, modal_deploy_plan
print(healthcheck())
print(modal_deploy_plan())
PY
```

Expected locally: `configured_not_deployed`, plus `modal_sdk_available` true or false depending on the environment. `modal_deploy_plan()` is contract metadata only; it does not prove the app has been deployed.

## Modal Asset Audit

Run the asset audit:

```bash
python -m autoalphafold3.agent audit-modal-assets
```

Run the search-readiness audit:

```bash
python -m autoalphafold3.agent audit-modal-assets --search-ready
```

Expected final layout:

- `autoalphafold3-data:/features/train_tiny.arrow`
- `autoalphafold3-data:/features/public_val_small.arrow`
- `autoalphafold3-data:/features/feature_fingerprints.json`
- `autoalphafold3-data:/provenance.json`
- `autoalphafold3-locked:/manifests/train_tiny.json`
- `autoalphafold3-locked:/manifests/public_val_small.json`
- `autoalphafold3-locked:/labels/public_val_labels.arrow`
- `autoalphafold3-locked:/scorer_version.txt`

## Verified May 30, 2026 Asset Preconditions

These data and lock-boundary conditions are already part of the hackathon-start contract:

- Cached Arrow features exist in `autoalphafold3-data`.
- Locked manifests, labels, and scorer assets are in `autoalphafold3-locked`.
- `python -m autoalphafold3.agent audit-modal-assets --search-ready` passes.

## Still Required Before Deployment

Do not deploy until these are true:

- NanoFold pin is verified.
- Baseline metrics are ready to freeze.
- Cost caps and resource tiers are reviewed.

## Contract

- Deploy once, call many.
- Local orchestrator writes `runs/ledger.jsonl`.
- Modal workers write only under `/runs/trials/<trial_id>/`.
- Workers return small JSON pointers, not large artifacts.
- GPU `min_containers` remains zero unless a human explicitly approves warm-pool cost.
- Trial, sampler, and debug workers do not mount locked labels.
- Scorer-only workers are the only workers allowed to mount `autoalphafold3-locked`.
