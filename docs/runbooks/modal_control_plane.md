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

## Modal Event Authority Proof

After deployment, record the no-side-effect proof that the Modal-hosted trusted
orchestrator is reachable and owns event submission authority:

```bash
python -m autoalphafold3.agent audit-modal-authority --mode dry-run
python -m autoalphafold3.agent audit-modal-authority \
  --mode modal \
  --approve I_APPROVE_MODAL_EVENT_AUTHORITY
```

The live command calls `TrustedOrchestrator.authority_health` on the deployed
app and writes only `runs/modal_event_authority.json`. It does not submit
trials, start autonomous search, write `runs/baseline/**`, write the canonical
ledger, or write the Discovery Ledger.

## One-Batch Checkpoint

Sampler-only trials require a frozen checkpoint. The minimum approved path is
the one-batch checkpoint runbook:

```bash
python -m autoalphafold3.agent run-one-batch-checkpoint \
  --mode modal \
  --approve I_APPROVE_ONE_BATCH_CHECKPOINT
```

This uses `TrialRunner.run_checkpoint` and writes only trial-scoped checkpoint
artifacts under `autoalphafold3-data:/runs/trials/T010/`.

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
- Trial workers mount `autoalphafold3-data` once at `/mnt/autoalphafold3`
  because Modal rejects mounting the same Volume twice for one function. The
  worker reads features from `/mnt/autoalphafold3/features` and writes
  trial-scoped artifacts under `/mnt/autoalphafold3/runs/trials/<trial_id>/`.
- During event search, the Modal-hosted trusted orchestrator writes
  `runs/ledger.jsonl` and the Discovery Ledger.
- Local orchestrator ledger writes are smoke-only scaffold behavior before the
  Modal-hosted trusted orchestrator is deployed.
- Modal trial workers write only under `/runs/trials/<trial_id>/`.
- Workers return small JSON pointers, not large artifacts.
- GPU `min_containers` remains zero unless a human explicitly approves warm-pool cost.
- Trial, sampler, and debug workers do not mount locked labels.
- Scorer-only workers are the only workers allowed to mount `autoalphafold3-locked`.
