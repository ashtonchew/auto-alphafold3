# One-Batch Checkpoint Runbook

This runbook produces the minimum defensible frozen checkpoint for sampler-only
trials. It is intentionally tiny: exactly one NanoFold training batch,
`diffusion_steps=1`, and `max_templates=0`.

## Plan Without Writing

```bash
python3 -m autoalphafold3.agent run-one-batch-checkpoint --mode dry-run
```

The dry-run prints the bounded plan and writes nothing.

## Run On Modal

After the Modal app is deployed, run the checkpoint producer with explicit
approval:

```bash
python3 -m autoalphafold3.agent run-one-batch-checkpoint \
  --mode modal \
  --approve I_APPROVE_ONE_BATCH_CHECKPOINT
```

The Modal worker writes trial-scoped artifacts under
`autoalphafold3-data:/runs/trials/T010/`, including:

- `checkpoint.pt`
- `checkpoint_manifest.json`
- `DONE`

The local command records the returned manifest at
`runs/trials/T010/checkpoint_manifest.json`.

## Claim Boundary

This is a real NanoFold-style AlphaFold3-lite checkpoint because the worker
initializes NanoFold, runs one optimizer step on one training batch, and saves
the actual model, optimizer, scheduler, and scaler state.

It is not a benchmark claim, not a good-model claim, and not autonomous search.
It must not write `runs/baseline/**`, the canonical ledger, the Discovery
Ledger, locked labels, scorer assets, cached feature outputs, fingerprints, or
Modal resource policy.

Sampler-only trials may use the returned `checkpoint_path` only after the
manifest is present, has `real_training_performed=true`, `training_steps=1`,
`diffusion_steps=1`, `max_templates=0`, and a valid `checkpoint_sha256`.
