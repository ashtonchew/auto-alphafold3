# One-Batch Checkpoint Runbook

This runbook produces the minimum defensible frozen checkpoint for sampler-only
trials. It is intentionally tiny: exactly one NanoFold training batch,
`diffusion_steps=1`, and `max_templates=0`.

## Plan Without Writing

```bash
python3 -m autoalphafold3.agent run-one-batch-checkpoint --mode dry-run
```

The dry-run prints the bounded plan and writes nothing.

By default the command uses
`autoalphafold3-data:/features/nanofold_event_small_no_templates.arrow`, because
that is the full no-template NanoFold feature artifact. The smaller
`train_tiny.arrow` helper is not a complete `ChainDataset` training input.

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

## Current Checkpoint State

PR #44 produced the approved live one-batch checkpoint on Modal:

- status: `CHECKPOINT_READY`
- trial id: `T010`
- checkpoint path: `/mnt/autoalphafold3/runs/trials/T010/checkpoint.pt`
- checkpoint sha256:
  `33a18f2a1595034b5a47018b47ae5733cd9e89df8f90e5664c953e928c5f1510`
- checkpoint size: `2666489` bytes
- feature input:
  `/mnt/autoalphafold3/features/nanofold_event_small_no_templates.arrow`
- config: `configs/nanofold_dev_cpu_smoke.json`
- `real_training_performed=true`
- `training_steps=1`
- `diffusion_steps=1`
- `max_templates=0`
- `starts_search=false`
- `writes_baseline=false`
- `writes_ledger=false`
- `writes_discovery_ledger=false`

The Modal Volume also contains:

- `/runs/trials/T010/checkpoint.pt`
- `/runs/trials/T010/checkpoint_manifest.json`
- `/runs/trials/T010/DONE`

The local repository records the returned manifest at
`runs/trials/T010/checkpoint_manifest.json`. The checkpoint itself remains in
the `autoalphafold3-data` Modal Volume and is not checked into git.

## Next Steps

Use this checkpoint only as a frozen sampler input for approved sampler-only
trials. Before a trial consumes it, verify that the manifest exists and still
matches the path, sha256, `real_training_performed=true`, `training_steps=1`,
`diffusion_steps=1`, and `max_templates=0`.

After PR #44 is merged, the next readiness step is to run the normal
pre-search readiness report from `main`. If readiness remains green, the system
has a real frozen checkpoint available without having started autonomous search.

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
