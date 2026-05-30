# First Sampler-Only Trial

`trials/T011.json` is the first prepared sampler-only trial after PR #44. It is
pre-registered against the frozen one-batch checkpoint from `T010`:

- checkpoint path: `/mnt/autoalphafold3/runs/trials/T010/checkpoint.pt`
- checkpoint sha256:
  `33a18f2a1595034b5a47018b47ae5733cd9e89df8f90e5664c953e928c5f1510`
- feature source:
  `/mnt/autoalphafold3/features/nanofold_event_small_no_templates.arrow`
- `training_steps=1`
- `diffusion_steps=1`
- `max_templates=0`

The trial is sampler-only: it must not train, update weights, or set
`max_steps`. Its purpose is to vary inference-time sampler behavior using the
frozen checkpoint, then score the produced trial artifacts through the locked
scorer path.

## Current Execution State

This PR prepares the trial request, proves local preflight, and adds the first
real frozen-checkpoint sampler worker path. It does not submit Modal work or
start autonomous search.

The sampler worker validates the checkpoint manifest and sha256, loads the
NanoFold checkpoint, runs an inference-only sample pass, and writes
trial-scoped artifacts:

- `predictions.json`
- `artifact_manifest.json`
- `sampler_manifest.json`
- `training_log.json`
- `stdout.log`
- `stderr.log`
- `patch.diff`
- `DONE`

## Next Implementation Needed

Before `T011` is used in an autonomous burst, run one human-approved Modal smoke
submission for this exact trial and scorer path. That live check must prove that
the public feature artifact used by the sampler produces complete
`public_val_small` target coverage for the locked scorer.

If the scorer reports missing targets, the next fix is to align sampler feature
selection with the locked public manifest. Do not fill missing predictions from
labels and do not fabricate coordinates.

The sampler worker must continue to:

- validate the checkpoint manifest and sha256 before use
- load the frozen checkpoint from `autoalphafold3-data`
- run inference only, with `max_templates=0`
- write artifacts only under `/runs/trials/T011/`
- avoid mounting locked labels
- return scorer-compatible prediction artifacts

Only after that live smoke passes should autonomous search submit a sampler
candidate burst.

## Live Smoke Result

After PR #45 merged, `T011` was submitted as the first live sampler smoke. It
proved the worker could write sampler artifacts, but the first implementation
only emitted one prediction and the locked scorer reported missing public
targets.

The follow-up smoke `T012` used the same frozen checkpoint with complete
`public_val_small` target coverage:

- trial id: `T012`
- checkpoint path: `/mnt/autoalphafold3/runs/trials/T010/checkpoint.pt`
- sampler artifacts: `/mnt/autoalphafold3/runs/trials/T012/`
- prediction count: `16`
- scorer status: `SCORED`
- `best_val_calpha_lddt`: `0.008276756926787072`
- `num_targets`: `16`
- `num_scored_targets`: `16`
- `num_failed_targets`: `0`
- `official_benchmark_result`: `true`

This is a successful live smoke of the sampler/scorer path. It is not a
discovery, not a provisional KEEP, and not autonomous search.
