# Research Contract

## Diagnostic Targets

Choose exactly one:

- `local_geometry_weak`: local lDDT or short-range contacts are weak.
- `long_range_topology_weak`: long-range contact precision is weak.
- `distogram_good_lddt_flat`: pair/contact signal improves but C-alpha lDDT
  stays flat, suggesting the 3D coordinate path is not using pair information.
- `stability_compute`: NaN, OOM, shape error, timeout, runtime, memory, or
  parameter count dominates.

## Move Families

Choose exactly one:

- `width_depth`
- `pairformer_attention`
- `diffusion_schedule`
- `diffusion_sampler_golf`
- `recycling`
- `auxiliary_loss`
- `geometry_loss`
- `curriculum`
- `optimizer_scheduler`
- `feature_handling`
- `memory_runtime`

## Allowed Surface

Allowed changes are limited to:

- `autoalphafold3/` patch modules that are not control plane, scorer, parser,
  locked-data, or baseline-ledger modules.
- selected NanoFold model/training modules listed in `editable_surface.md`.
- experiment configs, loss weights, optimizer/scheduler setup, curriculum,
  recycling, diffusion/noising schedule, sampler settings, label-free sample
  selection, Pairformer/attention variants, auxiliary heads, geometry losses,
  and memory/speed optimizations.

## Locked Surface

Never change:

- public or hidden validation split definitions.
- scoring script, metric computation, raw validation labels, locked Volume
  files, scorer version metadata, or result parser.
- preprocessed feature outputs or scripts that change benchmark fingerprints.
- `autoalphafold3/modal_app.py`, `autoalphafold3/orchestrator.py`, GPU type,
  timeout, `max_containers`, Volumes, or cost caps.
- `runs/baseline/` or the canonical ledger.

## Decision Rule

`KEEP` only when `best_val_calpha_lddt` improves over current best by the
configured threshold, no invalid evaluation explains the gain, and runtime,
memory, parameter count, NaN/OOM status, and shape checks remain within cap.

Use `DISCARD` for non-improving or incoherent patches, `FAIL` for model/training
path failures, and `INFRA_FAIL` for Modal worker, Volume, image, dependency, or
polling failures before trial logic runs.

## Stuck Rule

After 8 consecutive non-improving trials:

1. stop small variations of the same move.
2. switch to a move family not used in the last 20 trials.
3. switch diagnostic target if the current target was tried 4 times.
4. explain why the switch follows from diagnostics.
