# Trial Schema

## Required Fields

An `AutoFoldTrial` must include:

- `trial_id`
- `parent_commit`
- `hypothesis`
- `move_family`
- `diagnostic_target`
- `prediction`
- `patch_path`
- `config_path`
- `budget`
- `seed`
- `max_steps`
- `max_wall_minutes`
- `train_features`
- `public_val_features`
- `trial_output_root`
- `scorer_version`
- `primary_metric`

## Allowed Values

`move_family` must be one of:

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

`diagnostic_target` must be one of:

- `local_geometry_weak`
- `long_range_topology_weak`
- `distogram_good_lddt_flat`
- `stability_compute`

`budget` must be one of:

- `smoke`
- `trial`
- `sampler`
- `debug`
- `final`

During broad search, reject `final` unless a human explicitly declares final
validation mode.

## Rejection Checks

Reject if any path or instruction touches:

- `autoalphafold3/modal_app.py`
- `autoalphafold3/orchestrator.py`
- `autoalphafold3/locked_scorer.py`
- `autoalphafold3/result_parser.py`
- `locked/`
- `runs/baseline/`
- validation split definitions
- raw validation labels
- preprocessed validation features
- Modal GPU, timeout, Volume, `max_containers`, or cost cap settings

Reject multiple trial submissions. Fanout is owned by the orchestrator or parent
research controller, never by this skill.

## Sampler Rule

For `move_family = diffusion_sampler_golf` or `budget = sampler`, require:

- frozen checkpoint.
- no training-weight updates.
- label-free sample selection.
- no hidden validation.
