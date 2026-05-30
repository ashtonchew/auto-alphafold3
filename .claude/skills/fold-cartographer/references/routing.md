# Fold Cartographer Routing

Choose exactly one target.

## Target Rules

Prefer `stability_compute` when any of these dominate:

- `nan_oom_status` is not `ok`.
- runtime, peak GPU memory, parameter count, timeout, or shape errors exceed
  the current cap.

Prefer `distogram_good_lddt_flat` when:

- distogram/contact metrics improve or look strong.
- `best_val_calpha_lddt` is flat.
- coordinate loss or `distogram_coordinate_gap` suggests pair signal is not
  becoming 3D structure.

Prefer `long_range_topology_weak` when:

- long-range contact precision is weak, especially sequence separation >= 24.
- local geometry is acceptable enough that global topology is the clearer miss.

Prefer `local_geometry_weak` when:

- local lDDT or short-range contact precision is weak.
- failures look like loops, termini, short-range geometry, or backbone
  regularity problems.

## Tie Breaks

If multiple targets are plausible:

1. choose `stability_compute` if the trial cannot run reliably.
2. choose `distogram_good_lddt_flat` if pair/contact evidence is good but C-alpha
   lDDT is flat.
3. choose `long_range_topology_weak` before `local_geometry_weak` when local
   metrics are acceptable and long-range contacts are poor.
4. otherwise choose `local_geometry_weak`.

## Move Family Hints

- `local_geometry_weak`: `geometry_loss`, `recycling`, `curriculum`.
- `long_range_topology_weak`: `pairformer_attention`, `auxiliary_loss`,
  `curriculum`.
- `distogram_good_lddt_flat`: `geometry_loss`, `recycling`,
  `diffusion_sampler_golf`.
- `stability_compute`: `memory_runtime`, `width_depth`,
  `optimizer_scheduler`.
