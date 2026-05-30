# May 30 Modal Data Handoff

This document records the May 30, 2026 event-small data handoff. It is not an event-trial recipe and it is not a request to rebuild features during the hackathon.

The hackathon-start contract is:

- `autoalphafold3-data` contains public raw mmCIFs, cached Arrow features, provenance, fingerprints, runs, and renders.
- `autoalphafold3-locked` contains scorer-only manifests, public validation labels, and scorer metadata.
- Event search workers read cached features and write only trial-scoped artifacts.
- Event search workers must not rebuild shared feature artifacts, change split membership, mount locked labels, or mutate either shared Volume.

## Verified Layout

```text
autoalphafold3-data
  /raw/mmcif/*.cif
  /features/nanofold_event_small_no_templates.arrow
  /features/train_tiny.arrow
  /features/public_val_small.arrow
  /features/feature_fingerprints.json
  /runs/
  /provenance.json

autoalphafold3-locked
  /manifests/train_tiny.json
  /manifests/public_val_small.json
  /labels/public_val_labels.arrow
  /scorer_version.txt
```

The active May 30 split is 32 `train_tiny` chains and 16 `public_val_small` chains. `train_small` and hidden validation are not active in this handoff.

The public data Volume must not contain a `/locked` directory. Local staging paths under `locked/`, if recreated by a data-owner rebuild, must upload to `autoalphafold3-locked` without preserving the `locked/` prefix.

## Verification

Use the strict asset audit:

```bash
python3 -m autoalphafold3.agent audit-modal-assets --search-ready
```

Expected result:

- `status: PASS`
- `locked_asset_layout: separate_locked_volume`
- `official_lock_boundary: true`
- `split_counts.train_tiny: 32`
- `split_counts.public_val_small: 16`
- required data and locked files present

## Rebuild Boundary

Any future data rebuild is a separate data-owner task. It must produce a new immutable artifact set, update fingerprints, and pass the same two-Volume audit before any agent trial consumes it.

Do not run full PDB/mmCIF/MSA feature rebuilding from event search workers. Do not provision, search, or mutate PDB70/template databases. Official runs remain pinned to `max_templates=0`.
