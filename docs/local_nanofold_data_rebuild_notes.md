# Local NanoFold Data Rebuild Notes

The May 30, 2026 hackathon-start state does not require local data rebuilding. The event-small no-template assets are already uploaded to Modal and verified by `python3 -m autoalphafold3.agent audit-modal-assets --search-ready`.

This file exists only to define the boundary for any future data-owner rebuild. It should not be used as an event-trial checklist.

## Current Contract

- Use `autoalphafold3-data` for public raw mmCIFs, cached Arrow features, provenance, fingerprints, runs, and renders.
- Use `autoalphafold3-locked` for scorer-only manifests, public validation labels, and scorer metadata.
- Keep local `locked/` paths as staging only; they must upload into `autoalphafold3-locked` without preserving the `locked/` prefix.
- Keep official feature artifacts no-template: `max_templates=0`.
- Do not add `train_small` or hidden validation as active hackathon-start scope.

## Event-Small Scope

```text
train_tiny: 32
public_val_small: 16
```

The verified Modal artifact is:

```text
autoalphafold3-data:/features/nanofold_event_small_no_templates.arrow
```

It contains 48 records, empty template columns, and public feature rows aligned to the locked manifests.

## Rebuild Boundary

A future rebuild requires explicit human approval and must be treated as a data-owner operation outside agent search trials. A rebuild must:

- avoid full AlphaFold database downloads;
- avoid PDB70/template database provisioning;
- keep `max_templates=0`;
- write a new immutable feature artifact rather than mutating the current one;
- refresh provenance and fingerprints;
- upload public files to `autoalphafold3-data`;
- upload manifests and labels to `autoalphafold3-locked`;
- pass `python3 -m autoalphafold3.agent audit-modal-assets --search-ready`.

During the hackathon search loop, use cached features only.
