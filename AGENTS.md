# AGENTS.md

This repo is the May 30, 2026 hackathon-start foundation for a NanoFold-style AlphaFold3-lite autoresearch system. Before writing code, first explore the project structure, then invoke the `modal-docs` skill for Modal documentation.

Use `docs/spec/autoalphafold3-canonical (2).html` as the canonical design if contract drift appears.

## Read Order

1. `README.md`
2. `docs/framing.md`
3. `program.md`
4. `autoalphafold3/benchmark_contract.md`
5. `autoalphafold3/editable_surface.md`
6. `docs/runbooks/`

## Repo Layout

- `autoalphafold3/`: local control plane, contracts, scorer, adapters, and orchestration code.
- `autoalphafold3/scorer/`: locked scoring code. Do not edit during search.
- `external/nanofold/`: pinned NanoFold working-tree copy. Search edits are limited to the exact paths listed in `autoalphafold3/editable_surface.md`.
- `configs/`: baseline and experiment configs; `configs/experiments/` is the future agent-editable config area.
- `data/manifests/`, `data/fingerprints/`, and `data/toy/`: small metadata and toy fixtures only. Do not fabricate entries.
- `docs/`: framing, runbooks, data notes, and canonical spec.
- `trials/`: typed trial JSON files.
- `runs/`: generated run artifacts and ledger output. Event runs are append-only.
- `tests/`: local tests for scorer, contracts, Modal asset policy, and runner logic.

## Commands

- Run scorer tests: `python3 -m pytest tests/test_calpha_lddt.py`
- Run all local repo tests: `python3 -m pytest -p no:cacheprovider`
- Verify NanoFold pin/map: `python3 -m pytest tests/test_nanofold_adapter.py`
- Approved future experiment entrypoint: `python3 -m autoalphafold3.agent submit trials/T###.json`

There is no project-level formatter, linter, type checker, or packaging command configured yet. If one is added, document it here.

## Constraints

- This project does not train, reproduce, improve, or beat Google DeepMind AlphaFold3.
- Use the phrase NanoFold-style AlphaFold3-lite for the implementation target.
- Do not create fake benchmark results, fake Arrow files, fake Modal runs, or fake baseline metrics.
- Do not run full MSA/database feature rebuilding during the event.
- Pin official NanoFold runs to `max_templates=0`; do not provision, search, or mutate a template database.
- Use `autoalphafold3-data` for public data/runs and `autoalphafold3-locked` only for scorer/final-validation paths.
- Do not run upstream NanoFold download scripts or full feature rebuilding from `external/nanofold` without explicit approval.
- Do not start the hackathon search loop until baseline, scorer, manifests, Modal app, preflight, and ledger gates exist.

## Do Not Edit During Search

- `autoalphafold3/scorer/**`
- `autoalphafold3/benchmark_contract.md`
- `autoalphafold3/modal_app.py`
- public validation manifests
- validation labels, locked scorer assets, cached feature outputs, or fingerprints
- `runs/baseline/**`
- Modal GPU types, timeouts, `max_containers`, Volumes, or cost caps

## Done Means

For this foundation, done means the repo scaffold exists, the operating docs match the canonical spec, local tests pass, no process-log docs are present, and no unverified data or benchmark claims were created.
