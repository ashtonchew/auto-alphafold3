# Foundation Implementation Plan

## Grounding

- Current `main` was pulled on 2026-05-30 and was already up to date.
- PR #6 is merged: `6f4f6df`, merged 2026-05-30T18:20:55Z.
- Required docs were read in order: `README.md`, `docs/framing.md`,
  `program.md`, `autoalphafold3/benchmark_contract.md`,
  `autoalphafold3/editable_surface.md`, `docs/runbooks/`, canonical spec, and
  `.claude/skills/*/SKILL.md`.
- The repo-local `modal-docs` skill was read at
  `.claude/skills/modal-docs/SKILL.md`. It points to local snapshots of Modal
  docs for App/Function/Volume/Image/lifecycle APIs. Later Modal runtime work
  must cite those snapshots or official Modal docs instead of guessing APIs.

## Feature Stack

1. `feat/foundation-contracts`
   - Goal pack and checklist.
   - Official feature/data contract decision.
   - Modal asset readiness hardening for required files, provenance,
     fingerprints, scorer stamp, manifest counts, public-data lock-boundary,
     and Arrow readability when byte fixtures and `pyarrow` are available.
   - PR #6 Section 7.5 requirements recorded for later Modal implementation.

2. `feat/nanofold-adapters`
   - Repo-root-aware NanoFold adapters.
   - Explicit manifest-driven dataset boundaries.
   - No random split leakage in official mode.
   - Training label access limited to train splits.
   - Public validation inference remains label-free.
   - `max_templates=0` and empty template placeholder checks.

3. `feat/trial-runtime`
   - Fixed-budget runner boundary.
   - Trial-scoped artifact writer.
   - Canonical prediction artifact writer.
   - Local/mock paths remain explicitly non-official.

4. `feat/modal-scorer-orchestrator`
   - Real Modal app/classes/functions.
   - Trial/scorer worker separation.
   - Section 7.5 defaults exactly as specified: scorer and TrialRunner
     `min_containers=1`, scorer `scaledown_window=600`, TrialRunner
     `scaledown_window=300`, and scorer `@modal.concurrent(max_inputs=4,
     target_inputs=2)`.
   - Scorer-only locked-label scoring.
   - Fold Cartographer canonical diagnostics.
   - Modal error normalization, lifecycle validation, append-only ledger
     hardening, and CLI strict preflight/git-diff enforcement.

5. Optional `feat/readiness-baseline-gates`
   - Use only if readiness/baseline-freezing templates and end-to-end mocked
     validation become too large for PR 4.

## Official Feature Schema Decision

The official event feature contract is the canonical no-template NanoFold IPC
bundle in `autoalphafold3-data`:

- `/features/nanofold_event_small_no_templates.arrow` contains 48 records.
- `/features/train_tiny.arrow` contains 32 train records.
- `/features/public_val_small.arrow` contains 16 public-validation records.
- Template fields exist only as empty placeholders.
- Official runs are pinned to `max_templates=0`.

The locked scorer contract is separate:

- `autoalphafold3-locked:/manifests/train_tiny.json`
- `autoalphafold3-locked:/manifests/public_val_small.json`
- `autoalphafold3-locked:/labels/public_val_labels.arrow`
- `autoalphafold3-locked:/scorer_version.txt`

Trial/sampler/debug workers may consume public features and write
trial-scoped artifacts, but they must not mount or read locked validation
labels. Scorer-only workers may mount locked assets.

## Validation Log

- Passed: `python3 -m pytest tests/test_modal_assets.py tests/test_local_contracts.py -q`.
- Passed: `python3 -m pytest tests/test_modal_and_demo.py tests/test_nanofold_adapter.py -q`.
- Passed: `python3 -m pytest -p no:cacheprovider`.
- Passed: `python3 .claude/skill-evals/run_offline_evals.py`.

## PR 2 Grounding: `feat/nanofold-adapters`

Read-only grounding covered the canonical spec data boundary, `nanofold_adapter`,
`locked_dataset`, NanoFold `ChainDataset.construct_datasets`, no-template
verification scripts, config contracts, and adapter tests.

Best-practice approach:

- Keep official dataset-boundary behavior in `autoalphafold3/nanofold_adapter.py`.
- Reuse `locked_dataset.py` for manifest hashing, random-split rejection, and
  train-only label access instead of duplicating scorer lock logic.
- Add explicit official-mode validation that requires train/public-validation
  manifests, refuses random split behavior, and enforces `max_templates=0`.
- Make NanoFold path discovery repo-root-aware and remove personal absolute
  fallback paths from verification scripts.
- Verify empty-template support by source inspection and by no-template feature
  script behavior where real Arrow fixtures are unavailable.

Likely files:

- `autoalphafold3/nanofold_adapter.py`
- `scripts/verify_nanofold_no_template_features.py`
- `scripts/nanofold_preprocess_no_templates.py`
- `configs/nanofold_dataset_local.json`
- `tests/test_nanofold_adapter.py`
- `tests/test_local_contracts.py`
- goal-pack checklist/validation notes

Validation:

- Passed: `python3 -m pytest -p no:cacheprovider tests/test_nanofold_adapter.py tests/test_local_contracts.py -q`.
- Passed: `python3 -m pytest -p no:cacheprovider tests/test_modal_and_demo.py tests/test_prepare_nanofold_data.py -q`.
- Passed: `python3 -m pytest -p no:cacheprovider tests/test_runner_and_locked_scorer.py -q`.
- Passed: `rg -n "/Users/|naveenramasamy" autoalphafold3 scripts configs` with no matches.
- Passed: `python3 -m pytest -p no:cacheprovider`.
- Passed: `python3 .claude/skill-evals/run_offline_evals.py`.
