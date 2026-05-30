# AutoAlphaFold3 Pre-Run Readiness Plan

## Current Stack State

- PR #8 `feat/foundation-contracts` -> `main`: open draft.
- PR #10 `feat/nanofold-adapters` -> `feat/foundation-contracts`: open draft.
- PR #11 `feat/trial-runtime` -> `feat/nanofold-adapters`: open draft.
- PR #12 `feat/modal-scorer-orchestrator` -> `feat/trial-runtime`: open draft.
- PR #14 `feat/prerun-readiness-goal-prompt` -> `feat/modal-scorer-orchestrator`: open draft.

## Required Context Read

- `AGENTS.md`
- `README.md`
- `docs/framing.md`
- `program.md`
- `autoalphafold3/benchmark_contract.md`
- `autoalphafold3/editable_surface.md`
- `docs/runbooks/manifest_locking.md`
- `docs/runbooks/modal_control_plane.md`
- `docs/runbooks/nanofold_pin.md`
- `docs/spec/autoalphafold3-canonical (2).html` Sections 2-7
- `.claude/skills/modal-docs/SKILL.md`

## Feature Stack

1. `feat/falsification-contracts`
2. `feat/baseline-readiness`
3. `feat/discovery-ledger`
4. `feat/two-stage-orchestrator`
5. `feat/gate-wave-modal-adapter`
6. `feat/pre-run-readiness-cli`

## PR 1: `feat/falsification-contracts`

Grounding was performed with three read-only subagents:

- Schema/spec grounding: implement structured pre-registration, `FalsificationPlan`, `FalsificationResult`, verdict enums, discovery status fields, and pure verdict math in local modules.
- Test grounding: use synthetic in-memory fixtures only, cover all five verdicts, invalid prediction schemas, and missing/non-finite scored controls.
- Modal/spec grounding: do not implement Modal lookup, spawn, polling, map, starmap, or resource changes in this PR. Keep the contracts adapter-friendly for a later fakeable gate-wave adapter.

Best-practice approach:

- Keep the slice pure and local: no Modal imports, no filesystem side effects in verdict logic, no benchmark/data artifacts, no baseline metrics, and no Discovery Ledger writes.
- Add structured `RegisteredPrediction` to `AutoFoldTrial` instead of accepting a free-text prediction.
- Add `FalsificationAxis`, `PredictionDirection`, `FalsificationVerdict`, `DiscoveryStatus`, `FalsificationPlan`, and `FalsificationResult` schemas in `autoalphafold3/schema.py`.
- Add pure gate math in `autoalphafold3/falsification.py`, rejecting missing, non-finite, or incomplete control evidence rather than inventing verdicts.
- Extend `AutoFoldResult` with discovery/falsification fields while preserving the rule that a provisional `KEEP` is not a confirmed discovery.
- Add focused tests under `tests/test_falsification.py` plus fixture updates in existing local/Modal contract tests.
- Do not touch `autoalphafold3/modal_app.py`, `autoalphafold3/scorer/**`, public validation manifests, labels, fingerprints, cached features, or `runs/baseline/**`.
