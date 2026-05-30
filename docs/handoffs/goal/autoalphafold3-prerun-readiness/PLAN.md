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
- Lock implemented falsification verdict logic from future search edits by adding `autoalphafold3/falsification.py` to patch-policy denial coverage and testing that rejection. Future PRs must do the same for implemented gate-control construction, thresholds, and Discovery Ledger write paths.
- Do not touch `autoalphafold3/modal_app.py`, `autoalphafold3/scorer/**`, public validation manifests, labels, fingerprints, cached features, or `runs/baseline/**`.

Updated PR #14 guardrail:

- The readiness CLI must not treat gate calibration placeholders as search-ready. Autonomous search remains blocked until known-null and known-positive Falsification Gate calibration is complete, or the report names the exact human-approved live calibration action still pending.
- Optional live readiness is read-only/smoke by default and must not write `runs/baseline/**`, locked Volumes, canonical ledgers, Discovery Ledger entries, benchmark artifacts, or baseline metrics without a separate human-approved baseline-lock procedure.

## PR 2: `feat/baseline-readiness`

Grounding was performed with two read-only subagents:

- Baseline/spec grounding: implement a pure local baseline lock reader/validator that checks real readiness evidence and refuses to invent a current best when the baseline is missing.
- Test grounding: use only `tmp_path` synthetic contract payloads, never toy smoke scorer outputs as official baseline evidence, and do not write under repo `runs/` or `runs/baseline/`.

Best-practice approach:

- Add a small `autoalphafold3/baseline_readiness.py` module with strict report objects and no scorer, Modal, or locked-label access.
- Validate baseline `metrics.json`, `error_report.json`, and feature-fingerprint evidence without creating or mutating those artifacts.
- Require official evidence: `official_benchmark_result=true`, `split=public_val_small`, `primary_metric=best_val_calpha_lddt`, `scorer_version=calpha_lddt_v1`, finite score in `[0, 1]`, manifest hashes, feature fingerprints, and `max_templates=0`.
- Return an explicit not-ready report when the baseline lock is missing. Do not default current best to `0.0`.
- Add `current_best_from_baseline_and_ledger(...)` that starts from a ready baseline and only upgrades to valid `KEEP` rows with finite higher scores.
- Defer readiness CLI aggregation to `feat/pre-run-readiness-cli`.

## PR 3: `feat/discovery-ledger`

Grounding was performed with two read-only subagents:

- Discovery/spec grounding: implement confirmed-only Discovery Ledger schema and helpers with full provenance and no worker write authority.
- Test grounding: use `tmp_path` JSONL fixtures only, reject provisional `KEEP` and killed gate evidence, require stable JSONL, and add patch-policy denial for the implemented write path.

Best-practice approach:

- Add strict `DiscoveryProvenance` and `DiscoveryRecord` schemas in `autoalphafold3/schema.py`.
- Add `build_discovery_record(...)`, `append_discovery_record(...)`, `read_discovery_ledger(...)`, and `validate_discovery_record(...)` helpers in a dedicated `autoalphafold3/discovery_ledger.py` module.
- Require `AutoFoldResult.discovery=CONFIRMED`, `status=KEEP`, and `falsification.verdict=CONFIRMED`; a provisional `KEEP` is never a discovery.
- Require provenance fields: git SHA, scorer version, primary metric, manifest hashes, feature fingerprints, baseline/current-best reference, pre-registered axis/direction/component, verdict numbers, and gate thresholds.
- Use deterministic JSONL writes with exact duplicate idempotency and conflicting duplicate rejection.
- Lock `autoalphafold3/discovery_ledger.py` and canonical `runs/discovery_ledger.jsonl` paths in patch-policy coverage before search.
- Do not write real Discovery Ledger records, benchmark artifacts, baseline metrics, canonical ledgers, or `runs/**` files during tests.

## PR 4: `feat/two-stage-orchestrator`

Grounding was performed with two read-only subagents:

- Orchestrator/spec grounding: implement local stage-one decision logic only; require a ready baseline/current-best; make `KEEP` provisional and ledger-visible as gate-required.
- Test grounding: use `tmp_path` baseline/ledger/discovery paths only; cover KEEP, DISCARD, FAIL, INFRA_FAIL, missing baseline refusal, lifecycle transitions, and Discovery Ledger refusal for provisional `KEEP`.

Best-practice approach:

- Add stage-one decision helpers to `autoalphafold3/orchestrator.py` without changing Modal submission behavior.
- Require `current_best_from_baseline_and_ledger(...)`; never default current best to `0.0`.
- Emit `KEEP` only as provisional with `discovery=UNCONFIRMED` and no falsification evidence.
- Emit `DISCARD` for valid scores that do not beat the configured threshold.
- Emit `FAIL` for invalid/non-finite scoring payloads and preserve `INFRA_FAIL` rows for infrastructure failures.
- Append only canonical ledger rows in this slice; Discovery Ledger writes remain exclusively through `autoalphafold3/discovery_ledger.py` after confirmed gate evidence.
- Keep tests synthetic and local; do not write real `runs/**`, baseline metrics, benchmark artifacts, Modal runs, gate verdict artifacts, or Discovery Ledger records.

## PR 5: `feat/gate-wave-modal-adapter`

Grounding was performed with two read-only subagents:

- Modal/spec grounding: implement a narrow, fakeable Modal boundary that uses deployed function lookup plus `Function.starmap` for orchestrator-authored gate controls, without `modal run`, resource overrides, Volume changes, or ledger writes.
- Test/patch-policy grounding: prove bounded control construction, exact `starmap` exception-handling options, deterministic infra-failure evidence, and patch-policy denial for the implemented gate-control module and gate artifact paths.

Best-practice approach:

- Add a dedicated gate-wave adapter module separate from pure verdict math and from `autoalphafold3/modal_app.py`.
- Represent only orchestrator-authored controls: knock-out, placebo, predicted-axis check, and bounded seed reruns derived from a `FalsificationPlan`.
- Reject unbounded control waves before Modal submission with fixed maximum variants, finite positive timeout, and bounded seed count.
- Use dependency injection/fakes in tests; do not require a live Modal deployment and do not write benchmark, baseline, canonical ledger, or Discovery Ledger artifacts.
- For Modal execution, call the deployed `TrialRunner` method with `modal.Cls.from_name(APP_NAME, "TrialRunner")().run.starmap(...)` using `order_outputs=True`, `return_exceptions=True`, and `wrap_returned_exceptions=False`.
- Normalize Modal SDK lookup, starmap setup, and per-control returned exceptions into structured `INFRA_FAIL` evidence. Do not convert partial evidence into a gate verdict or confirmed discovery.
- Lock the implemented gate-wave module and gate-wave artifact paths in patch-policy coverage before autonomous search.
