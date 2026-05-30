# Plan

## Grounding

- Read required repo docs and runbooks before code changes.
- Invoked the repo-local `modal-docs` skill and read deployed-app,
  deployed-function invocation, and Volume reference pages.
- Verified PR #21 and PR #22 are merged into `main`; implementation branches
  start from the merged `f057aab` base.
- Worktree was clean before branching; no branch rebase or force-push has been
  used.

## Feature 1: `feat/modal-prerun-contracts`

Read-only agent findings:

- Current code already uses Modal deployed-app patterns for mocked submit/poll:
  `modal.Cls.from_name(APP_NAME, "TrialRunner")`, `.run.spawn(...)`, and
  `modal.FunctionCall.from_id(...).get(timeout=0)`.
- Tests should stay offline with narrow fake Modal SDK objects.
- The highest-value gap is explicit trusted-harness and execution-worker role
  metadata, plus static secret-boundary validation.

Implementation approach:

- Keep resource tiers, Volumes, GPU types, timeouts, and `max_containers`
  unchanged.
- Add a static trusted harness contract that marks local scaffold mode as
  smoke-only and records deploy-once/call-many lookup/spawn/poll behavior.
- Add worker role contracts for trial, sampler, scorer, debug, and final
  validation.
- Validate that execution workers do not receive harness secrets, cannot write
  ledgers, and only scorer workers can read locked labels.
- Add tests that local static contracts cannot be reported as event-search
  ready.
- Reconcile stale `program.md`, benchmark-contract, and runbook wording so
  event ledger/search authority belongs to the Modal-hosted trusted
  orchestrator, while local ledger writes remain smoke-only scaffold behavior.

## Feature 2: `feat/falsification-and-ledgers`

Read-only agent findings:

- Current schemas already cover registered predictions, orchestrator-authored
  `FalsificationPlan`, `FalsificationResult`, verdict enums, confirmed-only
  Discovery Ledger records, lifecycle validation, and patch-policy locks.
- Gate waves are bounded and fakeable, use `starmap(..., order_outputs=True,
  return_exceptions=True, wrap_returned_exceptions=False)`, and normalize
  ordinary Modal exceptions into `INFRA_FAIL`.
- Remaining hardening gaps are explicit aggregate timeout enforcement,
  cancellation-style `BaseException` normalization, and lifecycle-ledger
  patch-policy lock coverage.

Implementation approach:

- Keep gate controls orchestrator-authored and avoid live Modal calls.
- Add an aggregate gate-wave timeout cap derived from per-control timeouts.
- Normalize raised cancellation/base exceptions into structured gate evidence
  rather than letting the wave escape without a ledger-visible reason.
- Require canonical ledger and Discovery Ledger append helpers to reject
  non-orchestrator writer roles.
- Lock lifecycle ledger helpers against future search patches.
- Add focused local tests for oversized aggregate timeout, cancellation
  normalization, complete scored gate evidence, non-orchestrator writes, and
  lifecycle-ledger lock coverage.

## Feature 3: `feat/baseline-scorer-readiness`

Read-only agent findings:

- Baseline readiness, current-best lookup, scorer-only local artifact scoring,
  Modal asset audit, no-template checks, and readiness report bridge are already
  substantially implemented.
- Highest-value gaps that can be handled offline are baseline lock identity and
  artifact path validation, stronger public-data Volume locked-label detection,
  and search patch-policy locks for readiness infrastructure once implemented.
- Larger live/data checks, such as recomputing real Arrow fingerprints or
  validating live Modal Volume bytes, remain human-approved live readiness
  actions.

Implementation approach:

- Do not create baseline metrics, Arrow files, labels, fingerprints, or Modal
  runs.
- Reject baseline locks missing `trial_id` or `candidate_id`.
- Reject baseline artifact pointers outside `runs/baseline/**`.
- Fail Modal asset audit if obvious locked-label filenames appear in the public
  data Volume, not only a top-level `locked/` prefix.
- Lock baseline/scorer/asset readiness modules against future search patches.

## Feature 4: `feat/prerun-readiness-report`

Read-only agent findings:

- The readiness report already covers baseline lock, local NanoFold gates,
  gate calibration, live-smoke approval, CLI JSON serialization, and no
  side-effect tests.
- The main gap was exposing the canonical certification vocabulary directly in
  the JSON report, especially `PASS_MOCKED_MODAL`.
- Goal docs and checklist needed final feature updates before the last PR.

Implementation approach:

- Preserve existing section `status` values for compatibility while adding
  `certification_status` to every section.
- Add a mocked Modal contract section sourced from the offline harness/worker
  role validators.
- Report certification counts for `PASS_LOCAL`, `PASS_MOCKED_MODAL`,
  `PASS_LIVE`, `PENDING_HUMAN_LIVE_ACTION`, `BLOCKED`, and `NOT_REQUESTED`.
- Treat missing gate calibration as an exact pending human live action, not as
  implementation-complete search readiness.
- Keep live Modal asset audit opt-in and read-only.
