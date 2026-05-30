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

Pending until Feature 1 is merged.

## Feature 3: `feat/baseline-scorer-readiness`

Pending until Feature 2 is merged.

## Feature 4: `feat/prerun-readiness-report`

Pending until Feature 3 is merged.
