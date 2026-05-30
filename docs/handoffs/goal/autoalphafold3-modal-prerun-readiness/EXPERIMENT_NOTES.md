# Experiment Notes

## Feature 1

- All tests must remain offline and must not call live Modal, mutate Volumes, or
  write baseline/gate/discovery artifacts.
- Local scaffold readiness is intentionally `event_search_ready_locally=false`.
- Remaining event readiness requires a human-approved deployment/authentication
  action for the Modal-hosted trusted orchestrator.
- Slice 1 reconciles stale local-orchestrator authority language in
  `program.md`, `autoalphafold3/benchmark_contract.md`, and
  `docs/runbooks/modal_control_plane.md` with the PR #21 canonical contract.

## Feature 2

- Existing foundation code already covered most Falsification Gate and
  confirmed-only Discovery Ledger requirements.
- This slice hardens the mocked Modal gate-wave adapter: aggregate timeout is
  rejected before submission when unsafe, and cancellation-style platform
  failures become `INFRA_FAIL` evidence.
- Canonical ledger and Discovery Ledger append helpers now require
  `writer_role="orchestrator"`.
- Gate-wave evidence must be complete and scored before verdict math can use it.
- `autoalphafold3/ledger.py` is now locked by patch policy alongside
  falsification and Discovery Ledger surfaces.
- Tests remain offline and fakeable; no Modal jobs, benchmark artifacts, gate
  verdict artifacts, or Discovery Ledger records were created.

## Feature 3

- Existing baseline/scorer/asset readiness code already covered most required
  offline checks.
- This slice adds baseline label-hash, identity, and artifact-path validation
  without creating baseline artifacts.
- Public data Volume checks now reject obvious locked-label filenames as well
  as locked prefixes.
- Baseline readiness, locked scorer, and Modal asset audit modules are locked
  by patch policy for future search.

## Feature 4

- Readiness report JSON now includes canonical `certification_status` values
  and aggregate certification counts.
- Mocked Modal contract readiness is reported separately from local checks and
  live-smoke checks.
- Modal event-authority readiness is reported separately so local/mocked checks
  cannot make the scaffold event-search ready without a human-approved
  Modal-hosted trusted orchestrator action.
- Missing gate calibration is represented as an exact
  `PENDING_HUMAN_LIVE_ACTION`; it blocks autonomous search.
- The repo-local readiness report currently has zero `BLOCKED` sections:
  `PASS_MOCKED_MODAL=1`, `PENDING_HUMAN_LIVE_ACTION=4`, and
  `NOT_REQUESTED=1`.
- Mixed local gate skips are classified by exact cause; dependency-only skips
  keep the dependency action, cached Arrow fixture skips get a separate exact
  local-gate action, and unknown skips remain `BLOCKED`.
- No live Modal calls, baseline artifacts, gate verdict artifacts, canonical
  ledger rows, or Discovery Ledger records were created.
