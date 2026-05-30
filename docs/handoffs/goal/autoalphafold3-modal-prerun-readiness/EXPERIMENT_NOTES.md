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
