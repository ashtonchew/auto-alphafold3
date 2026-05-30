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
