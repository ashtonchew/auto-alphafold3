# Implementation Checklist

- [x] Latest `main` fetched.
- [x] PR #21 merge verified.
- [x] PR #22 merge verified.
- [x] Accepted merged base recorded.
- [x] Branch ownership and clean worktree state verified before branching.
- [x] Rebase skipped; no unowned branch rewritten.
- [x] Required repo docs and runbooks read.
- [x] Repo-local `modal-docs` skill read before Modal work.
- [x] PR #21 Modal-hosted orchestrator contract understood and cited in plan.
- [x] Local orchestration marked scaffold smoke-only, not event search ready.
- [x] Modal-hosted trusted orchestrator contract represented in code.
- [x] CPU-only harness contract represented in code.
- [x] Harness plane and execution plane represented in code and tests.
- [x] Secret-boundary validation implemented.
- [x] OpenAI, GitHub, Modal, dashboard, and judge/evaluator secrets excluded
  from trial/sampler/scorer/debug workers.
- [x] Worker role contracts implemented for trial, sampler, scorer, debug, and
  final validation.
- [x] Mocked/offline Modal contract tests extended for first feature.
- [x] Direct agent `modal run` and arbitrary Sandbox access remain forbidden by
  static contract.
- [ ] Structured pre-registration schema implemented.
- [ ] Falsification plan/result schemas implemented.
- [ ] Gate verdict outcomes covered by pure local tests.
- [ ] Gate controls are orchestrator-authored.
- [ ] Gate and Discovery Ledger files locked by patch policy.
- [ ] Baseline lock reader implemented.
- [ ] Missing or incomplete baseline lock fails readiness honestly.
- [ ] Scorer-only real artifact scoring contract implemented.
- [ ] Asset audit validates required files and separation.
- [ ] Official runs enforce `max_templates=0`.
- [ ] Readiness CLI/report implemented.
- [ ] Readiness report distinguishes local, mocked Modal, approved live,
  pending-human-live, and blocked evidence.
- [ ] Known-null and known-positive gate calibration complete or exact
  `PENDING_HUMAN_LIVE_ACTION` recorded.
- [ ] Four feature PRs merged.
- [ ] `python3 -m pytest -p no:cacheprovider` passes.
- [ ] `python3 .claude/skill-evals/run_offline_evals.py` passes.
- [ ] `git diff --check` passes.
- [x] No fake benchmark/data/Modal/gate/discovery artifacts created.
- [x] Autonomous research search not started.
