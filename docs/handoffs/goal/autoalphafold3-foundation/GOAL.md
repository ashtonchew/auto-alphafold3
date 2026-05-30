# NanoFold-style AlphaFold3-lite Foundation Goal

Implement the infrastructure foundation required before autonomous search can
begin. This goal is limited to contracts, adapters, trial runtime, Modal
control-plane wiring, orchestration, and readiness gates for the
NanoFold-style AlphaFold3-lite system.

This work must not start autonomous research trials, fabricate benchmark/data
artifacts, mutate locked benchmark assets, or claim to train, reproduce,
improve, or beat Google DeepMind AlphaFold3.

## Done Criteria

- Required repo docs and project skills have been read.
- PR #6 is verified merged into current `main`.
- Feature-sized stacked PRs are open with semantic `feat/...` branch names.
- `IMPLEMENTATION_CHECKLIST.md` is 100% complete.
- `python3 -m pytest -p no:cacheprovider` passes.
- `python3 .claude/skill-evals/run_offline_evals.py` passes.
- No fake benchmark results, fake Arrow files, fake Modal runs, fake baseline
  metrics, fake validation labels, or autonomous search trials were created.
