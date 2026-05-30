# AutoAlphaFold3 Pre-Run Readiness Experiment Notes

No benchmark, gate, discovery, baseline, Arrow, Modal-run, or validation-label artifacts have been created by this goal.

Synthetic unit tests may use in-memory scalar evidence and temporary directories only. They are not benchmark results and must not be reported as discoveries.

## 2026-05-30: `feat/falsification-contracts`

- Added only synthetic local contract tests.
- Ran `python3 -m pytest tests/test_falsification.py tests/test_local_contracts.py tests/test_modal_and_demo.py -q`: 32 passed.
- Pytest emitted a cache warning because the sibling worktree cannot create `.pytest_cache` under the current sandbox; this did not affect test results.
- Ran `python3 -m pytest -p no:cacheprovider`: 81 passed, 2 skipped.
- Ran `python3 .claude/skill-evals/run_offline_evals.py`: all 148 checks passed.
