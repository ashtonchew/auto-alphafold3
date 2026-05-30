# AutoAlphaFold3 Pre-Run Readiness Experiment Notes

No benchmark, gate, discovery, baseline, Arrow, Modal-run, or validation-label artifacts have been created by this goal.

Synthetic unit tests may use in-memory scalar evidence and temporary directories only. They are not benchmark results and must not be reported as discoveries.

## 2026-05-30: `feat/falsification-contracts`

- Added only synthetic local contract tests.
- Ran `python3 -m pytest tests/test_falsification.py tests/test_local_contracts.py tests/test_modal_and_demo.py -q`: 32 passed.
- Pytest emitted a cache warning because the sibling worktree cannot create `.pytest_cache` under the current sandbox; this did not affect test results.
- Ran `python3 -m pytest -p no:cacheprovider`: 81 passed, 2 skipped.
- Ran `python3 .claude/skill-evals/run_offline_evals.py`: all 148 checks passed.

## 2026-05-30: `feat/baseline-readiness`

- Added only synthetic `tmp_path` baseline-readiness contract tests.
- Ran `python3 -m pytest tests/test_baseline_readiness.py -q -p no:cacheprovider`: 21 passed.
- Ran `python3 -m pytest -p no:cacheprovider`: 102 passed, 2 skipped.
- Ran `python3 .claude/skill-evals/run_offline_evals.py`: all 148 checks passed.
- No baseline metrics, benchmark artifacts, `runs/baseline/**` files, locked Volume writes, canonical ledger entries, Discovery Ledger entries, Modal runs, or autonomous search trials were created.

## 2026-05-30: `feat/discovery-ledger`

- Added only synthetic `tmp_path` Discovery Ledger contract tests.
- Ran `python3 -m pytest tests/test_discovery_ledger.py tests/test_local_contracts.py -q -p no:cacheprovider`: 36 passed.
- Ran `python3 -m pytest tests/test_discovery_ledger.py tests/test_local_contracts.py -q -p no:cacheprovider`: 39 passed after aligning the provisional KEEP fixture with the stricter PR #15 gate-status schema.
- Ran `python3 -m pytest -p no:cacheprovider`: 121 passed, 2 skipped.
- Ran `python3 .claude/skill-evals/run_offline_evals.py`: all 148 checks passed.
- No real Discovery Ledger records, canonical ledger entries, benchmark artifacts, baseline metrics, Modal runs, `runs/**` files, or autonomous search trials were created.

## 2026-05-30: `feat/two-stage-orchestrator`

- Added only synthetic `tmp_path` two-stage orchestration contract tests.
- Ran `python3 -m pytest tests/test_two_stage_orchestrator.py -q -p no:cacheprovider`: 11 passed.
- Ran `python3 -m pytest -p no:cacheprovider`: 132 passed, 2 skipped.
- Ran `python3 .claude/skill-evals/run_offline_evals.py`: all 148 checks passed.
- No gate verdicts, real Discovery Ledger records, canonical ledger entries outside `tmp_path`, benchmark artifacts, baseline metrics, Modal runs, repo `runs/**` files, or autonomous search trials were created.
