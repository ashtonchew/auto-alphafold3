# Experiment Notes

## 2026-06-01

- PR #50 was confirmed merged into `origin/main`.
- Work started from sibling worktree
  `/Users/ashtonchew/projects/auto-alphafold3-worktrees/autoresearch-contract-docs`.
- PR 1 is documentation-only. It does not run training, call Modal, score
  candidates, write ledgers, write Discovery Ledger records, or create
  benchmark evidence.
- Validation for PR 1:
  - `git diff --check` passed.
  - `python3 .claude/skill-evals/run_offline_evals.py` passed all 148 checks.
  - Source tests were not run because PR 1 changes only documentation and goal
    progress files.

## Pending Human Live Actions

- `PENDING_HUMAN_LIVE_ACTION`: any future Modal short-training command must
  name the exact command and approval token after deterministic/local fixture
  validation exists.
- `PENDING_HUMAN_LIVE_ACTION`: any future open-ended LLM search command must
  name the exact command and approval token after deterministic ladder
  validation exists.

## 2026-06-01 PR 2

- Added a bounded short-training runner with `dry-run`, `local-fixture`, and
  approval-gated `modal` modes.
- Local fixture training is real NanoFold training over the synthetic fixture,
  but remains non-official smoke evidence.
- Targeted validation passed:
  `python3 -m pytest -p no:cacheprovider tests/test_short_training.py tests/test_checkpoint_training.py tests/test_runner_and_locked_scorer.py -q`
  reported 41 passed.
- Full local validation passed:
  `python3 -m pytest -p no:cacheprovider` reported 300 passed, 1 skipped.
