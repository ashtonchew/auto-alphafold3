# Experiments

## Feature 1: `feat/modal-prerun-contracts`

Commands run:

- `python3 -m pytest tests/test_modal_prerun_contracts.py tests/test_nanofold_adapter.py -p no:cacheprovider`
- `python3 -m pytest -p no:cacheprovider`
- `git diff --check`

Results:

- Targeted Modal contract suite: 19 passed, 1 skipped.
- Full local suite: 177 passed, 3 skipped.
- `git diff --check`: passed.
