# Experiments

## Feature 1: `feat/modal-prerun-contracts`

Commands run:

- `python3 -m pytest tests/test_modal_prerun_contracts.py tests/test_nanofold_adapter.py tests/test_modal_and_demo.py tests/test_gate_wave.py -p no:cacheprovider`
- `python3 -m pytest -p no:cacheprovider`
- `git diff --check`

Results:

- Targeted Modal contract suite: 41 passed, 1 skipped.
- Full local suite: 177 passed, 3 skipped.
- `git diff --check`: passed.

## Feature 2: `feat/falsification-and-ledgers`

Commands run:

- `python3 -m pytest tests/test_falsification.py tests/test_gate_wave.py tests/test_discovery_ledger.py tests/test_two_stage_orchestrator.py tests/test_local_contracts.py -p no:cacheprovider`
- `python3 -m pytest -p no:cacheprovider`
- `git diff --check`

Results:

- Targeted falsification/ledger suite: 85 passed.
- Full local suite: 183 passed, 3 skipped.
- `git diff --check`: passed.
