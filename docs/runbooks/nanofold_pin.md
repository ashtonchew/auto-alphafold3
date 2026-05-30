# NanoFold Pin Runbook

Use this runbook to refresh or verify the pinned NanoFold code without downloading large folding databases.

## Verify Current Pin

```bash
cat NANOFOLD_COMMIT
git -C external/nanofold rev-parse HEAD
python -m pytest tests/test_nanofold_adapter.py -p no:cacheprovider
```

Both commit values must match.

## Update Pin Intentionally

Only update the pin outside the locked search loop and with a clear reason.

```bash
git -C external/nanofold fetch origin
git -C external/nanofold checkout <new_commit>
git -C external/nanofold rev-parse HEAD > NANOFOLD_COMMIT
python -m pytest tests/test_nanofold_adapter.py -p no:cacheprovider
```

Then review `docs/nanofold_map.md` and `autoalphafold3/editable_surface.md` for path drift.

## Do Not Run Here

Do not run these during the locked hackathon search loop or without explicit approval:

- small BFD download
- Uniclust30 download
- PDB70 download or any template database provisioning
- full mmCIF/MSA feature rebuilding
- real Arrow feature generation
- real baseline training

## Local Preflight Fixture

The local `tiny_forward` and `finite_loss` readiness gates may use a
deterministic local-only Arrow fixture. This fixture is not benchmark data, not
a Modal run, not a baseline source, and not search evidence. It exists only so
the NanoFold parser/model path can be exercised without downloads or full
feature rebuilding.

Materialize it only with the exact approval token:

```bash
python3 -m autoalphafold3.agent materialize-local-fixture \
  --approve I_APPROVE_LOCAL_NANOFOLD_FIXTURE
```

The command writes under `data/toy/nanofold_fixture/`, records
`local_only=true`, `official_benchmark_result=false`, and `max_templates=0` in
fixture provenance, and does not write `runs/`, baseline artifacts, ledgers, or
Discovery Ledger records.
