# auto-AlphaFold3

May 30, 2026 hackathon-start scaffold for a NanoFold-style AlphaFold3-lite autoresearch system.

## Local Environment

Use a repo-local virtual environment so the same Python can run tests, import
Modal, and smoke-check the vendored NanoFold modules:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then verify the local foundation:

```bash
python -m pytest -p no:cacheprovider
python - <<'PY'
from autoalphafold3.modal_app import healthcheck
from autoalphafold3.nanofold_adapter import import_smoke_summary
print(healthcheck()["modal_sdk_available"])
print(import_smoke_summary()["imports"])
PY
```

Modal asset verification uses the live Modal service and requires authenticated
network access:

```bash
python -m autoalphafold3.agent audit-modal-assets --search-ready
```
