# auto-AlphaFold3

auto-AlphaFold3 is a hackathon foundation for a NanoFold-style AlphaFold3-lite autoresearch system.

The short version: an agent proposes one protein-folding change, registers what it expects to happen, runs a fixed-budget trial through a locked control plane, and keeps the result only when the score and the diagnostic story survive follow-up controls.

This project is intentionally smaller and stricter than its name sounds. Its target is a small monomer folding sandbox built around [`ogchen/nanofold`](https://github.com/ogchen/nanofold), with a locked C-alpha lDDT benchmark and Modal-backed trial infrastructure. Claims about training, reproducing, improving, or competing with Google DeepMind AlphaFold3 are outside scope.

## Highlights

- NanoFold-style AlphaFold3-lite research surface, scoped to monomer protein folding.
- Locked primary metric: `best_val_calpha_lddt`, computed by `calpha_lddt_v1`.
- Fold Cartographer diagnostics for local geometry, long-range topology, distogram-to-3D mismatch, and stability or compute failures.
- Falsification Gate for provisional wins: knock-out, placebo, predicted-axis check, and seed rerun.
- Typed `AutoFoldTrial` contracts, preflight checks, readiness reports, and append-only trial lifecycle records.
- Modal control-plane contract with separate public data and locked scorer Volumes.
- Local toy fixtures for contract tests, with no fabricated benchmark results.

## What This Repo Is

- local control-plane code in `autoalphafold3/`
- locked scorer code in `autoalphafold3/scorer/`
- a pinned NanoFold working tree in `external/nanofold/`
- experiment and smoke configs in `configs/`
- manifest templates and toy fixtures in `data/`
- runbooks and the canonical project spec in `docs/`
- typed trial JSON support in `trials/`
- tests for contracts, scorer behavior, Modal asset policy, and runner logic

The project is built around a narrow research claim: an agent can search a small folding model under a locked benchmark, and real progress should look like a documented mechanism instead of a lucky metric bump.

## How The Loop Works

Each trial follows a deliberate path:

1. Read the current best result, latest diagnostics, Discovery Ledger, and recent trial history.
2. Pick one diagnostic target and one move family.
3. Register a falsifiable prediction before editing, patch only the approved model, training, sampler, or config surface, and submit one `AutoFoldTrial` JSON object.
4. Run a fixed-budget trial through the Modal-hosted trusted orchestrator during event search.
5. Score with the locked C-alpha lDDT scorer, then route the result through Fold Cartographer.
6. Run Falsification Gate controls for any provisional `KEEP`.
7. Commit confirmed mechanisms, discard valid misses, and log failures honestly.

The primary optimization target is always `best_val_calpha_lddt`. Diagnostics guide hypotheses while the primary metric remains the decision point.

## LLM Phase Policy

The future Modal-hosted OpenAI Agents SDK harness must use the repo policy in `autoalphafold3/llm_policy.py` instead of relying on SDK defaults. The default model target is `gpt-5.4-mini`, with Priority processing for both search phases. Use `--model gpt-5.5` only as an explicit override/config choice:

- `hypothesis_generation`: web search enabled, reasoning effort `low`, text verbosity `low`.
- `patch_planning`: web search disabled, reasoning effort `low`, text verbosity `low`.

Inspect the concrete kwargs/spec with:

```bash
python -m autoalphafold3.agent llm-policy --format responses
python -m autoalphafold3.agent llm-policy --format agents-sdk
```

## Current Status

This repository now has the foundation evidence needed to describe a real NanoFold-style AlphaFold3-lite control loop: locked assets, a locked baseline, a Modal-hosted trusted orchestrator proof, gate calibration evidence, a frozen one-batch checkpoint, and live sampler/scorer smokes. It is still not a confirmed-discovery repo; sampler candidates so far are recorded as `DISCARD` globally.

Present today:

- pinned NanoFold source and adapter checks
- C-alpha lDDT scorer tests
- Fold Cartographer diagnostic contracts
- Falsification Gate verdict logic
- typed trial, ledger, and Discovery Ledger schemas
- Modal app contract metadata, asset audits, and event-authority proof
- locked baseline metrics and error report under `runs/baseline/`
- real one-batch frozen checkpoint manifest for sampler-only trials
- live Modal sampler/scorer smoke artifacts and canonical sampler summaries
- local toy data for contract tests

Still required before any discovery claim:

- a candidate that clears the locked global baseline threshold as a provisional `KEEP`
- a Falsification Gate run for that provisional `KEEP`
- a `CONFIRMED` mechanism written by the trusted orchestrator into the Discovery Ledger
- broader model/training search beyond the current frozen-checkpoint sampler-only evidence, if the project wants to claim more than sampler-family progress

Treat local stub artifacts as directory-shape and contract evidence only. Use only committed real Modal/scorer artifacts for result claims.

## Results

The locked global baseline is `T000` with `best_val_calpha_lddt=0.07941230438543605` on `public_val_small`, scored by `calpha_lddt_v1` across 16 of 16 targets with `max_templates=0` and `official_benchmark_result=true`. The Fold Cartographer signature is `toy_geometry_failed`, with canonical target `local_geometry_weak`.

The first live sampler/scorer path succeeded after the frozen checkpoint was created at `T010`. The same-family default sampler reference `T081` scored `best_val_calpha_lddt=0.008276756926787072` with 16 of 16 targets scored and no failed targets.

Sampler-family search found real improvements over that same-family reference, but not over the locked global baseline. The strongest committed sampler smoke candidate is `T088`, with `best_val_calpha_lddt=0.02098351201866366`, a `+0.012706755091876588` delta over `T081` and about `2.54x` the reference score. The canonical 12-candidate GPT-5.4 mini sampler run (`T092` through `T103`) recorded all 12 candidates as `SAMPLER_IMPROVED` versus `T081`; its best candidate was `T096` with `best_val_calpha_lddt=0.018519489370625704`, `+0.010242732443838632` over `T081`, and `-0.06089281501481034` versus the locked baseline.

No sampler candidate cleared the global baseline, so every candidate remains global `DISCARD`. No provisional `KEEP`, Falsification Gate run on a winning candidate, confirmed mechanism, or Discovery Ledger entry has been produced.

Primary evidence files:

- `runs/baseline/metrics.json`
- `runs/canonical_sampler_smokes_2026-05-30.json`
- `runs/canonical_sampler_12_candidate_run_2026-05-30.json`
- `runs/falsification_gate_calibration.json`
- `runs/modal_event_authority.json`

## Quick Start

Create a repo-local Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the local test suite:

```bash
python -m pytest -p no:cacheprovider
```

Run the scorer tests only:

```bash
python -m pytest tests/test_calpha_lddt.py
```

Verify the NanoFold pin and adapter map:

```bash
python -m pytest tests/test_nanofold_adapter.py -p no:cacheprovider
```

Check the local Modal contract metadata:

```bash
python - <<'PY'
from autoalphafold3.modal_app import healthcheck, modal_deploy_plan

print(healthcheck())
print(modal_deploy_plan())
PY
```

Build a read-only readiness report:

```bash
python -m autoalphafold3.agent readiness-report
```

## Common Commands

Validate the toy manifest:

```bash
python -m autoalphafold3.agent validate-manifest data/manifests/smoke.json
```

Validate empty official manifest templates:

```bash
python -m autoalphafold3.agent validate-manifest \
  data/manifests/train_tiny.template.json \
  data/manifests/public_val_small.template.json \
  --allow-empty --no-verify-assets
```

Audit Modal assets:

```bash
python -m autoalphafold3.agent audit-modal-assets
```

Audit Modal assets for search readiness:

```bash
python -m autoalphafold3.agent audit-modal-assets --search-ready
```

Record deployed Modal event authority proof:

```bash
python -m autoalphafold3.agent audit-modal-authority \
  --mode modal \
  --approve I_APPROVE_MODAL_EVENT_AUTHORITY
```

Produce and freeze Falsification Gate calibration evidence:

```bash
python -m autoalphafold3.agent run-gate-calibration \
  --mode modal \
  --approve I_APPROVE_GATE_CALIBRATION_RUN
python -m autoalphafold3.agent calibrate-gate \
  --mode from-evidence \
  --known-null-evidence runs/gate_calibration/known_null.json \
  --known-positive-evidence runs/gate_calibration/known_positive.json \
  --approve I_APPROVE_GATE_CALIBRATION
```

Produce a minimal real frozen checkpoint for sampler-only trials:

```bash
python -m autoalphafold3.agent run-one-batch-checkpoint --mode dry-run
python -m autoalphafold3.agent run-one-batch-checkpoint \
  --mode modal \
  --approve I_APPROVE_ONE_BATCH_CHECKPOINT
```

This performs exactly one NanoFold training batch with `diffusion_steps=1` and
`max_templates=0`, writes a real `checkpoint.pt` on the Modal data Volume, and
records `checkpoint_manifest.json`. It is a frozen checkpoint seed, not a
benchmark or quality claim.

Submit a future approved trial:

```bash
python -m autoalphafold3.agent submit trials/T###.json
```

During event search, submission authority belongs to the Modal-hosted trusted orchestrator. Local dry-run behavior is for scaffold checks before deployment.

## Benchmark Contract

The benchmark is locked by design.

Allowed during search:

- `autoalphafold3/patches/**`, when added
- `configs/experiments/**`
- selected NanoFold model and training files listed in `autoalphafold3/editable_surface.md`
- sampler wrappers that avoid validation-label reads
- loss weights, geometry losses, optimizer and scheduler config
- curriculum, diffusion/noising, recycling, Pairformer/attention, feature-handling, memory, and runtime experiments inside the approved surface

Forbidden during search:

- scorer code and metric math
- validation split membership
- validation labels and locked scorer assets
- cached feature outputs and fingerprints
- baseline ledger contents
- result parser behavior
- `autoalphafold3/modal_app.py`
- Modal GPU types, timeouts, Volumes, `max_containers`, and cost caps
- upstream NanoFold download scripts and full feature rebuilding

The official benchmark pins NanoFold template use to `max_templates=0`. Template tensors are schema placeholders here, outside the template-search path.

## Modal Layout

The intended Modal pattern is deploy once, call many. The agent submits typed trials, while Modal owns the controlled execution path.

Storage policy:

- `autoalphafold3-data`: public data, cached features, run artifacts, checkpoints, logs, and renders.
- `autoalphafold3-locked`: locked validation manifests, labels, scorer code, and scorer metadata.

Worker policy:

- trial, sampler, and debug workers run without locked validation-label mounts
- scorer-only workers may read locked labels
- workers write only under their own trial directories
- the trusted orchestrator writes canonical ledgers during event search
- direct agent `modal run`, arbitrary sandbox spawning, and infrastructure edits are outside the search contract

## Repository Map

```text
autoalphafold3/          control plane, contracts, scorer wrappers, orchestration
autoalphafold3/scorer/   locked scoring code
external/nanofold/       pinned NanoFold working-tree copy
configs/                baseline, smoke, and future experiment configs
data/                   manifest templates and toy fixtures
docs/                   framing, runbooks, data notes, and canonical spec
trials/                 typed trial JSON files
runs/                   generated artifacts and ledger output
tests/                  local contract and scorer tests
```

The canonical design lives in [`docs/spec/autoalphafold3-canonical (2).html`](<docs/spec/autoalphafold3-canonical%20(2).html>). If docs disagree, treat that file as the source of truth and fix the drift.

## Read Next

Before changing behavior, read [`docs/framing.md`](docs/framing.md) for scope, [`program.md`](program.md) for the operating contract, [`autoalphafold3/benchmark_contract.md`](autoalphafold3/benchmark_contract.md) for scoring rules, [`autoalphafold3/editable_surface.md`](autoalphafold3/editable_surface.md) for approved search files, and [`docs/runbooks/`](docs/runbooks/) for lock, manifest, Modal, and NanoFold procedures.

Useful runbooks:

- [`docs/runbooks/nanofold_pin.md`](docs/runbooks/nanofold_pin.md)
- [`docs/runbooks/manifest_locking.md`](docs/runbooks/manifest_locking.md)
- [`docs/runbooks/baseline_lock.md`](docs/runbooks/baseline_lock.md)
- [`docs/runbooks/modal_control_plane.md`](docs/runbooks/modal_control_plane.md)

## Development Notes

The repo currently has no project-level formatter, linter, type checker, or packaging command. Keep changes small, run the relevant tests, and update this README if a new standard command becomes part of the workflow.

For fast local iteration, keep pytest's default cache plugin enabled and use:

```bash
python -m pytest --lf
```

Live Modal checks require authenticated network access and real Modal assets. They are opt-in:

```bash
AUTOALPHAFOLD3_RUN_LIVE_MODAL_TESTS=1 python -m pytest -m live_modal
```

## Ground Rules

- Say NanoFold-style AlphaFold3-lite when describing the implementation target.
- Keep real AlphaFold3 training or improvement claims out of this project.
- Use only real benchmark metrics, Modal runs, Arrow files, and baseline evidence.
- Leave full MSA, mmCIF, and template database rebuilding outside the event loop.
- Keep failures visible. A clean `FAIL`, `INFRA_FAIL`, or Falsification Gate kill is useful evidence.

## License

See the repository files for project licensing. The vendored NanoFold source keeps its own upstream license under `external/nanofold/LICENSE`.
