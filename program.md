# auto-AlphaFold3: AlphaFold3-lite Autoresearch Contract

## Core Hypothesis

GPT-5.4 mini can act as an autoresearcher for protein folding, discovering AlphaFold3-lite design improvements in a locked NanoFold-style research sandbox. GPT-5.5 remains an explicit higher-tier override, not the default loop model.

We are not modifying, reproducing, training, improving, or beating Google DeepMind's real AlphaFold3. We are using `ogchen/nanofold` as a small NanoFold-style AlphaFold3-lite research sandbox.

## Locked Objective

Maximize `best_val_calpha_lddt` under the locked benchmark.

C-alpha lDDT measures how well predicted alpha-carbon backbone geometry preserves local target distances. It is superposition-free and is the only primary metric for trial decisions.

Diagnostics may route hypotheses, but they are not additional optimization targets:

- `local_geometry_weak`: local lDDT and short-range contact precision
- `long_range_topology_weak`: long-range contact precision at sequence separation >= 24
- `distogram_good_lddt_flat`: contacts/distogram improve but realized 3D lDDT does not
- `stability_compute`: runtime, peak GPU memory, parameter count, NaN, OOM, timeout, and infra status
- validation loss, coordinate loss, and distogram loss as supporting context
- final-validation seed variance

## Read Before Changes

1. `README.md`
2. `docs/framing.md`
3. `configs/auto_tiny.json`
4. `autoalphafold3/editable_surface.md`
5. `autoalphafold3/benchmark_contract.md`
6. `runs/baseline/metrics.json` when it exists
7. `runs/baseline/error_report.json` when it exists
8. the tail of `runs/ledger.jsonl` when it exists
9. selected NanoFold files listed in `editable_surface.md`

## Allowed Surface

During the search loop, the agent may edit only the approved model/training/sampler search surface:

- `autoalphafold3/patches/**` when added
- selected NanoFold model and training modules named in `editable_surface.md`
- `configs/experiments/**`
- loss weights and geometry losses
- optimizer and scheduler config
- training curriculum
- diffusion/noising and inference sampler settings
- recycling, Pairformer/attention, auxiliary heads, feature-handling, and memory/runtime optimizations

## Forbidden Surface

Do not modify:

- public validation split definitions
- scoring code or metric computation
- raw validation labels
- cached feature outputs during the event
- benchmark fingerprints
- baseline result ledger
- result parser
- files in the scorer-only locked Volume
- scripts that rebuild public validation features
- Modal app, resource tiers, GPU types, `max_containers`, timeout caps, or Volumes

If the scorer, split, or locked data looks wrong during search, write a note and stop that trial. Do not patch locked benchmark assets.

## Modal Contract

The Modal compute architecture is creator-designed infrastructure. During the
event, the Modal-hosted trusted orchestrator is the only search-loop authority;
local orchestration remains a scaffold smoke-test fallback before deployment.

The agent submits `AutoFoldTrial` JSON objects through the Modal-hosted
trusted orchestrator. It may not call `modal run` directly, spawn arbitrary
Modal Sandboxes, modify Volumes, change GPU type, raise `max_containers`,
change timeouts, edit `autoalphafold3/modal_app.py`, or author falsification
controls during search.

Approved experiment command:

```bash
python -m autoalphafold3.agent submit trials/T###.json
```

Workers write only to their own trial directories. During event search, the
Modal-hosted trusted orchestrator is the only writer for the canonical ledger
and Discovery Ledger; local ledger writes are scaffold smoke-test behavior only.

## Agent-Facing Skills And Evals

Use the project skills as narrow operating modes:

- `autoalphafold3-researcher`: choose one diagnostic target, one move family, and one falsifiable folding hypothesis.
- `fold-cartographer`: map scorer diagnostics to exactly one canonical target while keeping `best_val_calpha_lddt` as the only primary objective.
- `autoalphafold3-trial-submit`: validate one `AutoFoldTrial` JSON and return only the configured Modal-hosted orchestrator endpoint for event search. The local command is smoke-only before deployment.
- `autoalphafold3-subagent-worker`: generate bounded proposal artifacts for parallel hypothesis fanout; workers do not submit, integrate, call Modal, edit locked files, or write ledgers.

Skill evals must pass before autonomous research starts. Evals are synthetic and local: no Modal calls, no GPUs, no hidden validation, no gate edits, and no real trial submission. Run static/offline checks first, fan out sampled live evals on small fast models, and reserve stronger adjudication for failed or ambiguous outputs.

## LLM Phase Policy

The Modal-hosted harness must load `autoalphafold3.llm_policy.default_llm_phase_policies()` rather than relying on OpenAI Agents SDK defaults.

- Hypothesis generation defaults to `gpt-5.4-mini`, uses Priority processing, web search enabled, `reasoning.effort="low"`, and low verbosity.
- Patch planning defaults to `gpt-5.4-mini`, uses Priority processing, web search disabled, `reasoning.effort="low"`, and low verbosity.
- GPT-5.5 is available only behind an explicit model config/CLI override.

Hypothesis generation may use the web to broaden candidate ideas and cite current outside context. Patch planning must stay repo-local so implementation choices are grounded in the allowed edit surface, benchmark contract, diagnostics, and current code.

## Trial Lifecycle

```text
DRAFT -> PREFLIGHT_PASSED -> SUBMITTED -> RUNNING -> SCORED -> KEEP | DISCARD | FAIL | INFRA_FAIL -> ARCHIVED
```

For every trial:

1. Read current best result, latest Fold Cartographer report, and recent ledger entries.
2. Choose exactly one diagnostic target.
3. Choose exactly one move family.
4. Write a hypothesis before editing.
5. Predict the expected score and diagnostic effect.
6. Patch only the allowed surface.
7. Create one `AutoFoldTrial` JSON.
8. Submit through the Modal-hosted trusted orchestrator during event search, or the local smoke scaffold before deployment.
9. Read canonical metrics and diagnostics.
10. Decide `KEEP`, `DISCARD`, `FAIL`, or `INFRA_FAIL`.
11. Commit only valid `KEEP` runs.
12. Revert valid `DISCARD` runs.
13. Log `FAIL` and `INFRA_FAIL` honestly.
14. Write a postmortem.

## Failure Statuses

- `KEEP`: valid run, improves best validation C-alpha lDDT by the configured threshold, and does not violate evaluation rules.
- `DISCARD`: valid run, no meaningful improvement.
- `FAIL`: candidate patch, model, training, config, or scorer-output failure.
- `INFRA_FAIL`: Modal, Volume, image, dependency, polling, quota, or storage failure before trial logic completes.

Failures are part of the research trajectory. Do not hide them and do not convert infrastructure failures into benchmark changes.

## Data Rule

Do not run full PDB/mmCIF/MSA feature rebuilding during the event. The hackathon search loop starts from cached Arrow IPC features, fixed manifests, locked labels, and a baseline score.

Template policy:

- Pin `max_templates=0` for every official baseline, trial, sampler run, and final validation.
- Treat NanoFold template tensors as empty schema placeholders only.
- Do not provision, search, or mutate a template database.
- Do not use template-sensitive diagnostics; this benchmark is intentionally no-template.

Volume policy:

- Public data, reduced DBs, raw mmCIFs, Arrow features, run artifacts, checkpoints, logs, and renders live in the Modal Volume `autoalphafold3-data`.
- Locked validation manifests, labels, scorer code, and scorer metadata live in the scorer-only Modal Volume `autoalphafold3-locked`.
- Trial, sampler, and debug jobs may not mount locked labels. Scoring happens through scorer-only functions.
