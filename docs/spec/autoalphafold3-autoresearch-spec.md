# auto-AlphaFold3 Autoresearch Specification

Status: planning
Scope: post-hackathon side-project extension
Target: NanoFold-style AlphaFold3-lite

## Purpose

This specification defines the next version of auto-AlphaFold3: a
SimplexFold/Karpathy-style autonomous research loop for a NanoFold-style
AlphaFold3-lite folding sandbox.

The goal is not to train, reproduce, improve, or beat Google DeepMind
AlphaFold3. The goal is to test whether an AI agent can make real, bounded
model/loss/training-code changes inside a locked NanoFold-style research
sandbox, run fixed-budget training and evaluation, keep improvements, revert
misses, and leave a complete evidence trail.

The side-project claim is:

> An agent can act as a constrained research engineer for a
> NanoFold-style AlphaFold3-lite model by editing the approved
> architecture/loss/training surface, running fixed-budget experiments, and
> retaining only changes that improve locked C-alpha lDDT under honest controls.

This document extends the canonical hackathon-start design in
`docs/spec/autoalphafold3-canonical (2).html`, the operating contract in
`program.md`, the benchmark boundary in `autoalphafold3/benchmark_contract.md`,
and the search surface in `autoalphafold3/editable_surface.md`.

## Why This Shape

Karpathy-style autoresearch gives the agent a small real training system, a
fixed time budget, one ground-truth metric, and a narrow editable surface. The
agent patches code, runs training, keeps or reverts the change, records the
result, and repeats.

SimplexFold applies that spirit to protein folding. Its useful pattern is not
the particular simplex architecture alone; it is the research protocol:

- one falsifiable hypothesis
- one bounded edit surface
- fixed-budget train/eval runs
- matched ablations
- keep/revert discipline
- results logged as a trajectory, not a one-off claim

This repo should adopt that protocol while preserving stricter benchmark and
infrastructure boundaries than a free-form research fork.

## Non-Goals

- Do not claim real AlphaFold3 reproduction, training, improvement, or parity.
- Do not add ligand, RNA, DNA, multimer, template, or full biomolecular
  cofolding scope.
- Do not rebuild full MSA, PDB, mmCIF, Arrow, or template databases as part of
  the loop.
- Do not let the agent alter scorer math, validation membership, validation
  labels, cached feature outputs, fingerprints, baseline evidence, Modal GPU
  types, timeouts, Volumes, `max_containers`, or cost caps.
- Do not treat sampler-family gains over a frozen one-batch checkpoint as
  global discovery evidence.

## Core Research Program

The first autoresearch program should be deliberately narrow:

> Improve short-training local C-alpha geometry for NanoFold-style
> AlphaFold3-lite coordinate diffusion under a locked low-compute benchmark.

The first hypothesis family is:

> Short low-compute AF3-style diffusion training is bottlenecked by weak
> differentiable local C-alpha geometry supervision. Adding or restructuring
> local-geometry signals inside the allowed NanoFold model/loss/training
> surface should improve `best_val_calpha_lddt` more than sampler-only search or
> matched default short training.

The first loop should target `local_geometry_weak`. Later programs may target
`long_range_topology_weak`, `distogram_good_lddt_flat`, or `stability_compute`
only after the short-training harness works end to end.

## Agent Authority

The agent is a constrained research engineer. It may edit research code inside
the approved search surface, but it may not edit benchmark authority,
evaluation authority, infrastructure authority, or locked data.

### Allowed Edit Surface

The initial autoresearch loop may edit:

- `configs/experiments/**`
- `autoalphafold3/patches/**`
- `external/nanofold/nanofold/train/model/nanofold.py`
- `external/nanofold/nanofold/train/model/nanofold_trunk.py`
- `external/nanofold/nanofold/train/model/pairformer.py`
- `external/nanofold/nanofold/train/model/diffusion_model.py`
- `external/nanofold/nanofold/train/model/diffusion_transformer.py`
- `external/nanofold/nanofold/train/model/msa_module.py`
- `external/nanofold/nanofold/train/model/template_embedder.py`
- `external/nanofold/nanofold/train/loss.py`
- `external/nanofold/nanofold/train/trainer.py`
- `external/nanofold/nanofold/train/chain_dataset.py`

This matches the search surface in `autoalphafold3/editable_surface.md`.

### Forbidden Surface

The agent must not edit:

- `autoalphafold3/scorer/**`
- `autoalphafold3/benchmark_contract.md`
- `autoalphafold3/modal_app.py`
- `autoalphafold3/orchestrator.py`
- `autoalphafold3/preflight.py`
- `autoalphafold3/patch_policy.py`
- public validation manifests
- validation labels
- data fingerprints
- cached feature outputs
- `runs/baseline/**`
- `runs/ledger.jsonl`
- `runs/discovery_ledger.jsonl`
- Modal GPU type, timeout, Volume, retry, warm-pool, and `max_containers`
  settings

Any suspected benchmark, scorer, split, locked-label, or Modal resource bug is
a stop-and-report condition, not an agent-editable task.

## What The Agent May Try

The agent may make real code changes such as:

- differentiable local C-alpha pair-distance losses
- differentiable approximations to local lDDT
- config-driven diffusion, distogram, and local-geometry loss weights
- training-time diffusion noise distribution changes
- loss-weight schedules and short curricula
- Pairformer dropout, transition, attention, or residual-path changes
- lightweight auxiliary heads that use training labels only during training
- recycling behavior changes
- optimizer, scheduler, clipping, and warmup changes
- sampler schedules after trained checkpoints
- memory/runtime optimizations inside the approved NanoFold surface

The first implementation should avoid large sparse face/tetra architecture
changes. Those are valid future work, but they are too large for the first loop
because the current missing piece is the fixed-budget real train/eval harness.

## Fixed-Budget Tiers

Experiments must be comparable by budget. The first supported tiers are:

| Tier | Training steps | Intended use |
| --- | ---: | --- |
| `smoke` | 10 | catches code bugs, tensor shape failures, and impossible recipes |
| `trial` | 250 | first meaningful low-compute comparison under current preflight cap |
| `dev` | 1000 | human-approved escalation only |
| `final` | seed reruns plus gate controls | reserved for provisional winners |

The existing `trial` preflight cap is 250 steps. Increasing that cap is an
infrastructure/product decision, not an agent search edit.

Training steps are optimizer steps. In current NanoFold, `diffusion_steps`
controls inference sampling only; training uses one denoising task per optimizer
step, scaled primarily by `diffusion_batch_size`.

## Initial Experiment Ladder

The first deterministic ladder should run before open-ended LLM search:

```text
T120 short_train_baseline_smoke      10 steps
T121 first_geometry_patch_smoke      10 steps
T122 short_train_baseline_trial     250 steps
T123 best_geometry_patch_trial      250 steps
T124 no_geometry_aux_ablation       250 steps
T125 sampler_after_best_checkpoint  inference-only
```

Promotion rule:

- A patch that cannot pass smoke is `FAIL` or `INFRA_FAIL`.
- A patch that passes smoke but loses to the matched baseline at 250 steps is
  `DISCARD`.
- A patch that beats the matched 250-step baseline may become the new
  short-training current best.
- A patch that beats the global current best by the configured KEEP threshold is
  a provisional `KEEP` and must pass the Falsification Gate before discovery.

## System Architecture

```text
Research program docs
  -> agent reads current evidence and proposes one candidate
  -> agent patches allowed model/loss/training files
  -> patch policy validates changed paths
  -> preflight validates trial JSON, config, budget, and scorer version
  -> Modal trial worker trains and writes trial-scoped artifacts
  -> sampler/inference worker writes prediction artifacts when needed
  -> scorer-only worker reads locked labels and returns metrics
  -> orchestrator records lifecycle and stage-one decision
  -> Falsification Gate runs only for provisional KEEP
  -> Discovery Ledger accepts only CONFIRMED mechanisms
  -> evidence UI renders the trajectory from real artifacts
```

The research agent is never the scorer, never the ledger authority during event
search, and never the owner of locked labels.

## Current Implementation Baseline

The current repo already has the benchmark and control-plane skeleton needed for
this loop:

- `autoalphafold3/schema.py` defines typed trial, result, prediction,
  falsification, and discovery records around `best_val_calpha_lddt` and
  `calpha_lddt_v1`.
- `autoalphafold3/preflight.py` validates parent commit existence, patch scope,
  config shape, registered prediction, optional manifest hashes, scorer version,
  artifact directory state, budget caps, NanoFold gates, and canonical scorer
  dry-run metrics.
- `autoalphafold3/patch_policy.py` enforces the allowed and denied edit surface.
- `autoalphafold3/modal_app.py` defines the deploy-once/call-many Modal app,
  data and locked Volumes, worker roles, secret boundaries, and resource tiers.
- `autoalphafold3/checkpoint_training.py` implements one real NanoFold training
  batch for checkpoint smoke evidence.
- `autoalphafold3/sampler.py` implements frozen-checkpoint inference-only
  sampler trials with bounded sampler knobs.
- `autoalphafold3/orchestrator.py` records stage-one decisions and keeps
  provisional `KEEP` separate from confirmed discovery.

The missing piece is not the scientific boundary. The missing piece is the
Karpathy/SimplexFold-style train/eval harness:

```text
candidate patch
  -> fixed-budget short training
  -> prediction artifacts
  -> locked scorer
  -> keep/revert decision
  -> next candidate
```

The one-batch checkpoint path must remain an infrastructure smoke. It should
not become the scientific object of the autoresearch loop.

## Execution Contracts

### Training Worker

The short-training worker must:

- mount `autoalphafold3-data` only
- never mount `autoalphafold3-locked`
- read public training features from the data Volume
- preserve `max_templates=0`
- write only under `/mnt/autoalphafold3/runs/trials/<trial_id>/`
- write `checkpoint.pt`, `short_training_manifest.json`,
  `training_log.json`, `loss_history.json`, `artifact_manifest.json`,
  `stdout.log`, `stderr.log`, `patch.diff`, and `DONE`
- stamp `real_training_performed=true`
- stamp `writes_ledger=false`, `writes_baseline=false`,
  `writes_discovery_ledger=false`
- call `Volume.commit()` after writing persistent artifacts

Per Modal Volume semantics, workers that need newly committed Volume state must
reload before reading it. The spec requires commit/reload behavior because
Volume changes made in one container are not visible to another running
container until committed and reloaded.

The training worker should reuse existing schema fields where possible:

- `AutoFoldTrial.max_steps` for optimizer-step budget
- `AutoFoldTrial.budget` for budget tier
- `AutoFoldTrial.config_path` for the experiment config
- `AutoFoldTrial.seed` for reproducibility
- `AutoFoldTrial.artifact_dir` for trial-scoped outputs

It should not invent a parallel trial payload unless the existing schema cannot
represent a required safety field.

### Scorer Worker

The scorer worker is the only path allowed to read locked validation labels. It
must:

- mount `autoalphafold3-data` read-only
- mount `autoalphafold3-locked` read-only
- read trial prediction artifacts
- compute `calpha_lddt_v1`
- return `best_val_calpha_lddt` as the primary metric
- not mutate trial code, baseline artifacts, or discovery artifacts

The official split remains `public_val_small`. Toy and local stub outputs may
exercise contracts, but they are not benchmark evidence.

### Orchestrator

The orchestrator must:

- submit only preflight-passed trials
- record lifecycle transitions
- decide stage-one `KEEP`, `DISCARD`, `FAIL`, or `INFRA_FAIL`
- treat every `KEEP` as provisional until Falsification Gate confirmation
- write canonical ledger rows during event search
- write Discovery Ledger records only for confirmed mechanisms

## Autoresearch Loop

One candidate iteration is:

1. Read `program.md`, this spec, current baseline, current best, Fold
   Cartographer report, recent ledger rows, and prior candidate summaries.
2. Choose exactly one diagnostic target and one move family.
3. Write a pre-registered hypothesis and prediction.
4. Patch only the allowed surface.
5. Create or update one experiment config under `configs/experiments/`.
6. Create one trial JSON under `trials/`.
7. Run local preflight.
8. Create a candidate commit or equivalent safe patch snapshot.
9. Run fixed-budget training/evaluation.
10. Score with the locked scorer.
11. Decide:
    - `KEEP` if it clears global current-best threshold
    - `DISCARD` if it runs but does not improve enough
    - `FAIL` for candidate/model/training/scorer-output failure
    - `INFRA_FAIL` for Modal, Volume, image, quota, timeout, or storage failure
12. Keep the candidate commit only for valid improvements.
13. Revert or abandon candidate code for valid misses.
14. Record the result and postmortem.
15. Pick the next candidate from evidence, not from wishful metric chasing.

The agent should not continue from a crash by silently changing the benchmark,
data, or infrastructure. It may fix trivial bugs inside the candidate patch and
rerun within the same candidate only when the idea remains the same.

## Candidate Planner Modes

The loop should support three planner modes, enabled in this order:

### Manual

Manual mode consumes a prepared candidate plan, patch, config, and trial JSON.
It is used to verify the harness before allowing agent-authored patches.

### Deterministic

Deterministic mode runs a fixed ablation ladder for the first research program.
It should not call an LLM. This proves artifact creation, patch validation,
training, scoring, decisions, and safe revert behavior.

### LLM

LLM mode lets the research agent author the candidate hypothesis and patch. It
must still pass patch policy, preflight, budget checks, Modal authority checks,
scoring, and stage-one decision rules. The LLM may propose code edits only in
the approved NanoFold/model/training surface.

## Git Contract

The loop should use git as an audit boundary.

Required behavior:

- Create a dedicated branch for one autoresearch run.
- Record the base commit.
- Use one candidate commit per attempted code/config change.
- Keep only commits that improve according to the configured comparison.
- Revert or abandon code for `DISCARD`, `FAIL`, and `INFRA_FAIL` candidates.
- Never revert unrelated user changes.
- Never include generated binary artifacts, checkpoints, Arrow files, baseline
  artifacts, or ledger rows in candidate commits.
- Preserve candidate diffs and run summaries under the run artifact directory.

Implementation should prefer a safe repo-local wrapper over raw interactive git
commands. The wrapper must refuse to delete untracked files and must stage only
allowed candidate paths.

## Artifact Contract

Autoresearch run artifacts should live outside the locked benchmark and outside
the baseline directory:

```text
runs/autoresearch/<run_id>/
  run_manifest.json
  results.tsv
  summary.json
  candidates/
    T123/
      hypothesis.md
      patch.diff
      config.json
      trial.json
      preflight.json
      training_manifest.json
      loss_history.json
      metrics.json
      error_report.json
      decision.json
      postmortem.md
```

Trial execution artifacts remain under:

```text
runs/trials/<trial_id>/
```

Generated run artifacts are evidence, not source. They should not be committed
unless a human explicitly promotes a small summary file for docs or UI.

Every candidate artifact must distinguish:

- pre-registered prediction
- generated patch
- training evidence
- scorer evidence
- decision evidence
- postmortem interpretation

Interpretation is not evidence unless it points to real artifacts.

## Metrics And Decisions

Primary metric:

- `best_val_calpha_lddt`

Supporting diagnostics:

- `local_geometry_weak`
- `long_range_topology_weak`
- `distogram_good_lddt_flat`
- `stability_compute`
- loss history
- runtime
- peak memory
- parameter count
- NaN, OOM, timeout, and failed-target counts

Decision comparisons:

- The matched-budget baseline answers whether a candidate improved the current
  short-training recipe.
- The locked global baseline answers whether a candidate is eligible for
  provisional `KEEP`.
- Sampler-family improvements over a frozen checkpoint do not become global
  discoveries unless they also clear the locked global threshold and controls.

## First Implementation Phases

### Phase 1: Specs And Runbook

Create:

- this specification
- `docs/runbooks/autoresearch-loop.md`
- `docs/spec/autoresearch-agent-program.md` or equivalent program prompt

The runbook should be operational and narrow. This spec remains the canonical
design document for the side-project loop.

### Phase 2: Short-Training Runner

Add:

- `autoalphafold3/short_training.py`
- `autoalphafold3/short_training_runner.py`
- manifest validation tests
- fixture-backed 2-3 step local test

The runner generalizes the one-batch checkpoint path without weakening it. The
existing one-batch checkpoint remains an infrastructure smoke, not a research
result.

The short-training runner should accept a tiny fixture-backed path for tests,
but local scaffold mode must not produce official benchmark-ready evidence.

### Phase 3: NanoFold Search-Surface Patch

Add minimal code support inside the allowed NanoFold files:

- config-driven loss weights
- differentiable local C-alpha pair-distance loss
- optional training-noise parameters only after the loss-weight path is stable

The first patch should avoid large higher-order architecture changes.

The relevant current NanoFold mechanics are:

- `diffusion_steps` affects inference sampling schedule, not training compute.
- training uses one denoising task per optimizer step, scaled mainly by
  `diffusion_batch_size`
- `Nanofold.get_total_loss` currently hard-codes
  `4 * diffusion_loss + 0.03 * dist_loss`
- the current lDDT-style term is threshold-based and is not a smooth coordinate
  learning signal

Therefore, the first minimal scientific patch should make loss weights
config-driven and add a differentiable local C-alpha pair-distance loss, rather
than trying a large architecture rewrite.

### Phase 4: Candidate Manager

Add:

- candidate ID allocation
- run artifact directory creation
- candidate patch snapshots
- result table updates
- safe keep/revert wrapper

### Phase 5: Autoresearch Loop

Add planner modes:

- `manual`: consumes a prepared candidate spec
- `deterministic`: runs the first fixed ablation ladder
- `llm`: proposes patches from program docs and prior evidence

The deterministic ladder must pass before the open-ended LLM loop is enabled.

### Phase 6: UI Evidence

Extend the static board to render:

- autoresearch run ID
- candidate trajectory
- kept/discarded/failed/crashed statuses
- patch summaries
- loss curves
- matched-budget deltas
- global-baseline deltas
- gate status for provisional KEEPs

The UI must distinguish real artifacts from labelled samples.

## Required Tests

Add or update tests so they prove:

- short-training manifests reject fake training claims
- `max_templates != 0` is rejected
- unsafe feature paths are rejected
- non-trial output directories are rejected
- non-empty output directories are rejected
- local scaffold mode cannot fabricate benchmark-ready evidence
- training workers never mount locked labels
- scorer-only workers remain the only locked-label readers
- preflight enforces budget caps
- patch policy rejects edits outside the search surface
- sampler trials remain inference-only and reject `max_steps`
- provisional `KEEP` does not write the Discovery Ledger
- generated UI labels sample fallback data honestly

Additional contract tests should prove:

- `TrialRunner.run(...)` remains the official training entrypoint
- no dynamic Modal `.with_options(...)` resource escalation appears in source
- training artifacts do not write `runs/baseline/**`
- training artifacts do not write canonical ledger or Discovery Ledger records
- a stage-one `KEEP` remains provisional until a confirmed gate verdict exists

## Acceptance Criteria

The side-project autoresearch loop is ready for a first live run only when:

- the specs and runbooks are committed
- local tests pass
- the deterministic ladder works in dry-run/planning mode
- fixture-backed short training writes honest artifacts
- Modal live execution remains gated by explicit approval
- scorer-only evaluation remains the only official metric path
- no code path fabricates baseline, ledger, Discovery Ledger, Modal, Arrow, or
  benchmark evidence
- README and UI claim wording avoid real AlphaFold3 reproduction/improvement
  claims

## Related Documents To Maintain

When this spec is implemented, create or update:

- `docs/runbooks/autoresearch-loop.md`: operator commands for init, dry-run,
  smoke, trial, review, and UI render.
- `docs/spec/autoresearch-agent-program.md`: the concise program prompt the
  LLM planner reads before authoring candidate patches.
- `docs/spec/autoresearch-acceptance-criteria.md`: checklist form of the
  implementation gates if this spec becomes too large for PR review.
- `docs/spec/ui/README.md`: data-to-UI contract for autoresearch run summaries.
- `README.md`: only after the loop exists, update Current Status and commands.

## Handoff Prompt

Use this prompt for the implementation agent:

```text
Implement the SimplexFold/Karpathy-style autoresearch loop described in
docs/spec/autoalphafold3-autoresearch-spec.md for the NanoFold-style
AlphaFold3-lite project.

First read AGENTS.md, README.md, docs/framing.md, program.md,
autoalphafold3/benchmark_contract.md, autoalphafold3/editable_surface.md, and
docs/runbooks/. Invoke the repo-local modal-docs skill before Modal-related
work.

The agent should be allowed to edit the approved NanoFold model/loss/training
surface, not only configs. Preserve all locked benchmark, scorer, data, Modal,
baseline, and ledger boundaries.

Implement in phases:
1. runbook/program docs
2. bounded short-training runner
3. minimal NanoFold loss/config support
4. candidate artifact manager
5. deterministic autoresearch ladder
6. LLM planner mode
7. UI evidence integration

Do not edit scorer code, validation labels, manifests, cached features, baseline
artifacts, Modal resource caps, or canonical ledger authority. Do not fabricate
benchmark results, Modal runs, gate verdicts, discovery records, Arrow files, or
baseline metrics.

Done means the deterministic ladder can plan and run fixture-backed checks, all
local tests pass, and every real metric shown by the UI is backed by scorer
artifacts.
```
