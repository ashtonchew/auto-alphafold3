# Autoresearch Loop Runbook

This runbook operates the post-hackathon SimplexFold/Karpathy-style
autoresearch loop for the NanoFold-style AlphaFold3-lite sandbox. It does not
authorize live Modal execution or open-ended search by itself.

## Claim Boundary

The loop may test bounded model, loss, training, config, and sampler changes
inside the approved surface. It must not claim to train, reproduce, improve, or
beat Google DeepMind AlphaFold3.

The primary decision metric is always `best_val_calpha_lddt` from
`calpha_lddt_v1`. Diagnostics may route hypotheses, but they do not replace the
primary metric.

## Before Any Candidate

1. Confirm the base and worktree:

   ```bash
   git fetch origin main
   gh pr view 50 --json number,state,mergedAt,mergeCommit,baseRefName,headRefName,title,url
   git worktree list
   ```

2. Read the operating contract:

   ```text
   AGENTS.md
   README.md
   docs/framing.md
   program.md
   autoalphafold3/benchmark_contract.md
   autoalphafold3/editable_surface.md
   docs/runbooks/
   docs/spec/autoalphafold3-autoresearch-spec.md
   docs/spec/autoresearch-agent-program.md
   ```

3. For Modal-related work, read `.claude/skills/modal-docs/SKILL.md` and the
   smallest relevant Modal reference pages before changing code.

4. Review current evidence:

   ```bash
   python3 -m autoalphafold3.agent readiness-report
   python3 -m autoalphafold3.agent llm-policy --format responses
   ```

## Dry-Run Planning

Dry-run planning must not call Modal, score official benchmark data, write the
canonical ledger, write the Discovery Ledger, or create baseline evidence.

Planned deterministic ladder command:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode dry-run \
  --planner deterministic \
  --run-id local-deterministic-001 \
  --start-trial-id T120
```

Expected behavior after implementation:

- create or preview candidate plans for T120-T125
- validate paths and budgets
- report that live execution remains disabled
- write only non-official planning artifacts when an output directory is
  explicitly supplied

## Fixture Smoke

Fixture smoke is for contract and tensor-path validation only. It is not a
benchmark result, not a Modal run, and not search evidence.

Planned command:

```bash
python3 -m autoalphafold3.agent run-short-training \
  --trial trials/T120.json \
  --mode local-fixture \
  --max-steps 3
```

The runner must stamp local fixture artifacts as:

- `official_benchmark_result=false`
- `local_only=true`
- `real_training_performed=true` only when a real fixture training loop ran
- `writes_baseline=false`
- `writes_ledger=false`
- `writes_discovery_ledger=false`
- `max_templates=0`

It must reject unsafe feature paths, `max_templates != 0`, non-trial output
directories, and non-empty output directories unless an explicit resume mode is
implemented.

## Deterministic Ladder

The first ladder proves the loop without an LLM:

| Trial | Purpose | Budget |
| --- | --- | --- |
| `T120` | matched short-training baseline smoke | 10 steps |
| `T121` | first local-geometry patch smoke | 10 steps |
| `T122` | matched short-training baseline trial | 250 steps |
| `T123` | best local-geometry patch trial | 250 steps |
| `T124` | no-geometry auxiliary ablation | 250 steps |
| `T125` | sampler after best checkpoint | inference only |

Planned command:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode dry-run \
  --planner deterministic \
  --run-id local-deterministic-001 \
  --start-trial-id T120 \
  --max-candidates 6
```

Promotion rules:

- smoke failure is `FAIL` or `INFRA_FAIL`
- a valid 250-step miss is `DISCARD`
- a matched-budget improvement may become the short-training current best
- only a global current-best improvement can become provisional `KEEP`
- provisional `KEEP` remains non-discovery until Falsification Gate
  confirmation

Stop conditions:

- Do not retry a failed candidate more than once unless the failure is a
  trivial candidate-local bug and the hypothesis, move family, and budget stay
  unchanged.
- Stop a move family after three candidate-level `FAIL` results until a human
  reviews the failure pattern.
- Stop or downgrade the family after two OOM, NaN, timeout, or failed-target
  failures in the same budget tier; do not raise Modal resources from the
  loop.
- A repeated Falsification Gate kill for the same mechanism family blocks more
  variants of that family until the hypothesis is rewritten from evidence.

## Candidate Evidence

Each candidate should write or reference:

```text
runs/autoresearch/<run_id>/
  run_manifest.json
  results.tsv
  summary.json
  candidates/
    <trial_id>/
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

Generated run artifacts are not source commits unless a human explicitly
promotes a small summary artifact for documentation or UI samples.

## Safe Git Discipline

Use one branch per autoresearch run and one candidate commit per attempted
code/config change. Keep only commits that improve according to the configured
comparison. Revert or abandon valid misses and failures without touching
unrelated user changes.

The safe wrapper added in a later PR must:

- stage only approved candidate source/config paths
- refuse locked paths and generated binary artifacts
- refuse `runs/baseline/**`, ledger rows, Discovery Ledger rows, Arrow files,
  checkpoints, fingerprints, and validation labels
- refuse to delete untracked user files
- preserve candidate diffs under the run artifact directory

## Human-Approved Live Actions

No command in this section should run without the exact approval token.

Short-training Modal trial, after local fixture and deterministic planning pass:

```bash
python3 -m autoalphafold3.agent run-short-training \
  --trial trials/T123.json \
  --mode modal \
  --approve I_APPROVE_SHORT_TRAINING_TRIAL
```

This command is a planned human-operator wrapper. Its implementation must
delegate through the Modal-hosted trusted orchestrator and the approved
`AutoFoldTrial` submission boundary; it must not call trial workers directly or
create a parallel live execution authority.

Bounded live autoresearch smoke, after deterministic ladder passes and a human
reviews the plan:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode modal \
  --planner deterministic \
  --run-id live-autoresearch-001 \
  --start-trial-id T130 \
  --max-candidates 1 \
  --approve I_APPROVE_AUTORESEARCH_LIVE_SEARCH
```

This implemented live path is intentionally narrow: it plans exactly one
training candidate, submits it through the deployed Modal
`TrustedOrchestrator`, polls the returned worker call id, invokes the deployed
scorer-only worker, and writes only local autoresearch candidate artifacts. It
does not write the canonical ledger or Discovery Ledger. If the scorer returns
`SCORED`, the loop writes artifact-only metrics and a provisional `KEEP` or
`DISCARD` decision. If the scorer returns `FAIL` because required prediction
artifacts are missing or invalid, the loop records a local candidate
`error_report.json` and terminal `decision.json` instead of fabricating a score.

LLM-authored candidates for the same path still require a recorded one-candidate
plan and the same exact approval token:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode modal \
  --planner llm \
  --candidate-plan configs/experiments/recorded-live-candidate.json \
  --run-id live-autoresearch-llm-001 \
  --start-trial-id T130 \
  --max-candidates 1 \
  --approve I_APPROVE_AUTORESEARCH_LIVE_SEARCH
```

Open-ended autonomous search remains out of scope. Future extensions must still
submit one validated `AutoFoldTrial` at a time through the trusted orchestrator.
They may plan candidates and collect returned evidence, but must not bypass
preflight, scorer-only evaluation, canonical ledger authority, or Modal resource
policy.

After repeated smoke-budget `DISCARD` results show no score sensitivity, the LLM
planner may be moved to the bounded 250-step trial budget explicitly:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode modal \
  --planner llm \
  --candidate-budget trial \
  --prior-run-id live-autoresearch-<previous> \
  --run-id live-autoresearch-trial-001 \
  --start-trial-id T160 \
  --max-candidates 1 \
  --failure-streak-limit 2 \
  --approve I_APPROVE_AUTORESEARCH_LIVE_SEARCH
```

`--candidate-budget trial` changes only the generated LLM trial shape:
`budget=trial`, `max_steps=250`, `max_wall_minutes=45`, and
`timeout_cap=2700`. It does not change Modal GPU/resource settings, ledger
authority, scorer authority, or promotion rules. Keep this at one candidate per
live run until the cost and runtime profile are measured.

## Prediction Artifact Comparison

Repeated identical scorer values require artifact evidence before spending more
trial budget. Download the candidate `predictions.json` files from
`autoalphafold3-data` first, then compare them locally:

```bash
modal volume get autoalphafold3-data runs/trials/T150/predictions.json /tmp/T150-predictions.json
modal volume get autoalphafold3-data runs/trials/T157/predictions.json /tmp/T157-predictions.json
modal volume get autoalphafold3-data runs/trials/T150/metrics.json /tmp/T150-metrics.json
modal volume get autoalphafold3-data runs/trials/T157/metrics.json /tmp/T157-metrics.json

python3 -m autoalphafold3.agent compare-predictions \
  /tmp/T150-predictions.json \
  /tmp/T157-predictions.json \
  --left-metrics /tmp/T150-metrics.json \
  --right-metrics /tmp/T157-metrics.json \
  --output runs/autoresearch/prediction_comparisons/T150-vs-T157.json
```

The comparison report is diagnostic evidence only. It does not score a
candidate, write the canonical ledger, write the Discovery Ledger, or create an
official benchmark result. It includes artifact hashes, per-target prediction
hashes, optional metric deltas, target-level coordinate deltas such as RMSD and
mean absolute coordinate shift, and pairwise C-alpha distance deltas. Treat the
pairwise distance deltas as the scorer-aligned artifact signal; raw coordinate
RMSD can be misleading because C-alpha lDDT is based on pair-distance errors
and is invariant to global translation or rigid-body movement. If
`all_predictions_identical=true`, pause live trial-budget autoresearch and
diagnose stale artifacts, sampler determinism, or candidate patch
ineffectiveness before launching another candidate. If predictions differ but
metric deltas remain exactly zero, inspect the pairwise-distance summary before
deciding whether the scorer is saturated, the scorer inputs are miswired, or the
candidate is only moving already-failed geometry.

## Read-Only Scorer Sensitivity

When prediction artifacts differ but aggregate metrics remain pinned, run a
read-only scorer-only diagnostic before another live trial-budget candidate:

```bash
python3 -m autoalphafold3.agent scorer-sensitivity \
  --mode dry-run \
  --trial-id T150 \
  --trial-id T157

python3 -m autoalphafold3.agent scorer-sensitivity \
  --mode modal \
  --modal-env main \
  --trial-id T150 \
  --trial-id T157 \
  --approve I_APPROVE_SCORER_SENSITIVITY_DIAGNOSTIC \
  --output runs/autoresearch/scorer_sensitivity/T150-vs-T157.json
```

This command calls the deployed scorer-only worker for existing trial artifacts
only. It does not submit trials, start search, write the canonical ledger, write
the Discovery Ledger, or create candidate promotion evidence. Use it to
distinguish stale local metrics from true scorer/metric insensitivity.
The report includes per-target lDDT details from the scorer payload: target
score, eligible-pair count, scored-residue count, NaN prediction residue count,
threshold fractions, and per-target score deltas against the reference trial.
If aggregate metric deltas stay pinned while pairwise prediction distances
changed, inspect these per-target fields before running another live
trial-budget candidate.

## Targeted Diagnostic Candidate

If a bounded sampler/reference sweep fails but the read-only scorer-sensitivity
report shows real, target-mixed deltas, use `targeted_diagnostic` before
spending a larger sampler-only LLM loop. This planner consumes the existing
scorer-sensitivity JSON report and emits exactly one bounded training candidate
focused on recurring loser targets. It does not read labels, download data,
change manifests, alter Modal resources, write the canonical ledger, or write
the Discovery Ledger during planning.

Dry-run the candidate plan first:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode dry-run \
  --planner targeted_diagnostic \
  --candidate-budget trial \
  --diagnostic-report runs/autoresearch/scorer_sensitivity/T088-vs-T108-T111-reference-sweep.json \
  --run-id targeted-diagnostic-trial-001 \
  --start-trial-id T160
```

Review the generated candidate envelope under
`runs/autoresearch/targeted-diagnostic-trial-001/candidates/T160/`. The trial
must remain a NanoFold-style AlphaFold3-lite training candidate with
`max_templates=0`, `budget=trial`, `max_steps=250`, `max_wall_minutes=45`, and
`timeout_cap=2700`. The candidate note records the recurrent loser targets
from the diagnostic report; the executable `config_payload` stays a validated
NanoFold config object.

Only after reviewing the dry-run envelope and confirming readiness remains
green, run at most one live candidate:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode modal \
  --planner targeted_diagnostic \
  --candidate-budget trial \
  --diagnostic-report runs/autoresearch/scorer_sensitivity/T088-vs-T108-T111-reference-sweep.json \
  --run-id targeted-diagnostic-trial-001-live \
  --start-trial-id T160 \
  --modal-env main \
  --failure-streak-limit 1 \
  --approve I_APPROVE_AUTORESEARCH_LIVE_SEARCH
```

If the live candidate is `DISCARD`, archive the candidate evidence and pivot
the next plan using the new scorer-sensitivity report. If it is a provisional
`KEEP`, do not claim a result; run the Falsification Gate path first.

If the targeted diagnostic regresses across all targets, do not increase local
geometry pressure again. Use `schedule_diagnostic` for one narrower
optimizer/schedule candidate that backs off the failed geometry-loss shape:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode dry-run \
  --planner schedule_diagnostic \
  --candidate-budget trial \
  --diagnostic-report runs/autoresearch/scorer_sensitivity/T088-vs-T160-targeted-diagnostic.json \
  --run-id schedule-diagnostic-trial-001 \
  --start-trial-id T161
```

Review the generated `T161` envelope before live execution. It must still be a
single NanoFold-style AlphaFold3-lite trial-budget training candidate with
`max_templates=0`, no ledger writes, no Discovery Ledger writes, and no Modal
resource changes. The planner changes only validated inline config values such
as learning rate, warmup, clipping, and a lower local-geometry loss weight.

The matching one-candidate live command is:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode modal \
  --planner schedule_diagnostic \
  --candidate-budget trial \
  --diagnostic-report runs/autoresearch/scorer_sensitivity/T088-vs-T160-targeted-diagnostic.json \
  --run-id schedule-diagnostic-trial-001-live \
  --start-trial-id T161 \
  --modal-env main \
  --failure-streak-limit 1 \
  --approve I_APPROVE_AUTORESEARCH_LIVE_SEARCH
```

If the schedule diagnostic and the sampler-only pivot both regress or stay flat
against the sampler-family ceiling, use `capacity_diagnostic` for one bounded
model-capacity training candidate. This planner changes only validated inline
config values for small width/depth capacity plus conservative optimizer
settings. It does not edit source, scorer, manifests, fingerprints, Modal
resources, templates, baselines, the canonical ledger, or the Discovery Ledger.

Dry-run the candidate plan first:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode dry-run \
  --planner capacity_diagnostic \
  --candidate-budget trial \
  --diagnostic-report runs/autoresearch/scorer_sensitivity/T088-vs-T113-strategy-pivot.json \
  --run-id capacity-diagnostic-trial-001 \
  --start-trial-id T162
```

Review the generated `T162` envelope. It must remain a single
NanoFold-style AlphaFold3-lite trial-budget training candidate with
`max_templates=0`, `budget=trial`, `max_steps=250`, `max_wall_minutes=45`, and
`timeout_cap=2700`. The diagnostic note should record the failed
local-geometry, schedule, and sampler-only shapes it is avoiding.

Only after readiness remains green, run at most one live capacity candidate:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode modal \
  --planner capacity_diagnostic \
  --candidate-budget trial \
  --diagnostic-report runs/autoresearch/scorer_sensitivity/T088-vs-T113-strategy-pivot.json \
  --run-id capacity-diagnostic-trial-001-live \
  --start-trial-id T162 \
  --modal-env main \
  --failure-streak-limit 1 \
  --approve I_APPROVE_AUTORESEARCH_LIVE_SEARCH
```

If the capacity diagnostic is also `DISCARD`, treat the current tiny-model
training/sampler evidence as insufficient for broader autonomous search. The
next step should be a new explicit design review of allowed architecture or
data/feature surfaces, not another local variant of T160/T161/T113.

After that design review, the next distinct one-candidate surface is
`topology_recycling_diagnostic`. It tests the allowed recycling move family
instead of repeating sampler tuning, local-geometry pressure, optimizer/schedule
backoff, or width/depth capacity. The candidate changes only validated inline
config values: one extra trunk recycle, conservative optimizer settings,
`max_templates=0`, and no source, scorer, manifest, fingerprint, Modal resource,
baseline, canonical ledger, or Discovery Ledger edits.

Dry-run the candidate plan first:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode dry-run \
  --planner topology_recycling_diagnostic \
  --candidate-budget trial \
  --diagnostic-report runs/autoresearch/scorer_sensitivity/T088-vs-T162-capacity-diagnostic.json \
  --run-id topology-recycling-diagnostic-trial-001 \
  --start-trial-id T163
```

Review the generated `T163` envelope. It must remain a single
NanoFold-style AlphaFold3-lite trial-budget training candidate with
`diagnostic_target=long_range_topology_weak`, `move_family=recycling`,
`num_recycle=2`, `max_templates=0`, `budget=trial`, `max_steps=250`,
`max_wall_minutes=45`, and `timeout_cap=2700`.

Only after readiness remains green, run at most one live recycling candidate:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode modal \
  --planner topology_recycling_diagnostic \
  --candidate-budget trial \
  --diagnostic-report runs/autoresearch/scorer_sensitivity/T088-vs-T162-capacity-diagnostic.json \
  --run-id topology-recycling-diagnostic-trial-001-live \
  --start-trial-id T163 \
  --modal-env main \
  --failure-streak-limit 1 \
  --approve I_APPROVE_AUTORESEARCH_LIVE_SEARCH
```

If the recycling diagnostic is also `DISCARD`, pause live trial-budget spend.
At that point the repo has negative scorer-backed evidence across sampler,
local-geometry, optimizer/schedule, width/depth capacity, and recycling
surfaces. The next phase should be a new issue for a different allowed surface
or a deeper artifact diagnosis, not another immediate live candidate.

## Post-Discard Diagnosis

After pausing live trial-budget spend, classify the accumulated evidence
offline before approving a new surface:

```bash
python3 -m autoalphafold3.agent post-discard-diagnosis \
  --scorer-report runs/autoresearch/scorer_sensitivity/T088-vs-T113-strategy-pivot.json \
  --scorer-report runs/autoresearch/scorer_sensitivity/T088-vs-T162-capacity-diagnostic.json \
  --scorer-report runs/autoresearch/scorer_sensitivity/T088-vs-T163-topology-recycling.json \
  --prediction-comparison runs/autoresearch/prediction_comparisons/T088-vs-T113.json \
  --prediction-comparison runs/autoresearch/prediction_comparisons/T088-vs-T162.json \
  --prediction-comparison runs/autoresearch/prediction_comparisons/T088-vs-T163.json \
  --exhausted-surface sampler \
  --exhausted-surface local_geometry \
  --exhausted-surface optimizer_schedule \
  --exhausted-surface width_depth \
  --exhausted-surface recycling \
  --output runs/autoresearch/post_discard_diagnosis/T113-T162-T163.json
```

This is a local/offline evidence classifier. It does not submit trials, score
artifacts, start search, write the canonical ledger, write the Discovery
Ledger, or create benchmark claims. If it emits
`SHORT_TRAINING_FAMILY_SCORER_COLLAPSE`, do not run another immediate live
trial-budget candidate from the same short-training family. Define a new issue
around short-training initialization, artifact scale, or feature/curriculum
handling first, then dry-run exactly one candidate before any further Modal
spend.

## Review And UI Render

Before each implementation or source-behavior PR:

```bash
git diff --check
python3 -m pytest -p no:cacheprovider
```

Docs-only PRs may run the relevant documentation/eval checks instead, but any
skipped source test must be recorded in the PR body with the reason.

Current labelled sample render:

```bash
python3 -m autoalphafold3.ui.build --sample --out public
```

Planned autoresearch evidence render after the UI integration exists:

```bash
python3 -m autoalphafold3.ui.build \
  --autoresearch-summary runs/autoresearch/<run_id>/summary.json \
  --out public
```

Sample fallback rendering must be visibly labelled and must not invent missing
metrics.
