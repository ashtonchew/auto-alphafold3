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
official benchmark result. If `all_predictions_identical=true`, pause live
trial-budget autoresearch and diagnose stale artifacts, sampler determinism, or
candidate patch ineffectiveness before launching another candidate.

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
