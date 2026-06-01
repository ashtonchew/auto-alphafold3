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

Open-ended autoresearch loop, after deterministic ladder passes and a human
reviews the plan:

```bash
python3 -m autoalphafold3.agent autoresearch-loop \
  --mode modal \
  --planner llm \
  --run-id live-autoresearch-001 \
  --start-trial-id T130 \
  --max-candidates 1 \
  --approve I_APPROVE_AUTORESEARCH_LIVE_SEARCH
```

These are `PENDING_HUMAN_LIVE_ACTION` commands until the implementation exists
and a human explicitly approves them.

## Review And UI Render

Before each PR:

```bash
git diff --check
python3 -m pytest -p no:cacheprovider
```

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
