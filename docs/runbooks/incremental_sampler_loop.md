# Incremental Sampler Loop

This runbook operates the first repo-native autoresearch loop for
NanoFold-style AlphaFold3-lite sampler candidates. It is intentionally narrow:
it varies only inference-time sampler settings from the frozen checkpoint and
scores each candidate before choosing the next one.

## Dry Run

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 3 \
  --start-trial-id T013 \
  --mode dry-run \
  --planner deterministic
```

Dry-run writes candidate trial JSON files and does not call Modal, score, write
the ledger, or start search.

To exercise the LLM planning boundary without Modal submission, use:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 3 \
  --start-trial-id T013 \
  --mode dry-run \
  --planner llm \
  --model gpt-5.4-mini
```

The LLM planner uses OpenAI structured outputs and must return one
`SamplerCandidatePlan` at a time. The loop validates that plan into an
`AutoFoldTrial` before writing any trial file. When local OpenAI credentials are
not set, the planner calls the deployed Modal trusted orchestrator, where the
`openai-api-key` Secret is attached to the harness plane only. The secret is not
returned locally and is not passed to sampler, trial, or scorer workers.

## Live Loop

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 20 \
  --start-trial-id T013 \
  --mode modal \
  --planner llm \
  --model gpt-5.4-mini \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 180 \
  --failure-streak-limit 2 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

The loop is incremental, not a static sweep:

1. Ask the planner for one sampler candidate from current best and prior
   decisions.
2. Submit it through the trusted Modal orchestrator.
3. Wait for the sampler manifest.
4. Score with the locked scorer.
5. Record the scored row and stage-one decision in the canonical ledger.
6. Feed the observed score and diagnostics into the next planner call.

The deterministic planner remains available for reproducible dry-runs and
tests. The LLM planner plugs into the same boundary and may only choose
sampler-only frozen-checkpoint settings: `sampler_steps`, seed, hypothesis,
diagnostic target, predicted direction, and expected lDDT delta band. It cannot
author patches, submit directly, score directly, write ledgers, write the
Discovery Ledger, or change Modal resource policy.

The practical one-hour leg is:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 20 \
  --start-trial-id T013 \
  --mode modal \
  --planner llm \
  --model gpt-5.4-mini \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 180 \
  --failure-streak-limit 2 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

This is a legitimate autonomous research leg only when readiness is green and a
human explicitly approves the command. It starts search and may write canonical
ledger rows, but still never writes the Discovery Ledger; provisional `KEEP`
results must go through the Falsification Gate.

## Stop Rules

The loop stops when it reaches `--max-candidates` or when repeated live failures
hit `--failure-streak-limit`. It does not keep retrying indefinitely.

The loop never writes the Discovery Ledger. A stage-one `KEEP` is provisional
and must still pass the Falsification Gate before any discovery claim.

## 2026-05-30 Live Smoke

After deploying `autoalphafold3/modal_app.py` with Modal's documented deploy
path, a one-candidate live smoke passed:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 1 \
  --start-trial-id T081 \
  --mode modal \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 90 \
  --failure-streak-limit 1 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

Result: `T081` generated, sampler worker returned `SAMPLER_PREDICTED`, locked
scorer produced `best_val_calpha_lddt=0.008276756926787072`, and stage-one
recorded `DISCARD` with `num_failed_targets=0`. No Discovery Ledger row was
written.

## 2026-05-30 LLM Planner Smoke

After redeploying the trusted orchestrator with the harness-only
`openai-api-key` Secret, a one-candidate GPT-5.4 mini planner smoke passed:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 1 \
  --start-trial-id T082 \
  --mode modal \
  --planner llm \
  --model gpt-5.4-mini \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 180 \
  --failure-streak-limit 1 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

Result: `T082` generated from the LLM plan
`sampler_steps=6, seed=137`, sampler worker returned `SAMPLER_PREDICTED`,
locked scorer produced `best_val_calpha_lddt=0.008276756926787072`, and
stage-one recorded `DISCARD` with `num_failed_targets=0`. No Discovery Ledger
row was written.
