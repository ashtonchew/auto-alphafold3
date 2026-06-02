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

For scored candidates, the next planner prompt receives a compact Fold
Cartographer block in `prior_decisions`: `signature`, `canonical_target`,
`mean_target_calpha_lddt`, selected summary counts, and compact bucket counts.
This makes the loop Fold Cartographer-driven without passing validation labels,
raw scorer internals, or bulky per-target artifacts into the planner.

The loop also keeps two comparisons separate:

- `global_current_best`: the locked baseline/ledger best. Only this comparison
  can produce stage-one `KEEP` and trigger the Falsification Gate.
- `search_reference`: a same-family sampler reference, usually the first
  default sampler run from the same frozen checkpoint. This comparison can
  report `SAMPLER_IMPROVED`, but it is not a discovery claim and never writes
  the Discovery Ledger.

The deterministic planner remains available for reproducible dry-runs and
tests. The LLM planner plugs into the same boundary and may only choose
sampler-only frozen-checkpoint settings:

- `sampler_steps`: integer `1..12`
- `seed`: integer `>=0`
- `sampler_noise_scale`: float `0.25..2.0`
- `sampler_step_scale`: float `0.25..2.0`
- `sampler_schedule_shape`: `linear`, `cosine`, or `late_refine`
- `sampler_num_samples`: integer `1..4`
- `sampler_selection_policy`: `first`, `geometry`, or `compact_geometry`
- hypothesis, diagnostic target, predicted direction, and expected lDDT delta band

The selection policies are target-blind and do not read validation labels:
`geometry` picks the sample with the lowest internal C-alpha geometry penalty,
and `compact_geometry` adds a mild compactness penalty. The planner cannot
author patches, submit directly, score directly, write ledgers, write the
Discovery Ledger, or change Modal resource policy.

After canonical smoke evidence identifies a same-family sampler reference, use
`--planner reference_sweep` for a bounded deterministic follow-up around the
recorded T088 neighborhood before spending a full LLM search window:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 4 \
  --start-trial-id T108 \
  --mode modal \
  --planner reference_sweep \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 300 \
  --failure-streak-limit 2 \
  --search-reference-trial-id T088 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

The reference sweep varies only sampler-only frozen-checkpoint settings near
`T088`: late-refine schedule, high sample count, lower noise, higher step scale,
and target-blind geometry or compact-geometry selection. It still writes only
candidate trial files and canonical ledger rows; it does not write the
Discovery Ledger, change locked assets, or alter Modal resource policy.

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
  --search-reference-trial-id T081 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

This is a legitimate autonomous research leg only when readiness is green and a
human explicitly approves the command. It starts search and may write canonical
ledger rows, but still never writes the Discovery Ledger; provisional `KEEP`
results must go through the Falsification Gate.

## Stop Rules

The loop stops when it reaches `--max-candidates` or when repeated live failures
hit `--failure-streak-limit`. It does not keep retrying indefinitely.

The loop also computes a sampler strategy context before each planner turn. It
finds the best sampler-family result in the local ledger and checks recent
prior sampler decisions against scorer-sensitivity reports. If at least three
late-refine compact/geometry neighborhood candidates regress against that
sampler-family ceiling, the strategy gate blocks another candidate from that
same neighborhood before writing a trial file. When scorer-sensitivity evidence
also shows all-target regression, the context records the stronger
`stop_t088_neighborhood` recommendation; score-only repeated regression records
`avoid_t088_neighborhood` and is still enforced for that exhausted local
neighborhood. The next planner turn must pivot to a distinct sampler mechanism
outside the exhausted neighborhood or leave sampler-only search for a
model-capacity/training-horizon diagnostic.

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

## 2026-05-30 Three-Candidate Sequential Smoke

A three-candidate GPT-5.4 mini sequential loop passed after the one-candidate
smoke:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 3 \
  --start-trial-id T083 \
  --mode modal \
  --planner llm \
  --model gpt-5.4-mini \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 180 \
  --failure-streak-limit 1 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

Result: `T083`, `T084`, and `T085` all reached `SAMPLER_PREDICTED`, scored
with `num_failed_targets=0`, and were recorded as `DISCARD`. The best sampler
candidate in the run was `T084` with `best_val_calpha_lddt=0.008722378043985426`.
This improved over `T083` and `T085`, but did not clear the locked baseline
score `0.07941230438543605`, so no provisional `KEEP` or Falsification Gate
run was triggered.

Planner behavior was coherent but narrow. It first tried a longer diffusion
trajectory (`sampler_steps=6`), then increased to `sampler_steps=7` after a
small score improvement, then tested a shorter counter-move (`sampler_steps=5`)
after the longer run. All three hypotheses targeted `local_geometry_weak` and
the sampler-step-length mechanism, which is valid for the sampler-only leg but
not yet diverse enough to claim broad autonomous research quality.

## 2026-05-30 Expanded Sampler-Knob Smokes

After adding bounded sampler schedule knobs and target-blind multi-sample
selection, the live loop was redeployed and rerun.

GPT-5.4 mini, three candidates:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 3 \
  --start-trial-id T086 \
  --mode modal \
  --planner llm \
  --model gpt-5.4-mini \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 300 \
  --failure-streak-limit 1 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

Result: `T086`, `T087`, and `T088` all reached `SAMPLER_PREDICTED`,
scored with `num_failed_targets=0`, and were recorded as `DISCARD`. The best
candidate was `T088` with `best_val_calpha_lddt=0.02098351201866366`, using
`sampler_steps=12`, `sampler_noise_scale=0.6`, `sampler_step_scale=1.5`,
`sampler_schedule_shape=late_refine`, `sampler_num_samples=4`, and
`sampler_selection_policy=compact_geometry`.

Fold Cartographer diagnostics remained `signature=toy_geometry_failed` and
`canonical_target=local_geometry_weak` across the run. Mean target C-alpha lDDT
moved from `0.013645332384870059` on `T086`, down to
`0.008044645887676049` on `T087`, then up to `0.01999694034070508` on `T088`.

GPT-5.5 override, three candidates:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 3 \
  --start-trial-id T089 \
  --mode modal \
  --planner llm \
  --model gpt-5.5 \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 300 \
  --failure-streak-limit 1 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

The first GPT-5.5 run scored `T089`, then exposed a planner-schema edge where
the structured plan allowed `flat`/negative expected delta even though
`AutoFoldTrial` does not. The schema was tightened and the remaining two
candidates continued as `T090` and `T091`. All three GPT-5.5 candidates reached
`SAMPLER_PREDICTED`, scored with `num_failed_targets=0`, and were recorded as
`DISCARD`; the best GPT-5.5 candidate was `T090` with
`best_val_calpha_lddt=0.011894151878161945`.

Fold Cartographer diagnostics also remained `signature=toy_geometry_failed` and
`canonical_target=local_geometry_weak` for the GPT-5.5 run. Mean target C-alpha
lDDT was `0.008506080525658197` for `T089`,
`0.011453887030133602` for `T090`, and `0.008044645887676049` for `T091`.

The expanded sampler surface materially improved the sampler-only ceiling in
this smoke from the prior best `0.008722378043985426` to `0.02098351201866366`,
but it still did not clear the locked baseline score `0.07941230438543605`.
Therefore no provisional `KEEP`, Falsification Gate run, or Discovery Ledger
write occurred.

Interpreted against a same-family sampler reference, the best expanded sampler
candidate was materially better: default sampler reference `T081` scored
`0.008276756926787072`, while `T088` scored `0.02098351201866366`. That is a
`0.012706755091876588` sampler-reference delta, about `2.54x` the default
sampler score. This is reported as sampler-family progress only; global
discovery status remains `DISCARD`.

## 2026-05-30 Fold Cartographer Feedback Smoke

After separating `global_current_best` from the same-family `search_reference`
and adding Fold Cartographer diagnostics to planner feedback, a three-candidate
GPT-5.4 mini loop passed:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 3 \
  --start-trial-id T092 \
  --mode modal \
  --planner llm \
  --model gpt-5.4-mini \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 300 \
  --failure-streak-limit 1 \
  --search-reference-trial-id T081 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

Result: `T092`, `T093`, and `T094` all reached `SAMPLER_PREDICTED`, scored
with `num_failed_targets=0`, and were recorded as global `DISCARD` while also
reporting `SAMPLER_IMPROVED` against the same-family `T081` sampler reference.
The best candidate in this verification run was `T094` with
`best_val_calpha_lddt=0.014790689139951244`, `global_delta=-0.0646216152454848`,
and `search_reference_delta=0.006513932213164172`.

Fold Cartographer feedback was present in each decision and therefore available
to the next planner turn. The observed signature stayed `toy_geometry_failed`,
the canonical target stayed `local_geometry_weak`, and mean target C-alpha lDDT
moved from `0.008694094947111403` on `T092` to `0.013783696655218005` on
`T093` and `0.014178717028126628` on `T094`.

## 2026-05-30 Canonical 12-Candidate Sampler Run

The canonical sampler trajectory uses the successful `T092`-`T094` Fold
Cartographer feedback smoke as prior decisions, then continues with nine live
GPT-5.4 mini candidates:

```bash
python -m autoalphafold3.agent autonomous-sampler-loop \
  --seed-trial trials/T012.json \
  --max-candidates 9 \
  --start-trial-id T095 \
  --mode modal \
  --planner llm \
  --model gpt-5.4-mini \
  --poll-interval-s 2 \
  --per-candidate-timeout-s 300 \
  --failure-streak-limit 1 \
  --search-reference-trial-id T081 \
  --prior-decision-trial-id T092 \
  --prior-decision-trial-id T093 \
  --prior-decision-trial-id T094 \
  --approve I_APPROVE_AUTONOMOUS_SAMPLER_LOOP
```

Result: the continuation returned `PASS`, scored `T095` through `T103`, and
stopped because `max_candidates` was reached. The canonical UI artifact is
`runs/canonical_sampler_12_candidate_run_2026-05-30.json`; it records all 12
candidate rows, sampler knobs, hypotheses, Fold Cartographer diagnostics,
same-family sampler-reference comparisons, global baseline comparisons, and
artifact pointers.

Best candidate in the 12-run was `T096` with
`best_val_calpha_lddt=0.018519489370625704`. It beat the same-family sampler
reference `T081` by `0.010242732443838632`, about `2.24x` the reference score,
but remained below the locked global baseline `T000` by
`0.06089281501481034`. Therefore every candidate remained global `DISCARD`;
no `KEEP`, Falsification Gate run, or Discovery Ledger write occurred.

All 12 candidates beat the same-family sampler reference and are reported as
`SAMPLER_IMPROVED` only. The Fold Cartographer signature stayed
`toy_geometry_failed`, and the canonical diagnostic target stayed
`local_geometry_weak`, so the honest interpretation is sampler-family progress
under a frozen checkpoint, not a global discovery.
