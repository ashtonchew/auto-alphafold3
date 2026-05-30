# One-Hour Autonomous Search Plan

This note records the practical search plan to use after pre-run readiness gates
are complete. It is a short-window operating decision layered on top of the
canonical design in `docs/spec/autoalphafold3-canonical (2).html`.

The implementation target remains a NanoFold-style AlphaFold3-lite system. This
plan does not relax benchmark integrity, scorer locking, Modal control-plane
rules, or Falsification Gate requirements.

## Preconditions

Do not start this search until these are true:

- The readiness report has no `BLOCKED` items.
- Any remaining pending items are explicit human-approved live actions.
- The Modal-hosted trusted orchestrator path is deployed and authenticated.
- The real baseline is locked and current-best lookup works.
- The locked scorer, manifest hashes, feature fingerprints, and `max_templates=0`
  policy pass readiness checks.
- Known-null and known-positive Falsification Gate calibration is complete, or
  the exact human-approved live calibration action has been run before search.
- The canonical ledger and Discovery Ledger write boundaries are intact.

## Decision

For a one-hour search window, run a sampler-only search burst with 20 candidates,
then gate only the best provisional KEEP.

This intentionally cuts breadth, not integrity. The search still uses
pre-registered hypotheses, Fold Cartographer diagnostics, the locked scorer, the
trusted Modal control plane, and the full Falsification Gate for the selected
candidate.

## Why Sampler-Only

Sampler-only search is the fastest legitimate path in the canonical search
surface. It uses a frozen checkpoint and varies inference-time diffusion/sampler
behavior, such as:

- sampler steps
- noise schedule
- step scale
- stochastic sample count
- label-free sample selection

This gives up the broader claim that the agent improved architecture, training,
losses, curriculum, or learned representations. The demo becomes an
inference-time mechanism story rather than a model-learning story.

What is preserved:

- pre-registered hypothesis testing
- Fold Cartographer axis prediction
- locked C-alpha lDDT scoring
- Modal fanout
- knock-out, placebo, predicted-axis, and seed controls
- canonical ledger and confirmed-only Discovery Ledger behavior
- structure overlay and falsification-card demo artifacts

## Candidate Count

Run 20 sampler candidates.

Current practical worker cap is the deployed `TrialRunner` cap, not the sampler
policy number:

- `TrialRunner`: `max_containers=6`
- `Scorer`: `max_containers=10` with `@modal.concurrent(max_inputs=4, target_inputs=2)`
- `sampler` budget policy: `max_containers=50`, but there is no separate
  deployed sampler worker using that cap in the current contract

So the expected sampler burst wall time is approximately:

```text
ceil(20 / 6) * per_candidate_runtime + queue/scoring overhead
```

That means four waves at the current practical generation cap. If candidates
take 2 minutes each, expect roughly 8-12 minutes. If they take 5 minutes each,
expect roughly 20-25 minutes.

Do not increase worker caps during search. Modal GPU type, timeouts, Volumes,
`max_containers`, warm pools, and cost caps remain locked unless a human
explicitly approves a separate infrastructure change.

## Gate Policy

Gate only the best provisional KEEP.

If multiple sampler candidates improve the primary metric, rank them by
`best_val_calpha_lddt` and choose the best candidate that also satisfies
quality/stability constraints. Other improving candidates may be logged as
provisional or unverified, but they must not enter the Discovery Ledger.

A candidate is not a discovery until it survives the Falsification Gate:

- knock-out control
- matched-magnitude placebo
- predicted-axis check
- seed rerun

## One-Hour Schedule

```text
0-8 min:
  Run readiness report, confirm baseline/current best, and sanity-check Modal path.

8-30 min:
  Run 20 sampler candidates.
  If the first wave is slow or unstable, stop early after 12-18 candidates.

30-36 min:
  Score and rank candidates.
  Select the best provisional KEEP.

36-54 min:
  Run the full Falsification Gate only on the selected candidate.

54-60 min:
  Freeze search.
  Summarize ledger state and capture demo inputs: trajectory, falsification card,
  Fold Cartographer diagnostics, and overlay artifacts.
```

## Success Targets

Preferred outcome:

- one sampler candidate improves `best_val_calpha_lddt`
- the best provisional KEEP runs the full gate
- the gate returns `CONFIRMED`
- the Discovery Ledger records one confirmed mechanism with provenance

Acceptable fallback:

- baseline is locked
- at least one provisional KEEP runs the full gate
- the verdict is an honest kill such as `PLACEBO_KILL`, `AXIS_MISS`,
  `KNOCKOUT_SURVIVES`, or `SEED_FRAGILE`
- the demo shows the control loop catching the false positive

Do not fabricate a winner. If no candidate improves, freeze and report the
negative result honestly.

## Cuts For Practicality

Cut for this one-hour run:

- broad architecture search
- Pairformer/training/loss/curriculum edits
- hundreds-scale sampler fanout
- multi-candidate gate waves
- hidden validation
- multi-seed validation for every candidate
- dashboard or demo polish during active search
- debugging beyond one quick retry for infrastructure failures

Keep:

- trusted Modal orchestrator path
- locked scorer and manifests
- `max_templates=0`
- pre-registration
- Fold Cartographer axis prediction
- gate on the selected best provisional KEEP
- append-only canonical ledger
- confirmed-only Discovery Ledger

