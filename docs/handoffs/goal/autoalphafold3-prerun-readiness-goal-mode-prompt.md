# AutoAlphaFold3 Pre-Run Readiness Goal-Mode Prompt

This is the successor goal-mode handoff after
`docs/handoffs/goal/autoalphafold3-foundation-goal-mode-prompt.md`.

The foundation goal built the scaffold, contracts, adapters, runner boundary,
Modal control plane, scorer boundary, and local verification stack. This goal
extends that foundation through canonical spec Sections 2-7 up to the point
where a human can make an explicit decision to run controlled live readiness
jobs or start autonomous search.

It must not start autonomous research search. It must not fabricate benchmark
results, fake Arrow files, fake Modal runs, fake baseline metrics, fake gate
verdicts, fake discovery records, or fake validation labels.

## Why This Shape

The foundation goal deliberately stopped once the repo scaffold and control
plane contracts were implemented and tested. Canonical spec Sections 2-7 add a
larger pre-run system:

- structured pre-registration predictions
- Falsification Gate contracts and verdict logic
- baseline lock/readiness validation
- two-stage KEEP/DISCARD/FAIL/INFRA_FAIL orchestration
- confirmed-only Discovery Ledger
- fakeable Modal gate-wave fanout
- readiness CLI/reporting before the first autonomous search trial

Those pieces are implementation work, but they are not autonomous research.
They should be tracked as a separate goal so "done" remains measurable and so
live Modal/baseline actions cannot slip in under a broad foundation label.

## Short Goal Prompt

Paste this into Codex Goal mode after the foundation PR stack is open or merged
and local `main` is current:

```text
/goal
Implement the AutoAlphaFold3 pre-run readiness layer for the NanoFold-style AlphaFold3-lite system described in docs/handoffs/goal/autoalphafold3-prerun-readiness-goal-mode-prompt.md. First fetch latest main, inspect the existing stacked foundation PR state, verify branch ownership and clean worktree status, and only rebase branches you own with --force-with-lease; otherwise branch from the accepted stack tip or stop for human direction. Read the required repo docs and canonical spec Sections 2-7, and create docs/handoffs/goal/autoalphafold3-prerun-readiness/{GOAL.md,PLAN.md,IMPLEMENTATION_CHECKLIST.md,EXPERIMENTS.md,EXPERIMENT_NOTES.md}. Invoke the repo-local modal-docs skill before Modal-related work. Use local Codex worktrees and ship feature-sized stacked PRs with semantic feat/... branch names and conventional commits. Before each feature PR, use parallel read-only subagents to ground the implementation in docs/spec/tests/Modal contracts, explain the best-practice implementation approach, then implement autonomously. Implement only pre-run infrastructure and contracts: structured pre-registration, Falsification Gate contracts and pure verdict logic, baseline-readiness validation, confirmed-only Discovery Ledger, two-stage orchestration, fakeable bounded Modal gate-wave adapter, and readiness CLI/reporting. Do not create fake benchmark/data artifacts, fake gate verdicts, fake discovery records, mutate baseline artifacts, or start autonomous research trials. Done only when the checklist is 100% complete or explicitly marks real external Modal/baseline execution as pending human action, all PRs are open, gate calibration is complete or blocking readiness pending human-approved live action, and final tests pass: python3 -m pytest -p no:cacheprovider and python3 .claude/skill-evals/run_offline_evals.py.
```

## Full Goal Contract

### Objective

Implement the pre-run readiness layer required before autonomous search can
begin. This goal turns the canonical spec's Sections 2-7 into executable
contracts, refusal-safe orchestration, and readiness reporting.

This goal ends immediately before autonomous research search. Controlled live
Modal readiness or baseline jobs are not autonomous search, but they still
require explicit human approval and must be reported separately from local
offline tests.

### Required First Steps

1. Fetch and inspect latest `main`:

   ```bash
   git fetch origin
   git checkout main
   git pull --ff-only
   ```

2. Verify the foundation stack before rebasing anything:

   - Existing foundation PRs are open or merged.
   - Local worktrees and branches are clean.
   - Branch ownership is clear and no unrelated user work is present.
   - Local branches are rebased on the latest intended bases only if they are
     owned by this goal run.
   - Any force-push uses `--force-with-lease`.
   - If ownership or cleanliness is unclear, branch from the accepted stack tip
     or stop for human direction instead of rewriting active review branches.
   - `docs/handoffs/goal/autoalphafold3-foundation/IMPLEMENTATION_CHECKLIST.md`
     is complete if the foundation stack is already accepted as done.

3. Read, in order:

   - `AGENTS.md`
   - `README.md`
   - `docs/framing.md`
   - `program.md`
   - `autoalphafold3/benchmark_contract.md`
   - `autoalphafold3/editable_surface.md`
   - `docs/runbooks/`
   - `docs/spec/autoalphafold3-canonical (2).html`, especially Sections 2-7
   - `.claude/skills/modal-docs/SKILL.md`

4. Invoke the `modal-docs` skill before implementing or changing Modal-related
   behavior. Prefer the repo-local Modal references for:

   - deployed class/function lookup
   - `Cls.from_name`
   - method `.spawn`
   - `FunctionCall.from_id(...).get(...)`
   - `map`/`starmap`
   - `return_exceptions=True`
   - Volume `commit()`/`reload()`
   - Sandbox trigger boundaries

5. Create these persistent files:

   - `docs/handoffs/goal/autoalphafold3-prerun-readiness/GOAL.md`
   - `docs/handoffs/goal/autoalphafold3-prerun-readiness/PLAN.md`
   - `docs/handoffs/goal/autoalphafold3-prerun-readiness/IMPLEMENTATION_CHECKLIST.md`
   - `docs/handoffs/goal/autoalphafold3-prerun-readiness/EXPERIMENTS.md`
   - `docs/handoffs/goal/autoalphafold3-prerun-readiness/EXPERIMENT_NOTES.md`

### Engineering Principles

- Locality of behavior: keep schema in schema modules, gate math in
  falsification modules, baseline lock behavior in baseline modules, and ledger
  writing in ledger modules.
- Modularity by feature: each PR implements one reviewable readiness slice.
- Simplicity: encode explicit contracts and refusal paths before broad
  abstractions.
- Exact spec compliance: use `docs/spec/autoalphafold3-canonical (2).html` as
  canonical if repo docs drift.
- Honest evidence: mocks and stubs are allowed only when clearly marked as
  non-official.
- Locked benchmark integrity: do not edit scorer math, validation membership,
  locked labels, cached feature outputs, fingerprints, baseline artifacts, or
  the Modal resource/cost envelope.
- Falsification integrity: once implemented, gate control construction,
  thresholds, verdict logic, and Discovery Ledger writes become locked
  benchmark infrastructure. Add patch-policy and tests that prevent future
  search patches from mutating them.
- Discovery integrity: a provisional KEEP is not a discovery. Only a
  CONFIRMED Falsification Gate verdict may enter the Discovery Ledger.

### Constraints

- Use the phrase NanoFold-style AlphaFold3-lite for the implementation target.
- Do not claim to train, reproduce, improve, or beat Google DeepMind
  AlphaFold3.
- Do not create fake benchmark results, fake Arrow files, fake Modal runs,
  fake baseline metrics, fake gate verdicts, fake discovery records, or fake
  validation labels.
- Do not start autonomous research search.
- Do not run upstream NanoFold download scripts or full feature rebuilding.
- Pin official runs to `max_templates=0`.
- Trial/sampler/debug workers must not mount or read locked validation labels.
- The scorer-only path may read locked manifests and labels.
- Workers never write the canonical ledger or Discovery Ledger.
- The local orchestrator remains the only ledger authority.
- Gate controls are orchestrator-authored, not agent-authored.
- Gate thresholds and verdict logic are frozen before autonomous search.
- Discovery Ledger entries are confirmed-only and orchestrator-written.
- Hidden validation is final-only and out of scope for this goal.

### Feature Stack

Use local Codex worktrees under a sibling directory such as
`../auto-alphafold3-worktrees/`. Ship feature-sized stacked PRs. Use these
semantic branches unless a better feature boundary is documented in `PLAN.md`.

1. `feat/falsification-contracts`
   - Base: latest foundation stack tip.
   - Scope: structured `RegisteredPrediction`, `FalsificationPlan`,
     `FalsificationResult`, verdict enums, discovery status fields, prediction
     preflight checks, pure verdict math, and tests for all verdict outcomes.

2. `feat/baseline-readiness`
   - Base: `feat/falsification-contracts`.
   - Scope: baseline lock reader/validator, scorer-version/hash checks,
     missing-baseline refusal, current-best lookup, and readiness report fields.
     Do not create fake baseline metrics.

3. `feat/discovery-ledger`
   - Base: `feat/baseline-readiness`.
   - Scope: confirmed-only Discovery Ledger helpers, provenance validation,
     discovery-record tests, renderer/readiness integration if needed, and
     policy tests proving non-orchestrator/search patches cannot mutate
     confirmed Discovery Ledger entries.

4. `feat/two-stage-orchestrator`
   - Base: `feat/discovery-ledger`.
   - Scope: stage-one threshold decision, provisional KEEP/DISCARD/FAIL/
     INFRA_FAIL processing, refusal to claim discoveries without gate evidence,
     lifecycle ledger transitions, and CLI-visible status.

5. `feat/gate-wave-modal-adapter`
   - Base: `feat/two-stage-orchestrator`.
   - Scope: fakeable Modal gate-wave adapter for `starmap(...,
     return_exceptions=True, wrap_returned_exceptions=False)`, control variant
     construction, bounded fanout and seed caps, per-control and aggregate
     timeout handling, returned exception-to-evidence handling, oversized-plan
     rejection before Modal submission, and no live Modal calls in tests.

6. `feat/pre-run-readiness-cli`
   - Base: `feat/gate-wave-modal-adapter`.
   - Scope: a single readiness command/report that says whether assets,
     baseline lock, deployed app contract, gate calibration, ledgers, tests, and
     skill evals are ready for a human-approved real run. Readiness must block
     autonomous search unless gate calibration is complete or explicitly pending
     a human-approved live calibration action.

Optional controlled-live PR:

- `feat/live-readiness-validation`
- Use only if a human explicitly approves controlled live Modal readiness or
  baseline jobs. This PR must distinguish live readiness from autonomous search
  and must not create fake artifacts if credentials or assets are unavailable.
  By default it is read-only/smoke validation. It must not write
  `runs/baseline/**`, locked Volumes, canonical ledgers, Discovery Ledger
  entries, benchmark artifacts, or baseline metrics unless a separate
  human-approved baseline-lock procedure is invoked and documented.

### Per-PR Loop

For each feature PR:

1. Spawn parallel read-only subagents to ground the work in docs, canonical
   spec, tests, and relevant Modal docs.
2. Record the best-practice implementation approach in `PLAN.md` before
   editing.
3. Implement the smallest coherent feature slice.
4. Run targeted tests.
5. Run full local tests if shared contracts changed:

   ```bash
   python3 -m pytest -p no:cacheprovider
   ```

6. Run skill evals if skills/docs/agent behavior changed:

   ```bash
   python3 .claude/skill-evals/run_offline_evals.py
   ```

7. Review the diff for locked-surface violations.
8. Update `PLAN.md`, `IMPLEMENTATION_CHECKLIST.md`, `EXPERIMENTS.md`, and
   `EXPERIMENT_NOTES.md`.
9. Commit, push, and open or update the stacked PR.

### Minimum Checklist Items

`IMPLEMENTATION_CHECKLIST.md` must include measurable checkboxes for at least:

- Latest `main` fetched and foundation stack safety recorded.
- Existing foundation PR state recorded.
- Branch ownership and clean worktree state verified before any rebase.
- Rebase skipped or stopped when branch ownership/user-work safety is unclear.
- Required docs and `modal-docs` skill read.
- Structured pre-registration schema implemented.
- Prediction preflight rejects missing causal component, axis, direction, or
  invalid expected delta band.
- Falsification plan/result schemas implemented.
- All five gate verdicts covered by pure local tests.
- Gate verdict math rejects missing scored controls.
- Gate thresholds and verdict logic are frozen before autonomous search.
- Gate control construction, thresholds, verdict logic, and Discovery Ledger
  writes are added to locked patch-policy coverage after implementation.
- Patch-policy tests reject search edits to implemented falsification controls.
- Forbidden-file patch scope gate is tested.
- Config schema gate is tested.
- Parameter-count cap gate is tested.
- Tiny forward-pass gate is implemented or explicitly blocks live readiness
  when dependencies/assets are unavailable.
- One-batch finite-loss gate is implemented or explicitly blocks live readiness
  when dependencies/assets are unavailable.
- Scorer dry-run schema gate is tested.
- Baseline lock reader implemented.
- Missing or incomplete baseline lock fails readiness honestly.
- No fake baseline metrics created.
- Discovery Ledger helper implemented.
- Discovery Ledger rejects non-CONFIRMED records.
- Discovery provenance includes git SHA, scorer version, feature/manifest
  hashes, axis, verdict numbers, and design rule.
- Stage-one threshold decision implemented.
- Provisional KEEP cannot enter Discovery Ledger.
- Gate controls are orchestrator-authored.
- Modal gate-wave adapter is fakeable in tests.
- Gate `starmap` uses `return_exceptions=True` and
  `wrap_returned_exceptions=False`.
- Gate-wave plan enforces max control variants and seed count before Modal
  submission.
- Gate-wave plan enforces per-control and aggregate timeouts.
- Oversized or unbounded gate-wave plans are rejected locally.
- Returned exceptions from knock-out, placebo, and seed controls normalize into
  structured falsification evidence and ledger-visible verdict reasons.
- Modal lookup/spawn/poll failures normalize to `INFRA_FAIL`.
- Modal gate-wave timeout/cancel/poll failures normalize deterministically.
- Trial workers still cannot mount locked labels.
- Scorer-only path remains the locked-label boundary.
- Readiness CLI/report implemented.
- Readiness report distinguishes local offline checks from human-approved live
  Modal/baseline actions.
- Readiness report blocks autonomous search until known-null and known-positive
  Falsification Gate calibration is complete, or marks the exact
  human-approved live calibration action as pending.
- Optional live readiness remains read-only/smoke unless a separate
  human-approved baseline-lock procedure is invoked.
- Optional live readiness cannot write `runs/baseline/**`, locked Volumes,
  canonical ledgers, Discovery Ledger entries, benchmark artifacts, or baseline
  metrics without that separate approval.
- `python3 -m pytest -p no:cacheprovider` passes.
- `python3 .claude/skill-evals/run_offline_evals.py` passes.
- No fake benchmark/data/gate/discovery artifacts were created.
- Autonomous research search was not started.

### Stop Conditions

Stop and report `BLOCKED` rather than improvising if:

- Foundation stack is not available, cannot be used as an accepted base, or
  cannot be safely rebased by this goal run.
- Modal docs cannot be verified for a Modal behavior change.
- Required real Modal credentials/assets are unavailable for an explicitly live
  step.
- Baseline lock data is unavailable and the requested step requires a real
  baseline.
- Known-null and known-positive gate calibration cannot be completed or honestly
  marked pending human-approved live action.
- Repo docs/spec conflict and the canonical spec cannot resolve the drift.
- The only way forward would require fake metrics, fake discoveries, hidden
  validation access, scorer math edits, or locked benchmark mutation.

### Done Criteria

- Persistent pre-run readiness goal files exist and are current.
- Feature-sized stacked PRs are open or updated.
- `IMPLEMENTATION_CHECKLIST.md` is 100% complete, except any checklist item
  requiring real external Modal/baseline execution is explicitly marked pending
  human action with the exact command or approval needed.
- Gate calibration is complete for known-null and known-positive controls, or
  readiness explicitly blocks autonomous search pending the exact
  human-approved live calibration action.
- Implemented falsification controls, gate thresholds, verdict logic, and
  Discovery Ledger writes are locked against future search patches.
- Final local tests pass:

  ```bash
  python3 -m pytest -p no:cacheprovider
  python3 .claude/skill-evals/run_offline_evals.py
  ```

- The final response reports branch stack, PR URLs, tests, checklist status,
  live actions still pending, and confirms autonomous search was not started.
