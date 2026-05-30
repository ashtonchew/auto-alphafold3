# AutoAlphaFold3 Modal Pre-Run Readiness Goal-Mode Prompt

This is the successor goal-mode handoff after
`docs/handoffs/goal/autoalphafold3-foundation-goal-mode-prompt.md` and the PR
#21 canonical spec update that makes the event orchestrator a Modal-hosted
trusted CPU service.

The foundation goal built the scaffold, contracts, adapters, runner boundary,
Modal control-plane metadata, scorer boundary, and local verification stack.
PR #21 changes the pre-run target: local orchestration is now scaffold smoke
only. The event path must be a Modal-hosted trusted orchestrator/harness that
owns policy, secrets, gate execution, canonical ledger writes, and Discovery
Ledger writes while keeping trial, sampler, scorer, and debug workers on the
non-secret execution plane.

This goal brings the NanoFold-style AlphaFold3-lite system to the point where
the only remaining action is an explicit human-approved live run, baseline lock,
gate calibration, or autonomous search start. It must not start autonomous
research search. It must not fabricate benchmark results, Arrow files, Modal
runs, baseline metrics, gate verdicts, discovery records, or validation labels.

## Why This Shape

The earlier pre-run readiness prompt was correctly shaped, but PR #21 makes one
of its assumptions unsafe: the local orchestrator can no longer be treated as
the event search-loop authority. The canonical architecture now requires a
CPU-only Modal-hosted trusted orchestrator and a two-plane secret boundary:

- harness plane: orchestrator, OpenAI/agent credentials, Modal/GitHub/judge
  credentials, authenticated submission, policy, gate, and ledgers
- execution plane: trial, sampler, scorer, and debug workers with only the
  mounts and non-secret environment needed for their role

This handoff keeps the proven PR #14 Goal Mode style but folds in the hardening
learned during that PR: branch ownership before rebasing, bounded Modal fanout,
frozen gate thresholds, patch-policy lock coverage, honest baseline handling,
known-null and known-positive gate calibration, and explicit stop conditions.

The end state is not "search has started." The end state is "implementation,
local/offline validation, mocked Modal validation, readiness reporting, and PR
review surfaces are complete; any remaining action is explicitly listed as a
human-approved live action."

## Short Goal Prompt

Paste this into Codex Goal mode after PR #21 is open or merged and local `main`
or the accepted stack tip is current:

```text
/goal
Implement the post-PR-21 AutoAlphaFold3 Modal pre-run readiness layer for the NanoFold-style AlphaFold3-lite system described in docs/handoffs/goal/autoalphafold3-modal-prerun-readiness-goal-mode-prompt.md. First fetch latest main and PR #21, inspect the accepted stack tip, verify branch ownership and clean worktree status, and only rebase branches you own with --force-with-lease; otherwise branch from the accepted stack tip or stop for human direction. Read AGENTS.md, README.md, docs/framing.md, program.md, autoalphafold3/benchmark_contract.md, autoalphafold3/editable_surface.md, docs/runbooks/, docs/spec/autoalphafold3-canonical (2).html, and the repo-local modal-docs skill before Modal work. Create docs/handoffs/goal/autoalphafold3-modal-prerun-readiness/{GOAL.md,PLAN.md,IMPLEMENTATION_CHECKLIST.md,EXPERIMENTS.md,EXPERIMENT_NOTES.md}. Use local Codex worktrees and open four feature-sized stacked PRs: feat/modal-prerun-contracts, feat/falsification-and-ledgers, feat/baseline-scorer-readiness, and feat/prerun-readiness-report. Before each feature PR, use parallel read-only subagents to ground the implementation in docs/spec/tests/Modal contracts, record the best-practice approach in PLAN.md, then implement autonomously. Build only pre-run infrastructure: the Modal-hosted trusted orchestrator/harness contract, worker role and secret-boundary enforcement, Falsification Gate, canonical ledger and confirmed-only Discovery Ledger, baseline/scorer readiness, asset checks, and a readiness report that certifies whether the only remaining item is an explicit human-approved live action. Do not create fake benchmark/data artifacts, fake Modal runs, fake baseline metrics, fake gate verdicts, fake discovery records, mutate baseline artifacts, or start autonomous research trials. Done only when all four PRs are open or merged, the checklist is 100% complete except items explicitly marked PENDING_HUMAN_LIVE_ACTION with exact approval/command, gate calibration is complete or readiness blocks search pending exact human approval, and final validation passes: python3 -m pytest -p no:cacheprovider, python3 .claude/skill-evals/run_offline_evals.py, and git diff --check.
```

## Full Goal Contract

### Objective

Implement the pre-run readiness layer required after PR #21 before autonomous
search can begin. This goal turns the canonical spec into executable
infrastructure, refusal-safe orchestration, and a readiness certificate whose
only allowed unresolved items are explicit human-approved live actions.

This goal ends immediately before autonomous research search. Controlled live
Modal readiness jobs, baseline freezing, gate calibration, and autonomous
search start are separate live actions. They are allowed only after explicit
human approval and must be reported separately from local/offline tests.

### Certification Target

The final readiness report must classify every required gate as exactly one of:

- `PASS_LOCAL`: proven by local tests, static checks, or offline fixtures.
- `PASS_MOCKED_MODAL`: proven with fakeable Modal adapters and no live Modal
  call.
- `PASS_LIVE`: proven by an explicitly approved live Modal or baseline action.
- `PENDING_HUMAN_LIVE_ACTION`: implementation is complete, local/mocked checks
  pass, and the remaining item requires a named human-approved live command,
  credential, cost decision, or baseline lock action.
- `BLOCKED`: implementation cannot honestly proceed or the remaining work is
  not merely a live approval.

Autonomous search must remain blocked unless every required item is
`PASS_LOCAL`, `PASS_MOCKED_MODAL`, or `PASS_LIVE`, except for final items that
are explicitly `PENDING_HUMAN_LIVE_ACTION` with the exact command or approval
needed. Missing implementation, missing tests, missing baseline readers, missing
gate verdict logic, missing secret-boundary enforcement, or missing readiness
reporting may not be labeled as pending live action.

### Required First Steps

1. Fetch and inspect latest `main` and PR #21:

   ```bash
   git fetch origin
   gh pr view 21 --json number,title,headRefName,baseRefName,commits,files
   ```

2. Verify stack safety before rebasing or force-pushing:

   - The intended base is PR #21 head, a merged PR #21 commit, or another
     accepted stack tip named by a human.
   - Local worktrees and branches are clean.
   - Branch ownership is clear and no unrelated user work is present.
   - Local branches are rebased only if they are owned by this goal run.
   - Any force-push uses `--force-with-lease`.
   - If ownership, cleanliness, or base selection is unclear, branch from the
     accepted stack tip or stop for human direction instead of rewriting active
     review branches.

3. Read, in order:

   - `AGENTS.md`
   - `README.md`
   - `docs/framing.md`
   - `program.md`
   - `autoalphafold3/benchmark_contract.md`
   - `autoalphafold3/editable_surface.md`
   - `docs/runbooks/`
   - `docs/spec/autoalphafold3-canonical (2).html`, especially Sections 2-8
   - `.claude/skills/modal-docs/SKILL.md`

4. Invoke the repo-local `modal-docs` skill before implementing or changing
   Modal-related behavior. Prefer the repo-local Modal references for:

   - deployed apps and deploy-once/call-many behavior
   - web endpoint or queue worker shape
   - `Function.from_name`
   - `Cls.from_name`
   - `.spawn`
   - `FunctionCall.from_id(...).get(...)`
   - `map` and `starmap`
   - `return_exceptions=True`
   - Volume `commit()` and `reload()`
   - Secrets
   - Sandbox trigger boundaries

5. Create and maintain these persistent files:

   - `docs/handoffs/goal/autoalphafold3-modal-prerun-readiness/GOAL.md`
   - `docs/handoffs/goal/autoalphafold3-modal-prerun-readiness/PLAN.md`
   - `docs/handoffs/goal/autoalphafold3-modal-prerun-readiness/IMPLEMENTATION_CHECKLIST.md`
   - `docs/handoffs/goal/autoalphafold3-modal-prerun-readiness/EXPERIMENTS.md`
   - `docs/handoffs/goal/autoalphafold3-modal-prerun-readiness/EXPERIMENT_NOTES.md`

### Engineering Principles

- Exact spec compliance: use `docs/spec/autoalphafold3-canonical (2).html` as
  canonical if docs drift.
- Locality of behavior: keep schema in schema modules, Modal contract behavior
  in Modal/orchestrator modules, gate math in falsification modules, baseline
  behavior in baseline/readiness modules, and ledger writes in ledger modules.
- Modularity by feature: each PR implements one coherent readiness capability.
- Simplicity: encode explicit contracts, enums, validators, and refusal paths
  before broad abstractions.
- Honest evidence: mocks and stubs are allowed only when clearly marked
  non-official and not benchmark evidence.
- Locked benchmark integrity: do not edit scorer math, validation membership,
  locked labels, cached feature outputs, fingerprints, baseline artifacts,
  result parser behavior, Modal resource/cost envelope, or gate thresholds to
  make implementation easier.
- Falsification integrity: after implementation, gate control construction,
  thresholds, verdict logic, and Discovery Ledger writes are locked benchmark
  infrastructure. Add patch-policy tests that reject future search patches to
  those surfaces.
- Discovery integrity: provisional KEEP is not a discovery. Only a CONFIRMED
  Falsification Gate verdict may enter the Discovery Ledger.
- Secret integrity: OpenAI keys, Modal tokens, GitHub tokens, dashboard
  credentials, and judge/evaluator secrets belong only to the trusted
  harness/orchestrator plane.

### Constraints

- Use the phrase NanoFold-style AlphaFold3-lite for the implementation target.
- Do not claim to train, reproduce, improve, or beat Google DeepMind
  AlphaFold3.
- Do not create fake benchmark results, Arrow files, Modal runs, baseline
  metrics, gate verdicts, discovery records, or validation labels.
- Do not start autonomous research search.
- Do not run upstream NanoFold download scripts or full feature rebuilding.
- Pin official runs to `max_templates=0`.
- Do not provision, search, or mutate a template database.
- The Modal-hosted trusted orchestrator is the event search-loop authority.
- Local orchestration is allowed only for scaffold smoke tests before the Modal
  harness is deployed.
- The agent may not call `modal run` directly, spawn arbitrary Sandboxes,
  change GPU type, change timeouts, change Volumes, raise `max_containers`,
  edit `autoalphafold3/modal_app.py` during search, or author falsification
  controls.
- Trial, sampler, and debug workers must not mount or read locked validation
  labels.
- Scorer-only workers may read locked manifests and labels.
- Trial, sampler, scorer, and debug workers must not receive OpenAI, GitHub,
  Modal, dashboard, or judge/evaluator secrets.
- Workers never write the canonical ledger or Discovery Ledger.
- The Modal-hosted trusted orchestrator writes canonical ledger entries and
  confirmed-only Discovery Ledger entries.
- Gate controls are orchestrator-authored, not agent-authored.
- Gate thresholds and verdict logic are frozen before autonomous search.
- Hidden validation is final-only and out of scope for this goal.

### Feature Stack

Use local Codex worktrees under a sibling directory such as
`../auto-alphafold3-worktrees/`. Open four feature-sized stacked PRs. Each PR
must update `PLAN.md`, `IMPLEMENTATION_CHECKLIST.md`, `EXPERIMENTS.md`, and
`EXPERIMENT_NOTES.md` before it is opened or updated.

1. `feat/modal-prerun-contracts`
   - Base: PR #21 head or merged PR #21 commit.
   - Purpose: answer who is allowed to run things.
   - Scope:
     - reconcile stale local-orchestrator docs and contracts with PR #21
     - Modal-hosted trusted orchestrator/harness contract
     - CPU-only endpoint or queue worker contract
     - harness plane versus execution plane model
     - secret-boundary validators for OpenAI, Modal, GitHub, dashboard, and
       judge/evaluator credentials
     - worker role contracts for trial, sampler, scorer, and debug paths
     - deploy-once/call-many lookup/spawn/poll behavior
     - local scaffold fallback clearly labeled non-event and smoke-only
   - Validation:
     - static contract tests for worker mounts and secret exposure
     - mocked Modal lookup/spawn/poll tests
     - tests that local scaffold mode cannot be reported as event search ready
     - tests that workers cannot receive or serialize harness secrets
     - `git diff --check`
     - targeted tests listed in `PLAN.md`

2. `feat/falsification-and-ledgers`
   - Base: `feat/modal-prerun-contracts`.
   - Purpose: answer what counts as a real discovery.
   - Scope:
     - structured pre-registration schema for causal component, predicted axis,
       direction, and expected delta band
     - FalsificationPlan and FalsificationResult schemas
     - verdict enums and pure verdict math
     - orchestrator-authored knock-out, placebo, predicted-axis, and seed
       control plans
     - bounded control variants, seed caps, per-control timeouts, aggregate
       timeout, and oversized-plan rejection before Modal submission
     - returned exception, timeout, cancel, lookup, spawn, and poll failure
       normalization into structured evidence
     - canonical ledger lifecycle validation
     - confirmed-only Discovery Ledger helpers and provenance requirements
     - patch-policy locks for gate construction, thresholds, verdict logic,
       lifecycle validation, and Discovery Ledger writes
   - Validation:
     - pure local tests for every gate verdict outcome
     - tests that missing scored controls reject or kill deterministically
     - tests that provisional KEEP cannot enter Discovery Ledger
     - tests that non-orchestrator writes are rejected
     - tests that future search patches cannot mutate gate/discovery files
     - tests for bounded fanout, seed caps, and timeout handling
     - `python3 -m pytest -p no:cacheprovider` if shared contracts changed

3. `feat/baseline-scorer-readiness`
   - Base: `feat/falsification-and-ledgers`.
   - Purpose: answer whether the benchmark can run honestly.
   - Scope:
     - baseline lock reader and validator
     - missing or incomplete baseline refusal
     - current-best lookup
     - scorer version, manifest hash, feature hash, label hash, and
       provenance checks
     - asset audit hardening for manifests, split counts, scorer stamp, public
       data versus locked data separation, and Arrow readability where fixtures
       or approved live assets allow
     - scorer-only real artifact scoring contract over prediction artifacts and
       locked labels
     - no-template verification for official runs
     - explicit refusal to create fake baseline metrics or fake Arrow artifacts
   - Validation:
     - baseline-present and baseline-missing tests
     - scorer-only boundary tests
     - asset audit tests
     - locked-label access tests
     - no-template policy tests
     - tests that `runs/baseline/**` cannot be mutated without a separate
       human-approved baseline-lock procedure
     - `python3 -m pytest -p no:cacheprovider` if shared contracts changed

4. `feat/prerun-readiness-report`
   - Base: `feat/baseline-scorer-readiness`.
   - Purpose: answer whether the only thing left is the approved run.
   - Scope:
     - one readiness CLI/report
     - certification statuses: `PASS_LOCAL`, `PASS_MOCKED_MODAL`,
       `PASS_LIVE`, `PENDING_HUMAN_LIVE_ACTION`, and `BLOCKED`
     - readiness checks for Modal orchestrator, worker contracts, secrets,
       assets, baseline lock, scorer boundary, Falsification Gate, known-null
       calibration, known-positive calibration, canonical ledger, Discovery
       Ledger, tests, and skill evals
     - explicit block on autonomous search until all required implementation
       gates pass and remaining live items are named human approvals
     - final runbook text listing the exact command or approval needed for
       controlled live deploy, baseline lock, gate calibration, or search start
   - Validation:
     - readiness report fixture tests
     - tests for missing assets, missing baseline, missing Modal orchestrator,
       missing gate calibration, missing ledgers, and missing live approval
     - tests that incomplete implementation cannot be labeled
       `PENDING_HUMAN_LIVE_ACTION`
     - final local checks:

       ```bash
       python3 -m pytest -p no:cacheprovider
       python3 .claude/skill-evals/run_offline_evals.py
       git diff --check
       ```

Optional controlled-live follow-up:

- `feat/live-readiness-validation`
- Use only after explicit human approval.
- It may run deploy smoke checks, Modal endpoint/queue checks, baseline lock
  procedure, known-null gate calibration, or known-positive gate calibration.
- It must distinguish live readiness from autonomous search.
- It must not write `runs/baseline/**`, locked Volumes, canonical ledgers,
  Discovery Ledger entries, benchmark artifacts, or baseline metrics unless the
  exact baseline-lock procedure is separately approved and documented.

### Per-PR Loop

For each feature PR:

1. Spawn parallel read-only subagents to ground the work in docs, canonical
   spec, tests, and relevant Modal docs.
2. Record the best-practice implementation approach in `PLAN.md` before
   editing.
3. Implement the smallest coherent feature slice.
4. Add focused tests for the changed slice.
5. Run targeted tests and record exact commands and outcomes in
   `EXPERIMENTS.md`.
6. Run full local tests if shared contracts changed:

   ```bash
   python3 -m pytest -p no:cacheprovider
   ```

7. Run skill evals if skills, prompts, or agent behavior changed:

   ```bash
   python3 .claude/skill-evals/run_offline_evals.py
   ```

8. Run `git diff --check`.
9. Review the diff for locked-surface violations.
10. Update `PLAN.md`, `IMPLEMENTATION_CHECKLIST.md`, `EXPERIMENTS.md`, and
    `EXPERIMENT_NOTES.md`.
11. Commit, push, and open or update the stacked PR.

### Minimum Checklist Items

`IMPLEMENTATION_CHECKLIST.md` must include measurable checkboxes for at least:

- Latest `main` and PR #21 fetched.
- Accepted stack tip recorded.
- Branch ownership and clean worktree state verified before any rebase.
- Rebase skipped or stopped when ownership/user-work safety is unclear.
- Required docs and `modal-docs` skill read.
- PR #21 Modal-hosted orchestrator contract understood and cited.
- Local orchestration marked scaffold smoke-only, not event search ready.
- Modal-hosted trusted orchestrator contract implemented.
- CPU-only endpoint or queue worker contract implemented.
- Harness plane and execution plane represented in code and tests.
- Secret-boundary validation implemented.
- OpenAI, GitHub, Modal, dashboard, and judge/evaluator secrets excluded from
  trial/sampler/scorer/debug workers.
- Worker role contracts implemented for trial, sampler, scorer, and debug.
- Mocked Modal lookup/spawn/poll tests pass.
- Direct agent `modal run` and arbitrary Sandbox access remain forbidden.
- Structured pre-registration schema implemented.
- Prediction preflight rejects missing causal component, predicted axis,
  direction, or invalid expected delta band.
- Falsification plan/result schemas implemented.
- All gate verdict outcomes covered by pure local tests.
- Gate verdict math rejects missing scored controls.
- Gate controls are orchestrator-authored.
- Gate thresholds and verdict logic are frozen before autonomous search.
- Gate control construction, thresholds, verdict logic, lifecycle validation,
  and Discovery Ledger writes are added to locked patch-policy coverage after
  implementation.
- Patch-policy tests reject future search edits to implemented gate/discovery
  infrastructure.
- Gate-wave plan enforces max control variants and seed counts before Modal
  submission.
- Gate-wave plan enforces per-control and aggregate timeouts.
- Oversized or unbounded gate-wave plans are rejected locally.
- Returned exceptions from knock-out, placebo, and seed controls normalize into
  structured falsification evidence and ledger-visible verdict reasons.
- Modal lookup/spawn/poll failures normalize to `INFRA_FAIL`.
- Modal gate-wave timeout/cancel/poll failures normalize deterministically.
- Canonical ledger lifecycle validation implemented.
- Workers still cannot write canonical ledger entries.
- Modal-hosted trusted orchestrator is the only canonical ledger writer.
- Discovery Ledger helper implemented.
- Discovery Ledger rejects non-CONFIRMED records.
- Discovery provenance includes git SHA, scorer version, feature/manifest
  hashes, causal component, predicted axis, verdict numbers, and design rule.
- Provisional KEEP cannot enter Discovery Ledger.
- Baseline lock reader implemented.
- Missing or incomplete baseline lock fails readiness honestly.
- No fake baseline metrics created.
- Scorer-only real artifact scoring contract implemented.
- Trial/sampler/debug workers still cannot mount locked labels.
- Scorer-only path remains the locked-label boundary.
- Asset audit validates required files, split counts, provenance, fingerprints,
  scorer stamp, and Arrow readability where fixtures or approved assets allow.
- Public data Volume is checked for absence of locked labels.
- Official runs enforce `max_templates=0`.
- Readiness CLI/report implemented.
- Readiness report distinguishes local/offline, mocked Modal, approved live,
  pending-human-live, and blocked evidence.
- Readiness report blocks autonomous search until known-null and known-positive
  Falsification Gate calibration is complete, or marks the exact
  human-approved live calibration action as pending.
- Readiness report refuses to mark missing implementation as
  `PENDING_HUMAN_LIVE_ACTION`.
- Optional live readiness remains read-only/smoke unless a separate
  human-approved baseline-lock procedure is invoked.
- Optional live readiness cannot write `runs/baseline/**`, locked Volumes,
  canonical ledgers, Discovery Ledger entries, benchmark artifacts, or baseline
  metrics without that separate approval.
- Four stacked feature PRs are open or merged.
- `python3 -m pytest -p no:cacheprovider` passes.
- `python3 .claude/skill-evals/run_offline_evals.py` passes.
- `git diff --check` passes.
- No fake benchmark/data/Modal/gate/discovery artifacts were created.
- Autonomous research search was not started.

### Stop Conditions

Stop and report `BLOCKED` rather than improvising if:

- PR #21 is unavailable and no human has named an accepted replacement base.
- Foundation stack is not available, cannot be used as an accepted base, or
  cannot be safely rebased by this goal run.
- Modal docs cannot be verified for a Modal behavior change.
- Required real Modal credentials/assets are unavailable for an explicitly live
  step.
- Baseline lock data is unavailable and the requested step requires a real
  baseline.
- Known-null and known-positive gate calibration cannot be completed or
  honestly marked pending human-approved live action.
- Repo docs/spec conflict and the canonical spec cannot resolve the drift.
- The only way forward would require fake metrics, fake discoveries, hidden
  validation access, scorer math edits, locked data mutation, Modal cost-cap
  changes, or direct agent control of the gate.

### Done Criteria

- Persistent Modal pre-run readiness goal files exist and are current.
- Four feature-sized stacked PRs are open or merged:
  `feat/modal-prerun-contracts`, `feat/falsification-and-ledgers`,
  `feat/baseline-scorer-readiness`, and `feat/prerun-readiness-report`.
- `IMPLEMENTATION_CHECKLIST.md` is 100% complete except checklist items that
  require real external Modal/baseline execution and are explicitly marked
  `PENDING_HUMAN_LIVE_ACTION` with the exact command or approval needed.
- The readiness report contains no `BLOCKED` items.
- Any remaining `PENDING_HUMAN_LIVE_ACTION` item is a live approval/run item,
  not missing implementation, missing tests, missing schemas, missing gate
  logic, missing baseline readers, missing ledger enforcement, or missing
  readiness reporting.
- Gate calibration is complete for known-null and known-positive controls, or
  readiness explicitly blocks autonomous search pending the exact
  human-approved live calibration action.
- Implemented falsification controls, gate thresholds, verdict logic, lifecycle
  validation, and Discovery Ledger writes are locked against future search
  patches.
- Final local checks pass:

  ```bash
  python3 -m pytest -p no:cacheprovider
  python3 .claude/skill-evals/run_offline_evals.py
  git diff --check
  ```

- The final response reports branch stack, PR URLs, test commands and results,
  readiness status counts, checklist status, live actions still pending, and
  confirms autonomous research search was not started.
