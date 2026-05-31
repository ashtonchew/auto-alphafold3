# auto-AlphaFold3 Autoresearch Implementation Goal-Mode Prompt

This handoff turns `docs/spec/autoalphafold3-autoresearch-spec.md` into an
implementation goal for Codex Goal mode. It is the successor to the foundation,
pre-run readiness, Modal pre-run readiness, sampler loop, and UI evidence work.

The goal is to implement the SimplexFold/Karpathy-style autoresearch loop for a
NanoFold-style AlphaFold3-lite model: the agent may edit approved
model/loss/training code, run fixed-budget train/eval, keep improvements,
revert misses, and record an auditable evidence trail. It must not start
unapproved live autonomous search or weaken the locked benchmark boundary.

## Why This Shape

The prior goal prompts in this repo use a proven pattern:

1. Keep the in-chat `/goal` prompt short.
2. Point Codex at a versioned markdown handoff with the long contract.
3. Require Codex to create persistent progress files under
   `docs/handoffs/goal/<goal-name>/`.
4. Use parallel read-only subagents before each feature PR.
5. Ship feature-sized stacked PRs.
6. Make done criteria measurable with tests, PR state, and checklist state.

Official Codex guidance also supports this structure:

- `/goal` gives Codex a persistent target for larger tasks.
- `/plan` should be used before difficult implementation work starts.
- `/diff` and `/review` should be used to inspect work before committing or
  handing it off.
- Once the workflow stabilizes, the repeated long prompt should become a scoped
  Skill with clear triggers, inputs, and outputs.

This handoff keeps the prompt concrete enough for Goal mode while leaving room
for implementation agents to discover code-level details.

## Short Goal Prompt

Paste this into Codex Goal mode after PR #50 is merged or after branching from
the accepted `autoresearch-spec` tip:

```text
/goal
Implement the SimplexFold/Karpathy-style autoresearch loop for the NanoFold-style AlphaFold3-lite system described in docs/spec/autoalphafold3-autoresearch-spec.md and docs/handoffs/goal/autoalphafold3-autoresearch-implementation-goal-mode-prompt.md. First fetch latest main and PR #50, verify branch ownership and clean worktree status, read AGENTS.md, README.md, docs/framing.md, program.md, autoalphafold3/benchmark_contract.md, autoalphafold3/editable_surface.md, docs/runbooks/, docs/spec/autoalphafold3-autoresearch-spec.md, and invoke the repo-local modal-docs skill before Modal-related work. Create docs/handoffs/goal/autoalphafold3-autoresearch-implementation/{GOAL.md,PLAN.md,IMPLEMENTATION_CHECKLIST.md,EXPERIMENTS.md,EXPERIMENT_NOTES.md}. Use local Codex worktrees and open feature-sized stacked PRs. Before each feature PR, use parallel read-only subagents to ground the implementation in docs/spec/tests/NanoFold internals/Modal contracts, record the best-practice approach in PLAN.md, then implement autonomously. Build the bounded short-training runner, minimal NanoFold loss/config support, candidate artifact manager, safe git keep/revert wrapper, deterministic autoresearch ladder, LLM planner mode, and UI evidence integration. Preserve locked benchmark, scorer, validation-label, Modal-resource, baseline, and ledger boundaries. Do not fabricate benchmark/data artifacts, fake Modal runs, fake baseline metrics, fake gate verdicts, fake discovery records, or unapproved live search results. Done only when all implementation PRs are open or merged, the checklist is complete except exact PENDING_HUMAN_LIVE_ACTION items, local tests pass, fixture-backed short training produces honest artifacts, deterministic ladder works in dry-run/planning mode, and any live Modal/search action remains gated by explicit human approval.
```

## Full Goal Contract

### Objective

Implement the end-to-end autoresearch system described in
`docs/spec/autoalphafold3-autoresearch-spec.md`.

End-to-end means the repository can:

- let an agent author one candidate patch inside the approved NanoFold
  model/loss/training surface
- validate that patch and trial through patch policy and preflight
- run bounded short training
- produce scorer-compatible predictions
- score with the locked scorer boundary
- decide keep, discard, fail, or infra-fail
- preserve or revert code according to that decision
- record candidate artifacts and summaries
- render the trajectory in the evidence UI

End-to-end does not mean starting unapproved live autonomous search. Live Modal
execution, baseline writes, gate calibration, and open-ended search remain
explicit human-approved actions.

### Required First Steps

1. Fetch and inspect current repo state:

   ```bash
   git fetch origin
   git checkout main
   git pull --ff-only
   gh pr view 50 --json number,title,state,headRefName,baseRefName,commits,files
   ```

2. Verify stack safety before branching:

   - PR #50 is merged, or a human has named PR #50 head as the accepted base.
   - Local worktree is clean except known untracked scratch artifacts.
   - Branch ownership is clear.
   - No unrelated user work will be staged, reverted, or force-pushed.
   - If ownership or base selection is unclear, stop for human direction.

3. Read, in order:

   - `AGENTS.md`
   - `README.md`
   - `docs/framing.md`
   - `program.md`
   - `autoalphafold3/benchmark_contract.md`
   - `autoalphafold3/editable_surface.md`
   - `docs/runbooks/`
   - `docs/spec/autoalphafold3-canonical (2).html`
   - `docs/spec/autoalphafold3-autoresearch-spec.md`
   - `.claude/skills/modal-docs/SKILL.md`

4. Invoke the repo-local `modal-docs` skill before implementing or changing
   Modal-related behavior. Use the Volume, deployed class/function lookup,
   app/function, and migration references as needed. Do not guess Modal APIs.

5. Create and maintain:

   - `docs/handoffs/goal/autoalphafold3-autoresearch-implementation/GOAL.md`
   - `docs/handoffs/goal/autoalphafold3-autoresearch-implementation/PLAN.md`
   - `docs/handoffs/goal/autoalphafold3-autoresearch-implementation/IMPLEMENTATION_CHECKLIST.md`
   - `docs/handoffs/goal/autoalphafold3-autoresearch-implementation/EXPERIMENTS.md`
   - `docs/handoffs/goal/autoalphafold3-autoresearch-implementation/EXPERIMENT_NOTES.md`

### Engineering Principles

- Preserve the phrase NanoFold-style AlphaFold3-lite for the implementation
  target.
- Treat `best_val_calpha_lddt` and `calpha_lddt_v1` as locked decision
  contracts.
- Keep behavior local to the owning module.
- Prefer typed payloads, explicit validators, and refusal paths over implicit
  conventions.
- Keep local stubs and fixtures visibly non-official.
- Build deterministic/manual modes before LLM modes.
- Use one coherent feature slice per PR.
- Keep generated run artifacts out of source commits unless a human explicitly
  promotes a small summary artifact.
- Never rewrite user work or unrelated untracked files.

### Constraints

- Do not claim to train, reproduce, improve, or beat Google DeepMind
  AlphaFold3.
- Do not fabricate benchmark results, Arrow files, Modal runs, baseline
  metrics, gate verdicts, discovery records, validation labels, or live search
  results.
- Do not run upstream NanoFold download scripts or full feature rebuilding.
- Pin official runs to `max_templates=0`.
- Do not provision, search, or mutate a template database.
- Do not change scorer math, validation membership, validation labels, cached
  feature outputs, fingerprints, baseline artifacts, result parsing, Modal GPU
  types, timeouts, Volumes, retry policy, warm-pool behavior, `max_containers`,
  or cost caps.
- Trial, sampler, and debug workers must not mount locked labels.
- Scorer-only workers remain the locked-label boundary.
- Training and sampler workers do not write canonical ledger or Discovery
  Ledger records.
- Provisional `KEEP` is not discovery; Discovery Ledger writes require a
  confirmed Falsification Gate verdict.
- Open-ended LLM search remains disabled until deterministic/manual modes work.

## Feature Stack

Use local Codex worktrees under a sibling directory such as
`../auto-alphafold3-worktrees/`. Open stacked PRs with semantic `feat/...`
branches. Adjust branch count only if the implementation becomes materially
simpler, and document the change in `PLAN.md`.

### PR 1: `feat/autoresearch-contract-docs`

Purpose: create implementation runbooks and progress scaffolding.

Scope:

- create the persistent goal progress files
- add `docs/runbooks/autoresearch-loop.md`
- add `docs/spec/autoresearch-agent-program.md`
- optionally add `docs/spec/autoresearch-acceptance-criteria.md` if the main
  spec becomes too large to use as a checklist
- define exact commands for dry-run, fixture smoke, deterministic ladder, and
  human-approved live actions

Validation:

- docs lint/read-through
- `git diff --check`
- no source behavior changes

### PR 2: `feat/short-training-runner`

Purpose: make real bounded training possible beyond the one-batch checkpoint.

Scope:

- add `autoalphafold3/short_training.py`
- add `autoalphafold3/short_training_runner.py`
- add a short-training manifest schema and validator
- use `AutoFoldTrial.max_steps`, `budget`, `config_path`, `seed`, and
  `artifact_dir` where possible
- preserve the one-batch checkpoint path as infrastructure smoke
- write checkpoint, manifest, loss history, training log, stdout/stderr,
  patch diff, and `DONE`
- keep local fixture-backed tests non-official

Validation:

- fixture-backed 2-3 step short-training test
- manifest rejection tests for fake claims, unsafe paths, non-empty output, and
  `max_templates != 0`
- tests proving local scaffold mode cannot fabricate benchmark-ready evidence

### PR 3: `feat/nanofold-geometry-loss`

Purpose: expose the first useful scientific search surface.

Scope:

- make NanoFold loss weights config-driven
- add differentiable local C-alpha pair-distance loss
- add experiment configs under `configs/experiments/**`
- keep defaults equivalent to current behavior when the new loss weight is zero
- avoid large face/tetra architecture changes in this first patch

Validation:

- unit tests for local C-alpha loss shape, finite values, masking, and zero
  weight preserving current loss behavior
- tiny forward/backward fixture test if dependencies allow
- patch-policy coverage for the allowed NanoFold paths

### PR 4: `feat/autoresearch-candidates`

Purpose: create auditable candidate artifacts and safe git behavior.

Scope:

- add candidate ID allocation and run manifest helpers
- create `runs/autoresearch/<run_id>/candidates/<trial_id>/` artifact envelope
- write hypothesis, patch, config, trial, preflight, training manifest, metrics,
  error report, decision, and postmortem pointers
- add a safe git wrapper for candidate snapshot, keep, and revert behavior
- refuse to stage locked/generated artifacts
- refuse to delete untracked user files

Validation:

- tmp-repo tests for keep/revert behavior
- tests that unrelated user changes are preserved
- tests that generated binaries and locked paths are not staged

### PR 5: `feat/deterministic-autoresearch-ladder`

Purpose: prove the loop without an LLM.

Scope:

- add manual planner mode
- add deterministic planner mode for the first short-training ladder
- generate matched baseline and geometry candidate trials
- run dry-run/planning mode end to end
- wire stage-one decisions without Discovery Ledger writes
- keep live Modal execution behind explicit approval

Validation:

- deterministic ladder planning test
- mocked worker/scorer decision test
- no live Modal calls in tests
- `python3 -m pytest -p no:cacheprovider`

### PR 6: `feat/autoresearch-llm-planner`

Purpose: let the research agent author candidate patches after the deterministic
path works.

Scope:

- add LLM planner mode that reads `docs/spec/autoresearch-agent-program.md`
- use the existing LLM phase policy instead of SDK defaults
- require exactly one hypothesis, one move family, one diagnostic target, and
  one candidate patch per iteration
- enforce patch policy before any run
- keep web search limited to hypothesis generation, not patch planning
- record planner inputs/outputs as candidate artifacts

Validation:

- schema/structured-output tests or fakes for planner output
- tests for invalid multiple-move proposals
- tests for patch-policy rejection before execution
- no OpenAI secrets in execution workers

### PR 7: `feat/autoresearch-ui-evidence`

Purpose: make results inspectable.

Scope:

- extend UI state loading for `runs/autoresearch/<run_id>/summary.json`
- show candidate trajectory, kept/discarded/failed/infra-failed status,
  matched-budget delta, global-baseline delta, loss history, patch summary, and
  gate status
- keep sample fallback visibly labelled
- never invent metrics for missing run artifacts

Validation:

- UI build tests
- sample fallback tests
- real-artifact parsing tests
- `python -m autoalphafold3.ui.build --sample --out <tmp>`

## Per-PR Loop

For each PR:

1. Use `/plan` or write a short plan before editing.
2. Spawn parallel read-only subagents for independent context:
   - docs/spec/contracts
   - relevant code/tests
   - NanoFold internals or Modal docs when relevant
3. Record findings and approach in `PLAN.md`.
4. Implement the smallest coherent slice.
5. Run targeted tests.
6. Run full tests when shared contracts or critical paths changed:

   ```bash
   python3 -m pytest -p no:cacheprovider
   ```

7. Run skill evals if skills, prompt docs, or agent behavior changed:

   ```bash
   python3 .claude/skill-evals/run_offline_evals.py
   ```

8. Use `/diff` or `git diff` to review changes.
9. Use `/review` or a review pass before finalizing risky code changes.
10. Update `IMPLEMENTATION_CHECKLIST.md`, `EXPERIMENTS.md`, and
    `EXPERIMENT_NOTES.md`.
11. Commit, push, and open or update the stacked PR.

## Minimum Checklist Items

`IMPLEMENTATION_CHECKLIST.md` must include measurable checkboxes for at least:

- PR #50 base state verified.
- Required docs read in order.
- `modal-docs` skill invoked before Modal work.
- Goal progress files created.
- Autoresearch runbook created.
- Agent program prompt created.
- Short-training manifest schema implemented.
- Short-training runner writes trial-scoped artifacts only.
- Short-training runner rejects fake training claims.
- Short-training runner rejects `max_templates != 0`.
- Short-training runner rejects unsafe feature paths.
- Short-training runner rejects non-empty output directories.
- Fixture-backed 2-3 step short-training test passes.
- Local scaffold mode cannot stamp official benchmark evidence.
- Config-driven NanoFold loss weights implemented.
- Defaults preserve current loss behavior when new weights are zero.
- Differentiable local C-alpha geometry loss implemented.
- Local C-alpha geometry loss has finite-value and masking tests.
- Candidate artifact envelope implemented.
- Candidate patch snapshots implemented.
- Safe git keep/revert wrapper implemented.
- Safe git wrapper preserves unrelated user changes.
- Safe git wrapper refuses locked/generated artifacts.
- Manual planner mode implemented.
- Deterministic ladder planning implemented.
- Deterministic ladder dry-run/planning mode passes.
- Matched-budget baseline comparison implemented.
- Global-baseline provisional KEEP comparison preserved.
- Provisional KEEP does not write Discovery Ledger.
- LLM planner mode implemented only after deterministic mode passes.
- LLM planner emits exactly one hypothesis, move family, diagnostic target, and
  candidate patch.
- LLM planner cannot bypass patch policy.
- Web search is allowed only for hypothesis generation, not patch planning.
- Execution workers receive no OpenAI/GitHub/Modal/dashboard/judge secrets.
- UI reads autoresearch summary artifacts.
- UI labels sample fallback data honestly.
- Full tests pass or failures are recorded with exact blockers.
- Any live Modal/search step is marked `PENDING_HUMAN_LIVE_ACTION` with exact
  command and approval token.

## Done Criteria

The goal is complete when:

- all planned implementation PRs are open or merged
- the checklist is complete except exact `PENDING_HUMAN_LIVE_ACTION` items
- docs, runbooks, and agent program prompt exist
- fixture-backed short training writes honest artifacts
- deterministic ladder works in dry-run/planning mode
- LLM planner mode is implemented or explicitly deferred behind deterministic
  completion with documented reason
- UI can render autoresearch artifacts or labelled samples
- no tests create fake benchmark, Modal, baseline, gate, discovery, Arrow, or
  validation-label evidence
- final local validation passes:

  ```bash
  python3 -m pytest -p no:cacheprovider
  git diff --check
  ```

If skill docs, prompt docs, or agent behavior changed, also run:

```bash
python3 .claude/skill-evals/run_offline_evals.py
```

## Handoff Notes For The Agent

The strongest implementation path is not to jump directly to open-ended LLM
search. Build the harness first, then prove deterministic/manual loops, then
turn on LLM-authored candidate patches.

The first scientific patch should be modest: config-driven loss weights plus a
differentiable local C-alpha pair-distance loss. The large SimplexFold-style
face/tetra architecture family is future work after the loop is proven.

If this prompt becomes the standard way to run the project, convert it into a
repo skill. Official Codex guidance recommends turning repeatable long prompts
into scoped skills with clear inputs, outputs, and trigger phrases once the
workflow stabilizes.
