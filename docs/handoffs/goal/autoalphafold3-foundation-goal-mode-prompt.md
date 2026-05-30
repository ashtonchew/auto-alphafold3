# AutoAlphaFold3 Foundation Goal-Mode Prompt

This handoff keeps the in-chat `/goal` prompt short and puts the long-running
implementation contract in the repository where Codex can reread it after
compaction, pauses, or worktree switches.

## Why This Shape

OpenAI Codex guidance says Goal mode is for longer tasks where the goal text is
both the starting prompt and the completion criteria. Good goals need a
specific outcome, measurable target, or test criteria:

- Goal mode: https://developers.openai.com/codex/prompting#goal-mode
- Context/constraints/done-when prompting: https://developers.openai.com/codex/learn/best-practices#strong-first-use-context-and-prompts
- Plan-first workflows and PLANS.md-style templates: https://developers.openai.com/codex/learn/best-practices#plan-first-for-difficult-tasks
- Long-running agentic task guidance: https://developers.openai.com/api/docs/guides/prompt-engineering#coding

The practical pattern for this repo is:

1. Keep the composer prompt under the Goal-mode input limit.
2. Point the goal at this versioned markdown file.
3. Make Codex create and maintain progress files under
   `docs/handoffs/goal/autoalphafold3-foundation/`.
4. Define completion as checklist, PR, and test outcomes rather than a vague
   "finish implementation" instruction.

## Short Goal Prompt

Paste this into Codex Goal mode after PR #6 is merged and local `main` is
current:

```text
/goal
Implement the NanoFold-style AlphaFold3-lite foundation described in docs/handoffs/goal/autoalphafold3-foundation-goal-mode-prompt.md. First pull latest main, verify PR #6 is merged, read the required repo docs, and create docs/handoffs/goal/autoalphafold3-foundation/{GOAL.md,PLAN.md,IMPLEMENTATION_CHECKLIST.md,EXPERIMENTS.md,EXPERIMENT_NOTES.md}. Use local Codex worktrees and ship feature-sized stacked PRs with semantic feat/... branch names and conventional commits. Before each feature PR, use parallel read-only subagents to ground the implementation in the docs/spec/tests, explain the best-practice implementation approach, then implement autonomously. Preserve locality of behavior, modularity by feature, simplicity, and exact spec compliance. Do not create fake benchmark/data artifacts or start autonomous research trials. Done only when the checklist is 100% complete, all PRs are open, and final tests pass: python3 -m pytest -p no:cacheprovider and python3 .claude/skill-evals/run_offline_evals.py.
```

## Full Goal Contract

### Objective

Implement the complete foundation needed before autonomous search can begin.
This is infrastructure implementation only. Do not start the hackathon search
loop or submit research trials.

### Successor Goal

After this foundation goal is complete, use
`docs/handoffs/goal/autoalphafold3-prerun-readiness-goal-mode-prompt.md` for
the next implementation wave. That successor goal covers canonical spec
Sections 2-7 up to, but not including, autonomous research search: structured
pre-registration, Falsification Gate contracts, baseline readiness, two-stage
orchestration, confirmed-only Discovery Ledger, fakeable Modal gate fanout, and
pre-run readiness reporting.

### Required First Steps

1. Pull latest `main`:

   ```bash
   git fetch origin
   git checkout main
   git pull --ff-only
   ```

2. Verify PR #6 is merged:

   - No open PR #6 remains.
   - `git log` includes the PR #6 merge.
   - `docs/spec/autoalphafold3-canonical (2).html` contains Section 7.5.

3. Read, in order:

   - `README.md`
   - `docs/framing.md`
   - `program.md`
   - `autoalphafold3/benchmark_contract.md`
   - `autoalphafold3/editable_surface.md`
   - `docs/runbooks/`
   - `docs/spec/autoalphafold3-canonical (2).html`
   - `.claude/skills/*/SKILL.md`

4. Invoke the `modal-docs` skill before implementing Modal code. If that skill
   is unavailable, use official Modal docs only, cite the fallback in `PLAN.md`,
   and do not guess Modal APIs.

5. Create these persistent files:

   - `docs/handoffs/goal/autoalphafold3-foundation/GOAL.md`
   - `docs/handoffs/goal/autoalphafold3-foundation/PLAN.md`
   - `docs/handoffs/goal/autoalphafold3-foundation/IMPLEMENTATION_CHECKLIST.md`
   - `docs/handoffs/goal/autoalphafold3-foundation/EXPERIMENTS.md`
   - `docs/handoffs/goal/autoalphafold3-foundation/EXPERIMENT_NOTES.md`

### Engineering Principles

- Locality of behavior: put behavior in the module that owns it.
- Modularity by feature: each PR should implement one coherent capability.
- Simplicity: prefer explicit typed functions and local helpers over broad
  abstraction layers.
- No spec drift: do not deviate from `AGENTS.md`, `program.md`,
  `benchmark_contract.md`, `editable_surface.md`, runbooks, or the canonical
  spec. If they conflict, stop and document the conflict.
- Honest evidence: local mocks and stubs must remain clearly non-official.
- Locked benchmark integrity: never modify scorer math, validation splits,
  locked labels, cached feature outputs, fingerprints, or baseline metrics to
  make implementation easier.

### Section 7.5 Modal Defaults

PR #6 makes the Section 7.5 autoscaler settings canonical implementation
defaults. Apply the exact values in the spec when implementing the Modal scorer
and TrialRunner classes, including:

- `min_containers=1` for the scorer class where specified.
- `min_containers=1` for the TrialRunner class where specified.
- The listed `scaledown_window` values.
- `@modal.concurrent` on the scorer where specified.

Do not extrapolate beyond the spec. Raising `min_containers`, applying warm
pools to additional functions, changing GPU type, changing `max_containers`,
changing timeouts, changing Volumes, or changing cost caps remains out of scope
unless a human explicitly approves it and tests cover the change.

### Constraints

- Use the phrase NanoFold-style AlphaFold3-lite for the implementation target.
- Do not claim to train, reproduce, improve, or beat Google DeepMind
  AlphaFold3.
- Do not fabricate benchmark results, fake Arrow files, fake Modal runs, fake
  baseline metrics, or fake validation labels.
- Do not run full PDB/mmCIF/MSA feature rebuilding during the event.
- Pin official runs to `max_templates=0`.
- Do not provision, search, or mutate a template database.
- Trial, sampler, and debug workers must not mount or read locked validation
  labels.
- Scorer-only workers may read locked manifests and labels.
- The local orchestrator is the only canonical ledger writer.

### Worktree And PR Plan

Use local Codex worktrees under a sibling directory such as
`../auto-alphafold3-worktrees/`. Ship feature-sized stacked PRs. Four PRs should
be enough; use a fifth only if a slice becomes too large to review safely.

1. `feat/foundation-contracts`
   - Base: `main`
   - Scope: goal pack, implementation checklist, official feature/data
     contract decision, PR #6 spec alignment, asset audit hardening,
     provenance/fingerprint/scorer-stamp validation, no-template/schema
     readiness checks where fixtures allow.

2. `feat/nanofold-adapters`
   - Base: `feat/foundation-contracts`
   - Scope: repo-root-aware NanoFold adapters, no personal absolute paths,
     explicit manifest-driven dataset boundaries, no random split leakage,
     train-only label access, public-validation inference without label access,
     `max_templates=0`, empty-template verification.

3. `feat/trial-runtime`
   - Base: `feat/nanofold-adapters`
   - Scope: fixed-budget runner boundary, prediction artifact writer,
     checkpoint/log/stdout/stderr/patch manifest behavior, scorer-compatible
     prediction artifacts, local/mock non-official execution path.

4. `feat/modal-scorer-orchestrator`
   - Base: `feat/trial-runtime`
   - Scope: real Modal app/classes/functions, fixed image/Volume wiring,
     Section 7.5 safe performance defaults, trial/scorer worker separation,
     scorer-only locked-label scoring, Fold Cartographer canonical diagnostics,
     Modal failure normalization, lifecycle validation, ledger hardening, CLI
     strict preflight/git-diff enforcement.

Optional fifth PR:

- `feat/readiness-baseline-gates`
- Use only if final readiness, baseline-freezing procedure, trial templates, and
  end-to-end mocked validation need their own reviewable feature slice.

### Branch, Commit, And PR Rules

- Branches must be semantic `feat/...` names.
- Commits must be conventional, for example:
  - `feat(contracts): harden asset readiness checks`
  - `feat(nanofold): add manifest-driven adapter boundary`
  - `feat(runtime): write scorer-compatible predictions`
  - `feat(modal): wire scorer-only control plane`
  - `test(orchestrator): cover lifecycle transition rejection`
  - `docs(goal): add foundation goal-mode handoff`
- Set each stacked PR base to the previous feature branch, not `main`, except
  the first PR.
- Open draft PRs once each feature slice passes relevant tests.
- PR bodies should follow the recent repo format:

  ```markdown
  ## Summary

  - ...

  ## Validation

  - ...

  ## Notes

  - ...
  ```

### Per-PR Loop

For each feature PR:

1. Spawn parallel read-only subagents to ground the work in docs, spec, tests,
   and relevant upstream docs.
2. Record the best-practice implementation approach in `PLAN.md` before
   editing.
3. Implement with locality of behavior.
4. Run targeted tests.
5. Run full local tests if shared contracts changed:

   ```bash
   python3 -m pytest -p no:cacheprovider
   ```

6. Run skill evals if skills/docs/agent behavior changed:

   ```bash
   python3 .claude/skill-evals/run_offline_evals.py
   ```

7. Review diff for locked-surface violations.
8. Update `PLAN.md`, `IMPLEMENTATION_CHECKLIST.md`, `EXPERIMENTS.md`, and
   `EXPERIMENT_NOTES.md`.
9. Commit, push, and open or update the stacked PR.

### Minimum Checklist Items

`IMPLEMENTATION_CHECKLIST.md` must include measurable checkboxes for at least:

- PR #6 merged into `main`.
- Current `main` pulled.
- Official feature schema decision documented.
- No-template policy documented.
- Modal docs consulted and cited.
- Asset audit validates required files, provenance, fingerprints, scorer stamp,
  manifest counts, and Arrow readability where fixtures allow.
- NanoFold path resolution is repo-root-aware.
- No hard-coded personal absolute paths remain in NanoFold scripts.
- Official training path uses explicit manifests.
- Official validation inference path cannot read validation labels.
- Random split behavior is rejected for official runs.
- `max_templates=0` is enforced.
- Empty template placeholders are verified.
- Runner writes only trial-scoped artifacts.
- Runner creates canonical prediction artifacts.
- Runner does not stamp official benchmark success.
- Modal app uses real Modal definitions.
- Trial workers cannot mount locked labels.
- Scorer workers can mount locked labels.
- Scorer-only path can score real prediction artifacts.
- Fold Cartographer emits canonical targets.
- Orchestrator normalizes Modal errors to `INFRA_FAIL`.
- Ledger validates lifecycle transitions.
- Ledger writes remain append-only.
- CLI exposes strict preflight/git-diff enforcement.
- Targeted tests cover each implemented boundary.
- `python3 -m pytest -p no:cacheprovider` passes.
- `python3 .claude/skill-evals/run_offline_evals.py` passes.
- Four stacked PRs are open, or a fifth is justified in `PLAN.md`.
- No fake benchmark/data artifacts were created.
- Autonomous search trials were not started.

### Stop Conditions

Stop and report `BLOCKED` rather than improvising if:

- PR #6 is not merged.
- Modal docs cannot be accessed and Modal code cannot be verified safely.
- Live Modal credentials are required for a step and unavailable.
- Required real benchmark data is unavailable.
- Repo docs/spec conflict.
- The only way forward would require fake data, fake metrics, hidden validation
  access, scorer math edits, or locked benchmark mutation.

### Final Response Requirements

When the goal completes, report:

- Branch stack.
- PR URLs.
- Test commands and results.
- Checklist completion status.
- Live Modal/data validation that still needs human action.
- Confirmation that autonomous search trials were not started.
