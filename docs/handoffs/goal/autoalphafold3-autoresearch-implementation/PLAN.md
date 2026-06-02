# Autoresearch Implementation Plan

## Grounding

- Read required project docs: `AGENTS.md`, `README.md`, `docs/framing.md`,
  `program.md`, `autoalphafold3/benchmark_contract.md`,
  `autoalphafold3/editable_surface.md`, `docs/runbooks/`,
  `docs/spec/autoalphafold3-canonical (2).html`,
  `docs/spec/autoalphafold3-autoresearch-spec.md`, and the goal handoff.
- Invoked the repo-local `modal-docs` skill by reading
  `.claude/skills/modal-docs/SKILL.md`. Modal implementation work must consult
  the specific Volume, app/function, deployed-object, or migration references
  before code changes.
- PR #50 is merged into `origin/main`; implementation branches are based from
  `cf5844e43fb80562eebf31c741b2bd0f119b90f0`.

## PR 1: Contract Docs

Approach:

- Add operational runbook and concise agent program docs only.
- Create persistent goal progress files for the full implementation wave.
- Encode exact dry-run, fixture, deterministic ladder, and human-approved live
  command boundaries.
- Avoid source behavior changes and generated run artifacts.

Read-only grounding agents:

- `Carson`: docs/spec/contracts review. Findings: PR 1 should create the
  progress files, runbook, agent program, and acceptance checklist; encode
  locked metric/scorer, `max_templates=0`, no fake evidence, no locked surface
  edits, no locked-label mounts, provisional KEEP, and exact
  `PENDING_HUMAN_LIVE_ACTION` commands. Avoid restating stale scaffold wording
  from `benchmark_contract.md`; separate sampler-only evidence from the new
  short-training loop.
- `Hooke`: existing code/tests/CLI/UI hook review. Findings: keep existing
  `python -m autoalphafold3.agent <subcommand>` and
  `python -m autoalphafold3.ui.build` patterns; preserve artifact names such
  as `artifact_manifest.json`, `training_log.json`, `patch.diff`, `DONE`,
  `checkpoint_manifest.json`, and `sampler_manifest.json`; document exact
  approval-token refusal style; do not alter executable behavior in PR 1.
- `Huygens`: NanoFold internals and Modal boundary review. Findings: current
  NanoFold loss is hard-coded as `4 * diffusion_loss + 0.03 * dist_loss`;
  training is epoch-loop oriented and short-training docs must frame bounded
  runs as smoke/checkpoint evidence, not benchmark evidence; all official runs
  remain no-template; Modal writers must commit Volume writes and readers must
  reload to observe fresh artifacts; only scorer workers may mount locked
  labels.

Validation:

- `git diff --check`
- docs read-through

## Planned PR 2: Short-Training Runner

Approach:

- Add `autoalphafold3/short_training.py` as the pure training and manifest
  layer.
- Add `autoalphafold3/short_training_runner.py` as the dry-run/local-fixture/
  Modal guard layer.
- Keep the existing one-batch checkpoint path separate as infrastructure smoke.
- Keep `run_fixed_budget_trial()` stub behavior unless the payload explicitly
  requests `runner_mode=short_training`.
- Use direct `Trainer.training_loop(...)` steps instead of upstream
  epoch-oriented `Trainer.fit(...)`.

Read-only grounding agents:

- `Raman`: spec/contracts review. Findings: write only
  `runs/trials/<trial_id>/` artifacts, stamp fixture evidence as non-official,
  reject unsafe paths, non-empty outputs, fake claims, `max_templates != 0`,
  and Modal execution without `I_APPROVE_SHORT_TRAINING_TRIAL`.
- `Newton`: existing code/test pattern review. Findings: mirror
  checkpoint-runner style, structured JSON CLI failures, atomic JSON writes,
  fake Modal client tests, and side-effect assertions.
- `Leibniz`: NanoFold fixture feasibility review. Findings: local fixture
  supports a 2-3 step real training smoke without downloads; PR 2 should not
  change NanoFold loss yet.

Validation:

- `python3 -m pytest -p no:cacheprovider tests/test_short_training.py tests/test_checkpoint_training.py tests/test_runner_and_locked_scorer.py -q`
  passed: 41 passed.

## Planned PR 3: NanoFold Geometry Loss

Make loss weights config-driven and add a differentiable local C-alpha
pair-distance loss inside the approved NanoFold training surface. Defaults must
preserve current behavior when the new geometry weight is zero.

## Planned PR 4: Candidate Artifacts And Git Safety

Create the `runs/autoresearch/<run_id>/candidates/<trial_id>/` artifact
envelope and safe git helpers. The wrapper must stage only allowed candidate
paths, preserve unrelated user changes, refuse locked/generated artifacts, and
refuse to delete untracked files.

## Planned PR 5: Deterministic Ladder

Implement manual and deterministic planner modes before LLM mode. The first
ladder plans T120-T125 in dry-run/planning mode and keeps Discovery Ledger
writes disabled for stage-one decisions.

## Planned PR 6: LLM Planner

Use existing LLM phase policy. Hypothesis generation may use web search;
patch planning must be repo-local. The planner must emit exactly one
hypothesis, one diagnostic target, one move family, and one candidate patch.

## Planned PR 7: UI Evidence

Render autoresearch summaries, candidate status, patch summary, loss history,
matched-budget deltas, global-baseline deltas, and gate state. Sample fallback
data must remain visibly labelled.
