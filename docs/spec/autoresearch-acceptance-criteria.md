# Autoresearch Acceptance Criteria

This checklist summarizes the implementation gates from
`docs/spec/autoalphafold3-autoresearch-spec.md`.

## Documentation

- PR #50 is merged or an accepted base is documented.
- Goal progress files exist under
  `docs/handoffs/goal/autoalphafold3-autoresearch-implementation/`.
- `docs/runbooks/autoresearch-loop.md` exists.
- `docs/spec/autoresearch-agent-program.md` exists.
- Commands for dry-run, fixture smoke, deterministic ladder, UI render, and
  human-approved live actions are documented.

## Short Training

- Short-training manifests validate schema, paths, budget, and
  `max_templates=0`.
- `TrialRunner.run(...)` remains the official training entrypoint.
- Runner writes only trial-scoped artifacts.
- Local fixture artifacts cannot look like official benchmark evidence.
- Training workers do not mount locked labels.
- Training artifacts do not write `runs/baseline/**`, the canonical ledger, or
  Discovery Ledger records.
- Scorer-only workers remain the locked-label boundary.
- Modal resources are not escalated dynamically with `.with_options(...)`.
- Worker handoffs commit and reload Modal Volume state before cross-container
  reads.

## NanoFold Search Surface

- Loss weights are config-driven.
- Default loss behavior is preserved when new weights are zero.
- Differentiable local C-alpha geometry loss is implemented and tested.
- Patch policy still rejects edits outside the approved search surface.

## Candidate Management

- Candidate artifact envelopes distinguish hypothesis, patch, config, trial,
  preflight, training, scorer, decision, and postmortem evidence.
- Safe git helpers preserve unrelated user changes.
- Locked paths, generated binaries, checkpoints, Arrow files, baseline
  artifacts, ledgers, and Discovery Ledger rows are refused.

## Loop Modes

- Manual mode can consume a prepared candidate.
- Deterministic mode plans the T120-T125 ladder without an LLM.
- `run-short-training --mode modal` refuses absent or wrong approval tokens.
- `autoresearch-loop --mode modal` refuses absent or wrong approval tokens.
- Live-action refusal tests prove no Modal submission, canonical ledger write,
  Discovery Ledger write, or run artifact promotion occurs.
- Repeated candidate failures hit explicit stop rules instead of raising Modal
  resources or brute-forcing retries.
- Stage-one `KEEP` remains provisional.
- Discovery Ledger writes require confirmed Falsification Gate evidence.
- Canonical ledger, results TSV, and run summary writes are serialized through
  one writer boundary.
- LLM mode cannot bypass patch policy or emit multi-move candidates.
- Web search is limited to hypothesis generation, not patch planning.

## UI Evidence

- UI reads `runs/autoresearch/<run_id>/summary.json`.
- Candidate status, loss history, patch summaries, matched-budget deltas,
  global-baseline deltas, and gate state are visible.
- Sample fallback data is visibly labelled.
- Missing metrics are not invented.

## Final Validation

```bash
python3 -m pytest -p no:cacheprovider
git diff --check
```

If skills, prompt docs, or agent behavior changed:

```bash
python3 .claude/skill-evals/run_offline_evals.py
```
