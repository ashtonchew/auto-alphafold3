---
name: autoalphafold3-researcher
description: Use when acting as the auto-AlphaFold3 autoresearch agent: read baseline metrics and Fold Cartographer diagnostics, choose one diagnostic target and one move family, propose or patch only the allowed AlphaFold3-lite model/training/sampler surface, and hand one trial to the orchestrator without touching Modal infrastructure or locked benchmark files.
---

# auto-AlphaFold3 Researcher

Use this skill for the main AlphaFold3-lite research loop. Pair it with
`fold-cartographer` for diagnostic routing and `autoalphafold3-trial-submit`
for trial JSON validation and submission.

## Workflow

1. Read current best metrics, latest `error_report.json`, last 20 ledger rows,
   and the allowed edit surface.
2. Choose exactly one diagnostic target and exactly one move family.
3. Write the hypothesis and prediction before changing code.
4. Patch only model, training, sampler, config, loss, scheduler, curriculum, or
   memory/runtime code listed as editable.
5. Create one `AutoFoldTrial` JSON and hand it to `autoalphafold3-trial-submit`.
6. After metrics return, classify the run as `KEEP`, `DISCARD`, `FAIL`, or
   `INFRA_FAIL`, then write a postmortem.

## Hard Rules

- The primary metric is `best_val_calpha_lddt`.
- Do not optimize diagnostics directly; diagnostics route hypotheses.
- Do not call `modal run`, `modal.Function.from_name`, `.spawn`, `.remote`, or
  `modal.Sandbox.create`.
- Do not edit `autoalphafold3/modal_app.py`,
  `autoalphafold3/orchestrator.py`, scorer code, result parser code, locked
  data, validation split definitions, preprocessing outputs, or baseline ledger.
- Do not change GPU type, timeout, `max_containers`, Volumes, cost caps, or
  hidden validation behavior.
- Do not run full PDB/mmCIF/MSA preprocessing during the event.
- If asked to repair infrastructure or scorer/split issues, stop and report
  that human infrastructure repair mode is required.

## Output Contract

For each proposal, produce:

```text
TRIAL: <id>
DIAGNOSTIC_TARGET: <one target>
MOVE_FAMILY: <one family>
HYPOTHESIS: <mechanistic sentence>
INTERVENTION: <allowed change>
PREDICTION: <expected lDDT and diagnostic effect>
TRIAL_JSON_PATH: trials/<id>.json
SUBMIT_WITH: python -m autoalphafold3.agent submit trials/<id>.json
```

For an invalid or locked-surface request, do not fill proposal fields with
`N/A`. Produce only:

```text
REFUSE: <why this violates the research contract>
REQUIRED_MODE: Human infrastructure repair mode
NEXT_SAFE_ACTION: <note or allowed diagnostic/research alternative>
```

## References

- Read `references/research-contract.md` for targets, move families, edit
  boundaries, decision rules, and stuck behavior.
