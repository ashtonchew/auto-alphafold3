---
name: autoalphafold3-trial-submit
description: Use when validating and submitting exactly one auto-AlphaFold3 AutoFoldTrial JSON through the approved local orchestrator command, rejecting direct Modal calls, locked benchmark paths, hidden-validation search, missing hypotheses, and multi-trial submissions.
---

# auto-AlphaFold3 Trial Submit

Use this skill after a research hypothesis exists and before any experiment is
run. This skill is a narrow gate: validate one `AutoFoldTrial` JSON and submit
only through the local orchestrator.

## Workflow

1. Read `references/trial-schema.md`.
2. Confirm there is exactly one trial.
3. Confirm the trial has one hypothesis, one diagnostic target, one move family,
   allowed paths, a valid budget, and safe sampler semantics.
4. If valid, return the command:
   `python -m autoalphafold3.agent submit trials/<id>.json`
5. If invalid, refuse with reasons and do not provide any execution command.

## Hard Rules

- Do not call `modal run`, `modal.Function.from_name`, `.spawn`, `.remote`, or
  `modal.Sandbox.create`.
- Do not use hidden validation during broad search.
- Do not accept edits to scorer, result parser, locked data, baseline ledger,
  validation splits, preprocessing outputs, Modal app, orchestrator, Volumes,
  GPU type, timeout, `max_containers`, or cost caps.
- Sampler-only trials must use a frozen checkpoint and label-free selection.
- The orchestrator is the only component allowed to spawn Modal work or append
  the canonical ledger.

## Output Contract

For valid input:

```text
VALID: true
TRIAL_ID: <id>
COMMAND: python -m autoalphafold3.agent submit trials/<id>.json
```

For invalid input:

```text
VALID: false
REASONS:
- <reason>
COMMAND: <none>
```

## References

- Read `references/trial-schema.md` for required fields, allowed values, and
  rejection checks.
