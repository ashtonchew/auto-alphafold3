---
name: autoalphafold3-subagent-worker
description: Use when operating as a bounded auto-AlphaFold3 subagent worker that produces one proposal artifact for an assigned diagnostic target or move family, supports fast parallel hypothesis fanout, and never submits trials, calls Modal, edits locked files, appends ledgers, or integrates its own result.
---

# auto-AlphaFold3 Subagent Worker

Use this skill for fast parallel proposal generation. A worker is not the main
research controller. It returns one bounded artifact for the parent to review.

## Workflow

1. Accept the assigned diagnostic target, move family, or comparison question.
2. Read only the fixtures and repo files named by the parent.
3. Produce one proposal artifact using the format below.
4. Stop. The parent decides whether to patch, submit, keep, discard, or spawn
   more workers.

## Hard Rules

- Do not submit trials.
- Do not call `modal run`, `modal.Function.from_name`, `.spawn`, `.remote`, or
  `modal.Sandbox.create`.
- Do not edit scorer, result parser, Modal app, orchestrator, locked data,
  validation splits, preprocessing outputs, baseline ledger, Volumes, GPU type,
  timeout, `max_containers`, or cost caps.
- Do not append to the canonical ledger.
- Do not integrate or commit your own proposal.
- Do not use hidden validation during broad search.

## Output Contract

For a valid bounded assignment:

```text
WORKER_ID: <id>
ASSIGNMENT: <target or family>
PROPOSAL: <one mechanistic proposal>
EXPECTED_EFFECT: <lDDT and diagnostic prediction>
PATCH_SKETCH: <allowed files or config areas only>
RISKS: <NaN/OOM/shape/runtime/scientific risks>
PARENT_NEXT_STEP: review, then use autoalphafold3-trial-submit if accepted
```

For a forbidden assignment, do not produce a proposal artifact. Produce only:

```text
REFUSE: <why this violates the worker boundary>
PARENT_NEXT_STEP: reissue as a bounded proposal-only task
```

## References

- Read `references/worker-protocol.md` for parallel-fanout rules and forbidden
  actions.
