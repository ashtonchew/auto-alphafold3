# Worker Protocol

Subagent workers reduce wall-clock time by exploring independent hypothesis
branches in parallel. They do not own the research loop.

## Good Assignments

- "For `long_range_topology_weak`, propose one Pairformer change."
- "For `stability_compute`, inspect the fixture metrics and suggest one
  memory/runtime intervention."
- "Compare two candidate move families and recommend one for the parent."

## Bad Assignments

Reject assignments that ask the worker to:

- submit a trial.
- run or spawn Modal jobs.
- open a Sandbox.
- edit locked benchmark or control-plane files.
- change GPU, timeout, Volume, `max_containers`, or cost caps.
- use hidden validation.
- append or rewrite the ledger.

## Parallel Discipline

Workers should have disjoint assignments. Return short proposal artifacts so
the parent can compare them quickly. If evidence is insufficient, say what is
missing rather than inventing metrics.

## Model Policy

Use small fast models for routine worker evals or proposal fanout. Escalate to a
stronger model only when an output is ambiguous, safety-critical, or conflicts
with the canonical benchmark contract.
