# Experiment Notes

No benchmark claims, Modal runs, Arrow artifacts, baseline metrics, validation
labels, or autonomous trials were created during foundation setup.

## Notes

- PR 1 is contracts-only. It hardens readiness checks and records the work plan.
- Later Modal implementation must use docs-backed APIs and Section 7.5 defaults
  exactly as written.
- PR 2 is adapter-only. It does not run preprocessing or create feature,
  prediction, benchmark, baseline, or Modal artifacts.
- PR 3 is runtime-artifact-only. It adds scorer-compatible prediction artifact
  helpers for caller-supplied coordinates, keeps local stubs non-official, and
  does not create checkpoints, benchmark metrics, Modal runs, or NanoFold
  training outputs.
- PR 4 wires Modal/orchestrator contracts and static definitions. It does not
  deploy Modal, submit trials, score official benchmark results, or mutate
  locked assets.
