# Sampled Live Skill Eval Plan

Run this only after `run_offline_evals.py` passes. Live evals use synthetic
fixtures and small fast models first; they never call Modal, GPUs, hidden
validation, or real trial submission.

## Default Execution Policy

- model: `gpt-5.3-codex-spark` or the fastest available mini/spark-class model.
- fanout: one independent eval per skill in parallel.
- timeout: 60 to 90 seconds per worker.
- escalation: use a stronger model only for ambiguous or failed outputs.

## Worker Prompts

For each skill, give the worker only:

- the skill directory path.
- the rendered prompt from `build_live_eval_prompts.py`, which inlines
  `SKILL.md` and directly linked references.
- one eval case from `evals/evals.json`.
- instruction to produce a transcript and no file edits.

The parent grades transcripts with the same assertions used by the offline
runner and stores results under `.claude/skill-eval-workspace/iteration-N/`.

Do not rely on skill attachment resolution alone. If a worker returns a generic
answer instead of the skill output contract, rerun that case with the rendered
prompt before treating it as a skill failure.
