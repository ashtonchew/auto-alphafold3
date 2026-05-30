# Baseline Lock Runbook

The baseline lock is a human-approved freeze step. It records real scored
baseline evidence; it does not run NanoFold, call Modal, synthesize metrics, or
start autonomous search.

## Intended Source

Use the Modal-hosted baseline path from the canonical spec:

1. Verify Modal assets:

   ```bash
   python3 -m autoalphafold3.agent audit-modal-assets --search-ready
   ```

2. Plan the baseline source artifact run locally:

   ```bash
   python3 -m autoalphafold3.agent run-baseline --mode dry-run
   ```

   This writes nothing and is safe to use before deployment.

3. Run the NanoFold default/tiny baseline through the deployed Modal trial
   worker and scorer-only worker after explicit approval:

   ```bash
   python3 -m autoalphafold3.agent run-baseline \
     --mode modal \
     --approve I_APPROVE_BASELINE_RUN
   ```

   The command writes only trial-scoped source artifacts under
   `runs/trials/T000/`. It refuses to write `runs/baseline/**` and refuses
   scorer payloads that are not `official_benchmark_result=true`,
   `local_only=false`, `status=SCORED`, scorer-only, and `max_templates=0`.

4. Confirm the scored baseline artifact directory contains real
   `metrics.json` and `error_report.json`.
5. Freeze that evidence once:

   ```bash
   python3 -m autoalphafold3.agent lock-baseline \
     --source-dir runs/trials/T000 \
     --feature-fingerprints path/to/feature_fingerprints.json \
     --baseline-dir runs/baseline \
     --approve I_APPROVE_BASELINE_LOCK
   ```

Use `--dry-run` first to validate the source payloads without writing
`runs/baseline`.

## Guardrails

- The command refuses to run without the exact approval token.
- The source metrics must be `official_benchmark_result=true`.
- The source metrics must not be `local_only=true`.
- The error report must prove `scorer_only=true`.
- Existing `runs/baseline` contents are never overwritten.
- The command writes only `metrics.json`, `error_report.json`, and
  `feature_fingerprints.json` under the requested baseline directory.
- Readiness is audited immediately after freezing.

## Runtime Expectation

The baseline run should be much faster than the full autonomous search because
it is a single tiny cached-feature run plus one scorer pass, not a 20-candidate
sampler burst or a Falsification Gate wave. It is still a real Modal/NanoFold
action, so wall time depends on deployment state, image/container warmup, GPU
queueing, and the fixed trial timeout. Expect minutes rather than an hour when
containers are warm; cold starts can dominate the first run.
