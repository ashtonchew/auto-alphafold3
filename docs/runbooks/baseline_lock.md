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

2. Run the NanoFold default/tiny baseline through the trusted Modal
   orchestrator and scorer-only worker.
3. Confirm the scored baseline artifact directory contains real
   `metrics.json` and `error_report.json`.
4. Freeze that evidence once:

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
