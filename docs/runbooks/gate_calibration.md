# Falsification Gate Calibration Runbook

The Falsification Gate calibration file is a human-approved readiness artifact.
It proves the gate can kill a known-null control and confirm a known-positive
control before autonomous search starts.

## Plan Without Writing

```bash
python3 -m autoalphafold3.agent calibrate-gate --mode dry-run
```

This prints the required evidence contract and writes nothing.

## Produce Evidence With Modal Controls

After the approved baseline is locked, produce known-null and known-positive
control evidence through the deployed Modal gate-control runner:

```bash
python3 -m autoalphafold3.agent run-gate-calibration --mode dry-run
python3 -m autoalphafold3.agent run-gate-calibration \
  --mode modal \
  --approve I_APPROVE_GATE_CALIBRATION_RUN
```

The live command writes only:

- `runs/gate_calibration/known_null.json`
- `runs/gate_calibration/known_positive.json`

It is calibration-only evidence. It does not start autonomous search, write the
canonical ledger, write the Discovery Ledger, write baseline artifacts, or
create benchmark claims.

## Write From Real Evidence

After known-null and known-positive control evidence files exist, write the
calibration file with explicit approval:

```bash
python3 -m autoalphafold3.agent calibrate-gate \
  --mode from-evidence \
  --known-null-evidence runs/gate_calibration/known_null.json \
  --known-positive-evidence runs/gate_calibration/known_positive.json \
  --approve I_APPROVE_GATE_CALIBRATION
```

The command writes only `runs/falsification_gate_calibration.json`. It does not
write baseline artifacts, the canonical ledger, the Discovery Ledger, benchmark
metrics, or trial artifacts.

## Evidence Rules

Both evidence records must be complete, real calibration records:

- `status=complete`
- `scorer_version=calpha_lddt_v1`
- `primary_metric=best_val_calpha_lddt`
- `split=public_val_small`
- non-empty `baseline_id` and `current_best_trial_id`
- non-empty `manifest_hashes` and `feature_fingerprints`
- finite `gate_thresholds` for `tau_attribution`, `rho_placebo`, and `k_seed`
- non-empty `control_evidence_ids`

The known-null record must have a kill verdict: `PLACEBO_KILL`,
`KNOCKOUT_SURVIVES`, `AXIS_MISS`, or `SEED_FRAGILE`.

The known-positive record must have `CONFIRMED`.

Synthetic fixture evidence is refused. Hand-writing a calibration record without
real control evidence would be fake gate evidence and must not be used to start
search.
