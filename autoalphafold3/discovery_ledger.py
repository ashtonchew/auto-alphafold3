"""Confirmed-only Discovery Ledger helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

from autoalphafold3.falsification import FalsificationError, decide_falsification_verdict
from autoalphafold3.schema import (
    AutoFoldResult,
    DiscoveryProvenance,
    DiscoveryRecord,
    DiscoveryStatus,
    FalsificationVerdict,
    TrialStatus,
)

DEFAULT_DISCOVERY_LEDGER = Path("runs/discovery_ledger.jsonl")
REQUIRED_VERDICT_NUMBERS = (
    "gain_full",
    "gain_knockout",
    "gain_placebo",
    "attributable_fraction",
    "axis_delta_observed",
    "seed_mean",
    "seed_std",
)
REQUIRED_GATE_THRESHOLDS = ("tau_attribution", "rho_placebo", "k_seed")


class DiscoveryLedgerError(ValueError):
    """Raised when a Discovery Ledger record is not confirmed evidence."""


def build_discovery_record(
    result: AutoFoldResult,
    *,
    mechanism: str,
    design_rule: str,
    provenance: DiscoveryProvenance | dict[str, object],
) -> DiscoveryRecord:
    """Build a confirmed Discovery Ledger record from a gated result."""

    _require_confirmed_result(result)
    if result.falsification is None:
        raise DiscoveryLedgerError("confirmed discovery requires falsification evidence")
    provenance_model = (
        provenance if isinstance(provenance, DiscoveryProvenance) else DiscoveryProvenance.model_validate(provenance)
    )
    return validate_discovery_record(
        DiscoveryRecord(
            trial_id=result.trial_id,
            candidate_id=result.candidate_id,
            mechanism=mechanism,
            axis_moved=provenance_model.predicted_axis,
            design_rule=design_rule,
            falsification=result.falsification,
            provenance=provenance_model,
        )
    )


def append_discovery_record(
    record: DiscoveryRecord | dict[str, object],
    *,
    ledger_path: str | Path = DEFAULT_DISCOVERY_LEDGER,
    dedupe: bool = True,
) -> None:
    """Append a confirmed Discovery Ledger record with duplicate protection."""

    row = validate_discovery_record(record)
    path = Path(ledger_path)
    existing = read_discovery_ledger(ledger_path=path)
    duplicate = _matching_record(existing, row)
    if duplicate is not None:
        if duplicate.model_dump(mode="json") == row.model_dump(mode="json") and dedupe:
            return
        raise DiscoveryLedgerError(f"conflicting Discovery Ledger record for {row.trial_id}/{row.candidate_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")


def read_discovery_ledger(*, ledger_path: str | Path = DEFAULT_DISCOVERY_LEDGER) -> list[DiscoveryRecord]:
    """Read and validate confirmed Discovery Ledger records."""

    path = Path(ledger_path)
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(validate_discovery_record(DiscoveryRecord.model_validate_json(line)))
        except ValueError as exc:
            raise DiscoveryLedgerError(f"invalid Discovery Ledger row {line_no} in {path}: {exc}") from exc
    return rows


def validate_discovery_record(record: DiscoveryRecord | dict[str, object]) -> DiscoveryRecord:
    """Validate one prospective confirmed Discovery Ledger row."""

    row = record if isinstance(record, DiscoveryRecord) else DiscoveryRecord.model_validate(record)
    if row.falsification.verdict != FalsificationVerdict.CONFIRMED:
        raise DiscoveryLedgerError("Discovery Ledger rows require CONFIRMED falsification verdicts")
    _require_axis_consistency(row)
    _require_verdict_evidence_consistency(row)
    return row


def _require_confirmed_result(result: AutoFoldResult) -> None:
    if result.status != TrialStatus.KEEP:
        raise DiscoveryLedgerError("Discovery Ledger requires KEEP status after gate confirmation")
    if result.discovery != DiscoveryStatus.CONFIRMED:
        raise DiscoveryLedgerError("Discovery Ledger requires discovery=CONFIRMED")
    if result.falsification is None or result.falsification.verdict != FalsificationVerdict.CONFIRMED:
        raise DiscoveryLedgerError("Discovery Ledger requires a CONFIRMED falsification verdict")


def _matching_record(records: list[DiscoveryRecord], row: DiscoveryRecord) -> DiscoveryRecord | None:
    for existing in records:
        if existing.trial_id == row.trial_id and existing.candidate_id == row.candidate_id:
            return existing
    return None


def _require_axis_consistency(row: DiscoveryRecord) -> None:
    if row.axis_moved != row.provenance.predicted_axis:
        raise DiscoveryLedgerError("Discovery Ledger axis_moved must match provenance predicted_axis")


def _require_verdict_evidence_consistency(row: DiscoveryRecord) -> None:
    numbers = row.provenance.verdict_numbers
    thresholds = row.provenance.gate_thresholds
    _require_keys("verdict_numbers", numbers, REQUIRED_VERDICT_NUMBERS)
    _require_keys("gate_thresholds", thresholds, REQUIRED_GATE_THRESHOLDS)

    falsification = row.falsification
    for name in REQUIRED_VERDICT_NUMBERS:
        _require_close(f"verdict_numbers.{name}", numbers[name], getattr(falsification, name))

    try:
        expected_verdict = decide_falsification_verdict(
            gain_full=falsification.gain_full,
            gain_knockout=falsification.gain_knockout,
            gain_placebo=falsification.gain_placebo,
            axis_prediction_held=falsification.axis_prediction_held,
            seed_std=falsification.seed_std,
            tau_attribution=thresholds["tau_attribution"],
            rho_placebo=thresholds["rho_placebo"],
            k_seed=thresholds["k_seed"],
        )
    except FalsificationError as exc:
        raise DiscoveryLedgerError(f"Discovery Ledger has invalid gate evidence: {exc}") from exc
    if expected_verdict != FalsificationVerdict.CONFIRMED:
        raise DiscoveryLedgerError(f"Discovery Ledger evidence recomputes to {expected_verdict}, not CONFIRMED")


def _require_keys(collection_name: str, collection: dict[str, float], required: tuple[str, ...]) -> None:
    missing = [key for key in required if key not in collection]
    if missing:
        raise DiscoveryLedgerError(f"{collection_name} missing required keys: {', '.join(missing)}")


def _require_close(name: str, observed: float, expected: float) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise DiscoveryLedgerError(f"{name} does not match falsification evidence")
