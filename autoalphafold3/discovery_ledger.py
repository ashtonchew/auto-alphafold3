"""Confirmed-only Discovery Ledger helpers."""

from __future__ import annotations

import json
from pathlib import Path

from autoalphafold3.schema import (
    AutoFoldResult,
    DiscoveryProvenance,
    DiscoveryRecord,
    DiscoveryStatus,
    FalsificationVerdict,
    TrialStatus,
)

DEFAULT_DISCOVERY_LEDGER = Path("runs/discovery_ledger.jsonl")


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
    return DiscoveryRecord(
        trial_id=result.trial_id,
        candidate_id=result.candidate_id,
        mechanism=mechanism,
        axis_moved=provenance_model.predicted_axis,
        design_rule=design_rule,
        falsification=result.falsification,
        provenance=provenance_model,
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
            rows.append(DiscoveryRecord.model_validate_json(line))
        except ValueError as exc:
            raise DiscoveryLedgerError(f"invalid Discovery Ledger row {line_no} in {path}: {exc}") from exc
    return rows


def validate_discovery_record(record: DiscoveryRecord | dict[str, object]) -> DiscoveryRecord:
    """Validate one prospective confirmed Discovery Ledger row."""

    row = record if isinstance(record, DiscoveryRecord) else DiscoveryRecord.model_validate(record)
    if row.falsification.verdict != FalsificationVerdict.CONFIRMED:
        raise DiscoveryLedgerError("Discovery Ledger rows require CONFIRMED falsification verdicts")
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
