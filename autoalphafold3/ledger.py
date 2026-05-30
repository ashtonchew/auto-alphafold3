"""JSONL ledger helpers for local and future Modal results."""

from __future__ import annotations

import json
from pathlib import Path

from autoalphafold3.schema import AutoFoldResult, TrialStatus

DEFAULT_LEDGER = Path("runs/ledger.jsonl")


def append_ledger(
    row: AutoFoldResult | dict[str, object],
    *,
    ledger_path: str | Path = DEFAULT_LEDGER,
    dedupe: bool = False,
    validate_lifecycle: bool = False,
) -> None:
    """Append a validated canonical result row to the JSONL ledger."""

    result = row if isinstance(row, AutoFoldResult) else AutoFoldResult.model_validate(row)
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dedupe and _has_duplicate_row(result, ledger_path=path):
        return
    if validate_lifecycle:
        validate_lifecycle_transition(read_ledger(ledger_path=path), result)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")


def read_ledger(*, ledger_path: str | Path = DEFAULT_LEDGER) -> list[AutoFoldResult]:
    """Read and validate all rows from the JSONL ledger."""

    path = Path(ledger_path)
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(AutoFoldResult.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid ledger row {line_no} in {path}: {exc}") from exc
    return rows


def validate_ledger_row(row: dict[str, object]) -> AutoFoldResult:
    """Validate one prospective ledger row."""

    return AutoFoldResult.model_validate(row)


def latest_result_for_trial(trial_id: str, *, ledger_path: str | Path = DEFAULT_LEDGER) -> AutoFoldResult | None:
    """Return the latest ledger result for one trial, if present."""

    for row in reversed(read_ledger(ledger_path=ledger_path)):
        if row.trial_id == trial_id:
            return row
    return None


ALLOWED_TRANSITIONS: dict[TrialStatus, set[TrialStatus]] = {
    TrialStatus.DRAFT: {TrialStatus.PREFLIGHT_PASSED, TrialStatus.FAIL, TrialStatus.INFRA_FAIL, TrialStatus.ARCHIVED},
    TrialStatus.PREFLIGHT_PASSED: {TrialStatus.SUBMITTED, TrialStatus.FAIL, TrialStatus.INFRA_FAIL, TrialStatus.ARCHIVED},
    TrialStatus.SUBMITTED: {TrialStatus.RUNNING, TrialStatus.FAIL, TrialStatus.INFRA_FAIL},
    TrialStatus.RUNNING: {TrialStatus.SCORED, TrialStatus.FAIL, TrialStatus.INFRA_FAIL},
    TrialStatus.SCORED: {TrialStatus.KEEP, TrialStatus.DISCARD, TrialStatus.FAIL, TrialStatus.ARCHIVED},
    TrialStatus.KEEP: {TrialStatus.ARCHIVED},
    TrialStatus.DISCARD: {TrialStatus.ARCHIVED},
    TrialStatus.FAIL: {TrialStatus.ARCHIVED},
    TrialStatus.INFRA_FAIL: {TrialStatus.ARCHIVED},
    TrialStatus.ARCHIVED: set(),
}


def validate_lifecycle_transition(existing_rows: list[AutoFoldResult], row: AutoFoldResult) -> None:
    """Require lifecycle rows for one trial to move forward only."""

    previous = next((item for item in reversed(existing_rows) if item.trial_id == row.trial_id), None)
    if previous is None:
        return
    if previous.status == row.status:
        return
    allowed = ALLOWED_TRANSITIONS.get(previous.status, set())
    if row.status not in allowed:
        raise ValueError(f"invalid lifecycle transition for {row.trial_id}: {previous.status} -> {row.status}")


def _has_duplicate_row(result: AutoFoldResult, *, ledger_path: Path) -> bool:
    for row in read_ledger(ledger_path=ledger_path):
        if (
            row.trial_id == result.trial_id
            and row.status == result.status
            and row.candidate_id == result.candidate_id
            and row.failure_signature == result.failure_signature
        ):
            return True
    return False
