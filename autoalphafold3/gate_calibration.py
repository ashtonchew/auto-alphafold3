"""Human-approved Falsification Gate calibration evidence writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoalphafold3.schema import FalsificationVerdict, PRIMARY_METRIC, SCORER_VERSION

APPROVAL_TEXT = "I_APPROVE_GATE_CALIBRATION"
DEFAULT_CALIBRATION_PATH = Path("runs/falsification_gate_calibration.json")
PUBLIC_VAL_SPLIT = "public_val_small"
KILL_VERDICTS = {
    FalsificationVerdict.PLACEBO_KILL.value,
    FalsificationVerdict.KNOCKOUT_SURVIVES.value,
    FalsificationVerdict.AXIS_MISS.value,
    FalsificationVerdict.SEED_FRAGILE.value,
}


class GateCalibrationError(RuntimeError):
    """Raised when calibration evidence is incomplete or unsafe to write."""


@dataclass(frozen=True)
class GateCalibrationResult:
    """JSON-friendly gate calibration command result."""

    status: str
    mode: str
    calibration_path: str
    wrote_files: list[str]
    plan: dict[str, object]
    calibration: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "calibration_path": self.calibration_path,
            "wrote_files": self.wrote_files,
            "plan": self.plan,
            "calibration": self.calibration,
        }


def calibrate_gate(
    *,
    repo_root: str | Path = ".",
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    known_null_evidence: str | Path | None = None,
    known_positive_evidence: str | Path | None = None,
    approval: str | None = None,
    mode: str = "dry-run",
) -> GateCalibrationResult:
    """Plan or write Falsification Gate calibration from real evidence files."""

    root = Path(repo_root)
    output_path = root / calibration_path
    plan = calibration_plan(calibration_path=calibration_path)
    if mode == "dry-run":
        return GateCalibrationResult(
            status="PLANNED",
            mode=mode,
            calibration_path=str(output_path),
            wrote_files=[],
            plan=plan,
        )
    if mode != "from-evidence":
        raise GateCalibrationError(f"unsupported calibration mode: {mode}")
    if approval != APPROVAL_TEXT:
        raise GateCalibrationError(f"gate calibration requires --approve {APPROVAL_TEXT}")
    if known_null_evidence is None or known_positive_evidence is None:
        raise GateCalibrationError("known-null and known-positive evidence paths are required")
    if output_path.exists():
        raise GateCalibrationError(f"calibration output already exists: {output_path}")

    known_null = _load_record(root / known_null_evidence, name="known_null", expected_positive=False)
    known_positive = _load_record(root / known_positive_evidence, name="known_positive", expected_positive=True)
    payload = {"known_null": known_null, "known_positive": known_positive}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, payload)
    return GateCalibrationResult(
        status="PASS",
        mode=mode,
        calibration_path=str(output_path),
        wrote_files=[str(output_path)],
        plan=plan,
        calibration=payload,
    )


def calibration_plan(*, calibration_path: str | Path = DEFAULT_CALIBRATION_PATH) -> dict[str, object]:
    """Return the required calibration evidence contract without writing."""

    return {
        "calibration_path": str(calibration_path),
        "requires_approval": APPROVAL_TEXT,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "required_records": {
            "known_null": {
                "verdict": sorted(KILL_VERDICTS),
                "status": "complete",
                "split": PUBLIC_VAL_SPLIT,
            },
            "known_positive": {
                "verdict": FalsificationVerdict.CONFIRMED.value,
                "status": "complete",
                "split": PUBLIC_VAL_SPLIT,
            },
        },
        "required_fields": [
            "status",
            "verdict",
            "scorer_version",
            "primary_metric",
            "split",
            "baseline_id",
            "current_best_trial_id",
            "manifest_hashes",
            "feature_fingerprints",
            "gate_thresholds",
            "control_evidence_ids",
        ],
    }


def _load_record(path: Path, *, name: str, expected_positive: bool) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateCalibrationError(f"{name} evidence file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateCalibrationError(f"{name} evidence file is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise GateCalibrationError(f"{name} evidence must be a JSON object")
    _validate_record(payload, name=name, expected_positive=expected_positive)
    return payload


def _validate_record(record: dict[str, object], *, name: str, expected_positive: bool) -> None:
    if str(record.get("status", "")).lower() != "complete":
        raise GateCalibrationError(f"{name} calibration status must be complete")
    verdict = record.get("verdict")
    if expected_positive:
        if verdict != FalsificationVerdict.CONFIRMED.value:
            raise GateCalibrationError("known_positive calibration must have CONFIRMED verdict")
    elif verdict not in KILL_VERDICTS:
        raise GateCalibrationError("known_null calibration must be killed by the gate")
    if record.get("synthetic_fixture") is True:
        raise GateCalibrationError(f"{name} calibration evidence must not be a synthetic fixture")
    if record.get("scorer_version") != SCORER_VERSION:
        raise GateCalibrationError(f"{name} calibration scorer_version must be {SCORER_VERSION}")
    if record.get("primary_metric") != PRIMARY_METRIC:
        raise GateCalibrationError(f"{name} calibration primary_metric must be {PRIMARY_METRIC}")
    if record.get("split") != PUBLIC_VAL_SPLIT:
        raise GateCalibrationError(f"{name} calibration split must be {PUBLIC_VAL_SPLIT}")
    for field_name in ("baseline_id", "current_best_trial_id"):
        if not isinstance(record.get(field_name), str) or not record[field_name]:
            raise GateCalibrationError(f"{name} calibration {field_name} is required")
    for collection_name in ("manifest_hashes", "feature_fingerprints"):
        collection = record.get(collection_name)
        if not isinstance(collection, dict) or not collection:
            raise GateCalibrationError(f"{name} calibration {collection_name} must be non-empty")
    control_ids = record.get("control_evidence_ids")
    if not isinstance(control_ids, list) or not control_ids or not all(isinstance(item, str) and item for item in control_ids):
        raise GateCalibrationError(f"{name} calibration requires control_evidence_ids")
    thresholds = record.get("gate_thresholds")
    if not isinstance(thresholds, dict):
        raise GateCalibrationError(f"{name} calibration gate_thresholds are required")
    for threshold_name in ("tau_attribution", "rho_placebo", "k_seed"):
        if threshold_name not in thresholds:
            raise GateCalibrationError(f"{name} calibration gate_thresholds.{threshold_name} is required")
        threshold = thresholds[threshold_name]
        if not isinstance(threshold, int | float):
            raise GateCalibrationError(f"{name} calibration gate_thresholds.{threshold_name} must be numeric")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
