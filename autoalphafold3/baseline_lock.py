"""Human-approved baseline lock writer.

This module freezes already-produced official baseline evidence. It does not
run NanoFold, score artifacts, call Modal, or synthesize benchmark numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoalphafold3.baseline_readiness import (
    DEFAULT_ERROR_REPORT,
    DEFAULT_FEATURE_FINGERPRINTS,
    DEFAULT_METRICS,
    BaselineReadinessError,
    audit_baseline_readiness,
)

APPROVAL_TEXT = "I_APPROVE_BASELINE_LOCK"


class BaselineLockError(RuntimeError):
    """Raised when the baseline lock command refuses unsafe input."""


@dataclass(frozen=True)
class BaselineLockResult:
    """JSON-friendly result from a baseline lock attempt."""

    status: str
    baseline_dir: str
    dry_run: bool
    wrote_files: list[str]
    readiness: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "baseline_dir": self.baseline_dir,
            "dry_run": self.dry_run,
            "wrote_files": self.wrote_files,
            "readiness": self.readiness,
        }


def lock_baseline_from_scored_artifacts(
    *,
    source_dir: str | Path,
    feature_fingerprints_path: str | Path,
    baseline_dir: str | Path = "runs/baseline",
    approval: str | None = None,
    dry_run: bool = False,
) -> BaselineLockResult:
    """Freeze real scored baseline files into ``runs/baseline``.

    The source directory must already contain official, scorer-produced
    ``metrics.json`` and ``error_report.json`` files. The feature fingerprints
    must come from the approved data handoff, not from this command.
    """

    if approval != APPROVAL_TEXT:
        raise BaselineLockError(f"baseline lock requires --approve {APPROVAL_TEXT}")

    source = Path(source_dir)
    baseline = Path(baseline_dir)
    fingerprints_path = Path(feature_fingerprints_path)
    if not source.is_dir():
        raise BaselineLockError(f"source_dir is missing: {source}")
    if not fingerprints_path.is_file():
        raise BaselineLockError(f"feature_fingerprints_path is missing: {fingerprints_path}")

    metrics = _read_json(source / DEFAULT_METRICS)
    error_report = _read_json(source / DEFAULT_ERROR_REPORT)
    fingerprints = _read_json(fingerprints_path)

    _require_real_official_baseline(metrics, error_report)
    frozen_metrics = _baseline_artifact_payload(metrics, owner="metrics")
    frozen_error_report = _baseline_artifact_payload(error_report, owner="error_report")

    wrote: list[str] = []
    if not dry_run:
        _require_empty_target(baseline)
        baseline.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(baseline / DEFAULT_METRICS, frozen_metrics)
        wrote.append(str(baseline / DEFAULT_METRICS))
        _atomic_write_json(baseline / DEFAULT_ERROR_REPORT, frozen_error_report)
        wrote.append(str(baseline / DEFAULT_ERROR_REPORT))
        _atomic_write_json(baseline / DEFAULT_FEATURE_FINGERPRINTS, fingerprints)
        wrote.append(str(baseline / DEFAULT_FEATURE_FINGERPRINTS))
        readiness = audit_baseline_readiness(baseline_dir=baseline)
    else:
        readiness = _audit_payloads_without_writing(
            baseline=baseline,
            metrics=frozen_metrics,
            error_report=frozen_error_report,
            fingerprints=fingerprints,
        )

    readiness_payload = readiness.to_dict() if hasattr(readiness, "to_dict") else readiness
    status = "PASS" if readiness_payload.get("status") == "PASS" else "FAIL"
    if status != "PASS":
        raise BaselineLockError(f"baseline lock evidence is not ready: {readiness_payload.get('problems')}")
    return BaselineLockResult(
        status=status,
        baseline_dir=str(baseline),
        dry_run=dry_run,
        wrote_files=wrote,
        readiness=readiness_payload,
    )


def _require_real_official_baseline(metrics: dict[str, object], error_report: dict[str, object]) -> None:
    if metrics.get("official_benchmark_result") is not True:
        raise BaselineLockError("baseline metrics must be an official benchmark result")
    if metrics.get("local_only") is True:
        raise BaselineLockError("baseline metrics must not be local_only")
    if error_report.get("scorer_only") is not True:
        raise BaselineLockError("baseline error_report must prove scorer_only=true")


def _baseline_artifact_payload(payload: dict[str, object], *, owner: str) -> dict[str, object]:
    frozen = dict(payload)
    if owner == "metrics":
        artifacts = {
            "metrics_json": "runs/baseline/metrics.json",
            "error_report_json": "runs/baseline/error_report.json",
            "feature_fingerprints_json": "runs/baseline/feature_fingerprints.json",
        }
    elif owner == "error_report":
        artifacts = {
            "metrics_json": "runs/baseline/metrics.json",
            "error_report_json": "runs/baseline/error_report.json",
        }
    else:
        raise BaselineLockError(f"unsupported baseline artifact owner: {owner}")
    frozen["artifacts"] = artifacts
    return frozen


def _require_empty_target(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise BaselineLockError(f"baseline target exists and is not a directory: {path}")
    existing = sorted(child.name for child in path.iterdir())
    if existing:
        raise BaselineLockError(f"baseline target is not empty: {path}")


def _audit_payloads_without_writing(
    *,
    baseline: Path,
    metrics: dict[str, object],
    error_report: dict[str, object],
    fingerprints: dict[str, object],
) -> dict[str, object]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="autoaf3-baseline-lock-") as tmp:
        candidate = Path(tmp) / baseline.name
        candidate.mkdir()
        _atomic_write_json(candidate / DEFAULT_METRICS, metrics)
        _atomic_write_json(candidate / DEFAULT_ERROR_REPORT, error_report)
        _atomic_write_json(candidate / DEFAULT_FEATURE_FINGERPRINTS, fingerprints)
        return audit_baseline_readiness(baseline_dir=candidate).to_dict()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BaselineLockError(f"required source file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BaselineLockError(f"required source file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BaselineLockError(f"required source file must contain a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
