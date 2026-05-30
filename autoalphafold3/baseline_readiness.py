"""Baseline lock readiness checks for pre-run search safety."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from autoalphafold3.ledger import read_ledger
from autoalphafold3.schema import PRIMARY_METRIC, SCORER_VERSION, AutoFoldResult, TrialStatus

BaselineReadinessStatus = Literal["PASS", "FAIL"]

DEFAULT_BASELINE_DIR = Path("runs/baseline")
DEFAULT_METRICS = "metrics.json"
DEFAULT_ERROR_REPORT = "error_report.json"
DEFAULT_FEATURE_FINGERPRINTS = "feature_fingerprints.json"
PUBLIC_VAL_SPLIT = "public_val_small"
REQUIRED_MANIFEST_HASHES = ("train_tiny", "public_val_small")


class BaselineReadinessError(RuntimeError):
    """Raised when current-best lookup is requested without a ready baseline."""


@dataclass(frozen=True)
class BaselineReadinessReport:
    """JSON-friendly report for baseline lock readiness."""

    status: BaselineReadinessStatus
    baseline_dir: str
    baseline_metrics_present: bool
    baseline_error_report_present: bool
    feature_fingerprints_present: bool
    official_benchmark_result: bool = False
    primary_metric: str | None = None
    baseline_score: float | None = None
    scorer_version: str | None = None
    split: str | None = None
    manifest_hashes_valid: bool = False
    feature_fingerprints_valid: bool = False
    max_templates_zero: bool = False
    current_best_trial_id: str | None = None
    current_best_score: float | None = None
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pending_human_action: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "baseline_dir": self.baseline_dir,
            "baseline_metrics_present": self.baseline_metrics_present,
            "baseline_error_report_present": self.baseline_error_report_present,
            "feature_fingerprints_present": self.feature_fingerprints_present,
            "official_benchmark_result": self.official_benchmark_result,
            "primary_metric": self.primary_metric,
            "baseline_score": self.baseline_score,
            "scorer_version": self.scorer_version,
            "split": self.split,
            "manifest_hashes_valid": self.manifest_hashes_valid,
            "feature_fingerprints_valid": self.feature_fingerprints_valid,
            "max_templates_zero": self.max_templates_zero,
            "current_best_trial_id": self.current_best_trial_id,
            "current_best_score": self.current_best_score,
            "problems": self.problems,
            "notes": self.notes,
            "pending_human_action": self.pending_human_action,
        }


@dataclass(frozen=True)
class CurrentBest:
    """Best score available after a ready baseline and optional ledger rows."""

    trial_id: str
    candidate_id: str
    score: float
    source: Literal["baseline", "ledger_keep"]

    def to_dict(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "candidate_id": self.candidate_id,
            "score": self.score,
            "source": self.source,
        }


def audit_baseline_readiness(
    *,
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
) -> BaselineReadinessReport:
    """Validate existing baseline lock evidence without creating artifacts."""

    base = Path(baseline_dir)
    metrics_path = base / DEFAULT_METRICS
    error_report_path = base / DEFAULT_ERROR_REPORT
    fingerprints_path = base / DEFAULT_FEATURE_FINGERPRINTS
    problems: list[str] = []
    notes: list[str] = []

    metrics_present = metrics_path.exists()
    error_present = error_report_path.exists()
    fingerprints_present = fingerprints_path.exists()

    if not metrics_present:
        problems.append("baseline metrics.json is missing")
    if not error_present:
        problems.append("baseline error_report.json is missing")
    if not fingerprints_present:
        problems.append("baseline feature_fingerprints.json is missing")
    if not metrics_present or not error_present or not fingerprints_present:
        return BaselineReadinessReport(
            status="FAIL",
            baseline_dir=str(base),
            baseline_metrics_present=metrics_present,
            baseline_error_report_present=error_present,
            feature_fingerprints_present=fingerprints_present,
            problems=problems,
            notes=notes,
            pending_human_action="Provide real locked baseline metrics, error report, and feature fingerprints.",
        )

    try:
        metrics = _read_json(metrics_path)
        error_report = _read_json(error_report_path)
        fingerprints = _read_json(fingerprints_path)
    except (json.JSONDecodeError, OSError) as exc:
        problems.append(f"baseline lock JSON is unreadable: {exc}")
        return _failed_report(base, metrics_present, error_present, fingerprints_present, problems)

    if metrics.get("schema_version") != "autoaf3.metrics.v1":
        problems.append("baseline metrics schema_version must be autoaf3.metrics.v1")
    official = metrics.get("official_benchmark_result") is True
    if not official:
        problems.append("baseline metrics must set official_benchmark_result=true")
    if metrics.get("local_only") is True:
        problems.append("baseline metrics must not be local_only")
    primary_metric = metrics.get("primary_metric")
    if primary_metric != PRIMARY_METRIC:
        problems.append(f"baseline primary_metric must be {PRIMARY_METRIC}")
    scorer_version = metrics.get("scorer_version")
    if scorer_version != SCORER_VERSION:
        problems.append(f"baseline scorer_version must be {SCORER_VERSION}")
    split = metrics.get("split")
    if split != PUBLIC_VAL_SPLIT:
        problems.append(f"baseline split must be {PUBLIC_VAL_SPLIT}")
    if metrics.get("status") != TrialStatus.SCORED.value:
        problems.append("baseline status must be SCORED")

    score = _extract_score(metrics, problems)
    manifest_hashes_valid = _validate_manifest_hashes(metrics, problems)
    feature_fingerprints_valid = _validate_feature_fingerprints(fingerprints, problems)
    max_templates_zero = _has_zero_templates(metrics, fingerprints, error_report)
    if not max_templates_zero:
        problems.append("baseline lock must prove max_templates=0")
    if not isinstance(metrics.get("fold_cartographer"), dict):
        problems.append("baseline metrics must include fold_cartographer")
    if not isinstance(error_report, dict) or error_report.get("scorer_only") is not True:
        problems.append("baseline error_report must prove scorer_only=true")

    status: BaselineReadinessStatus = "FAIL" if problems else "PASS"
    return BaselineReadinessReport(
        status=status,
        baseline_dir=str(base),
        baseline_metrics_present=metrics_present,
        baseline_error_report_present=error_present,
        feature_fingerprints_present=fingerprints_present,
        official_benchmark_result=official,
        primary_metric=str(primary_metric) if primary_metric is not None else None,
        baseline_score=score,
        scorer_version=str(scorer_version) if scorer_version is not None else None,
        split=str(split) if split is not None else None,
        manifest_hashes_valid=manifest_hashes_valid,
        feature_fingerprints_valid=feature_fingerprints_valid,
        max_templates_zero=max_templates_zero,
        current_best_trial_id=str(metrics.get("trial_id")) if score is not None else None,
        current_best_score=score,
        problems=problems,
        notes=notes,
        pending_human_action=None if not problems else "Repair or provide a real locked baseline before search.",
    )


def require_baseline_ready(report: BaselineReadinessReport) -> BaselineReadinessReport:
    """Raise if the baseline lock is not ready for search."""

    if report.status != "PASS":
        raise BaselineReadinessError(f"baseline is not ready: {report.problems}")
    return report


def current_best_from_baseline_and_ledger(
    *,
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
    ledger_path: str | Path = "runs/ledger.jsonl",
) -> CurrentBest:
    """Return the current best score, refusing to proceed without a ready baseline."""

    report = require_baseline_ready(audit_baseline_readiness(baseline_dir=baseline_dir))
    if report.baseline_score is None or report.current_best_trial_id is None:
        raise BaselineReadinessError("baseline report is missing current-best score")
    best = CurrentBest(
        trial_id=report.current_best_trial_id,
        candidate_id="baseline_lock",
        score=report.baseline_score,
        source="baseline",
    )
    for row in read_ledger(ledger_path=ledger_path):
        maybe_score = _score_from_result(row)
        if row.status == TrialStatus.KEEP and maybe_score is not None and maybe_score > best.score:
            best = CurrentBest(
                trial_id=row.trial_id,
                candidate_id=row.candidate_id,
                score=maybe_score,
                source="ledger_keep",
            )
    return best


def _failed_report(
    base: Path,
    metrics_present: bool,
    error_present: bool,
    fingerprints_present: bool,
    problems: list[str],
) -> BaselineReadinessReport:
    return BaselineReadinessReport(
        status="FAIL",
        baseline_dir=str(base),
        baseline_metrics_present=metrics_present,
        baseline_error_report_present=error_present,
        feature_fingerprints_present=fingerprints_present,
        problems=problems,
        pending_human_action="Repair or provide a real locked baseline before search.",
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _extract_score(metrics: dict[str, object], problems: list[str]) -> float | None:
    values = metrics.get("metrics")
    if not isinstance(values, dict):
        problems.append("baseline metrics must contain a metrics object")
        return None
    score = values.get(PRIMARY_METRIC)
    if not isinstance(score, int | float) or not math.isfinite(float(score)):
        problems.append(f"baseline metrics.{PRIMARY_METRIC} must be finite")
        return None
    score_float = float(score)
    if score_float < 0.0 or score_float > 1.0:
        problems.append(f"baseline metrics.{PRIMARY_METRIC} must be in [0, 1]")
        return None
    return score_float


def _validate_manifest_hashes(metrics: dict[str, object], problems: list[str]) -> bool:
    manifests = metrics.get("manifests")
    if not isinstance(manifests, dict):
        problems.append("baseline metrics must include manifest hashes")
        return False
    missing = [name for name in REQUIRED_MANIFEST_HASHES if not _looks_like_sha256(manifests.get(name))]
    if missing:
        problems.append(f"baseline manifest hashes missing or invalid: {', '.join(missing)}")
        return False
    return True


def _validate_feature_fingerprints(fingerprints: dict[str, object], problems: list[str]) -> bool:
    if not fingerprints:
        problems.append("baseline feature fingerprints are empty")
        return False
    values = fingerprints.get("features", fingerprints)
    if not isinstance(values, dict):
        problems.append("baseline feature fingerprints must be a JSON object")
        return False
    if not any(_looks_like_sha256(value) for value in values.values()):
        problems.append("baseline feature fingerprints must include at least one SHA256")
        return False
    return True


def _has_zero_templates(
    metrics: dict[str, object],
    fingerprints: dict[str, object],
    error_report: dict[str, object],
) -> bool:
    present = False
    for payload in (metrics, fingerprints, error_report):
        for key in ("max_templates", "template_policy"):
            if key not in payload:
                continue
            present = True
            if not _is_zero_template_evidence(payload[key]):
                return False
    return present


def _is_zero_template_evidence(value: object) -> bool:
    if value == 0:
        return True
    if isinstance(value, str) and value in {"0", "max_templates=0", "no_templates"}:
        return True
    if isinstance(value, dict) and value.get("max_templates") == 0:
        return True
    return False


def _score_from_result(row: AutoFoldResult) -> float | None:
    score = row.metrics.get(PRIMARY_METRIC)
    if isinstance(score, int | float) and math.isfinite(float(score)):
        return float(score)
    return None


def _looks_like_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)
