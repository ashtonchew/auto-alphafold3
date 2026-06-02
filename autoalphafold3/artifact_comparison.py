"""Local prediction-artifact comparison for autoresearch diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autoalphafold3.runner import RunnerError, validate_prediction_artifact


class ArtifactComparisonError(RuntimeError):
    """Raised when prediction artifacts cannot be compared."""


@dataclass(frozen=True)
class ArtifactSideSummary:
    """Stable summary for one prediction artifact."""

    path: str
    trial_id: str
    candidate_id: str | None
    split: str
    source: str | None
    max_templates: int | None
    prediction_count: int
    artifact_sha256: str
    target_hashes: dict[str, str]
    metrics: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionArtifactComparison:
    """JSON-friendly comparison between two prediction artifacts."""

    schema_version: str
    left: ArtifactSideSummary
    right: ArtifactSideSummary
    same_artifact_sha256: bool
    same_split: bool
    same_target_set: bool
    common_target_count: int
    left_only_targets: list[str]
    right_only_targets: list[str]
    identical_targets: list[str]
    changed_targets: list[str]
    all_common_predictions_identical: bool
    all_predictions_identical: bool
    metric_deltas: dict[str, float] | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["left"] = self.left.to_dict()
        payload["right"] = self.right.to_dict()
        return payload


def compare_prediction_artifacts(
    *,
    left_predictions: str | Path,
    right_predictions: str | Path,
    left_metrics: str | Path | None = None,
    right_metrics: str | Path | None = None,
) -> PredictionArtifactComparison:
    """Compare two local ``predictions.json`` artifacts without scoring them."""

    left_payload = _read_prediction_artifact(left_predictions)
    right_payload = _read_prediction_artifact(right_predictions)
    left_summary = _side_summary(
        path=left_predictions,
        payload=left_payload,
        metrics=_read_metrics(left_metrics) if left_metrics is not None else None,
    )
    right_summary = _side_summary(
        path=right_predictions,
        payload=right_payload,
        metrics=_read_metrics(right_metrics) if right_metrics is not None else None,
    )
    left_targets = set(left_summary.target_hashes)
    right_targets = set(right_summary.target_hashes)
    common = sorted(left_targets & right_targets)
    identical = [target for target in common if left_summary.target_hashes[target] == right_summary.target_hashes[target]]
    changed = [target for target in common if left_summary.target_hashes[target] != right_summary.target_hashes[target]]
    metric_deltas = _metric_deltas(left_summary.metrics, right_summary.metrics)
    same_target_set = left_targets == right_targets
    all_common_identical = len(identical) == len(common)
    return PredictionArtifactComparison(
        schema_version="autoaf3.prediction_artifact_comparison.v1",
        left=left_summary,
        right=right_summary,
        same_artifact_sha256=left_summary.artifact_sha256 == right_summary.artifact_sha256,
        same_split=left_summary.split == right_summary.split,
        same_target_set=same_target_set,
        common_target_count=len(common),
        left_only_targets=sorted(left_targets - right_targets),
        right_only_targets=sorted(right_targets - left_targets),
        identical_targets=identical,
        changed_targets=changed,
        all_common_predictions_identical=all_common_identical,
        all_predictions_identical=same_target_set and all_common_identical,
        metric_deltas=metric_deltas,
    )


def _read_prediction_artifact(path: str | Path) -> dict[str, object]:
    try:
        payload = _read_json(path)
        return validate_prediction_artifact(payload)
    except (OSError, json.JSONDecodeError, RunnerError) as exc:
        raise ArtifactComparisonError(f"invalid prediction artifact {path}: {exc}") from exc


def _read_metrics(path: str | Path) -> dict[str, object]:
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactComparisonError(f"invalid metrics artifact {path}: {exc}") from exc
    metrics = payload.get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else {}


def _read_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactComparisonError(f"JSON artifact must be an object: {path}")
    return payload


def _side_summary(
    *,
    path: str | Path,
    payload: dict[str, object],
    metrics: dict[str, object] | None,
) -> ArtifactSideSummary:
    predictions = payload["predictions"]
    assert isinstance(predictions, list)
    return ArtifactSideSummary(
        path=str(path),
        trial_id=str(payload["trial_id"]),
        candidate_id=_optional_str(payload.get("candidate_id")),
        split=str(payload["split"]),
        source=_optional_str(payload.get("source")),
        max_templates=_optional_int(payload.get("max_templates")),
        prediction_count=len(predictions),
        artifact_sha256=_sha256_json(payload),
        target_hashes=_target_hashes(predictions),
        metrics=metrics,
    )


def _target_hashes(predictions: list[object]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for prediction in predictions:
        assert isinstance(prediction, dict)
        target_id = str(prediction["target_id"])
        hashes[target_id] = _sha256_json(prediction)
    return hashes


def _metric_deltas(
    left_metrics: dict[str, object] | None,
    right_metrics: dict[str, object] | None,
) -> dict[str, float] | None:
    if left_metrics is None or right_metrics is None:
        return None
    deltas: dict[str, float] = {}
    for key in sorted(set(left_metrics) & set(right_metrics)):
        left_value = left_metrics.get(key)
        right_value = right_metrics.get(key)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            deltas[key] = float(right_value) - float(left_value)
    return deltas


def _sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
