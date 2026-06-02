"""Local prediction-artifact comparison for autoresearch diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
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
class TargetCoordinateDelta:
    """Coordinate-distance summary for one shared target."""

    target_id: str
    left_residue_count: int
    right_residue_count: int
    comparable_residue_count: int
    mean_abs_coordinate_delta: float | None
    max_abs_coordinate_delta: float | None
    rmsd: float | None


@dataclass(frozen=True)
class TargetDistanceDelta:
    """Pairwise-distance summary for one shared target."""

    target_id: str
    left_residue_count: int
    right_residue_count: int
    comparable_pair_count: int
    mean_abs_pair_distance_delta: float | None
    max_abs_pair_distance_delta: float | None
    pair_distance_rmsd: float | None
    fraction_lt_0_5A: float | None
    fraction_lt_1A: float | None
    fraction_lt_2A: float | None
    fraction_lt_4A: float | None


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
    coordinate_deltas: dict[str, TargetCoordinateDelta]
    coordinate_delta_summary: dict[str, object]
    distance_deltas: dict[str, TargetDistanceDelta]
    distance_delta_summary: dict[str, object]
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
    coordinate_deltas = _coordinate_deltas(left_payload, right_payload, common)
    distance_deltas = _distance_deltas(left_payload, right_payload, common)
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
        coordinate_deltas=coordinate_deltas,
        coordinate_delta_summary=_coordinate_delta_summary(coordinate_deltas),
        distance_deltas=distance_deltas,
        distance_delta_summary=_distance_delta_summary(distance_deltas),
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


def _coordinate_deltas(
    left_payload: dict[str, object],
    right_payload: dict[str, object],
    common_targets: list[str],
) -> dict[str, TargetCoordinateDelta]:
    left_by_target = _predictions_by_target(left_payload)
    right_by_target = _predictions_by_target(right_payload)
    return {
        target_id: _target_coordinate_delta(target_id, left_by_target[target_id], right_by_target[target_id])
        for target_id in common_targets
    }


def _predictions_by_target(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    predictions = payload["predictions"]
    assert isinstance(predictions, list)
    by_target: dict[str, dict[str, object]] = {}
    for prediction in predictions:
        assert isinstance(prediction, dict)
        by_target[str(prediction["target_id"])] = prediction
    return by_target


def _target_coordinate_delta(
    target_id: str,
    left_prediction: dict[str, object],
    right_prediction: dict[str, object],
) -> TargetCoordinateDelta:
    left = _coordinates(left_prediction)
    right = _coordinates(right_prediction)
    comparable = min(len(left), len(right))
    if comparable == 0 or len(left) != len(right):
        return TargetCoordinateDelta(
            target_id=target_id,
            left_residue_count=len(left),
            right_residue_count=len(right),
            comparable_residue_count=comparable,
            mean_abs_coordinate_delta=None,
            max_abs_coordinate_delta=None,
            rmsd=None,
        )
    abs_deltas: list[float] = []
    squared_residue_distances: list[float] = []
    for left_xyz, right_xyz in zip(left, right, strict=True):
        squared = 0.0
        for left_coord, right_coord in zip(left_xyz, right_xyz, strict=True):
            delta = float(right_coord) - float(left_coord)
            abs_deltas.append(abs(delta))
            squared += delta * delta
        squared_residue_distances.append(squared)
    return TargetCoordinateDelta(
        target_id=target_id,
        left_residue_count=len(left),
        right_residue_count=len(right),
        comparable_residue_count=comparable,
        mean_abs_coordinate_delta=sum(abs_deltas) / len(abs_deltas),
        max_abs_coordinate_delta=max(abs_deltas),
        rmsd=math.sqrt(sum(squared_residue_distances) / len(squared_residue_distances)),
    )


def _coordinates(prediction: dict[str, object]) -> list[list[float]]:
    coordinates = prediction["predicted_ca"]
    assert isinstance(coordinates, list)
    return [[float(coord) for coord in xyz] for xyz in coordinates if isinstance(xyz, list)]


def _coordinate_delta_summary(deltas: dict[str, TargetCoordinateDelta]) -> dict[str, object]:
    comparable = [delta for delta in deltas.values() if delta.rmsd is not None]
    mismatched = [
        delta.target_id
        for delta in deltas.values()
        if delta.left_residue_count != delta.right_residue_count or delta.rmsd is None
    ]
    rmsds = [float(delta.rmsd) for delta in comparable if delta.rmsd is not None]
    mean_abs_values = [
        float(delta.mean_abs_coordinate_delta)
        for delta in comparable
        if delta.mean_abs_coordinate_delta is not None
    ]
    return {
        "target_count": len(deltas),
        "comparable_target_count": len(comparable),
        "residue_count_mismatch_targets": sorted(mismatched),
        "mean_target_rmsd": sum(rmsds) / len(rmsds) if rmsds else None,
        "max_target_rmsd": max(rmsds) if rmsds else None,
        "mean_target_mean_abs_coordinate_delta": sum(mean_abs_values) / len(mean_abs_values)
        if mean_abs_values
        else None,
    }


def _distance_deltas(
    left_payload: dict[str, object],
    right_payload: dict[str, object],
    common_targets: list[str],
) -> dict[str, TargetDistanceDelta]:
    left_by_target = _predictions_by_target(left_payload)
    right_by_target = _predictions_by_target(right_payload)
    return {
        target_id: _target_distance_delta(target_id, left_by_target[target_id], right_by_target[target_id])
        for target_id in common_targets
    }


def _target_distance_delta(
    target_id: str,
    left_prediction: dict[str, object],
    right_prediction: dict[str, object],
) -> TargetDistanceDelta:
    left = _coordinates(left_prediction)
    right = _coordinates(right_prediction)
    if len(left) < 2 or len(left) != len(right):
        return TargetDistanceDelta(
            target_id=target_id,
            left_residue_count=len(left),
            right_residue_count=len(right),
            comparable_pair_count=0,
            mean_abs_pair_distance_delta=None,
            max_abs_pair_distance_delta=None,
            pair_distance_rmsd=None,
            fraction_lt_0_5A=None,
            fraction_lt_1A=None,
            fraction_lt_2A=None,
            fraction_lt_4A=None,
        )
    abs_deltas: list[float] = []
    for left_index in range(len(left)):
        for right_index in range(left_index + 1, len(left)):
            left_distance = _euclidean_distance(left[left_index], left[right_index])
            right_distance = _euclidean_distance(right[left_index], right[right_index])
            abs_deltas.append(abs(right_distance - left_distance))
    return TargetDistanceDelta(
        target_id=target_id,
        left_residue_count=len(left),
        right_residue_count=len(right),
        comparable_pair_count=len(abs_deltas),
        mean_abs_pair_distance_delta=sum(abs_deltas) / len(abs_deltas),
        max_abs_pair_distance_delta=max(abs_deltas),
        pair_distance_rmsd=math.sqrt(sum(delta * delta for delta in abs_deltas) / len(abs_deltas)),
        fraction_lt_0_5A=_fraction_below(abs_deltas, 0.5),
        fraction_lt_1A=_fraction_below(abs_deltas, 1.0),
        fraction_lt_2A=_fraction_below(abs_deltas, 2.0),
        fraction_lt_4A=_fraction_below(abs_deltas, 4.0),
    )


def _distance_delta_summary(deltas: dict[str, TargetDistanceDelta]) -> dict[str, object]:
    comparable = [delta for delta in deltas.values() if delta.pair_distance_rmsd is not None]
    mismatched = [
        delta.target_id
        for delta in deltas.values()
        if delta.left_residue_count != delta.right_residue_count or delta.pair_distance_rmsd is None
    ]
    rmsds = [float(delta.pair_distance_rmsd) for delta in comparable if delta.pair_distance_rmsd is not None]
    mean_abs_values = [
        float(delta.mean_abs_pair_distance_delta)
        for delta in comparable
        if delta.mean_abs_pair_distance_delta is not None
    ]
    max_abs_values = [
        float(delta.max_abs_pair_distance_delta)
        for delta in comparable
        if delta.max_abs_pair_distance_delta is not None
    ]
    return {
        "target_count": len(deltas),
        "comparable_target_count": len(comparable),
        "residue_count_mismatch_targets": sorted(mismatched),
        "mean_target_pair_distance_rmsd": sum(rmsds) / len(rmsds) if rmsds else None,
        "max_target_pair_distance_rmsd": max(rmsds) if rmsds else None,
        "mean_target_mean_abs_pair_distance_delta": sum(mean_abs_values) / len(mean_abs_values)
        if mean_abs_values
        else None,
        "max_target_max_abs_pair_distance_delta": max(max_abs_values) if max_abs_values else None,
        "mean_fraction_lt_0_5A": _mean_optional(delta.fraction_lt_0_5A for delta in comparable),
        "mean_fraction_lt_1A": _mean_optional(delta.fraction_lt_1A for delta in comparable),
        "mean_fraction_lt_2A": _mean_optional(delta.fraction_lt_2A for delta in comparable),
        "mean_fraction_lt_4A": _mean_optional(delta.fraction_lt_4A for delta in comparable),
    }


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((right_coord - left_coord) ** 2 for left_coord, right_coord in zip(left, right, strict=True)))


def _fraction_below(values: list[float], threshold: float) -> float:
    return sum(1 for value in values if value < threshold) / len(values)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    checked = [float(value) for value in values if value is not None]
    return sum(checked) / len(checked) if checked else None


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
