"""Label-free prediction geometry diagnostics for autoresearch artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autoalphafold3.runner import RunnerError, validate_prediction_artifact


class PredictionGeometryError(RuntimeError):
    """Raised when prediction geometry artifacts cannot be audited."""


@dataclass(frozen=True)
class TargetGeometrySummary:
    target_id: str
    residue_count: int
    finite_coordinate_count: int
    coordinate_min: float
    coordinate_max: float
    centroid_norm: float
    radius_mean: float
    radius_max: float
    adjacent_distance_count: int
    adjacent_distance_mean: float | None
    adjacent_distance_min: float | None
    adjacent_distance_max: float | None
    adjacent_distance_rms: float | None
    pair_distance_count: int
    pair_distance_mean: float | None
    pair_distance_min: float | None
    pair_distance_max: float | None
    fraction_pair_distance_lt_4A: float | None
    fraction_pair_distance_lt_8A: float | None
    fraction_pair_distance_gt_40A: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactGeometrySummary:
    path: str
    trial_id: str
    candidate_id: str | None
    split: str
    source: str | None
    max_templates: int | None
    target_count: int
    residue_count_total: int
    mean_target_radius_mean: float | None
    max_target_radius_max: float | None
    mean_adjacent_distance: float | None
    max_adjacent_distance: float | None
    mean_pair_distance: float | None
    max_pair_distance: float | None
    mean_fraction_pair_distance_lt_8A: float | None
    mean_fraction_pair_distance_gt_40A: float | None
    scale_flags: list[str]
    target_summaries: dict[str, TargetGeometrySummary]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["target_summaries"] = {
            target_id: summary.to_dict() for target_id, summary in self.target_summaries.items()
        }
        return payload


@dataclass(frozen=True)
class ReferenceGeometryDelta:
    reference_trial_id: str
    candidate_trial_id: str
    same_split: bool
    same_target_set: bool
    common_target_count: int
    mean_radius_scale_ratio: float | None
    mean_adjacent_distance_delta: float | None
    mean_pair_distance_delta: float | None
    candidate_flags: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_prediction_geometry(
    *,
    predictions: list[str | Path],
    reference_predictions: str | Path | None = None,
) -> dict[str, object]:
    """Summarize label-free coordinate geometry for prediction artifacts.

    The audit intentionally does not read labels, invoke the scorer, or write
    ledgers. It is for offline triage before deciding whether another bounded
    live candidate is justified.
    """

    if not predictions:
        raise PredictionGeometryError("at least one predictions artifact is required")
    summaries = [_summarize_artifact(path) for path in predictions]
    reference = _summarize_artifact(reference_predictions) if reference_predictions is not None else None
    reference_deltas = (
        [_reference_delta(reference, summary).to_dict() for summary in summaries] if reference is not None else []
    )
    return {
        "schema_version": "autoaf3.prediction_geometry_audit.v1",
        "official_benchmark_result": False,
        "starts_search": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "reference": reference.to_dict() if reference is not None else None,
        "artifacts": [summary.to_dict() for summary in summaries],
        "reference_deltas": reference_deltas,
        "recommendation": _recommendation(summaries=summaries, reference_deltas=reference_deltas),
    }


def _summarize_artifact(path: str | Path) -> ArtifactGeometrySummary:
    payload = _read_prediction_artifact(path)
    predictions = payload["predictions"]
    assert isinstance(predictions, list)
    target_summaries = {
        str(prediction["target_id"]): _target_summary(str(prediction["target_id"]), _coordinates(prediction))
        for prediction in predictions
        if isinstance(prediction, dict)
    }
    radius_means = [summary.radius_mean for summary in target_summaries.values()]
    radius_maxes = [summary.radius_max for summary in target_summaries.values()]
    adjacent_means = _present(summary.adjacent_distance_mean for summary in target_summaries.values())
    adjacent_maxes = _present(summary.adjacent_distance_max for summary in target_summaries.values())
    pair_means = _present(summary.pair_distance_mean for summary in target_summaries.values())
    pair_maxes = _present(summary.pair_distance_max for summary in target_summaries.values())
    fraction_lt_8 = _present(summary.fraction_pair_distance_lt_8A for summary in target_summaries.values())
    fraction_gt_40 = _present(summary.fraction_pair_distance_gt_40A for summary in target_summaries.values())
    flags = _scale_flags(
        target_summaries=list(target_summaries.values()),
        mean_adjacent_distance=_mean(adjacent_means),
        max_adjacent_distance=max(adjacent_maxes) if adjacent_maxes else None,
        mean_pair_distance=_mean(pair_means),
        max_pair_distance=max(pair_maxes) if pair_maxes else None,
    )
    return ArtifactGeometrySummary(
        path=str(path),
        trial_id=str(payload["trial_id"]),
        candidate_id=_optional_str(payload.get("candidate_id")),
        split=str(payload["split"]),
        source=_optional_str(payload.get("source")),
        max_templates=_optional_int(payload.get("max_templates")),
        target_count=len(target_summaries),
        residue_count_total=sum(summary.residue_count for summary in target_summaries.values()),
        mean_target_radius_mean=_mean(radius_means),
        max_target_radius_max=max(radius_maxes) if radius_maxes else None,
        mean_adjacent_distance=_mean(adjacent_means),
        max_adjacent_distance=max(adjacent_maxes) if adjacent_maxes else None,
        mean_pair_distance=_mean(pair_means),
        max_pair_distance=max(pair_maxes) if pair_maxes else None,
        mean_fraction_pair_distance_lt_8A=_mean(fraction_lt_8),
        mean_fraction_pair_distance_gt_40A=_mean(fraction_gt_40),
        scale_flags=flags,
        target_summaries=target_summaries,
    )


def _target_summary(target_id: str, coordinates: list[list[float]]) -> TargetGeometrySummary:
    flattened = [coord for xyz in coordinates for coord in xyz]
    centroid = [sum(xyz[axis] for xyz in coordinates) / len(coordinates) for axis in range(3)]
    radii = [_euclidean_distance(xyz, centroid) for xyz in coordinates]
    adjacent_distances = [
        _euclidean_distance(coordinates[index], coordinates[index + 1])
        for index in range(len(coordinates) - 1)
    ]
    pair_distances = [
        _euclidean_distance(coordinates[left], coordinates[right])
        for left in range(len(coordinates))
        for right in range(left + 1, len(coordinates))
    ]
    return TargetGeometrySummary(
        target_id=target_id,
        residue_count=len(coordinates),
        finite_coordinate_count=len(flattened),
        coordinate_min=min(flattened),
        coordinate_max=max(flattened),
        centroid_norm=math.sqrt(sum(value * value for value in centroid)),
        radius_mean=sum(radii) / len(radii),
        radius_max=max(radii),
        adjacent_distance_count=len(adjacent_distances),
        adjacent_distance_mean=_mean(adjacent_distances),
        adjacent_distance_min=min(adjacent_distances) if adjacent_distances else None,
        adjacent_distance_max=max(adjacent_distances) if adjacent_distances else None,
        adjacent_distance_rms=_rms(adjacent_distances),
        pair_distance_count=len(pair_distances),
        pair_distance_mean=_mean(pair_distances),
        pair_distance_min=min(pair_distances) if pair_distances else None,
        pair_distance_max=max(pair_distances) if pair_distances else None,
        fraction_pair_distance_lt_4A=_fraction_below(pair_distances, 4.0),
        fraction_pair_distance_lt_8A=_fraction_below(pair_distances, 8.0),
        fraction_pair_distance_gt_40A=_fraction_above(pair_distances, 40.0),
    )


def _reference_delta(
    reference: ArtifactGeometrySummary,
    candidate: ArtifactGeometrySummary,
) -> ReferenceGeometryDelta:
    reference_targets = set(reference.target_summaries)
    candidate_targets = set(candidate.target_summaries)
    common = sorted(reference_targets & candidate_targets)
    radius_ratios: list[float] = []
    adjacent_deltas: list[float] = []
    pair_deltas: list[float] = []
    for target_id in common:
        reference_target = reference.target_summaries[target_id]
        candidate_target = candidate.target_summaries[target_id]
        if reference_target.radius_mean > 0.0:
            radius_ratios.append(candidate_target.radius_mean / reference_target.radius_mean)
        if reference_target.adjacent_distance_mean is not None and candidate_target.adjacent_distance_mean is not None:
            adjacent_deltas.append(candidate_target.adjacent_distance_mean - reference_target.adjacent_distance_mean)
        if reference_target.pair_distance_mean is not None and candidate_target.pair_distance_mean is not None:
            pair_deltas.append(candidate_target.pair_distance_mean - reference_target.pair_distance_mean)
    flags = list(candidate.scale_flags)
    mean_radius_ratio = _mean(radius_ratios)
    if mean_radius_ratio is not None and (mean_radius_ratio < 0.5 or mean_radius_ratio > 2.0):
        flags.append("reference_radius_scale_shift")
    mean_pair_delta = _mean(pair_deltas)
    if mean_pair_delta is not None and abs(mean_pair_delta) > 20.0:
        flags.append("reference_pair_distance_shift_gt_20A")
    return ReferenceGeometryDelta(
        reference_trial_id=reference.trial_id,
        candidate_trial_id=candidate.trial_id,
        same_split=reference.split == candidate.split,
        same_target_set=reference_targets == candidate_targets,
        common_target_count=len(common),
        mean_radius_scale_ratio=mean_radius_ratio,
        mean_adjacent_distance_delta=_mean(adjacent_deltas),
        mean_pair_distance_delta=mean_pair_delta,
        candidate_flags=sorted(set(flags)),
    )


def _recommendation(
    *,
    summaries: list[ArtifactGeometrySummary],
    reference_deltas: list[dict[str, object]],
) -> dict[str, object]:
    artifact_flags = sorted({flag for summary in summaries for flag in summary.scale_flags})
    reference_flags = sorted(
        {
            str(flag)
            for delta in reference_deltas
            for flag in delta.get("candidate_flags", [])
            if isinstance(delta.get("candidate_flags"), list)
        }
    )
    flags = sorted(set(artifact_flags + reference_flags))
    if flags:
        next_goal = (
            "Review prediction geometry scale before another live candidate; changed artifacts may still be "
            "outside plausible coordinate or distance regimes."
        )
    else:
        next_goal = (
            "Geometry scale audit did not flag obvious label-free coordinate pathologies; require scorer-backed "
            "offline review before approving another bounded live candidate."
        )
    return {
        "status": "REVIEW_REQUIRED" if flags else "NO_LABEL_FREE_SCALE_FLAGS",
        "flags": flags,
        "next_goal": next_goal,
        "stop_live_trial_budget": True,
        "do_not_start_open_ended_loop": True,
    }


def _scale_flags(
    *,
    target_summaries: list[TargetGeometrySummary],
    mean_adjacent_distance: float | None,
    max_adjacent_distance: float | None,
    mean_pair_distance: float | None,
    max_pair_distance: float | None,
) -> list[str]:
    flags: list[str] = []
    if any(summary.residue_count < 2 for summary in target_summaries):
        flags.append("single_residue_target")
    if mean_adjacent_distance is not None and mean_adjacent_distance < 1.0:
        flags.append("adjacent_ca_distance_collapsed")
    if mean_adjacent_distance is not None and mean_adjacent_distance > 10.0:
        flags.append("adjacent_ca_distance_exploded")
    if max_adjacent_distance is not None and max_adjacent_distance > 30.0:
        flags.append("adjacent_ca_distance_outlier_gt_30A")
    if mean_pair_distance is not None and mean_pair_distance < 2.0:
        flags.append("pair_distance_collapsed")
    if mean_pair_distance is not None and mean_pair_distance > 100.0:
        flags.append("pair_distance_exploded")
    if max_pair_distance is not None and max_pair_distance > 500.0:
        flags.append("pair_distance_outlier_gt_500A")
    return sorted(set(flags))


def _read_prediction_artifact(path: str | Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return validate_prediction_artifact(payload)
    except (OSError, json.JSONDecodeError, RunnerError) as exc:
        raise PredictionGeometryError(f"invalid prediction artifact {path}: {exc}") from exc


def _coordinates(prediction: dict[str, object]) -> list[list[float]]:
    coordinates = prediction["predicted_ca"]
    assert isinstance(coordinates, list)
    return [[float(coord) for coord in xyz] for xyz in coordinates if isinstance(xyz, list)]


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((right_coord - left_coord) ** 2 for left_coord, right_coord in zip(left, right, strict=True)))


def _fraction_below(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value < threshold) / len(values)


def _fraction_above(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value > threshold) / len(values)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _rms(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _present(values: Any) -> list[float]:
    return [float(value) for value in values if value is not None]


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None

