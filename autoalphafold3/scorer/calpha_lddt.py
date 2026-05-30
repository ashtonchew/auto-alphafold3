"""C-alpha lDDT scorer for the locked AlphaFold3-lite benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable

import numpy as np

SCORER_VERSION = "calpha_lddt_v1"
DEFAULT_DISTANCE_CUTOFF = 15.0
DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class CalphaLddtResult:
    """Result for one target scored with C-alpha lDDT."""

    score: float
    eligible_pair_count: int
    scored_residue_count: int
    nan_prediction_residue_count: int
    threshold_fractions: dict[str, float]
    target_id: str | None = None
    scorer_version: str = SCORER_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def score_calpha_lddt(
    predicted_ca: np.ndarray,
    target_ca: np.ndarray,
    target_mask: np.ndarray,
    *,
    target_id: str | None = None,
    distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> CalphaLddtResult:
    """Score one target with superposition-free C-alpha lDDT.

    NaN or infinite predictions are treated as non-preserved pairs. NaN or
    infinite target coordinates are rejected because labels are locked assets.
    """

    predicted = _as_coordinate_array(predicted_ca, "predicted_ca")
    target = _as_coordinate_array(target_ca, "target_ca")
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted_ca and target_ca must have the same shape; "
            f"got {predicted.shape} and {target.shape}"
        )

    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != (target.shape[0],):
        raise ValueError(f"target_mask must have shape ({target.shape[0]},); got {mask.shape}")

    threshold_values = tuple(float(threshold) for threshold in thresholds)
    if not threshold_values:
        raise ValueError("thresholds must contain at least one value")
    if any(threshold <= 0.0 for threshold in threshold_values):
        raise ValueError("thresholds must all be positive")
    if distance_cutoff <= 0.0:
        raise ValueError("distance_cutoff must be positive")
    if not np.isfinite(target).all():
        raise ValueError("target_ca must contain only finite coordinates")

    scored_residue_count = int(mask.sum())
    nan_prediction_residue_count = int((~np.isfinite(predicted).all(axis=1) & mask).sum())
    if scored_residue_count < 2:
        return _empty_result(
            target_id,
            scored_residue_count,
            nan_prediction_residue_count,
            threshold_values,
        )

    true_distances = _pairwise_distances(target)
    pred_distances = _pairwise_distances(predicted)

    upper_triangle = np.triu(np.ones(true_distances.shape, dtype=bool), k=1)
    resolved_pairs = mask[:, None] & mask[None, :]
    eligible_pairs = upper_triangle & resolved_pairs & (true_distances < distance_cutoff)
    eligible_pair_count = int(eligible_pairs.sum())

    if eligible_pair_count == 0:
        return _empty_result(
            target_id,
            scored_residue_count,
            nan_prediction_residue_count,
            threshold_values,
        )

    distance_errors = np.abs(pred_distances[eligible_pairs] - true_distances[eligible_pairs])
    threshold_fractions = {
        _threshold_key(threshold): float(np.mean(distance_errors < threshold))
        for threshold in threshold_values
    }
    score = float(np.mean(tuple(threshold_fractions.values())))

    return CalphaLddtResult(
        score=score,
        eligible_pair_count=eligible_pair_count,
        scored_residue_count=scored_residue_count,
        nan_prediction_residue_count=nan_prediction_residue_count,
        threshold_fractions=threshold_fractions,
        target_id=target_id,
    )


def aggregate_calpha_lddt(results: Iterable[CalphaLddtResult]) -> dict[str, object]:
    """Aggregate per-target C-alpha lDDT results into canonical metric fields."""

    result_list = list(results)
    total_pairs = sum(result.eligible_pair_count for result in result_list)
    scored = [result for result in result_list if result.eligible_pair_count > 0]
    scores = [result.score for result in scored]

    weighted_score = (
        sum(result.score * result.eligible_pair_count for result in scored) / total_pairs
        if total_pairs
        else 0.0
    )

    return {
        "schema_version": "autoaf3.metrics.v1",
        "scorer_version": SCORER_VERSION,
        "primary_metric": "best_val_calpha_lddt",
        "metrics": {
            "best_val_calpha_lddt": float(weighted_score),
            "mean_val_calpha_lddt": float(np.mean(scores)) if scores else 0.0,
            "median_val_calpha_lddt": float(median(scores)) if scores else 0.0,
            "eligible_pair_count": int(total_pairs),
            "num_targets": len(result_list),
            "num_scored_targets": len(scored),
            "num_failed_targets": len(result_list) - len(scored),
        },
    }


def _as_coordinate_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (L, 3); got {array.shape}")
    return array


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(deltas, axis=-1)


def _threshold_key(threshold: float) -> str:
    return f"lt_{threshold:g}A"


def _empty_result(
    target_id: str | None,
    scored_residue_count: int,
    nan_prediction_residue_count: int,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> CalphaLddtResult:
    return CalphaLddtResult(
        score=0.0,
        eligible_pair_count=0,
        scored_residue_count=scored_residue_count,
        nan_prediction_residue_count=nan_prediction_residue_count,
        threshold_fractions={_threshold_key(threshold): 0.0 for threshold in thresholds},
        target_id=target_id,
    )
