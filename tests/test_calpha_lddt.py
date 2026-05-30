from __future__ import annotations

import numpy as np
import pytest

from autoalphafold3.scorer.calpha_lddt import (
    SCORER_VERSION,
    aggregate_calpha_lddt,
    score_calpha_lddt,
)


def test_identical_structures_score_one() -> None:
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    result = score_calpha_lddt(target.copy(), target, np.ones(4, dtype=bool), target_id="toy_A")

    assert result.score == pytest.approx(1.0)
    assert result.eligible_pair_count == 6
    assert result.scorer_version == SCORER_VERSION
    assert all(value == pytest.approx(1.0) for value in result.threshold_fractions.values())


def test_perturbed_structure_scores_lower_than_identical() -> None:
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    predicted = target.copy()
    predicted[-1] = [12.0, 0.0, 0.0]

    identical = score_calpha_lddt(target, target, np.ones(4, dtype=bool))
    perturbed = score_calpha_lddt(predicted, target, np.ones(4, dtype=bool))

    assert perturbed.score < identical.score
    assert perturbed.score < 1.0
    assert perturbed.eligible_pair_count == identical.eligible_pair_count


def test_target_mask_excludes_unresolved_residues() -> None:
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    predicted = target.copy()
    predicted[-1] = [100.0, 100.0, 100.0]
    mask = np.array([True, True, True, False])

    result = score_calpha_lddt(predicted, target, mask)

    assert result.score == pytest.approx(1.0)
    assert result.eligible_pair_count == 3
    assert result.scored_residue_count == 3


def test_nan_predictions_do_not_crash_and_score_lower() -> None:
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    predicted = target.copy()
    predicted[1] = [np.nan, np.nan, np.nan]

    result = score_calpha_lddt(predicted, target, np.ones(4, dtype=bool))

    assert result.score < 1.0
    assert result.nan_prediction_residue_count == 1
    assert result.eligible_pair_count == 6


@pytest.mark.parametrize(
    ("predicted", "target", "mask"),
    [
        (np.zeros((4, 2)), np.zeros((4, 3)), np.ones(4, dtype=bool)),
        (np.zeros((5, 3)), np.zeros((4, 3)), np.ones(4, dtype=bool)),
        (np.zeros((4, 3)), np.zeros((4, 3)), np.ones((4, 1), dtype=bool)),
    ],
)
def test_invalid_shapes_are_rejected(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        score_calpha_lddt(predicted, target, mask)


def test_aggregate_uses_eligible_pair_weighting() -> None:
    target_short = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target_long = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    perturbed_short = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    low_weight_small = score_calpha_lddt(perturbed_short, target_short, np.ones(2, dtype=bool))
    high_weight_large = score_calpha_lddt(target_long, target_long, np.ones(4, dtype=bool))
    aggregate = aggregate_calpha_lddt([low_weight_small, high_weight_large])

    assert aggregate["metrics"]["eligible_pair_count"] == 7
    assert aggregate["metrics"]["best_val_calpha_lddt"] > aggregate["metrics"]["mean_val_calpha_lddt"]
