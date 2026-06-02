from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.artifact_comparison import ArtifactComparisonError, compare_prediction_artifacts
from autoalphafold3.runner import prediction_artifact_shape


def test_compare_prediction_artifacts_detects_identical_targets_and_metric_deltas(tmp_path: Path) -> None:
    left = _write_predictions(tmp_path / "left.json", trial_id="T150", y_offset=0.0)
    right = _write_predictions(tmp_path / "right.json", trial_id="T157", y_offset=0.0)
    left_metrics = _write_metrics(tmp_path / "left_metrics.json", score=0.10, mean=0.08)
    right_metrics = _write_metrics(tmp_path / "right_metrics.json", score=0.12, mean=0.09)

    report = compare_prediction_artifacts(
        left_predictions=left,
        right_predictions=right,
        left_metrics=left_metrics,
        right_metrics=right_metrics,
    )

    assert report.same_artifact_sha256 is False
    assert report.same_split is True
    assert report.same_target_set is True
    assert report.all_common_predictions_identical is True
    assert report.all_predictions_identical is True
    assert report.identical_targets == ["TARGET_A", "TARGET_B"]
    assert report.changed_targets == []
    assert report.metric_deltas == {
        "best_val_calpha_lddt": pytest.approx(0.02),
        "mean_val_calpha_lddt": pytest.approx(0.01),
        "num_scored_targets": pytest.approx(0.0),
    }
    assert report.left.trial_id == "T150"
    assert report.right.trial_id == "T157"


def test_compare_prediction_artifacts_detects_changed_and_missing_targets(tmp_path: Path) -> None:
    left = _write_predictions(tmp_path / "left.json", trial_id="T150", targets=("TARGET_A", "TARGET_B"))
    right = _write_predictions(
        tmp_path / "right.json",
        trial_id="T157",
        targets=("TARGET_A", "TARGET_C"),
        y_offset=1.0,
    )

    report = compare_prediction_artifacts(left_predictions=left, right_predictions=right)

    assert report.same_target_set is False
    assert report.all_common_predictions_identical is False
    assert report.all_predictions_identical is False
    assert report.common_target_count == 1
    assert report.left_only_targets == ["TARGET_B"]
    assert report.right_only_targets == ["TARGET_C"]
    assert report.identical_targets == []
    assert report.changed_targets == ["TARGET_A"]
    assert report.metric_deltas is None


def test_compare_prediction_artifacts_rejects_invalid_prediction_schema(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = _write_predictions(tmp_path / "right.json", trial_id="T157")
    left.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    with pytest.raises(ArtifactComparisonError, match="invalid prediction artifact"):
        compare_prediction_artifacts(left_predictions=left, right_predictions=right)


def _write_predictions(
    path: Path,
    *,
    trial_id: str,
    targets: tuple[str, ...] = ("TARGET_A", "TARGET_B"),
    y_offset: float = 0.0,
) -> Path:
    predictions = [
        {
            "target_id": target,
            "predicted_ca": [[0.0, y_offset + index, 0.0], [1.0, y_offset + index, 0.0]],
        }
        for index, target in enumerate(targets)
    ]
    payload = prediction_artifact_shape(
        trial_id=trial_id,
        split="public_val_small",
        predictions=predictions,
        source="comparison_fixture",
    )
    payload["candidate_id"] = trial_id
    payload["max_templates"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_metrics(path: Path, *, score: float, mean: float) -> Path:
    payload = {
        "schema_version": "autoaf3.metrics.v1",
        "metrics": {
            "best_val_calpha_lddt": score,
            "mean_val_calpha_lddt": mean,
            "num_scored_targets": 2,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
