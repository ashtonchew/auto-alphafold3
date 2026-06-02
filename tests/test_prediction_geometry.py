from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.prediction_geometry import PredictionGeometryError, audit_prediction_geometry
from autoalphafold3.runner import prediction_artifact_shape


def test_prediction_geometry_audit_reports_reference_scale_shift(tmp_path: Path) -> None:
    reference = _write_predictions(tmp_path / "reference.json", trial_id="T088", scale=3.8)
    candidate = _write_predictions(tmp_path / "candidate.json", trial_id="T165", scale=40.0)

    report = audit_prediction_geometry(predictions=[candidate], reference_predictions=reference)

    assert report["schema_version"] == "autoaf3.prediction_geometry_audit.v1"
    assert report["official_benchmark_result"] is False
    assert report["starts_search"] is False
    assert report["writes_ledger"] is False
    assert report["writes_discovery_ledger"] is False
    assert report["reference"]["trial_id"] == "T088"
    candidate_summary = report["artifacts"][0]
    assert candidate_summary["trial_id"] == "T165"
    assert candidate_summary["target_count"] == 2
    assert candidate_summary["mean_adjacent_distance"] == pytest.approx(40.0)
    assert "adjacent_ca_distance_exploded" in candidate_summary["scale_flags"]
    delta = report["reference_deltas"][0]
    assert delta["reference_trial_id"] == "T088"
    assert delta["candidate_trial_id"] == "T165"
    assert delta["same_target_set"] is True
    assert delta["mean_radius_scale_ratio"] == pytest.approx(40.0 / 3.8)
    assert "reference_radius_scale_shift" in delta["candidate_flags"]
    assert report["recommendation"]["stop_live_trial_budget"] is True
    assert report["recommendation"]["do_not_start_open_ended_loop"] is True


def test_prediction_geometry_audit_allows_label_free_no_flag_case(tmp_path: Path) -> None:
    reference = _write_predictions(tmp_path / "reference.json", trial_id="T088", scale=3.8)
    candidate = _write_predictions(tmp_path / "candidate.json", trial_id="T166", scale=4.0)

    report = audit_prediction_geometry(predictions=[candidate], reference_predictions=reference)

    assert report["artifacts"][0]["scale_flags"] == []
    assert report["reference_deltas"][0]["candidate_flags"] == []
    assert report["recommendation"]["status"] == "NO_LABEL_FREE_SCALE_FLAGS"
    assert report["recommendation"]["stop_live_trial_budget"] is True


def test_prediction_geometry_audit_rejects_invalid_prediction_schema(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    with pytest.raises(PredictionGeometryError, match="invalid prediction artifact"):
        audit_prediction_geometry(predictions=[invalid])


def _write_predictions(path: Path, *, trial_id: str, scale: float) -> Path:
    predictions = [
        {
            "target_id": target,
            "predicted_ca": [[0.0, float(index), 0.0], [scale, float(index), 0.0], [scale * 2.0, float(index), 0.0]],
        }
        for index, target in enumerate(("TARGET_A", "TARGET_B"))
    ]
    payload = prediction_artifact_shape(
        trial_id=trial_id,
        split="public_val_small",
        predictions=predictions,
        source="prediction_geometry_fixture",
    )
    payload["candidate_id"] = trial_id
    payload["max_templates"] = 0
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
