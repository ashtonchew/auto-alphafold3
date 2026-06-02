from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.post_discard_diagnosis import (
    PostDiscardDiagnosisError,
    diagnose_post_discard_evidence,
)


def test_post_discard_diagnosis_detects_short_training_family_collapse(tmp_path: Path) -> None:
    scorer_reports = [
        _write_scorer_report(tmp_path, candidate_trial_id="T113"),
        _write_scorer_report(tmp_path, candidate_trial_id="T162"),
        _write_scorer_report(tmp_path, candidate_trial_id="T163"),
    ]
    comparisons = [
        _write_prediction_comparison(tmp_path, left_trial_id="T088", right_trial_id="T113"),
        _write_prediction_comparison(tmp_path, left_trial_id="T088", right_trial_id="T162"),
        _write_prediction_comparison(tmp_path, left_trial_id="T088", right_trial_id="T163"),
    ]

    report = diagnose_post_discard_evidence(
        repo_root=tmp_path,
        scorer_reports=scorer_reports,
        prediction_comparisons=comparisons,
        exhausted_surfaces=["sampler", "width_depth", "recycling"],
    )

    payload = report.to_dict()
    assert payload["status"] == "PASS"
    assert payload["verdict"] == "SHORT_TRAINING_FAMILY_SCORER_COLLAPSE"
    assert payload["candidate_trial_ids"] == ["T113", "T162", "T163"]
    assert payload["score_summary"]["candidate_scores_identical"] is True
    assert payload["score_summary"]["all_candidate_per_target_deltas_negative"] is True
    assert payload["artifact_summary"]["all_comparisons_changed"] is True
    assert payload["artifact_summary"]["any_all_predictions_identical"] is False
    assert payload["recommendation"]["stop_live_trial_budget"] is True
    assert payload["recommendation"]["do_not_start_open_ended_loop"] is True
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False


def test_post_discard_diagnosis_detects_stale_prediction_artifacts(tmp_path: Path) -> None:
    scorer_report = _write_scorer_report(tmp_path, candidate_trial_id="T162")
    comparison = _write_prediction_comparison(
        tmp_path,
        left_trial_id="T113",
        right_trial_id="T162",
        all_predictions_identical=True,
    )

    report = diagnose_post_discard_evidence(
        repo_root=tmp_path,
        scorer_reports=[scorer_report],
        prediction_comparisons=[comparison],
    )

    assert report.verdict == "STALE_PREDICTION_ARTIFACTS"
    assert report.artifact_summary["any_all_predictions_identical"] is True
    assert report.recommendation["stop_live_trial_budget"] is True


def test_post_discard_diagnosis_refuses_reports_with_authority_claims(tmp_path: Path) -> None:
    scorer_report = _write_scorer_report(tmp_path, candidate_trial_id="T162", writes_ledger=True)
    comparison = _write_prediction_comparison(tmp_path, left_trial_id="T088", right_trial_id="T162")

    with pytest.raises(PostDiscardDiagnosisError, match="writes_ledger"):
        diagnose_post_discard_evidence(
            repo_root=tmp_path,
            scorer_reports=[scorer_report],
            prediction_comparisons=[comparison],
        )


def _write_scorer_report(
    tmp_path: Path,
    *,
    candidate_trial_id: str,
    writes_ledger: bool = False,
) -> Path:
    path = tmp_path / f"runs/autoresearch/scorer_sensitivity/T088-vs-{candidate_trial_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "autoaf3.scorer_sensitivity.v1",
        "status": "PASS",
        "mode": "modal",
        "starts_search": False,
        "writes_ledger": writes_ledger,
        "writes_discovery_ledger": False,
        "trial_ids": ["T088", candidate_trial_id],
        "reference_trial_id": "T088",
        "scored_trials": [
            {
                "trial_id": "T088",
                "status": "SCORED",
                "score": 0.02098351201866366,
                "metrics": {"best_val_calpha_lddt": 0.02098351201866366},
                "per_target_results": [],
                "fold_cartographer_signature": "toy_geometry_failed",
                "official_benchmark_result": True,
                "local_only": False,
                "artifacts": {},
            },
            {
                "trial_id": candidate_trial_id,
                "status": "SCORED",
                "score": 0.008276756926787072,
                "metrics": {"best_val_calpha_lddt": 0.008276756926787072},
                "per_target_results": [],
                "fold_cartographer_signature": "toy_geometry_failed",
                "official_benchmark_result": True,
                "local_only": False,
                "artifacts": {},
            },
        ],
        "metric_deltas_vs_reference": {
            candidate_trial_id: {"best_val_calpha_lddt": -0.012706755091876588}
        },
        "per_target_score_deltas_vs_reference": {
            candidate_trial_id: {
                "1MBD_A": -0.0213496566139146,
                "2BP2_A": -0.0188131785847997,
                "2LZM_A": -0.01565860215053763,
            }
        },
        "all_primary_scores_identical": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.relative_to(tmp_path)


def _write_prediction_comparison(
    tmp_path: Path,
    *,
    left_trial_id: str,
    right_trial_id: str,
    all_predictions_identical: bool = False,
) -> Path:
    path = tmp_path / f"runs/autoresearch/prediction_comparisons/{left_trial_id}-vs-{right_trial_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    changed_targets = [] if all_predictions_identical else ["1MBD_A", "2BP2_A", "2LZM_A"]
    identical_targets = ["1MBD_A", "2BP2_A", "2LZM_A"] if all_predictions_identical else []
    payload = {
        "schema_version": "autoaf3.prediction_artifact_comparison.v1",
        "left": {"trial_id": left_trial_id},
        "right": {"trial_id": right_trial_id},
        "same_artifact_sha256": all_predictions_identical,
        "same_split": True,
        "same_target_set": True,
        "common_target_count": 3,
        "left_only_targets": [],
        "right_only_targets": [],
        "identical_targets": identical_targets,
        "changed_targets": changed_targets,
        "all_common_predictions_identical": all_predictions_identical,
        "all_predictions_identical": all_predictions_identical,
        "coordinate_deltas": {},
        "coordinate_delta_summary": {},
        "distance_deltas": {},
        "distance_delta_summary": {"mean_target_mean_abs_pair_distance_delta": 1358.1},
        "metric_deltas": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.relative_to(tmp_path)
