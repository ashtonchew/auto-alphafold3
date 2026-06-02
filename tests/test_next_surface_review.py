from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.next_surface_review import NextSurfaceReviewError, review_next_surface


def test_next_surface_review_approves_coordinate_scale_locality_pr_only(tmp_path: Path) -> None:
    diagnosis = _write_diagnosis(tmp_path)

    report = review_next_surface(repo_root=tmp_path, diagnosis_path=diagnosis)

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.next_surface_review.v1"
    assert payload["status"] == "PASS"
    assert payload["source_verdict"] == "MIXED_EVIDENCE_REVIEW_REQUIRED"
    assert payload["decision"] == "APPROVE_OFFLINE_PLANNER_PR_ONLY"
    assert payload["approved_next_surface"] == "coordinate_scale_locality_diagnostic"
    assert payload["rejected_surfaces"] == [
        "sampler",
        "local_geometry",
        "optimizer_schedule",
        "width_depth",
        "recycling",
        "feature_curriculum",
    ]
    assert payload["required_next_pr"]["planner"] == "coordinate_scale_locality_diagnostic"
    assert payload["required_next_pr"]["candidate_limit"] == 1
    assert payload["required_next_pr"]["mode_before_merge"] == "dry-run"
    assert payload["required_next_pr"]["must_consume_review"] is True
    assert payload["evidence_summary"]["all_candidate_per_target_deltas_negative"] is True
    assert payload["evidence_summary"]["all_comparisons_changed"] is True
    assert payload["stop_live_trial_budget"] is True
    assert payload["do_not_start_open_ended_loop"] is True
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False


def test_next_surface_review_refuses_authority_claims(tmp_path: Path) -> None:
    diagnosis = _write_diagnosis(tmp_path, writes_ledger=True)

    with pytest.raises(NextSurfaceReviewError, match="writes_ledger"):
        review_next_surface(repo_root=tmp_path, diagnosis_path=diagnosis)


def test_next_surface_review_does_not_approve_without_feature_curriculum_evidence(tmp_path: Path) -> None:
    diagnosis = _write_diagnosis(tmp_path, exhausted_surfaces=["sampler", "recycling"])

    report = review_next_surface(repo_root=tmp_path, diagnosis_path=diagnosis)

    assert report.decision == "NO_NEXT_SURFACE_APPROVED"
    assert report.approved_next_surface is None
    assert report.required_next_pr["candidate_limit"] == 0


def _write_diagnosis(
    tmp_path: Path,
    *,
    writes_ledger: bool = False,
    exhausted_surfaces: list[str] | None = None,
) -> Path:
    path = tmp_path / "runs/autoresearch/post_discard_diagnosis/T113-T162-T163-T164.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "autoaf3.post_discard_diagnosis.v1",
        "status": "PASS",
        "verdict": "MIXED_EVIDENCE_REVIEW_REQUIRED",
        "reference_trial_id": "T088",
        "candidate_trial_ids": ["T113", "T162", "T163", "T164"],
        "exhausted_surfaces": exhausted_surfaces
        if exhausted_surfaces is not None
        else [
            "sampler",
            "local_geometry",
            "optimizer_schedule",
            "width_depth",
            "recycling",
            "feature_curriculum",
        ],
        "score_summary": {
            "primary_metric": "best_val_calpha_lddt",
            "candidate_scores": {
                "T113": 0.008276756926787072,
                "T162": 0.008276756926787072,
                "T163": 0.008276756926787072,
                "T164": 0.009960942619727908,
            },
            "candidate_scores_identical": False,
            "all_candidate_per_target_deltas_negative": True,
            "per_target_delta_summary": {
                "candidate_delta_sets": 4,
                "negative_delta_count": 64,
                "positive_delta_count": 0,
                "target_count": 16,
                "worst_target": "1MBD_A",
                "worst_delta": -0.0213496566139146,
            },
        },
        "artifact_summary": {
            "comparison_count": 4,
            "all_comparisons_changed": True,
            "any_all_predictions_identical": False,
        },
        "recommendation": {
            "stop_live_trial_budget": True,
            "do_not_start_open_ended_loop": True,
        },
        "starts_search": False,
        "writes_ledger": writes_ledger,
        "writes_discovery_ledger": False,
        "official_benchmark_result": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.relative_to(tmp_path)
