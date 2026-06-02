from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.surface_strategy_review import SurfaceStrategyReviewError, review_surface_strategy


def test_surface_strategy_blocks_after_t171_no_surface_approved(tmp_path: Path) -> None:
    diagnosis = _write_diagnosis(
        tmp_path,
        exhausted_surfaces=[
            "sampler_coordinate_scale",
            "sampler_geometry_selection",
            "sampler_low_noise",
            "diffusion_data_scale",
        ],
    )
    review = _write_next_surface_review(
        tmp_path,
        decision="NO_NEXT_SURFACE_APPROVED",
        approved_next_surface=None,
        planner=None,
        candidate_limit=0,
        rejected_surfaces=[
            "sampler_coordinate_scale",
            "sampler_geometry_selection",
            "sampler_low_noise",
            "diffusion_data_scale",
        ],
    )

    report = review_surface_strategy(
        repo_root=tmp_path,
        next_surface_reviews=[review],
        diagnoses=[diagnosis],
    )

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.surface_strategy_review.v1"
    assert payload["decision"] == "NO_NON_OVERLAPPING_PLANNER_APPROVED"
    assert payload["approved_next_surface"] is None
    assert payload["approved_planner"] is None
    assert payload["candidate_limit"] == 0
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert "diffusion_data_scale" in payload["exhausted_surfaces"]
    assert "feature_handling" in payload["unimplemented_candidate_surfaces"]
    assert "pairformer_attention" not in payload["unimplemented_candidate_surfaces"]
    assert "auxiliary_loss" not in payload["unimplemented_candidate_surfaces"]
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False


def test_surface_strategy_allows_unexhausted_approved_offline_planner(tmp_path: Path) -> None:
    review = _write_next_surface_review(
        tmp_path,
        decision="APPROVE_OFFLINE_PLANNER_PR_ONLY",
        approved_next_surface="diffusion_data_scale_diagnostic",
        planner="diffusion_data_scale_diagnostic",
        candidate_limit=1,
        rejected_surfaces=[
            "sampler_coordinate_scale",
            "sampler_geometry_selection",
            "sampler_low_noise",
        ],
    )

    report = review_surface_strategy(repo_root=tmp_path, next_surface_reviews=[review])

    assert report.decision == "APPROVE_OFFLINE_PLANNER_PR_ONLY"
    assert report.approved_next_surface == "diffusion_data_scale_diagnostic"
    assert report.approved_planner == "diffusion_data_scale_diagnostic"
    assert report.candidate_limit == 1
    assert report.may_start_live_candidate is False
    assert report.may_start_open_ended_loop is False


def test_surface_strategy_refuses_exhausted_approved_surface(tmp_path: Path) -> None:
    diagnosis = _write_diagnosis(tmp_path, exhausted_surfaces=["diffusion_data_scale"])
    review = _write_next_surface_review(
        tmp_path,
        decision="APPROVE_OFFLINE_PLANNER_PR_ONLY",
        approved_next_surface="diffusion_data_scale_diagnostic",
        planner="diffusion_data_scale_diagnostic",
        candidate_limit=1,
        rejected_surfaces=["diffusion_data_scale"],
    )

    report = review_surface_strategy(repo_root=tmp_path, next_surface_reviews=[review], diagnoses=[diagnosis])

    assert report.decision == "NO_NON_OVERLAPPING_PLANNER_APPROVED"
    assert report.approved_next_surface is None
    assert report.candidate_limit == 0
    assert "already exhausted" in str(report.bench_blocked_reason)


def test_surface_strategy_refuses_authority_claims(tmp_path: Path) -> None:
    review = _write_next_surface_review(
        tmp_path,
        decision="NO_NEXT_SURFACE_APPROVED",
        approved_next_surface=None,
        planner=None,
        candidate_limit=0,
        rejected_surfaces=[],
        writes_ledger=True,
    )

    with pytest.raises(SurfaceStrategyReviewError, match="writes_ledger"):
        review_surface_strategy(repo_root=tmp_path, next_surface_reviews=[review])


def _write_next_surface_review(
    tmp_path: Path,
    *,
    decision: str,
    approved_next_surface: str | None,
    planner: str | None,
    candidate_limit: int,
    rejected_surfaces: list[str],
    writes_ledger: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/next_surface_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.next_surface_review.v1",
                "status": "PASS",
                "source_diagnosis": "runs/autoresearch/post_discard_diagnosis/diagnosis.json",
                "source_verdict": "MIXED_EVIDENCE_REVIEW_REQUIRED",
                "decision": decision,
                "approved_next_surface": approved_next_surface,
                "rejected_surfaces": rejected_surfaces,
                "evidence_summary": {},
                "required_next_pr": {
                    "planner": planner,
                    "candidate_limit": candidate_limit,
                },
                "allowed_next_step": "test",
                "stop_live_trial_budget": True,
                "do_not_start_open_ended_loop": True,
                "starts_search": False,
                "writes_ledger": writes_ledger,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)


def _write_diagnosis(tmp_path: Path, *, exhausted_surfaces: list[str]) -> Path:
    path = tmp_path / "runs/autoresearch/post_discard_diagnosis/diagnosis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.post_discard_diagnosis.v1",
                "status": "PASS",
                "verdict": "MIXED_EVIDENCE_REVIEW_REQUIRED",
                "reference_trial_id": "T088",
                "candidate_trial_ids": ["T168", "T169", "T170", "T171"],
                "exhausted_surfaces": exhausted_surfaces,
                "score_summary": {},
                "artifact_summary": {},
                "recommendation": {
                    "stop_live_trial_budget": True,
                    "do_not_start_open_ended_loop": True,
                },
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)
