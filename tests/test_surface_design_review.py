from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.surface_design_review import SurfaceDesignReviewError, review_surface_design


def test_surface_design_approves_pairformer_after_blocked_strategy(tmp_path: Path) -> None:
    strategy = _write_strategy_review(
        tmp_path,
        unimplemented_candidate_surfaces=["pairformer_attention", "auxiliary_loss"],
        exhausted_surfaces=["sampler_low_noise", "diffusion_data_scale"],
    )

    report = review_surface_design(
        repo_root=tmp_path,
        strategy_review=strategy,
        proposed_surface="pairformer_attention",
    )

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.surface_design_review.v1"
    assert payload["decision"] == "APPROVE_DRY_RUN_PLANNER_IMPLEMENTATION_ONLY"
    assert payload["approved_next_surface"] == "pairformer_attention"
    assert payload["approved_planner"] == "pairformer_attention_diagnostic"
    assert payload["candidate_limit"] == 1
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["required_next_pr"]["planner"] == "pairformer_attention_diagnostic"


def test_surface_design_approves_auxiliary_loss_after_blocked_strategy(tmp_path: Path) -> None:
    strategy = _write_strategy_review(
        tmp_path,
        unimplemented_candidate_surfaces=["auxiliary_loss", "feature_handling"],
        exhausted_surfaces=["pairformer_attention", "diffusion_data_scale"],
    )

    report = review_surface_design(
        repo_root=tmp_path,
        strategy_review=strategy,
        proposed_surface="auxiliary_loss",
    )

    payload = report.to_dict()
    assert payload["decision"] == "APPROVE_DRY_RUN_PLANNER_IMPLEMENTATION_ONLY"
    assert payload["approved_next_surface"] == "auxiliary_loss"
    assert payload["approved_planner"] == "auxiliary_contact_loss_diagnostic"
    assert payload["candidate_limit"] == 1
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert payload["required_next_pr"]["planner"] == "auxiliary_contact_loss_diagnostic"


def test_surface_design_approves_feature_handling_after_blocked_strategy(tmp_path: Path) -> None:
    strategy = _write_strategy_review(
        tmp_path,
        unimplemented_candidate_surfaces=["feature_handling", "memory_runtime"],
        exhausted_surfaces=["auxiliary_loss", "pairformer_attention"],
    )

    report = review_surface_design(
        repo_root=tmp_path,
        strategy_review=strategy,
        proposed_surface="feature_handling",
    )

    payload = report.to_dict()
    assert payload["decision"] == "APPROVE_DRY_RUN_PLANNER_IMPLEMENTATION_ONLY"
    assert payload["approved_next_surface"] == "feature_handling"
    assert payload["approved_planner"] == "feature_ref_pos_scale_diagnostic"
    assert payload["candidate_limit"] == 1
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert payload["required_next_pr"]["planner"] == "feature_ref_pos_scale_diagnostic"


def test_surface_design_refuses_exhausted_or_unlisted_surface(tmp_path: Path) -> None:
    exhausted = _write_strategy_review(
        tmp_path,
        unimplemented_candidate_surfaces=["pairformer_attention"],
        exhausted_surfaces=["pairformer_attention"],
    )
    with pytest.raises(SurfaceDesignReviewError, match="already exhausted"):
        review_surface_design(
            repo_root=tmp_path,
            strategy_review=exhausted,
            proposed_surface="pairformer_attention",
        )

    unlisted = _write_strategy_review(
        tmp_path,
        name="unlisted.json",
        unimplemented_candidate_surfaces=["auxiliary_loss"],
        exhausted_surfaces=[],
    )
    with pytest.raises(SurfaceDesignReviewError, match="unimplemented and available"):
        review_surface_design(
            repo_root=tmp_path,
            strategy_review=unlisted,
            proposed_surface="pairformer_attention",
        )

    exhausted_alias = _write_strategy_review(
        tmp_path,
        name="exhausted-alias.json",
        unimplemented_candidate_surfaces=["auxiliary_loss"],
        exhausted_surfaces=["auxiliary_contact_loss"],
    )
    with pytest.raises(SurfaceDesignReviewError, match="already exhausted"):
        review_surface_design(
            repo_root=tmp_path,
            strategy_review=exhausted_alias,
            proposed_surface="auxiliary_loss",
        )

    exhausted_feature_alias = _write_strategy_review(
        tmp_path,
        name="exhausted-feature-alias.json",
        unimplemented_candidate_surfaces=["feature_handling"],
        exhausted_surfaces=["ref_pos_scale"],
    )
    with pytest.raises(SurfaceDesignReviewError, match="already exhausted"):
        review_surface_design(
            repo_root=tmp_path,
            strategy_review=exhausted_feature_alias,
            proposed_surface="feature_handling",
        )


def test_surface_design_refuses_live_authority_claims(tmp_path: Path) -> None:
    strategy = _write_strategy_review(
        tmp_path,
        unimplemented_candidate_surfaces=["pairformer_attention"],
        exhausted_surfaces=[],
        writes_ledger=True,
    )

    with pytest.raises(SurfaceDesignReviewError, match="writes_ledger"):
        review_surface_design(
            repo_root=tmp_path,
            strategy_review=strategy,
            proposed_surface="pairformer_attention",
        )


def _write_strategy_review(
    tmp_path: Path,
    *,
    unimplemented_candidate_surfaces: list[str],
    exhausted_surfaces: list[str],
    name: str = "strategy.json",
    writes_ledger: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/surface_strategy_review" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.surface_strategy_review.v1",
                "status": "PASS",
                "decision": "NO_NON_OVERLAPPING_PLANNER_APPROVED",
                "approved_next_surface": None,
                "approved_planner": None,
                "candidate_limit": 0,
                "may_start_live_candidate": False,
                "may_start_open_ended_loop": False,
                "bench_blocked_reason": "latest next-surface review did not approve a non-overlapping planner",
                "consumed_next_surface_reviews": ["runs/autoresearch/next_surface_review/T171.json"],
                "consumed_diagnoses": ["runs/autoresearch/post_discard_diagnosis/T171.json"],
                "exhausted_surfaces": exhausted_surfaces,
                "unimplemented_candidate_surfaces": unimplemented_candidate_surfaces,
                "required_next_step": "Do offline design review.",
                "starts_search": False,
                "writes_ledger": writes_ledger,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)
