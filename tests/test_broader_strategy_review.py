from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.broader_strategy_review import BroaderStrategyReviewError, review_broader_strategy


def test_broader_strategy_approves_one_non_overlapping_dry_run_planner(tmp_path: Path) -> None:
    surface_strategy = _write_surface_strategy(
        tmp_path,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
    )
    bench_readiness = _write_bench_readiness(
        tmp_path,
        decision="BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED",
        autonomous_search_ready=True,
    )

    report = review_broader_strategy(
        repo_root=tmp_path,
        surface_strategy_review=surface_strategy,
        bench_readiness_review=bench_readiness,
    )

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.broader_strategy_review.v1"
    assert payload["decision"] == "APPROVE_DRY_RUN_PLANNER_PR_ONLY"
    assert payload["approved_next_surface"] == "diffusion_initialization_scale"
    assert payload["approved_planner"] == "diffusion_initialization_scale_diagnostic"
    assert payload["candidate_limit"] == 1
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert "model-internal diffusion initial state scale" in str(payload["non_overlap_rationale"])
    assert payload["blocked_reason"] is None
    assert "scorer" in payload["forbidden_edits"]
    assert "post-merge readiness is not green" in payload["stop_conditions"]
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False


def test_broader_strategy_keeps_blocked_when_readiness_is_not_green(tmp_path: Path) -> None:
    surface_strategy = _write_surface_strategy(
        tmp_path,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
    )
    bench_readiness = _write_bench_readiness(
        tmp_path,
        decision="BLOCK_OPEN_ENDED_BENCH_READINESS_NOT_GREEN",
        autonomous_search_ready=False,
    )

    report = review_broader_strategy(
        repo_root=tmp_path,
        surface_strategy_review=surface_strategy,
        bench_readiness_review=bench_readiness,
    )

    assert report.decision == "NO_BROADER_STRATEGY_APPROVED"
    assert report.candidate_limit == 0
    assert report.approved_planner is None
    assert "Foundation readiness is not green" in str(report.blocked_reason)


def test_broader_strategy_no_go_after_diffusion_initialization_is_exhausted(tmp_path: Path) -> None:
    surface_strategy = _write_surface_strategy(
        tmp_path,
        exhausted_surfaces=[
            "auxiliary_loss",
            "feature_handling",
            "memory_runtime",
            "diffusion_initialization_scale",
        ],
    )
    bench_readiness = _write_bench_readiness(
        tmp_path,
        decision="BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED",
        autonomous_search_ready=True,
    )

    report = review_broader_strategy(
        repo_root=tmp_path,
        surface_strategy_review=surface_strategy,
        bench_readiness_review=bench_readiness,
    )

    assert report.decision == "NO_BROADER_STRATEGY_APPROVED"
    assert report.approved_next_surface is None
    assert report.approved_planner is None
    assert report.candidate_limit == 0
    assert report.may_start_live_candidate is False
    assert report.may_start_open_ended_loop is False
    assert "diffusion_initialization_scale" in report.exhausted_surfaces
    assert "already exhausted" in str(report.blocked_reason)
    assert "Reapproving it would repeat a discarded surface" in str(report.blocked_reason)
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.writes_discovery_ledger is False
    assert report.official_benchmark_result is False


def test_broader_strategy_refuses_authority_claims(tmp_path: Path) -> None:
    surface_strategy = _write_surface_strategy(
        tmp_path,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        writes_ledger=True,
    )
    bench_readiness = _write_bench_readiness(
        tmp_path,
        decision="BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED",
        autonomous_search_ready=True,
    )

    with pytest.raises(BroaderStrategyReviewError, match="writes_ledger"):
        review_broader_strategy(
            repo_root=tmp_path,
            surface_strategy_review=surface_strategy,
            bench_readiness_review=bench_readiness,
        )


def test_broader_strategy_requires_repo_relative_autoresearch_paths(tmp_path: Path) -> None:
    bench_readiness = _write_bench_readiness(
        tmp_path,
        decision="BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED",
        autonomous_search_ready=True,
    )

    with pytest.raises(BroaderStrategyReviewError, match="repo-relative"):
        review_broader_strategy(
            repo_root=tmp_path,
            surface_strategy_review=Path("/tmp/review.json"),
            bench_readiness_review=bench_readiness,
        )


def _write_surface_strategy(
    tmp_path: Path,
    *,
    exhausted_surfaces: list[str],
    writes_ledger: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/surface_strategy_review/review.json"
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
                "bench_blocked_reason": "all planned surfaces exhausted",
                "consumed_next_surface_reviews": [],
                "consumed_diagnoses": [],
                "exhausted_surfaces": exhausted_surfaces,
                "unimplemented_candidate_surfaces": [],
                "required_next_step": "broader review",
                "starts_search": False,
                "writes_ledger": writes_ledger,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)


def _write_bench_readiness(
    tmp_path: Path,
    *,
    decision: str,
    autonomous_search_ready: bool,
) -> Path:
    path = tmp_path / "runs/autoresearch/bench_readiness_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.bench_readiness_review.v1",
                "status": "PASS",
                "decision": decision,
                "can_start_open_ended_bench": False,
                "autonomous_search_ready": autonomous_search_ready,
                "surface_strategy_decision": "NO_NON_OVERLAPPING_PLANNER_APPROVED",
                "may_start_live_candidate": False,
                "may_start_open_ended_loop": False,
                "exhausted_surfaces": ["auxiliary_loss", "feature_handling", "memory_runtime"],
                "unimplemented_candidate_surfaces": [],
                "required_objectives": [],
                "roadmap": [],
                "evidence": {},
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)
