from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.post_exhaustion_strategy import (
    PostExhaustionStrategyError,
    design_post_exhaustion_strategy,
)


def test_post_exhaustion_strategy_approves_dry_run_prd_only(tmp_path: Path) -> None:
    bench = _write_bench_readiness(tmp_path, autonomous_search_ready=True)
    audit = _write_strategy_exhaustion(tmp_path, remaining_implemented_planners=[])

    report = design_post_exhaustion_strategy(
        repo_root=tmp_path,
        bench_readiness_review=bench,
        strategy_exhaustion_audit=audit,
    )
    payload = report.to_dict()

    assert payload["schema_version"] == "autoaf3.post_exhaustion_strategy_prd.v1"
    assert payload["decision"] == "APPROVE_DRY_RUN_STRATEGY_PRD_ONLY"
    assert payload["approved_strategy_family"] == "evidence_guided_failure_mode_bridge"
    assert payload["approved_planner"] == "evidence_guided_failure_mode_bridge_diagnostic"
    assert payload["candidate_limit"] == 1
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert "pre-live evidence bridge" in str(payload["non_overlap_rationale"])
    assert "autoalphafold3/autoresearch_loop.py" in payload["allowed_edit_areas"][0]
    assert "scorer" in payload["forbidden_edits"]
    assert payload["dry_run_candidate_shape"]["candidate_limit"] == 1
    assert "target_level_non_regression_expectations" in payload["dry_run_candidate_shape"]["must_emit"]
    assert "dry-run planner consumes real local scorer-sensitivity evidence" in payload["required_evidence_before_live_smoke"]
    assert "post-merge full local gate is not green" in payload["stop_conditions"]
    assert "planning evidence only" in payload["ui_reporting_constraint"]
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False


def test_post_exhaustion_strategy_blocks_when_readiness_is_not_green(tmp_path: Path) -> None:
    bench = _write_bench_readiness(tmp_path, autonomous_search_ready=False)
    audit = _write_strategy_exhaustion(tmp_path, remaining_implemented_planners=[])

    report = design_post_exhaustion_strategy(
        repo_root=tmp_path,
        bench_readiness_review=bench,
        strategy_exhaustion_audit=audit,
    )

    assert report.decision == "NO_POST_EXHAUSTION_STRATEGY_APPROVED"
    assert report.candidate_limit == 0
    assert report.approved_planner is None
    assert "Foundation readiness is not green" in str(report.blocked_reason)


def test_post_exhaustion_strategy_blocks_when_planner_remains(tmp_path: Path) -> None:
    bench = _write_bench_readiness(tmp_path, autonomous_search_ready=True)
    audit = _write_strategy_exhaustion(
        tmp_path,
        remaining_implemented_planners=["diffusion_initialization_scale_diagnostic"],
    )

    report = design_post_exhaustion_strategy(
        repo_root=tmp_path,
        bench_readiness_review=bench,
        strategy_exhaustion_audit=audit,
    )

    assert report.decision == "NO_POST_EXHAUSTION_STRATEGY_APPROVED"
    assert report.candidate_limit == 0
    assert "does not report all implemented planners exhausted" in str(report.blocked_reason)


def test_post_exhaustion_strategy_refuses_authority_claims(tmp_path: Path) -> None:
    bench = _write_bench_readiness(tmp_path, autonomous_search_ready=True, writes_ledger=True)
    audit = _write_strategy_exhaustion(tmp_path, remaining_implemented_planners=[])

    with pytest.raises(PostExhaustionStrategyError, match="writes_ledger"):
        design_post_exhaustion_strategy(
            repo_root=tmp_path,
            bench_readiness_review=bench,
            strategy_exhaustion_audit=audit,
        )


def test_post_exhaustion_strategy_requires_safe_evidence_paths(tmp_path: Path) -> None:
    audit = _write_strategy_exhaustion(tmp_path, remaining_implemented_planners=[])

    with pytest.raises(PostExhaustionStrategyError, match="repo-relative"):
        design_post_exhaustion_strategy(
            repo_root=tmp_path,
            bench_readiness_review=Path("/tmp/bench.json"),
            strategy_exhaustion_audit=audit,
        )


def _write_bench_readiness(
    tmp_path: Path,
    *,
    autonomous_search_ready: bool,
    writes_ledger: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/bench_readiness_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.bench_readiness_review.v1",
                "status": "PASS",
                "decision": "BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED",
                "can_start_open_ended_bench": False,
                "autonomous_search_ready": autonomous_search_ready,
                "starts_search": False,
                "writes_ledger": writes_ledger,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)


def _write_strategy_exhaustion(
    tmp_path: Path,
    *,
    remaining_implemented_planners: list[str],
) -> Path:
    path = tmp_path / "runs/autoresearch/strategy_exhaustion_audit/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    exhausted = ["targeted_diagnostic"] if not remaining_implemented_planners else []
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.strategy_exhaustion_audit.v1",
                "status": "PASS",
                "decision": "NO_IMPLEMENTED_PLANNER_REMAINING"
                if not remaining_implemented_planners
                else "IMPLEMENTED_PLANNER_REMAINS_UNEXHAUSTED",
                "implemented_planner_count": len(exhausted) + len(remaining_implemented_planners),
                "exhausted_implemented_planners": exhausted,
                "remaining_implemented_planners": remaining_implemented_planners,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)
