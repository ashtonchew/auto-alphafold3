from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.strategy_exhaustion_audit import (
    StrategyExhaustionAuditError,
    audit_strategy_exhaustion,
)
from autoalphafold3.surface_strategy_review import IMPLEMENTED_PLANNER_SURFACES


def test_strategy_exhaustion_audit_reports_no_remaining_implemented_planners(tmp_path: Path) -> None:
    _write_strategy_evidence(
        tmp_path,
        exhausted_surfaces=[
            next(iter(aliases))
            for aliases in IMPLEMENTED_PLANNER_SURFACES.values()
        ],
    )
    bench = _write_bench_readiness(tmp_path, can_start_open_ended_bench=False)

    report = audit_strategy_exhaustion(repo_root=tmp_path, bench_readiness_review=bench)

    assert report.schema_version == "autoaf3.strategy_exhaustion_audit.v1"
    assert report.decision == "NO_IMPLEMENTED_PLANNER_REMAINING"
    assert report.implemented_planner_count == len(IMPLEMENTED_PLANNER_SURFACES)
    assert report.remaining_implemented_planners == []
    assert sorted(report.exhausted_implemented_planners) == sorted(IMPLEMENTED_PLANNER_SURFACES)
    assert report.can_start_open_ended_bench is False
    assert "Keep the bench blocked" in report.required_next_step
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.writes_discovery_ledger is False
    assert report.official_benchmark_result is False


def test_strategy_exhaustion_audit_reports_remaining_implemented_planners(tmp_path: Path) -> None:
    _write_strategy_evidence(tmp_path, exhausted_surfaces=["auxiliary_loss", "feature_handling"])

    report = audit_strategy_exhaustion(repo_root=tmp_path)

    assert report.decision == "IMPLEMENTED_PLANNER_REMAINS_UNEXHAUSTED"
    assert "auxiliary_contact_loss_diagnostic" in report.exhausted_implemented_planners
    assert "feature_ref_pos_scale_diagnostic" in report.exhausted_implemented_planners
    assert "diffusion_initialization_scale_diagnostic" in report.remaining_implemented_planners
    assert report.can_start_open_ended_bench is None


def test_strategy_exhaustion_audit_refuses_authority_claims(tmp_path: Path) -> None:
    _write_strategy_evidence(tmp_path, exhausted_surfaces=["auxiliary_loss"], writes_ledger=True)

    with pytest.raises(StrategyExhaustionAuditError, match="writes_ledger"):
        audit_strategy_exhaustion(repo_root=tmp_path)


def test_strategy_exhaustion_audit_requires_safe_evidence_root(tmp_path: Path) -> None:
    with pytest.raises(StrategyExhaustionAuditError, match="runs/autoresearch"):
        audit_strategy_exhaustion(repo_root=tmp_path, evidence_root="runs")


def _write_strategy_evidence(
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
                "exhausted_surfaces": exhausted_surfaces,
                "unimplemented_candidate_surfaces": [],
                "starts_search": False,
                "writes_ledger": writes_ledger,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)


def _write_bench_readiness(tmp_path: Path, *, can_start_open_ended_bench: bool) -> Path:
    path = tmp_path / "runs/autoresearch/bench_readiness_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.bench_readiness_review.v1",
                "status": "PASS",
                "decision": "APPROVE_OPEN_ENDED_BENCH"
                if can_start_open_ended_bench
                else "BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED",
                "can_start_open_ended_bench": can_start_open_ended_bench,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)
