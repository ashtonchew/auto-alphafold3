from __future__ import annotations

import json
from pathlib import Path

import pytest

import autoalphafold3.bench_readiness_review as bench_review
from autoalphafold3.bench_readiness_review import BenchReadinessReviewError, review_bench_readiness


class FakeReadiness:
    def __init__(self, *, ready: bool) -> None:
        self.autonomous_search_ready = ready

    def to_dict(self) -> dict[str, object]:
        return {
            "autonomous_search_ready": self.autonomous_search_ready,
            "mode": "offline",
            "problems": [] if self.autonomous_search_ready else ["readiness problem"],
            "pending_human_actions": [],
            "baseline_lock": {"status": "PASS" if self.autonomous_search_ready else "BLOCKED"},
            "local_gates": {"status": "PASS"},
            "modal_event_authority": {"status": "PASS"},
            "gate_calibration": {"status": "PASS"},
        }


def test_bench_readiness_blocks_when_strategy_surfaces_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_bench_readiness(repo_root=tmp_path, surface_strategy_review=strategy)

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.bench_readiness_review.v1"
    assert payload["decision"] == "BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED"
    assert payload["can_start_open_ended_bench"] is False
    assert payload["autonomous_search_ready"] is True
    assert payload["unimplemented_candidate_surfaces"] == []
    assert payload["broader_strategy_decision"] is None
    assert payload["approved_broader_planner"] is None
    assert payload["required_objectives"][0]["name"] == "broader_offline_strategy_review"
    assert payload["roadmap"][0]["step"] == "stop_live_spend"
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False


def test_bench_readiness_consumes_broader_strategy_without_opening_bench(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    broader_strategy = _write_broader_strategy(tmp_path)
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_bench_readiness(
        repo_root=tmp_path,
        surface_strategy_review=strategy,
        broader_strategy_review=broader_strategy,
    )

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_DRY_RUN_PLANNER_REQUIRED"
    assert report.can_start_open_ended_bench is False
    assert report.broader_strategy_decision == "APPROVE_DRY_RUN_PLANNER_PR_ONLY"
    assert report.approved_broader_surface == "diffusion_initialization_scale"
    assert report.approved_broader_planner == "diffusion_initialization_scale_diagnostic"
    assert report.required_objectives[0]["name"] == "implement_broader_dry_run_planner"
    assert report.roadmap[1]["step"] == "implement_broader_dry_run_planner"
    assert report.evidence["broader_strategy_review"] == str(broader_strategy)
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.official_benchmark_result is False


def test_bench_readiness_blocks_when_foundation_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=True,
        exhausted_surfaces=[],
        unimplemented_candidate_surfaces=[],
    )
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=False))

    report = review_bench_readiness(repo_root=tmp_path, surface_strategy_review=strategy)

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_READINESS_NOT_GREEN"
    assert report.can_start_open_ended_bench is False
    assert report.required_objectives[0]["name"] == "restore_foundation_readiness"


def test_bench_readiness_approves_only_when_readiness_and_strategy_approve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        decision="APPROVE_OPEN_ENDED_BENCH",
        may_start_open_ended_loop=True,
        exhausted_surfaces=[],
        unimplemented_candidate_surfaces=[],
    )
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_bench_readiness(repo_root=tmp_path, surface_strategy_review=strategy)

    assert report.decision == "APPROVE_OPEN_ENDED_BENCH"
    assert report.can_start_open_ended_bench is True
    assert report.required_objectives[0]["name"] == "start_open_ended_bench"
    assert report.roadmap[0]["step"] == "run_open_ended_bench"


def test_bench_readiness_refuses_authority_claims(tmp_path: Path) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=[],
        unimplemented_candidate_surfaces=[],
        writes_ledger=True,
    )

    with pytest.raises(BenchReadinessReviewError, match="writes_ledger"):
        review_bench_readiness(repo_root=tmp_path, surface_strategy_review=strategy)


def test_bench_readiness_refuses_live_authority_in_broader_strategy(tmp_path: Path) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    broader_strategy = _write_broader_strategy(tmp_path, may_start_live_candidate=True)

    with pytest.raises(BenchReadinessReviewError, match="must not authorize live"):
        review_bench_readiness(
            repo_root=tmp_path,
            surface_strategy_review=strategy,
            broader_strategy_review=broader_strategy,
        )


def _write_surface_strategy(
    tmp_path: Path,
    *,
    decision: str = "NO_NON_OVERLAPPING_PLANNER_APPROVED",
    may_start_open_ended_loop: bool,
    exhausted_surfaces: list[str],
    unimplemented_candidate_surfaces: list[str],
    writes_ledger: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/surface_strategy_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.surface_strategy_review.v1",
                "status": "PASS",
                "decision": decision,
                "approved_next_surface": None,
                "approved_planner": None,
                "candidate_limit": 0,
                "may_start_live_candidate": may_start_open_ended_loop,
                "may_start_open_ended_loop": may_start_open_ended_loop,
                "bench_blocked_reason": None,
                "consumed_next_surface_reviews": [],
                "consumed_diagnoses": [],
                "exhausted_surfaces": exhausted_surfaces,
                "unimplemented_candidate_surfaces": unimplemented_candidate_surfaces,
                "required_next_step": "test",
                "starts_search": False,
                "writes_ledger": writes_ledger,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)


def _write_broader_strategy(
    tmp_path: Path,
    *,
    may_start_live_candidate: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/broader_strategy_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.broader_strategy_review.v1",
                "status": "PASS",
                "decision": "APPROVE_DRY_RUN_PLANNER_PR_ONLY",
                "approved_next_surface": "diffusion_initialization_scale",
                "approved_planner": "diffusion_initialization_scale_diagnostic",
                "candidate_limit": 1,
                "may_start_live_candidate": may_start_live_candidate,
                "may_start_open_ended_loop": False,
                "non_overlap_rationale": "model-internal diffusion initial state scale",
                "forbidden_edits": ["scorer"],
                "stop_conditions": ["post-merge readiness is not green"],
                "consumed_surface_strategy_review": "runs/autoresearch/surface_strategy_review/review.json",
                "consumed_bench_readiness_review": "runs/autoresearch/bench_readiness_review/review.json",
                "exhausted_surfaces": ["auxiliary_loss", "feature_handling", "memory_runtime"],
                "required_next_step": "implement planner",
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)
