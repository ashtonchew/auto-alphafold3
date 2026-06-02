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


def test_bench_readiness_consumes_evidence_bridge_as_next_candidate_pr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    bridge = _write_evidence_bridge_review(tmp_path)
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_bench_readiness(
        repo_root=tmp_path,
        surface_strategy_review=strategy,
        evidence_bridge_review=bridge,
    )

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_CANDIDATE_IMPLEMENTATION_REQUIRED"
    assert report.can_start_open_ended_bench is False
    assert report.evidence_bridge_decision == "APPROVE_NEXT_CANDIDATE_IMPLEMENTATION_PR_ONLY"
    assert report.approved_evidence_bridge_planner == "evidence_guided_failure_mode_bridge_diagnostic"
    assert report.required_objectives[0]["name"] == "implement_evidence_guided_candidate_pr"
    assert report.roadmap[1]["step"] == "implement_evidence_guided_candidate_pr"
    assert report.evidence["evidence_bridge_review"] == str(bridge)
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.writes_discovery_ledger is False
    assert report.official_benchmark_result is False


def test_bench_readiness_consumes_candidate_implementation_as_live_smoke_gate_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    bridge = _write_evidence_bridge_review(tmp_path)
    candidate = _write_candidate_implementation_review(tmp_path)
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_bench_readiness(
        repo_root=tmp_path,
        surface_strategy_review=strategy,
        evidence_bridge_review=bridge,
        candidate_implementation_review=candidate,
    )

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_GATE_REQUIRED"
    assert report.can_start_open_ended_bench is False
    assert report.candidate_implementation_decision == "APPROVE_LIVE_SMOKE_GATE_PR_ONLY"
    assert report.approved_candidate_implementation == "sampler_locality_guard"
    assert report.required_objectives[0]["name"] == "implement_live_smoke_approval_gate"
    assert report.roadmap[1]["step"] == "implement_live_smoke_approval_gate"
    assert report.evidence["candidate_implementation_review"] == str(candidate)
    assert report.may_start_live_candidate is False
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.official_benchmark_result is False


def test_bench_readiness_consumes_live_smoke_gate_as_bounded_smoke_approved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    candidate = _write_candidate_implementation_review(tmp_path)
    live_smoke = _write_live_smoke_gate(tmp_path)
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_bench_readiness(
        repo_root=tmp_path,
        surface_strategy_review=strategy,
        candidate_implementation_review=candidate,
        live_smoke_gate=live_smoke,
    )

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_BOUNDED_LIVE_SMOKE_APPROVED"
    assert report.can_start_open_ended_bench is False
    assert report.may_start_live_candidate is True
    assert report.may_start_open_ended_loop is False
    assert report.live_smoke_gate_decision == "APPROVE_BOUNDED_LIVE_SMOKE_ONLY"
    assert report.approved_live_smoke_candidate == "sampler_locality_guard"
    assert report.required_objectives[0]["name"] == "run_one_bounded_live_smoke"
    assert report.roadmap[1]["step"] == "run_one_bounded_live_smoke"
    assert report.evidence["live_smoke_gate"] == str(live_smoke)
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.official_benchmark_result is False


def test_bench_readiness_consumes_live_smoke_result_as_post_smoke_strategy_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    live_smoke = _write_live_smoke_gate(tmp_path)
    live_smoke_result = _write_live_smoke_result_review(tmp_path)
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_bench_readiness(
        repo_root=tmp_path,
        surface_strategy_review=strategy,
        live_smoke_gate=live_smoke,
        live_smoke_result_review=live_smoke_result,
    )

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED"
    assert report.can_start_open_ended_bench is False
    assert report.may_start_live_candidate is False
    assert report.live_smoke_result_decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED"
    assert report.live_smoke_result_status == "DISCARD"
    assert report.live_smoke_result_trial_id == "T179"
    assert report.required_objectives[0]["name"] == "post_smoke_strategy_review"
    assert report.roadmap[1]["step"] == "write_post_smoke_strategy"
    assert report.evidence["live_smoke_result_review"] == str(live_smoke_result)
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.official_benchmark_result is False


def test_bench_readiness_live_smoke_result_overrides_stale_open_ended_approval(
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
    live_smoke_result = _write_live_smoke_result_review(tmp_path)
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_bench_readiness(
        repo_root=tmp_path,
        surface_strategy_review=strategy,
        live_smoke_result_review=live_smoke_result,
    )

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED"
    assert report.can_start_open_ended_bench is False
    assert report.may_start_live_candidate is False
    assert report.required_objectives[0]["name"] == "post_smoke_strategy_review"


def test_bench_readiness_routes_provisional_keep_to_falsification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    live_smoke_result = _write_live_smoke_result_review(
        tmp_path,
        decision="BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_PROVISIONAL_KEEP",
        smoke_status="KEEP",
    )
    monkeypatch.setattr(bench_review, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_bench_readiness(
        repo_root=tmp_path,
        surface_strategy_review=strategy,
        live_smoke_result_review=live_smoke_result,
    )

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_FALSIFICATION_REQUIRED"
    assert report.required_objectives[0]["name"] == "run_falsification_gate"
    assert report.roadmap[1]["step"] == "run_falsification_gate"


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


def test_bench_readiness_refuses_live_authority_in_evidence_bridge(tmp_path: Path) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    bridge = _write_evidence_bridge_review(tmp_path, may_start_live_candidate=True)

    with pytest.raises(BenchReadinessReviewError, match="must not authorize live"):
        review_bench_readiness(
            repo_root=tmp_path,
            surface_strategy_review=strategy,
            evidence_bridge_review=bridge,
        )


def test_bench_readiness_refuses_live_authority_in_candidate_implementation(tmp_path: Path) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    candidate = _write_candidate_implementation_review(tmp_path, may_start_live_candidate=True)

    with pytest.raises(BenchReadinessReviewError, match="must not authorize live"):
        review_bench_readiness(
            repo_root=tmp_path,
            surface_strategy_review=strategy,
            candidate_implementation_review=candidate,
        )


def test_bench_readiness_refuses_open_ended_authority_in_live_smoke_gate(tmp_path: Path) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    live_smoke = _write_live_smoke_gate(tmp_path, may_start_open_ended_loop=True)

    with pytest.raises(BenchReadinessReviewError, match="must not authorize open-ended"):
        review_bench_readiness(
            repo_root=tmp_path,
            surface_strategy_review=strategy,
            live_smoke_gate=live_smoke,
        )


def test_bench_readiness_refuses_live_authority_in_live_smoke_result(tmp_path: Path) -> None:
    strategy = _write_surface_strategy(
        tmp_path,
        may_start_open_ended_loop=False,
        exhausted_surfaces=["auxiliary_loss", "feature_handling", "memory_runtime"],
        unimplemented_candidate_surfaces=[],
    )
    live_smoke_result = _write_live_smoke_result_review(tmp_path, may_start_live_candidate=True)

    with pytest.raises(BenchReadinessReviewError, match="must not authorize live"):
        review_bench_readiness(
            repo_root=tmp_path,
            surface_strategy_review=strategy,
            live_smoke_result_review=live_smoke_result,
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
                "blocked_reason": None,
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


def _write_evidence_bridge_review(
    tmp_path: Path,
    *,
    may_start_live_candidate: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/evidence_bridge_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.evidence_bridge_review.v1",
                "status": "PASS",
                "decision": "APPROVE_NEXT_CANDIDATE_IMPLEMENTATION_PR_ONLY",
                "approved_planner": "evidence_guided_failure_mode_bridge_diagnostic",
                "approved_strategy_family": "evidence_guided_failure_mode_bridge",
                "candidate_limit": 1,
                "reviewed_trial_id": "T177",
                "blocked_reasons": [],
                "may_start_live_candidate": may_start_live_candidate,
                "may_start_open_ended_loop": False,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)


def _write_candidate_implementation_review(
    tmp_path: Path,
    *,
    may_start_live_candidate: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/candidate_implementation_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.candidate_implementation_review.v1",
                "status": "PASS",
                "decision": "APPROVE_LIVE_SMOKE_GATE_PR_ONLY",
                "approved_candidate": "sampler_locality_guard",
                "blocked_reasons": [],
                "may_start_live_candidate": may_start_live_candidate,
                "may_start_open_ended_loop": False,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)


def _write_live_smoke_gate(
    tmp_path: Path,
    *,
    may_start_open_ended_loop: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/live_smoke_gate/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.live_smoke_gate.v1",
                "status": "PASS",
                "decision": "APPROVE_BOUNDED_LIVE_SMOKE_ONLY",
                "approved_candidate": "sampler_locality_guard",
                "candidate_limit": 1,
                "may_start_live_candidate": True,
                "may_start_open_ended_loop": may_start_open_ended_loop,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)


def _write_live_smoke_result_review(
    tmp_path: Path,
    *,
    decision: str = "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED",
    smoke_status: str = "DISCARD",
    may_start_live_candidate: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/live_smoke_result_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.live_smoke_result_review.v1",
                "status": "PASS",
                "decision": decision,
                "reviewed_run_dir": "runs/autoresearch/T179-smoke",
                "reviewed_trial_id": "T179",
                "candidate_id": "T179",
                "smoke_status": smoke_status,
                "result_status": "SCORED",
                "candidate_score": 0.07791816299247686,
                "global_baseline_delta": -0.0014941413929591835,
                "provisional_keep": smoke_status == "KEEP",
                "promotion_status": "FALSIFICATION_REQUIRED" if smoke_status == "KEEP" else "NOT_ELIGIBLE",
                "failure_signature": None,
                "required_objectives": [],
                "roadmap": [],
                "may_start_live_candidate": may_start_live_candidate,
                "may_start_open_ended_loop": False,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)
