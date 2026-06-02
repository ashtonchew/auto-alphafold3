from __future__ import annotations

import json
from pathlib import Path

import pytest

import autoalphafold3.open_ended_bench_gate as gate
from autoalphafold3.autoresearch_loop import APPROVAL_TEXT
from autoalphafold3.open_ended_bench_gate import OpenEndedBenchGateError, review_open_ended_bench_gate


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


def test_open_ended_bench_gate_approves_after_strategy_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _write_strategy_exhaustion(tmp_path)
    monkeypatch.setattr(gate, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_open_ended_bench_gate(repo_root=tmp_path, strategy_exhaustion_audit=audit)

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.open_ended_bench_gate.v1"
    assert payload["decision"] == "APPROVE_OPEN_ENDED_BENCH_ONLY"
    assert payload["approved_mode"] == "modal"
    assert payload["approved_planner"] == "llm"
    assert payload["approved_candidate_budget"] == "smoke"
    assert payload["approved_max_candidates"] == 3
    assert payload["approved_failure_streak_limit"] == 2
    assert payload["required_approval_token"] == APPROVAL_TEXT
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is True
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False
    assert payload["blocked_reasons"] == []
    assert payload["required_objectives"][0]["name"] == "run_open_ended_bench"


def test_open_ended_bench_gate_blocks_when_readiness_not_green(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _write_strategy_exhaustion(tmp_path)
    monkeypatch.setattr(gate, "build_readiness_report", lambda **_: FakeReadiness(ready=False))

    report = review_open_ended_bench_gate(repo_root=tmp_path, strategy_exhaustion_audit=audit)

    payload = report.to_dict()
    assert payload["decision"] == "BLOCK_OPEN_ENDED_BENCH_GATE"
    assert payload["may_start_open_ended_loop"] is False
    assert "foundation readiness is not autonomous_search_ready=true" in payload["blocked_reasons"]
    assert payload["approved_planner"] is None


def test_open_ended_bench_gate_blocks_when_planners_remain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _write_strategy_exhaustion(tmp_path, remaining_implemented_planners=["llm"])
    monkeypatch.setattr(gate, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_open_ended_bench_gate(repo_root=tmp_path, strategy_exhaustion_audit=audit)

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_GATE"
    assert "no remaining implemented planners" in report.blocked_reasons[0]


def test_open_ended_bench_gate_refuses_unsafe_audit_authority(tmp_path: Path) -> None:
    audit = _write_strategy_exhaustion(tmp_path, writes_ledger=True)

    with pytest.raises(OpenEndedBenchGateError, match="writes_ledger"):
        review_open_ended_bench_gate(repo_root=tmp_path, strategy_exhaustion_audit=audit)


def _write_strategy_exhaustion(
    tmp_path: Path,
    *,
    remaining_implemented_planners: list[str] | None = None,
    writes_ledger: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/strategy_exhaustion_audit/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.strategy_exhaustion_audit.v1",
                "status": "PASS",
                "decision": "NO_IMPLEMENTED_PLANNER_REMAINING",
                "remaining_implemented_planners": remaining_implemented_planners or [],
                "can_start_open_ended_bench": False,
                "bench_decision": "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED",
                "starts_search": False,
                "writes_ledger": writes_ledger,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)
