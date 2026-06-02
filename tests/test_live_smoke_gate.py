from __future__ import annotations

import json
from pathlib import Path

import pytest

import autoalphafold3.live_smoke_gate as live_gate
from autoalphafold3.live_smoke_gate import LiveSmokeGateError, review_live_smoke_gate


class FakeReadiness:
    def __init__(self, *, ready: bool) -> None:
        self.autonomous_search_ready = ready

    def to_dict(self) -> dict[str, object]:
        status = "PASS" if self.autonomous_search_ready else "BLOCKED"
        return {
            "autonomous_search_ready": self.autonomous_search_ready,
            "problems": [] if self.autonomous_search_ready else ["readiness problem"],
            "pending_human_actions": [],
            "baseline_lock": {"status": status},
            "local_gates": {"status": "PASS"},
            "gate_calibration": {"status": "PASS"},
            "modal_event_authority": {"status": "PASS"},
        }


def test_live_smoke_gate_approves_one_bounded_smoke_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _write_candidate_implementation_review(tmp_path)
    monkeypatch.setattr(live_gate, "build_readiness_report", lambda **_: FakeReadiness(ready=True))

    report = review_live_smoke_gate(repo_root=tmp_path, candidate_implementation_review=candidate)

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.live_smoke_gate.v1"
    assert payload["decision"] == "APPROVE_BOUNDED_LIVE_SMOKE_ONLY"
    assert payload["approved_candidate"] == "sampler_locality_guard"
    assert payload["candidate_limit"] == 1
    assert payload["required_approval_token"] == "I_APPROVE_AUTORESEARCH_LIVE_SEARCH"
    assert payload["may_start_live_candidate"] is True
    assert payload["may_start_open_ended_loop"] is False
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False


def test_live_smoke_gate_blocks_when_readiness_not_green(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _write_candidate_implementation_review(tmp_path)
    monkeypatch.setattr(live_gate, "build_readiness_report", lambda **_: FakeReadiness(ready=False))

    report = review_live_smoke_gate(repo_root=tmp_path, candidate_implementation_review=candidate)

    assert report.decision == "BLOCK_LIVE_SMOKE_GATE"
    assert report.candidate_limit == 0
    assert report.may_start_live_candidate is False
    assert "foundation readiness is not autonomous_search_ready=true" in report.blocked_reasons


def test_live_smoke_gate_refuses_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(LiveSmokeGateError, match="repo-relative"):
        review_live_smoke_gate(
            repo_root=tmp_path,
            candidate_implementation_review=Path("/tmp/candidate.json"),
        )


def _write_candidate_implementation_review(tmp_path: Path) -> Path:
    path = tmp_path / "runs/autoresearch/candidate_implementation_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.candidate_implementation_review.v1",
                "status": "PASS",
                "decision": "APPROVE_LIVE_SMOKE_GATE_PR_ONLY",
                "approved_candidate": "sampler_locality_guard",
                "may_start_live_candidate": False,
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
