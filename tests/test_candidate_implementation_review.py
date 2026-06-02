from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.candidate_implementation_review import (
    CandidateImplementationReviewError,
    review_candidate_implementation,
)


def test_candidate_implementation_review_approves_live_smoke_gate_pr_only(tmp_path: Path) -> None:
    bridge = _write_evidence_bridge_review(tmp_path)

    report = review_candidate_implementation(repo_root=tmp_path, evidence_bridge_review=bridge)

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.candidate_implementation_review.v1"
    assert payload["decision"] == "APPROVE_LIVE_SMOKE_GATE_PR_ONLY"
    assert payload["approved_candidate"] == "sampler_locality_guard"
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False
    assert all(check["status"] == "PASS" for check in payload["behavior_checks"])
    assert payload["required_objectives"][0]["name"] == "implement_live_smoke_approval_gate"


def test_candidate_implementation_review_blocks_wrong_candidate(tmp_path: Path) -> None:
    bridge = _write_evidence_bridge_review(tmp_path)

    report = review_candidate_implementation(
        repo_root=tmp_path,
        evidence_bridge_review=bridge,
        candidate="unsupported",
    )

    assert report.decision == "BLOCK_CANDIDATE_IMPLEMENTATION_REVIEW"
    assert report.approved_candidate is None
    assert "unsupported candidate implementation review: unsupported" in report.blocked_reasons
    assert report.may_start_live_candidate is False


def test_candidate_implementation_review_refuses_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(CandidateImplementationReviewError, match="repo-relative"):
        review_candidate_implementation(
            repo_root=tmp_path,
            evidence_bridge_review=Path("/tmp/bridge.json"),
        )


def _write_evidence_bridge_review(tmp_path: Path) -> Path:
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
