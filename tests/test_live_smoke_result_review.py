from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.live_smoke_result_review import (
    LiveSmokeResultReviewError,
    review_live_smoke_result,
)


def test_live_smoke_result_review_blocks_after_scored_discard(tmp_path: Path) -> None:
    run_dir = _write_live_smoke_run(tmp_path, status="DISCARD", result_status="SCORED")

    report = review_live_smoke_result(repo_root=tmp_path, live_smoke_run_dir=run_dir)

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.live_smoke_result_review.v1"
    assert payload["decision"] == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED"
    assert payload["reviewed_trial_id"] == "T179"
    assert payload["smoke_status"] == "DISCARD"
    assert payload["result_status"] == "SCORED"
    assert payload["candidate_score"] == pytest.approx(0.07791816299247686)
    assert payload["global_baseline_delta"] == pytest.approx(-0.0014941413929591835)
    assert payload["required_objectives"][0]["name"] == "post_smoke_strategy_review"
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False


def test_live_smoke_result_review_blocks_after_failure(tmp_path: Path) -> None:
    run_dir = _write_live_smoke_run(tmp_path, status="FAIL", result_status=None)

    report = review_live_smoke_result(repo_root=tmp_path, live_smoke_run_dir=run_dir)

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_FAILED"
    assert report.failure_signature == "modal_worker_exception"
    assert report.required_objectives[0]["name"] == "diagnose_live_smoke_failure"
    assert report.roadmap[1]["step"] == "diagnose_failure"


def test_live_smoke_result_review_blocks_after_provisional_keep(tmp_path: Path) -> None:
    run_dir = _write_live_smoke_run(
        tmp_path,
        status="KEEP",
        result_status="SCORED",
        provisional_keep=True,
        promotion_status="FALSIFICATION_REQUIRED",
    )

    report = review_live_smoke_result(repo_root=tmp_path, live_smoke_run_dir=run_dir)

    assert report.decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_PROVISIONAL_KEEP"
    assert report.provisional_keep is True
    assert report.required_objectives[0]["name"] == "run_falsification_gate"
    assert report.roadmap[1]["step"] == "run_falsification_gate"


def test_live_smoke_result_review_refuses_unsafe_or_nonterminal_evidence(tmp_path: Path) -> None:
    with pytest.raises(LiveSmokeResultReviewError, match="repo-relative"):
        review_live_smoke_result(repo_root=tmp_path, live_smoke_run_dir=Path("/tmp/run"))

    run_dir = _write_live_smoke_run(tmp_path, status="DRAFT", result_status=None)
    with pytest.raises(LiveSmokeResultReviewError, match="unsupported live smoke terminal status"):
        review_live_smoke_result(repo_root=tmp_path, live_smoke_run_dir=run_dir)

    run_dir = _write_live_smoke_run(tmp_path, status="DISCARD", result_status="SCORED", writes_ledger=True)
    with pytest.raises(LiveSmokeResultReviewError, match="writes_ledger"):
        review_live_smoke_result(repo_root=tmp_path, live_smoke_run_dir=run_dir)


def _write_live_smoke_run(
    tmp_path: Path,
    *,
    status: str,
    result_status: str | None,
    provisional_keep: bool = False,
    promotion_status: str = "NOT_ELIGIBLE",
    writes_ledger: bool = False,
) -> str:
    run_dir = tmp_path / "runs/autoresearch/live-smoke/candidates/T179"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "autoaf3.autoresearch_summary.v1",
        "run_id": "live-smoke",
        "official_benchmark_result": False,
        "candidates": [
            {
                "trial_id": "T179",
                "candidate_id": "T179",
                "status": status,
                "decision_path": "runs/autoresearch/live-smoke/candidates/T179/decision.json",
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ],
    }
    (tmp_path / "runs/autoresearch/live-smoke/summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    decision = {
        "schema_version": "autoaf3.autoresearch_decision.v1",
        "run_id": "live-smoke",
        "trial_id": "T179",
        "candidate_id": "T179",
        "status": status,
        "global_baseline_delta": -0.0014941413929591835,
        "provisional_keep": provisional_keep,
        "promotion_status": promotion_status,
        "reason": "modal_worker_exception" if status == "FAIL" else "candidate did not clear threshold",
        "writes_ledger": writes_ledger,
        "writes_discovery_ledger": False,
        "official_benchmark_result": False,
    }
    (run_dir / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
    if result_status is not None:
        metrics = {
            "schema_version": "autoaf3.autoresearch_comparison_metrics.v1",
            "trial_id": "T179",
            "result_status": result_status,
            "comparison": {
                "candidate_score": 0.07791816299247686,
                "status": status,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
            },
            "official_benchmark_result": False,
            "writes_ledger": False,
            "writes_discovery_ledger": False,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if status == "FAIL":
        error = {
            "schema_version": "autoaf3.metrics.v1",
            "status": "FAIL",
            "trial_id": "T179",
            "candidate_id": "T179",
            "error_report": {
                "failure_signature": "modal_worker_exception",
                "reason": "worker failed",
            },
            "official_benchmark_result": False,
            "writes_ledger": False,
            "writes_discovery_ledger": False,
        }
        (run_dir / "error_report.json").write_text(json.dumps(error), encoding="utf-8")
    return "runs/autoresearch/live-smoke"
