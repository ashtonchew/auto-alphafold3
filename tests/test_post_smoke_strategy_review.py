from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.post_smoke_strategy_review import (
    PostSmokeStrategyReviewError,
    review_post_smoke_strategy,
)


def test_post_smoke_strategy_approves_next_candidate_plan_only(tmp_path: Path) -> None:
    live_result = _write_live_smoke_result(tmp_path)
    bench = _write_bench_readiness(tmp_path)
    _write_live_smoke_run(tmp_path)

    report = review_post_smoke_strategy(
        repo_root=tmp_path,
        live_smoke_result_review=live_result,
        bench_readiness_review=bench,
    )
    payload = report.to_dict()

    assert payload["schema_version"] == "autoaf3.post_smoke_strategy_review.v1"
    assert payload["decision"] == "APPROVE_NEXT_BOUNDED_CANDIDATE_PLAN_ONLY"
    assert payload["approved_strategy_family"] == "sampler_low_noise_locality_refinement"
    assert payload["approved_next_candidate"] == "sampler_low_noise_locality_refinement"
    assert payload["approved_next_planner"] == "manual_sampler_low_noise_locality_refinement"
    assert payload["candidate_limit"] == 1
    assert payload["candidate_score"] == pytest.approx(0.07791816299247686)
    assert payload["global_baseline_delta"] == pytest.approx(-0.0014941413929591835)
    assert payload["num_scored_targets"] == 16
    assert payload["num_failed_targets"] == 0
    assert payload["sampler_manifest_summary"]["sampler_locality_guard"] == "reject_exploded"
    assert payload["next_candidate_plan"]["required_sampler_settings"]["sampler_noise_scale"] == pytest.approx(0.6)
    assert payload["required_objectives"][0]["name"] == "create_next_bounded_candidate_plan"
    assert payload["roadmap"][1]["step"] == "create_next_bounded_candidate_plan"
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False


def test_post_smoke_strategy_lowers_second_clean_discard_to_final_noise_step(tmp_path: Path) -> None:
    live_result = _write_live_smoke_result(tmp_path, trial_id="T180", run_dir="runs/autoresearch/live-smoke-t180")
    bench = _write_bench_readiness(tmp_path)
    _write_live_smoke_run(tmp_path, trial_id="T180", run_dir="runs/autoresearch/live-smoke-t180", sampler_noise_scale=0.6)

    report = review_post_smoke_strategy(
        repo_root=tmp_path,
        live_smoke_result_review=live_result,
        bench_readiness_review=bench,
    )

    assert report.decision == "APPROVE_NEXT_BOUNDED_CANDIDATE_PLAN_ONLY"
    assert report.reviewed_trial_id == "T180"
    assert report.next_candidate_plan["required_sampler_settings"]["sampler_noise_scale"] == pytest.approx(0.3)


def test_post_smoke_strategy_blocks_when_noise_ladder_is_exhausted(tmp_path: Path) -> None:
    live_result = _write_live_smoke_result(tmp_path, trial_id="T181", run_dir="runs/autoresearch/live-smoke-t181")
    bench = _write_bench_readiness(tmp_path)
    _write_live_smoke_run(tmp_path, trial_id="T181", run_dir="runs/autoresearch/live-smoke-t181", sampler_noise_scale=0.3)

    report = review_post_smoke_strategy(
        repo_root=tmp_path,
        live_smoke_result_review=live_result,
        bench_readiness_review=bench,
    )

    assert report.decision == "BLOCK_POST_SMOKE_STRATEGY_REVIEW"
    assert report.candidate_limit == 0
    assert "sampler noise refinement ladder is exhausted for this source smoke" in report.blocked_reasons


def test_post_smoke_strategy_blocks_provisional_keep(tmp_path: Path) -> None:
    live_result = _write_live_smoke_result(
        tmp_path,
        decision="BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_PROVISIONAL_KEEP",
        smoke_status="KEEP",
        provisional_keep=True,
    )
    bench = _write_bench_readiness(tmp_path, decision="BLOCK_OPEN_ENDED_BENCH_FALSIFICATION_REQUIRED")
    _write_live_smoke_run(tmp_path, comparison_status="KEEP", provisional_keep=True)

    report = review_post_smoke_strategy(
        repo_root=tmp_path,
        live_smoke_result_review=live_result,
        bench_readiness_review=bench,
    )

    assert report.decision == "BLOCK_POST_SMOKE_STRATEGY_REVIEW"
    assert report.candidate_limit == 0
    assert "live smoke result review must be a scored DISCARD" in report.blocked_reasons
    assert "provisional KEEP requires falsification, not a next strategy" in report.blocked_reasons
    assert report.required_objectives[0]["name"] == "repair_post_smoke_strategy_inputs"


def test_post_smoke_strategy_blocks_failed_target_metrics(tmp_path: Path) -> None:
    live_result = _write_live_smoke_result(tmp_path)
    bench = _write_bench_readiness(tmp_path)
    _write_live_smoke_run(tmp_path, num_scored_targets=15, num_failed_targets=1)

    report = review_post_smoke_strategy(
        repo_root=tmp_path,
        live_smoke_result_review=live_result,
        bench_readiness_review=bench,
    )

    assert report.decision == "BLOCK_POST_SMOKE_STRATEGY_REVIEW"
    assert "discarded smoke must not have failed scored targets before strategy refinement" in report.blocked_reasons
    assert "discarded smoke must score every target before strategy refinement" in report.blocked_reasons


def test_post_smoke_strategy_refuses_authority_claims(tmp_path: Path) -> None:
    live_result = _write_live_smoke_result(tmp_path, writes_ledger=True)
    bench = _write_bench_readiness(tmp_path)
    _write_live_smoke_run(tmp_path)

    with pytest.raises(PostSmokeStrategyReviewError, match="writes_ledger"):
        review_post_smoke_strategy(
            repo_root=tmp_path,
            live_smoke_result_review=live_result,
            bench_readiness_review=bench,
        )


def test_post_smoke_strategy_requires_safe_paths(tmp_path: Path) -> None:
    bench = _write_bench_readiness(tmp_path)

    with pytest.raises(PostSmokeStrategyReviewError, match="repo-relative"):
        review_post_smoke_strategy(
            repo_root=tmp_path,
            live_smoke_result_review=Path("/tmp/live.json"),
            bench_readiness_review=bench,
        )


def _write_live_smoke_result(
    tmp_path: Path,
    *,
    trial_id: str = "T179",
    run_dir: str = "runs/autoresearch/live-smoke",
    decision: str = "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED",
    smoke_status: str = "DISCARD",
    provisional_keep: bool = False,
    writes_ledger: bool = False,
) -> Path:
    path = tmp_path / "runs/autoresearch/live_smoke_result_review/review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.live_smoke_result_review.v1",
                "status": "PASS",
                "decision": decision,
                "reviewed_run_dir": run_dir,
                "reviewed_trial_id": trial_id,
                "candidate_id": trial_id,
                "smoke_status": smoke_status,
                "result_status": "SCORED",
                "candidate_score": 0.07791816299247686,
                "global_baseline_delta": -0.0014941413929591835,
                "provisional_keep": provisional_keep,
                "promotion_status": "FALSIFICATION_REQUIRED" if provisional_keep else "NOT_ELIGIBLE",
                "failure_signature": None,
                "required_objectives": [],
                "roadmap": [],
                "may_start_live_candidate": False,
                "may_start_open_ended_loop": False,
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
    decision: str = "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED",
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
                "autonomous_search_ready": True,
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


def _write_live_smoke_run(
    tmp_path: Path,
    *,
    trial_id: str = "T179",
    run_dir: str = "runs/autoresearch/live-smoke",
    comparison_status: str = "DISCARD",
    provisional_keep: bool = False,
    num_scored_targets: int = 16,
    num_failed_targets: int = 0,
    sampler_noise_scale: float = 1.0,
) -> None:
    candidate_dir = tmp_path / run_dir / "candidates" / trial_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.autoresearch_comparison_metrics.v1",
                "trial_id": trial_id,
                "result_status": "SCORED",
                "comparison": {
                    "candidate_score": 0.07791816299247686,
                    "global_baseline_delta": -0.0014941413929591835,
                    "global_current_best_score": 0.07941230438543605,
                    "status": comparison_status,
                    "provisional_keep": provisional_keep,
                    "writes_ledger": False,
                    "writes_discovery_ledger": False,
                },
                "metrics": {
                    "best_val_calpha_lddt": 0.07791816299247686,
                    "num_targets": 16,
                    "num_scored_targets": num_scored_targets,
                    "num_failed_targets": num_failed_targets,
                },
                "official_benchmark_result": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "sampler_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.sampler_manifest.v1",
                "trial_id": trial_id,
                "status": "SAMPLER_PREDICTED",
                "sampler_locality_guard": "reject_exploded",
                "sampler_coordinate_normalization": "ca_bond",
                "sampler_coordinate_scale": 1.0,
                "sampler_noise_scale": sampler_noise_scale,
                "sampler_num_samples": 4,
                "sampler_selection_policy": "geometry",
                "max_templates": 0,
                "prediction_count": 16,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
