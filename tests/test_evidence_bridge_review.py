from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.evidence_bridge_review import EvidenceBridgeReviewError, review_evidence_bridge


def test_evidence_bridge_review_approves_next_candidate_pr_only(tmp_path: Path) -> None:
    prd = _write_post_exhaustion_prd(tmp_path)
    run_dir = _write_bridge_candidate_run(tmp_path, prd=prd)

    report = review_evidence_bridge(
        repo_root=tmp_path,
        post_exhaustion_strategy=prd,
        candidate_run_dir=run_dir,
    )

    payload = report.to_dict()
    assert payload["schema_version"] == "autoaf3.evidence_bridge_review.v1"
    assert payload["decision"] == "APPROVE_NEXT_CANDIDATE_IMPLEMENTATION_PR_ONLY"
    assert payload["candidate_limit"] == 1
    assert payload["reviewed_trial_id"] == "T177"
    assert payload["target_level_non_regression_count"] == 1
    assert payload["geometry_failure_mode_count"] == 1
    assert payload["may_start_live_candidate"] is False
    assert payload["may_start_open_ended_loop"] is False
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert payload["official_benchmark_result"] is False
    assert payload["required_objectives"][0]["name"] == "implement_evidence_guided_candidate_pr"
    assert payload["roadmap"][1]["step"] == "implement_candidate_pr"


def test_evidence_bridge_review_blocks_live_or_search_claims(tmp_path: Path) -> None:
    prd = _write_post_exhaustion_prd(tmp_path)
    run_dir = _write_bridge_candidate_run(tmp_path, prd=prd, stop_before_live_modal=False)

    report = review_evidence_bridge(
        repo_root=tmp_path,
        post_exhaustion_strategy=prd,
        candidate_run_dir=run_dir,
    )

    assert report.decision == "BLOCK_EVIDENCE_BRIDGE_REVIEW"
    assert report.candidate_limit == 0
    assert "bridge config must stop before live Modal" in report.blocked_reasons
    assert report.may_start_live_candidate is False
    assert report.official_benchmark_result is False


def test_evidence_bridge_review_refuses_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(EvidenceBridgeReviewError, match="repo-relative"):
        review_evidence_bridge(
            repo_root=tmp_path,
            post_exhaustion_strategy=Path("/tmp/prd.json"),
            candidate_run_dir="runs/autoresearch/run",
        )


def _write_post_exhaustion_prd(tmp_path: Path) -> Path:
    path = tmp_path / "runs/autoresearch/post_exhaustion_strategy/prd.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.post_exhaustion_strategy_prd.v1",
                "status": "PASS",
                "decision": "APPROVE_DRY_RUN_STRATEGY_PRD_ONLY",
                "approved_strategy_family": "evidence_guided_failure_mode_bridge",
                "approved_planner": "evidence_guided_failure_mode_bridge_diagnostic",
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


def _write_bridge_candidate_run(
    tmp_path: Path,
    *,
    prd: Path,
    stop_before_live_modal: bool = True,
) -> Path:
    run_dir = tmp_path / "runs/autoresearch/evidence-bridge-run"
    candidate_dir = run_dir / "candidates/T177"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    scorer = _write_scorer_sensitivity(tmp_path)
    geometry = _write_prediction_geometry(tmp_path)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.autoresearch_run_manifest.v1",
                "run_id": "evidence-bridge-run",
                "mode": "dry-run",
                "planner": "evidence_guided_failure_mode_bridge_diagnostic",
                "candidate_count": 1,
                "live_modal_execution": False,
                "target": "NanoFold-style AlphaFold3-lite",
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.autoresearch_summary.v1",
                "run_id": "evidence-bridge-run",
                "official_benchmark_result": False,
                "candidates": [
                    {
                        "trial_id": "T177",
                        "planning_status": "PLANNED",
                        "official_benchmark_result": False,
                        "writes_ledger": False,
                        "writes_discovery_ledger": False,
                        "provisional_keep": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.autoresearch_preflight_plan.v1",
                "trial_id": "T177",
                "mode": "dry-run",
                "planning_status": "PLANNED",
                "status": "DRAFT",
                "max_templates": 0,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "config.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.evidence_guided_failure_mode_bridge_plan.v1",
                "approved_planner": "evidence_guided_failure_mode_bridge_diagnostic",
                "approved_strategy_family": "evidence_guided_failure_mode_bridge",
                "candidate_limit": 1,
                "artifact_only": True,
                "stop_before_live_modal": stop_before_live_modal,
                "not_a_benchmark_claim": True,
                "official_benchmark_result": False,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "max_templates": 0,
                "source_post_exhaustion_strategy_prd": str(prd),
                "source_scorer_sensitivity": str(scorer),
                "source_geometry_audit": str(geometry),
                "target_level_non_regression_expectations": [
                    {
                        "target_id": "1HEL_A",
                        "kill_if_regresses": True,
                        "expected_score_delta_floor": 0.0,
                    }
                ],
                "geometry_failure_modes_to_avoid": ["pair_distance_exploded"],
                "forbidden_edit_attestation": {"all_forbidden_edits_unchanged": True},
                "config_path": "configs/experiments/T177_evidence_guided_failure_mode_bridge_diagnostic.json",
            }
        ),
        encoding="utf-8",
    )
    return run_dir.relative_to(tmp_path)


def _write_scorer_sensitivity(tmp_path: Path) -> Path:
    path = tmp_path / "runs/autoresearch/scorer_sensitivity/report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.scorer_sensitivity.v1",
                "all_primary_scores_identical": False,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)


def _write_prediction_geometry(tmp_path: Path) -> Path:
    path = tmp_path / "runs/autoresearch/prediction_geometry/report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.prediction_geometry_audit.v1",
                "artifacts": [],
                "reference_deltas": [],
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return path.relative_to(tmp_path)
