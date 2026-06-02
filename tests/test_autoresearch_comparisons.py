from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.autoresearch_candidates import CandidateEnvelope, create_candidate_envelope, create_run_manifest
from autoalphafold3.autoresearch_comparisons import (
    AutoresearchComparisonError,
    compare_and_write_candidate_decision,
    compare_candidate_result,
)
from autoalphafold3.ledger import LEDGER_WRITER_ROLE, append_ledger
from autoalphafold3.schema import AutoFoldResult, FoldCartographerReport, TrialStatus

SHA = "a" * 64


def test_autoresearch_comparison_computes_and_writes_delta_axes(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.42)
    envelope = _candidate_envelope(tmp_path, "T123")

    comparison = compare_and_write_candidate_decision(
        envelope,
        candidate_result=_result("T123", 0.43),
        matched_budget_result=_result("T122", 0.40),
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path="runs/ledger.jsonl",
        keep_delta=0.001,
    )

    assert comparison.status == "KEEP"
    assert comparison.matched_budget_delta == pytest.approx(0.03)
    assert comparison.global_baseline_delta == pytest.approx(0.01)
    assert comparison.provisional_keep is True
    decision = json.loads(envelope.decision_path.read_text(encoding="utf-8"))
    assert decision["status"] == "KEEP"
    assert decision["matched_budget_delta"] == pytest.approx(0.03)
    assert decision["global_baseline_delta"] == pytest.approx(0.01)
    assert decision["keep_threshold_delta"] == pytest.approx(0.001)
    assert decision["discovery_status"] == "UNCONFIRMED"
    assert decision["promotion_status"] == "FALSIFICATION_REQUIRED"
    assert decision["promotion_plan_path"] == str(envelope.promotion_plan_path)
    assert decision["writes_ledger"] is False
    assert decision["writes_discovery_ledger"] is False
    promotion_plan = json.loads(envelope.promotion_plan_path.read_text(encoding="utf-8"))
    assert promotion_plan["status"] == "FALSIFICATION_REQUIRED"
    assert promotion_plan["provisional_keep"] is True
    assert promotion_plan["discovery_status"] == "UNCONFIRMED"
    assert promotion_plan["writes_ledger"] is False
    assert promotion_plan["writes_discovery_ledger"] is False
    assert promotion_plan["allowed_writer"] == "modal_hosted_trusted_orchestrator"
    assert "confirmed falsification verdict" in promotion_plan["required_evidence"]
    metrics = json.loads(envelope.metrics_path.read_text(encoding="utf-8"))
    assert metrics["result_status"] == "SCORED"
    assert metrics["fold_cartographer"]["signature"] == "comparison_fixture"
    assert metrics["candidate_artifacts"]["metrics_json"] == "runs/trials/T123/metrics.json"
    summary = json.loads((envelope.root / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"][0]["provisional_keep"] is True
    assert summary["candidates"][0]["promotion_status"] == "FALSIFICATION_REQUIRED"
    assert summary["candidates"][0]["promotion_plan_path"] == str(envelope.promotion_plan_path)
    results = (envelope.root / "results.tsv").read_text(encoding="utf-8")
    assert "T123\tT123\tKEEP\tbest_val_calpha_lddt" in results
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery").exists()


def test_autoresearch_comparison_matched_win_global_miss_discards(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.45)
    envelope = _candidate_envelope(tmp_path, "T124")

    comparison = compare_and_write_candidate_decision(
        envelope,
        candidate_result=_result("T124", 0.43),
        matched_budget_result=_result("T122", 0.40),
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path="runs/ledger.jsonl",
        keep_delta=0.001,
    )

    assert comparison.status == "DISCARD"
    assert comparison.matched_budget_delta == pytest.approx(0.03)
    assert comparison.global_baseline_delta == pytest.approx(-0.02)
    assert comparison.provisional_keep is False
    decision = json.loads(envelope.decision_path.read_text(encoding="utf-8"))
    assert decision["promotion_status"] == "NOT_ELIGIBLE"
    assert decision["promotion_plan_path"] is None
    assert not envelope.promotion_plan_path.exists()


def test_autoresearch_comparison_uses_global_current_best_from_prior_keep(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.42)
    ledger = tmp_path / "runs/ledger.jsonl"
    append_ledger(_result("T090", 0.45, status=TrialStatus.KEEP), ledger_path=ledger, writer_role=LEDGER_WRITER_ROLE)

    discard = compare_candidate_result(
        candidate_result=_result("T125", 0.451),
        matched_budget_result=_result("T122", 0.40),
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path=ledger.relative_to(tmp_path),
        keep_delta=0.001,
    )
    keep = compare_candidate_result(
        candidate_result=_result("T126", 0.4521),
        matched_budget_result=_result("T122", 0.40),
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path=ledger.relative_to(tmp_path),
        keep_delta=0.001,
    )

    assert discard.status == "DISCARD"
    assert discard.global_current_best_score == pytest.approx(0.45)
    assert keep.status == "KEEP"
    assert keep.global_current_best_score == pytest.approx(0.45)


def test_autoresearch_comparison_refuses_missing_baseline_without_fake_deltas(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="baseline"):
        compare_candidate_result(
            candidate_result=_result("T123", 0.43),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir="runs/baseline",
            ledger_path="runs/ledger.jsonl",
        )

    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_autoresearch_comparison_refuses_invalid_candidate_score(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.42)

    with pytest.raises(AutoresearchComparisonError, match="missing finite"):
        compare_candidate_result(
            candidate_result=_result("T123", 2.0),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir=baseline.relative_to(tmp_path),
            ledger_path="runs/ledger.jsonl",
        )


def test_autoresearch_comparison_refuses_candidate_envelope_trial_mismatch(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.42)
    envelope = _candidate_envelope(tmp_path, "T123")

    with pytest.raises(AutoresearchComparisonError, match="does not match envelope"):
        compare_and_write_candidate_decision(
            envelope,
            candidate_result=_result("T999", 0.43),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir=baseline.relative_to(tmp_path),
            ledger_path="runs/ledger.jsonl",
        )

    assert not envelope.metrics_path.exists()
    assert not envelope.decision_path.exists()


def test_autoresearch_comparison_refuses_absolute_or_traversing_comparison_paths(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.42)

    with pytest.raises(AutoresearchComparisonError, match="baseline_dir must be repo-relative"):
        compare_candidate_result(
            candidate_result=_result("T123", 0.43),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir=baseline,
            ledger_path="runs/ledger.jsonl",
        )
    with pytest.raises(AutoresearchComparisonError, match="baseline_dir must be runs/baseline"):
        compare_candidate_result(
            candidate_result=_result("T123", 0.43),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir="runs/autoresearch/fake-baseline",
            ledger_path="runs/ledger.jsonl",
        )
    with pytest.raises(AutoresearchComparisonError, match="ledger_path must be runs/ledger.jsonl"):
        compare_candidate_result(
            candidate_result=_result("T123", 0.43),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir=baseline.relative_to(tmp_path),
            ledger_path="runs/autoresearch/fake-ledger.jsonl",
        )
    with pytest.raises(AutoresearchComparisonError, match="ledger_path must be runs/ledger.jsonl"):
        compare_candidate_result(
            candidate_result=_result("T123", 0.43),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir=baseline.relative_to(tmp_path),
            ledger_path="../ledger.jsonl",
        )


def test_autoresearch_comparison_refuses_malicious_envelope_paths_before_writes(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.42)
    forged = CandidateEnvelope(
        run_id="comparison",
        trial_id="T123",
        root=tmp_path / "runs" / "autoresearch" / "comparison",
        candidate_dir=tmp_path / "runs" / "baseline",
        candidate_id="T123",
    )

    with pytest.raises(AutoresearchComparisonError, match="candidate envelope directory"):
        compare_and_write_candidate_decision(
            forged,
            candidate_result=_result("T123", 0.43),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir=baseline.relative_to(tmp_path),
            ledger_path="runs/ledger.jsonl",
        )

    assert json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))["trial_id"] == "baseline_auto_tiny"
    assert not (tmp_path / "runs" / "baseline" / "decision.json").exists()


def test_autoresearch_comparison_refuses_symlinked_envelope_paths(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.42)
    run_root = tmp_path / "runs" / "autoresearch" / "comparison"
    candidates = run_root / "candidates"
    outside = tmp_path / "outside-candidate"
    candidates.mkdir(parents=True)
    outside.mkdir()
    (candidates / "T123").symlink_to(outside, target_is_directory=True)
    forged = CandidateEnvelope(
        run_id="comparison",
        trial_id="T123",
        root=run_root,
        candidate_dir=candidates / "T123",
        candidate_id="T123",
    )

    with pytest.raises(AutoresearchComparisonError, match="candidate envelope directory"):
        compare_and_write_candidate_decision(
            forged,
            candidate_result=_result("T123", 0.43),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir=baseline.relative_to(tmp_path),
            ledger_path="runs/ledger.jsonl",
        )

    assert not (outside / "metrics.json").exists()


def test_autoresearch_comparison_refuses_invalid_keep_delta(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.42)

    with pytest.raises(AutoresearchComparisonError, match="keep_delta"):
        compare_candidate_result(
            candidate_result=_result("T123", 0.43),
            matched_budget_result=_result("T122", 0.40),
            repo_root=tmp_path,
            baseline_dir=baseline.relative_to(tmp_path),
            ledger_path="runs/ledger.jsonl",
            keep_delta=-0.1,
        )


def test_autoresearch_comparison_missing_matched_baseline_is_unavailable_not_zero(tmp_path: Path) -> None:
    baseline = _write_baseline_lock(tmp_path, score=0.42)

    comparison = compare_candidate_result(
        candidate_result=_result("T123", 0.421),
        matched_budget_result=None,
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path="runs/ledger.jsonl",
        keep_delta=0.001,
    )

    assert comparison.status == "DISCARD"
    assert comparison.matched_budget_score is None
    assert comparison.matched_budget_delta is None
    assert comparison.global_baseline_delta == pytest.approx(0.001)


def _candidate_envelope(tmp_path: Path, trial_id: str):
    create_run_manifest(
        repo_root=tmp_path,
        run_id="comparison",
        base_commit="abc1234",
        planner="manual",
        mode="dry-run",
        description="comparison fixture",
    )
    return create_candidate_envelope(
        repo_root=tmp_path,
        run_id="comparison",
        trial_id=trial_id,
        hypothesis="Local geometry loss should improve matched budget and global current best.",
        trial={"trial_id": trial_id, "candidate_id": trial_id},
    )


def _result(trial_id: str, score: float, *, status: TrialStatus = TrialStatus.SCORED) -> AutoFoldResult:
    return AutoFoldResult(
        trial_id=trial_id,
        status=status,
        candidate_id=f"{trial_id}_candidate",
        metrics={"best_val_calpha_lddt": score},
        fold_cartographer=FoldCartographerReport(signature="comparison_fixture"),
        artifacts={"metrics_json": f"runs/trials/{trial_id}/metrics.json"},
    )


def _write_baseline_lock(tmp_path: Path, *, score: float) -> Path:
    baseline = tmp_path / "runs" / "baseline"
    baseline.mkdir(parents=True)
    metrics = {
        "schema_version": "autoaf3.metrics.v1",
        "status": "SCORED",
        "trial_id": "baseline_auto_tiny",
        "candidate_id": "baseline_lock",
        "split": "public_val_small",
        "official_benchmark_result": True,
        "primary_metric": "best_val_calpha_lddt",
        "scorer_version": "calpha_lddt_v1",
        "max_templates": 0,
        "manifests": {"train_tiny": SHA, "public_val_small": SHA},
        "label_hashes": {"public_val_small": SHA},
        "metrics": {"best_val_calpha_lddt": score},
        "fold_cartographer": {"signature": "baseline_locked", "summary": {}, "buckets": {}},
        "artifacts": {"metrics_json": "runs/baseline/metrics.json"},
    }
    (baseline / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (baseline / "error_report.json").write_text(json.dumps({"scorer_only": True}), encoding="utf-8")
    (baseline / "feature_fingerprints.json").write_text(
        json.dumps(
            {
                "files": {
                    "features/train_tiny.arrow": SHA,
                    "features/public_val_small.arrow": SHA,
                },
                "max_templates": 0,
            }
        ),
        encoding="utf-8",
    )
    return baseline
