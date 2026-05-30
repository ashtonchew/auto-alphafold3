from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.baseline_readiness import (
    BaselineReadinessError,
    audit_baseline_readiness,
    current_best_from_baseline_and_ledger,
)
from autoalphafold3.ledger import append_ledger
from autoalphafold3.schema import AutoFoldResult, FoldCartographerReport, TrialStatus


SHA = "a" * 64


def write_baseline_lock(tmp_path: Path, *, metrics_overrides: dict[str, object] | None = None) -> Path:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    metrics: dict[str, object] = {
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
        "metrics": {"best_val_calpha_lddt": 0.42},
        "fold_cartographer": {"signature": "baseline_locked", "summary": {}, "buckets": {}},
        "artifacts": {"metrics_json": "runs/baseline/metrics.json"},
    }
    if metrics_overrides:
        metrics.update(metrics_overrides)
    (baseline / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (baseline / "error_report.json").write_text(
        json.dumps({"scorer_only": True, "failure_signature": None}),
        encoding="utf-8",
    )
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


def test_baseline_readiness_passes_with_complete_tmp_lock(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)

    report = audit_baseline_readiness(baseline_dir=baseline)

    assert report.status == "PASS"
    assert report.baseline_score == pytest.approx(0.42)
    assert report.official_benchmark_result is True
    assert report.manifest_hashes_valid is True
    assert report.feature_fingerprints_valid is True
    assert report.max_templates_zero is True
    assert report.to_dict()["current_best_score"] == pytest.approx(0.42)


def test_baseline_readiness_fails_when_required_files_missing(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()

    before = sorted(path.name for path in baseline.iterdir())
    report = audit_baseline_readiness(baseline_dir=baseline)
    after = sorted(path.name for path in baseline.iterdir())

    assert report.status == "FAIL"
    assert "baseline metrics.json is missing" in report.problems
    assert "baseline error_report.json is missing" in report.problems
    assert "baseline feature_fingerprints.json is missing" in report.problems
    assert report.pending_human_action is not None
    assert before == after


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"schema_version": "wrong"}, "schema_version"),
        ({"official_benchmark_result": False}, "official_benchmark_result=true"),
        ({"local_only": True}, "local_only"),
        ({"primary_metric": "loss"}, "primary_metric"),
        ({"scorer_version": "wrong"}, "scorer_version"),
        ({"split": "smoke"}, "public_val_small"),
        ({"status": "FAIL"}, "status must be SCORED"),
        ({"metrics": {}}, "best_val_calpha_lddt"),
        ({"metrics": {"best_val_calpha_lddt": float("nan")}}, "best_val_calpha_lddt"),
        ({"metrics": {"best_val_calpha_lddt": True}}, "best_val_calpha_lddt"),
        ({"metrics": {"best_val_calpha_lddt": 1.5}}, "[0, 1]"),
        ({"manifests": {"train_tiny": SHA}}, "manifest hashes"),
        ({"max_templates": 1}, "max_templates=0"),
        ({"fold_cartographer": None}, "fold_cartographer"),
    ],
)
def test_baseline_readiness_rejects_invalid_metrics(tmp_path: Path, override: dict[str, object], problem: str) -> None:
    baseline = write_baseline_lock(tmp_path, metrics_overrides=override)

    report = audit_baseline_readiness(baseline_dir=baseline)

    assert report.status == "FAIL"
    assert any(problem in item for item in report.problems)


def test_baseline_readiness_rejects_invalid_error_report(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    (baseline / "error_report.json").write_text(json.dumps({"scorer_only": False}), encoding="utf-8")

    report = audit_baseline_readiness(baseline_dir=baseline)

    assert report.status == "FAIL"
    assert any("scorer_only=true" in item for item in report.problems)


def test_baseline_readiness_rejects_invalid_feature_fingerprints(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    (baseline / "feature_fingerprints.json").write_text(json.dumps({"files": {}}), encoding="utf-8")

    report = audit_baseline_readiness(baseline_dir=baseline)

    assert report.status == "FAIL"
    assert any("feature_fingerprints.json" in item for item in report.problems)


def test_baseline_readiness_rejects_missing_public_val_feature_fingerprint(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    (baseline / "feature_fingerprints.json").write_text(
        json.dumps({"files": {"features/train_tiny.arrow": SHA}, "max_templates": 0}),
        encoding="utf-8",
    )

    report = audit_baseline_readiness(baseline_dir=baseline)

    assert report.status == "FAIL"
    assert any("features/public_val_small.arrow" in item for item in report.problems)


def test_baseline_readiness_rejects_missing_label_hashes(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, metrics_overrides={"label_hashes": {}})

    report = audit_baseline_readiness(baseline_dir=baseline)

    assert report.status == "FAIL"
    assert any("label hashes" in item for item in report.problems)


def test_baseline_readiness_rejects_artifacts_outside_baseline_dir(tmp_path: Path) -> None:
    baseline = write_baseline_lock(
        tmp_path,
        metrics_overrides={"artifacts": {"metrics_json": "runs/trials/T001/metrics.json"}},
    )

    report = audit_baseline_readiness(baseline_dir=baseline)

    assert report.status == "FAIL"
    assert any("runs/baseline" in item for item in report.problems)


def test_baseline_readiness_rejects_missing_identity_fields(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, metrics_overrides={"trial_id": "", "candidate_id": ""})

    report = audit_baseline_readiness(baseline_dir=baseline)

    assert report.status == "FAIL"
    assert any("trial_id" in item for item in report.problems)
    assert any("candidate_id" in item for item in report.problems)


def test_baseline_readiness_reports_non_object_json_as_not_ready(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    (baseline / "metrics.json").write_text("[]", encoding="utf-8")

    report = audit_baseline_readiness(baseline_dir=baseline)

    assert report.status == "FAIL"
    assert any("baseline lock JSON is unreadable" in item for item in report.problems)


def test_current_best_lookup_refuses_without_ready_baseline(tmp_path: Path) -> None:
    with pytest.raises(BaselineReadinessError, match="baseline is not ready"):
        current_best_from_baseline_and_ledger(
            baseline_dir=tmp_path / "missing-baseline",
            ledger_path=tmp_path / "ledger.jsonl",
        )


def test_current_best_lookup_returns_baseline_when_no_keep_rows(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)

    best = current_best_from_baseline_and_ledger(
        baseline_dir=baseline,
        ledger_path=tmp_path / "missing-ledger.jsonl",
    )

    assert best.source == "baseline"
    assert best.score == pytest.approx(0.42)


def test_current_best_lookup_prefers_valid_keep_above_baseline(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    append_ledger(_result("T200", TrialStatus.FAIL, 0.99), ledger_path=ledger)
    append_ledger(_result("T201", TrialStatus.DISCARD, 0.98), ledger_path=ledger)
    append_ledger(_result("T202", TrialStatus.KEEP, 0.43), ledger_path=ledger)

    best = current_best_from_baseline_and_ledger(baseline_dir=baseline, ledger_path=ledger)

    assert best.source == "ledger_keep"
    assert best.trial_id == "T202"
    assert best.score == pytest.approx(0.43)


def test_current_best_lookup_ignores_keep_below_baseline(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    append_ledger(_result("T203", TrialStatus.KEEP, 0.41), ledger_path=ledger)

    best = current_best_from_baseline_and_ledger(baseline_dir=baseline, ledger_path=ledger)

    assert best.source == "baseline"
    assert best.score == pytest.approx(0.42)


def test_current_best_lookup_ignores_out_of_range_keep_score(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    append_ledger(_result("T204", TrialStatus.KEEP, 2.0), ledger_path=ledger)

    best = current_best_from_baseline_and_ledger(baseline_dir=baseline, ledger_path=ledger)

    assert best.source == "baseline"
    assert best.score == pytest.approx(0.42)


def _result(trial_id: str, status: TrialStatus, score: float) -> AutoFoldResult:
    return AutoFoldResult(
        trial_id=trial_id,
        status=status,
        candidate_id=f"{trial_id}_candidate",
        metrics={"best_val_calpha_lddt": score},
        fold_cartographer=FoldCartographerReport(signature="synthetic_contract_fixture"),
    )
