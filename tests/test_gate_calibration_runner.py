from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3 import gate_calibration_runner
from autoalphafold3.gate_calibration_runner import (
    APPROVAL_TEXT,
    GateCalibrationRunError,
    run_gate_calibration,
)
from autoalphafold3.gate_wave import GateWaveReport, _evidence_from_result
from autoalphafold3.modal_app import calibration_gate_control_result
from autoalphafold3.schema import TrialStatus

SHA = "a" * 64


def write_baseline_lock(tmp_path: Path) -> Path:
    baseline = tmp_path / "runs/baseline"
    baseline.mkdir(parents=True)
    metrics = {
        "schema_version": "autoaf3.metrics.v1",
        "status": "SCORED",
        "trial_id": "T000",
        "candidate_id": "baseline_auto_tiny",
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
    (baseline / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (baseline / "error_report.json").write_text(json.dumps({"scorer_only": True, "max_templates": 0}), encoding="utf-8")
    (baseline / "feature_fingerprints.json").write_text(
        json.dumps({"files": {"features/train_tiny.arrow": SHA, "features/public_val_small.arrow": SHA}, "max_templates": 0}),
        encoding="utf-8",
    )
    return baseline


def fake_modal_gate_wave(
    controls,
    *,
    modal_module=None,
    class_name="TrialRunner",
    method_name="run_gate_control",
    environment_name=None,
):
    del modal_module, class_name, method_name, environment_name
    evidence = []
    for control in controls:
        raw = calibration_gate_control_result(control.payload, seed=control.seed)
        evidence.append(_evidence_from_result(control, raw))
    return GateWaveReport(candidate_trial_id=controls[0].candidate_trial_id, status=TrialStatus.SCORED, controls=evidence)


def test_gate_calibration_run_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = run_gate_calibration(repo_root=tmp_path, mode="dry-run")

    assert result.status == "PLANNED"
    assert result.wrote_files == []
    assert result.plan["starts_search"] is False
    assert not (tmp_path / "runs").exists()


def test_gate_calibration_run_requires_approval(tmp_path: Path) -> None:
    write_baseline_lock(tmp_path)

    with pytest.raises(GateCalibrationRunError, match=APPROVAL_TEXT):
        run_gate_calibration(repo_root=tmp_path, mode="modal", approval="yes")


def test_gate_calibration_run_writes_known_null_and_positive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_baseline_lock(tmp_path)
    monkeypatch.setattr(gate_calibration_runner, "run_modal_gate_wave", fake_modal_gate_wave)

    result = run_gate_calibration(repo_root=tmp_path, mode="modal", approval=APPROVAL_TEXT)

    null_path = tmp_path / "runs/gate_calibration/known_null.json"
    positive_path = tmp_path / "runs/gate_calibration/known_positive.json"
    known_null = json.loads(null_path.read_text(encoding="utf-8"))
    known_positive = json.loads(positive_path.read_text(encoding="utf-8"))
    assert result.status == "PASS"
    assert result.wrote_files == [str(null_path), str(positive_path)]
    assert known_null["verdict"] != "CONFIRMED"
    assert known_positive["verdict"] == "CONFIRMED"
    assert known_positive["calibration_only"] is True
    assert known_positive["starts_search"] is False
    assert not (tmp_path / "runs/falsification_gate_calibration.json").exists()


def test_gate_calibration_run_refuses_missing_baseline(tmp_path: Path) -> None:
    with pytest.raises(GateCalibrationRunError, match="baseline must be locked"):
        run_gate_calibration(repo_root=tmp_path, mode="modal", approval=APPROVAL_TEXT)
