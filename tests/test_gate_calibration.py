from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoalphafold3.gate_calibration import APPROVAL_TEXT, GateCalibrationError, calibrate_gate

SHA = "a" * 64
REPO_ROOT = Path(__file__).resolve().parents[1]


def calibration_record(*, verdict: str) -> dict[str, object]:
    return {
        "status": "complete",
        "verdict": verdict,
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "split": "public_val_small",
        "baseline_id": "baseline_auto_tiny",
        "current_best_trial_id": "baseline_auto_tiny",
        "manifest_hashes": {"train_tiny": SHA, "public_val_small": SHA},
        "feature_fingerprints": {"features/train_tiny.arrow": SHA},
        "gate_thresholds": {"tau_attribution": 0.5, "rho_placebo": 0.5, "k_seed": 2.0},
        "control_evidence_ids": ["knockout", "placebo", "axis", "seed"],
    }


def write_evidence(tmp_path: Path, *, null_verdict: str = "PLACEBO_KILL", positive_verdict: str = "CONFIRMED") -> tuple[Path, Path]:
    null_path = tmp_path / "known_null.json"
    positive_path = tmp_path / "known_positive.json"
    null_path.write_text(json.dumps(calibration_record(verdict=null_verdict)), encoding="utf-8")
    positive_path.write_text(json.dumps(calibration_record(verdict=positive_verdict)), encoding="utf-8")
    return null_path, positive_path


def test_calibrate_gate_dry_run_plans_without_writing(tmp_path: Path) -> None:
    result = calibrate_gate(repo_root=tmp_path, mode="dry-run")

    assert result.status == "PLANNED"
    assert result.wrote_files == []
    assert result.plan["writes_baseline"] is False
    assert result.plan["writes_ledger"] is False
    assert result.plan["starts_search"] is False
    assert not (tmp_path / "runs").exists()


def test_calibrate_gate_requires_exact_approval(tmp_path: Path) -> None:
    null_path, positive_path = write_evidence(tmp_path)

    with pytest.raises(GateCalibrationError, match=APPROVAL_TEXT):
        calibrate_gate(
            repo_root=tmp_path,
            mode="from-evidence",
            known_null_evidence=null_path.relative_to(tmp_path),
            known_positive_evidence=positive_path.relative_to(tmp_path),
            approval="yes",
        )

    assert not (tmp_path / "runs").exists()


def test_calibrate_gate_writes_valid_readiness_payload_from_real_evidence(tmp_path: Path) -> None:
    null_path, positive_path = write_evidence(tmp_path)

    result = calibrate_gate(
        repo_root=tmp_path,
        mode="from-evidence",
        known_null_evidence=null_path.relative_to(tmp_path),
        known_positive_evidence=positive_path.relative_to(tmp_path),
        approval=APPROVAL_TEXT,
    )

    output_path = tmp_path / "runs/falsification_gate_calibration.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result.status == "PASS"
    assert result.wrote_files == [str(output_path)]
    assert payload["known_null"]["verdict"] == "PLACEBO_KILL"
    assert payload["known_positive"]["verdict"] == "CONFIRMED"
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_calibrate_gate_refuses_unconfirmed_positive(tmp_path: Path) -> None:
    null_path, positive_path = write_evidence(tmp_path, positive_verdict="AXIS_MISS")

    with pytest.raises(GateCalibrationError, match="known_positive"):
        calibrate_gate(
            repo_root=tmp_path,
            mode="from-evidence",
            known_null_evidence=null_path.relative_to(tmp_path),
            known_positive_evidence=positive_path.relative_to(tmp_path),
            approval=APPROVAL_TEXT,
        )


def test_calibrate_gate_refuses_synthetic_fixture_evidence(tmp_path: Path) -> None:
    null_path, positive_path = write_evidence(tmp_path)
    null_payload = json.loads(null_path.read_text(encoding="utf-8"))
    null_payload["synthetic_fixture"] = True
    null_path.write_text(json.dumps(null_payload), encoding="utf-8")

    with pytest.raises(GateCalibrationError, match="synthetic fixture"):
        calibrate_gate(
            repo_root=tmp_path,
            mode="from-evidence",
            known_null_evidence=null_path.relative_to(tmp_path),
            known_positive_evidence=positive_path.relative_to(tmp_path),
            approval=APPROVAL_TEXT,
        )


def test_calibrate_gate_refuses_to_overwrite_output(tmp_path: Path) -> None:
    null_path, positive_path = write_evidence(tmp_path)
    output_path = tmp_path / "runs/falsification_gate_calibration.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(GateCalibrationError, match="already exists"):
        calibrate_gate(
            repo_root=tmp_path,
            mode="from-evidence",
            known_null_evidence=null_path.relative_to(tmp_path),
            known_positive_evidence=positive_path.relative_to(tmp_path),
            approval=APPROVAL_TEXT,
        )


def test_calibrate_gate_cli_dry_run_is_structured_json(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "calibrate-gate",
            "--repo-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "PLANNED"
    assert payload["plan"]["requires_approval"] == APPROVAL_TEXT
    assert not (tmp_path / "runs").exists()
