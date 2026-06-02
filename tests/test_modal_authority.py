from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3 import agent
from autoalphafold3.modal_app import modal_event_authority_health
from autoalphafold3.modal_authority import (
    APPROVAL_TEXT,
    ModalAuthorityError,
    audit_modal_event_authority,
)
from autoalphafold3.readiness import build_readiness_report

SHA = "a" * 64


def passed_gates():
    from autoalphafold3.nanofold_checks import NanoFoldGateResult

    return [
        NanoFoldGateResult("parameter_count", "passed", "counted", {"parameter_count": 1}),
        NanoFoldGateResult("tiny_forward", "passed", "finite", {}),
        NanoFoldGateResult("finite_loss", "passed", "finite", {}),
    ]


def write_baseline_lock(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
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
    (baseline / "error_report.json").write_text(json.dumps({"scorer_only": True}), encoding="utf-8")
    (baseline / "feature_fingerprints.json").write_text(
        json.dumps({"files": {"features/train_tiny.arrow": SHA, "features/public_val_small.arrow": SHA}, "max_templates": 0}),
        encoding="utf-8",
    )
    return baseline


def write_calibration(tmp_path: Path) -> Path:
    path = tmp_path / "gate_calibration.json"
    record = {
        "status": "complete",
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "split": "public_val_small",
        "baseline_id": "baseline_auto_tiny",
        "current_best_trial_id": "T000",
        "manifest_hashes": {"train_tiny": SHA, "public_val_small": SHA},
        "feature_fingerprints": {"features/train_tiny.arrow": SHA},
        "gate_thresholds": {"tau_attribution": 0.5, "rho_placebo": 0.5, "k_seed": 2.0},
        "control_evidence_ids": ["knockout", "placebo", "axis", "seed"],
    }
    path.write_text(
        json.dumps({"known_null": {**record, "verdict": "PLACEBO_KILL"}, "known_positive": {**record, "verdict": "CONFIRMED"}}),
        encoding="utf-8",
    )
    return path


class FakeAuthorityClient:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload or modal_event_authority_health()

    def authority_health(self) -> dict[str, object]:
        return self.payload


def test_modal_authority_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = audit_modal_event_authority(repo_root=tmp_path, mode="dry-run")

    assert result.status == "PLANNED"
    assert result.wrote_files == []
    assert result.plan["starts_search"] is False
    assert not (tmp_path / "runs").exists()


def test_modal_authority_requires_exact_approval(tmp_path: Path) -> None:
    with pytest.raises(ModalAuthorityError, match=APPROVAL_TEXT):
        audit_modal_event_authority(
            repo_root=tmp_path,
            mode="modal",
            approval="yes",
            client=FakeAuthorityClient(),
        )


def test_modal_authority_writes_readiness_proof_from_live_payload(tmp_path: Path) -> None:
    result = audit_modal_event_authority(
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        client=FakeAuthorityClient(),
    )

    output = tmp_path / "runs/modal_event_authority.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.status == "PASS"
    assert result.wrote_files == [str(output)]
    assert payload["trusted_orchestrator"] is True
    assert payload["runtime_capabilities"]["post_training_sampler_coordinate_normalization"] is True
    assert payload["starts_search"] is False


def test_modal_authority_rejects_bad_live_payload(tmp_path: Path) -> None:
    payload = modal_event_authority_health()
    payload["starts_search"] = True

    with pytest.raises(ModalAuthorityError, match="starts_search"):
        audit_modal_event_authority(
            repo_root=tmp_path,
            mode="modal",
            approval=APPROVAL_TEXT,
            client=FakeAuthorityClient(payload),
        )


def test_readiness_passes_modal_authority_proof(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    calibration = write_calibration(tmp_path)
    authority = tmp_path / "modal_event_authority.json"
    authority.write_text(json.dumps(modal_event_authority_health()), encoding="utf-8")

    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path=calibration.relative_to(tmp_path),
        modal_authority_path=authority.relative_to(tmp_path),
        nanofold_gates=passed_gates(),
    )

    assert report.modal_event_authority.status == "PASS"
    assert report.modal_event_authority.certification_status == "PASS_LIVE"
    assert report.autonomous_search_ready is True


def test_modal_authority_cli_reexecs_to_repo_venv_when_system_python_lacks_modal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    monkeypatch.setattr(agent, "_current_python_can_import_modal", lambda: False)
    monkeypatch.setattr(agent, "_python_can_import_modal", lambda python: python == venv_python)
    monkeypatch.setattr(agent.sys, "executable", "/usr/bin/python3")
    args = type(
        "Args",
        (),
        {"command": "audit-modal-authority", "mode": "modal", "repo_root": str(tmp_path)},
    )()

    reexec = agent._modal_authority_venv_reexec_argv(
        args,
        ["audit-modal-authority", "--mode", "modal"],
    )

    assert reexec == [
        str(venv_python),
        "-m",
        "autoalphafold3.agent",
        "audit-modal-authority",
        "--mode",
        "modal",
    ]


def test_modal_authority_cli_does_not_reexec_when_current_python_has_modal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "_current_python_can_import_modal", lambda: True)
    args = type(
        "Args",
        (),
        {"command": "audit-modal-authority", "mode": "modal", "repo_root": str(tmp_path)},
    )()

    assert agent._modal_authority_venv_reexec_argv(args, ["audit-modal-authority"]) is None
