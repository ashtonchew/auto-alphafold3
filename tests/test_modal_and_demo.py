from __future__ import annotations

import json
import subprocess
import sys
import types
import builtins
from pathlib import Path

import pytest

import autoalphafold3.agent as agent_cli
from autoalphafold3.ledger import read_ledger
from autoalphafold3.orchestrator import poll_trial, submit_trial
from autoalphafold3.render_overlay import ca_trace_to_pdb, render_sample_overlay
from autoalphafold3.render_trajectory import render_trajectory
from autoalphafold3.scorer.locked_dataset import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_MANIFEST = "data/manifests/smoke.json"


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _trial_payload(trial_id: str = "T010") -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "parent_commit": _head(),
        "agent_session_id": "pytest-modal",
        "trial_kind": "training",
        "hypothesis": "Modal mode should use explicit mocked submission.",
        "move_family": "geometry_loss",
        "diagnostic_target": "local_geometry_weak",
        "prediction": {
            "causal_component": "mocked Modal submission fixture",
            "predicted_axis": "local_geometry",
            "predicted_direction": "up",
            "expected_lddt_delta_band": [0.01, 0.05],
        },
        "patch_path": None,
        "config_path": "configs/auto_tiny.json",
        "budget": "dry_run",
        "seed": 0,
        "max_steps": 1,
        "max_wall_minutes": 1,
        "manifest_hashes": {"smoke": sha256_file(REPO_ROOT / SMOKE_MANIFEST)},
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "param_cap": 1,
        "gpu_memory_cap": 0.0,
        "cost_cap": 0.0,
        "timeout_cap": 60,
        "artifact_dir": f"runs/{trial_id}_artifacts",
    }


def _write_trial(tmp_path: Path, trial_id: str = "T010") -> Path:
    path = tmp_path / f"{trial_id}.json"
    path.write_text(json.dumps(_trial_payload(trial_id)), encoding="utf-8")
    return path


def test_agent_strict_preflight_enables_strict_nanofold_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_submit_trial(*args: object, **kwargs: object) -> str:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "dryrun:T010"

    monkeypatch.setattr(agent_cli, "submit_trial", fake_submit_trial)

    rc = agent_cli.main(["submit", "trial.json", "--strict-preflight"])

    assert rc == 0
    assert captured["kwargs"]["enforce_git_diff"] is True
    assert captured["kwargs"]["strict_nanofold_gates"] is True


def test_modal_submit_and_poll_with_mocked_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeCall:
        object_id = "fake-call-123"

        def get(self, timeout: int) -> dict[str, object]:
            assert timeout == 0
            return {
                "trial_id": "T010",
                "status": "SCORED",
                "candidate_id": "mocked_modal",
                "metrics": {"best_val_calpha_lddt": 0.5},
                "fold_cartographer": {"signature": "mocked_modal_scored", "summary": {}, "buckets": {}},
                "artifacts": {"metrics_json": "/runs/trials/T010/metrics.json"},
                "postmortem": "Mocked Modal result for contract test.",
            }

    class FakeRunMethod:
        def spawn(self, payload: dict[str, object]) -> FakeCall:
            assert payload["trial_id"] == "T010"
            return FakeCall()

    class FakeRunner:
        run = FakeRunMethod()

    class FakeCls:
        @staticmethod
        def from_name(app_name: str, class_name: str) -> type[FakeRunner]:
            assert app_name == "autoalphafold3-modal"
            assert class_name == "TrialRunner"
            return FakeRunner

    class FakeFunctionCall:
        @staticmethod
        def from_id(object_id: str) -> FakeCall:
            assert object_id == "fake-call-123"
            return FakeCall()

    fake_modal = types.SimpleNamespace(Cls=FakeCls, FunctionCall=FakeFunctionCall)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    ledger_path = tmp_path / "ledger.jsonl"
    call_id = submit_trial(
        _write_trial(tmp_path),
        repo_root=REPO_ROOT,
        ledger_path=ledger_path,
        manifest_paths={"smoke": SMOKE_MANIFEST},
        mode="modal",
    )
    result = poll_trial(call_id, repo_root=REPO_ROOT, ledger_path=ledger_path)
    rows = read_ledger(ledger_path=ledger_path)

    assert call_id == "modal:fake-call-123"
    assert result.status == "SCORED"
    assert result.metrics["best_val_calpha_lddt"] == 0.5
    assert rows[-1].candidate_id == "mocked_modal"


def test_modal_submit_spawn_error_normalizes_to_infra_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeCls:
        @staticmethod
        def from_name(app_name: str, class_name: str) -> object:
            raise RuntimeError(f"cannot spawn {app_name}.{class_name}")

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Cls=FakeCls))
    ledger_path = tmp_path / "ledger.jsonl"

    call_id = submit_trial(
        _write_trial(tmp_path, trial_id="T012"),
        repo_root=REPO_ROOT,
        ledger_path=ledger_path,
        manifest_paths={"smoke": SMOKE_MANIFEST},
        mode="modal",
    )
    result = poll_trial(call_id, repo_root=REPO_ROOT, ledger_path=ledger_path)

    assert call_id == "modal:INFRA_FAIL:T012"
    assert result.status == "INFRA_FAIL"
    assert result.failure_signature == "modal_RuntimeError"


def test_modal_mode_records_infra_fail_when_sdk_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "modal":
            raise ModuleNotFoundError("No module named 'modal'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ledger_path = tmp_path / "ledger.jsonl"

    call_id = submit_trial(
        _write_trial(tmp_path, trial_id="T011"),
        repo_root=REPO_ROOT,
        ledger_path=ledger_path,
        manifest_paths={"smoke": SMOKE_MANIFEST},
        mode="modal",
    )
    result = poll_trial(call_id, repo_root=REPO_ROOT, ledger_path=ledger_path)

    assert call_id == "modal:INFRA_FAIL:T011"
    assert result.status == "INFRA_FAIL"
    assert result.failure_signature == "modal_sdk_missing"


def test_trajectory_renderer_marks_sample_data(tmp_path: Path) -> None:
    output = render_trajectory(tmp_path / "missing.jsonl", tmp_path / "trajectory.html", sample=True)
    html = output.read_text(encoding="utf-8")

    assert "Sample/local dry-run data only. Not benchmark results." in html
    assert "T003" in html
    assert "toy_geometry_preserved" in html


def test_overlay_renderer_marks_sample_data_and_embeds_pdb(tmp_path: Path) -> None:
    pdb = ca_trace_to_pdb([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    output = render_sample_overlay(tmp_path / "overlay.html")
    html = output.read_text(encoding="utf-8")

    assert "ATOM" in pdb
    assert "Sample/local C-alpha traces only. Not benchmark results." in html
    assert "3Dmol" in html
    assert "ATOM" in html


def test_agent_manifest_validation_cli_for_smoke_and_empty_templates() -> None:
    smoke = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "validate-manifest",
            "data/manifests/smoke.json",
        ],
        cwd=REPO_ROOT,
        text=True,
    )
    smoke_report = json.loads(smoke)
    assert smoke_report[0]["entry_count"] == 1
    assert smoke_report[0]["verified_assets"] is True

    templates = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "validate-manifest",
            "data/manifests/train_tiny.template.json",
            "data/manifests/public_val_small.template.json",
            "--allow-empty",
            "--no-verify-assets",
        ],
        cwd=REPO_ROOT,
        text=True,
    )
    template_report = json.loads(templates)
    assert [row["entry_count"] for row in template_report] == [0, 0]
    assert all(row["manifest_kind"] == "official_template" for row in template_report)
