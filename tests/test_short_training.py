from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoalphafold3.local_fixtures import APPROVAL_TOKEN, materialize_local_nanofold_fixture
from autoalphafold3.short_training import (
    DEFAULT_SHORT_TRAINING_MANIFEST,
    ShortTrainingError,
    run_short_nanofold_training,
    short_training_payload,
    validate_short_training_manifest,
)
from autoalphafold3.short_training_runner import (
    APPROVAL_TEXT,
    ShortTrainingRunError,
    run_short_training,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA = "b" * 64


class FakeModalShortTrainingClient:
    def __init__(self, manifest: dict[str, object]) -> None:
        self.manifest = manifest
        self.payload: dict[str, object] | None = None

    def run_short_training(self, payload: dict[str, object]) -> dict[str, object]:
        self.payload = payload
        return self.manifest


def trial_payload(trial_id: str = "T120", max_steps: int = 3) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "parent_commit": "c13e68b000000000000000000000000000000000",
        "created_at": "2026-06-01T00:00:00Z",
        "agent_session_id": "pytest-short-training",
        "trial_kind": "training",
        "hypothesis": "Bounded short training can produce honest fixture artifacts.",
        "move_family": "geometry_loss",
        "diagnostic_target": "local_geometry_weak",
        "prediction": {
            "causal_component": "local_calpha_geometry_loss",
            "predicted_axis": "local_geometry",
            "predicted_direction": "up",
            "expected_lddt_delta_band": [0.001, 0.01],
        },
        "patch_path": None,
        "config_path": "configs/nanofold_dev_cpu_smoke.json",
        "budget": "smoke",
        "seed": 0,
        "n_res": 32,
        "max_steps": max_steps,
        "max_wall_minutes": 5,
        "manifest_hashes": {},
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "param_cap": 176514,
        "gpu_memory_cap": 80.0,
        "cost_cap": 2.0,
        "timeout_cap": 300,
        "artifact_dir": f"runs/trials/{trial_id}",
        "checkpoint_path": None,
    }


def manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "autoaf3.short_training_manifest.v1",
        "status": "SHORT_TRAINING_READY",
        "trial_id": "T120",
        "candidate_id": "T120",
        "budget": "smoke",
        "real_training_performed": True,
        "local_only": False,
        "official_benchmark_result": False,
        "training_steps": 3,
        "max_steps": 3,
        "max_templates": 0,
        "seed": 0,
        "config_path": "configs/nanofold_dev_cpu_smoke.json",
        "features_path": "/mnt/autoalphafold3/features/tiny_features.arrow",
        "checkpoint_path": "/mnt/autoalphafold3/runs/trials/T120/checkpoint.pt",
        "checkpoint_sha256": SHA,
        "checkpoint_size_bytes": 1234,
        "checkpoint_source": "short_nanofold_training",
        "loss_history_path": "/mnt/autoalphafold3/runs/trials/T120/loss_history.json",
        "training_log_path": "/mnt/autoalphafold3/runs/trials/T120/training_log.json",
        "artifact_manifest_path": "/mnt/autoalphafold3/runs/trials/T120/artifact_manifest.json",
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "final_losses": {"total_loss": 1.0},
        "runtime_s": 1.0,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "reads_locked_labels": False,
    }


def write_trial(path: Path, payload: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or trial_payload(), indent=2) + "\n", encoding="utf-8")


def test_validate_short_training_manifest_accepts_valid_payload() -> None:
    assert validate_short_training_manifest(manifest_payload())["checkpoint_sha256"] == SHA


def test_validate_short_training_manifest_rejects_fake_training_claim() -> None:
    bad = manifest_payload()
    bad["real_training_performed"] = False

    with pytest.raises(ShortTrainingError, match="real_training_performed"):
        validate_short_training_manifest(bad)


def test_validate_short_training_manifest_rejects_official_benchmark_claim() -> None:
    bad = manifest_payload()
    bad["official_benchmark_result"] = True

    with pytest.raises(ShortTrainingError, match="official_benchmark_result"):
        validate_short_training_manifest(bad)


def test_short_training_refuses_max_templates_nonzero(tmp_path: Path) -> None:
    payload = short_training_payload(
        trial_id="T120",
        candidate_id="T120",
        config_path="configs/nanofold_dev_cpu_smoke.json",
        features_path="tiny_features.arrow",
        max_steps=1,
        budget="smoke",
        seed=0,
    )
    payload["max_templates"] = 1

    with pytest.raises(ShortTrainingError, match="max_templates=0"):
        run_short_nanofold_training(
            payload,
            features_dir=tmp_path,
            output_dir=tmp_path / "runs/trials/T120",
            repo_root=REPO_ROOT,
        )


def test_short_training_refuses_unsafe_feature_paths(tmp_path: Path) -> None:
    payload = short_training_payload(
        trial_id="T120",
        candidate_id="T120",
        config_path="configs/nanofold_dev_cpu_smoke.json",
        features_path="../labels.arrow",
        max_steps=1,
        budget="smoke",
        seed=0,
    )

    with pytest.raises(ShortTrainingError, match="safe relative"):
        run_short_nanofold_training(
            payload,
            features_dir=tmp_path,
            output_dir=tmp_path / "runs/trials/T120",
            repo_root=REPO_ROOT,
        )


def test_short_training_refuses_locked_feature_paths(tmp_path: Path) -> None:
    payload = short_training_payload(
        trial_id="T120",
        candidate_id="T120",
        config_path="configs/nanofold_dev_cpu_smoke.json",
        features_path="autoalphafold3-locked/labels/public_val_labels.arrow",
        max_steps=1,
        budget="smoke",
        seed=0,
    )

    with pytest.raises(ShortTrainingError, match="forbidden path"):
        run_short_nanofold_training(
            payload,
            features_dir=tmp_path,
            output_dir=tmp_path / "runs/trials/T120",
            repo_root=REPO_ROOT,
        )


def test_short_training_refuses_non_trial_output_dir(tmp_path: Path) -> None:
    payload = short_training_payload(
        trial_id="T120",
        candidate_id="T120",
        config_path="configs/nanofold_dev_cpu_smoke.json",
        features_path="tiny_features.arrow",
        max_steps=1,
        budget="smoke",
        seed=0,
    )

    with pytest.raises(ShortTrainingError, match="trial-scoped"):
        run_short_nanofold_training(
            payload,
            features_dir=tmp_path,
            output_dir=tmp_path / "runs/not-trials/T120",
            repo_root=REPO_ROOT,
        )


def test_short_training_refuses_non_empty_output_dir(tmp_path: Path) -> None:
    output = tmp_path / "runs/trials/T120"
    output.mkdir(parents=True)
    (output / "user.txt").write_text("do not overwrite\n", encoding="utf-8")
    payload = short_training_payload(
        trial_id="T120",
        candidate_id="T120",
        config_path="configs/nanofold_dev_cpu_smoke.json",
        features_path="tiny_features.arrow",
        max_steps=1,
        budget="smoke",
        seed=0,
    )

    with pytest.raises(ShortTrainingError, match="not empty"):
        run_short_nanofold_training(
            payload,
            features_dir=tmp_path,
            output_dir=output,
            repo_root=REPO_ROOT,
        )


def test_short_training_refuses_steps_above_budget(tmp_path: Path) -> None:
    payload = short_training_payload(
        trial_id="T120",
        candidate_id="T120",
        config_path="configs/nanofold_dev_cpu_smoke.json",
        features_path="tiny_features.arrow",
        max_steps=11,
        budget="smoke",
        seed=0,
    )

    with pytest.raises(ShortTrainingError, match="budget cap"):
        run_short_nanofold_training(
            payload,
            features_dir=tmp_path,
            output_dir=tmp_path / "runs/trials/T120",
            repo_root=REPO_ROOT,
        )


def test_fixture_backed_short_training_writes_honest_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    materialize_local_nanofold_fixture(
        repo_root=tmp_path,
        output_dir="features",
        approval=APPROVAL_TOKEN,
    )

    manifest = run_short_nanofold_training(
        short_training_payload(
            trial_id="T120",
            candidate_id="T120",
            config_path="configs/nanofold_dev_cpu_smoke.json",
            features_path="tiny_features.arrow",
            max_steps=2,
            budget="smoke",
            seed=0,
            local_only=True,
        ),
        features_dir=tmp_path / "features",
        output_dir=tmp_path / "runs/trials/T120",
        repo_root=REPO_ROOT,
        local_only=True,
    )

    output = tmp_path / "runs/trials/T120"
    assert validate_short_training_manifest(manifest)["local_only"] is True
    assert manifest["official_benchmark_result"] is False
    assert manifest["real_training_performed"] is True
    assert manifest["training_steps"] == 2
    assert manifest["max_templates"] == 0
    assert (output / "checkpoint.pt").exists()
    assert (output / DEFAULT_SHORT_TRAINING_MANIFEST).exists()
    assert (output / "loss_history.json").exists()
    assert (output / "artifact_manifest.json").exists()
    assert (output / "training_log.json").exists()
    assert (output / "stdout.log").exists()
    assert (output / "stderr.log").exists()
    assert (output / "patch.diff").exists()
    assert (output / "DONE").exists()
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    loss_history = json.loads((output / "loss_history.json").read_text(encoding="utf-8"))
    assert len(loss_history["losses"]) == 2


def test_run_short_training_dry_run_writes_nothing(tmp_path: Path) -> None:
    trial = tmp_path / "trials/T120.json"
    write_trial(trial)

    result = run_short_training(trial_path=trial.relative_to(tmp_path), repo_root=tmp_path, mode="dry-run")

    assert result.status == "PLANNED"
    assert result.wrote_files == []
    assert result.plan["training_steps"] == 3
    assert result.plan["writes_baseline_dir"] is False
    assert not (tmp_path / "runs").exists()


def test_run_short_training_requires_exact_approval(tmp_path: Path) -> None:
    trial = tmp_path / "trials/T120.json"
    write_trial(trial)

    with pytest.raises(ShortTrainingRunError, match=APPROVAL_TEXT):
        run_short_training(
            trial_path=trial.relative_to(tmp_path),
            repo_root=tmp_path,
            mode="modal",
            approval="yes",
            modal_client=FakeModalShortTrainingClient(manifest_payload()),
        )


def test_run_short_training_records_returned_modal_manifest_only(tmp_path: Path) -> None:
    trial = tmp_path / "trials/T120.json"
    write_trial(trial)
    client = FakeModalShortTrainingClient(manifest_payload())

    result = run_short_training(
        trial_path=trial.relative_to(tmp_path),
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        modal_client=client,
    )

    manifest_path = tmp_path / "runs/trials/T120/short_training_manifest.json"
    assert result.status == "PASS"
    assert result.wrote_files == [str(manifest_path)]
    assert client.payload is not None
    assert client.payload["max_templates"] == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["checkpoint_sha256"] == SHA
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_run_short_training_rejects_bad_returned_modal_manifest(tmp_path: Path) -> None:
    trial = tmp_path / "trials/T120.json"
    write_trial(trial)
    bad = manifest_payload()
    bad["writes_ledger"] = True

    with pytest.raises(ShortTrainingRunError, match="writes_ledger"):
        run_short_training(
            trial_path=trial.relative_to(tmp_path),
            repo_root=tmp_path,
            mode="modal",
            approval=APPROVAL_TEXT,
            modal_client=FakeModalShortTrainingClient(bad),
        )


def test_run_short_training_cli_dry_run_is_structured_json(tmp_path: Path) -> None:
    trial = tmp_path / "trials/T120.json"
    write_trial(trial)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "run-short-training",
            "--repo-root",
            str(tmp_path),
            "--trial",
            "trials/T120.json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "PLANNED"
    assert payload["plan"]["trial_worker"] == "TrialRunner.run"
    assert not (tmp_path / "runs").exists()
