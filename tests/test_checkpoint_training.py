from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoalphafold3.checkpoint_runner import (
    APPROVAL_TEXT,
    CheckpointRunError,
    run_one_batch_checkpoint,
)
from autoalphafold3.checkpoint_training import (
    DEFAULT_CHECKPOINT_MANIFEST,
    one_batch_checkpoint_payload,
    run_one_batch_nanofold_checkpoint,
    validate_checkpoint_manifest,
)
from autoalphafold3.local_fixtures import APPROVAL_TOKEN, materialize_local_nanofold_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


class FakeModalCheckpointClient:
    def __init__(self, manifest: dict[str, object]) -> None:
        self.manifest = manifest
        self.payload: dict[str, object] | None = None

    def run_checkpoint(self, payload: dict[str, object]) -> dict[str, object]:
        self.payload = payload
        return self.manifest


def manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "autoaf3.checkpoint_manifest.v1",
        "status": "CHECKPOINT_READY",
        "trial_id": "T010",
        "candidate_id": "one_batch_nanofold_checkpoint",
        "checkpoint_path": "/mnt/autoalphafold3/runs/trials/T010/checkpoint.pt",
        "checkpoint_sha256": SHA,
        "checkpoint_size_bytes": 1234,
        "checkpoint_source": "one_batch_nanofold_training",
        "real_training_performed": True,
        "training_steps": 1,
        "diffusion_steps": 1,
        "max_templates": 0,
        "seed": 0,
        "config_path": "configs/nanofold_dev_cpu_smoke.json",
        "features_path": "/mnt/autoalphafold3/features/train_tiny.arrow",
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "losses": {"total_loss": 1.0},
        "runtime_s": 2.0,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
    }


def test_run_one_batch_checkpoint_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = run_one_batch_checkpoint(repo_root=tmp_path, mode="dry-run")

    assert result.status == "PLANNED"
    assert result.wrote_files == []
    assert result.plan["training_steps"] == 1
    assert result.plan["diffusion_steps"] == 1
    assert result.plan["writes_baseline_dir"] is False
    assert not (tmp_path / "runs").exists()


def test_run_one_batch_checkpoint_requires_exact_approval(tmp_path: Path) -> None:
    with pytest.raises(CheckpointRunError, match=APPROVAL_TEXT):
        run_one_batch_checkpoint(
            repo_root=tmp_path,
            mode="modal",
            approval="yes",
            modal_client=FakeModalCheckpointClient(manifest_payload()),
        )


def test_run_one_batch_checkpoint_records_live_manifest_only(tmp_path: Path) -> None:
    client = FakeModalCheckpointClient(manifest_payload())

    result = run_one_batch_checkpoint(
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        modal_client=client,
    )

    manifest_path = tmp_path / "runs/trials/T010/checkpoint_manifest.json"
    assert result.status == "PASS"
    assert result.wrote_files == [str(manifest_path)]
    assert client.payload == one_batch_checkpoint_payload()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["checkpoint_sha256"] == SHA
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_run_one_batch_checkpoint_rejects_bad_manifest(tmp_path: Path) -> None:
    bad = manifest_payload()
    bad["real_training_performed"] = False

    with pytest.raises(CheckpointRunError, match="real_training_performed"):
        run_one_batch_checkpoint(
            repo_root=tmp_path,
            mode="modal",
            approval=APPROVAL_TEXT,
            modal_client=FakeModalCheckpointClient(bad),
        )


def test_validate_checkpoint_manifest_rejects_search_side_effect_claims() -> None:
    bad = manifest_payload()
    bad["starts_search"] = True

    with pytest.raises(Exception, match="starts_search"):
        validate_checkpoint_manifest(bad)


def test_validate_checkpoint_manifest_rejects_unscoped_checkpoint_path() -> None:
    bad = manifest_payload()
    bad["checkpoint_path"] = "/tmp/checkpoint.pt"

    with pytest.raises(Exception, match="checkpoint_path"):
        validate_checkpoint_manifest(bad)


def test_one_batch_nanofold_checkpoint_writes_real_checkpoint_from_fixture(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    materialize_local_nanofold_fixture(
        repo_root=tmp_path,
        output_dir="features",
        approval=APPROVAL_TOKEN,
    )

    manifest = run_one_batch_nanofold_checkpoint(
        one_batch_checkpoint_payload(features_path="tiny_features.arrow"),
        features_dir=tmp_path / "features",
        output_dir=tmp_path / "runs/trials/T010",
        repo_root=REPO_ROOT,
    )

    checkpoint = tmp_path / "runs/trials/T010/checkpoint.pt"
    manifest_path = tmp_path / "runs/trials/T010" / DEFAULT_CHECKPOINT_MANIFEST
    assert checkpoint.exists()
    assert manifest_path.exists()
    assert manifest["real_training_performed"] is True
    assert manifest["training_steps"] == 1
    assert manifest["diffusion_steps"] == 1
    assert manifest["max_templates"] == 0
    assert manifest["checkpoint_size_bytes"] == checkpoint.stat().st_size


def test_run_one_batch_checkpoint_cli_dry_run_is_structured_json(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "run-one-batch-checkpoint",
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
    assert payload["plan"]["trial_worker"] == "TrialRunner.run_checkpoint"
    assert not (tmp_path / "runs").exists()
