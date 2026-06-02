from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.checkpoint_training import one_batch_checkpoint_payload, run_one_batch_nanofold_checkpoint
from autoalphafold3.local_fixtures import APPROVAL_TOKEN, materialize_local_nanofold_fixture
from autoalphafold3.runner import validate_prediction_artifact
from autoalphafold3.runner import validate_artifact_manifest
from autoalphafold3.sampler import (
    SamplerError,
    _label_free_ca_quality,
    _normalize_ca_coordinates,
    _sampler_settings,
    run_checkpoint_prediction_artifacts,
    run_sampler_trial,
)
from autoalphafold3.short_training import run_short_nanofold_training, short_training_payload

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_sampler_trial_loads_checkpoint_and_writes_predictions(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    materialize_local_nanofold_fixture(
        repo_root=tmp_path,
        output_dir="features",
        approval=APPROVAL_TOKEN,
    )
    run_one_batch_nanofold_checkpoint(
        one_batch_checkpoint_payload(features_path="tiny_features.arrow"),
        features_dir=tmp_path / "features",
        output_dir=tmp_path / "runs/trials/T010",
        repo_root=REPO_ROOT,
    )

    manifest = run_sampler_trial(
        {
            "trial_id": "T011",
            "trial_kind": "sampler",
            "checkpoint_path": str(tmp_path / "runs/trials/T010/checkpoint.pt"),
            "seed": 0,
            "sampler_steps": 1,
        },
        features_dir=tmp_path / "features",
        output_dir=tmp_path / "runs/trials/T011",
        repo_root=REPO_ROOT,
        split="smoke",
    )

    output = tmp_path / "runs/trials/T011"
    predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    artifact_manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    validate_prediction_artifact(predictions)
    validate_artifact_manifest(artifact_manifest)
    assert manifest["status"] == "SAMPLER_PREDICTED"
    assert manifest["inference_only"] is True
    assert manifest["real_training_performed"] is False
    assert manifest["max_templates"] == 0
    assert manifest["starts_search"] is False
    assert artifact_manifest["runner_mode"] == "frozen_checkpoint_sampler"
    assert predictions["source"] == "frozen_checkpoint_nanofold_sampler"
    assert predictions["max_templates"] == 0
    assert manifest["sampler_coordinate_normalization"] == "none"
    assert predictions["predictions"][0]["target_id"] == "TARGET_0_A"
    assert len(predictions["predictions"][0]["predicted_ca"]) > 0
    assert (output / "DONE").exists()


def test_run_sampler_trial_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(SamplerError, match="checkpoint_path does not exist"):
        run_sampler_trial(
            {
                "trial_id": "T011",
                "trial_kind": "sampler",
                "checkpoint_path": str(tmp_path / "runs/trials/T010/checkpoint.pt"),
            },
            features_dir=tmp_path / "features",
            output_dir=tmp_path / "runs/trials/T011",
            repo_root=REPO_ROOT,
        )


def test_run_sampler_trial_rejects_training_steps(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs/trials/T010/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not-a-real-checkpoint")

    with pytest.raises(SamplerError, match="must not set max_steps"):
        run_sampler_trial(
            {
                "trial_id": "T011",
                "trial_kind": "sampler",
                "checkpoint_path": str(checkpoint),
                "max_steps": 1,
            },
            features_dir=tmp_path / "features",
            output_dir=tmp_path / "runs/trials/T011",
            repo_root=REPO_ROOT,
        )


def test_checkpoint_prediction_artifacts_accept_short_training_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    materialize_local_nanofold_fixture(
        repo_root=tmp_path,
        output_dir="features",
        approval=APPROVAL_TOKEN,
    )
    training_manifest = run_short_nanofold_training(
        short_training_payload(
            trial_id="T120",
            candidate_id="T120",
            config_path="configs/nanofold_dev_cpu_smoke.json",
            features_path="tiny_features.arrow",
            max_steps=1,
            budget="smoke",
            seed=0,
            local_only=True,
        ),
        features_dir=tmp_path / "features",
        output_dir=tmp_path / "runs/trials/T120",
        repo_root=REPO_ROOT,
        local_only=True,
    )

    sampler_manifest = run_checkpoint_prediction_artifacts(
        {
            "trial_id": "T120",
            "candidate_id": "T120",
            "checkpoint_path": training_manifest["checkpoint_path"],
            "seed": 0,
            "sampler_steps": 1,
        },
        features_dir=tmp_path / "features",
        output_dir=tmp_path / "runs/trials/T120",
        repo_root=REPO_ROOT,
        split="smoke",
    )

    output = tmp_path / "runs/trials/T120"
    predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    artifact_manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    validate_prediction_artifact(predictions)
    assert sampler_manifest["status"] == "SHORT_TRAINING_PREDICTED"
    assert sampler_manifest["real_training_performed"] is True
    assert sampler_manifest["inference_only"] is True
    assert sampler_manifest["max_templates"] == 0
    assert predictions["source"] == "short_training_checkpoint_nanofold_sampler"
    assert predictions["candidate_id"] == "T120"
    assert artifact_manifest["status"] == "SHORT_TRAINING_PREDICTED"
    assert artifact_manifest["predictions_ready"] is True
    assert (output / "short_training_manifest.json").exists()
    assert (output / "checkpoint.pt").exists()
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_sampler_ca_bond_coordinate_normalization_rescales_exploded_trace() -> None:
    exploded = [[0.0, 0.0, 0.0], [380.0, 0.0, 0.0], [760.0, 0.0, 0.0]]

    normalized = _normalize_ca_coordinates(exploded, policy="ca_bond")

    assert normalized[0] == pytest.approx([-3.8, 0.0, 0.0])
    assert normalized[1] == pytest.approx([0.0, 0.0, 0.0])
    assert normalized[2] == pytest.approx([3.8, 0.0, 0.0])
    assert _label_free_ca_quality(normalized, policy="geometry") == pytest.approx(0.0)


def test_sampler_rejects_unknown_coordinate_normalization() -> None:
    with pytest.raises(SamplerError, match="sampler_coordinate_normalization must be none or ca_bond"):
        _sampler_settings(
            {
                "sampler_steps": 1,
                "sampler_coordinate_normalization": "bad",
            }
        )


def test_run_sampler_trial_accepts_short_training_checkpoint_manifest(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    materialize_local_nanofold_fixture(
        repo_root=tmp_path,
        output_dir="features",
        approval=APPROVAL_TOKEN,
    )
    training_manifest = run_short_nanofold_training(
        short_training_payload(
            trial_id="T120",
            candidate_id="T120",
            config_path="configs/nanofold_dev_cpu_smoke.json",
            features_path="tiny_features.arrow",
            max_steps=1,
            budget="smoke",
            seed=0,
            local_only=True,
        ),
        features_dir=tmp_path / "features",
        output_dir=tmp_path / "runs/trials/T120",
        repo_root=REPO_ROOT,
        local_only=True,
    )

    sampler_manifest = run_sampler_trial(
        {
            "trial_id": "T121",
            "trial_kind": "sampler",
            "checkpoint_path": training_manifest["checkpoint_path"],
            "seed": 0,
            "sampler_steps": 1,
        },
        features_dir=tmp_path / "features",
        output_dir=tmp_path / "runs/trials/T121",
        repo_root=REPO_ROOT,
        split="smoke",
    )

    output = tmp_path / "runs/trials/T121"
    predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    validate_prediction_artifact(predictions)
    assert sampler_manifest["status"] == "SAMPLER_PREDICTED"
    assert sampler_manifest["checkpoint_source_trial_id"] == "T120"
    assert sampler_manifest["real_training_performed"] is False
    assert sampler_manifest["inference_only"] is True
