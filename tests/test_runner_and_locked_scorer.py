from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.locked_scorer import LockedScorerError, score_trial_artifacts
from autoalphafold3.runner import (
    RunnerError,
    artifact_manifest_shape,
    initialize_trial_directory,
    plan_trial_artifacts,
    run_fixed_budget_trial,
    safe_child_path,
    validate_artifact_manifest,
    validate_trial_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_toy_predictions(artifact_dir: Path, *, split: str = "smoke") -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "predictions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.predictions.v1",
                "split": split,
                "predictions": [
                    {
                        "target_id": "smoke_A",
                        "predicted_ca": [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [3.0, 0.0, 0.0],
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_runner_artifact_manifest_shape_is_deterministic(tmp_path: Path) -> None:
    first = artifact_manifest_shape(trial_id="T123", output_dir=tmp_path / "T123", features_dir="/features")
    second = artifact_manifest_shape(trial_id="T123", output_dir=tmp_path / "T123", features_dir="/features")

    assert first == second
    assert first["schema_version"] == "autoaf3.artifact_manifest.v1"
    assert first["status"] == "STUB_ONLY"
    assert first["real_training_performed"] is False
    assert first["lifecycle"]["initialized"] is True
    assert "does not represent a NanoFold run" in str(first["disclaimer"])
    validate_artifact_manifest(first)


def test_runner_plans_trial_without_creating_artifacts(tmp_path: Path) -> None:
    plan = plan_trial_artifacts(
        {"trial_id": "T124"},
        features_dir="/features",
        output_root=tmp_path,
    )

    assert plan["status"] == "PLANNED"
    assert plan["lifecycle"]["planned"] is True
    assert not (tmp_path / "T124" / "artifact_manifest.json").exists()


def test_runner_refuses_to_claim_real_training(tmp_path: Path) -> None:
    with pytest.raises(RunnerError, match="not implemented"):
        run_fixed_budget_trial({"trial_id": "T123"}, features_dir="/features", output_dir=tmp_path / "T123")

    manifest = run_fixed_budget_trial(
        {"trial_id": "T123"},
        features_dir="/features",
        output_dir=tmp_path / "T123",
        allow_local_stub=True,
    )

    assert manifest["real_training_performed"] is False
    assert (tmp_path / "T123" / "artifact_manifest.json").exists()
    assert (tmp_path / "T123" / "training_log.json").exists()
    assert (tmp_path / "T123" / "DONE").exists()

    manifest["real_training_performed"] = True
    with pytest.raises(RunnerError, match="must not claim real training"):
        validate_artifact_manifest(manifest)


def test_runner_initialization_is_idempotency_guarded(tmp_path: Path) -> None:
    output = tmp_path / "T125"
    initialize_trial_directory({"trial_id": "T125"}, features_dir="/features", output_dir=output)

    with pytest.raises(RunnerError, match="already completed"):
        initialize_trial_directory({"trial_id": "T125"}, features_dir="/features", output_dir=output)


def test_runner_rejects_unsafe_ids_and_paths(tmp_path: Path) -> None:
    assert validate_trial_id("T001") == "T001"
    with pytest.raises(RunnerError, match="invalid trial_id"):
        validate_trial_id("../bad")
    with pytest.raises(RunnerError, match="unsafe artifact path"):
        safe_child_path(tmp_path, "../escape")


def test_locked_scorer_scores_toy_artifact_directory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "T777"
    predictions = _write_toy_predictions(artifact_dir)

    result = score_trial_artifacts(
        artifact_dir=artifact_dir,
        manifest_path="data/manifests/smoke.json",
        repo_root=REPO_ROOT,
        split="smoke",
        allow_local_smoke=True,
    )

    assert result["status"] == "SCORED"
    assert result["scorer_version"] == "calpha_lddt_v1"
    assert result["primary_metric"] == "best_val_calpha_lddt"
    assert result["official_benchmark_result"] is False
    assert result["local_only"] is True
    assert result["metrics"]["best_val_calpha_lddt"] == pytest.approx(1.0)
    assert result["fold_cartographer"]["signature"] == "toy_geometry_preserved"
    assert result["artifacts"]["predictions_json"] == str(predictions)
    assert result["error_report"]["scorer_only"] is True
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "error_report.json").exists()


def test_locked_scorer_missing_prediction_artifact_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "T778"
    artifact_dir.mkdir()

    result = score_trial_artifacts(
        artifact_dir=artifact_dir,
        manifest_path="data/manifests/smoke.json",
        repo_root=REPO_ROOT,
        split="smoke",
        allow_local_smoke=True,
    )

    assert result["status"] == "FAIL"
    assert result["error_report"]["failure_signature"] == "prediction_artifact_missing"
    assert result["fold_cartographer"]["signature"] == "prediction_artifact_missing"


def test_locked_scorer_refuses_unsupported_split(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "T779"
    _write_toy_predictions(artifact_dir, split="train_tiny")

    with pytest.raises(PermissionError, match="unsupported scorer-only split"):
        score_trial_artifacts(
            artifact_dir=artifact_dir,
            manifest_path="data/manifests/smoke.json",
            repo_root=REPO_ROOT,
            split="train_tiny",
        )


def test_locked_scorer_rejects_bad_prediction_schema(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "T780"
    artifact_dir.mkdir()
    (artifact_dir / "predictions.json").write_text(
        json.dumps({"schema_version": "wrong", "split": "smoke", "predictions": []}),
        encoding="utf-8",
    )

    with pytest.raises(LockedScorerError, match="schema_version"):
        score_trial_artifacts(
            artifact_dir=artifact_dir,
            manifest_path="data/manifests/smoke.json",
            repo_root=REPO_ROOT,
            split="smoke",
            allow_local_smoke=True,
        )


def test_locked_scorer_rejects_duplicate_targets_and_bad_shapes(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "T781"
    artifact_dir.mkdir()
    (artifact_dir / "predictions.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.predictions.v1",
                "split": "smoke",
                "predictions": [
                    {"target_id": "smoke_A", "predicted_ca": [[0.0, 0.0, 0.0]]},
                    {"target_id": "smoke_A", "predicted_ca": [[0.0, 0.0, 0.0]]},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LockedScorerError, match="duplicate prediction"):
        score_trial_artifacts(
            artifact_dir=artifact_dir,
            manifest_path="data/manifests/smoke.json",
            repo_root=REPO_ROOT,
            split="smoke",
            allow_local_smoke=True,
        )

    (artifact_dir / "predictions.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.predictions.v1",
                "split": "smoke",
                "predictions": [{"target_id": "smoke_A", "predicted_ca": [0.0, 1.0, 2.0]}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LockedScorerError, match="shape"):
        score_trial_artifacts(
            artifact_dir=artifact_dir,
            manifest_path="data/manifests/smoke.json",
            repo_root=REPO_ROOT,
            split="smoke",
            allow_local_smoke=True,
        )
