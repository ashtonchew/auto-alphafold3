from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.locked_scorer import LockedScorerError, load_locked_state, score_trial_artifacts
from autoalphafold3.scorer.locked_dataset import sha256_file
from autoalphafold3.runner import (
    RunnerError,
    artifact_manifest_shape,
    initialize_trial_directory,
    plan_trial_artifacts,
    prediction_artifact_shape,
    run_sequence_linear_baseline,
    run_fixed_budget_trial,
    safe_child_path,
    validate_artifact_manifest,
    validate_prediction_artifact,
    validate_trial_id,
    write_prediction_artifact,
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
    assert not (tmp_path / "T123" / "checkpoint.pt").exists()
    assert not (tmp_path / "T123" / "predictions.json").exists()

    manifest["real_training_performed"] = True
    with pytest.raises(RunnerError, match="must not claim real training"):
        validate_artifact_manifest(manifest)


def test_runner_manifest_rejects_artifact_paths_outside_trial_dir(tmp_path: Path) -> None:
    manifest = artifact_manifest_shape(
        trial_id="T132",
        output_dir=tmp_path / "T132",
        features_dir="/features",
    )
    artifacts = dict(manifest["artifacts"])
    artifacts["predictions_json"] = str(tmp_path / "other" / "predictions.json")
    manifest["artifacts"] = artifacts

    with pytest.raises(RunnerError, match="escapes trial directory"):
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
    with pytest.raises(RunnerError, match="trial-scoped"):
        artifact_manifest_shape(trial_id="T001", output_dir=tmp_path, features_dir="/features")


def test_prediction_artifact_writer_creates_canonical_non_official_payload(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "T126"
    payload = write_prediction_artifact(
        trial_id="T126",
        split="smoke",
        output_dir=artifact_dir,
        predictions=[
            {
                "target_id": "smoke_A",
                "predicted_ca": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
            }
        ],
    )

    assert payload["schema_version"] == "autoaf3.predictions.v1"
    assert payload["official_benchmark_result"] is False
    assert "not a benchmark result" in payload["disclaimer"]
    validate_prediction_artifact(payload)
    assert json.loads((artifact_dir / "predictions.json").read_text()) == payload


def test_prediction_artifact_validation_rejects_official_or_bad_shapes() -> None:
    payload = prediction_artifact_shape(
        trial_id="T127",
        split="smoke",
        predictions=[{"target_id": "smoke_A", "predicted_ca": [[0.0, 0.0, 0.0]]}],
    )
    payload["official_benchmark_result"] = True
    with pytest.raises(RunnerError, match="official benchmark"):
        validate_prediction_artifact(payload)

    with pytest.raises(RunnerError, match="shape"):
        prediction_artifact_shape(
            trial_id="T128",
            split="smoke",
            predictions=[{"target_id": "smoke_A", "predicted_ca": [0.0, 0.0, 0.0]}],
        )
    with pytest.raises(RunnerError, match="non-finite"):
        prediction_artifact_shape(
            trial_id="T128",
            split="smoke",
            predictions=[{"target_id": "smoke_A", "predicted_ca": [[float("nan"), 0.0, 0.0]]}],
        )
    with pytest.raises(RunnerError, match="at least one prediction"):
        prediction_artifact_shape(trial_id="T129", split="smoke", predictions=[])
    with pytest.raises(RunnerError, match="duplicate prediction"):
        prediction_artifact_shape(
            trial_id="T130",
            split="smoke",
            predictions=[
                {"target_id": "smoke_A", "predicted_ca": [[0.0, 0.0, 0.0]]},
                {"target_id": "smoke_A", "predicted_ca": [[0.0, 0.0, 0.0]]},
            ],
        )
    with pytest.raises(RunnerError, match="requires split"):
        prediction_artifact_shape(
            trial_id="T131",
            split="",
            predictions=[{"target_id": "smoke_A", "predicted_ca": [[0.0, 0.0, 0.0]]}],
        )


def test_sequence_linear_baseline_reads_public_features_without_labels(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    ipc = pytest.importorskip("pyarrow.ipc")
    features = tmp_path / "features"
    features.mkdir()
    schema = pa.schema(
        [
            ("record_id", pa.string()),
            ("sequence_length", pa.int32()),
        ]
    )
    table = pa.Table.from_pylist(
        [{"record_id": "target_A", "sequence_length": 3}],
        schema=schema,
    )
    with (features / "public_val_small.arrow").open("wb") as handle:
        with ipc.new_file(handle, schema) as writer:
            writer.write_table(table)

    manifest = run_sequence_linear_baseline(
        {"trial_id": "T900", "candidate_id": "baseline_auto_tiny", "max_templates": 0},
        features_dir=features,
        output_dir=tmp_path / "T900",
    )

    payload = json.loads((tmp_path / "T900" / "predictions.json").read_text(encoding="utf-8"))
    assert manifest["runner_mode"] == "sequence_linear_baseline"
    assert payload["candidate_id"] == "baseline_auto_tiny"
    assert payload["predictions"][0]["predicted_ca"] == [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]]
    assert not (tmp_path / "T900" / "checkpoint.pt").exists()


def test_locked_scorer_scores_toy_artifact_directory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "T777"
    payload = write_prediction_artifact(
        trial_id="T777",
        split="smoke",
        output_dir=artifact_dir,
        predictions=[
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
    )
    predictions = artifact_dir / "predictions.json"

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
    assert result["fold_cartographer"]["summary"]["canonical_target"] == "local_geometry_weak"
    assert result["artifacts"]["predictions_json"] == str(predictions)
    assert payload["official_benchmark_result"] is False
    assert result["error_report"]["scorer_only"] is True
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "error_report.json").exists()


def test_locked_scorer_accepts_preloaded_state_for_local_smoke(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "T782"
    _write_toy_predictions(artifact_dir)
    state = load_locked_state(REPO_ROOT)
    state = state.__class__(
        locked_root=state.locked_root,
        manifest_path="data/manifests/smoke.json",
        scorer_version_path="missing-version.txt",
    )

    result = score_trial_artifacts(
        artifact_dir=artifact_dir,
        manifest_path="ignored.json",
        split="smoke",
        allow_local_smoke=True,
        locked=state,
    )

    assert result["status"] == "SCORED"
    assert result["official_benchmark_result"] is False


def test_locked_scorer_scores_arrow_labels_as_official_with_locked_state(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    ipc = pytest.importorskip("pyarrow.ipc")
    locked = tmp_path / "locked"
    labels = locked / "labels"
    manifests = locked / "manifests"
    labels.mkdir(parents=True)
    manifests.mkdir()
    label_schema = pa.schema(
        [
            ("record_id", pa.string()),
            ("pdb_id", pa.string()),
            ("chain_id", pa.string()),
            ("sequence_length", pa.int32()),
            ("ca_positions", pa.list_(pa.list_(pa.float32()))),
            ("ca_mask", pa.list_(pa.bool_())),
        ]
    )
    label_path = labels / "public_val_labels.arrow"
    label_table = pa.Table.from_pylist(
        [
            {
                "record_id": "target_A",
                "pdb_id": "1ABC",
                "chain_id": "A",
                "sequence_length": 3,
                "ca_positions": [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]],
                "ca_mask": [True, True, True],
            }
        ],
        schema=label_schema,
    )
    with label_path.open("wb") as handle:
        with ipc.new_file(handle, label_schema) as writer:
            writer.write_table(label_table)
    feature = tmp_path / "feature.arrow"
    feature.write_text("feature", encoding="utf-8")
    manifest = {
        "manifest_kind": "locked_manifest",
        "schema_version": "autoaf3.manifest.v1",
        "entries": [
            {
                "target_id": "target_A",
                "pdb_id": "1ABC",
                "chain_id": "A",
                "sequence_sha256": "a" * 64,
                "feature_sha256": sha256_file(feature),
                "label_sha256": sha256_file(label_path),
                "length": 3,
                "msa_depth_bucket": "tiny",
                "length_bucket": "tiny",
                "split": "public_val_small",
                "feature_path": "feature.arrow",
                "label_path": "labels/public_val_labels.arrow",
            }
        ],
    }
    (manifests / "public_val_small.json").write_text(json.dumps(manifest), encoding="utf-8")
    (manifests / "train_tiny.json").write_text(json.dumps({**manifest, "entries": []}), encoding="utf-8")
    (locked / "scorer_version.txt").write_text("calpha_lddt_v1", encoding="utf-8")
    artifact_dir = tmp_path / "T901"
    write_prediction_artifact(
        trial_id="T901",
        split="public_val_small",
        output_dir=artifact_dir,
        predictions=[{"target_id": "target_A", "predicted_ca": [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]]}],
    )

    result = score_trial_artifacts(
        artifact_dir=artifact_dir,
        split="public_val_small",
        locked=load_locked_state(locked),
        write_outputs=False,
    )

    assert result["status"] == "SCORED"
    assert result["official_benchmark_result"] is True
    assert result["max_templates"] == 0
    assert result["label_hashes"]["public_val_small"] == sha256_file(label_path)


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
