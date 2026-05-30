"""Local runner artifact lifecycle for future NanoFold trials.

The functions here deliberately stop short of training. They create and
validate the artifact envelope a real Modal/NanoFold worker must later fill,
while refusing to claim that local scaffold code produced a checkpoint,
cached Arrow load, GPU job, or benchmark result.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

TRIAL_ID_RE = re.compile(r"^T[0-9]{3,}$")
ARTIFACT_MANIFEST_SCHEMA = "autoaf3.artifact_manifest.v1"
PREDICTIONS_SCHEMA = "autoaf3.predictions.v1"
DONE_FILENAME = "DONE"
PREDICTIONS_FILENAME = "predictions.json"
ALLOWED_STUB_STATUSES = {"PLANNED", "STUB_ONLY", "REAL_MODE_UNAVAILABLE", "BASELINE_PREDICTED"}


class RunnerError(RuntimeError):
    """Raised when the local runner stub refuses unsafe or unsupported work."""


def validate_trial_id(trial_id: str) -> str:
    """Validate a trial id for filesystem-safe artifact paths."""

    if not TRIAL_ID_RE.fullmatch(trial_id):
        raise RunnerError(f"invalid trial_id: {trial_id}")
    return trial_id


def safe_child_path(root: str | Path, child: str | Path) -> Path:
    """Resolve a child path and require it to stay under root."""

    root_path = Path(root).resolve()
    candidate = Path(child)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RunnerError(f"unsafe artifact path: {child}")
    resolved = (root_path / candidate).resolve()
    if root_path not in {resolved, *resolved.parents}:
        raise RunnerError(f"artifact path escapes root: {child}")
    return resolved


def artifact_manifest_shape(
    *,
    trial_id: str,
    output_dir: str | Path,
    features_dir: str | Path,
    split: str = "public_val_small",
    status: str = "STUB_ONLY",
) -> dict[str, object]:
    """Return the deterministic artifact manifest shape for one trial."""

    checked_trial_id = validate_trial_id(trial_id)
    output = Path(output_dir)
    _require_trial_output_dir(output, checked_trial_id)
    if status not in ALLOWED_STUB_STATUSES:
        raise RunnerError(f"unsupported local runner status: {status}")
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "trial_id": checked_trial_id,
        "status": status,
        "real_training_performed": False,
        "runner_mode": "local_stub",
        "split": split,
        "features_dir": str(features_dir),
        "artifacts": {
            "artifact_manifest_json": str(output / "artifact_manifest.json"),
            "predictions_json": str(output / PREDICTIONS_FILENAME),
            "training_log_json": str(output / "training_log.json"),
            "stdout_log": str(output / "stdout.log"),
            "stderr_log": str(output / "stderr.log"),
            "patch_diff": str(output / "patch.diff"),
            "checkpoint": str(output / "checkpoint.pt"),
            "done_marker": str(output / DONE_FILENAME),
        },
        "lifecycle": {
            "planned": True,
            "initialized": status == "STUB_ONLY",
            "real_training_available": False,
            "scored": False,
        },
        "disclaimer": (
            "Local stub only. This manifest does not represent a NanoFold run, "
            "Modal job, checkpoint, Arrow feature load, or benchmark result."
        ),
    }


def plan_trial_artifacts(
    trial_json: dict[str, Any],
    *,
    features_dir: str | Path,
    output_root: str | Path,
    split: str = "public_val_small",
) -> dict[str, object]:
    """Return the future artifact plan for a trial without creating files."""

    trial_id = validate_trial_id(str(trial_json.get("trial_id", "")))
    output_dir = Path(output_root) / trial_id
    return artifact_manifest_shape(
        trial_id=trial_id,
        output_dir=output_dir,
        features_dir=features_dir,
        split=split,
        status="PLANNED",
    )


def write_artifact_manifest_stub(
    trial_json: dict[str, Any],
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    split: str = "public_val_small",
) -> dict[str, object]:
    """Write a local stub artifact manifest without claiming training happened."""

    trial_id = validate_trial_id(str(trial_json.get("trial_id", "")))
    output = Path(output_dir)
    _require_trial_output_dir(output, trial_id)
    output.mkdir(parents=True, exist_ok=True)
    manifest = artifact_manifest_shape(
        trial_id=trial_id,
        output_dir=output,
        features_dir=features_dir,
        split=split,
    )
    manifest_path = output / "artifact_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest


def prediction_artifact_shape(
    *,
    trial_id: str,
    split: str,
    predictions: list[dict[str, object]],
    source: str = "provided_coordinates",
) -> dict[str, object]:
    """Return a scorer-compatible prediction artifact without scoring it."""

    checked_trial_id = validate_trial_id(trial_id)
    checked_split = _validate_split(split)
    checked_predictions = _validate_prediction_entries(predictions)
    return {
        "schema_version": PREDICTIONS_SCHEMA,
        "trial_id": checked_trial_id,
        "split": checked_split,
        "source": source,
        "official_benchmark_result": False,
        "predictions": checked_predictions,
        "disclaimer": (
            "Prediction artifact only. This file is not a benchmark result and "
            "does not imply that local stub code ran NanoFold training."
        ),
    }


def write_prediction_artifact(
    *,
    trial_id: str,
    split: str,
    predictions: list[dict[str, object]],
    output_dir: str | Path,
    source: str = "provided_coordinates",
) -> dict[str, object]:
    """Write a canonical local prediction artifact for scorer-compatible tests."""

    checked_trial_id = validate_trial_id(trial_id)
    output = Path(output_dir)
    _require_trial_output_dir(output, checked_trial_id)
    output.mkdir(parents=True, exist_ok=True)
    payload = prediction_artifact_shape(
        trial_id=checked_trial_id,
        split=split,
        predictions=predictions,
        source=source,
    )
    _atomic_write_json(output / PREDICTIONS_FILENAME, payload)
    return payload


def initialize_trial_directory(
    trial_json: dict[str, Any],
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    split: str = "public_val_small",
    patch_text: str = "",
    allow_existing_done: bool = False,
) -> dict[str, object]:
    """Create a deterministic local trial directory envelope.

    This writes only scaffold artifacts and a completion marker for the local
    stub. It never writes a checkpoint or predictions file, because those would
    imply model execution.
    """

    trial_id = validate_trial_id(str(trial_json.get("trial_id", "")))
    output = Path(output_dir)
    _require_trial_output_dir(output, trial_id)
    done_marker = output / DONE_FILENAME
    if done_marker.exists() and not allow_existing_done:
        raise RunnerError(f"trial directory already completed: {output}")

    output.mkdir(parents=True, exist_ok=True)
    manifest = write_artifact_manifest_stub(
        {**trial_json, "trial_id": trial_id},
        features_dir=features_dir,
        output_dir=output,
        split=split,
    )
    _atomic_write_json(
        output / "training_log.json",
        {
            "schema_version": "autoaf3.training_log.v1",
            "trial_id": trial_id,
            "status": "STUB_ONLY",
            "real_training_performed": False,
            "events": [
                {
                    "event": "initialized_local_stub",
                    "timestamp_unix": 0,
                    "message": "Artifact envelope initialized; no NanoFold training was run.",
                }
            ],
        },
    )
    (output / "stdout.log").write_text("", encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")
    (output / "patch.diff").write_text(patch_text, encoding="utf-8")
    done_marker.write_text(f"local_stub_completed_at={int(time.time())}\n", encoding="utf-8")
    return manifest


def validate_artifact_manifest(manifest: dict[str, object]) -> dict[str, object]:
    """Validate the local artifact manifest contract and return it unchanged."""

    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA:
        raise RunnerError("artifact manifest schema_version mismatch")
    validate_trial_id(str(manifest.get("trial_id", "")))
    if manifest.get("status") not in ALLOWED_STUB_STATUSES:
        raise RunnerError("artifact manifest status is not allowed for local stub")
    if manifest.get("real_training_performed") is not False:
        raise RunnerError("local runner stub must not claim real training")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RunnerError("artifact manifest missing artifacts map")
    required = {
        "artifact_manifest_json",
        "predictions_json",
        "training_log_json",
        "stdout_log",
        "stderr_log",
        "patch_diff",
        "checkpoint",
        "done_marker",
    }
    missing = sorted(required - artifacts.keys())
    if missing:
        raise RunnerError(f"artifact manifest missing keys: {', '.join(missing)}")
    _require_manifest_artifacts_trial_scoped(artifacts)
    return manifest


def validate_prediction_artifact(payload: dict[str, object]) -> dict[str, object]:
    """Validate the local prediction artifact contract and return it unchanged."""

    if payload.get("schema_version") != PREDICTIONS_SCHEMA:
        raise RunnerError("prediction artifact schema_version mismatch")
    validate_trial_id(str(payload.get("trial_id", "")))
    if payload.get("official_benchmark_result") is not False:
        raise RunnerError("prediction artifacts must not claim official benchmark status")
    _validate_split(payload.get("split"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise RunnerError("prediction artifact must contain a predictions list")
    _validate_prediction_entries(predictions)
    return payload


def run_fixed_budget_trial(
    trial_json: dict[str, Any],
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    split: str = "public_val_small",
    allow_local_stub: bool = False,
) -> dict[str, object]:
    """Future Modal/NanoFold trial interface.

    The local implementation refuses to run unless `allow_local_stub` is
    explicitly set. That makes tests and scaffolding possible without implying a
    real NanoFold training job, GPU run, Arrow load, or benchmark metric exists.
    """

    if trial_json.get("baseline") is True:
        return run_sequence_linear_baseline(
            trial_json,
            features_dir=features_dir,
            output_dir=output_dir,
            split=split,
        )
    if not allow_local_stub:
        raise RunnerError(
            "run_fixed_budget_trial is not implemented without NanoFold features, "
            "dependencies, and Modal/GPU execution; pass allow_local_stub=True only "
            "for local artifact-manifest tests"
        )
    return initialize_trial_directory(
        trial_json,
        features_dir=features_dir,
        output_dir=output_dir,
        split=split,
    )


def run_sequence_linear_baseline(
    trial_json: dict[str, Any],
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    split: str = "public_val_small",
) -> dict[str, object]:
    """Write a deterministic no-template, sequence-only baseline artifact.

    This is a real baseline artifact generator, not a training claim. It reads
    only public feature rows and emits simple C-alpha coordinates from sequence
    length, leaving benchmark scoring to the scorer-only worker.
    """

    trial_id = validate_trial_id(str(trial_json.get("trial_id", "")))
    if trial_json.get("max_templates") != 0:
        raise RunnerError("official baseline must preserve max_templates=0")
    output = Path(output_dir)
    _require_trial_output_dir(output, trial_id)
    output.mkdir(parents=True, exist_ok=True)
    predictions = _sequence_linear_predictions(Path(features_dir) / f"{split}.arrow")
    prediction_payload = prediction_artifact_shape(
        trial_id=trial_id,
        split=split,
        predictions=predictions,
        source="sequence_linear_no_template_baseline",
    )
    prediction_payload["candidate_id"] = str(trial_json.get("candidate_id", "sequence_linear_baseline"))
    prediction_payload["max_templates"] = 0
    _atomic_write_json(output / PREDICTIONS_FILENAME, prediction_payload)
    manifest = artifact_manifest_shape(
        trial_id=trial_id,
        output_dir=output,
        features_dir=features_dir,
        split=split,
        status="STUB_ONLY",
    )
    manifest.update(
        {
            "status": "BASELINE_PREDICTED",
            "real_training_performed": False,
            "runner_mode": "sequence_linear_baseline",
            "max_templates": 0,
            "disclaimer": (
                "Sequence-only no-template baseline predictions. This is not a "
                "NanoFold training run; scorer-only metrics determine the real baseline score."
            ),
        }
    )
    _atomic_write_json(output / "artifact_manifest.json", manifest)
    _atomic_write_json(
        output / "training_log.json",
        {
            "schema_version": "autoaf3.training_log.v1",
            "trial_id": trial_id,
            "status": "BASELINE_PREDICTED",
            "real_training_performed": False,
            "max_templates": 0,
            "events": [
                {
                    "event": "sequence_linear_baseline_predicted",
                    "timestamp_unix": 0,
                    "message": "Generated sequence-only no-template baseline predictions from public features.",
                }
            ],
        },
    )
    (output / "stdout.log").write_text("", encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")
    (output / "patch.diff").write_text("", encoding="utf-8")
    (output / DONE_FILENAME).write_text("sequence_linear_baseline_completed\n", encoding="utf-8")
    return manifest


def run_final_validation(
    trial_json: dict[str, Any],
    *,
    seed: int,
    features_dir: str | Path,
    output_dir: str | Path,
    allow_local_stub: bool = False,
) -> dict[str, object]:
    """Future final-validation interface with the same no-fake-run guard."""

    if seed < 0:
        raise RunnerError(f"invalid seed: {seed}")
    payload = dict(trial_json)
    payload["seed"] = seed
    return run_fixed_budget_trial(
        payload,
        features_dir=features_dir,
        output_dir=output_dir,
        allow_local_stub=allow_local_stub,
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _sequence_linear_predictions(feature_path: Path) -> list[dict[str, object]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - depends on Modal/local image deps.
        raise RunnerError("pyarrow is required for sequence baseline feature loading") from exc
    if not feature_path.exists():
        raise RunnerError(f"baseline feature file is missing: {feature_path}")
    with pa.memory_map(str(feature_path)) as source:
        with ipc.open_file(source) as reader:
            table = reader.read_all()
    predictions: list[dict[str, object]] = []
    for row in table.to_pylist():
        target_id = str(row.get("record_id") or f"{row['pdb_id']}_{row['chain_id']}")
        sequence_length = int(row["sequence_length"])
        predicted_ca = [[float(index) * 3.8, 0.0, 0.0] for index in range(sequence_length)]
        predictions.append({"target_id": target_id, "predicted_ca": predicted_ca})
    return predictions


def _require_trial_output_dir(output: Path, trial_id: str) -> None:
    if output.name != trial_id:
        raise RunnerError(f"trial artifacts must be written under a trial-scoped directory named {trial_id}")


def _validate_prediction_entries(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    if not predictions:
        raise RunnerError("prediction artifact must contain at least one prediction")
    seen: set[str] = set()
    checked = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise RunnerError("prediction entries must be objects")
        target_id = prediction.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise RunnerError("prediction entries require target_id")
        if target_id in seen:
            raise RunnerError(f"duplicate prediction target_id: {target_id}")
        seen.add(target_id)
        predicted_ca = _validate_predicted_ca(prediction.get("predicted_ca"), target_id=target_id)
        checked.append({"target_id": target_id, "predicted_ca": predicted_ca})
    return checked


def _validate_predicted_ca(value: object, *, target_id: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise RunnerError(f"predicted_ca must be a non-empty list for {target_id}")
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise RunnerError(f"predicted_ca must have shape (L, 3) for {target_id}")
        try:
            coords = [float(row[0]), float(row[1]), float(row[2])]
        except (TypeError, ValueError) as exc:
            raise RunnerError(f"predicted_ca is not numeric for {target_id}") from exc
        if not all(math.isfinite(coord) for coord in coords):
            raise RunnerError(f"predicted_ca contains non-finite coordinate for {target_id}")
        rows.append(coords)
    return rows


def _validate_split(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RunnerError("prediction artifact requires split")
    return value


def _require_manifest_artifacts_trial_scoped(artifacts: dict[str, object]) -> None:
    manifest_path = Path(str(artifacts["artifact_manifest_json"]))
    trial_root = manifest_path.parent
    validate_trial_id(trial_root.name)
    resolved_root = trial_root.resolve()
    for key, value in artifacts.items():
        if not isinstance(value, str):
            raise RunnerError(f"artifact path for {key} must be a string")
        artifact_path = Path(value)
        if ".." in artifact_path.parts:
            raise RunnerError(f"artifact path for {key} escapes trial directory")
        resolved = artifact_path.resolve()
        if resolved_root not in {resolved, *resolved.parents}:
            raise RunnerError(f"artifact path for {key} escapes trial directory")
