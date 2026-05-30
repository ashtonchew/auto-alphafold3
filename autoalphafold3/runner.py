"""Local runner artifact lifecycle for future NanoFold trials.

The functions here deliberately stop short of training. They create and
validate the artifact envelope a real Modal/NanoFold worker must later fill,
while refusing to claim that local scaffold code produced a checkpoint,
cached Arrow load, GPU job, or benchmark result.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

TRIAL_ID_RE = re.compile(r"^T[0-9]{3,}$")
ARTIFACT_MANIFEST_SCHEMA = "autoaf3.artifact_manifest.v1"
PREDICTIONS_SCHEMA = "autoaf3.predictions.v1"
DONE_FILENAME = "DONE"
ALLOWED_STUB_STATUSES = {"PLANNED", "STUB_ONLY", "REAL_MODE_UNAVAILABLE"}


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
            "predictions_json": str(output / "predictions.json"),
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
    return manifest


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
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
