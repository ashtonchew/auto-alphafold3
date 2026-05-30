"""Scorer-only trial artifact scoring wrapper.

This module represents the boundary that may read locked labels. Trial,
sampler, and debug workers must not import or call it during official search.
The local implementation supports toy JSON artifacts only; it does not produce
official benchmark metrics or read real Arrow label files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autoalphafold3.schema import PRIMARY_METRIC
from autoalphafold3.scorer.calpha_lddt import SCORER_VERSION, aggregate_calpha_lddt, score_calpha_lddt
from autoalphafold3.scorer.fold_cartographer import summarize_fold_cartographer
from autoalphafold3.scorer.locked_dataset import load_locked_manifest, manifest_hashes, sha256_file

PREDICTIONS_FILENAME = "predictions.json"
SUPPORTED_SCORING_SPLITS = {"public_val_small"}
LOCAL_ONLY_SPLITS = {"smoke"}


class LockedScorerError(RuntimeError):
    """Raised when the scorer-only wrapper refuses unsafe access."""


@dataclass(frozen=True)
class LockedScorerState:
    """Immutable scorer-only paths loaded once by a Modal Scorer container."""

    locked_root: str
    manifest_path: str = "manifests/public_val_small.json"
    train_manifest_path: str = "manifests/train_tiny.json"
    scorer_version_path: str = "scorer_version.txt"


def load_locked_state(locked_root: str | Path) -> LockedScorerState:
    """Return immutable scorer-only state for Modal memory snapshots."""

    return LockedScorerState(locked_root=str(locked_root))


def score_trial_artifacts(
    *,
    artifact_dir: str | Path,
    manifest_path: str | Path,
    labels_path: str | Path | None = None,
    scorer_version_path: str | Path | None = None,
    repo_root: str | Path = ".",
    split: str = "public_val_small",
    allow_local_smoke: bool = False,
    write_outputs: bool = True,
    locked: LockedScorerState | None = None,
) -> dict[str, object]:
    """Score one trial artifact directory through the scorer-only boundary."""

    _require_supported_split(split, allow_local_smoke=allow_local_smoke)
    if locked is not None:
        repo_root = locked.locked_root
        manifest_path = locked.manifest_path
        scorer_version_path = locked.scorer_version_path
    root = Path(repo_root)
    artifact_root = _safe_artifact_dir(artifact_dir)
    prediction_path = artifact_root / PREDICTIONS_FILENAME
    if not prediction_path.exists():
        failure = _failure_payload(
            trial_id=artifact_root.name,
            split=split,
            artifact_dir=artifact_root,
            failure_signature="prediction_artifact_missing",
            reason=f"missing {PREDICTIONS_FILENAME}",
        )
        if write_outputs:
            _write_score_outputs(artifact_root, failure)
        return failure

    verified = load_locked_manifest(manifest_path, repo_root=root, verify_assets=locked is None)
    _require_manifest_split(verified.manifest.entries, split=split)
    predictions = _load_predictions(prediction_path, split=split)
    labels = _load_labels(root, verified.manifest.entries)

    results = []
    failed_targets: list[str] = []
    for entry in verified.manifest.entries:
        predicted = predictions.get(entry.target_id)
        if predicted is None:
            failed_targets.append(entry.target_id)
            continue
        label = labels[entry.target_id]
        results.append(
            score_calpha_lddt(
                np.asarray(predicted, dtype=np.float64),
                np.asarray(label["target_ca"], dtype=np.float64),
                np.asarray(label["target_mask"], dtype=bool),
                target_id=entry.target_id,
            )
        )

    aggregate = aggregate_calpha_lddt(results)
    metrics = dict(aggregate["metrics"])
    metrics["num_failed_targets"] = int(metrics["num_failed_targets"]) + len(failed_targets)

    status = "SCORED" if not failed_targets else "FAIL"
    failure_signature = None if not failed_targets else "prediction_target_missing"
    official_result = locked is not None and split == "public_val_small"
    label_paths = sorted({entry.label_path for entry in verified.manifest.entries})
    payload = {
        "schema_version": "autoaf3.metrics.v1",
        "scorer_version": _read_scorer_version(root, scorer_version_path),
        "primary_metric": PRIMARY_METRIC,
        "status": status,
        "trial_id": artifact_root.name,
        "candidate_id": _prediction_candidate_id(prediction_path) or "local_artifact_score",
        "seed": 0,
        "split": split,
        "official_benchmark_result": official_result,
        "local_only": split in LOCAL_ONLY_SPLITS,
        "max_templates": 0,
        "manifests": _manifest_hash_payload(
            root=root,
            public_manifest_path=manifest_path,
            train_manifest_path=locked.train_manifest_path if locked is not None else None,
        ),
        "label_hashes": _label_hash_payload(root=root, label_paths=label_paths),
        "metrics": metrics,
        "quality_gates": {
            "nan_detected": any(result.nan_prediction_residue_count > 0 for result in results),
            "oom_detected": False,
            "timeout_detected": False,
            "runtime_s": 0.0,
            "peak_gpu_mem_gb": 0.0,
            "parameter_count": 0,
        },
        "fold_cartographer": summarize_fold_cartographer(results),
        "error_report": {
            "failure_signature": failure_signature,
            "failed_targets": failed_targets,
            "scorer_only": True,
            "template_policy": "max_templates=0",
            "max_templates": 0,
            "labels_path": str(labels_path) if labels_path is not None else "manifest_label_paths",
        },
        "artifacts": {
            "artifact_dir": str(artifact_root),
            "predictions_json": str(prediction_path),
            "manifest": str(manifest_path),
            "metrics_json": str(artifact_root / "metrics.json"),
            "error_report_json": str(artifact_root / "error_report.json"),
        },
    }
    if write_outputs:
        _write_score_outputs(artifact_root, payload)
    return payload


def _require_supported_split(split: str, *, allow_local_smoke: bool) -> None:
    if split in SUPPORTED_SCORING_SPLITS:
        return
    if allow_local_smoke and split in LOCAL_ONLY_SPLITS:
        return
    raise PermissionError(f"unsupported scorer-only split: {split}")


def _safe_artifact_dir(artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir)
    if ".." in path.parts:
        raise LockedScorerError(f"unsafe artifact_dir: {artifact_dir}")
    return path


def _require_manifest_split(entries: list[Any], *, split: str) -> None:
    mismatched = [entry.target_id for entry in entries if entry.split != split]
    if mismatched:
        raise LockedScorerError(
            f"manifest split mismatch for {len(mismatched)} targets: expected {split}"
        )


def _load_predictions(path: Path, *, split: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "autoaf3.predictions.v1":
        raise LockedScorerError("prediction artifact schema_version mismatch")
    payload_split = payload.get("split")
    if payload_split != split:
        raise LockedScorerError(f"prediction split mismatch: expected {split}, got {payload_split}")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise LockedScorerError("prediction artifact must contain a predictions list")
    by_target = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or "target_id" not in prediction or "predicted_ca" not in prediction:
            raise LockedScorerError("prediction entries require target_id and predicted_ca")
        target_id = str(prediction["target_id"])
        if target_id in by_target:
            raise LockedScorerError(f"duplicate prediction target_id: {target_id}")
        predicted_ca = _validate_predicted_ca(prediction["predicted_ca"], target_id=target_id)
        by_target[target_id] = predicted_ca
    return by_target


def _prediction_candidate_id(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    candidate_id = payload.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id:
        return candidate_id
    return None


def _validate_predicted_ca(value: object, *, target_id: str) -> list[list[float]]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LockedScorerError(f"predicted_ca is not numeric for {target_id}") from exc
    if array.ndim != 2 or array.shape[1] != 3:
        raise LockedScorerError(f"predicted_ca must have shape (L, 3) for {target_id}")
    if array.shape[0] == 0:
        raise LockedScorerError(f"predicted_ca must contain at least one residue for {target_id}")
    return array.tolist()


def _load_labels(root: Path, entries: list[Any]) -> dict[str, dict[str, object]]:
    labels: dict[str, dict[str, object]] = {}
    json_entries = []
    arrow_paths = sorted({entry.label_path for entry in entries if str(entry.label_path).endswith(".arrow")})
    for rel_path in arrow_paths:
        labels.update(_load_arrow_labels(root / rel_path))
    for entry in entries:
        if str(entry.label_path).endswith(".arrow"):
            continue
        json_entries.append(entry)
    for entry in json_entries:
        path = root / entry.label_path
        label = json.loads(path.read_text(encoding="utf-8"))
        labels[entry.target_id] = label
    return labels


def _load_arrow_labels(path: Path) -> dict[str, dict[str, object]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - depends on scorer image deps.
        raise LockedScorerError("pyarrow is required to read locked Arrow labels") from exc
    with pa.memory_map(str(path)) as source:
        with ipc.open_file(source) as reader:
            table = reader.read_all()
    labels = {}
    for row in table.to_pylist():
        target_id = str(row.get("record_id") or f"{row['pdb_id']}_{row['chain_id']}")
        labels[target_id] = {
            "target_ca": row["ca_positions"],
            "target_mask": row["ca_mask"],
        }
    return labels


def _manifest_hash_payload(
    *,
    root: Path,
    public_manifest_path: str | Path | None,
    train_manifest_path: str | Path | None,
) -> dict[str, str]:
    paths: dict[str, str | Path] = {}
    if train_manifest_path is not None and (root / train_manifest_path).exists():
        paths["train_tiny"] = train_manifest_path
    if public_manifest_path is not None:
        paths["public_val_small"] = public_manifest_path
    if not paths and public_manifest_path is not None:
        paths["scoring"] = public_manifest_path
    return manifest_hashes(paths, repo_root=root)


def _label_hash_payload(*, root: Path, label_paths: list[str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for rel_path in label_paths:
        if rel_path.endswith("public_val_labels.arrow"):
            payload["public_val_small"] = sha256_file(root / rel_path)
    return payload


def _read_scorer_version(root: Path, scorer_version_path: str | Path | None) -> str:
    if scorer_version_path is None:
        return SCORER_VERSION
    path = Path(scorer_version_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return SCORER_VERSION
    return path.read_text(encoding="utf-8").strip() or SCORER_VERSION


def _failure_payload(
    *,
    trial_id: str,
    split: str,
    artifact_dir: Path,
    failure_signature: str,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": "autoaf3.metrics.v1",
        "scorer_version": SCORER_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "status": "FAIL",
        "trial_id": trial_id,
        "candidate_id": "local_artifact_score",
        "seed": 0,
        "split": split,
        "official_benchmark_result": False,
        "metrics": {
            "best_val_calpha_lddt": 0.0,
            "mean_val_calpha_lddt": 0.0,
            "median_val_calpha_lddt": 0.0,
            "eligible_pair_count": 0,
            "num_targets": 0,
            "num_scored_targets": 0,
            "num_failed_targets": 1,
        },
        "fold_cartographer": {
            "signature": failure_signature,
            "summary": {"reason": reason},
            "buckets": {},
        },
        "error_report": {
            "failure_signature": failure_signature,
            "reason": reason,
            "scorer_only": True,
        },
        "artifacts": {
            "artifact_dir": str(artifact_dir),
            "predictions_json": str(artifact_dir / PREDICTIONS_FILENAME),
        },
    }


def _write_score_outputs(artifact_root: Path, payload: dict[str, object]) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics_payload = {key: value for key, value in payload.items() if key != "error_report"}
    error_payload = payload.get("error_report")
    _atomic_write_json(artifact_root / "metrics.json", metrics_payload)
    if isinstance(error_payload, dict):
        _atomic_write_json(artifact_root / "error_report.json", error_payload)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
