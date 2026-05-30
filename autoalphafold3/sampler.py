"""Frozen-checkpoint NanoFold sampler worker helpers."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from autoalphafold3.checkpoint_training import DEFAULT_CHECKPOINT_MANIFEST, validate_checkpoint_manifest
from autoalphafold3.nanofold_adapter import NANOFOLD_PATH
from autoalphafold3.runner import (
    DONE_FILENAME,
    artifact_manifest_shape,
    validate_trial_id,
    write_prediction_artifact,
)
from autoalphafold3.schema import PRIMARY_METRIC, SCORER_VERSION

SAMPLER_MANIFEST_SCHEMA = "autoaf3.sampler_manifest.v1"


class SamplerError(RuntimeError):
    """Raised when a sampler trial cannot honestly produce predictions."""


def run_sampler_trial(
    trial_json: dict[str, Any],
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    split: str = "public_val_small",
) -> dict[str, object]:
    """Run a bounded inference-only NanoFold sample from a frozen checkpoint."""

    import torch

    root = Path(repo_root)
    trial_id = validate_trial_id(str(trial_json.get("trial_id", "")))
    if trial_json.get("trial_kind") != "sampler":
        raise SamplerError("run_sampler_trial only accepts sampler trials")
    if trial_json.get("max_steps") is not None:
        raise SamplerError("sampler trials must not set max_steps or train")

    output = Path(output_dir)
    _require_trial_output_dir(output, trial_id)
    if output.exists() and any(output.iterdir()):
        raise SamplerError(f"sampler output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _checkpoint_path(trial_json)
    checkpoint_manifest = _load_and_validate_checkpoint_manifest(checkpoint_path)
    _require_checkpoint_sha(checkpoint_path, str(checkpoint_manifest["checkpoint_sha256"]))
    _require_checkpoint_matches_trial(checkpoint_manifest, checkpoint_path)

    _ensure_nanofold_import_path(root)
    from nanofold.train.chain_dataset import ChainDataset
    from nanofold.train.trainer import Trainer

    started = time.time()
    torch.manual_seed(int(trial_json.get("seed", 0)))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config") or {})
    config["device"] = _sampler_device(config)
    config["max_templates"] = 0
    _require_checkpoint_config(config)

    feature_file = _sampler_feature_file(
        features_dir=features_dir,
        checkpoint_manifest=checkpoint_manifest,
    )
    train, _held_out = ChainDataset.construct_datasets(
        feature_file,
        config["train_split"],
        config["residue_crop_size"],
        config["num_msa_samples"],
    )
    trainer = Trainer(config, loggers=[], checkpoint_save_freq=0, checkpoint=checkpoint)
    batch = trainer.load_batch(next(iter(train)))
    predicted_ca = _sample_ca_coordinates(trainer.model, batch, sampler_steps=int(trial_json.get("sampler_steps", 1)))
    target_id = _first_feature_target_id(feature_file)
    prediction_payload = write_prediction_artifact(
        trial_id=trial_id,
        split=split,
        output_dir=output,
        source="frozen_checkpoint_nanofold_sampler",
        predictions=[{"target_id": target_id, "predicted_ca": predicted_ca}],
    )
    prediction_payload["candidate_id"] = str(trial_json.get("candidate_id", f"{trial_id}_sampler"))
    prediction_payload["max_templates"] = 0
    _atomic_write_json(output / "predictions.json", prediction_payload)

    artifact_manifest = artifact_manifest_shape(
        trial_id=trial_id,
        output_dir=output,
        features_dir=features_dir,
        split=split,
        status="REAL_MODE_UNAVAILABLE",
    )
    artifact_manifest.update(
        {
            "status": "SAMPLER_PREDICTED",
            "real_training_performed": False,
            "runner_mode": "frozen_checkpoint_sampler",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": str(checkpoint_manifest["checkpoint_sha256"]),
            "max_templates": 0,
            "sampler_steps": int(trial_json.get("sampler_steps", 1)),
            "primary_metric": PRIMARY_METRIC,
            "scorer_version": SCORER_VERSION,
            "disclaimer": (
                "Inference-only frozen-checkpoint sampler artifact. This file is "
                "not a benchmark result; scorer-only metrics determine trial status."
            ),
        }
    )
    _atomic_write_json(output / "artifact_manifest.json", artifact_manifest)
    sampler_manifest = {
        "schema_version": SAMPLER_MANIFEST_SCHEMA,
        "status": "SAMPLER_PREDICTED",
        "trial_id": trial_id,
        "candidate_id": prediction_payload["candidate_id"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint_manifest["checkpoint_sha256"]),
        "checkpoint_source_trial_id": str(checkpoint_manifest["trial_id"]),
        "feature_file": str(feature_file),
        "target_ids": [target_id],
        "prediction_count": 1,
        "real_training_performed": False,
        "inference_only": True,
        "max_templates": 0,
        "sampler_steps": int(trial_json.get("sampler_steps", 1)),
        "starts_search": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "runtime_s": time.time() - started,
    }
    _atomic_write_json(output / "sampler_manifest.json", sampler_manifest)
    _atomic_write_json(
        output / "training_log.json",
        {
            "schema_version": "autoaf3.training_log.v1",
            "trial_id": trial_id,
            "status": "SAMPLER_PREDICTED",
            "real_training_performed": False,
            "inference_only": True,
            "events": [
                {
                    "event": "frozen_checkpoint_sampler_predicted",
                    "timestamp_unix": int(started),
                    "message": "Loaded frozen NanoFold checkpoint and wrote sampler predictions.",
                }
            ],
        },
    )
    (output / "stdout.log").write_text("", encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")
    (output / "patch.diff").write_text("", encoding="utf-8")
    (output / DONE_FILENAME).write_text("frozen_checkpoint_sampler_completed\n", encoding="utf-8")
    return sampler_manifest


def _sample_ca_coordinates(model: Any, features: dict[str, Any], *, sampler_steps: int) -> list[list[float]]:
    import torch

    if sampler_steps < 1:
        raise SamplerError("sampler_steps must be at least 1")
    model.eval()
    with torch.no_grad():
        model.diffusion_model.inference = True
        _install_sampler_schedule(model.diffusion_model, sampler_steps=sampler_steps)
        input_embedding, single_rep_init, pair_rep_init = model.nanofold_input(features)
        single_rep_prev = torch.zeros_like(single_rep_init)
        pair_rep_prev = torch.zeros_like(pair_rep_init)
        num_recycle = int(model.num_recycle)
        for _ in range(num_recycle):
            single_rep, pair_rep = model.nanofold_trunk(
                features,
                input_embedding,
                pair_rep_init,
                single_rep_init,
                pair_rep_prev,
                single_rep_prev,
            )
            single_rep_prev, pair_rep_prev = single_rep, pair_rep
        sampled = model.diffusion_model.sample_diffusion(features, input_embedding, single_rep, pair_rep)
    if sampled.ndim != 2 or sampled.shape[-1] != 3:
        raise SamplerError(f"sampler returned unexpected coordinate shape: {tuple(sampled.shape)}")
    local_coords = features["local_coords"]
    residue_count = int(local_coords.shape[0])
    atoms_per_residue = int(local_coords.shape[1])
    if atoms_per_residue < 2:
        raise SamplerError("sampler features do not contain C-alpha atom positions")
    expected_atoms = residue_count * atoms_per_residue
    if int(sampled.shape[0]) != expected_atoms:
        raise SamplerError(f"sampler returned {sampled.shape[0]} atoms for expected {expected_atoms}")
    ca = sampled.reshape(residue_count, atoms_per_residue, 3)[:, 1, :]
    values = ca.detach().cpu().to(torch.float64).tolist()
    for row in values:
        if len(row) != 3 or not all(math.isfinite(float(coord)) for coord in row):
            raise SamplerError("sampler produced non-finite C-alpha coordinates")
    return [[float(coord) for coord in row] for row in values]


def _install_sampler_schedule(diffusion_model: Any, *, sampler_steps: int) -> None:
    import torch

    s_max = 160
    s_min = 0.0004
    p = 7
    schedule_points = max(2, sampler_steps + 1)
    steps = torch.arange(schedule_points, device=next(diffusion_model.parameters()).device) / (schedule_points - 1)
    schedule = diffusion_model.data_std_dev * (
        s_max ** (1 / p)
        + steps * (s_min ** (1 / p) - s_max ** (1 / p))
    ) ** p
    diffusion_model.schedule = torch.cat([schedule, torch.zeros_like(schedule[:1])])


def _sampler_device(config: dict[str, Any]) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard.
        raise SamplerError("torch is required for sampler execution") from exc
    configured = str(config.get("device", "cpu"))
    if configured == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return configured


def _checkpoint_path(trial_json: dict[str, Any]) -> Path:
    value = trial_json.get("checkpoint_path")
    if not isinstance(value, str) or not value:
        raise SamplerError("sampler trials require checkpoint_path")
    path = Path(value)
    if path.name != "checkpoint.pt" or "runs" not in path.parts or "trials" not in path.parts:
        raise SamplerError("checkpoint_path must point at a trial-scoped checkpoint.pt")
    if not path.exists():
        raise SamplerError(f"checkpoint_path does not exist: {path}")
    return path


def _load_and_validate_checkpoint_manifest(checkpoint_path: Path) -> dict[str, object]:
    manifest_path = checkpoint_path.parent / DEFAULT_CHECKPOINT_MANIFEST
    if not manifest_path.exists():
        raise SamplerError(f"checkpoint manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        return validate_checkpoint_manifest(payload)
    except Exception as exc:  # noqa: BLE001 - normalize checkpoint validation failures.
        raise SamplerError(f"checkpoint manifest is invalid: {exc}") from exc


def _require_checkpoint_sha(path: Path, expected: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise SamplerError(f"checkpoint sha256 mismatch: expected {expected}, got {actual}")


def _require_checkpoint_matches_trial(manifest: dict[str, object], checkpoint_path: Path) -> None:
    manifest_path = Path(str(manifest["checkpoint_path"]))
    if manifest_path != checkpoint_path:
        raise SamplerError(f"checkpoint_path does not match manifest: {checkpoint_path} != {manifest_path}")


def _require_checkpoint_config(config: dict[str, Any]) -> None:
    if config.get("max_templates") != 0:
        raise SamplerError("sampler checkpoint config must preserve max_templates=0")
    if not config:
        raise SamplerError("checkpoint is missing NanoFold config")


def _sampler_feature_file(*, features_dir: str | Path, checkpoint_manifest: dict[str, object]) -> Path:
    manifest_feature = Path(str(checkpoint_manifest["features_path"]))
    feature_file = manifest_feature if manifest_feature.is_absolute() else Path(features_dir) / manifest_feature
    if not feature_file.exists():
        feature_file = Path(features_dir) / manifest_feature.name
    if not feature_file.exists():
        raise SamplerError(f"sampler feature file is missing: {feature_file}")
    return feature_file


def _first_feature_target_id(feature_file: Path) -> str:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - depends on sampler image deps.
        raise SamplerError("pyarrow is required for sampler feature loading") from exc
    with pa.memory_map(str(feature_file)) as source:
        with ipc.open_file(source) as reader:
            table = reader.read_all()
    if table.num_rows < 1:
        raise SamplerError(f"sampler feature file has no rows: {feature_file}")
    row = table.slice(0, 1).to_pylist()[0]
    target_id = row.get("record_id") or (
        f"{row['pdb_id']}_{row['chain_id']}" if "pdb_id" in row and "chain_id" in row else None
    )
    if not isinstance(target_id, str) or not target_id:
        target_id = f"{feature_file.stem}_0"
    return target_id


def _require_trial_output_dir(path: Path, trial_id: str) -> None:
    as_posix = path.as_posix()
    if "runs/baseline" in as_posix:
        raise SamplerError("sampler must not write runs/baseline")
    if path.name != trial_id or "runs/trials" not in as_posix:
        raise SamplerError(f"sampler output must be trial-scoped under runs/trials/{trial_id}: {path}")


def _ensure_nanofold_import_path(repo_root: Path) -> None:
    nanofold_root = str(repo_root / NANOFOLD_PATH)
    if nanofold_root not in sys.path:
        sys.path.insert(0, nanofold_root)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
