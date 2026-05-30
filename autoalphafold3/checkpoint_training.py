"""Minimal real NanoFold checkpoint production helpers."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from autoalphafold3.nanofold_adapter import NANOFOLD_PATH, load_nanofold_config
from autoalphafold3.runner import DONE_FILENAME, validate_trial_id
from autoalphafold3.schema import PRIMARY_METRIC, SCORER_VERSION

CHECKPOINT_MANIFEST_SCHEMA = "autoaf3.checkpoint_manifest.v1"
DEFAULT_CHECKPOINT_FILENAME = "checkpoint.pt"
DEFAULT_CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
DEFAULT_TRAIN_FEATURES = "nanofold_event_small_no_templates.arrow"


class CheckpointTrainingError(RuntimeError):
    """Raised when the checkpoint producer cannot create honest evidence."""


def one_batch_checkpoint_payload(
    *,
    trial_id: str = "T010",
    candidate_id: str = "one_batch_nanofold_checkpoint",
    config_path: str = "configs/nanofold_dev_cpu_smoke.json",
    features_path: str = DEFAULT_TRAIN_FEATURES,
    seed: int = 0,
) -> dict[str, object]:
    """Return the bounded one-batch checkpoint payload for Modal."""

    checked_trial_id = validate_trial_id(trial_id)
    return {
        "trial_id": checked_trial_id,
        "candidate_id": candidate_id,
        "trial_kind": "training",
        "budget_tier": "trial",
        "config_path": config_path,
        "features_path": features_path,
        "seed": seed,
        "max_templates": 0,
        "diffusion_steps": 1,
        "training_steps": 1,
        "description": "One-batch NanoFold-style AlphaFold3-lite checkpoint seed.",
    }


def run_one_batch_nanofold_checkpoint(
    payload: dict[str, object],
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, object]:
    """Run exactly one NanoFold training batch and write a checkpoint manifest."""

    import torch

    root = Path(repo_root)
    trial_id = validate_trial_id(str(payload.get("trial_id", "")))
    if payload.get("max_templates") != 0:
        raise CheckpointTrainingError("checkpoint training must preserve max_templates=0")
    if payload.get("training_steps") != 1:
        raise CheckpointTrainingError("checkpoint training is bounded to exactly one training step")
    if payload.get("diffusion_steps") != 1:
        raise CheckpointTrainingError("checkpoint training is bounded to exactly one diffusion step")

    output = Path(output_dir)
    _require_trial_output_dir(output, trial_id)
    output.mkdir(parents=True, exist_ok=True)

    _ensure_nanofold_import_path(root)
    from nanofold.train.chain_dataset import ChainDataset
    from nanofold.train.trainer import Trainer

    torch.manual_seed(int(payload.get("seed", 0)))
    config_path = str(payload.get("config_path", "configs/nanofold_dev_cpu_smoke.json"))
    config = dict(load_nanofold_config(config_path, repo_root=root))
    config["diffusion_steps"] = 1
    config["max_templates"] = 0
    relative_features_path = Path(str(payload.get("features_path", DEFAULT_TRAIN_FEATURES)))
    _reject_unsafe_relative_feature_path(relative_features_path)
    feature_file = Path(features_dir) / relative_features_path
    train, _held_out = ChainDataset.construct_datasets(
        feature_file,
        config["train_split"],
        config["residue_crop_size"],
        config["num_msa_samples"],
    )
    trainer = Trainer(config, loggers=[], checkpoint_save_freq=1, checkpoint=None)
    started = time.time()
    losses = trainer.training_loop(trainer.load_batch(next(iter(train))))
    elapsed_s = time.time() - started
    _require_finite_losses(losses)
    trainer.epoch = 1

    checkpoint_path = output / DEFAULT_CHECKPOINT_FILENAME
    checkpoint = {
        "schema_version": "autoaf3.nanofold_checkpoint.v1",
        "epoch": trainer.epoch,
        "training_steps": 1,
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    sha256 = _sha256_file(checkpoint_path)
    manifest = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "status": "CHECKPOINT_READY",
        "trial_id": trial_id,
        "candidate_id": str(payload.get("candidate_id", "one_batch_nanofold_checkpoint")),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_source": "one_batch_nanofold_training",
        "real_training_performed": True,
        "training_steps": 1,
        "diffusion_steps": 1,
        "max_templates": 0,
        "seed": int(payload.get("seed", 0)),
        "config_path": config_path,
        "features_path": str(feature_file),
        "scorer_version": SCORER_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "losses": {key: float(value) for key, value in losses.items()},
        "runtime_s": elapsed_s,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
    }
    _atomic_write_json(output / DEFAULT_CHECKPOINT_MANIFEST, manifest)
    (output / DONE_FILENAME).write_text("one_batch_checkpoint_completed\n", encoding="utf-8")
    return manifest


def validate_checkpoint_manifest(payload: object) -> dict[str, object]:
    """Validate a one-batch checkpoint manifest returned by Modal."""

    if not isinstance(payload, dict):
        raise CheckpointTrainingError("checkpoint manifest must be a JSON object")
    required = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "status": "CHECKPOINT_READY",
        "checkpoint_source": "one_batch_nanofold_training",
        "real_training_performed": True,
        "training_steps": 1,
        "diffusion_steps": 1,
        "max_templates": 0,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise CheckpointTrainingError(f"checkpoint manifest has invalid {key}: {payload.get(key)!r}")
    validate_trial_id(str(payload.get("trial_id", "")))
    checkpoint_path = payload.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or "runs/trials" not in checkpoint_path or not checkpoint_path.endswith("/checkpoint.pt"):
        raise CheckpointTrainingError("checkpoint manifest checkpoint_path must be trial-scoped and end in checkpoint.pt")
    sha = payload.get("checkpoint_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise CheckpointTrainingError("checkpoint manifest missing checkpoint_sha256")
    losses = payload.get("losses")
    if not isinstance(losses, dict) or "total_loss" not in losses:
        raise CheckpointTrainingError("checkpoint manifest missing losses.total_loss")
    _require_finite_losses(losses)
    return payload


def _ensure_nanofold_import_path(repo_root: Path) -> None:
    nanofold_root = str(repo_root / NANOFOLD_PATH)
    if nanofold_root not in sys.path:
        sys.path.insert(0, nanofold_root)


def _require_trial_output_dir(path: Path, trial_id: str) -> None:
    expected_suffix = Path("runs") / "trials" / trial_id
    as_posix = path.as_posix()
    if "runs/baseline" in as_posix:
        raise CheckpointTrainingError("checkpoint training must not write runs/baseline")
    if path.name != trial_id or "runs/trials" not in as_posix:
        raise CheckpointTrainingError(f"checkpoint output must be trial-scoped under runs/trials/{trial_id}: {path}")
    if path.exists() and any(path.iterdir()):
        raise CheckpointTrainingError(f"checkpoint output already exists and is not empty: {path}")
    if not as_posix.endswith(expected_suffix.as_posix()) and f"runs/trials/{trial_id}" not in as_posix:
        raise CheckpointTrainingError(f"checkpoint output must end in runs/trials/{trial_id}: {path}")


def _reject_unsafe_relative_feature_path(path: Path) -> None:
    if path.is_absolute() or ".." in path.parts:
        raise CheckpointTrainingError(f"checkpoint features_path must stay under features_dir: {path}")


def _require_finite_losses(losses: dict[str, object]) -> None:
    for key, value in losses.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise CheckpointTrainingError(f"checkpoint training loss {key} is not finite")


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
