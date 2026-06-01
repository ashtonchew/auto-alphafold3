"""Bounded NanoFold short-training helpers for autoresearch trials."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from autoalphafold3.config_contract import validate_config_file
from autoalphafold3.nanofold_adapter import NANOFOLD_PATH, load_nanofold_config
from autoalphafold3.runner import ARTIFACT_MANIFEST_SCHEMA, DONE_FILENAME, validate_trial_id
from autoalphafold3.schema import PRIMARY_METRIC, SCORER_VERSION

SHORT_TRAINING_MANIFEST_SCHEMA = "autoaf3.short_training_manifest.v1"
LOSS_HISTORY_SCHEMA = "autoaf3.loss_history.v1"
TRAINING_LOG_SCHEMA = "autoaf3.training_log.v1"
DEFAULT_SHORT_TRAINING_MANIFEST = "short_training_manifest.json"
DEFAULT_LOSS_HISTORY = "loss_history.json"
DEFAULT_CHECKPOINT = "checkpoint.pt"
MAX_STEPS_BY_BUDGET = {
    "smoke": 10,
    "trial": 250,
    "debug": 250,
    "dry_run": 0,
}


class ShortTrainingError(RuntimeError):
    """Raised when bounded short training would violate the contract."""


def short_training_payload(
    *,
    trial_id: str,
    candidate_id: str,
    config_path: str,
    features_path: str,
    max_steps: int,
    budget: str,
    seed: int,
    artifact_dir: str | None = None,
    local_only: bool = False,
) -> dict[str, object]:
    """Return a JSON-friendly bounded short-training payload."""

    checked_trial_id = validate_trial_id(trial_id)
    if max_steps < 1:
        raise ShortTrainingError("short training max_steps must be positive")
    if seed < 0:
        raise ShortTrainingError("short training seed must be non-negative")
    return {
        "trial_id": checked_trial_id,
        "candidate_id": candidate_id,
        "runner_mode": "short_training",
        "trial_kind": "training",
        "budget": budget,
        "config_path": config_path,
        "features_path": features_path,
        "max_steps": max_steps,
        "seed": seed,
        "artifact_dir": artifact_dir,
        "max_templates": 0,
        "local_only": local_only,
    }


def run_short_nanofold_training(
    payload: dict[str, object],
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    local_only: bool = False,
) -> dict[str, object]:
    """Run bounded NanoFold training and write honest trial-scoped artifacts."""

    import torch

    root = Path(repo_root)
    trial_id = validate_trial_id(str(payload.get("trial_id", "")))
    candidate_id = str(payload.get("candidate_id", trial_id))
    max_steps = _require_positive_int(payload.get("max_steps"), name="max_steps")
    seed = _require_nonnegative_int(payload.get("seed", 0), name="seed")
    if payload.get("max_templates") != 0:
        raise ShortTrainingError("short training must preserve max_templates=0")
    budget = str(payload.get("budget", ""))
    _require_budget_cap(max_steps=max_steps, budget=budget)

    output = Path(output_dir)
    _require_trial_output_dir(output, trial_id)
    output.mkdir(parents=True, exist_ok=True)

    config_path = str(payload.get("config_path", ""))
    _reject_unsafe_relative_path(Path(config_path), label="config_path")
    config_report = validate_config_file(config_path, repo_root=root)
    if not config_report.valid:
        raise ShortTrainingError(f"short training config is invalid: {config_report.missing_keys}")
    config = dict(load_nanofold_config(config_path, repo_root=root))
    if config.get("max_templates") != 0:
        raise ShortTrainingError("short training config must pin max_templates=0")
    config["max_templates"] = 0

    relative_features_path = Path(str(payload.get("features_path", "")))
    _reject_unsafe_relative_path(relative_features_path, label="features_path")
    feature_file = Path(features_dir) / relative_features_path
    if not feature_file.exists():
        raise ShortTrainingError(f"short training feature file is missing: {feature_file}")

    _ensure_nanofold_import_path(root)
    from nanofold.train.chain_dataset import ChainDataset
    from nanofold.train.trainer import Trainer

    torch.manual_seed(seed)
    train, _held_out = ChainDataset.construct_datasets(
        feature_file,
        config["train_split"],
        config["residue_crop_size"],
        config["num_msa_samples"],
    )
    trainer = Trainer(config, loggers=[], checkpoint_save_freq=max_steps, checkpoint=None)
    train_iter = iter(train)
    loss_history: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    started = time.time()
    for step in range(1, max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train)
            try:
                batch = next(train_iter)
            except StopIteration as exc:
                raise ShortTrainingError("short training requires at least one training row") from exc
        losses = trainer.training_loop(trainer.load_batch(batch))
        _require_finite_losses(losses)
        loss_history.append(
            {
                "step": step,
                "losses": {key: float(value) for key, value in losses.items()},
            }
        )
        events.append({"event": "training_step", "step": step, "status": "complete"})
    trainer.epoch = max_steps
    elapsed_s = time.time() - started

    checkpoint_path = output / DEFAULT_CHECKPOINT
    checkpoint = {
        "schema_version": "autoaf3.nanofold_short_training_checkpoint.v1",
        "epoch": trainer.epoch,
        "training_steps": max_steps,
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    manifest = {
        "schema_version": SHORT_TRAINING_MANIFEST_SCHEMA,
        "status": "SHORT_TRAINING_READY",
        "trial_id": trial_id,
        "candidate_id": candidate_id,
        "budget": str(payload.get("budget", "")),
        "real_training_performed": True,
        "local_only": bool(local_only or payload.get("local_only") is True),
        "official_benchmark_result": False,
        "training_steps": max_steps,
        "max_steps": max_steps,
        "max_templates": 0,
        "seed": seed,
        "config_path": config_path,
        "features_path": str(feature_file),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_source": "short_nanofold_training",
        "loss_history_path": str(output / DEFAULT_LOSS_HISTORY),
        "training_log_path": str(output / "training_log.json"),
        "artifact_manifest_path": str(output / "artifact_manifest.json"),
        "scorer_version": SCORER_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "final_losses": loss_history[-1]["losses"],
        "runtime_s": elapsed_s,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "reads_locked_labels": False,
    }
    _atomic_write_json(output / DEFAULT_SHORT_TRAINING_MANIFEST, manifest)
    _atomic_write_json(
        output / DEFAULT_LOSS_HISTORY,
        {
            "schema_version": LOSS_HISTORY_SCHEMA,
            "trial_id": trial_id,
            "training_steps": max_steps,
            "losses": loss_history,
        },
    )
    _atomic_write_json(
        output / "training_log.json",
        {
            "schema_version": TRAINING_LOG_SCHEMA,
            "trial_id": trial_id,
            "status": "SHORT_TRAINING_READY",
            "real_training_performed": True,
            "local_only": bool(local_only or payload.get("local_only") is True),
            "official_benchmark_result": False,
            "max_templates": 0,
            "events": events,
            "writes_baseline": False,
            "writes_ledger": False,
            "writes_discovery_ledger": False,
        },
    )
    _atomic_write_json(output / "artifact_manifest.json", _artifact_manifest(payload, output, manifest))
    (output / "stdout.log").write_text("", encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")
    (output / "patch.diff").write_text(str(payload.get("patch_diff", "")), encoding="utf-8")
    (output / DONE_FILENAME).write_text("short_training_completed\n", encoding="utf-8")
    return manifest


def validate_short_training_manifest(payload: object) -> dict[str, object]:
    """Validate a short-training manifest returned by a worker."""

    if not isinstance(payload, dict):
        raise ShortTrainingError("short training manifest must be a JSON object")
    required = {
        "schema_version": SHORT_TRAINING_MANIFEST_SCHEMA,
        "status": "SHORT_TRAINING_READY",
        "real_training_performed": True,
        "official_benchmark_result": False,
        "checkpoint_source": "short_nanofold_training",
        "max_templates": 0,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "reads_locked_labels": False,
        "scorer_version": SCORER_VERSION,
        "primary_metric": PRIMARY_METRIC,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ShortTrainingError(f"short training manifest has invalid {key}: {payload.get(key)!r}")
    validate_trial_id(str(payload.get("trial_id", "")))
    steps = _require_positive_int(payload.get("training_steps"), name="training_steps")
    if payload.get("max_steps") != steps:
        raise ShortTrainingError("short training manifest max_steps must match training_steps")
    checkpoint_path = payload.get("checkpoint_path")
    if (
        not isinstance(checkpoint_path, str)
        or "runs/trials" not in checkpoint_path
        or not checkpoint_path.endswith("/checkpoint.pt")
    ):
        raise ShortTrainingError("short training checkpoint_path must be trial-scoped and end in checkpoint.pt")
    sha = payload.get("checkpoint_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ShortTrainingError("short training manifest missing checkpoint_sha256")
    final_losses = payload.get("final_losses")
    if not isinstance(final_losses, dict) or "total_loss" not in final_losses:
        raise ShortTrainingError("short training manifest missing final_losses.total_loss")
    _require_finite_losses(final_losses)
    return payload


def _artifact_manifest(
    payload: dict[str, object],
    output: Path,
    short_manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "trial_id": short_manifest["trial_id"],
        "status": "SHORT_TRAINING_READY",
        "real_training_performed": True,
        "runner_mode": "short_training",
        "local_only": short_manifest["local_only"],
        "official_benchmark_result": False,
        "split": "train",
        "features_dir": str(Path(str(short_manifest["features_path"])).parent),
        "artifacts": {
            "artifact_manifest_json": str(output / "artifact_manifest.json"),
            "short_training_manifest_json": str(output / DEFAULT_SHORT_TRAINING_MANIFEST),
            "loss_history_json": str(output / DEFAULT_LOSS_HISTORY),
            "training_log_json": str(output / "training_log.json"),
            "stdout_log": str(output / "stdout.log"),
            "stderr_log": str(output / "stderr.log"),
            "patch_diff": str(output / "patch.diff"),
            "checkpoint": str(output / DEFAULT_CHECKPOINT),
            "done_marker": str(output / DONE_FILENAME),
        },
        "lifecycle": {
            "planned": True,
            "initialized": True,
            "real_training_available": True,
            "scored": False,
        },
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "disclaimer": (
            "Bounded short-training artifact. This is not a scored benchmark "
            "result and does not grant Discovery Ledger authority."
        ),
    }


def _ensure_nanofold_import_path(repo_root: Path) -> None:
    nanofold_root = str(repo_root / NANOFOLD_PATH)
    if nanofold_root not in sys.path:
        sys.path.insert(0, nanofold_root)


def _require_trial_output_dir(path: Path, trial_id: str) -> None:
    expected_suffix = Path("runs") / "trials" / trial_id
    as_posix = path.as_posix()
    if "runs/baseline" in as_posix:
        raise ShortTrainingError("short training must not write runs/baseline")
    if path.name != trial_id or "runs/trials" not in as_posix:
        raise ShortTrainingError(f"short training output must be trial-scoped under runs/trials/{trial_id}: {path}")
    if path.exists() and any(path.iterdir()):
        raise ShortTrainingError(f"short training output already exists and is not empty: {path}")
    if not as_posix.endswith(expected_suffix.as_posix()) and f"runs/trials/{trial_id}" not in as_posix:
        raise ShortTrainingError(f"short training output must end in runs/trials/{trial_id}: {path}")


def _reject_unsafe_relative_path(path: Path, *, label: str) -> None:
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ShortTrainingError(f"short training {label} must be a safe relative path: {path}")
    as_posix = path.as_posix()
    forbidden_fragments = ("autoalphafold3-locked", "locked", "labels", "public_val_labels", "runs/baseline")
    if any(fragment in as_posix for fragment in forbidden_fragments):
        raise ShortTrainingError(f"short training {label} references a forbidden path: {path}")


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ShortTrainingError(f"short training {name} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ShortTrainingError(f"short training {name} must be a non-negative integer")
    return value


def _require_budget_cap(*, max_steps: int, budget: str) -> None:
    cap = MAX_STEPS_BY_BUDGET.get(budget)
    if cap is None:
        raise ShortTrainingError(f"short training budget is unsupported: {budget}")
    if cap == 0 or max_steps > cap:
        raise ShortTrainingError(f"short training max_steps={max_steps} exceeds {budget} budget cap {cap}")


def _require_finite_losses(losses: dict[str, object]) -> None:
    for key, value in losses.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ShortTrainingError(f"short training loss {key} is not finite")


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
