"""Human-approved one-batch checkpoint runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from autoalphafold3.checkpoint_training import (
    DEFAULT_CHECKPOINT_MANIFEST,
    CheckpointTrainingError,
    one_batch_checkpoint_payload,
    validate_checkpoint_manifest,
)
from autoalphafold3.modal_app import APP_NAME, validate_execution_payload
from autoalphafold3.runner import validate_trial_id

APPROVAL_TEXT = "I_APPROVE_ONE_BATCH_CHECKPOINT"
DEFAULT_CHECKPOINT_TRIAL_ID = "T010"
DEFAULT_CHECKPOINT_CANDIDATE_ID = "one_batch_nanofold_checkpoint"


class CheckpointRunError(RuntimeError):
    """Raised when the approved checkpoint run cannot complete honestly."""


class ModalCheckpointClient(Protocol):
    """Small protocol for deployed Modal checkpoint execution."""

    def run_checkpoint(self, payload: dict[str, object]) -> dict[str, object]:
        """Run the one-batch checkpoint worker and return its manifest."""


@dataclass(frozen=True)
class CheckpointRunResult:
    """JSON-friendly one-batch checkpoint runner result."""

    status: str
    mode: str
    trial_id: str
    source_dir: str
    wrote_files: list[str]
    plan: dict[str, object]
    checkpoint_manifest: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "trial_id": self.trial_id,
            "source_dir": self.source_dir,
            "wrote_files": self.wrote_files,
            "plan": self.plan,
            "checkpoint_manifest": self.checkpoint_manifest,
        }


def run_one_batch_checkpoint(
    *,
    repo_root: str | Path = ".",
    trial_id: str = DEFAULT_CHECKPOINT_TRIAL_ID,
    source_dir: str | Path | None = None,
    config_path: str = "configs/nanofold_dev_cpu_smoke.json",
    features_path: str = "train_tiny.arrow",
    approval: str | None = None,
    mode: str = "dry-run",
    modal_env: str | None = None,
    modal_client: ModalCheckpointClient | None = None,
) -> CheckpointRunResult:
    """Plan or execute the human-approved one-batch checkpoint run."""

    checked_trial_id = validate_trial_id(trial_id)
    root = Path(repo_root)
    source = root / (source_dir or Path("runs/trials") / checked_trial_id)
    _require_trial_source_dir(source, checked_trial_id, dry_run=mode == "dry-run")
    plan = checkpoint_run_plan(
        trial_id=checked_trial_id,
        source_dir=source,
        config_path=config_path,
        features_path=features_path,
    )
    if mode == "dry-run":
        return CheckpointRunResult(
            status="PLANNED",
            mode=mode,
            trial_id=checked_trial_id,
            source_dir=str(source),
            wrote_files=[],
            plan=plan,
        )
    if mode != "modal":
        raise CheckpointRunError(f"unsupported checkpoint run mode: {mode}")
    if approval != APPROVAL_TEXT:
        raise CheckpointRunError(f"one-batch checkpoint run requires --approve {APPROVAL_TEXT}")

    client = modal_client if modal_client is not None else DeployedModalCheckpointClient(environment_name=modal_env)
    payload = one_batch_checkpoint_payload(
        trial_id=checked_trial_id,
        candidate_id=DEFAULT_CHECKPOINT_CANDIDATE_ID,
        config_path=config_path,
        features_path=features_path,
    )
    validate_execution_payload(payload, role="trial")
    manifest = _require_checkpoint_manifest(client.run_checkpoint(payload))

    source.mkdir(parents=True, exist_ok=True)
    manifest_path = source / DEFAULT_CHECKPOINT_MANIFEST
    _atomic_write_json(manifest_path, manifest)
    return CheckpointRunResult(
        status="PASS",
        mode=mode,
        trial_id=checked_trial_id,
        source_dir=str(source),
        wrote_files=[str(manifest_path)],
        plan=plan,
        checkpoint_manifest=manifest,
    )


def checkpoint_run_plan(
    *,
    trial_id: str = DEFAULT_CHECKPOINT_TRIAL_ID,
    source_dir: str | Path = "runs/trials/T010",
    config_path: str = "configs/nanofold_dev_cpu_smoke.json",
    features_path: str = "train_tiny.arrow",
) -> dict[str, object]:
    """Return the bounded checkpoint-run intent without touching Modal."""

    checked_trial_id = validate_trial_id(trial_id)
    return {
        "trial_id": checked_trial_id,
        "candidate_id": DEFAULT_CHECKPOINT_CANDIDATE_ID,
        "source_dir": str(source_dir),
        "config_path": config_path,
        "features_path": features_path,
        "requires_approval": APPROVAL_TEXT,
        "requires_modal_deployment": APP_NAME,
        "trial_worker": "TrialRunner.run_checkpoint",
        "training_steps": 1,
        "diffusion_steps": 1,
        "max_templates": 0,
        "checkpoint_filename": "checkpoint.pt",
        "checkpoint_manifest": str(Path(source_dir) / DEFAULT_CHECKPOINT_MANIFEST),
        "writes_baseline_dir": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "claim": "real one-batch NanoFold training checkpoint, not a quality or benchmark claim",
    }


class DeployedModalCheckpointClient:
    """Modal SDK client for the deployed one-batch checkpoint method."""

    def __init__(self, *, environment_name: str | None = None) -> None:
        self.environment_name = environment_name
        try:
            import modal
        except ModuleNotFoundError as exc:
            raise CheckpointRunError("Modal SDK is required for --mode modal checkpoint runs") from exc
        self._modal = modal

    def run_checkpoint(self, payload: dict[str, object]) -> dict[str, object]:
        runner_cls = self._modal.Cls.from_name(APP_NAME, "TrialRunner", environment_name=self.environment_name)
        runner = runner_cls()
        result = runner.run_checkpoint.remote(payload)
        if not isinstance(result, dict):
            raise CheckpointRunError("TrialRunner.run_checkpoint returned a non-object payload")
        return result


def _require_checkpoint_manifest(payload: dict[str, object]) -> dict[str, object]:
    try:
        return validate_checkpoint_manifest(payload)
    except CheckpointTrainingError as exc:
        raise CheckpointRunError(str(exc)) from exc


def _require_trial_source_dir(path: Path, trial_id: str, *, dry_run: bool) -> None:
    as_posix = path.as_posix()
    if "runs/baseline" in as_posix:
        raise CheckpointRunError("one-batch checkpoint must not write runs/baseline")
    if path.name != trial_id or "runs/trials" not in as_posix:
        raise CheckpointRunError(f"one-batch checkpoint output must be under runs/trials/{trial_id}: {path}")
    if not dry_run and path.exists() and any(path.iterdir()):
        raise CheckpointRunError(f"checkpoint source output already exists and is not empty: {path}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
