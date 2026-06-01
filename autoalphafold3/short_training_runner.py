"""Planning and guarded execution for bounded short-training trials."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from autoalphafold3.modal_app import APP_NAME, validate_execution_payload
from autoalphafold3.runner import validate_trial_id
from autoalphafold3.schema import AutoFoldTrial
from autoalphafold3.short_training import (
    DEFAULT_SHORT_TRAINING_MANIFEST,
    ShortTrainingError,
    run_short_nanofold_training,
    short_training_payload,
    validate_short_training_manifest,
)

APPROVAL_TEXT = "I_APPROVE_SHORT_TRAINING_TRIAL"


class ShortTrainingRunError(RuntimeError):
    """Raised when a guarded short-training run cannot complete honestly."""


class ModalShortTrainingClient(Protocol):
    """Small protocol for deployed Modal short-training execution."""

    def run_short_training(self, payload: dict[str, object]) -> dict[str, object]:
        """Run the deployed trial worker and return a short-training manifest."""


@dataclass(frozen=True)
class ShortTrainingRunResult:
    """JSON-friendly short-training runner result."""

    status: str
    mode: str
    trial_id: str
    source_dir: str
    wrote_files: list[str]
    plan: dict[str, object]
    short_training_manifest: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "trial_id": self.trial_id,
            "source_dir": self.source_dir,
            "wrote_files": self.wrote_files,
            "plan": self.plan,
            "short_training_manifest": self.short_training_manifest,
        }


def run_short_training(
    *,
    trial_path: str | Path,
    repo_root: str | Path = ".",
    source_dir: str | Path | None = None,
    features_dir: str | Path = "data/toy/nanofold_fixture",
    features_path: str = "tiny_features.arrow",
    approval: str | None = None,
    mode: str = "dry-run",
    modal_env: str | None = None,
    modal_client: ModalShortTrainingClient | None = None,
) -> ShortTrainingRunResult:
    """Plan or execute a bounded short-training run."""

    root = Path(repo_root)
    trial = _load_trial(root / trial_path)
    checked_trial_id = validate_trial_id(trial.trial_id)
    source = root / (source_dir or Path("runs/trials") / checked_trial_id)
    _require_trial_source_dir(source, checked_trial_id, dry_run=mode == "dry-run")
    plan = short_training_run_plan(
        trial=trial,
        source_dir=source,
        features_dir=features_dir,
        features_path=features_path,
    )
    if mode == "dry-run":
        return ShortTrainingRunResult(
            status="PLANNED",
            mode=mode,
            trial_id=checked_trial_id,
            source_dir=str(source),
            wrote_files=[],
            plan=plan,
        )
    payload = _payload_from_trial(trial, features_path=features_path, local_only=mode == "local-fixture")
    if mode == "local-fixture":
        try:
            manifest = run_short_nanofold_training(
                payload,
                features_dir=root / features_dir,
                output_dir=source,
                repo_root=root,
                local_only=True,
            )
        except ShortTrainingError as exc:
            raise ShortTrainingRunError(str(exc)) from exc
        return ShortTrainingRunResult(
            status="PASS",
            mode=mode,
            trial_id=checked_trial_id,
            source_dir=str(source),
            wrote_files=_short_training_wrote_files(source),
            plan=plan,
            short_training_manifest=manifest,
        )
    if mode != "modal":
        raise ShortTrainingRunError(f"unsupported short-training mode: {mode}")
    if approval != APPROVAL_TEXT:
        raise ShortTrainingRunError(f"short-training Modal run requires --approve {APPROVAL_TEXT}")
    client = modal_client if modal_client is not None else DeployedModalShortTrainingClient(environment_name=modal_env)
    validate_execution_payload(payload, role="trial")
    manifest = _require_short_training_manifest(client.run_short_training(payload))
    source.mkdir(parents=True, exist_ok=True)
    manifest_path = source / DEFAULT_SHORT_TRAINING_MANIFEST
    _atomic_write_json(manifest_path, manifest)
    return ShortTrainingRunResult(
        status="PASS",
        mode=mode,
        trial_id=checked_trial_id,
        source_dir=str(source),
        wrote_files=[str(manifest_path)],
        plan=plan,
        short_training_manifest=manifest,
    )


def short_training_run_plan(
    *,
    trial: AutoFoldTrial,
    source_dir: str | Path,
    features_dir: str | Path,
    features_path: str,
) -> dict[str, object]:
    """Return the bounded short-training intent without touching workers."""

    return {
        "trial_id": trial.trial_id,
        "candidate_id": trial.trial_id,
        "source_dir": str(source_dir),
        "config_path": trial.config_path,
        "features_dir": str(features_dir),
        "features_path": features_path,
        "requires_approval": APPROVAL_TEXT,
        "requires_modal_deployment": APP_NAME,
        "trial_worker": "TrialRunner.run",
        "training_steps": trial.max_steps,
        "max_templates": 0,
        "budget": trial.budget.value,
        "checkpoint_filename": "checkpoint.pt",
        "short_training_manifest": str(Path(source_dir) / DEFAULT_SHORT_TRAINING_MANIFEST),
        "writes_baseline_dir": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "claim": "bounded NanoFold short-training artifact, not a scored benchmark or discovery claim",
    }


class DeployedModalShortTrainingClient:
    """Modal SDK client for the deployed short-training worker."""

    def __init__(self, *, environment_name: str | None = None) -> None:
        self.environment_name = environment_name
        try:
            import modal
        except ModuleNotFoundError as exc:
            raise ShortTrainingRunError("Modal SDK is required for --mode modal short-training runs") from exc
        self._modal = modal

    def run_short_training(self, payload: dict[str, object]) -> dict[str, object]:
        runner_cls = self._modal.Cls.from_name(APP_NAME, "TrialRunner", environment_name=self.environment_name)
        runner = runner_cls()
        result = runner.run.remote(payload)
        if not isinstance(result, dict):
            raise ShortTrainingRunError("TrialRunner.run returned a non-object payload")
        return result


def _load_trial(path: Path) -> AutoFoldTrial:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AutoFoldTrial.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - CLI reports schema errors as refusal text.
        raise ShortTrainingRunError(f"invalid short-training trial: {exc}") from exc


def _payload_from_trial(
    trial: AutoFoldTrial,
    *,
    features_path: str,
    local_only: bool,
) -> dict[str, object]:
    if trial.max_steps is None:
        raise ShortTrainingRunError("short-training trials require max_steps")
    return short_training_payload(
        trial_id=trial.trial_id,
        candidate_id=trial.trial_id,
        config_path=trial.config_path,
        features_path=features_path,
        max_steps=trial.max_steps,
        budget=trial.budget.value,
        seed=trial.seed,
        artifact_dir=trial.artifact_dir,
        local_only=local_only,
    )


def _require_short_training_manifest(payload: dict[str, object]) -> dict[str, object]:
    try:
        return validate_short_training_manifest(payload)
    except ShortTrainingError as exc:
        raise ShortTrainingRunError(str(exc)) from exc


def _require_trial_source_dir(path: Path, trial_id: str, *, dry_run: bool) -> None:
    as_posix = path.as_posix()
    if "runs/baseline" in as_posix:
        raise ShortTrainingRunError("short training must not write runs/baseline")
    if path.name != trial_id or "runs/trials" not in as_posix:
        raise ShortTrainingRunError(f"short-training output must be under runs/trials/{trial_id}: {path}")
    if not dry_run and path.exists() and any(path.iterdir()):
        raise ShortTrainingRunError(f"short-training output already exists and is not empty: {path}")


def _short_training_wrote_files(source: Path) -> list[str]:
    names = [
        "checkpoint.pt",
        DEFAULT_SHORT_TRAINING_MANIFEST,
        "loss_history.json",
        "artifact_manifest.json",
        "training_log.json",
        "stdout.log",
        "stderr.log",
        "patch.diff",
        "DONE",
    ]
    return [str(source / name) for name in names]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
