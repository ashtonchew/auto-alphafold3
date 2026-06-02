"""Read-only local fetch helpers for Modal trial artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from typing import Protocol

from autoalphafold3.modal_app import DATA_VOLUME
from autoalphafold3.runner import RunnerError, validate_trial_id

ALLOWED_TRIAL_ARTIFACTS = {
    "artifact_manifest.json",
    "error_report.json",
    "loss_history.json",
    "metrics.json",
    "predictions.json",
    "sampler_manifest.json",
    "training_log.json",
}


class ModalTrialArtifactError(RuntimeError):
    """Raised when a Modal trial artifact fetch is unsafe or fails."""


class CommandRunner(Protocol):
    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Run a command and return the completed process."""


@dataclass(frozen=True)
class FetchedTrialArtifact:
    artifact: str
    remote_path: str
    local_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ModalTrialArtifactFetchReport:
    schema_version: str
    status: str
    trial_id: str
    volume: str
    modal_env: str | None
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    writes_modal_volume: bool
    fetched: list[FetchedTrialArtifact]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["fetched"] = [artifact.to_dict() for artifact in self.fetched]
        return payload


def fetch_modal_trial_artifacts(
    *,
    trial_id: str,
    output_dir: str | Path,
    artifacts: list[str],
    modal_env: str | None = None,
    volume: str = DATA_VOLUME,
    force: bool = False,
    runner: CommandRunner | None = None,
) -> ModalTrialArtifactFetchReport:
    """Fetch selected trial-scoped artifacts from the public Modal data Volume."""

    checked_trial_id = _checked_trial_id(trial_id)
    artifact_names = _checked_artifacts(artifacts)
    destination_root = Path(output_dir) / checked_trial_id
    fetched: list[FetchedTrialArtifact] = []
    command_runner = runner or _run_command
    for artifact in artifact_names:
        remote_path = f"runs/trials/{checked_trial_id}/{artifact}"
        local_path = destination_root / artifact
        if local_path.exists() and not force:
            raise ModalTrialArtifactError(f"local artifact already exists: {local_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["modal", "volume", "get", volume, remote_path, str(local_path)]
        if modal_env is not None:
            cmd.extend(["--env", modal_env])
        completed = command_runner(cmd)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            message = f"modal volume get failed for {remote_path}"
            if detail:
                message = f"{message}: {detail}"
            raise ModalTrialArtifactError(message)
        fetched.append(
            FetchedTrialArtifact(
                artifact=artifact,
                remote_path=remote_path,
                local_path=str(local_path),
            )
        )
    return ModalTrialArtifactFetchReport(
        schema_version="autoaf3.modal_trial_artifact_fetch.v1",
        status="PASS",
        trial_id=checked_trial_id,
        volume=volume,
        modal_env=modal_env,
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        writes_modal_volume=False,
        fetched=fetched,
    )


def _checked_trial_id(trial_id: str) -> str:
    try:
        return validate_trial_id(trial_id)
    except RunnerError as exc:
        raise ModalTrialArtifactError(str(exc)) from exc


def _checked_artifacts(artifacts: list[str]) -> list[str]:
    if not artifacts:
        raise ModalTrialArtifactError("at least one artifact is required")
    checked: list[str] = []
    for artifact in artifacts:
        name = Path(artifact).name
        if name != artifact or artifact not in ALLOWED_TRIAL_ARTIFACTS:
            allowed = ", ".join(sorted(ALLOWED_TRIAL_ARTIFACTS))
            raise ModalTrialArtifactError(f"unsupported trial artifact {artifact!r}; allowed: {allowed}")
        if artifact not in checked:
            checked.append(artifact)
    return checked


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)
