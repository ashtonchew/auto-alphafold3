from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from autoalphafold3.modal_trial_artifacts import (
    ModalTrialArtifactError,
    fetch_modal_trial_artifacts,
)


class FakeCommandRunner:
    def __init__(self, *, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        output_path = Path(args[5])
        output_path.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=self.returncode, stdout="", stderr=self.stderr)


def test_fetch_modal_trial_artifacts_downloads_allowed_artifacts(tmp_path: Path) -> None:
    runner = FakeCommandRunner()

    report = fetch_modal_trial_artifacts(
        trial_id="T164",
        artifacts=["predictions.json", "metrics.json", "predictions.json"],
        output_dir=tmp_path,
        modal_env="main",
        runner=runner,
    )

    assert report.status == "PASS"
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.writes_discovery_ledger is False
    assert report.writes_modal_volume is False
    assert [artifact.artifact for artifact in report.fetched] == ["predictions.json", "metrics.json"]
    assert runner.calls == [
        [
            "modal",
            "volume",
            "get",
            "autoalphafold3-data",
            "runs/trials/T164/predictions.json",
            str(tmp_path / "T164/predictions.json"),
            "--env",
            "main",
        ],
        [
            "modal",
            "volume",
            "get",
            "autoalphafold3-data",
            "runs/trials/T164/metrics.json",
            str(tmp_path / "T164/metrics.json"),
            "--env",
            "main",
        ],
    ]
    assert (tmp_path / "T164/predictions.json").exists()
    assert (tmp_path / "T164/metrics.json").exists()


def test_fetch_modal_trial_artifacts_refuses_unsafe_inputs(tmp_path: Path) -> None:
    runner = FakeCommandRunner()

    with pytest.raises(ModalTrialArtifactError, match="invalid trial_id"):
        fetch_modal_trial_artifacts(
            trial_id="../T164",
            artifacts=["predictions.json"],
            output_dir=tmp_path,
            runner=runner,
        )

    with pytest.raises(ModalTrialArtifactError, match="unsupported trial artifact"):
        fetch_modal_trial_artifacts(
            trial_id="T164",
            artifacts=["../predictions.json"],
            output_dir=tmp_path,
            runner=runner,
        )

    assert runner.calls == []


def test_fetch_modal_trial_artifacts_refuses_overwrite_without_force(tmp_path: Path) -> None:
    destination = tmp_path / "T164/predictions.json"
    destination.parent.mkdir(parents=True)
    destination.write_text("existing", encoding="utf-8")

    with pytest.raises(ModalTrialArtifactError, match="already exists"):
        fetch_modal_trial_artifacts(
            trial_id="T164",
            artifacts=["predictions.json"],
            output_dir=tmp_path,
            runner=FakeCommandRunner(),
        )


def test_fetch_modal_trial_artifacts_reports_modal_failure(tmp_path: Path) -> None:
    with pytest.raises(ModalTrialArtifactError, match="modal volume get failed"):
        fetch_modal_trial_artifacts(
            trial_id="T164",
            artifacts=["predictions.json"],
            output_dir=tmp_path,
            runner=FakeCommandRunner(returncode=1, stderr="missing file"),
        )
