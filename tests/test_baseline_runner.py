from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoalphafold3.baseline_runner import (
    APPROVAL_TEXT,
    BaselineRunError,
    baseline_trial_payload,
    run_baseline,
)

SHA = "a" * 64
REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeModalBaselineClient:
    def __init__(self, scored: dict[str, object]) -> None:
        self.scored = scored
        self.ran_payload: dict[str, object] | None = None

    def run_trial(self, trial_payload: dict[str, object]) -> dict[str, object]:
        self.ran_payload = trial_payload
        return {"status": "DONE", "trial_id": trial_payload["trial_id"]}

    def score_trial(self, trial_id: str) -> dict[str, object]:
        return {**self.scored, "trial_id": trial_id}


def official_score() -> dict[str, object]:
    return {
        "schema_version": "autoaf3.metrics.v1",
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "status": "SCORED",
        "candidate_id": "baseline_auto_tiny",
        "seed": 0,
        "split": "public_val_small",
        "official_benchmark_result": True,
        "local_only": False,
        "max_templates": 0,
        "manifests": {"train_tiny": SHA, "public_val_small": SHA},
        "label_hashes": {"public_val_small": SHA},
        "metrics": {"best_val_calpha_lddt": 0.42},
        "fold_cartographer": {"signature": "baseline_run", "summary": {}, "buckets": {}},
        "error_report": {"failure_signature": None, "scorer_only": True},
        "artifacts": {"metrics_json": "runs/trials/T000/metrics.json"},
    }


def test_run_baseline_dry_run_plans_without_writing(tmp_path: Path) -> None:
    result = run_baseline(repo_root=tmp_path, mode="dry-run")

    assert result.status == "PLANNED"
    assert result.wrote_files == []
    assert result.plan["writes_baseline_dir"] is False
    assert result.plan["writes_ledger"] is False
    assert not (tmp_path / "runs").exists()


def test_run_baseline_modal_requires_exact_approval(tmp_path: Path) -> None:
    with pytest.raises(BaselineRunError, match=APPROVAL_TEXT):
        run_baseline(
            repo_root=tmp_path,
            mode="modal",
            approval="yes",
            modal_client=FakeModalBaselineClient(official_score()),
        )

    assert not (tmp_path / "runs").exists()


def test_run_baseline_writes_only_trial_scoped_lock_source_artifacts(tmp_path: Path) -> None:
    client = FakeModalBaselineClient(official_score())

    result = run_baseline(
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        modal_client=client,
    )

    source = tmp_path / "runs/trials/T000"
    assert result.status == "PASS"
    assert client.ran_payload == baseline_trial_payload(trial_id="T000")
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == [
        "runs",
        "runs/trials",
        "runs/trials/T000",
        "runs/trials/T000/error_report.json",
        "runs/trials/T000/metrics.json",
    ]
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    error_report = json.loads((source / "error_report.json").read_text(encoding="utf-8"))
    assert metrics["official_benchmark_result"] is True
    assert metrics["local_only"] is False
    assert error_report["scorer_only"] is True
    assert not (tmp_path / "runs/baseline").exists()


def test_run_baseline_refuses_local_or_scaffold_scorer_evidence(tmp_path: Path) -> None:
    score = official_score()
    score["official_benchmark_result"] = False

    with pytest.raises(BaselineRunError, match="official_benchmark_result"):
        run_baseline(
            repo_root=tmp_path,
            mode="modal",
            approval=APPROVAL_TEXT,
            modal_client=FakeModalBaselineClient(score),
        )

    assert not (tmp_path / "runs").exists()


def test_run_baseline_refuses_to_write_baseline_dir(tmp_path: Path) -> None:
    with pytest.raises(BaselineRunError, match="runs/baseline"):
        run_baseline(
            repo_root=tmp_path,
            source_dir="runs/baseline/T000",
            mode="dry-run",
        )


def test_run_baseline_refuses_to_overwrite_source_outputs(tmp_path: Path) -> None:
    source = tmp_path / "runs/trials/T000"
    source.mkdir(parents=True)
    (source / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(BaselineRunError, match="already exists"):
        run_baseline(
            repo_root=tmp_path,
            mode="modal",
            approval=APPROVAL_TEXT,
            modal_client=FakeModalBaselineClient(official_score()),
        )


def test_run_baseline_cli_dry_run_is_structured_json(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "run-baseline",
            "--repo-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "PLANNED"
    assert payload["plan"]["expected_lock_step"].startswith("python3 -m autoalphafold3.agent lock-baseline")
    assert not (tmp_path / "runs").exists()
