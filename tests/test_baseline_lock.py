from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoalphafold3.baseline_lock import APPROVAL_TEXT, BaselineLockError, lock_baseline_from_scored_artifacts

SHA = "a" * 64


def write_source(tmp_path: Path, *, official: bool = True, local_only: bool = False) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    metrics = {
        "schema_version": "autoaf3.metrics.v1",
        "trial_id": "baseline_auto_tiny",
        "candidate_id": "baseline_lock",
        "official_benchmark_result": official,
        "local_only": local_only,
        "primary_metric": "best_val_calpha_lddt",
        "scorer_version": "calpha_lddt_v1",
        "split": "public_val_small",
        "status": "SCORED",
        "seed": 0,
        "max_templates": 0,
        "metrics": {"best_val_calpha_lddt": 0.42},
        "manifests": {"train_tiny": SHA, "public_val_small": SHA},
        "label_hashes": {"public_val_small": SHA},
        "fold_cartographer": {"signature": "baseline_locked", "summary": {}, "buckets": {}},
        "artifacts": {"metrics_json": "runs/trials/T000/metrics.json"},
    }
    error_report = {
        "scorer_only": True,
        "failure_signature": None,
        "template_policy": "max_templates=0",
        "artifacts": {"error_report_json": "runs/trials/T000/error_report.json"},
    }
    fingerprints = {
        "template_policy": "max_templates=0",
        "files": {
            "features/train_tiny.arrow": SHA,
            "features/public_val_small.arrow": SHA,
        },
    }
    (source / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (source / "error_report.json").write_text(json.dumps(error_report), encoding="utf-8")
    fingerprint_path = tmp_path / "feature_fingerprints.json"
    fingerprint_path.write_text(json.dumps(fingerprints), encoding="utf-8")
    return source, fingerprint_path, tmp_path / "runs" / "baseline"


def test_lock_baseline_requires_explicit_human_approval(tmp_path: Path) -> None:
    source, fingerprints, baseline = write_source(tmp_path)

    with pytest.raises(BaselineLockError, match=APPROVAL_TEXT):
        lock_baseline_from_scored_artifacts(
            source_dir=source,
            feature_fingerprints_path=fingerprints,
            baseline_dir=baseline,
        )

    assert not baseline.exists()


def test_lock_baseline_dry_run_validates_without_writing(tmp_path: Path) -> None:
    source, fingerprints, baseline = write_source(tmp_path)

    result = lock_baseline_from_scored_artifacts(
        source_dir=source,
        feature_fingerprints_path=fingerprints,
        baseline_dir=baseline,
        approval=APPROVAL_TEXT,
        dry_run=True,
    )

    assert result.status == "PASS"
    assert result.wrote_files == []
    assert result.readiness["status"] == "PASS"
    assert not baseline.exists()


def test_lock_baseline_writes_only_locked_baseline_files(tmp_path: Path) -> None:
    source, fingerprints, baseline = write_source(tmp_path)

    result = lock_baseline_from_scored_artifacts(
        source_dir=source,
        feature_fingerprints_path=fingerprints,
        baseline_dir=baseline,
        approval=APPROVAL_TEXT,
    )

    assert result.status == "PASS"
    assert sorted(path.name for path in baseline.iterdir()) == [
        "error_report.json",
        "feature_fingerprints.json",
        "metrics.json",
    ]
    metrics = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["artifacts"]["metrics_json"] == "runs/baseline/metrics.json"
    assert result.readiness["baseline_score"] == pytest.approx(0.42)


@pytest.mark.parametrize(
    ("official", "local_only", "problem"),
    [
        (False, False, "official benchmark"),
        (True, True, "must not be local_only"),
    ],
)
def test_lock_baseline_refuses_non_official_sources(
    tmp_path: Path,
    official: bool,
    local_only: bool,
    problem: str,
) -> None:
    source, fingerprints, baseline = write_source(tmp_path, official=official, local_only=local_only)

    with pytest.raises(BaselineLockError, match=problem):
        lock_baseline_from_scored_artifacts(
            source_dir=source,
            feature_fingerprints_path=fingerprints,
            baseline_dir=baseline,
            approval=APPROVAL_TEXT,
        )


def test_lock_baseline_refuses_to_overwrite_existing_baseline(tmp_path: Path) -> None:
    source, fingerprints, baseline = write_source(tmp_path)
    baseline.mkdir(parents=True)
    (baseline / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(BaselineLockError, match="not empty"):
        lock_baseline_from_scored_artifacts(
            source_dir=source,
            feature_fingerprints_path=fingerprints,
            baseline_dir=baseline,
            approval=APPROVAL_TEXT,
        )


def test_lock_baseline_cli_dry_run(tmp_path: Path) -> None:
    source, fingerprints, baseline = write_source(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "lock-baseline",
            "--source-dir",
            str(source),
            "--feature-fingerprints",
            str(fingerprints),
            "--baseline-dir",
            str(baseline),
            "--approve",
            APPROVAL_TEXT,
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["dry_run"] is True
    assert not baseline.exists()


def test_lock_baseline_cli_refusal_is_structured_json(tmp_path: Path) -> None:
    source, fingerprints, baseline = write_source(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "lock-baseline",
            "--source-dir",
            str(source),
            "--feature-fingerprints",
            str(fingerprints),
            "--baseline-dir",
            str(baseline),
            "--approve",
            "wrong",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["status"] == "FAIL"
    assert APPROVAL_TEXT in payload["error"]
    assert "Traceback" not in result.stderr
