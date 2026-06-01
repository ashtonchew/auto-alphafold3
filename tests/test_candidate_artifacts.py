from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.autoresearch_candidates import (
    CandidateArtifactError,
    create_candidate_envelope,
    create_run_manifest,
    validate_run_id,
    write_candidate_decision,
    write_candidate_evidence,
)


def trial_payload(trial_id: str = "T123") -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "candidate_id": trial_id,
        "config_path": "configs/experiments/local_calpha_geometry_smoke.json",
        "budget": "smoke",
        "max_steps": 10,
        "seed": 0,
    }


def test_candidate_envelope_creates_expected_layout(tmp_path: Path) -> None:
    create_run_manifest(
        repo_root=tmp_path,
        run_id="local-deterministic-001",
        base_commit="abc1234",
        planner="deterministic",
        mode="dry-run",
        description="fixture run",
    )

    envelope = create_candidate_envelope(
        repo_root=tmp_path,
        run_id="local-deterministic-001",
        trial_id="T123",
        hypothesis="Local geometry loss should improve the matched smoke objective.",
        trial=trial_payload(),
        config={"local_calpha_geometry_loss_weight": 0.25},
        patch_text="diff --git a/configs/experiments/x.json b/configs/experiments/x.json\n",
    )

    expected = {
        "candidate_manifest.json",
        "hypothesis.md",
        "patch.diff",
        "config.json",
        "trial.json",
    }
    assert expected <= {path.name for path in envelope.candidate_dir.iterdir()}
    manifest = json.loads(envelope.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "autoaf3.autoresearch_candidate_manifest.v1"
    assert manifest["artifact_dir"].endswith("runs/autoresearch/local-deterministic-001/candidates/T123")
    assert manifest["trial_artifact_dir"] == "runs/trials/T123"
    assert manifest["writes_discovery_ledger"] is False
    run_manifest = json.loads((envelope.root / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["candidate_count"] == 1


def test_run_id_and_trial_id_reject_unsafe_values(tmp_path: Path) -> None:
    with pytest.raises(CandidateArtifactError, match="invalid"):
        validate_run_id("../escape")

    create_run_manifest(
        repo_root=tmp_path,
        run_id="run1",
        base_commit="abc1234",
        planner="manual",
        mode="dry-run",
        description="fixture run",
    )
    with pytest.raises(Exception, match="invalid trial_id"):
        create_candidate_envelope(
            repo_root=tmp_path,
            run_id="run1",
            trial_id="../T123",
            hypothesis="hypothesis",
            trial=trial_payload(),
        )


def test_candidate_dir_non_empty_is_refused(tmp_path: Path) -> None:
    create_run_manifest(
        repo_root=tmp_path,
        run_id="run1",
        base_commit="abc1234",
        planner="manual",
        mode="dry-run",
        description="fixture run",
    )
    existing = tmp_path / "runs/autoresearch/run1/candidates/T123"
    existing.mkdir(parents=True)
    (existing / "user.txt").write_text("preserve\n", encoding="utf-8")

    with pytest.raises(CandidateArtifactError, match="not empty"):
        create_candidate_envelope(
            repo_root=tmp_path,
            run_id="run1",
            trial_id="T123",
            hypothesis="hypothesis",
            trial=trial_payload(),
        )


def test_candidate_decision_updates_summary_and_results(tmp_path: Path) -> None:
    create_run_manifest(
        repo_root=tmp_path,
        run_id="run1",
        base_commit="abc1234",
        planner="manual",
        mode="dry-run",
        description="fixture run",
    )
    envelope = create_candidate_envelope(
        repo_root=tmp_path,
        run_id="run1",
        trial_id="T123",
        hypothesis="Local geometry loss should improve the matched smoke objective.",
        trial=trial_payload(),
    )
    wrote = write_candidate_evidence(
        envelope,
        preflight={"status": "PASS"},
        training_manifest={"status": "SHORT_TRAINING_READY", "official_benchmark_result": False},
        loss_history={"losses": []},
        metrics={"official_benchmark_result": False},
        error_report={"scorer_only": True},
    )
    decision = write_candidate_decision(
        envelope,
        status="DISCARD",
        matched_budget_delta=-0.1,
        global_baseline_delta=-0.5,
        reason="missed matched baseline",
        postmortem="Valid miss; revert candidate.",
    )

    assert len(wrote) == 5
    assert decision["schema_version"] == "autoaf3.autoresearch_decision.v1"
    assert decision["writes_ledger"] is False
    assert decision["writes_discovery_ledger"] is False
    summary = json.loads((envelope.root / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"][0]["status"] == "DISCARD"
    results = (envelope.root / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert results[0].split("\t") == [
        "trial_id",
        "candidate_id",
        "status",
        "primary_metric",
        "matched_budget_delta",
        "global_baseline_delta",
        "provisional_keep",
        "decision_path",
    ]
    assert results[1].startswith("T123\tT123\tDISCARD\tbest_val_calpha_lddt")


def test_candidate_artifacts_do_not_create_locked_run_outputs(tmp_path: Path) -> None:
    create_run_manifest(
        repo_root=tmp_path,
        run_id="run1",
        base_commit="abc1234",
        planner="manual",
        mode="dry-run",
        description="fixture run",
    )

    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
