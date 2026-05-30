from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoalphafold3.ledger import append_ledger, read_ledger
from autoalphafold3.orchestrator import poll_trial, record_trial_status, submit_trial
from autoalphafold3.patch_policy import PatchPolicyError, validate_patch_scope
from autoalphafold3.preflight import PreflightError, changed_paths_from_parent, run_preflight
from autoalphafold3.schema import (
    AutoFoldResult,
    AutoFoldTrial,
    FoldCartographerReport,
    TrialStatus,
)
from autoalphafold3.scorer.dry_run import run_scorer_dry_run
from autoalphafold3.scorer.locked_dataset import (
    label_path_for_entry,
    load_locked_manifest,
    sha256_file,
    validate_manifest_file,
    validate_manifest_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_MANIFEST = "data/manifests/smoke.json"


def current_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def smoke_manifest_hash() -> str:
    return sha256_file(REPO_ROOT / SMOKE_MANIFEST)


def valid_trial_dict(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "trial_id": "T001",
        "parent_commit": current_head(),
        "agent_session_id": "pytest",
        "trial_kind": "training",
        "hypothesis": "Dry-run contract smoke test.",
        "move_family": "geometry_loss",
        "diagnostic_target": "local_geometry_weak",
        "prediction": "Preflight should pass without Modal.",
        "patch_path": None,
        "config_path": "configs/auto_tiny.json",
        "budget": "dry_run",
        "seed": 0,
        "max_steps": 1,
        "max_wall_minutes": 1,
        "manifest_hashes": {"smoke": smoke_manifest_hash()},
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "param_cap": 1,
        "gpu_memory_cap": 0.0,
        "cost_cap": 0.0,
        "timeout_cap": 60,
        "artifact_dir": "runs/test_dry_run_artifacts",
    }
    data.update(overrides)
    return data


def write_trial(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(valid_trial_dict(**overrides)), encoding="utf-8")
    return path


def test_schema_rejects_sampler_without_checkpoint() -> None:
    data = valid_trial_dict(trial_kind="sampler", max_steps=None)

    with pytest.raises(ValidationError, match="checkpoint_path"):
        AutoFoldTrial.model_validate(data)


def test_locked_manifest_loads_and_blocks_validation_label_training_access() -> None:
    verified = load_locked_manifest(SMOKE_MANIFEST, repo_root=REPO_ROOT)

    assert verified.manifest.entries[0].target_id == "smoke_A"
    assert len(verified.sha256) == 64
    public_like = verified.manifest.entries[0].model_copy(update={"split": "public_val_small"})
    with pytest.raises(PermissionError):
        label_path_for_entry(public_like, access_mode="training")


def test_manifest_templates_validate_only_when_empty_allowed() -> None:
    templates = [
        "data/manifests/train_tiny.template.json",
        "data/manifests/public_val_small.template.json",
    ]

    reports = validate_manifest_files(
        templates,
        repo_root=REPO_ROOT,
        verify_assets=False,
        allow_empty=True,
    )

    assert [report.entry_count for report in reports] == [0, 0]
    assert all(report.manifest_kind == "official_template" for report in reports)
    with pytest.raises(ValueError, match="no entries"):
        validate_manifest_file(templates[0], repo_root=REPO_ROOT, verify_assets=False)


def test_locked_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    bad_feature = tmp_path / "feature.json"
    bad_label = tmp_path / "label.json"
    bad_feature.write_text("{}", encoding="utf-8")
    bad_label.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_kind": "local_smoke_only",
                "schema_version": "autoaf3.manifest.v1",
                "entries": [
                    {
                        "target_id": "bad_A",
                        "pdb_id": "BAD",
                        "chain_id": "A",
                        "sequence_sha256": "0" * 64,
                        "feature_sha256": "1" * 64,
                        "label_sha256": "2" * 64,
                        "length": 1,
                        "msa_depth_bucket": "toy",
                        "length_bucket": "toy",
                        "split": "smoke",
                        "feature_path": bad_feature.name,
                        "label_path": bad_label.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        load_locked_manifest(manifest.name, repo_root=tmp_path)


def test_patch_policy_accepts_allowed_paths_and_rejects_locked_paths() -> None:
    assert validate_patch_scope(["configs/experiments/T001.json"], repo_root=REPO_ROOT) == [
        "configs/experiments/T001.json"
    ]

    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["autoalphafold3/scorer/calpha_lddt.py"], repo_root=REPO_ROOT)
    with pytest.raises(PatchPolicyError, match="path traversal"):
        validate_patch_scope(["configs/experiments/../escape.json"], repo_root=REPO_ROOT)
    with pytest.raises(PatchPolicyError, match="binary"):
        validate_patch_scope(["configs/experiments/checkpoint.pt"], repo_root=REPO_ROOT)


def test_scorer_dry_run_emits_canonical_metrics() -> None:
    metrics = run_scorer_dry_run(repo_root=REPO_ROOT, trial_id="T001")

    assert metrics["schema_version"] == "autoaf3.metrics.v1"
    assert metrics["scorer_version"] == "calpha_lddt_v1"
    assert metrics["primary_metric"] == "best_val_calpha_lddt"
    assert metrics["metrics"]["best_val_calpha_lddt"] == pytest.approx(1.0)
    assert metrics["fold_cartographer"]["signature"] == "toy_geometry_preserved"


def test_preflight_passes_and_rejects_bad_manifest_hash(tmp_path: Path) -> None:
    trial_path = write_trial(tmp_path)

    result = run_preflight(
        trial_path,
        repo_root=REPO_ROOT,
        manifest_paths={"smoke": SMOKE_MANIFEST},
    )

    assert result.status == TrialStatus.PREFLIGHT_PASSED
    assert result.budget_resources["gpu"] == "none"
    assert {gate.name for gate in result.nanofold_gates} == {
        "parameter_count",
        "tiny_forward",
        "finite_loss",
    }

    bad_trial_path = write_trial(tmp_path, manifest_hashes={"smoke": "0" * 64})
    with pytest.raises(PreflightError, match="manifest hash mismatch"):
        run_preflight(bad_trial_path, repo_root=REPO_ROOT, manifest_paths={"smoke": SMOKE_MANIFEST})


def test_preflight_rejects_budget_overrun(tmp_path: Path) -> None:
    trial_path = write_trial(tmp_path, max_steps=2)

    with pytest.raises(PreflightError, match="max_steps"):
        run_preflight(trial_path, repo_root=REPO_ROOT, manifest_paths={"smoke": SMOKE_MANIFEST})


def test_preflight_can_enforce_git_diff_patch_policy(tmp_path: Path) -> None:
    trial_path = write_trial(tmp_path)

    changed = changed_paths_from_parent(current_head(), repo_root=REPO_ROOT)
    assert isinstance(changed, list)
    with pytest.raises(PatchPolicyError, match="locked"):
        run_preflight(
            trial_path,
            repo_root=REPO_ROOT,
            changed_paths=["autoalphafold3/scorer/calpha_lddt.py"],
            manifest_paths={"smoke": SMOKE_MANIFEST},
        )


def test_preflight_strict_nanofold_gates_fail_when_dependencies_missing(tmp_path: Path) -> None:
    trial_path = write_trial(
        tmp_path,
        config_path="configs/nanofold_dev_cpu_smoke.json",
        artifact_dir="runs/test_strict_nanofold_artifacts",
    )

    with pytest.raises(PreflightError, match="NanoFold-dependent gates did not pass"):
        run_preflight(
            trial_path,
            repo_root=REPO_ROOT,
            manifest_paths={"smoke": SMOKE_MANIFEST},
            strict_nanofold_gates=True,
        )


def test_ledger_append_read_roundtrip(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    row = AutoFoldResult(
        trial_id="T001",
        status=TrialStatus.PREFLIGHT_PASSED,
        candidate_id="local_dry_run",
        metrics={"best_val_calpha_lddt": 1.0},
        fold_cartographer=FoldCartographerReport(signature="toy_geometry_preserved"),
    )

    append_ledger(row, ledger_path=ledger_path)
    append_ledger(row, ledger_path=ledger_path, dedupe=True)
    rows = read_ledger(ledger_path=ledger_path)

    assert len(rows) == 1
    assert rows[0].trial_id == "T001"
    assert rows[0].status == TrialStatus.PREFLIGHT_PASSED


def test_ledger_rejects_invalid_lifecycle_transition(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    first = AutoFoldResult(
        trial_id="T030",
        status=TrialStatus.PREFLIGHT_PASSED,
        candidate_id="local_dry_run",
        metrics={},
        fold_cartographer=FoldCartographerReport(signature="preflight"),
    )
    second = first.model_copy(update={"status": TrialStatus.RUNNING})

    append_ledger(first, ledger_path=ledger_path, validate_lifecycle=True)
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        append_ledger(second, ledger_path=ledger_path, validate_lifecycle=True)


def test_orchestrator_records_lifecycle_transition_once(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    payload = {
        "trial_id": "T020",
        "status": "INFRA_FAIL",
        "candidate_id": "modal_poll",
        "metrics": {},
        "fold_cartographer": {"signature": "modal_timeout", "summary": {}, "buckets": {}},
        "failure_signature": "modal_timeout",
        "postmortem": "Synthetic lifecycle transition for local contract test.",
    }

    record_trial_status(payload, repo_root=REPO_ROOT, ledger_path=ledger_path)
    record_trial_status(payload, repo_root=REPO_ROOT, ledger_path=ledger_path)

    rows = read_ledger(ledger_path=ledger_path)
    assert len(rows) == 1
    assert rows[0].status == TrialStatus.INFRA_FAIL


def test_orchestrator_submit_and_poll_dry_run(tmp_path: Path) -> None:
    trial_path = write_trial(tmp_path, trial_id="T002", artifact_dir="runs/test_dry_run_artifacts_T002")
    ledger_path = tmp_path / "ledger.jsonl"

    call_id = submit_trial(
        trial_path,
        repo_root=REPO_ROOT,
        ledger_path=ledger_path,
        manifest_paths={"smoke": SMOKE_MANIFEST},
    )
    result = poll_trial(call_id, repo_root=REPO_ROOT, ledger_path=ledger_path)

    assert call_id == "dryrun:T002"
    assert result.status == TrialStatus.PREFLIGHT_PASSED
    assert result.metrics["best_val_calpha_lddt"] == pytest.approx(1.0)
