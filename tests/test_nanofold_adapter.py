from __future__ import annotations

import pytest

from autoalphafold3.config_contract import validate_config_file
from autoalphafold3.modal_app import (
    RESOURCE_TIERS,
    final_validate_seed,
    healthcheck,
    modal_deploy_plan,
    sample_once,
    score_final_seed,
    trial_dir,
    worker_artifact_paths,
)
from autoalphafold3.nanofold_adapter import (
    actual_nanofold_commit,
    expected_nanofold_commit,
    import_smoke_summary,
    nanofold_path_map,
    validate_nanofold_pin,
)
from autoalphafold3.nanofold_checks import run_nanofold_preflight_gates
from autoalphafold3.patch_policy import PatchPolicyError, validate_patch_scope


def test_nanofold_pin_matches_checkout() -> None:
    assert expected_nanofold_commit() == actual_nanofold_commit()
    validate_nanofold_pin()


def test_nanofold_key_paths_exist() -> None:
    path_map = nanofold_path_map()

    assert path_map["train_entrypoint"] == "external/nanofold/nanofold/train/__main__.py"
    assert path_map["pairformer"] == "external/nanofold/nanofold/train/model/pairformer.py"
    assert path_map["docker_train"] == "external/nanofold/docker/Dockerfile.train"


def test_nanofold_import_smoke_reports_dependency_status() -> None:
    summary = import_smoke_summary()

    assert summary["expected_commit"] == summary["actual_commit"]
    modules = {row["module"]: row for row in summary["imports"]}
    assert modules["nanofold"]["ok"] is True
    assert "nanofold.train.model.nanofold" in modules
    assert "nanofold.preprocess.__main__" in modules


def test_config_contract_accepts_local_and_nanofold_configs() -> None:
    local_result = validate_config_file("configs/auto_tiny.json")
    nanofold_result = validate_config_file("configs/nanofold_dev_cpu_smoke.json")

    assert local_result.valid is True
    assert local_result.config_kind == "auto_tiny_scaffold"
    assert nanofold_result.valid is True
    assert nanofold_result.config_kind == "nanofold_training"


def test_patch_policy_allows_mapped_nanofold_surface_and_rejects_preprocess() -> None:
    assert validate_patch_scope(["external/nanofold/nanofold/train/model/pairformer.py"]) == [
        "external/nanofold/nanofold/train/model/pairformer.py"
    ]

    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["external/nanofold/nanofold/preprocess/__main__.py"])
    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["NANOFOLD_COMMIT"])


def test_modal_control_plane_static_contract() -> None:
    status = healthcheck()

    assert status["app_name"] == "autoalphafold3-modal"
    assert status["volumes"]["data"] == "autoalphafold3-data"
    assert status["volumes"]["locked"] == "autoalphafold3-locked"
    assert status["locked_asset_policy"]["target_layout"] == "two_volume"
    assert status["locked_asset_policy"]["search_ready_requires_locked_volume"] is True
    assert "/mnt/autoalphafold3-locked" not in status["mounts"]["trial_workers"]
    assert "/mnt/autoalphafold3-locked" in status["mounts"]["scorer_workers"]
    assert RESOURCE_TIERS["trial"].gpu == "A100-80GB"
    assert RESOURCE_TIERS["trial"].max_containers == 6
    assert RESOURCE_TIERS["trial"].min_containers == 0
    assert str(trial_dir("T123")) == "/mnt/autoalphafold3-runs/trials/T123"
    assert worker_artifact_paths("T123")["artifact_manifest_json"] == (
        "/mnt/autoalphafold3-runs/trials/T123/artifact_manifest.json"
    )
    with pytest.raises(ValueError):
        trial_dir("../bad")


def test_modal_deploy_plan_encodes_non_download_control_plane() -> None:
    plan = modal_deploy_plan()

    assert plan["deploy_command"] == "modal deploy autoalphafold3/modal_app.py"
    assert plan["local_import_safe_without_sdk"] is True
    assert plan["official_training_function"] == "run_trial"
    assert plan["official_training_gpu"] == "A100-80GB"
    assert plan["benchmark_result_produced_locally"] is False
    assert plan["function_contracts"]["run_trial"]["reads_locked_labels"] is False
    assert plan["function_contracts"]["score_trial"]["reads_locked_labels"] is True
    assert plan["locked_asset_policy"]["official_locked_volume"] == "autoalphafold3-locked"
    assert plan["locked_asset_policy"]["search_ready_requires_locked_volume"] is True


def test_modal_control_plane_v4_entrypoints_are_import_safe() -> None:
    sample = sample_once({"trial_id": "T200"})
    final = final_validate_seed({"trial_id": "T201"}, seed=2)
    scored = score_final_seed("T201", seed=2)

    assert sample["function_name"] == "sample_once"
    assert final["function_name"] == "final_validate_seed"
    assert scored["status"] == "INFRA_FAIL"
    assert scored["split"] == "public_val_small"
    with pytest.raises(PermissionError):
        score_final_seed("T201", seed=2, split="smoke")


def test_nanofold_preflight_gates_report_dependency_gaps() -> None:
    gates = run_nanofold_preflight_gates(config_path="configs/nanofold_dev_cpu_smoke.json")
    by_name = {gate.name: gate for gate in gates}

    assert set(by_name) == {"parameter_count", "tiny_forward", "finite_loss"}
    assert by_name["parameter_count"].status in {"passed", "skipped"}
    if by_name["parameter_count"].status == "skipped":
        assert by_name["parameter_count"].reason == "dependency_missing"
    assert by_name["tiny_forward"].status == "skipped"
    assert by_name["finite_loss"].status == "skipped"
