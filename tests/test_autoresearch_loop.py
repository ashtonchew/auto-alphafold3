from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoalphafold3.autoresearch_loop import APPROVAL_TEXT, AutoresearchLoopError, run_autoresearch_loop

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_autoresearch_ladder_plans_t120_to_t125(tmp_path: Path) -> None:
    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="pytest-deterministic",
        mode="dry-run",
        planner="deterministic",
        start_trial_id="T120",
        max_candidates=6,
    )

    assert result.status == "PLANNED"
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    assert result.generated_trials == ["T120", "T121", "T122", "T123", "T124", "T125"]
    assert all(decision["status"] == "DRAFT" for decision in result.decisions)
    assert all(decision["planning_status"] == "PLANNED" for decision in result.decisions)
    run_dir = Path(result.run_dir)
    assert run_dir.exists()
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()
    trials = [
        json.loads((run_dir / "candidates" / trial_id / "trial.json").read_text(encoding="utf-8"))
        for trial_id in result.generated_trials
    ]
    assert [trial["budget"] for trial in trials] == ["smoke", "smoke", "trial", "trial", "trial", "sampler"]
    assert [trial.get("max_steps") for trial in trials] == [10, 10, 250, 250, 250, None]
    assert trials[-1]["trial_kind"] == "sampler"
    assert trials[-1]["checkpoint_path"] == "runs/trials/T123/checkpoint.pt"
    assert "max_steps" not in trials[-1]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [candidate["status"] for candidate in summary["candidates"]] == ["DRAFT"] * 6
    results = (run_dir / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(results) == 7


def test_deterministic_autoresearch_ladder_partial_count(tmp_path: Path) -> None:
    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="partial",
        mode="dry-run",
        planner="deterministic",
        start_trial_id="T130",
        max_candidates=2,
    )

    assert result.generated_trials == ["T130", "T131"]
    assert all(Path(path).is_relative_to(tmp_path) for path in result.candidate_dirs)
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_deterministic_sampler_checkpoint_follows_start_trial_id(tmp_path: Path) -> None:
    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="custom-start",
        mode="dry-run",
        planner="deterministic",
        start_trial_id="T130",
        max_candidates=6,
    )

    sampler_trial = json.loads(
        (Path(result.run_dir) / "candidates" / "T135" / "trial.json").read_text(encoding="utf-8")
    )
    assert sampler_trial["checkpoint_path"] == "runs/trials/T133/checkpoint.pt"


def test_manual_autoresearch_plan_consumes_prepared_candidate(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/candidate.json"
    candidate_plan.parent.mkdir(parents=True)
    candidate_plan.write_text(
        json.dumps(
            {
                "hypothesis": "Manual candidate validates prepared candidate planning.",
                "trial": {
                    "trial_id": "T200",
                    "parent_commit": "abc1234",
                    "agent_session_id": "pytest-manual",
                    "trial_kind": "training",
                    "hypothesis": "Manual candidate validates prepared candidate planning.",
                    "move_family": "geometry_loss",
                    "diagnostic_target": "local_geometry_weak",
                    "prediction": {
                        "causal_component": "local_calpha_geometry_loss",
                        "predicted_axis": "local_geometry",
                        "predicted_direction": "up",
                        "expected_lddt_delta_band": [0.001, 0.01],
                    },
                    "patch_path": None,
                    "config_path": "configs/experiments/local_calpha_geometry_smoke.json",
                    "budget": "smoke",
                    "seed": 0,
                    "n_res": 32,
                    "max_steps": 10,
                    "max_wall_minutes": 5,
                    "manifest_hashes": {},
                    "scorer_version": "calpha_lddt_v1",
                    "primary_metric": "best_val_calpha_lddt",
                    "param_cap": 176514,
                    "gpu_memory_cap": 80.0,
                    "cost_cap": 2.0,
                    "timeout_cap": 300,
                    "artifact_dir": "runs/trials/T200",
                    "checkpoint_path": None,
                },
                "config": {"local_calpha_geometry_loss_weight": 0.25, "max_templates": 0},
                "patch_text": "diff --git a/configs/experiments/x.json b/configs/experiments/x.json\n",
            }
        ),
        encoding="utf-8",
    )

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="manual",
        mode="dry-run",
        planner="manual",
        candidate_plan="configs/experiments/candidate.json",
    )

    assert result.generated_trials == ["T200"]
    candidate_dir = tmp_path / "runs/autoresearch/manual/candidates/T200"
    assert (candidate_dir / "trial.json").exists()
    assert (candidate_dir / "config.json").exists()
    assert (candidate_dir / "preflight.json").exists()
    assert not (candidate_dir / "decision.json").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()


def test_manual_autoresearch_plan_refuses_absolute_path(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/candidate.json"
    candidate_plan.parent.mkdir(parents=True)
    candidate_plan.write_text("{}\n", encoding="utf-8")

    with pytest.raises(AutoresearchLoopError, match="repo-relative"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="manual",
            mode="dry-run",
            planner="manual",
            candidate_plan=candidate_plan,
        )
    assert not (tmp_path / "runs/autoresearch/manual").exists()


def test_manual_autoresearch_plan_refuses_non_object_json(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/candidate.json"
    candidate_plan.parent.mkdir(parents=True)
    candidate_plan.write_text("[]\n", encoding="utf-8")

    with pytest.raises(AutoresearchLoopError, match="JSON object"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="manual",
            mode="dry-run",
            planner="manual",
            candidate_plan="configs/experiments/candidate.json",
        )
    assert not (tmp_path / "runs/autoresearch/manual").exists()


def test_manual_autoresearch_plan_refuses_authority_claims_and_locked_patch(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/candidate.json"
    candidate_plan.parent.mkdir(parents=True)
    payload = {
        "hypothesis": "Manual candidate validates prepared candidate planning.",
        "trial": {
            "trial_id": "T200",
            "parent_commit": "abc1234",
            "agent_session_id": "pytest-manual",
            "trial_kind": "training",
            "hypothesis": "Manual candidate validates prepared candidate planning.",
            "move_family": "geometry_loss",
            "diagnostic_target": "local_geometry_weak",
            "prediction": {
                "causal_component": "local_calpha_geometry_loss",
                "predicted_axis": "local_geometry",
                "predicted_direction": "up",
                "expected_lddt_delta_band": [0.001, 0.01],
            },
            "patch_path": None,
            "config_path": "configs/experiments/local_calpha_geometry_smoke.json",
            "budget": "smoke",
            "seed": 0,
            "n_res": 32,
            "max_steps": 10,
            "max_wall_minutes": 5,
            "manifest_hashes": {},
            "scorer_version": "calpha_lddt_v1",
            "primary_metric": "best_val_calpha_lddt",
            "param_cap": 176514,
            "gpu_memory_cap": 80.0,
            "cost_cap": 2.0,
            "timeout_cap": 300,
            "artifact_dir": "runs/trials/T200",
            "checkpoint_path": None,
        },
        "config": {"official_benchmark_result": True},
        "patch_text": "diff --git a/autoalphafold3/modal_app.py b/autoalphafold3/modal_app.py\n",
    }
    candidate_plan.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AutoresearchLoopError, match="official_benchmark_result"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="manual",
            mode="dry-run",
            planner="manual",
            candidate_plan="configs/experiments/candidate.json",
        )
    assert not (tmp_path / "runs/autoresearch/manual").exists()

    payload["config"] = {"local_calpha_geometry_loss_weight": 0.25}
    candidate_plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AutoresearchLoopError, match="locked"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="manual-locked-patch",
            mode="dry-run",
            planner="manual",
            candidate_plan="configs/experiments/candidate.json",
        )


def test_manual_autoresearch_plan_refuses_bad_artifact_dir_and_templates(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/candidate.json"
    candidate_plan.parent.mkdir(parents=True)
    payload = {
        "hypothesis": "Manual candidate validates prepared candidate planning.",
        "trial": {
            "trial_id": "T200",
            "parent_commit": "abc1234",
            "agent_session_id": "pytest-manual",
            "trial_kind": "training",
            "hypothesis": "Manual candidate validates prepared candidate planning.",
            "move_family": "geometry_loss",
            "diagnostic_target": "local_geometry_weak",
            "prediction": {
                "causal_component": "local_calpha_geometry_loss",
                "predicted_axis": "local_geometry",
                "predicted_direction": "up",
                "expected_lddt_delta_band": [0.001, 0.01],
            },
            "patch_path": None,
            "config_path": "configs/experiments/local_calpha_geometry_smoke.json",
            "budget": "smoke",
            "seed": 0,
            "n_res": 32,
            "max_steps": 10,
            "max_wall_minutes": 5,
            "manifest_hashes": {},
            "scorer_version": "calpha_lddt_v1",
            "primary_metric": "best_val_calpha_lddt",
            "param_cap": 176514,
            "gpu_memory_cap": 80.0,
            "cost_cap": 2.0,
            "timeout_cap": 300,
            "artifact_dir": "runs/baseline",
            "checkpoint_path": None,
        },
        "config": {"local_calpha_geometry_loss_weight": 0.25, "max_templates": 0},
        "patch_text": "diff --git a/configs/experiments/x.json b/configs/experiments/x.json\n",
    }
    candidate_plan.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AutoresearchLoopError, match="artifact_dir"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="bad-artifact-dir",
            mode="dry-run",
            planner="manual",
            candidate_plan="configs/experiments/candidate.json",
        )

    payload["trial"]["artifact_dir"] = "runs/trials/T200"
    payload["config"]["max_templates"] = 1
    candidate_plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AutoresearchLoopError, match="max_templates"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="bad-templates",
            mode="dry-run",
            planner="manual",
            candidate_plan="configs/experiments/candidate.json",
        )


def test_manual_autoresearch_plan_refuses_duplicate_trial_ids_before_publish(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/candidate.json"
    candidate_plan.parent.mkdir(parents=True)
    trial = {
        "trial_id": "T200",
        "parent_commit": "abc1234",
        "agent_session_id": "pytest-manual",
        "trial_kind": "training",
        "hypothesis": "Manual candidate validates prepared candidate planning.",
        "move_family": "geometry_loss",
        "diagnostic_target": "local_geometry_weak",
        "prediction": {
            "causal_component": "local_calpha_geometry_loss",
            "predicted_axis": "local_geometry",
            "predicted_direction": "up",
            "expected_lddt_delta_band": [0.001, 0.01],
        },
        "patch_path": None,
        "config_path": "configs/experiments/local_calpha_geometry_smoke.json",
        "budget": "smoke",
        "seed": 0,
        "n_res": 32,
        "max_steps": 10,
        "max_wall_minutes": 5,
        "manifest_hashes": {},
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "param_cap": 176514,
        "gpu_memory_cap": 80.0,
        "cost_cap": 2.0,
        "timeout_cap": 300,
        "artifact_dir": "runs/trials/T200",
        "checkpoint_path": None,
    }
    candidate_plan.write_text(json.dumps({"candidates": [{"trial": trial}, {"trial": trial}]}), encoding="utf-8")

    with pytest.raises(AutoresearchLoopError, match="duplicate"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="manual-duplicate",
            mode="dry-run",
            planner="manual",
            candidate_plan="configs/experiments/candidate.json",
        )
    assert not (tmp_path / "runs/autoresearch/manual-duplicate").exists()


def test_manual_autoresearch_plan_refuses_bad_config_path_and_locked_label_patch(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/candidate.json"
    candidate_plan.parent.mkdir(parents=True)
    payload = {
        "hypothesis": "Manual candidate validates prepared candidate planning.",
        "trial": {
            "trial_id": "T200",
            "parent_commit": "abc1234",
            "agent_session_id": "pytest-manual",
            "trial_kind": "training",
            "hypothesis": "Manual candidate validates prepared candidate planning.",
            "move_family": "geometry_loss",
            "diagnostic_target": "local_geometry_weak",
            "prediction": {
                "causal_component": "local_calpha_geometry_loss",
                "predicted_axis": "local_geometry",
                "predicted_direction": "up",
                "expected_lddt_delta_band": [0.001, 0.01],
            },
            "patch_path": None,
            "config_path": "runs/baseline/metrics.json",
            "budget": "smoke",
            "seed": 0,
            "n_res": 32,
            "max_steps": 10,
            "max_wall_minutes": 5,
            "manifest_hashes": {},
            "scorer_version": "calpha_lddt_v1",
            "primary_metric": "best_val_calpha_lddt",
            "param_cap": 176514,
            "gpu_memory_cap": 80.0,
            "cost_cap": 2.0,
            "timeout_cap": 300,
            "artifact_dir": "runs/trials/T200",
            "checkpoint_path": None,
        },
        "config": {"local_calpha_geometry_loss_weight": 0.25, "max_templates": 0},
        "patch_text": "diff --git a/configs/experiments/x.json b/configs/experiments/x.json\n",
    }
    candidate_plan.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AutoresearchLoopError, match="config_path"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="bad-config-path",
            mode="dry-run",
            planner="manual",
            candidate_plan="configs/experiments/candidate.json",
        )

    payload["trial"]["config_path"] = "configs/experiments/local_calpha_geometry_smoke.json"
    payload["patch_text"] = (
        "diff --git a/configs/experiments/x.json b/configs/experiments/x.json\n"
        "--- a/configs/experiments/x.json\n"
        "+++ b/configs/experiments/x.json\n"
        "@@ -0,0 +1 @@\n"
        "+{\"labels\":\"/mnt/autoalphafold3-locked/public_val_labels.json\"}\n"
    )
    candidate_plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AutoresearchLoopError, match="locked labels"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="locked-label-patch",
            mode="dry-run",
            planner="manual",
            candidate_plan="configs/experiments/candidate.json",
        )


def test_autoresearch_loop_refuses_live_mode_even_with_approval(tmp_path: Path) -> None:
    with pytest.raises(AutoresearchLoopError, match="not implemented"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="live",
            mode="modal",
            planner="deterministic",
            approval=APPROVAL_TEXT,
        )


def test_autoresearch_loop_requires_live_approval(tmp_path: Path) -> None:
    with pytest.raises(AutoresearchLoopError, match=APPROVAL_TEXT):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="live",
            mode="modal",
            planner="deterministic",
        )


def test_autoresearch_loop_cli_dry_run_is_structured_json(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "autoresearch-loop",
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "cli-deterministic",
            "--mode",
            "dry-run",
            "--planner",
            "deterministic",
            "--start-trial-id",
            "T120",
            "--max-candidates",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "PLANNED"
    assert payload["generated_trials"] == ["T120", "T121"]
    assert payload["starts_search"] is False
    assert payload["writes_ledger"] is False
    assert payload["writes_discovery_ledger"] is False
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_autoresearch_loop_cli_duplicate_run_is_structured_json(tmp_path: Path) -> None:
    args = [
        sys.executable,
        "-m",
        "autoalphafold3.agent",
        "autoresearch-loop",
        "--repo-root",
        str(tmp_path),
        "--run-id",
        "cli-duplicate",
        "--mode",
        "dry-run",
        "--planner",
        "deterministic",
        "--max-candidates",
        "1",
    ]
    subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=True)

    result = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert "already exists" in payload["error"]
