from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from autoalphafold3 import agent
from autoalphafold3.autoresearch_loop import (
    APPROVAL_TEXT,
    AutoresearchCandidatePlan,
    DeployedTrustedAutoresearchClient,
    MODAL_WORKER_RESULT_TIMEOUT_S,
    AutoresearchLoopError,
    run_autoresearch_loop,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeLLMPlanner:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def plan(self, **kwargs: object) -> AutoresearchCandidatePlan:
        self.calls.append(kwargs)
        return AutoresearchCandidatePlan.model_validate(self.payload)


class FakeTrustedAutoresearchClient:
    def __init__(self, payload: dict[str, object], score_payload: dict[str, object] | None = None) -> None:
        self.payload = payload
        self.score_payload = score_payload
        self.submitted_trials: list[dict[str, object]] = []
        self.scored_trials: list[str] = []

    def submit_and_poll_trial(self, trial: dict[str, object]) -> dict[str, object]:
        self.submitted_trials.append(trial)
        return self.payload

    def score_trial(self, trial_id: str) -> dict[str, object]:
        self.scored_trials.append(trial_id)
        return self.score_payload or _scorer_fail_payload(trial_id)


SHA = "c" * 64


def _short_training_manifest(trial_id: str = "T130") -> dict[str, object]:
    return {
        "schema_version": "autoaf3.short_training_manifest.v1",
        "status": "SHORT_TRAINING_READY",
        "trial_id": trial_id,
        "candidate_id": trial_id,
        "budget": "smoke",
        "real_training_performed": True,
        "local_only": False,
        "official_benchmark_result": False,
        "training_steps": 10,
        "max_steps": 10,
        "max_templates": 0,
        "seed": 0,
        "config_path": "configs/nanofold_dev_cpu_smoke.json",
        "features_path": "nanofold_event_small_no_templates.arrow",
        "feature_sha256": SHA,
        "checkpoint_path": f"/mnt/autoalphafold3/runs/trials/{trial_id}/checkpoint.pt",
        "checkpoint_sha256": SHA,
        "checkpoint_size_bytes": 1234,
        "checkpoint_source": "short_nanofold_training",
        "loss_history_path": f"/mnt/autoalphafold3/runs/trials/{trial_id}/loss_history.json",
        "training_log_path": f"/mnt/autoalphafold3/runs/trials/{trial_id}/training_log.json",
        "artifact_manifest_path": f"/mnt/autoalphafold3/runs/trials/{trial_id}/artifact_manifest.json",
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "final_losses": {"total_loss": 1.0},
        "runtime_s": 1.0,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "reads_locked_labels": False,
    }


def _infra_fail_result(trial_id: str = "T130") -> dict[str, object]:
    return {
        "schema_version": "autoaf3.result.v1",
        "trial_id": trial_id,
        "status": "INFRA_FAIL",
        "candidate_id": trial_id,
        "metrics": {},
        "fold_cartographer": {"signature": "modal_worker_failed", "summary": {}, "buckets": {}},
        "artifacts": {},
        "failure_signature": "pytest_modal_worker_failed",
        "postmortem": "Fake Modal worker failure.",
    }


def _scorer_fail_payload(trial_id: str = "T130") -> dict[str, object]:
    return {
        "schema_version": "autoaf3.metrics.v1",
        "status": "FAIL",
        "trial_id": trial_id,
        "candidate_id": trial_id,
        "primary_metric": "best_val_calpha_lddt",
        "metrics": {"best_val_calpha_lddt": 0.0},
        "fold_cartographer": {"signature": "prediction_artifact_missing", "summary": {"reason": "missing predictions.json"}, "buckets": {}},
        "error_report": {"failure_signature": "prediction_artifact_missing", "reason": "missing predictions.json", "scorer_only": True},
        "official_benchmark_result": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "artifacts": {"predictions_json": f"/mnt/autoalphafold3/runs/trials/{trial_id}/predictions.json"},
    }


def _scored_metrics_payload(trial_id: str = "T130", score: float = 0.09) -> dict[str, object]:
    return {
        "schema_version": "autoaf3.metrics.v1",
        "status": "SCORED",
        "trial_id": trial_id,
        "candidate_id": trial_id,
        "primary_metric": "best_val_calpha_lddt",
        "metrics": {"best_val_calpha_lddt": score},
        "fold_cartographer": {"signature": "candidate_scored", "summary": {}, "buckets": {}},
        "error_report": {"failure_signature": None, "failed_targets": [], "scorer_only": True},
        "official_benchmark_result": True,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "artifacts": {
            "predictions_json": f"/mnt/autoalphafold3/runs/trials/{trial_id}/predictions.json",
            "metrics_json": f"/mnt/autoalphafold3/runs/trials/{trial_id}/metrics.json",
        },
    }


def _write_baseline_lock(tmp_path: Path, *, score: float = 0.08) -> None:
    baseline = tmp_path / "runs/baseline"
    baseline.mkdir(parents=True)
    metrics = {
        "schema_version": "autoaf3.metrics.v1",
        "status": "SCORED",
        "trial_id": "baseline_auto_tiny",
        "candidate_id": "baseline_lock",
        "split": "public_val_small",
        "official_benchmark_result": True,
        "primary_metric": "best_val_calpha_lddt",
        "scorer_version": "calpha_lddt_v1",
        "max_templates": 0,
        "manifests": {"train_tiny": SHA, "public_val_small": SHA},
        "label_hashes": {"public_val_small": SHA},
        "metrics": {"best_val_calpha_lddt": score},
        "fold_cartographer": {"signature": "baseline_locked", "summary": {}, "buckets": {}},
        "artifacts": {"metrics_json": "runs/baseline/metrics.json"},
    }
    (baseline / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (baseline / "error_report.json").write_text(json.dumps({"scorer_only": True}), encoding="utf-8")
    (baseline / "feature_fingerprints.json").write_text(
        json.dumps(
            {
                "files": {
                    "features/train_tiny.arrow": SHA,
                    "features/public_val_small.arrow": SHA,
                },
                "max_templates": 0,
            }
        ),
        encoding="utf-8",
    )


def _llm_candidate_payload(*, changed_paths: list[str] | None = None) -> dict[str, object]:
    trial_id = "T120"
    return {
        "hypothesis": "LLM planner tests one local geometry loss candidate without starting live search.",
        "rationale": "Exercise the strict one-candidate planning seam with patch policy before artifacts.",
        "changed_paths": changed_paths if changed_paths is not None else ["configs/experiments/llm_geometry.json"],
        "trial": {
            "trial_id": trial_id,
            "parent_commit": "abc1234",
            "agent_session_id": "pytest-llm",
            "trial_kind": "training",
            "hypothesis": "LLM planner tests one local geometry loss candidate without starting live search.",
            "move_family": "geometry_loss",
            "diagnostic_target": "local_geometry_weak",
            "prediction": {
                "causal_component": "local_calpha_geometry_loss",
                "predicted_axis": "local_geometry",
                "predicted_direction": "up",
                "expected_lddt_delta_band": [0.001, 0.01],
            },
            "patch_path": None,
            "config_path": "configs/experiments/llm_geometry.json",
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
            "artifact_dir": f"runs/trials/{trial_id}",
            "checkpoint_path": None,
        },
        "config": {"config_path": "configs/experiments/llm_geometry.json", "max_templates": 0},
        "patch_text": (
            "diff --git a/configs/experiments/llm_geometry.json b/configs/experiments/llm_geometry.json\n"
            "--- a/configs/experiments/llm_geometry.json\n"
            "+++ b/configs/experiments/llm_geometry.json\n"
            "@@ -0,0 +1 @@\n"
            "+{\"max_templates\": 0}\n"
        ),
    }


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
                "patch_text": (
                    "diff --git a/configs/experiments/x.json b/configs/experiments/x.json\n"
                    "--- a/configs/experiments/x.json\n"
                    "+++ b/configs/experiments/x.json\n"
                    "@@ -0,0 +1 @@\n"
                    "+{\"max_templates\": 0}\n"
                ),
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


def test_manual_autoresearch_plan_refuses_non_planning_surface_path(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "runs/autoresearch/recorded.json"
    candidate_plan.parent.mkdir(parents=True)
    candidate_plan.write_text("{}\n", encoding="utf-8")

    with pytest.raises(AutoresearchLoopError, match="configs/experiments"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="manual",
            mode="dry-run",
            planner="manual",
            candidate_plan="runs/autoresearch/recorded.json",
        )
    assert not (tmp_path / "runs/autoresearch/manual").exists()


def test_manual_autoresearch_plan_refuses_symlinked_plan_path(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    plan_dir = tmp_path / "configs/experiments"
    plan_dir.mkdir(parents=True)
    (plan_dir / "candidate.json").symlink_to(outside)

    with pytest.raises(AutoresearchLoopError, match="symlink"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="manual",
            mode="dry-run",
            planner="manual",
            candidate_plan="configs/experiments/candidate.json",
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
        "patch_text": (
            "diff --git a/configs/experiments/x.json b/configs/experiments/x.json\n"
            "--- a/configs/experiments/x.json\n"
            "+++ b/configs/experiments/x.json\n"
            "@@ -0,0 +1 @@\n"
            "+{\"max_templates\": 0}\n"
        ),
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
        "patch_text": (
            "diff --git a/configs/experiments/x.json b/configs/experiments/x.json\n"
            "--- a/configs/experiments/x.json\n"
            "+++ b/configs/experiments/x.json\n"
            "@@ -0,0 +1 @@\n"
            "+{\"max_templates\": 0}\n"
        ),
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


def test_autoresearch_loop_modal_scores_after_training_and_records_scorer_fail(tmp_path: Path) -> None:
    client = FakeTrustedAutoresearchClient(_short_training_manifest("T130"))

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="live",
        mode="modal",
        planner="deterministic",
        start_trial_id="T130",
        max_candidates=1,
        approval=APPROVAL_TEXT,
        modal_client=client,
    )

    assert result.status == "PASS"
    assert result.starts_search is True
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    assert result.generated_trials == ["T130"]
    assert len(result.wrote_files) == len(set(result.wrote_files))
    assert client.submitted_trials[0]["trial_id"] == "T130"
    assert client.submitted_trials[0]["trial_kind"] == "training"
    assert client.submitted_trials[0]["runner_mode"] == "short_training"
    assert client.submitted_trials[0]["features_path"] == "nanofold_event_small_no_templates.arrow"
    assert client.submitted_trials[0]["short_training_approval"] == "I_APPROVE_SHORT_TRAINING_TRIAL"
    assert client.scored_trials == ["T130"]
    run_manifest = json.loads((tmp_path / "runs/autoresearch/live/run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["live_modal_execution"] is True
    assert run_manifest["starts_search"] is True
    candidate_dir = tmp_path / "runs/autoresearch/live/candidates/T130"
    training_manifest = json.loads((candidate_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert training_manifest["status"] == "SHORT_TRAINING_READY"
    summary = json.loads((tmp_path / "runs/autoresearch/live/summary.json").read_text(encoding="utf-8"))
    candidate = summary["candidates"][0]
    assert candidate["status"] == "FAIL"
    assert candidate["execution_status"] == "FAIL"
    assert candidate["training_status"] == "SHORT_TRAINING_READY"
    assert candidate["training_manifest_path"] == str(candidate_dir / "training_manifest.json")
    assert candidate["official_benchmark_result"] is False
    assert json.loads((candidate_dir / "decision.json").read_text(encoding="utf-8"))["status"] == "FAIL"
    assert json.loads((candidate_dir / "error_report.json").read_text(encoding="utf-8"))["error_report"]["failure_signature"] == "prediction_artifact_missing"
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/baseline").exists()


def test_autoresearch_loop_modal_writes_artifact_only_decision_for_scored_candidate(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    client = FakeTrustedAutoresearchClient(_short_training_manifest("T130"), _scored_metrics_payload("T130", score=0.09))

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="live-scored",
        mode="modal",
        planner="deterministic",
        start_trial_id="T130",
        max_candidates=1,
        approval=APPROVAL_TEXT,
        modal_client=client,
    )

    assert result.status == "PASS"
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = tmp_path / "runs/autoresearch/live-scored/candidates/T130"
    decision = json.loads((candidate_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "KEEP"
    assert decision["official_benchmark_result"] is False
    assert decision["global_baseline_delta"] == pytest.approx(0.01)
    metrics = json.loads((candidate_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["comparison"]["provisional_keep"] is True
    summary = json.loads((tmp_path / "runs/autoresearch/live-scored/summary.json").read_text(encoding="utf-8"))
    candidate = summary["candidates"][0]
    assert candidate["status"] == "KEEP"
    assert candidate["training_status"] == "SHORT_TRAINING_READY"
    assert candidate["training_manifest_path"] == str(candidate_dir / "training_manifest.json")
    assert candidate["official_benchmark_result"] is False
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_autoresearch_loop_modal_records_terminal_fail_without_ledger(tmp_path: Path) -> None:
    client = FakeTrustedAutoresearchClient(_infra_fail_result("T130"))

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="live-fail",
        mode="modal",
        planner="deterministic",
        start_trial_id="T130",
        max_candidates=1,
        approval=APPROVAL_TEXT,
        modal_client=client,
    )

    assert result.status == "PASS"
    candidate_dir = tmp_path / "runs/autoresearch/live-fail/candidates/T130"
    assert json.loads((candidate_dir / "decision.json").read_text(encoding="utf-8"))["status"] == "INFRA_FAIL"
    assert json.loads((candidate_dir / "error_report.json").read_text(encoding="utf-8"))["failure_signature"] == "pytest_modal_worker_failed"
    summary = json.loads((tmp_path / "runs/autoresearch/live-fail/summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"][0]["status"] == "INFRA_FAIL"
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_autoresearch_loop_requires_live_approval(tmp_path: Path) -> None:
    with pytest.raises(AutoresearchLoopError, match=APPROVAL_TEXT):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="live",
            mode="modal",
            planner="deterministic",
        )


def test_autoresearch_loop_modal_requires_one_candidate_before_artifacts(tmp_path: Path) -> None:
    with pytest.raises(AutoresearchLoopError, match="exactly one"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="live-two",
            mode="modal",
            planner="deterministic",
            max_candidates=2,
            approval=APPROVAL_TEXT,
            modal_client=FakeTrustedAutoresearchClient(_short_training_manifest("T130")),
        )

    assert not (tmp_path / "runs/autoresearch/live-two").exists()


def test_autoresearch_loop_modal_reexecs_through_repo_venv_when_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    venv_python = tmp_path / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    monkeypatch.setattr(agent, "_current_python_can_import_modal", lambda: False)
    monkeypatch.setattr(agent, "_python_can_import_modal", lambda python: python == venv_python)
    monkeypatch.setattr(agent.sys, "executable", "/usr/bin/python3")

    reexec = agent._modal_venv_reexec_argv(
        Namespace(command="autoresearch-loop", mode="modal", repo_root=str(tmp_path)),
        ["autoresearch-loop", "--mode", "modal"],
    )

    assert reexec == [str(venv_python), "-m", "autoalphafold3.agent", "autoresearch-loop", "--mode", "modal"]


def test_deployed_trusted_autoresearch_client_waits_for_worker_result() -> None:
    class FakeSubmit:
        def remote(self, trial: dict[str, object]) -> dict[str, object]:
            assert trial["trial_id"] == "T130"
            return {"status": "SUBMITTED", "artifacts": {"worker_call_id": "fc-123"}}

    class FakeOrchestrator:
        submit_trial = FakeSubmit()

    class FakeCls:
        @staticmethod
        def from_name(app_name: str, class_name: str, environment_name: str | None = None) -> type[FakeOrchestrator]:
            assert app_name == "autoalphafold3-modal"
            assert class_name == "TrustedOrchestrator"
            assert environment_name == "main"
            return FakeOrchestrator

    class FakeCall:
        timeout: int | None = None

        def get(self, *, timeout: int) -> dict[str, object]:
            FakeCall.timeout = timeout
            return _short_training_manifest("T130")

    class FakeFunctionCall:
        @staticmethod
        def from_id(object_id: str) -> FakeCall:
            assert object_id == "fc-123"
            return FakeCall()

    class FakeModal:
        Cls = FakeCls
        FunctionCall = FakeFunctionCall

    client = DeployedTrustedAutoresearchClient.__new__(DeployedTrustedAutoresearchClient)
    client.environment_name = "main"
    client._modal = FakeModal

    payload = client.submit_and_poll_trial({"trial_id": "T130"})

    assert payload["status"] == "SHORT_TRAINING_READY"
    assert FakeCall.timeout == MODAL_WORKER_RESULT_TIMEOUT_S


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


def test_llm_autoresearch_planner_accepts_exactly_one_candidate(tmp_path: Path) -> None:
    fake = FakeLLMPlanner(_llm_candidate_payload())

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="llm-one",
        mode="dry-run",
        planner="llm",
        max_candidates=1,
        planner_client=fake,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T120"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    assert result.llm_policy is not None
    assert result.llm_policy["hypothesis_generation"]["tools"] == [{"type": "web_search", "search_context_size": "medium"}]
    assert result.llm_policy["patch_planning"]["tools"] == []
    assert fake.calls[0]["policy"] == result.llm_policy
    candidate_dir = tmp_path / "runs/autoresearch/llm-one/candidates/T120"
    assert (candidate_dir / "trial.json").exists()
    summary = json.loads((tmp_path / "runs/autoresearch/llm-one/summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"][0]["status"] == "DRAFT"
    assert not (candidate_dir / "decision.json").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_llm_autoresearch_rejects_multi_candidate_shape_before_writing(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/multi.json"
    candidate_plan.parent.mkdir(parents=True)
    candidate_plan.write_text(
        json.dumps({"candidates": [_llm_candidate_payload(), _llm_candidate_payload()]}),
        encoding="utf-8",
    )

    with pytest.raises(AutoresearchLoopError, match="exactly one"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-multi",
            mode="dry-run",
            planner="llm",
            max_candidates=1,
            candidate_plan="configs/experiments/multi.json",
        )

    assert not (tmp_path / "runs/autoresearch/llm-multi").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()


def test_llm_autoresearch_refuses_locked_surface_before_artifacts(tmp_path: Path) -> None:
    fake = FakeLLMPlanner(_llm_candidate_payload(changed_paths=["autoalphafold3/scorer/calpha_lddt.py"]))

    with pytest.raises(AutoresearchLoopError, match="locked during search"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-locked",
            mode="dry-run",
            planner="llm",
            max_candidates=1,
            planner_client=fake,
        )

    assert not (tmp_path / "runs/autoresearch/llm-locked").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/baseline").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_llm_autoresearch_refuses_patch_path_mismatch_and_authority_flags(tmp_path: Path) -> None:
    mismatch = _llm_candidate_payload(changed_paths=["configs/experiments/declared.json"])
    fake = FakeLLMPlanner(mismatch)

    with pytest.raises(AutoresearchLoopError, match="must match"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-mismatch",
            mode="dry-run",
            planner="llm",
            max_candidates=1,
            planner_client=fake,
        )

    blank_patch = _llm_candidate_payload()
    blank_patch["patch_text"] = ""
    fake = FakeLLMPlanner(blank_patch)

    with pytest.raises(AutoresearchLoopError, match="must match"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-blank-patch",
            mode="dry-run",
            planner="llm",
            max_candidates=1,
            planner_client=fake,
        )

    header_only = _llm_candidate_payload()
    header_only["patch_text"] = "diff --git a/configs/experiments/llm_geometry.json b/configs/experiments/llm_geometry.json\n"
    fake = FakeLLMPlanner(header_only)

    with pytest.raises(AutoresearchLoopError, match="hunk content"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-header-only",
            mode="dry-run",
            planner="llm",
            max_candidates=1,
            planner_client=fake,
        )

    malformed_hunk = _llm_candidate_payload()
    malformed_hunk["patch_text"] = (
        "diff --git a/configs/experiments/llm_geometry.json b/configs/experiments/llm_geometry.json\n"
        "--- a/configs/experiments/llm_geometry.json\n"
        "+++ b/configs/experiments/llm_geometry.json\n"
        "+{\"max_templates\": 0}\n"
    )
    fake = FakeLLMPlanner(malformed_hunk)

    with pytest.raises(AutoresearchLoopError, match="hunk header"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-malformed-hunk",
            mode="dry-run",
            planner="llm",
            max_candidates=1,
            planner_client=fake,
        )

    wrong_trial = _llm_candidate_payload()
    wrong_trial["trial"]["trial_id"] = "T999"
    wrong_trial["trial"]["artifact_dir"] = "runs/trials/T999"
    fake = FakeLLMPlanner(wrong_trial)

    with pytest.raises(AutoresearchLoopError, match="start_trial_id"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-wrong-trial",
            mode="dry-run",
            planner="llm",
            max_candidates=1,
            planner_client=fake,
        )

    authority = _llm_candidate_payload()
    authority["patch_text"] = (
        "diff --git a/configs/experiments/llm_geometry.json b/configs/experiments/llm_geometry.json\n"
        "--- a/configs/experiments/llm_geometry.json\n"
        "+++ b/configs/experiments/llm_geometry.json\n"
        "@@ -0,0 +3 @@\n"
        "+{\"official_benchmark_result\": false,\n"
        "+ \"writes_ledger\"\n"
        "+ : true, \"max_templates\": 1}\n"
    )
    fake = FakeLLMPlanner(authority)

    with pytest.raises(AutoresearchLoopError, match="max_templates|official_benchmark_result|writes_ledger"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-authority-patch",
            mode="dry-run",
            planner="llm",
            max_candidates=1,
            planner_client=fake,
        )
    assert not (tmp_path / "runs/autoresearch/llm-mismatch").exists()
    assert not (tmp_path / "runs/autoresearch/llm-blank-patch").exists()
    assert not (tmp_path / "runs/autoresearch/llm-header-only").exists()
    assert not (tmp_path / "runs/autoresearch/llm-malformed-hunk").exists()
    assert not (tmp_path / "runs/autoresearch/llm-wrong-trial").exists()
    assert not (tmp_path / "runs/autoresearch/llm-authority-patch").exists()


def test_llm_autoresearch_cli_requires_recorded_plan(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "autoresearch-loop",
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "cli-llm-missing",
            "--mode",
            "dry-run",
            "--planner",
            "llm",
            "--max-candidates",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert "requires --candidate-plan" in payload["error"]
    assert not (tmp_path / "runs/autoresearch/cli-llm-missing").exists()


def test_llm_autoresearch_cli_consumes_recorded_plan(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/recorded-llm.json"
    candidate_plan.parent.mkdir(parents=True)
    candidate_plan.write_text(json.dumps(_llm_candidate_payload()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "autoresearch-loop",
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "cli-llm-recorded",
            "--mode",
            "dry-run",
            "--planner",
            "llm",
            "--candidate-plan",
            "configs/experiments/recorded-llm.json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "PLANNED"
    assert payload["planner"] == "llm"
    assert payload["generated_trials"] == ["T120"]
    assert payload["llm_policy"]["patch_planning"]["tools"] == []
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_llm_autoresearch_cli_rejects_explicit_multi_candidate_request(tmp_path: Path) -> None:
    candidate_plan = tmp_path / "configs/experiments/recorded-llm.json"
    candidate_plan.parent.mkdir(parents=True)
    candidate_plan.write_text(json.dumps(_llm_candidate_payload()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "autoresearch-loop",
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "cli-llm-two",
            "--mode",
            "dry-run",
            "--planner",
            "llm",
            "--max-candidates",
            "2",
            "--candidate-plan",
            "configs/experiments/recorded-llm.json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert "exactly one" in payload["error"]
