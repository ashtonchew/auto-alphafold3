from __future__ import annotations

import json
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from autoalphafold3 import agent
from autoalphafold3.autoresearch_loop import (
    APPROVAL_TEXT,
    AutoresearchCandidatePlan,
    DeployedTrustedAutoresearchClient,
    MODAL_WORKER_RESULT_TIMEOUT_S,
    OpenAIAutoresearchPlanner,
    AutoresearchLoopError,
    run_autoresearch_loop,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeLLMPlanner:
    def __init__(self, payload: dict[str, object] | list[dict[str, object]]) -> None:
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.calls: list[dict[str, object]] = []

    def plan(self, **kwargs: object) -> AutoresearchCandidatePlan:
        self.calls.append(kwargs)
        index = int(kwargs.get("candidate_index", 0))
        return AutoresearchCandidatePlan.model_validate(self.payloads[index])


class FakeTrustedAutoresearchClient:
    def __init__(self, payload: dict[str, object], score_payload: dict[str, object] | None = None) -> None:
        self.payload = payload
        self.score_payload = score_payload
        self.authority_payload: dict[str, object] = {
            "runtime_capabilities": {
                "post_training_sampler_coordinate_normalization": True,
                "post_training_sampler_coordinate_scale": True,
                "post_training_sampler_selection": True,
                "post_training_sampler_schedule": True,
            }
        }
        self.submitted_trials: list[dict[str, object]] = []
        self.scored_trials: list[str] = []

    def authority_health(self) -> dict[str, object]:
        return self.authority_payload

    def submit_and_poll_trial(self, trial: dict[str, object]) -> dict[str, object]:
        self.submitted_trials.append(trial)
        return self.payload

    def score_trial(self, trial_id: str) -> dict[str, object]:
        self.scored_trials.append(trial_id)
        return self.score_payload or _scorer_fail_payload(trial_id)


class FakeSequencedTrustedAutoresearchClient:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.authority_payload: dict[str, object] = {
            "runtime_capabilities": {
                "post_training_sampler_coordinate_normalization": True,
                "post_training_sampler_coordinate_scale": True,
                "post_training_sampler_selection": True,
                "post_training_sampler_schedule": True,
            }
        }
        self.submitted_trials: list[dict[str, object]] = []
        self.scored_trials: list[str] = []

    def authority_health(self) -> dict[str, object]:
        return self.authority_payload

    def submit_and_poll_trial(self, trial: dict[str, object]) -> dict[str, object]:
        self.submitted_trials.append(trial)
        if trial.get("trial_kind") == "sampler":
            return _sampler_manifest(str(trial["trial_id"]), checkpoint_path=str(trial["checkpoint_path"]))
        return _short_training_manifest(str(trial["trial_id"]))

    def score_trial(self, trial_id: str) -> dict[str, object]:
        self.scored_trials.append(trial_id)
        return _scored_metrics_payload(trial_id, score=self.scores[trial_id])


class FakeSequencedFailingTrustedAutoresearchClient:
    def __init__(self) -> None:
        self.authority_payload: dict[str, object] = {
            "runtime_capabilities": {
                "post_training_sampler_coordinate_normalization": True,
                "post_training_sampler_coordinate_scale": True,
                "post_training_sampler_selection": True,
                "post_training_sampler_schedule": True,
            }
        }
        self.submitted_trials: list[dict[str, object]] = []
        self.scored_trials: list[str] = []

    def authority_health(self) -> dict[str, object]:
        return self.authority_payload

    def submit_and_poll_trial(self, trial: dict[str, object]) -> dict[str, object]:
        self.submitted_trials.append(trial)
        return _short_training_manifest(str(trial["trial_id"]))

    def score_trial(self, trial_id: str) -> dict[str, object]:
        self.scored_trials.append(trial_id)
        return _scorer_fail_payload(trial_id)


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


def _sampler_manifest(trial_id: str = "T135", *, checkpoint_path: str = "/mnt/autoalphafold3/runs/trials/T133/checkpoint.pt") -> dict[str, object]:
    return {
        "schema_version": "autoaf3.sampler_manifest.v1",
        "status": "SAMPLER_PREDICTED",
        "trial_id": trial_id,
        "candidate_id": trial_id,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": SHA,
        "checkpoint_source_trial_id": "T133",
        "feature_file": "/mnt/autoalphafold3/features/nanofold_event_small_no_templates.arrow",
        "target_ids": ["2LZM_A"],
        "prediction_count": 1,
        "real_training_performed": False,
        "inference_only": True,
        "max_templates": 0,
        "sampler_steps": 2,
        "sampler_noise_scale": 1.0,
        "sampler_step_scale": 1.0,
        "sampler_schedule_shape": "linear",
        "sampler_num_samples": 1,
        "sampler_selection_policy": "first",
        "sampler_coordinate_normalization": "ca_bond",
        "sampler_coordinate_scale": 13.126702,
        "starts_search": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "runtime_s": 1.0,
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


def _llm_candidate_payload(
    *,
    trial_id: str = "T120",
    config_path: str = "configs/experiments/llm_geometry.json",
    changed_paths: list[str] | None = None,
    config_payload: dict[str, object] | None = None,
    budget: str = "smoke",
    max_steps: int = 10,
    max_wall_minutes: int = 5,
    timeout_cap: int = 300,
) -> dict[str, object]:
    inline_config = config_payload or json.loads((REPO_ROOT / "configs/nanofold_dev_cpu_smoke.json").read_text(encoding="utf-8"))
    inline_config.setdefault("diffusion_loss_weight", 4.0)
    inline_config.setdefault("dist_loss_weight", 0.0)
    inline_config.setdefault("distogram_loss_weight", 0.03)
    inline_config.setdefault("local_calpha_geometry_loss_weight", 0.0)
    return {
        "hypothesis": "LLM planner tests one local geometry loss candidate without starting live search.",
        "rationale": "Exercise the strict one-candidate planning seam with patch policy before artifacts.",
        "changed_paths": changed_paths if changed_paths is not None else [config_path],
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
            "config_path": config_path,
            "budget": budget,
            "seed": 0,
            "n_res": 32,
            "max_steps": max_steps,
            "max_wall_minutes": max_wall_minutes,
            "manifest_hashes": {},
            "scorer_version": "calpha_lddt_v1",
            "primary_metric": "best_val_calpha_lddt",
            "param_cap": 176514,
            "gpu_memory_cap": 80.0,
            "cost_cap": 2.0,
            "timeout_cap": timeout_cap,
            "artifact_dir": f"runs/trials/{trial_id}",
            "checkpoint_path": None,
            "config_payload": inline_config,
        },
        "config": {
            "config_path": config_path,
            "max_templates": 0,
            "learning_rate": inline_config["learning_rate"],
            "local_calpha_geometry_loss_weight": inline_config["local_calpha_geometry_loss_weight"],
        },
        "patch_text": (
            f"diff --git a/{config_path} b/{config_path}\n"
            f"--- a/{config_path}\n"
            f"+++ b/{config_path}\n"
            "@@ -0,0 +1 @@\n"
            "+{\"max_templates\": 0}\n"
        ),
    }


def _write_targeted_diagnostic_inputs(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True)
    config_dir.joinpath("local_calpha_geometry_smoke.json").write_text(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report = tmp_path / "runs/autoresearch/scorer_sensitivity/T088-vs-T108-T111.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.scorer_sensitivity.v1",
                "status": "PASS",
                "reference_trial_id": "T088",
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "per_target_score_deltas_vs_reference": {
                    "T108": {
                        "2LZM_A": -0.013,
                        "2BP2_A": -0.011,
                        "1PFC_A": 0.012,
                    },
                    "T109": {
                        "2BP2_A": -0.012,
                        "1MBD_A": -0.014,
                        "1NXB_A": 0.012,
                    },
                    "T110": {
                        "2BP2_A": -0.013,
                        "1MBD_A": -0.012,
                        "1BP2_A": -0.011,
                    },
                    "T111": {
                        "2LZM_A": -0.012,
                        "3CHY_A": -0.011,
                        "2BP2_A": -0.011,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return report.relative_to(tmp_path)


def _write_schedule_diagnostic_inputs(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True)
    config_dir.joinpath("local_calpha_geometry_smoke.json").write_text(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report = tmp_path / "runs/autoresearch/scorer_sensitivity/T088-vs-T160.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.scorer_sensitivity.v1",
                "status": "PASS",
                "reference_trial_id": "T088",
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "per_target_score_deltas_vs_reference": {
                    "T160": {
                        "2BP2_A": -0.0188,
                        "1MBD_A": -0.0213,
                        "2LZM_A": -0.0156,
                        "1BP2_A": -0.0147,
                        "1NXB_A": -0.0048,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return report.relative_to(tmp_path)


def _write_capacity_diagnostic_inputs(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True)
    config_dir.joinpath("local_calpha_geometry_smoke.json").write_text(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report = tmp_path / "runs/autoresearch/scorer_sensitivity/T088-vs-T113.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.scorer_sensitivity.v1",
                "status": "PASS",
                "reference_trial_id": "T088",
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "per_target_score_deltas_vs_reference": {
                    "T113": {
                        "1MBD_A": -0.0213,
                        "2BP2_A": -0.0188,
                        "2LZM_A": -0.0156,
                        "1BP2_A": -0.0147,
                        "3CHY_A": -0.0145,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return report.relative_to(tmp_path)


def _write_topology_recycling_diagnostic_inputs(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True)
    config_dir.joinpath("local_calpha_geometry_smoke.json").write_text(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report = tmp_path / "runs/autoresearch/scorer_sensitivity/T088-vs-T162.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.scorer_sensitivity.v1",
                "status": "PASS",
                "reference_trial_id": "T088",
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "per_target_score_deltas_vs_reference": {
                    "T162": {
                        "1MBD_A": -0.0213,
                        "2BP2_A": -0.0188,
                        "2LZM_A": -0.0156,
                        "1BP2_A": -0.0147,
                        "3CHY_A": -0.0145,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return report.relative_to(tmp_path)


def _write_feature_curriculum_diagnostic_inputs(
    tmp_path: Path,
    *,
    verdict: str = "SHORT_TRAINING_FAMILY_SCORER_COLLAPSE",
    report_name: str = "T113-T162-T163.json",
) -> Path:
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dir.joinpath("local_calpha_geometry_smoke.json").write_text(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report = tmp_path / "runs/autoresearch/post_discard_diagnosis" / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.post_discard_diagnosis.v1",
                "status": "PASS",
                "verdict": verdict,
                "reference_trial_id": "T088",
                "candidate_trial_ids": ["T113", "T162", "T163"],
                "exhausted_surfaces": [
                    "sampler",
                    "local_geometry",
                    "optimizer_schedule",
                    "width_depth",
                    "recycling",
                ],
                "score_summary": {
                    "primary_metric": "best_val_calpha_lddt",
                    "candidate_scores": {
                        "T113": 0.008276756926787072,
                        "T162": 0.008276756926787072,
                        "T163": 0.008276756926787072,
                    },
                    "candidate_scores_identical": True,
                    "all_candidate_per_target_deltas_negative": True,
                    "per_target_delta_summary": {
                        "candidate_delta_sets": 3,
                        "target_count": 16,
                        "negative_delta_count": 48,
                        "positive_delta_count": 0,
                        "zero_delta_count": 0,
                        "worst_target": "1MBD_A",
                        "worst_delta": -0.0213496566139146,
                    },
                },
                "artifact_summary": {
                    "comparison_count": 3,
                    "all_comparisons_changed": True,
                    "any_all_predictions_identical": False,
                },
                "recommendation": {
                    "stop_live_trial_budget": True,
                    "do_not_start_open_ended_loop": True,
                },
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return report.relative_to(tmp_path)


def _write_coordinate_scale_locality_review_inputs(
    tmp_path: Path,
    *,
    decision: str = "APPROVE_OFFLINE_PLANNER_PR_ONLY",
    approved_surface: str | None = "coordinate_scale_locality_diagnostic",
    report_name: str = "T164-mixed-evidence.json",
    source_diagnosis: str = "runs/autoresearch/post_discard_diagnosis/T113-T162-T163-T164.json",
    rejected_surfaces: list[str] | None = None,
    candidate_trial_ids: list[str] | None = None,
    negative_delta_count: int = 64,
    positive_delta_count: int = 0,
    worst_delta: float = -0.0213496566139146,
) -> Path:
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dir.joinpath("local_calpha_geometry_smoke.json").write_text(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    planner = approved_surface or "coordinate_scale_locality_diagnostic"
    report = tmp_path / "runs/autoresearch/next_surface_review" / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.next_surface_review.v1",
                "status": "PASS",
                "source_diagnosis": source_diagnosis,
                "source_verdict": "MIXED_EVIDENCE_REVIEW_REQUIRED",
                "decision": decision,
                "approved_next_surface": approved_surface,
                "rejected_surfaces": rejected_surfaces or [
                    "sampler",
                    "local_geometry",
                    "optimizer_schedule",
                    "width_depth",
                    "recycling",
                    "feature_curriculum",
                ],
                "evidence_summary": {
                    "reference_trial_id": "T088",
                    "candidate_trial_ids": candidate_trial_ids or ["T113", "T162", "T163", "T164"],
                    "all_candidate_per_target_deltas_negative": positive_delta_count == 0,
                    "negative_delta_count": negative_delta_count,
                    "positive_delta_count": positive_delta_count,
                    "worst_target": "1MBD_A",
                    "worst_delta": worst_delta,
                    "all_comparisons_changed": True,
                    "comparison_count": 4,
                },
                "required_next_pr": {
                    "planner": planner,
                    "candidate_limit": 1,
                    "mode_before_merge": "dry-run",
                    "candidate_budget": "trial",
                    "must_consume_review": True,
                },
                "allowed_next_step": "Implement one dry-run-only planner.",
                "stop_live_trial_budget": True,
                "do_not_start_open_ended_loop": True,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    return report.relative_to(tmp_path)


def _write_prediction_geometry_audit_inputs(
    tmp_path: Path,
    *,
    flags: list[str] | None = None,
    starts_search: bool = False,
    report_name: str = "T164-T165-vs-T088.json",
) -> Path:
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dir.joinpath("local_calpha_geometry_smoke.json").write_text(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report = tmp_path / "runs/autoresearch/prediction_geometry" / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    scale_flags = flags or [
        "adjacent_ca_distance_exploded",
        "adjacent_ca_distance_outlier_gt_30A",
        "pair_distance_exploded",
        "pair_distance_outlier_gt_500A",
        "reference_pair_distance_shift_gt_20A",
        "reference_radius_scale_shift",
    ]
    report.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.prediction_geometry_audit.v1",
                "official_benchmark_result": False,
                "starts_search": starts_search,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "reference": {"trial_id": "T088", "split": "public_val_small"},
                "artifacts": [
                    {"trial_id": "T164", "split": "public_val_small", "scale_flags": scale_flags},
                    {"trial_id": "T165", "split": "public_val_small", "scale_flags": scale_flags},
                ],
                "reference_deltas": [
                    {
                        "reference_trial_id": "T088",
                        "candidate_trial_id": "T164",
                        "mean_radius_scale_ratio": 55.315869408410755,
                        "mean_pair_distance_delta": 2619.6425616268834,
                        "candidate_flags": scale_flags,
                    },
                    {
                        "reference_trial_id": "T088",
                        "candidate_trial_id": "T165",
                        "mean_radius_scale_ratio": 55.50376670178062,
                        "mean_pair_distance_delta": 2637.861416018026,
                        "candidate_flags": scale_flags,
                    },
                ],
                "recommendation": {
                    "status": "REVIEW_REQUIRED",
                    "flags": scale_flags,
                    "next_goal": "Review prediction geometry scale before another live candidate.",
                    "stop_live_trial_budget": True,
                    "do_not_start_open_ended_loop": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return report.relative_to(tmp_path)


def _write_collapsed_prediction_geometry_audit_inputs(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dir.joinpath("local_calpha_geometry_smoke.json").write_text(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    flags = [
        "adjacent_ca_distance_collapsed",
        "reference_pair_distance_shift_gt_20A",
        "reference_radius_scale_shift",
    ]
    report = tmp_path / "runs/autoresearch/prediction_geometry/T167-vs-T088.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.prediction_geometry_audit.v1",
                "official_benchmark_result": False,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "reference": {"trial_id": "T088", "split": "public_val_small"},
                "artifacts": [{"trial_id": "T167", "split": "public_val_small", "scale_flags": flags}],
                "reference_deltas": [
                    {
                        "reference_trial_id": "T088",
                        "candidate_trial_id": "T167",
                        "mean_radius_scale_ratio": 0.07618059739178347,
                        "mean_pair_distance_delta": -45.5238705754392,
                        "candidate_flags": flags,
                    }
                ],
                "recommendation": {
                    "status": "REVIEW_REQUIRED",
                    "flags": flags,
                    "next_goal": "Review collapsed coordinate scale before another live candidate.",
                    "stop_live_trial_budget": True,
                    "do_not_start_open_ended_loop": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return report.relative_to(tmp_path)


def _write_controlled_scale_prediction_geometry_audit_inputs(
    tmp_path: Path,
    *,
    source_trial_id: str = "T168",
    report_name: str = "T168-vs-T088.json",
    mean_radius_scale_ratio: float = 0.995071710549596,
    mean_pair_distance_delta: float = -0.9456763615706882,
) -> Path:
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dir.joinpath("local_calpha_geometry_smoke.json").write_text(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    flags = ["adjacent_ca_distance_outlier_gt_30A"]
    report = tmp_path / "runs/autoresearch/prediction_geometry" / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.prediction_geometry_audit.v1",
                "official_benchmark_result": False,
                "starts_search": False,
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "reference": {"trial_id": "T088", "split": "public_val_small"},
                "artifacts": [{"trial_id": source_trial_id, "split": "public_val_small", "scale_flags": flags}],
                "reference_deltas": [
                    {
                        "reference_trial_id": "T088",
                        "candidate_trial_id": source_trial_id,
                        "mean_radius_scale_ratio": mean_radius_scale_ratio,
                        "mean_pair_distance_delta": mean_pair_distance_delta,
                        "candidate_flags": flags,
                    }
                ],
                "recommendation": {
                    "status": "REVIEW_REQUIRED",
                    "flags": flags,
                    "next_goal": "Review sampler locality before another live candidate.",
                    "stop_live_trial_budget": True,
                    "do_not_start_open_ended_loop": True,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_dir = tmp_path / "runs/autoresearch/modal_artifacts" / source_trial_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.joinpath("sampler_manifest.json").write_text(
        json.dumps(
            _sampler_manifest(
                source_trial_id,
                checkpoint_path=f"/mnt/autoalphafold3/runs/trials/{source_trial_id}/checkpoint.pt",
            )
        ),
        encoding="utf-8",
    )
    return report.relative_to(tmp_path)


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


def test_targeted_diagnostic_autoresearch_plans_one_report_driven_candidate(tmp_path: Path) -> None:
    report = _write_targeted_diagnostic_inputs(tmp_path)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="targeted-diagnostic",
        mode="dry-run",
        planner="targeted_diagnostic",
        start_trial_id="T160",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T160"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T160"
    assert str(candidate_dir / "config.json") in result.wrote_files
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["max_wall_minutes"] == 45
    assert trial["timeout_cap"] == 2700
    assert trial["move_family"] == "geometry_loss"
    assert trial["diagnostic_target"] == "local_geometry_weak"
    assert trial["prediction"]["causal_component"] == "local_calpha_geometry_loss_weight"
    assert trial["config_path"] == "configs/experiments/T160_targeted_geometry_diagnostic.json"
    assert trial["config_payload"]["max_templates"] == 0
    assert trial["config_payload"]["learning_rate"] == pytest.approx(0.0012)
    assert trial["config_payload"]["local_calpha_geometry_loss_weight"] == pytest.approx(0.4)
    assert trial["config_payload"]["distogram_loss_weight"] == pytest.approx(0.05)
    assert config["reference_trial_id"] == "T088"
    assert config["worst_targets"][:2] == ["2BP2_A", "1MBD_A"]
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T160_targeted_geometry_diagnostic.json" in patch_text
    assert "2BP2_A" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_targeted_diagnostic_autoresearch_requires_single_candidate_and_report(tmp_path: Path) -> None:
    report = _write_targeted_diagnostic_inputs(tmp_path)

    with pytest.raises(AutoresearchLoopError, match="max_candidates=1"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="targeted-too-many",
            mode="dry-run",
            planner="targeted_diagnostic",
            start_trial_id="T160",
            max_candidates=2,
            diagnostic_report=report,
        )

    with pytest.raises(AutoresearchLoopError, match="requires --diagnostic-report"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="targeted-missing-report",
            mode="dry-run",
            planner="targeted_diagnostic",
            start_trial_id="T160",
            max_candidates=1,
        )

    assert not (tmp_path / "runs/autoresearch/targeted-too-many").exists()
    assert not (tmp_path / "runs/autoresearch/targeted-missing-report").exists()


def test_schedule_diagnostic_autoresearch_plans_one_optimizer_candidate(tmp_path: Path) -> None:
    report = _write_schedule_diagnostic_inputs(tmp_path)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="schedule-diagnostic",
        mode="dry-run",
        planner="schedule_diagnostic",
        start_trial_id="T161",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T161"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T161"
    assert str(candidate_dir / "config.json") in result.wrote_files
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["max_wall_minutes"] == 45
    assert trial["timeout_cap"] == 2700
    assert trial["move_family"] == "optimizer_scheduler"
    assert trial["diagnostic_target"] == "local_geometry_weak"
    assert trial["prediction"]["causal_component"] == "optimizer_schedule_stability"
    assert trial["config_path"] == "configs/experiments/T161_schedule_diagnostic.json"
    assert trial["config_payload"]["max_templates"] == 0
    assert trial["config_payload"]["learning_rate"] == pytest.approx(0.0008)
    assert trial["config_payload"]["lr_start_factor"] == pytest.approx(0.01)
    assert trial["config_payload"]["lr_warmup"] == 250
    assert trial["config_payload"]["clip_norm"] == pytest.approx(3.0)
    assert trial["config_payload"]["local_calpha_geometry_loss_weight"] == pytest.approx(0.1)
    assert config["schema_version"] == "autoaf3.schedule_diagnostic_plan.v1"
    assert config["failed_shape_avoided"] == "T160 stronger local-geometry pressure"
    assert config["worst_targets"][:2] == ["1MBD_A", "2BP2_A"]
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T161_schedule_diagnostic.json" in patch_text
    assert "bounded optimizer/schedule training diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_schedule_diagnostic_autoresearch_requires_single_candidate_and_report(tmp_path: Path) -> None:
    report = _write_schedule_diagnostic_inputs(tmp_path)

    with pytest.raises(AutoresearchLoopError, match="max_candidates=1"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="schedule-too-many",
            mode="dry-run",
            planner="schedule_diagnostic",
            start_trial_id="T161",
            max_candidates=2,
            diagnostic_report=report,
        )

    with pytest.raises(AutoresearchLoopError, match="requires --diagnostic-report"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="schedule-missing-report",
            mode="dry-run",
            planner="schedule_diagnostic",
            start_trial_id="T161",
            max_candidates=1,
        )

    assert not (tmp_path / "runs/autoresearch/schedule-too-many").exists()
    assert not (tmp_path / "runs/autoresearch/schedule-missing-report").exists()


def test_capacity_diagnostic_autoresearch_plans_one_capacity_candidate(tmp_path: Path) -> None:
    report = _write_capacity_diagnostic_inputs(tmp_path)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="capacity-diagnostic",
        mode="dry-run",
        planner="capacity_diagnostic",
        start_trial_id="T162",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T162"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T162"
    assert str(candidate_dir / "config.json") in result.wrote_files
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["max_wall_minutes"] == 45
    assert trial["timeout_cap"] == 2700
    assert trial["move_family"] == "width_depth"
    assert trial["diagnostic_target"] == "local_geometry_weak"
    assert trial["prediction"]["causal_component"] == "bounded_width_depth_capacity"
    assert trial["config_path"] == "configs/experiments/T162_capacity_diagnostic.json"
    assert trial["config_payload"]["max_templates"] == 0
    assert trial["config_payload"]["single_embedding_size"] == 16
    assert trial["config_payload"]["pair_embedding_size"] == 4
    assert trial["config_payload"]["msa_embedding_size"] == 12
    assert trial["config_payload"]["token_embedding_size"] == 21
    assert trial["config_payload"]["atom_embedding_size"] == 12
    assert trial["config_payload"]["num_pairformer_blocks"] == 4
    assert trial["config_payload"]["num_diffusion_transformer_blocks"] == 4
    assert trial["config_payload"]["learning_rate"] == pytest.approx(0.0009)
    assert trial["config_payload"]["local_calpha_geometry_loss_weight"] == pytest.approx(0.05)
    assert config["schema_version"] == "autoaf3.capacity_diagnostic_plan.v1"
    assert config["failed_shapes_avoided"] == [
        "T160 stronger local-geometry pressure",
        "T161 optimizer/schedule backoff",
        "T113 sampler-only pivot",
    ]
    assert config["worst_targets"][:2] == ["1MBD_A", "2BP2_A"]
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T162_capacity_diagnostic.json" in patch_text
    assert "bounded model-capacity training diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_capacity_diagnostic_autoresearch_requires_single_candidate_and_report(tmp_path: Path) -> None:
    report = _write_capacity_diagnostic_inputs(tmp_path)

    with pytest.raises(AutoresearchLoopError, match="max_candidates=1"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="capacity-too-many",
            mode="dry-run",
            planner="capacity_diagnostic",
            start_trial_id="T162",
            max_candidates=2,
            diagnostic_report=report,
        )

    with pytest.raises(AutoresearchLoopError, match="requires --diagnostic-report"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="capacity-missing-report",
            mode="dry-run",
            planner="capacity_diagnostic",
            start_trial_id="T162",
            max_candidates=1,
        )

    assert not (tmp_path / "runs/autoresearch/capacity-too-many").exists()
    assert not (tmp_path / "runs/autoresearch/capacity-missing-report").exists()


def test_topology_recycling_diagnostic_plans_one_recycling_candidate(tmp_path: Path) -> None:
    report = _write_topology_recycling_diagnostic_inputs(tmp_path)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="topology-recycling-diagnostic",
        mode="dry-run",
        planner="topology_recycling_diagnostic",
        start_trial_id="T163",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T163"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T163"
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["max_wall_minutes"] == 45
    assert trial["timeout_cap"] == 2700
    assert trial["move_family"] == "recycling"
    assert trial["diagnostic_target"] == "long_range_topology_weak"
    assert trial["prediction"]["causal_component"] == "extra_trunk_recycle"
    assert trial["prediction"]["predicted_axis"] == "long_range_topology"
    assert trial["config_path"] == "configs/experiments/T163_topology_recycling_diagnostic.json"
    assert trial["config_payload"]["max_templates"] == 0
    assert trial["config_payload"]["num_recycle"] == 2
    assert trial["config_payload"]["learning_rate"] == pytest.approx(0.0007)
    assert trial["config_payload"]["distogram_loss_weight"] == pytest.approx(0.06)
    assert trial["config_payload"]["local_calpha_geometry_loss_weight"] == pytest.approx(0.0)
    assert config["schema_version"] == "autoaf3.topology_recycling_diagnostic_plan.v1"
    assert config["failed_shapes_avoided"] == [
        "T160 stronger local-geometry pressure",
        "T161 optimizer/schedule backoff",
        "T113 sampler-only pivot",
        "T162 width/depth capacity",
    ]
    assert config["worst_targets"][:2] == ["1MBD_A", "2BP2_A"]
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T163_topology_recycling_diagnostic.json" in patch_text
    assert "bounded topology/recycling training diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_topology_recycling_diagnostic_requires_single_candidate_and_report(tmp_path: Path) -> None:
    report = _write_topology_recycling_diagnostic_inputs(tmp_path)

    with pytest.raises(AutoresearchLoopError, match="max_candidates=1"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="topology-recycling-too-many",
            mode="dry-run",
            planner="topology_recycling_diagnostic",
            start_trial_id="T163",
            max_candidates=2,
            diagnostic_report=report,
        )

    with pytest.raises(AutoresearchLoopError, match="requires --diagnostic-report"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="topology-recycling-missing-report",
            mode="dry-run",
            planner="topology_recycling_diagnostic",
            start_trial_id="T163",
            max_candidates=1,
        )

    assert not (tmp_path / "runs/autoresearch/topology-recycling-too-many").exists()
    assert not (tmp_path / "runs/autoresearch/topology-recycling-missing-report").exists()


def test_feature_curriculum_diagnostic_plans_one_curriculum_candidate(tmp_path: Path) -> None:
    report = _write_feature_curriculum_diagnostic_inputs(tmp_path)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="feature-curriculum-diagnostic",
        mode="dry-run",
        planner="feature_curriculum_diagnostic",
        start_trial_id="T164",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T164"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T164"
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["max_wall_minutes"] == 45
    assert trial["timeout_cap"] == 2700
    assert trial["move_family"] == "curriculum"
    assert trial["diagnostic_target"] == "stability_compute"
    assert trial["prediction"]["causal_component"] == "reduced_crop_msa_curriculum"
    assert trial["prediction"]["predicted_axis"] == "stability_compute"
    assert trial["config_path"] == "configs/experiments/T164_feature_curriculum_diagnostic.json"
    assert trial["config_payload"]["max_templates"] == 0
    assert trial["config_payload"]["residue_crop_size"] == 16
    assert trial["config_payload"]["num_msa_samples"] == 2
    assert trial["config_payload"]["learning_rate"] == pytest.approx(0.0007)
    assert trial["config_payload"]["local_calpha_geometry_loss_weight"] == pytest.approx(0.0)
    assert config["schema_version"] == "autoaf3.feature_curriculum_diagnostic_plan.v1"
    assert config["post_discard_verdict"] == "SHORT_TRAINING_FAMILY_SCORER_COLLAPSE"
    assert config["exhausted_surfaces"] == [
        "sampler",
        "local_geometry",
        "optimizer_schedule",
        "width_depth",
        "recycling",
    ]
    assert config["failed_shapes_avoided"] == [
        "T160 stronger local-geometry pressure",
        "T161 optimizer/schedule backoff",
        "T113 sampler-only pivot",
        "T162 width/depth capacity",
        "T163 topology/recycling",
    ]
    assert config["worst_targets"] == ["1MBD_A"]
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T164_feature_curriculum_diagnostic.json" in patch_text
    assert "bounded feature/curriculum short-training diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_feature_curriculum_diagnostic_requires_single_candidate_and_collapse_report(tmp_path: Path) -> None:
    report = _write_feature_curriculum_diagnostic_inputs(tmp_path)
    mixed_report = _write_feature_curriculum_diagnostic_inputs(
        tmp_path,
        verdict="MIXED_EVIDENCE_REVIEW_REQUIRED",
        report_name="mixed-evidence.json",
    )

    with pytest.raises(AutoresearchLoopError, match="max_candidates=1"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="feature-curriculum-too-many",
            mode="dry-run",
            planner="feature_curriculum_diagnostic",
            start_trial_id="T164",
            max_candidates=2,
            diagnostic_report=report,
        )

    with pytest.raises(AutoresearchLoopError, match="requires --diagnostic-report"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="feature-curriculum-missing-report",
            mode="dry-run",
            planner="feature_curriculum_diagnostic",
            start_trial_id="T164",
            max_candidates=1,
        )

    with pytest.raises(AutoresearchLoopError, match="SHORT_TRAINING_FAMILY_SCORER_COLLAPSE"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="feature-curriculum-wrong-verdict",
            mode="dry-run",
            planner="feature_curriculum_diagnostic",
            start_trial_id="T164",
            max_candidates=1,
            diagnostic_report=mixed_report,
        )

    assert not (tmp_path / "runs/autoresearch/feature-curriculum-too-many").exists()
    assert not (tmp_path / "runs/autoresearch/feature-curriculum-missing-report").exists()
    assert not (tmp_path / "runs/autoresearch/feature-curriculum-wrong-verdict").exists()


def test_coordinate_scale_locality_diagnostic_plans_one_diffusion_candidate(tmp_path: Path) -> None:
    report = _write_coordinate_scale_locality_review_inputs(tmp_path)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="coordinate-scale-locality-diagnostic",
        mode="dry-run",
        planner="coordinate_scale_locality_diagnostic",
        start_trial_id="T165",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T165"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T165"
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["max_wall_minutes"] == 45
    assert trial["timeout_cap"] == 2700
    assert trial["move_family"] == "diffusion_schedule"
    assert trial["diagnostic_target"] == "distogram_good_lddt_flat"
    assert trial["prediction"]["causal_component"] == "diffusion_loss_scale_distogram_locality"
    assert trial["prediction"]["predicted_axis"] == "distogram_vs_3d"
    assert trial["config_path"] == "configs/experiments/T165_coordinate_scale_locality_diagnostic.json"
    assert trial["config_payload"]["max_templates"] == 0
    assert trial["config_payload"]["residue_crop_size"] == 16
    assert trial["config_payload"]["num_msa_samples"] == 2
    assert trial["config_payload"]["diffusion_loss_weight"] == pytest.approx(1.0)
    assert trial["config_payload"]["distogram_loss_weight"] == pytest.approx(0.08)
    assert trial["config_payload"]["local_calpha_geometry_loss_weight"] == pytest.approx(0.0)
    assert trial["config_payload"]["diffusion_steps"] == 20
    assert config["schema_version"] == "autoaf3.coordinate_scale_locality_diagnostic_plan.v1"
    assert config["source_verdict"] == "MIXED_EVIDENCE_REVIEW_REQUIRED"
    assert config["reference_trial_id"] == "T088"
    assert config["candidate_trial_ids"] == ["T113", "T162", "T163", "T164"]
    assert config["rejected_surfaces"] == [
        "sampler",
        "local_geometry",
        "optimizer_schedule",
        "width_depth",
        "recycling",
        "feature_curriculum",
    ]
    assert config["worst_targets"] == ["1MBD_A"]
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T165_coordinate_scale_locality_diagnostic.json" in patch_text
    assert "bounded coordinate-scale/locality diffusion diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_coordinate_scale_locality_diagnostic_requires_single_candidate_and_approved_review(tmp_path: Path) -> None:
    report = _write_coordinate_scale_locality_review_inputs(tmp_path)
    wrong_review = _write_coordinate_scale_locality_review_inputs(
        tmp_path,
        decision="NO_NEXT_SURFACE_APPROVED",
        approved_surface=None,
    )

    with pytest.raises(AutoresearchLoopError, match="max_candidates=1"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="coordinate-scale-too-many",
            mode="dry-run",
            planner="coordinate_scale_locality_diagnostic",
            start_trial_id="T165",
            max_candidates=2,
            diagnostic_report=report,
        )

    with pytest.raises(AutoresearchLoopError, match="requires --diagnostic-report"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="coordinate-scale-missing-review",
            mode="dry-run",
            planner="coordinate_scale_locality_diagnostic",
            start_trial_id="T165",
            max_candidates=1,
        )

    with pytest.raises(AutoresearchLoopError, match="APPROVE_OFFLINE_PLANNER_PR_ONLY"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="coordinate-scale-wrong-review",
            mode="dry-run",
            planner="coordinate_scale_locality_diagnostic",
            start_trial_id="T165",
            max_candidates=1,
            diagnostic_report=wrong_review,
        )

    assert not (tmp_path / "runs/autoresearch/coordinate-scale-too-many").exists()
    assert not (tmp_path / "runs/autoresearch/coordinate-scale-missing-review").exists()
    assert not (tmp_path / "runs/autoresearch/coordinate-scale-wrong-review").exists()


def test_coordinate_normalized_sampler_diagnostic_plans_one_normalized_candidate(tmp_path: Path) -> None:
    report = _write_prediction_geometry_audit_inputs(tmp_path)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="coordinate-normalized-sampler-diagnostic",
        mode="dry-run",
        planner="coordinate_normalized_sampler_diagnostic",
        start_trial_id="T166",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T166"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T166"
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["move_family"] == "diffusion_sampler_golf"
    assert trial["diagnostic_target"] == "distogram_good_lddt_flat"
    assert trial["sampler_coordinate_normalization"] == "ca_bond"
    assert trial["prediction"]["causal_component"] == "ca_bond_sampler_coordinate_normalization"
    assert trial["prediction"]["predicted_axis"] == "distogram_vs_3d"
    assert trial["config_path"] == "configs/experiments/T166_coordinate_normalized_sampler_diagnostic.json"
    assert trial["config_payload"]["max_templates"] == 0
    assert trial["config_payload"]["diffusion_loss_weight"] == pytest.approx(1.0)
    assert config["schema_version"] == "autoaf3.coordinate_normalized_sampler_diagnostic_plan.v1"
    assert config["reference_trial_id"] == "T088"
    assert config["candidate_trial_ids"] == ["T164", "T165"]
    assert config["sampler_coordinate_normalization"] == "ca_bond"
    assert "reference_radius_scale_shift" in config["scale_flags"]
    assert config["reference_deltas"][0]["mean_radius_scale_ratio"] == pytest.approx(55.315869408410755)
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T166_coordinate_normalized_sampler_diagnostic.json" in patch_text
    assert "bounded coordinate-normalized sampler diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_coordinate_normalized_sampler_diagnostic_requires_deployed_runtime_capability(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    report = _write_prediction_geometry_audit_inputs(tmp_path)
    client = FakeTrustedAutoresearchClient(_short_training_manifest("T166"))
    client.authority_payload = {"runtime_capabilities": {}}

    with pytest.raises(AutoresearchLoopError, match="post_training_sampler_coordinate_normalization"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="coordinate-normalized-stale-modal",
            mode="modal",
            planner="coordinate_normalized_sampler_diagnostic",
            start_trial_id="T166",
            max_candidates=1,
            candidate_budget="trial",
            diagnostic_report=report,
            approval=APPROVAL_TEXT,
            modal_client=client,
        )

    assert client.submitted_trials == []
    assert client.scored_trials == []


def test_coordinate_normalized_sampler_diagnostic_requires_geometry_scale_audit(tmp_path: Path) -> None:
    report = _write_prediction_geometry_audit_inputs(tmp_path)
    weak_report = _write_prediction_geometry_audit_inputs(
        tmp_path,
        flags=["adjacent_ca_distance_exploded"],
        report_name="weak-scale-flags.json",
    )
    unsafe_report = _write_prediction_geometry_audit_inputs(
        tmp_path,
        starts_search=True,
        report_name="unsafe-starts-search.json",
    )

    with pytest.raises(AutoresearchLoopError, match="max_candidates=1"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="coordinate-normalized-too-many",
            mode="dry-run",
            planner="coordinate_normalized_sampler_diagnostic",
            start_trial_id="T166",
            max_candidates=2,
            diagnostic_report=report,
        )

    with pytest.raises(AutoresearchLoopError, match="requires --diagnostic-report"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="coordinate-normalized-missing-report",
            mode="dry-run",
            planner="coordinate_normalized_sampler_diagnostic",
            start_trial_id="T166",
            max_candidates=1,
        )

    with pytest.raises(AutoresearchLoopError, match="requires reference coordinate-scale flags"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="coordinate-normalized-weak-report",
            mode="dry-run",
            planner="coordinate_normalized_sampler_diagnostic",
            start_trial_id="T166",
            max_candidates=1,
            diagnostic_report=weak_report,
        )

    with pytest.raises(AutoresearchLoopError, match="must not claim starts_search=true"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="coordinate-normalized-unsafe-report",
            mode="dry-run",
            planner="coordinate_normalized_sampler_diagnostic",
            start_trial_id="T166",
            max_candidates=1,
            diagnostic_report=unsafe_report,
        )

    assert not (tmp_path / "runs/autoresearch/coordinate-normalized-too-many").exists()
    assert not (tmp_path / "runs/autoresearch/coordinate-normalized-missing-report").exists()
    assert not (tmp_path / "runs/autoresearch/coordinate-normalized-weak-report").exists()
    assert not (tmp_path / "runs/autoresearch/coordinate-normalized-unsafe-report").exists()


def test_calibrated_coordinate_normalized_sampler_diagnostic_plans_scaled_candidate(tmp_path: Path) -> None:
    report = _write_collapsed_prediction_geometry_audit_inputs(tmp_path)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="calibrated-coordinate-normalized-sampler-diagnostic",
        mode="dry-run",
        planner="calibrated_coordinate_normalized_sampler_diagnostic",
        start_trial_id="T168",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T168"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T168"
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["sampler_coordinate_normalization"] == "ca_bond"
    assert trial["sampler_coordinate_scale"] == pytest.approx(13.126698)
    assert trial["prediction"]["causal_component"] == "calibrated_ca_bond_sampler_coordinate_scale"
    assert trial["prediction"]["predicted_axis"] == "distogram_vs_3d"
    assert trial["config_path"] == "configs/experiments/T168_calibrated_coordinate_normalized_sampler_diagnostic.json"
    assert config["schema_version"] == "autoaf3.calibrated_coordinate_normalized_sampler_diagnostic_plan.v1"
    assert config["reference_trial_id"] == "T088"
    assert config["candidate_trial_ids"] == ["T167"]
    assert config["sampler_coordinate_normalization"] == "ca_bond"
    assert config["sampler_coordinate_scale"] == pytest.approx(13.126698)
    assert config["calibration_rule"] == "inverse_mean_radius_scale_ratio"
    assert "adjacent_ca_distance_collapsed" in config["scale_flags"]
    assert "configs/experiments/T168_calibrated_coordinate_normalized_sampler_diagnostic.json" in patch_text
    assert "bounded calibrated coordinate-normalized sampler diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_calibrated_coordinate_normalized_sampler_diagnostic_requires_collapsed_geometry(tmp_path: Path) -> None:
    report = _write_prediction_geometry_audit_inputs(tmp_path)

    with pytest.raises(AutoresearchLoopError, match="collapsed coordinate-scale evidence"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="calibrated-coordinate-normalized-wrong-geometry",
            mode="dry-run",
            planner="calibrated_coordinate_normalized_sampler_diagnostic",
            start_trial_id="T168",
            max_candidates=1,
            candidate_budget="trial",
            diagnostic_report=report,
        )


def test_calibrated_coordinate_normalized_sampler_diagnostic_requires_scale_runtime_capability(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    report = _write_collapsed_prediction_geometry_audit_inputs(tmp_path)
    client = FakeTrustedAutoresearchClient(_short_training_manifest("T168"))
    client.authority_payload = {
        "runtime_capabilities": {"post_training_sampler_coordinate_normalization": True}
    }

    with pytest.raises(AutoresearchLoopError, match="post_training_sampler_coordinate_scale"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="calibrated-coordinate-normalized-stale-modal",
            mode="modal",
            planner="calibrated_coordinate_normalized_sampler_diagnostic",
            start_trial_id="T168",
            max_candidates=1,
            candidate_budget="trial",
            diagnostic_report=report,
            approval=APPROVAL_TEXT,
            modal_client=client,
        )

    assert client.submitted_trials == []
    assert client.scored_trials == []


def test_calibrated_sampler_locality_selection_diagnostic_plans_geometry_selected_candidate(tmp_path: Path) -> None:
    report = _write_controlled_scale_prediction_geometry_audit_inputs(tmp_path)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="calibrated-sampler-locality-selection-diagnostic",
        mode="dry-run",
        planner="calibrated_sampler_locality_selection_diagnostic",
        start_trial_id="T169",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T169"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T169"
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["sampler_coordinate_normalization"] == "ca_bond"
    assert trial["sampler_coordinate_scale"] == pytest.approx(13.126702)
    assert trial["sampler_num_samples"] == 4
    assert trial["sampler_selection_policy"] == "geometry"
    assert trial["prediction"]["causal_component"] == "label_free_sampler_geometry_selection"
    assert trial["config_path"] == "configs/experiments/T169_calibrated_sampler_locality_selection_diagnostic.json"
    assert config["schema_version"] == "autoaf3.calibrated_sampler_locality_selection_diagnostic_plan.v1"
    assert config["source_trial_id"] == "T168"
    assert config["source_sampler_manifest"] == "runs/autoresearch/modal_artifacts/T168/sampler_manifest.json"
    assert config["sampler_coordinate_scale"] == pytest.approx(13.126702)
    assert config["sampler_num_samples"] == 4
    assert config["sampler_selection_policy"] == "geometry"
    assert config["reference_deltas"][0]["mean_radius_scale_ratio"] == pytest.approx(0.995071710549596)
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T169_calibrated_sampler_locality_selection_diagnostic.json" in patch_text
    assert "bounded calibrated sampler locality-selection diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_calibrated_sampler_locality_selection_diagnostic_requires_resolved_scale(tmp_path: Path) -> None:
    report = _write_collapsed_prediction_geometry_audit_inputs(tmp_path)

    with pytest.raises(AutoresearchLoopError, match="controlled radius scale"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="calibrated-sampler-locality-selection-unresolved-scale",
            mode="dry-run",
            planner="calibrated_sampler_locality_selection_diagnostic",
            start_trial_id="T169",
            max_candidates=1,
            candidate_budget="trial",
            diagnostic_report=report,
        )


def test_calibrated_sampler_locality_selection_diagnostic_requires_selection_runtime_capability(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    report = _write_controlled_scale_prediction_geometry_audit_inputs(tmp_path)
    client = FakeTrustedAutoresearchClient(_short_training_manifest("T169"))
    client.authority_payload = {
        "runtime_capabilities": {
            "post_training_sampler_coordinate_normalization": True,
            "post_training_sampler_coordinate_scale": True,
        }
    }

    with pytest.raises(AutoresearchLoopError, match="post_training_sampler_selection"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="calibrated-sampler-locality-selection-stale-modal",
            mode="modal",
            planner="calibrated_sampler_locality_selection_diagnostic",
            start_trial_id="T169",
            max_candidates=1,
            candidate_budget="trial",
            diagnostic_report=report,
            approval=APPROVAL_TEXT,
            modal_client=client,
        )

    assert client.submitted_trials == []
    assert client.scored_trials == []


def test_calibrated_sampler_low_noise_diagnostic_plans_first_sample_candidate(tmp_path: Path) -> None:
    report = _write_controlled_scale_prediction_geometry_audit_inputs(
        tmp_path,
        source_trial_id="T169",
        report_name="T169-vs-T088.json",
        mean_radius_scale_ratio=0.9749788633633896,
        mean_pair_distance_delta=-2.583829989066608,
    )

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="calibrated-sampler-low-noise-diagnostic",
        mode="dry-run",
        planner="calibrated_sampler_low_noise_diagnostic",
        start_trial_id="T170",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T170"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T170"
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["sampler_coordinate_normalization"] == "ca_bond"
    assert trial["sampler_coordinate_scale"] == pytest.approx(13.126702)
    assert trial["sampler_noise_scale"] == pytest.approx(0.5)
    assert trial["sampler_num_samples"] == 1
    assert trial["sampler_selection_policy"] == "first"
    assert trial["prediction"]["causal_component"] == "calibrated_low_noise_first_sample"
    assert trial["config_path"] == "configs/experiments/T170_calibrated_sampler_low_noise_diagnostic.json"
    assert config["schema_version"] == "autoaf3.calibrated_sampler_low_noise_diagnostic_plan.v1"
    assert config["source_trial_id"] == "T169"
    assert config["source_sampler_manifest"] == "runs/autoresearch/modal_artifacts/T169/sampler_manifest.json"
    assert config["sampler_coordinate_scale"] == pytest.approx(13.126702)
    assert config["sampler_noise_scale"] == pytest.approx(0.5)
    assert config["sampler_num_samples"] == 1
    assert config["sampler_selection_policy"] == "first"
    assert config["reference_deltas"][0]["mean_radius_scale_ratio"] == pytest.approx(0.9749788633633896)
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T170_calibrated_sampler_low_noise_diagnostic.json" in patch_text
    assert "bounded calibrated sampler low-noise diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_calibrated_sampler_low_noise_diagnostic_requires_resolved_scale(tmp_path: Path) -> None:
    report = _write_collapsed_prediction_geometry_audit_inputs(tmp_path)

    with pytest.raises(AutoresearchLoopError, match="controlled radius scale"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="calibrated-sampler-low-noise-unresolved-scale",
            mode="dry-run",
            planner="calibrated_sampler_low_noise_diagnostic",
            start_trial_id="T170",
            max_candidates=1,
            candidate_budget="trial",
            diagnostic_report=report,
        )


def test_calibrated_sampler_low_noise_diagnostic_requires_schedule_runtime_capability(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    report = _write_controlled_scale_prediction_geometry_audit_inputs(
        tmp_path,
        source_trial_id="T169",
        report_name="T169-vs-T088.json",
        mean_radius_scale_ratio=0.9749788633633896,
        mean_pair_distance_delta=-2.583829989066608,
    )
    client = FakeTrustedAutoresearchClient(_short_training_manifest("T170"))
    client.authority_payload = {
        "runtime_capabilities": {
            "post_training_sampler_coordinate_normalization": True,
            "post_training_sampler_coordinate_scale": True,
            "post_training_sampler_selection": True,
        }
    }

    with pytest.raises(AutoresearchLoopError, match="post_training_sampler_schedule"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="calibrated-sampler-low-noise-stale-modal",
            mode="modal",
            planner="calibrated_sampler_low_noise_diagnostic",
            start_trial_id="T170",
            max_candidates=1,
            candidate_budget="trial",
            diagnostic_report=report,
            approval=APPROVAL_TEXT,
            modal_client=client,
        )

    assert client.submitted_trials == []
    assert client.scored_trials == []


def test_diffusion_data_scale_diagnostic_plans_one_model_scale_candidate(tmp_path: Path) -> None:
    report = _write_coordinate_scale_locality_review_inputs(
        tmp_path,
        approved_surface="diffusion_data_scale_diagnostic",
        report_name="T170-mixed-evidence.json",
        source_diagnosis="runs/autoresearch/post_discard_diagnosis/T170-vs-T169-T168-T088.json",
        rejected_surfaces=[
            "sampler_coordinate_scale",
            "sampler_geometry_selection",
            "sampler_low_noise",
        ],
        candidate_trial_ids=["T168", "T169", "T170"],
        negative_delta_count=27,
        positive_delta_count=20,
        worst_delta=-0.018736936398925055,
    )

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="diffusion-data-scale-diagnostic",
        mode="dry-run",
        planner="diffusion_data_scale_diagnostic",
        start_trial_id="T171",
        max_candidates=1,
        candidate_budget="trial",
        diagnostic_report=report,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T171"]
    assert result.starts_search is False
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    candidate_dir = Path(result.run_dir) / "candidates/T171"
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    patch_text = (candidate_dir / "patch.diff").read_text(encoding="utf-8")
    assert trial["trial_kind"] == "training"
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["move_family"] == "diffusion_schedule"
    assert trial["diagnostic_target"] == "distogram_good_lddt_flat"
    assert trial["sampler_coordinate_normalization"] == "ca_bond"
    assert trial["sampler_coordinate_scale"] == pytest.approx(13.126702)
    assert trial["sampler_num_samples"] == 1
    assert trial["sampler_selection_policy"] == "first"
    assert trial["prediction"]["causal_component"] == "diffusion_data_std_dev_scale"
    assert trial["config_path"] == "configs/experiments/T171_diffusion_data_scale_diagnostic.json"
    assert trial["config_payload"]["max_templates"] == 0
    assert trial["config_payload"]["diffusion_data_std_dev"] == pytest.approx(8.0)
    assert trial["config_payload"]["diffusion_gamma_0"] == pytest.approx(0.6)
    assert config["schema_version"] == "autoaf3.diffusion_data_scale_diagnostic_plan.v1"
    assert config["source_next_surface_review"] == str(report)
    assert config["candidate_trial_ids"] == ["T168", "T169", "T170"]
    assert config["rejected_surfaces"] == [
        "sampler_coordinate_scale",
        "sampler_geometry_selection",
        "sampler_low_noise",
    ]
    assert config["config_payload_overrides"]["diffusion_data_std_dev"] == pytest.approx(8.0)
    assert config["writes_ledger"] is False
    assert config["writes_discovery_ledger"] is False
    assert "configs/experiments/T171_diffusion_data_scale_diagnostic.json" in patch_text
    assert "bounded diffusion data-scale diagnostic" in patch_text
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()
    assert not (tmp_path / "runs/trials").exists()


def test_diffusion_data_scale_diagnostic_requires_matching_next_surface_review(tmp_path: Path) -> None:
    report = _write_coordinate_scale_locality_review_inputs(tmp_path)

    with pytest.raises(AutoresearchLoopError, match="diffusion_data_scale_diagnostic"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="diffusion-data-scale-wrong-review",
            mode="dry-run",
            planner="diffusion_data_scale_diagnostic",
            start_trial_id="T171",
            max_candidates=1,
            candidate_budget="trial",
            diagnostic_report=report,
        )

    assert not (tmp_path / "runs/autoresearch/diffusion-data-scale-wrong-review").exists()


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
    assert result.stopped_reason == "max_candidates_reached"
    assert result.starts_search is True
    assert result.writes_ledger is False
    assert result.writes_discovery_ledger is False
    assert result.generated_trials == ["T130"]
    assert len(result.wrote_files) == len(set(result.wrote_files))
    assert client.submitted_trials[0]["trial_id"] == "T130"
    assert client.submitted_trials[0]["trial_kind"] == "training"
    assert client.submitted_trials[0]["runner_mode"] == "short_training"
    assert client.submitted_trials[0]["features_path"] == "nanofold_event_small_no_templates.arrow"
    assert client.submitted_trials[0]["predict_after_training"] is True
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
    assert decision["promotion_status"] == "FALSIFICATION_REQUIRED"
    assert decision["promotion_plan_path"] == str(candidate_dir / "promotion_plan.json")
    assert decision["official_benchmark_result"] is False
    assert decision["global_baseline_delta"] == pytest.approx(0.01)
    promotion_plan = json.loads((candidate_dir / "promotion_plan.json").read_text(encoding="utf-8"))
    assert promotion_plan["status"] == "FALSIFICATION_REQUIRED"
    assert promotion_plan["writes_ledger"] is False
    assert promotion_plan["writes_discovery_ledger"] is False
    metrics = json.loads((candidate_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["comparison"]["provisional_keep"] is True
    summary = json.loads((tmp_path / "runs/autoresearch/live-scored/summary.json").read_text(encoding="utf-8"))
    candidate = summary["candidates"][0]
    assert candidate["status"] == "KEEP"
    assert candidate["promotion_status"] == "FALSIFICATION_REQUIRED"
    assert candidate["promotion_plan_path"] == str(candidate_dir / "promotion_plan.json")
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
    assert result.stopped_reason == "max_candidates_reached"
    candidate_dir = tmp_path / "runs/autoresearch/live-fail/candidates/T130"
    assert json.loads((candidate_dir / "decision.json").read_text(encoding="utf-8"))["status"] == "INFRA_FAIL"
    assert json.loads((candidate_dir / "decision.json").read_text(encoding="utf-8"))["promotion_status"] == "NOT_ELIGIBLE"
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


def test_autoresearch_loop_modal_runs_multiple_candidates_without_ledger(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    client = FakeSequencedTrustedAutoresearchClient({"T130": 0.07, "T131": 0.09})

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="live-two",
        mode="modal",
        planner="deterministic",
        start_trial_id="T130",
        max_candidates=2,
        approval=APPROVAL_TEXT,
        modal_client=client,
    )

    assert result.status == "PASS"
    assert result.generated_trials == ["T130", "T131"]
    assert [trial["trial_id"] for trial in client.submitted_trials] == ["T130", "T131"]
    assert all(trial["predict_after_training"] is True for trial in client.submitted_trials)
    assert client.scored_trials == ["T130", "T131"]
    assert [decision["status"] for decision in result.decisions] == ["DISCARD", "KEEP"]
    assert result.decisions[0]["promotion_status"] == "NOT_ELIGIBLE"
    assert result.decisions[1]["promotion_status"] == "FALSIFICATION_REQUIRED"
    assert result.decisions[0]["matched_budget_delta"] is None
    assert result.decisions[1]["matched_budget_delta"] == pytest.approx(0.02)
    summary = json.loads((tmp_path / "runs/autoresearch/live-two/summary.json").read_text(encoding="utf-8"))
    assert [candidate["status"] for candidate in summary["candidates"]] == ["DISCARD", "KEEP"]
    assert [candidate["promotion_status"] for candidate in summary["candidates"]] == [
        "NOT_ELIGIBLE",
        "FALSIFICATION_REQUIRED",
    ]
    assert summary["candidates"][1]["matched_budget_delta"] == pytest.approx(0.02)
    assert len(result.wrote_files) == len(set(result.wrote_files))
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_autoresearch_loop_modal_stops_after_failure_streak(tmp_path: Path) -> None:
    client = FakeSequencedFailingTrustedAutoresearchClient()

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="live-failure-streak",
        mode="modal",
        planner="deterministic",
        start_trial_id="T130",
        max_candidates=3,
        approval=APPROVAL_TEXT,
        modal_client=client,
        failure_streak_limit=2,
    )

    assert result.status == "PASS"
    assert result.stopped_reason == "failure_streak_limit:FAIL"
    assert result.generated_trials == ["T130", "T131"]
    assert [trial["trial_id"] for trial in client.submitted_trials] == ["T130", "T131"]
    assert client.scored_trials == ["T130", "T131"]
    assert [decision["status"] for decision in result.decisions] == ["FAIL", "FAIL"]
    assert not (tmp_path / "runs/autoresearch/live-failure-streak/candidates/T132").exists()
    summary = json.loads((tmp_path / "runs/autoresearch/live-failure-streak/summary.json").read_text(encoding="utf-8"))
    assert [candidate["status"] for candidate in summary["candidates"]] == ["FAIL", "FAIL"]
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_autoresearch_loop_modal_runs_full_ladder_including_sampler(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    scores = {f"T{number:03d}": 0.07 for number in range(130, 136)}
    client = FakeSequencedTrustedAutoresearchClient(scores)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="live-full",
        mode="modal",
        planner="deterministic",
        start_trial_id="T130",
        max_candidates=6,
        approval=APPROVAL_TEXT,
        modal_client=client,
    )

    assert result.status == "PASS"
    assert result.generated_trials == ["T130", "T131", "T132", "T133", "T134", "T135"]
    submitted = client.submitted_trials
    assert [trial["trial_kind"] for trial in submitted] == ["training", "training", "training", "training", "training", "sampler"]
    assert submitted[-1]["checkpoint_path"] == "/mnt/autoalphafold3/runs/trials/T133/checkpoint.pt"
    assert client.scored_trials == result.generated_trials
    assert all(decision["status"] == "DISCARD" for decision in result.decisions)
    assert result.decisions[-1]["sampler_status"] == "SAMPLER_PREDICTED"
    sampler_manifest = tmp_path / "runs/autoresearch/live-full/candidates/T135/sampler_manifest.json"
    assert json.loads(sampler_manifest.read_text(encoding="utf-8"))["checkpoint_path"] == submitted[-1]["checkpoint_path"]
    summary = json.loads((tmp_path / "runs/autoresearch/live-full/summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"][-1]["status"] == "DISCARD"
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


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


def test_llm_autoresearch_plans_bounded_batch_with_prior_plan_context(tmp_path: Path) -> None:
    fake = FakeLLMPlanner(
        [
            _llm_candidate_payload(trial_id="T120", config_path="configs/experiments/llm_geometry_a.json"),
            _llm_candidate_payload(trial_id="T121", config_path="configs/experiments/llm_geometry_b.json"),
        ]
    )

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="llm-two",
        mode="dry-run",
        planner="llm",
        max_candidates=2,
        planner_client=fake,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T120", "T121"]
    assert [call["trial_id"] for call in fake.calls] == ["T120", "T121"]
    assert fake.calls[0]["prior_plans"] == []
    assert fake.calls[1]["prior_plans"][0]["trial_id"] == "T120"
    assert fake.calls[1]["prior_plans"][0]["config"]["config_path"] == "configs/experiments/llm_geometry_a.json"
    assert (tmp_path / "runs/autoresearch/llm-two/candidates/T121/trial.json").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_llm_autoresearch_plans_trial_budget_candidate(tmp_path: Path) -> None:
    fake = FakeLLMPlanner(
        _llm_candidate_payload(
            trial_id="T140",
            config_path="configs/experiments/llm_trial_budget.json",
            budget="trial",
            max_steps=250,
            max_wall_minutes=45,
            timeout_cap=2700,
        )
    )

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="llm-trial-budget",
        mode="dry-run",
        planner="llm",
        start_trial_id="T140",
        max_candidates=1,
        planner_client=fake,
        candidate_budget="trial",
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T140"]
    assert fake.calls[0]["candidate_budget"] == "trial"
    candidate_dir = tmp_path / "runs/autoresearch/llm-trial-budget/candidates/T140"
    trial = json.loads((candidate_dir / "trial.json").read_text(encoding="utf-8"))
    assert trial["budget"] == "trial"
    assert trial["max_steps"] == 250
    assert trial["max_wall_minutes"] == 45
    assert trial["timeout_cap"] == 2700
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_llm_autoresearch_refuses_budget_shape_mismatch(tmp_path: Path) -> None:
    fake = FakeLLMPlanner(_llm_candidate_payload(trial_id="T140"))

    with pytest.raises(AutoresearchLoopError, match="budget must be trial"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-trial-budget-mismatch",
            mode="dry-run",
            planner="llm",
            start_trial_id="T140",
            max_candidates=1,
            planner_client=fake,
            candidate_budget="trial",
        )

    assert not (tmp_path / "runs/autoresearch/llm-trial-budget-mismatch").exists()
    assert fake.calls[0]["candidate_budget"] == "trial"


def test_llm_autoresearch_planner_receives_prior_run_outcomes(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    prior_client = FakeSequencedTrustedAutoresearchClient({"T130": 0.07})
    run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="prior-live",
        mode="modal",
        planner="deterministic",
        start_trial_id="T130",
        max_candidates=1,
        approval=APPROVAL_TEXT,
        modal_client=prior_client,
    )
    fake = FakeLLMPlanner(_llm_candidate_payload(trial_id="T140"))

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="llm-with-prior",
        mode="dry-run",
        planner="llm",
        start_trial_id="T140",
        max_candidates=1,
        planner_client=fake,
        prior_run_ids=["prior-live"],
    )

    assert result.generated_trials == ["T140"]
    prior_outcomes = fake.calls[0]["prior_outcomes"]
    assert len(prior_outcomes) == 1
    assert prior_outcomes[0] == {
        "run_id": "prior-live",
        "trial_id": "T130",
        "status": "DISCARD",
        "promotion_status": "NOT_ELIGIBLE",
        "provisional_keep": False,
        "matched_budget_delta": None,
        "global_baseline_delta": pytest.approx(-0.01),
        "candidate_score": pytest.approx(0.07),
        "fold_cartographer_signature": "candidate_scored",
        "candidate_artifacts": {
            "metrics_json": "/mnt/autoalphafold3/runs/trials/T130/metrics.json",
            "predictions_json": "/mnt/autoalphafold3/runs/trials/T130/predictions.json",
        },
        "hypothesis": "Deterministic ladder candidate short_train_baseline_smoke tests bounded local-geometry short-training behavior.",
        "move_family": "curriculum",
        "diagnostic_target": "local_geometry_weak",
        "config_path": "configs/nanofold_dev_cpu_smoke.json",
        "budget": "smoke",
    }
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_llm_autoresearch_rejects_unbounded_batch_before_artifacts(tmp_path: Path) -> None:
    fake = FakeLLMPlanner(_llm_candidate_payload())

    with pytest.raises(AutoresearchLoopError, match="between 1 and 3"):
        run_autoresearch_loop(
            repo_root=tmp_path,
            run_id="llm-four",
            mode="dry-run",
            planner="llm",
            max_candidates=4,
            planner_client=fake,
        )

    assert not (tmp_path / "runs/autoresearch/llm-four").exists()
    assert fake.calls == []


def test_llm_modal_candidate_passes_inline_config_payload_without_ledger(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    config_payload = json.loads((REPO_ROOT / "configs/nanofold_dev_cpu_smoke.json").read_text(encoding="utf-8"))
    config_payload["learning_rate"] = 0.0016
    fake = FakeLLMPlanner(_llm_candidate_payload(config_payload=config_payload))
    client = FakeTrustedAutoresearchClient(_short_training_manifest("T120"), _scored_metrics_payload("T120", score=0.07))

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="llm-inline-config",
        mode="modal",
        planner="llm",
        max_candidates=1,
        approval=APPROVAL_TEXT,
        planner_client=fake,
        modal_client=client,
    )

    assert result.status == "PASS"
    assert client.submitted_trials[0]["config_payload"]["learning_rate"] == pytest.approx(0.0016)
    assert client.submitted_trials[0]["config_payload"]["max_templates"] == 0
    assert result.decisions[0]["status"] == "DISCARD"
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


def test_llm_modal_runs_bounded_batch_with_matched_budget_without_ledger(tmp_path: Path) -> None:
    _write_baseline_lock(tmp_path, score=0.08)
    fake = FakeLLMPlanner(
        [
            _llm_candidate_payload(trial_id="T120", config_path="configs/experiments/llm_geometry_a.json"),
            _llm_candidate_payload(trial_id="T121", config_path="configs/experiments/llm_geometry_b.json"),
        ]
    )
    client = FakeSequencedTrustedAutoresearchClient({"T120": 0.07, "T121": 0.09})

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="llm-live-two",
        mode="modal",
        planner="llm",
        max_candidates=2,
        approval=APPROVAL_TEXT,
        planner_client=fake,
        modal_client=client,
    )

    assert result.status == "PASS"
    assert result.generated_trials == ["T120", "T121"]
    assert [trial["trial_id"] for trial in client.submitted_trials] == ["T120", "T121"]
    assert client.scored_trials == ["T120", "T121"]
    assert [decision["status"] for decision in result.decisions] == ["DISCARD", "KEEP"]
    assert result.decisions[0]["matched_budget_delta"] is None
    assert result.decisions[1]["matched_budget_delta"] == pytest.approx(0.02)
    assert fake.calls[1]["prior_plans"][0]["trial_id"] == "T120"
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    assert not (tmp_path / "runs/discovery_ledger.jsonl").exists()


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


def test_llm_autoresearch_uses_live_planner_when_no_recorded_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _llm_candidate_payload()
    calls: list[dict[str, object]] = []

    class FakeLivePlanner:
        def __init__(self, *, repo_root: str | Path = ".", model: str = "ignored") -> None:
            calls.append({"repo_root": Path(repo_root), "model": model})

        def plan(self, **kwargs: object) -> AutoresearchCandidatePlan:
            calls.append(kwargs)
            return AutoresearchCandidatePlan.model_validate(payload)

    monkeypatch.setattr("autoalphafold3.autoresearch_loop.OpenAIAutoresearchPlanner", FakeLivePlanner)

    result = run_autoresearch_loop(
        repo_root=tmp_path,
        run_id="llm-live-planner",
        mode="dry-run",
        planner="llm",
        max_candidates=1,
    )

    assert result.status == "PLANNED"
    assert result.generated_trials == ["T120"]
    assert result.llm_policy is not None
    assert calls[0]["repo_root"] == tmp_path
    assert calls[1]["trial_id"] == "T120"
    assert calls[1]["base_commit"] == "unknown"
    assert (tmp_path / "runs/autoresearch/llm-live-planner/candidates/T120/trial.json").exists()
    assert not (tmp_path / "runs/ledger.jsonl").exists()


def test_openai_autoresearch_planner_falls_back_to_modal_harness_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, object]] = []

    fake_openai = types.ModuleType("openai")

    class MissingKeyOpenAI:
        def __init__(self) -> None:
            raise RuntimeError("The api_key client option must be set by environment variable")

    fake_openai.OpenAI = MissingKeyOpenAI

    class FakeRemoteMethod:
        def remote(self, payload: dict[str, object]) -> dict[str, object]:
            payloads.append(payload)
            return _llm_candidate_payload()

    class FakeOrchestrator:
        plan_autoresearch_candidate = FakeRemoteMethod()

    class FakeCls:
        @staticmethod
        def from_name(app_name: str, class_name: str):
            assert app_name == "autoalphafold3-modal"
            assert class_name == "TrustedOrchestrator"
            return FakeOrchestrator

    fake_modal = types.ModuleType("modal")
    fake_modal.Cls = FakeCls
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    plan = OpenAIAutoresearchPlanner(repo_root=tmp_path).plan(
        run_id="llm-modal-secret",
        trial_id="T120",
        candidate_index=0,
        model="gpt-5.4-mini",
        policy={"patch_planning": {"tools": []}},
        base_commit="abc1234",
    )

    assert plan.trial.trial_id == "T120"
    assert payloads[0]["trial_id"] == "T120"
    assert "NanoFold-style AlphaFold3-lite" in str(payloads[0]["prompt"])


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
    assert "recorded LLM candidate plans can replay exactly one candidate" in payload["error"]
