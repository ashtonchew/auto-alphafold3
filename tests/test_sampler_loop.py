from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.sampler_loop import (
    APPROVAL_TEXT,
    SamplerCandidatePlan,
    SamplerLoopError,
    run_incremental_sampler_loop,
)
from tests.test_two_stage_orchestrator import write_baseline_lock


def seed_trial(tmp_path: Path) -> Path:
    path = tmp_path / "trials/T012.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "trial_id": "T012",
                "parent_commit": "a" * 40,
                "agent_session_id": "pytest",
                "trial_kind": "sampler",
                "hypothesis": "seed",
                "move_family": "diffusion_sampler_golf",
                "diagnostic_target": "local_geometry_weak",
                "prediction": {
                    "causal_component": "sampler",
                    "predicted_axis": "local_geometry",
                    "predicted_direction": "up",
                    "expected_lddt_delta_band": [0.001, 0.01],
                },
                "patch_path": None,
                "config_path": "configs/nanofold_dev_cpu_smoke.json",
                "budget": "sampler",
                "seed": 0,
                "n_res": 32,
                "sampler_steps": 1,
                "sampler_noise_scale": 1.0,
                "sampler_step_scale": 1.0,
                "sampler_schedule_shape": "linear",
                "sampler_num_samples": 1,
                "sampler_selection_policy": "first",
                "max_wall_minutes": 5,
                "manifest_hashes": {},
                "scorer_version": "calpha_lddt_v1",
                "primary_metric": "best_val_calpha_lddt",
                "param_cap": 176514,
                "gpu_memory_cap": 80.0,
                "cost_cap": 2.0,
                "timeout_cap": 300,
                "artifact_dir": "runs/trials/T012",
                "checkpoint_path": "/mnt/autoalphafold3/runs/trials/T010/checkpoint.pt",
            }
        ),
        encoding="utf-8",
    )
    return path


class FakeSamplerClient:
    def __init__(self, scores: list[float] | None = None, fail: bool = False) -> None:
        self.scores = scores or [0.1]
        self.fail = fail
        self.submitted: list[str] = []

    def submit(self, trial_path: Path) -> str:
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
        self.submitted.append(trial["trial_id"])
        return f"modal:orchestrator-{trial['trial_id']}"

    def wait_for_sampler(self, call_id: str, *, timeout_s: int, poll_interval_s: float) -> dict[str, object]:
        if self.fail:
            raise TimeoutError("synthetic timeout")
        trial_id = call_id.rsplit("-", 1)[-1]
        return {
            "schema_version": "autoaf3.sampler_manifest.v1",
            "status": "SAMPLER_PREDICTED",
            "trial_id": trial_id,
            "real_training_performed": False,
            "inference_only": True,
            "writes_discovery_ledger": False,
        }

    def score(self, trial_id: str) -> dict[str, object]:
        score = self.scores[min(len(self.submitted) - 1, len(self.scores) - 1)]
        return {
            "schema_version": "autoaf3.metrics.v1",
            "status": "SCORED",
            "trial_id": trial_id,
            "candidate_id": f"{trial_id}_sampler",
            "primary_metric": "best_val_calpha_lddt",
            "metrics": {
                "best_val_calpha_lddt": score,
                "num_targets": 16,
                "num_scored_targets": 16,
                "num_failed_targets": 0,
            },
            "fold_cartographer": {"signature": "scored", "summary": {}, "buckets": {}},
            "artifacts": {"metrics_json": f"runs/trials/{trial_id}/metrics.json"},
            "error_report": {"failure_signature": None},
        }


class ScoreAwarePlanner:
    def __init__(self) -> None:
        self.observed_scores: list[float | None] = []
        self.observed_fold_cartographer: list[dict[str, object] | None] = []
        self.observed_strategy_recommendations: list[str | None] = []

    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        global_current_best: dict[str, object],
        search_reference: dict[str, object],
        strategy_context: dict[str, object],
    ) -> SamplerCandidatePlan:
        latest_score = prior_decisions[-1].get("score") if prior_decisions else None
        self.observed_scores.append(float(latest_score) if isinstance(latest_score, int | float) else None)
        latest_fold_cartographer = prior_decisions[-1].get("fold_cartographer") if prior_decisions else None
        self.observed_fold_cartographer.append(
            latest_fold_cartographer if isinstance(latest_fold_cartographer, dict) else None
        )
        recommendation = strategy_context.get("recommendation")
        self.observed_strategy_recommendations.append(str(recommendation) if recommendation is not None else None)
        sampler_steps = 4 if latest_score is not None and latest_score < 0.42 else 1
        return SamplerCandidatePlan(
            diagnostic_target="local_geometry_weak",
            hypothesis=f"LLM-style planner chooses sampler candidate {candidate_index} after observing prior score.",
            intervention=f"Use sampler_steps={sampler_steps} from score-aware planner.",
            predicted_direction="up",
            expected_lddt_delta_band=[0.001, 0.01],
            sampler_steps=sampler_steps,
            sampler_noise_scale=1.0,
            sampler_step_scale=1.0,
            sampler_schedule_shape="linear",
            sampler_num_samples=1,
            sampler_selection_policy="first",
            seed=10 + candidate_index,
            rationale=(
                f"latest_score={latest_score}; global_best={global_current_best.get('score')}; "
                f"search_reference={search_reference.get('score')}"
            ),
        )


class InvalidPlanner:
    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        global_current_best: dict[str, object],
        search_reference: dict[str, object],
        strategy_context: dict[str, object],
    ) -> SamplerCandidatePlan:
        return SamplerCandidatePlan(
            diagnostic_target="local_geometry_weak",
            hypothesis="Invalid planner tries to edit scorer labels and Modal GPU policy.",
            intervention="Change scorer labels and Modal GPU policy.",
            predicted_direction="up",
            expected_lddt_delta_band=[0.001, 0.01],
            sampler_steps=1,
            sampler_noise_scale=1.0,
            sampler_step_scale=1.0,
            sampler_schedule_shape="linear",
            sampler_num_samples=1,
            sampler_selection_policy="first",
            seed=0,
            rationale="This should fail validation.",
        )


class LocalNeighborhoodPlanner:
    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        global_current_best: dict[str, object],
        search_reference: dict[str, object],
        strategy_context: dict[str, object],
    ) -> SamplerCandidatePlan:
        return SamplerCandidatePlan(
            diagnostic_target="local_geometry_weak",
            hypothesis="Try one more late-refine compact-geometry neighborhood sampler candidate near T088.",
            intervention="Use late_refine, compact_geometry, high step scale, and low noise again.",
            predicted_direction="up",
            expected_lddt_delta_band=[0.001, 0.01],
            sampler_steps=10,
            sampler_noise_scale=0.8,
            sampler_step_scale=1.3,
            sampler_schedule_shape="late_refine",
            sampler_num_samples=4,
            sampler_selection_policy="compact_geometry",
            seed=123,
            rationale="This intentionally repeats the exhausted local neighborhood.",
        )


class DistinctSamplerPlanner:
    def __init__(self) -> None:
        self.observed_strategy: dict[str, object] | None = None

    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        global_current_best: dict[str, object],
        search_reference: dict[str, object],
        strategy_context: dict[str, object],
    ) -> SamplerCandidatePlan:
        self.observed_strategy = strategy_context
        return SamplerCandidatePlan(
            diagnostic_target="stability_compute",
            hypothesis="Pivot away from the exhausted T088 late-refine neighborhood with a distinct linear sampler mechanism.",
            intervention="Use a linear schedule with first-sample selection and neutral scales.",
            predicted_direction="up",
            expected_lddt_delta_band=[0.001, 0.01],
            sampler_steps=4,
            sampler_noise_scale=1.0,
            sampler_step_scale=1.0,
            sampler_schedule_shape="linear",
            sampler_num_samples=1,
            sampler_selection_policy="first",
            seed=124,
            rationale="The strategy gate requires a non-neighborhood mechanism after repeated all-target regressions.",
        )


class DiagnosticSamplerClient(FakeSamplerClient):
    def score(self, trial_id: str) -> dict[str, object]:
        payload = super().score(trial_id)
        payload["fold_cartographer"] = {
            "signature": "toy_geometry_failed",
            "summary": {
                "canonical_target": "local_geometry_weak",
                "mean_target_calpha_lddt": 0.123,
                "nan_prediction_residue_count": 0,
                "num_scored_targets": 16,
                "num_targets": 16,
                "ignored_verbose_field": "not needed by planner",
            },
            "buckets": {
                "toy_all": {
                    "eligible_pair_count": 38149,
                    "target_ids": ["A", "B", "C", "D", "E", "F"],
                }
            },
        }
        return payload


def _write_sampler_ledger_row(root: Path, *, trial_id: str, score: float) -> None:
    ledger = root / "runs/ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "autoaf3.result.v1",
        "status": "DISCARD",
        "trial_id": trial_id,
        "candidate_id": f"{trial_id}_sampler",
        "primary_metric": "best_val_calpha_lddt",
        "metrics": {
            "best_val_calpha_lddt": score,
            "num_targets": 16,
            "num_scored_targets": 16,
            "num_failed_targets": 0,
        },
        "fold_cartographer": {
            "signature": "toy_geometry_failed",
            "summary": {
                "canonical_target": "local_geometry_weak",
                "mean_target_calpha_lddt": score,
                "nan_prediction_residue_count": 0,
                "num_scored_targets": 16,
                "num_targets": 16,
            },
            "buckets": {},
        },
        "artifacts": {"metrics_json": f"runs/trials/{trial_id}/metrics.json"},
        "postmortem": "Synthetic sampler row for strategy gate tests.",
    }
    ledger.write_text(
        ledger.read_text(encoding="utf-8") + json.dumps(payload, sort_keys=True) + "\n"
        if ledger.exists()
        else json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_sampler_trial_knobs(root: Path, *, trial_id: str, score_rank: int) -> None:
    trials = root / "trials"
    trials.mkdir(parents=True, exist_ok=True)
    payload = json.loads((trials / "T012.json").read_text(encoding="utf-8"))
    payload.update(
        {
            "trial_id": trial_id,
            "artifact_dir": f"runs/trials/{trial_id}",
            "sampler_steps": 12 - min(score_rank, 2),
            "sampler_noise_scale": 0.6 + score_rank * 0.05,
            "sampler_step_scale": 1.5 - score_rank * 0.05,
            "sampler_schedule_shape": "late_refine",
            "sampler_num_samples": 4,
            "sampler_selection_policy": "compact_geometry",
            "seed": 88000 + score_rank,
        }
    )
    (trials / f"{trial_id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_all_target_regression_report(root: Path, *, reference_trial_id: str, trial_ids: list[str]) -> None:
    out = root / "runs/autoresearch/scorer_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "autoaf3.scorer_sensitivity.v1",
        "reference_trial_id": reference_trial_id,
        "per_target_score_deltas_vs_reference": {
            trial_id: {"A": -0.01, "B": -0.02, "C": -0.03} for trial_id in trial_ids
        },
        "metric_deltas_vs_reference": {
            trial_id: {"best_val_calpha_lddt": -0.01} for trial_id in trial_ids
        },
    }
    (out / f"{reference_trial_id}-vs-{'-'.join(trial_ids)}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_exhausted_sampler_strategy_fixture(root: Path) -> Path:
    seed_path = seed_trial(root)
    _write_sampler_ledger_row(root, trial_id="T088", score=0.0209)
    for rank, (trial_id, score) in enumerate((("T108", 0.0192), ("T109", 0.0177), ("T110", 0.0166))):
        _write_sampler_trial_knobs(root, trial_id=trial_id, score_rank=rank)
        _write_sampler_ledger_row(root, trial_id=trial_id, score=score)
    _write_all_target_regression_report(root, reference_trial_id="T088", trial_ids=["T108", "T109", "T110"])
    return seed_path


def _write_score_regressed_sampler_strategy_fixture(root: Path) -> Path:
    seed_path = seed_trial(root)
    _write_sampler_ledger_row(root, trial_id="T088", score=0.0209)
    for rank, (trial_id, score) in enumerate((("T108", 0.0192), ("T109", 0.0177), ("T110", 0.0166))):
        _write_sampler_trial_knobs(root, trial_id=trial_id, score_rank=rank)
        _write_sampler_ledger_row(root, trial_id=trial_id, score=score)
    return seed_path


def test_sampler_loop_dry_run_generates_incremental_trials(tmp_path: Path) -> None:
    result = run_incremental_sampler_loop(
        seed_trial_path=seed_trial(tmp_path),
        repo_root=tmp_path,
        max_candidates=3,
        start_trial_id="T020",
    )

    assert result.status == "PASS"
    assert result.mode == "dry-run"
    assert result.generated_trials == ["T020", "T021", "T022"]
    assert result.scored_trials == []
    assert result.planner == "deterministic"
    assert len(result.wrote_files) == 3
    trial = json.loads((tmp_path / "trials/T020.json").read_text())
    assert trial["sampler_steps"] == 4
    assert trial["sampler_noise_scale"] == 1.0
    assert trial["sampler_schedule_shape"] == "linear"
    assert trial["sampler_num_samples"] == 1


def test_sampler_loop_reference_sweep_starts_near_recorded_t088_settings(tmp_path: Path) -> None:
    result = run_incremental_sampler_loop(
        seed_trial_path=seed_trial(tmp_path),
        repo_root=tmp_path,
        max_candidates=2,
        start_trial_id="T120",
        planner="reference_sweep",
        search_reference_trial_id="T088",
    )

    assert result.status == "PASS"
    assert result.planner == "reference_sweep"
    assert result.generated_trials == ["T120", "T121"]
    first = json.loads((tmp_path / "trials/T120.json").read_text())
    second = json.loads((tmp_path / "trials/T121.json").read_text())
    assert first["sampler_steps"] == 12
    assert first["sampler_noise_scale"] == pytest.approx(0.6)
    assert first["sampler_step_scale"] == pytest.approx(1.5)
    assert first["sampler_schedule_shape"] == "late_refine"
    assert first["sampler_num_samples"] == 4
    assert first["sampler_selection_policy"] == "compact_geometry"
    assert first["seed"] == 88000
    assert second["sampler_noise_scale"] == pytest.approx(0.55)
    assert second["sampler_step_scale"] == pytest.approx(1.6)


def test_sampler_loop_strategy_pivot_avoids_blocked_t088_neighborhood(tmp_path: Path) -> None:
    seed_path = _write_exhausted_sampler_strategy_fixture(tmp_path)

    result = run_incremental_sampler_loop(
        seed_trial_path=seed_path,
        repo_root=tmp_path,
        max_candidates=2,
        start_trial_id="T130",
        planner="strategy_pivot",
        search_reference_trial_id="T081",
        prior_decision_trial_ids=["T108", "T109", "T110"],
    )

    assert result.status == "PASS"
    assert result.planner == "strategy_pivot"
    assert result.generated_trials == ["T130", "T131"]
    first = json.loads((tmp_path / "trials/T130.json").read_text())
    second = json.loads((tmp_path / "trials/T131.json").read_text())
    assert first["trial_kind"] == "sampler"
    assert "max_steps" not in first
    assert first["sampler_schedule_shape"] == "cosine"
    assert first["sampler_selection_policy"] == "geometry"
    assert first["sampler_noise_scale"] == pytest.approx(1.35)
    assert first["sampler_step_scale"] == pytest.approx(0.75)
    assert second["sampler_schedule_shape"] == "linear"
    assert second["sampler_selection_policy"] == "first"
    assert result.decisions[-1]["strategy_context"]["recommendation"] == "stop_t088_neighborhood"


def test_sampler_loop_modal_requires_exact_approval(tmp_path: Path) -> None:
    with pytest.raises(SamplerLoopError, match=APPROVAL_TEXT):
        run_incremental_sampler_loop(
            seed_trial_path=seed_trial(tmp_path),
            repo_root=tmp_path,
            mode="modal",
            client=FakeSamplerClient(),
        )


def test_sampler_loop_modal_scores_and_records_stage_one(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    result = run_incremental_sampler_loop(
        seed_trial_path=seed_trial(tmp_path),
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        max_candidates=2,
        start_trial_id="T030",
        baseline_dir=baseline.relative_to(tmp_path),
        client=FakeSamplerClient(scores=[0.1, 0.43]),
    )

    assert result.status == "PASS"
    assert result.scored_trials == ["T030", "T031"]
    assert result.best_trial_id == "T031"
    assert result.decisions[0]["status"] == "DISCARD"
    assert result.decisions[1]["status"] == "KEEP"
    assert result.decisions[0]["beats_global_current_best"] is False
    assert result.decisions[1]["beats_global_current_best"] is True
    ledger = (tmp_path / "runs/ledger.jsonl").read_text(encoding="utf-8")
    assert '"status": "SCORED"' in ledger
    assert '"status": "KEEP"' in ledger


def test_sampler_loop_reports_search_reference_separately_from_global_keep(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    result = run_incremental_sampler_loop(
        seed_trial_path=seed_trial(tmp_path),
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        max_candidates=2,
        start_trial_id="T080",
        baseline_dir=baseline.relative_to(tmp_path),
        client=FakeSamplerClient(scores=[0.1, 0.2]),
        search_reference_trial_id="T080",
    )

    assert result.search_reference["trial_id"] == "T080"
    assert result.search_reference["score"] == 0.1
    assert result.decisions[0]["status"] == "DISCARD"
    assert result.decisions[0]["sampler_search_status"] == "SAMPLER_NOT_IMPROVED"
    assert result.decisions[0]["search_reference_delta"] == 0.0
    assert result.decisions[1]["status"] == "DISCARD"
    assert result.decisions[1]["beats_global_current_best"] is False
    assert result.decisions[1]["beats_search_reference"] is True
    assert result.decisions[1]["sampler_search_status"] == "SAMPLER_IMPROVED"
    assert result.decisions[1]["search_reference_delta"] == pytest.approx(0.1)
    assert result.decisions[1]["global_delta"] == pytest.approx(-0.22)


def test_sampler_loop_reference_sweep_keeps_stage_one_boundaries(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    result = run_incremental_sampler_loop(
        seed_trial_path=seed_trial(tmp_path),
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        max_candidates=1,
        start_trial_id="T122",
        planner="reference_sweep",
        baseline_dir=baseline.relative_to(tmp_path),
        client=FakeSamplerClient(scores=[0.2]),
        search_reference_trial_id="T088",
    )

    assert result.status == "PASS"
    assert result.scored_trials == ["T122"]
    assert result.decisions[0]["planner"] == "reference_sweep"
    assert result.decisions[0]["status"] == "DISCARD"
    assert result.decisions[0]["beats_global_current_best"] is False
    assert result.writes_discovery_ledger is False
    trial = json.loads((tmp_path / "trials/T122.json").read_text())
    assert trial["sampler_steps"] == 12
    assert trial["sampler_selection_policy"] == "compact_geometry"


def test_sampler_loop_strategy_pivot_keeps_stage_one_boundaries(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    result = run_incremental_sampler_loop(
        seed_trial_path=seed_trial(tmp_path),
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        max_candidates=1,
        start_trial_id="T132",
        planner="strategy_pivot",
        baseline_dir=baseline.relative_to(tmp_path),
        client=FakeSamplerClient(scores=[0.2]),
        search_reference_trial_id="T088",
    )

    assert result.status == "PASS"
    assert result.scored_trials == ["T132"]
    assert result.decisions[0]["planner"] == "strategy_pivot"
    assert result.decisions[0]["status"] == "DISCARD"
    assert result.decisions[0]["beats_global_current_best"] is False
    assert result.writes_discovery_ledger is False
    trial = json.loads((tmp_path / "trials/T132.json").read_text())
    assert trial["sampler_schedule_shape"] == "cosine"
    assert trial["sampler_selection_policy"] == "geometry"


def test_sampler_loop_stops_on_repeated_infra_failures(tmp_path: Path) -> None:
    result = run_incremental_sampler_loop(
        seed_trial_path=seed_trial(tmp_path),
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        max_candidates=5,
        start_trial_id="T040",
        failure_streak_limit=2,
        client=FakeSamplerClient(fail=True),
    )

    assert result.status == "FAIL"
    assert result.generated_trials == ["T040", "T041"]
    assert result.stopped_reason.startswith("failure_streak_limit")


def test_sampler_loop_planner_observes_prior_score_before_next_candidate(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    planner = ScoreAwarePlanner()
    result = run_incremental_sampler_loop(
        seed_trial_path=seed_trial(tmp_path),
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        max_candidates=2,
        start_trial_id="T050",
        baseline_dir=baseline.relative_to(tmp_path),
        client=FakeSamplerClient(scores=[0.1, 0.43]),
        planner="llm",
        planner_client=planner,
    )

    assert result.status == "PASS"
    assert planner.observed_scores == [None, 0.1]
    assert planner.observed_fold_cartographer[0] is None
    assert json.loads((tmp_path / "trials/T050.json").read_text())["sampler_steps"] == 1
    assert json.loads((tmp_path / "trials/T051.json").read_text())["sampler_steps"] == 4
    assert result.decisions[1]["planner"] == "llm"


def test_sampler_loop_feeds_fold_cartographer_diagnostics_to_next_plan(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    planner = ScoreAwarePlanner()
    result = run_incremental_sampler_loop(
        seed_trial_path=seed_trial(tmp_path),
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        max_candidates=2,
        start_trial_id="T070",
        baseline_dir=baseline.relative_to(tmp_path),
        client=DiagnosticSamplerClient(scores=[0.1, 0.2]),
        planner="llm",
        planner_client=planner,
    )

    diagnostic = planner.observed_fold_cartographer[1]
    assert result.status == "PASS"
    assert diagnostic is not None
    assert diagnostic["signature"] == "toy_geometry_failed"
    assert diagnostic["canonical_target"] == "local_geometry_weak"
    assert diagnostic["mean_target_calpha_lddt"] == 0.123
    assert diagnostic["summary"] == {
        "canonical_target": "local_geometry_weak",
        "mean_target_calpha_lddt": 0.123,
        "nan_prediction_residue_count": 0,
        "num_scored_targets": 16,
        "num_targets": 16,
    }
    assert diagnostic["buckets"] == {
        "toy_all": {
            "eligible_pair_count": 38149,
            "target_count": 6,
            "target_ids_head": ["A", "B", "C", "D", "E"],
        }
    }


def test_sampler_loop_can_continue_from_scored_ledger_decisions(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    seed_path = seed_trial(tmp_path)
    first_planner = ScoreAwarePlanner()
    first = run_incremental_sampler_loop(
        seed_trial_path=seed_path,
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        max_candidates=2,
        start_trial_id="T080",
        baseline_dir=baseline.relative_to(tmp_path),
        client=DiagnosticSamplerClient(scores=[0.1, 0.2]),
        planner="llm",
        planner_client=first_planner,
        search_reference_trial_id="T080",
    )
    continuation_planner = ScoreAwarePlanner()

    second = run_incremental_sampler_loop(
        seed_trial_path=seed_path,
        repo_root=tmp_path,
        mode="modal",
        approval=APPROVAL_TEXT,
        max_candidates=1,
        start_trial_id="T082",
        baseline_dir=baseline.relative_to(tmp_path),
        client=DiagnosticSamplerClient(scores=[0.3]),
        planner="llm",
        planner_client=continuation_planner,
        search_reference_trial_id="T080",
        prior_decision_trial_ids=["T080", "T081"],
    )

    assert first.status == "PASS"
    assert second.status == "PASS"
    assert continuation_planner.observed_scores == [0.2]
    diagnostic = continuation_planner.observed_fold_cartographer[0]
    assert diagnostic is not None
    assert diagnostic["signature"] == "toy_geometry_failed"
    assert second.decisions[0]["continuation_source"] == "ledger"
    assert second.decisions[-1]["trial_id"] == "T082"
    assert second.decisions[-1]["beats_search_reference"] is True


def test_sampler_strategy_gate_blocks_repeated_t088_neighborhood_regressions(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    seed_path = _write_exhausted_sampler_strategy_fixture(tmp_path)

    with pytest.raises(SamplerLoopError, match="strategy gate blocks"):
        run_incremental_sampler_loop(
            seed_trial_path=seed_path,
            repo_root=tmp_path,
            mode="dry-run",
            max_candidates=1,
            start_trial_id="T120",
            baseline_dir=baseline.relative_to(tmp_path),
            planner="llm",
            planner_client=LocalNeighborhoodPlanner(),
            search_reference_trial_id="T081",
            prior_decision_trial_ids=["T108", "T109", "T110"],
        )

    assert not (tmp_path / "trials/T120.json").exists()


def test_sampler_strategy_gate_blocks_score_only_neighborhood_regressions(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    seed_path = _write_score_regressed_sampler_strategy_fixture(tmp_path)

    with pytest.raises(SamplerLoopError, match="strategy gate blocks"):
        run_incremental_sampler_loop(
            seed_trial_path=seed_path,
            repo_root=tmp_path,
            mode="dry-run",
            max_candidates=1,
            start_trial_id="T120",
            baseline_dir=baseline.relative_to(tmp_path),
            planner="llm",
            planner_client=LocalNeighborhoodPlanner(),
            search_reference_trial_id="T081",
            prior_decision_trial_ids=["T108", "T109", "T110"],
        )

    assert not (tmp_path / "trials/T120.json").exists()


def test_sampler_strategy_gate_allows_distinct_pivot_with_context(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    seed_path = _write_exhausted_sampler_strategy_fixture(tmp_path)
    planner = DistinctSamplerPlanner()

    result = run_incremental_sampler_loop(
        seed_trial_path=seed_path,
        repo_root=tmp_path,
        mode="dry-run",
        max_candidates=1,
        start_trial_id="T121",
        baseline_dir=baseline.relative_to(tmp_path),
        planner="llm",
        planner_client=planner,
        search_reference_trial_id="T081",
        prior_decision_trial_ids=["T108", "T109", "T110"],
    )

    assert result.status == "PASS"
    assert result.generated_trials == ["T121"]
    assert planner.observed_strategy is not None
    assert planner.observed_strategy["recommendation"] == "stop_t088_neighborhood"
    assert result.decisions[-1]["strategy_context"]["recommendation"] == "stop_t088_neighborhood"
    trial = json.loads((tmp_path / "trials/T121.json").read_text(encoding="utf-8"))
    assert trial["sampler_schedule_shape"] == "linear"
    assert trial["sampler_selection_policy"] == "first"


def test_sampler_loop_rejects_invalid_planner_output_before_writing(tmp_path: Path) -> None:
    with pytest.raises(SamplerLoopError, match="planner failed"):
        run_incremental_sampler_loop(
            seed_trial_path=seed_trial(tmp_path),
            repo_root=tmp_path,
            max_candidates=1,
            start_trial_id="T060",
            planner="llm",
            planner_client=InvalidPlanner(),
        )

    assert not (tmp_path / "trials/T060.json").exists()
