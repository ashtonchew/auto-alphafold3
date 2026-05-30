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

    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        current_best: dict[str, object],
    ) -> SamplerCandidatePlan:
        latest_score = prior_decisions[-1].get("score") if prior_decisions else None
        self.observed_scores.append(float(latest_score) if isinstance(latest_score, int | float) else None)
        sampler_steps = 4 if latest_score is not None and latest_score < 0.42 else 1
        return SamplerCandidatePlan(
            diagnostic_target="local_geometry_weak",
            hypothesis=f"LLM-style planner chooses sampler candidate {candidate_index} after observing prior score.",
            intervention=f"Use sampler_steps={sampler_steps} from score-aware planner.",
            predicted_direction="up",
            expected_lddt_delta_band=[0.001, 0.01],
            sampler_steps=sampler_steps,
            seed=10 + candidate_index,
            rationale=f"latest_score={latest_score}; current_best={current_best.get('score')}",
        )


class InvalidPlanner:
    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        current_best: dict[str, object],
    ) -> SamplerCandidatePlan:
        return SamplerCandidatePlan(
            diagnostic_target="local_geometry_weak",
            hypothesis="Invalid planner tries to edit scorer labels and Modal GPU policy.",
            intervention="Change scorer labels and Modal GPU policy.",
            predicted_direction="up",
            expected_lddt_delta_band=[0.001, 0.01],
            sampler_steps=1,
            seed=0,
            rationale="This should fail validation.",
        )


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
    assert json.loads((tmp_path / "trials/T020.json").read_text())["sampler_steps"] == 1


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
    ledger = (tmp_path / "runs/ledger.jsonl").read_text(encoding="utf-8")
    assert '"status": "SCORED"' in ledger
    assert '"status": "KEEP"' in ledger


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
    assert json.loads((tmp_path / "trials/T050.json").read_text())["sampler_steps"] == 1
    assert json.loads((tmp_path / "trials/T051.json").read_text())["sampler_steps"] == 4
    assert result.decisions[1]["planner"] == "llm"


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
