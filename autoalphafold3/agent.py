"""CLI entrypoint for local auto-AlphaFold3 agent operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from autoalphafold3.orchestrator import poll_trial, submit_trial
from autoalphafold3.artifact_comparison import ArtifactComparisonError, compare_prediction_artifacts
from autoalphafold3.autoresearch_loop import AutoresearchLoopError, run_autoresearch_loop
from autoalphafold3.autoresearch_candidates import CandidateArtifactError
from autoalphafold3.baseline_lock import BaselineLockError, lock_baseline_from_scored_artifacts
from autoalphafold3.baseline_runner import BaselineRunError, run_baseline
from autoalphafold3.bench_readiness_review import BenchReadinessReviewError, review_bench_readiness
from autoalphafold3.broader_strategy_review import BroaderStrategyReviewError, review_broader_strategy
from autoalphafold3.candidate_implementation_review import (
    CandidateImplementationReviewError,
    review_candidate_implementation,
)
from autoalphafold3.checkpoint_runner import CheckpointRunError, run_one_batch_checkpoint
from autoalphafold3.evidence_bridge_review import EvidenceBridgeReviewError, review_evidence_bridge
from autoalphafold3.gate_calibration import GateCalibrationError, calibrate_gate
from autoalphafold3.gate_calibration_runner import GateCalibrationRunError, run_gate_calibration
from autoalphafold3.local_fixtures import LocalFixtureError, materialize_local_nanofold_fixture
from autoalphafold3.llm_policy import DEFAULT_LLM_MODEL, AgentSearchPhase, default_llm_phase_policies, default_llm_phase_policy
from autoalphafold3.live_smoke_gate import LiveSmokeGateError, review_live_smoke_gate
from autoalphafold3.modal_authority import ModalAuthorityError, audit_modal_event_authority
from autoalphafold3.modal_trial_artifacts import (
    ModalTrialArtifactError,
    fetch_modal_trial_artifacts,
)
from autoalphafold3.next_surface_review import NextSurfaceReviewError, review_next_surface
from autoalphafold3.post_discard_diagnosis import (
    PostDiscardDiagnosisError,
    diagnose_post_discard_evidence,
)
from autoalphafold3.post_exhaustion_strategy import (
    PostExhaustionStrategyError,
    design_post_exhaustion_strategy,
)
from autoalphafold3.prediction_geometry import PredictionGeometryError, audit_prediction_geometry
from autoalphafold3.readiness import build_readiness_report, readiness_exit_code
from autoalphafold3.scorer_sensitivity import ScorerSensitivityError, run_scorer_sensitivity
from autoalphafold3.sampler_loop import APPROVAL_TEXT as SAMPLER_LOOP_APPROVAL_TEXT
from autoalphafold3.sampler_loop import SamplerLoopError, run_incremental_sampler_loop
from autoalphafold3.short_training_runner import ShortTrainingRunError, run_short_training
from autoalphafold3.strategy_exhaustion_audit import (
    StrategyExhaustionAuditError,
    audit_strategy_exhaustion,
)
from autoalphafold3.surface_design_review import SurfaceDesignReviewError, review_surface_design
from autoalphafold3.surface_strategy_review import SurfaceStrategyReviewError, review_surface_strategy
from autoalphafold3.modal_assets import (
    ModalAssetAuditError,
    audit_modal_assets,
    require_search_ready_assets,
)
from autoalphafold3.scorer.locked_dataset import validate_manifest_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m autoalphafold3.agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("trial_path")
    submit_parser.add_argument("--repo-root", default=".")
    submit_parser.add_argument("--ledger-path", default="runs/ledger.jsonl")
    submit_parser.add_argument("--manifest", action="append", default=[], help="name=path manifest mapping")
    submit_parser.add_argument("--mode", choices=("dry_run", "modal"), default="dry_run")
    submit_parser.add_argument("--enforce-git-diff", action="store_true")
    submit_parser.add_argument("--strict-preflight", action="store_true")

    poll_parser = subparsers.add_parser("poll")
    poll_parser.add_argument("call_id")
    poll_parser.add_argument("--repo-root", default=".")
    poll_parser.add_argument("--ledger-path", default="runs/ledger.jsonl")

    manifest_parser = subparsers.add_parser("validate-manifest")
    manifest_parser.add_argument("manifest", nargs="+")
    manifest_parser.add_argument("--repo-root", default=".")
    manifest_parser.add_argument("--no-verify-assets", action="store_true")
    manifest_parser.add_argument("--allow-empty", action="store_true")

    modal_assets_parser = subparsers.add_parser("audit-modal-assets")
    modal_assets_parser.add_argument("--env", default=None)
    modal_assets_parser.add_argument("--data-volume", default="autoalphafold3-data")
    modal_assets_parser.add_argument("--locked-volume", default="autoalphafold3-locked")
    modal_assets_parser.add_argument("--search-ready", action="store_true")

    readiness_parser = subparsers.add_parser("readiness-report")
    readiness_parser.add_argument("--repo-root", default=".")
    readiness_parser.add_argument("--baseline-dir", default="runs/baseline")
    readiness_parser.add_argument("--config-path", default="configs/nanofold_dev_cpu_smoke.json")
    readiness_parser.add_argument("--calibration-path", default="runs/falsification_gate_calibration.json")
    readiness_parser.add_argument("--modal-authority-path", default="runs/modal_event_authority.json")
    readiness_parser.add_argument("--pending-human-calibration-action", default=None)
    readiness_parser.add_argument("--include-live-smoke", action="store_true")
    readiness_parser.add_argument("--human-approved-live-smoke-action", default=None)

    baseline_lock_parser = subparsers.add_parser("lock-baseline")
    baseline_lock_parser.add_argument("--source-dir", required=True)
    baseline_lock_parser.add_argument("--feature-fingerprints", required=True)
    baseline_lock_parser.add_argument("--baseline-dir", default="runs/baseline")
    baseline_lock_parser.add_argument("--approve", required=True)
    baseline_lock_parser.add_argument("--dry-run", action="store_true")

    baseline_run_parser = subparsers.add_parser("run-baseline")
    baseline_run_parser.add_argument("--repo-root", default=".")
    baseline_run_parser.add_argument("--trial-id", default="T000")
    baseline_run_parser.add_argument("--source-dir", default="runs/trials/T000")
    baseline_run_parser.add_argument("--mode", choices=("dry-run", "modal"), default="dry-run")
    baseline_run_parser.add_argument("--modal-env", default=None)
    baseline_run_parser.add_argument("--approve", default=None)

    checkpoint_run_parser = subparsers.add_parser("run-one-batch-checkpoint")
    checkpoint_run_parser.add_argument("--repo-root", default=".")
    checkpoint_run_parser.add_argument("--trial-id", default="T010")
    checkpoint_run_parser.add_argument("--source-dir", default=None)
    checkpoint_run_parser.add_argument("--config-path", default="configs/nanofold_dev_cpu_smoke.json")
    checkpoint_run_parser.add_argument("--features-path", default="nanofold_event_small_no_templates.arrow")
    checkpoint_run_parser.add_argument("--mode", choices=("dry-run", "modal"), default="dry-run")
    checkpoint_run_parser.add_argument("--modal-env", default=None)
    checkpoint_run_parser.add_argument("--approve", default=None)

    short_training_parser = subparsers.add_parser("run-short-training")
    short_training_parser.add_argument("--repo-root", default=".")
    short_training_parser.add_argument("--trial", required=True)
    short_training_parser.add_argument("--source-dir", default=None)
    short_training_parser.add_argument("--features-dir", default="data/toy/nanofold_fixture")
    short_training_parser.add_argument("--features-path", default=None)
    short_training_parser.add_argument("--mode", choices=("dry-run", "local-fixture", "modal"), default="dry-run")
    short_training_parser.add_argument("--modal-env", default=None)
    short_training_parser.add_argument("--approve", default=None)

    autoresearch_loop_parser = subparsers.add_parser("autoresearch-loop")
    autoresearch_loop_parser.add_argument("--repo-root", default=".")
    autoresearch_loop_parser.add_argument("--run-id", required=True)
    autoresearch_loop_parser.add_argument("--mode", choices=("dry-run", "modal"), default="dry-run")
    autoresearch_loop_parser.add_argument(
        "--planner",
        choices=(
            "manual",
            "deterministic",
            "targeted_diagnostic",
            "schedule_diagnostic",
            "capacity_diagnostic",
            "topology_recycling_diagnostic",
            "feature_curriculum_diagnostic",
            "coordinate_scale_locality_diagnostic",
            "coordinate_normalized_sampler_diagnostic",
            "calibrated_coordinate_normalized_sampler_diagnostic",
            "calibrated_sampler_locality_selection_diagnostic",
            "calibrated_sampler_low_noise_diagnostic",
            "diffusion_data_scale_diagnostic",
            "pairformer_attention_diagnostic",
            "auxiliary_contact_loss_diagnostic",
            "feature_ref_pos_scale_diagnostic",
            "gradient_checkpointing_runtime_diagnostic",
            "diffusion_initialization_scale_diagnostic",
            "evidence_guided_failure_mode_bridge_diagnostic",
            "llm",
        ),
        default="deterministic",
    )
    autoresearch_loop_parser.add_argument("--start-trial-id", default="T120")
    autoresearch_loop_parser.add_argument("--max-candidates", type=int, default=None)
    autoresearch_loop_parser.add_argument("--candidate-plan", default=None)
    autoresearch_loop_parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    autoresearch_loop_parser.add_argument("--modal-env", default=None)
    autoresearch_loop_parser.add_argument("--approve", default=None)
    autoresearch_loop_parser.add_argument("--failure-streak-limit", type=int, default=2)
    autoresearch_loop_parser.add_argument("--prior-run-id", action="append", default=[])
    autoresearch_loop_parser.add_argument("--candidate-budget", choices=("smoke", "trial"), default="smoke")
    autoresearch_loop_parser.add_argument("--diagnostic-report", default=None)
    autoresearch_loop_parser.add_argument("--scorer-report", default=None)
    autoresearch_loop_parser.add_argument("--geometry-report", default=None)
    autoresearch_loop_parser.add_argument("--live-smoke-gate", default=None)

    compare_predictions_parser = subparsers.add_parser("compare-predictions")
    compare_predictions_parser.add_argument("left_predictions")
    compare_predictions_parser.add_argument("right_predictions")
    compare_predictions_parser.add_argument("--left-metrics", default=None)
    compare_predictions_parser.add_argument("--right-metrics", default=None)
    compare_predictions_parser.add_argument("--output", default=None)

    prediction_geometry_parser = subparsers.add_parser("prediction-geometry-audit")
    prediction_geometry_parser.add_argument("--prediction", action="append", required=True)
    prediction_geometry_parser.add_argument("--reference-predictions", default=None)
    prediction_geometry_parser.add_argument("--output", default=None)

    fetch_trial_artifacts_parser = subparsers.add_parser("fetch-modal-trial-artifacts")
    fetch_trial_artifacts_parser.add_argument("--trial-id", required=True)
    fetch_trial_artifacts_parser.add_argument("--artifact", action="append", required=True)
    fetch_trial_artifacts_parser.add_argument("--output-dir", default="runs/autoresearch/modal_artifacts")
    fetch_trial_artifacts_parser.add_argument("--modal-env", default=None)
    fetch_trial_artifacts_parser.add_argument("--volume", default="autoalphafold3-data")
    fetch_trial_artifacts_parser.add_argument("--force", action="store_true")

    scorer_sensitivity_parser = subparsers.add_parser("scorer-sensitivity")
    scorer_sensitivity_parser.add_argument("--trial-id", action="append", required=True)
    scorer_sensitivity_parser.add_argument("--mode", choices=("dry-run", "modal"), default="dry-run")
    scorer_sensitivity_parser.add_argument("--modal-env", default=None)
    scorer_sensitivity_parser.add_argument("--approve", default=None)
    scorer_sensitivity_parser.add_argument("--output", default=None)

    post_discard_parser = subparsers.add_parser("post-discard-diagnosis")
    post_discard_parser.add_argument("--repo-root", default=".")
    post_discard_parser.add_argument("--scorer-report", action="append", required=True)
    post_discard_parser.add_argument("--prediction-comparison", action="append", required=True)
    post_discard_parser.add_argument("--exhausted-surface", action="append", default=[])
    post_discard_parser.add_argument("--output", default=None)

    next_surface_parser = subparsers.add_parser("next-surface-review")
    next_surface_parser.add_argument("--repo-root", default=".")
    next_surface_parser.add_argument("--diagnosis", required=True)
    next_surface_parser.add_argument("--output", default=None)

    surface_strategy_parser = subparsers.add_parser("surface-strategy-review")
    surface_strategy_parser.add_argument("--repo-root", default=".")
    surface_strategy_parser.add_argument("--next-surface-review", action="append", required=True)
    surface_strategy_parser.add_argument("--diagnosis", action="append", default=[])
    surface_strategy_parser.add_argument("--output", default=None)

    surface_design_parser = subparsers.add_parser("surface-design-review")
    surface_design_parser.add_argument("--repo-root", default=".")
    surface_design_parser.add_argument("--strategy-review", required=True)
    surface_design_parser.add_argument("--proposed-surface", required=True)
    surface_design_parser.add_argument("--output", default=None)

    bench_readiness_parser = subparsers.add_parser("bench-readiness-review")
    bench_readiness_parser.add_argument("--repo-root", default=".")
    bench_readiness_parser.add_argument("--surface-strategy-review", required=True)
    bench_readiness_parser.add_argument("--broader-strategy-review", default=None)
    bench_readiness_parser.add_argument("--evidence-bridge-review", default=None)
    bench_readiness_parser.add_argument("--candidate-implementation-review", default=None)
    bench_readiness_parser.add_argument("--live-smoke-gate", default=None)
    bench_readiness_parser.add_argument("--baseline-dir", default="runs/baseline")
    bench_readiness_parser.add_argument("--config-path", default="configs/nanofold_dev_cpu_smoke.json")
    bench_readiness_parser.add_argument("--calibration-path", default="runs/falsification_gate_calibration.json")
    bench_readiness_parser.add_argument("--modal-authority-path", default="runs/modal_event_authority.json")
    bench_readiness_parser.add_argument("--output", default=None)

    broader_strategy_parser = subparsers.add_parser("broader-strategy-review")
    broader_strategy_parser.add_argument("--repo-root", default=".")
    broader_strategy_parser.add_argument("--surface-strategy-review", required=True)
    broader_strategy_parser.add_argument("--bench-readiness-review", required=True)
    broader_strategy_parser.add_argument("--output", default=None)

    strategy_exhaustion_parser = subparsers.add_parser("strategy-exhaustion-audit")
    strategy_exhaustion_parser.add_argument("--repo-root", default=".")
    strategy_exhaustion_parser.add_argument("--evidence-root", default="runs/autoresearch")
    strategy_exhaustion_parser.add_argument("--bench-readiness-review", default=None)
    strategy_exhaustion_parser.add_argument("--output", default=None)

    post_exhaustion_parser = subparsers.add_parser("post-exhaustion-strategy")
    post_exhaustion_parser.add_argument("--repo-root", default=".")
    post_exhaustion_parser.add_argument("--bench-readiness-review", required=True)
    post_exhaustion_parser.add_argument("--strategy-exhaustion-audit", required=True)
    post_exhaustion_parser.add_argument("--output", default=None)

    evidence_bridge_parser = subparsers.add_parser("evidence-bridge-review")
    evidence_bridge_parser.add_argument("--repo-root", default=".")
    evidence_bridge_parser.add_argument("--post-exhaustion-strategy", required=True)
    evidence_bridge_parser.add_argument("--candidate-run-dir", required=True)
    evidence_bridge_parser.add_argument("--output", default=None)

    candidate_implementation_parser = subparsers.add_parser("candidate-implementation-review")
    candidate_implementation_parser.add_argument("--repo-root", default=".")
    candidate_implementation_parser.add_argument("--evidence-bridge-review", required=True)
    candidate_implementation_parser.add_argument("--candidate", default="sampler_locality_guard")
    candidate_implementation_parser.add_argument("--output", default=None)

    live_smoke_gate_parser = subparsers.add_parser("live-smoke-gate")
    live_smoke_gate_parser.add_argument("--repo-root", default=".")
    live_smoke_gate_parser.add_argument("--candidate-implementation-review", required=True)
    live_smoke_gate_parser.add_argument("--baseline-dir", default="runs/baseline")
    live_smoke_gate_parser.add_argument("--config-path", default="configs/nanofold_dev_cpu_smoke.json")
    live_smoke_gate_parser.add_argument("--calibration-path", default="runs/falsification_gate_calibration.json")
    live_smoke_gate_parser.add_argument("--modal-authority-path", default="runs/modal_event_authority.json")
    live_smoke_gate_parser.add_argument("--output", default=None)

    sampler_loop_parser = subparsers.add_parser("autonomous-sampler-loop")
    sampler_loop_parser.add_argument("--repo-root", default=".")
    sampler_loop_parser.add_argument("--seed-trial", default="trials/T012.json")
    sampler_loop_parser.add_argument("--output-dir", default="trials")
    sampler_loop_parser.add_argument("--ledger-path", default="runs/ledger.jsonl")
    sampler_loop_parser.add_argument("--baseline-dir", default="runs/baseline")
    sampler_loop_parser.add_argument("--mode", choices=("dry-run", "modal"), default="dry-run")
    sampler_loop_parser.add_argument("--max-candidates", type=int, default=3)
    sampler_loop_parser.add_argument("--start-trial-id", default=None)
    sampler_loop_parser.add_argument("--poll-interval-s", type=float, default=2.0)
    sampler_loop_parser.add_argument("--per-candidate-timeout-s", type=int, default=180)
    sampler_loop_parser.add_argument("--failure-streak-limit", type=int, default=2)
    sampler_loop_parser.add_argument(
        "--planner",
        choices=("deterministic", "reference_sweep", "strategy_pivot", "llm"),
        default="deterministic",
    )
    sampler_loop_parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    sampler_loop_parser.add_argument("--search-reference-trial-id", default=None)
    sampler_loop_parser.add_argument("--prior-decision-trial-id", action="append", default=[])
    sampler_loop_parser.add_argument("--approve", default=None)

    calibrate_parser = subparsers.add_parser("calibrate-gate")
    calibrate_parser.add_argument("--repo-root", default=".")
    calibrate_parser.add_argument("--calibration-path", default="runs/falsification_gate_calibration.json")
    calibrate_parser.add_argument("--known-null-evidence", default=None)
    calibrate_parser.add_argument("--known-positive-evidence", default=None)
    calibrate_parser.add_argument("--mode", choices=("dry-run", "from-evidence"), default="dry-run")
    calibrate_parser.add_argument("--approve", default=None)

    gate_calibration_run_parser = subparsers.add_parser("run-gate-calibration")
    gate_calibration_run_parser.add_argument("--repo-root", default=".")
    gate_calibration_run_parser.add_argument("--evidence-dir", default="runs/gate_calibration")
    gate_calibration_run_parser.add_argument("--baseline-dir", default="runs/baseline")
    gate_calibration_run_parser.add_argument("--mode", choices=("dry-run", "modal"), default="dry-run")
    gate_calibration_run_parser.add_argument("--modal-env", default=None)
    gate_calibration_run_parser.add_argument("--approve", default=None)

    modal_authority_parser = subparsers.add_parser("audit-modal-authority")
    modal_authority_parser.add_argument("--repo-root", default=".")
    modal_authority_parser.add_argument("--authority-path", default="runs/modal_event_authority.json")
    modal_authority_parser.add_argument("--mode", choices=("dry-run", "modal"), default="dry-run")
    modal_authority_parser.add_argument("--modal-env", default=None)
    modal_authority_parser.add_argument("--approve", default=None)

    fixture_parser = subparsers.add_parser("materialize-local-fixture")
    fixture_parser.add_argument("--repo-root", default=".")
    fixture_parser.add_argument("--output-dir", default="data/toy/nanofold_fixture")
    fixture_parser.add_argument("--approve", required=True)
    fixture_parser.add_argument("--overwrite", action="store_true")

    llm_policy_parser = subparsers.add_parser("llm-policy")
    llm_policy_parser.add_argument(
        "--phase",
        choices=[phase.value for phase in AgentSearchPhase],
        default=None,
        help="Show only one autonomous-search LLM phase policy.",
    )
    llm_policy_parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    llm_policy_parser.add_argument(
        "--format",
        choices=("policy", "responses", "agents-sdk"),
        default="policy",
        help="Render the raw policy, Responses API kwargs, or dependency-free Agents SDK spec.",
    )

    args = parser.parse_args(argv)
    original_argv = list(sys.argv[1:] if argv is None else argv)
    if args.command == "submit":
        manifest_paths = _parse_manifest_args(args.manifest)
        call_id = submit_trial(
            args.trial_path,
            repo_root=args.repo_root,
            ledger_path=args.ledger_path,
            manifest_paths=manifest_paths,
            mode=args.mode,
            enforce_git_diff=args.enforce_git_diff or args.strict_preflight,
            strict_nanofold_gates=args.strict_preflight,
        )
        print(json.dumps({"call_id": call_id}, sort_keys=True))
        return 0
    if args.command == "poll":
        result = poll_trial(args.call_id, repo_root=args.repo_root, ledger_path=args.ledger_path)
        print(result.model_dump_json())
        return 0
    if args.command == "validate-manifest":
        reports = validate_manifest_files(
            args.manifest,
            repo_root=args.repo_root,
            verify_assets=not args.no_verify_assets,
            allow_empty=args.allow_empty,
        )
        print(json.dumps([report.to_dict() for report in reports], sort_keys=True))
        return 0
    if args.command == "audit-modal-assets":
        report = audit_modal_assets(
            data_volume=args.data_volume,
            locked_volume=args.locked_volume,
            env=args.env,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        if args.search_ready:
            try:
                require_search_ready_assets(report)
            except ModalAssetAuditError:
                return 1
        return 1 if report.status == "FAIL" else 0
    if args.command == "readiness-report":
        report = build_readiness_report(
            repo_root=args.repo_root,
            baseline_dir=args.baseline_dir,
            config_path=args.config_path,
            calibration_path=args.calibration_path,
            modal_authority_path=args.modal_authority_path,
            pending_human_calibration_action=args.pending_human_calibration_action,
            include_live_smoke=args.include_live_smoke,
            approved_live_smoke_action=args.human_approved_live_smoke_action,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return readiness_exit_code(report)
    if args.command == "lock-baseline":
        try:
            result = lock_baseline_from_scored_artifacts(
                source_dir=args.source_dir,
                feature_fingerprints_path=args.feature_fingerprints,
                baseline_dir=args.baseline_dir,
                approval=args.approve,
                dry_run=args.dry_run,
            )
        except BaselineLockError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        else:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0
    if args.command == "run-baseline":
        try:
            result = run_baseline(
                repo_root=args.repo_root,
                trial_id=args.trial_id,
                source_dir=args.source_dir,
                approval=args.approve,
                mode=args.mode,
                modal_env=args.modal_env,
            )
        except BaselineRunError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "run-one-batch-checkpoint":
        try:
            result = run_one_batch_checkpoint(
                repo_root=args.repo_root,
                trial_id=args.trial_id,
                source_dir=args.source_dir,
                config_path=args.config_path,
                features_path=args.features_path,
                approval=args.approve,
                mode=args.mode,
                modal_env=args.modal_env,
            )
        except CheckpointRunError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "run-short-training":
        try:
            result = run_short_training(
                trial_path=args.trial,
                repo_root=args.repo_root,
                source_dir=args.source_dir,
                features_dir=args.features_dir,
                features_path=args.features_path,
                approval=args.approve,
                mode=args.mode,
                modal_env=args.modal_env,
            )
        except ShortTrainingRunError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "autoresearch-loop":
        reexec_argv = _modal_venv_reexec_argv(args, original_argv)
        if reexec_argv is not None:
            os.execv(reexec_argv[0], reexec_argv)
        single_candidate_planners = {
            "llm",
            "targeted_diagnostic",
            "schedule_diagnostic",
            "evidence_guided_failure_mode_bridge_diagnostic",
        }
        try:
            result = run_autoresearch_loop(
                repo_root=args.repo_root,
                run_id=args.run_id,
                mode=args.mode,
                planner=args.planner,
                start_trial_id=args.start_trial_id,
                max_candidates=args.max_candidates
                if args.max_candidates is not None
                else (
                    1
                    if args.mode == "modal" or args.planner in single_candidate_planners
                    else 6
                ),
                candidate_plan=args.candidate_plan,
                approval=args.approve,
                model=args.model,
                modal_env=args.modal_env,
                failure_streak_limit=args.failure_streak_limit,
                prior_run_ids=args.prior_run_id,
                candidate_budget=args.candidate_budget,
                diagnostic_report=args.diagnostic_report,
                scorer_report=args.scorer_report,
                geometry_report=args.geometry_report,
                live_smoke_gate=args.live_smoke_gate,
            )
        except (AutoresearchLoopError, CandidateArtifactError, OSError, ValueError) as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "compare-predictions":
        try:
            result = compare_prediction_artifacts(
                left_predictions=args.left_predictions,
                right_predictions=args.right_predictions,
                left_metrics=args.left_metrics,
                right_metrics=args.right_metrics,
            )
        except ArtifactComparisonError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "prediction-geometry-audit":
        try:
            payload = audit_prediction_geometry(
                predictions=args.prediction,
                reference_predictions=args.reference_predictions,
            )
        except PredictionGeometryError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "fetch-modal-trial-artifacts":
        try:
            result = fetch_modal_trial_artifacts(
                trial_id=args.trial_id,
                artifacts=args.artifact,
                output_dir=args.output_dir,
                modal_env=args.modal_env,
                volume=args.volume,
                force=args.force,
            )
        except ModalTrialArtifactError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "scorer-sensitivity":
        reexec_argv = _modal_venv_reexec_argv(args, original_argv)
        if reexec_argv is not None:
            os.execv(reexec_argv[0], reexec_argv)
        try:
            result = run_scorer_sensitivity(
                trial_ids=args.trial_id,
                mode=args.mode,
                approval=args.approve,
                modal_env=args.modal_env,
            )
        except ScorerSensitivityError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "post-discard-diagnosis":
        try:
            result = diagnose_post_discard_evidence(
                repo_root=args.repo_root,
                scorer_reports=args.scorer_report,
                prediction_comparisons=args.prediction_comparison,
                exhausted_surfaces=args.exhausted_surface,
            )
        except PostDiscardDiagnosisError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "next-surface-review":
        try:
            result = review_next_surface(repo_root=args.repo_root, diagnosis_path=args.diagnosis)
        except NextSurfaceReviewError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "surface-strategy-review":
        try:
            result = review_surface_strategy(
                repo_root=args.repo_root,
                next_surface_reviews=args.next_surface_review,
                diagnoses=args.diagnosis,
            )
        except SurfaceStrategyReviewError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "surface-design-review":
        try:
            result = review_surface_design(
                repo_root=args.repo_root,
                strategy_review=args.strategy_review,
                proposed_surface=args.proposed_surface,
            )
        except SurfaceDesignReviewError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "bench-readiness-review":
        try:
            result = review_bench_readiness(
                repo_root=args.repo_root,
                surface_strategy_review=args.surface_strategy_review,
                broader_strategy_review=args.broader_strategy_review,
                evidence_bridge_review=args.evidence_bridge_review,
                candidate_implementation_review=args.candidate_implementation_review,
                live_smoke_gate=args.live_smoke_gate,
                baseline_dir=args.baseline_dir,
                config_path=args.config_path,
                calibration_path=args.calibration_path,
                modal_authority_path=args.modal_authority_path,
            )
        except BenchReadinessReviewError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "live-smoke-gate":
        try:
            result = review_live_smoke_gate(
                repo_root=args.repo_root,
                candidate_implementation_review=args.candidate_implementation_review,
                baseline_dir=args.baseline_dir,
                config_path=args.config_path,
                calibration_path=args.calibration_path,
                modal_authority_path=args.modal_authority_path,
            )
        except LiveSmokeGateError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "candidate-implementation-review":
        try:
            result = review_candidate_implementation(
                repo_root=args.repo_root,
                evidence_bridge_review=args.evidence_bridge_review,
                candidate=args.candidate,
            )
        except CandidateImplementationReviewError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "evidence-bridge-review":
        try:
            result = review_evidence_bridge(
                repo_root=args.repo_root,
                post_exhaustion_strategy=args.post_exhaustion_strategy,
                candidate_run_dir=args.candidate_run_dir,
            )
        except EvidenceBridgeReviewError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "broader-strategy-review":
        try:
            result = review_broader_strategy(
                repo_root=args.repo_root,
                surface_strategy_review=args.surface_strategy_review,
                bench_readiness_review=args.bench_readiness_review,
            )
        except BroaderStrategyReviewError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "strategy-exhaustion-audit":
        try:
            result = audit_strategy_exhaustion(
                repo_root=args.repo_root,
                evidence_root=args.evidence_root,
                bench_readiness_review=args.bench_readiness_review,
            )
        except StrategyExhaustionAuditError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "post-exhaustion-strategy":
        try:
            result = design_post_exhaustion_strategy(
                repo_root=args.repo_root,
                bench_readiness_review=args.bench_readiness_review,
                strategy_exhaustion_audit=args.strategy_exhaustion_audit,
            )
        except PostExhaustionStrategyError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        payload = result.to_dict()
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "autonomous-sampler-loop":
        try:
            result = run_incremental_sampler_loop(
                seed_trial_path=args.seed_trial,
                repo_root=args.repo_root,
                output_dir=args.output_dir,
                ledger_path=args.ledger_path,
                baseline_dir=args.baseline_dir,
                mode=args.mode,
                approval=args.approve,
                max_candidates=args.max_candidates,
                start_trial_id=args.start_trial_id,
                poll_interval_s=args.poll_interval_s,
                per_candidate_timeout_s=args.per_candidate_timeout_s,
                failure_streak_limit=args.failure_streak_limit,
                planner=args.planner,
                model=args.model,
                search_reference_trial_id=args.search_reference_trial_id,
                prior_decision_trial_ids=args.prior_decision_trial_id,
            )
        except SamplerLoopError as exc:
            expected = SAMPLER_LOOP_APPROVAL_TEXT if args.mode == "modal" else None
            print(json.dumps({"status": "FAIL", "error": str(exc), "approval": expected}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "calibrate-gate":
        try:
            result = calibrate_gate(
                repo_root=args.repo_root,
                calibration_path=args.calibration_path,
                known_null_evidence=args.known_null_evidence,
                known_positive_evidence=args.known_positive_evidence,
                approval=args.approve,
                mode=args.mode,
            )
        except GateCalibrationError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "run-gate-calibration":
        try:
            result = run_gate_calibration(
                repo_root=args.repo_root,
                evidence_dir=args.evidence_dir,
                baseline_dir=args.baseline_dir,
                approval=args.approve,
                mode=args.mode,
                modal_env=args.modal_env,
            )
        except GateCalibrationRunError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "audit-modal-authority":
        reexec_argv = _modal_venv_reexec_argv(args, original_argv)
        if reexec_argv is not None:
            os.execv(reexec_argv[0], reexec_argv)
        try:
            result = audit_modal_event_authority(
                repo_root=args.repo_root,
                authority_path=args.authority_path,
                approval=args.approve,
                mode=args.mode,
                modal_env=args.modal_env,
            )
        except ModalAuthorityError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "materialize-local-fixture":
        try:
            result = materialize_local_nanofold_fixture(
                repo_root=args.repo_root,
                output_dir=args.output_dir,
                approval=args.approve,
                overwrite=args.overwrite,
            )
        except LocalFixtureError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "llm-policy":
        if args.phase is None:
            policies = default_llm_phase_policies(model=args.model)
            payload = {
                phase.value: _render_llm_policy(policy, args.format)
                for phase, policy in policies.items()
            }
        else:
            policy = default_llm_phase_policy(args.phase, model=args.model)
            payload = _render_llm_policy(policy, args.format)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    return 2


def _modal_authority_venv_reexec_argv(args: argparse.Namespace, argv: list[str]) -> list[str] | None:
    """Return a repo-venv re-exec command when live Modal SDK is only there."""

    return _modal_venv_reexec_argv(args, argv)


def _modal_venv_reexec_argv(args: argparse.Namespace, argv: list[str]) -> list[str] | None:
    """Return a repo-venv re-exec command for live Modal commands."""

    if getattr(args, "command", None) not in {"audit-modal-authority", "autoresearch-loop", "scorer-sensitivity"}:
        return None
    if getattr(args, "mode", None) != "modal":
        return None
    if _current_python_can_import_modal():
        return None
    venv_python = Path(getattr(args, "repo_root", ".")) / ".venv" / "bin" / "python"
    if not venv_python.exists() or venv_python.absolute() == Path(sys.executable).absolute():
        return None
    if not _python_can_import_modal(venv_python):
        return None
    return [str(venv_python), "-m", "autoalphafold3.agent", *argv]


def _current_python_can_import_modal() -> bool:
    try:
        import modal  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _python_can_import_modal(python: Path) -> bool:
    result = subprocess.run(
        [str(python), "-c", "import modal"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _render_llm_policy(policy, output_format: str) -> dict[str, object]:
    if output_format == "responses":
        return policy.to_responses_create_kwargs()
    if output_format == "agents-sdk":
        return policy.to_agents_sdk_spec()
    return policy.model_dump(mode="json")


def _parse_manifest_args(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"manifest must use name=path form: {value}")
        parsed[name] = str(Path(path))
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
