"""CLI entrypoint for local auto-AlphaFold3 agent operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoalphafold3.orchestrator import poll_trial, submit_trial
from autoalphafold3.baseline_lock import BaselineLockError, lock_baseline_from_scored_artifacts
from autoalphafold3.baseline_runner import BaselineRunError, run_baseline
from autoalphafold3.gate_calibration import GateCalibrationError, calibrate_gate
from autoalphafold3.local_fixtures import LocalFixtureError, materialize_local_nanofold_fixture
from autoalphafold3.llm_policy import AgentSearchPhase, default_llm_phase_policies, default_llm_phase_policy
from autoalphafold3.readiness import build_readiness_report, readiness_exit_code
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

    calibrate_parser = subparsers.add_parser("calibrate-gate")
    calibrate_parser.add_argument("--repo-root", default=".")
    calibrate_parser.add_argument("--calibration-path", default="runs/falsification_gate_calibration.json")
    calibrate_parser.add_argument("--known-null-evidence", default=None)
    calibrate_parser.add_argument("--known-positive-evidence", default=None)
    calibrate_parser.add_argument("--mode", choices=("dry-run", "from-evidence"), default="dry-run")
    calibrate_parser.add_argument("--approve", default=None)

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
    llm_policy_parser.add_argument("--model", default="gpt-5.5")
    llm_policy_parser.add_argument(
        "--format",
        choices=("policy", "responses", "agents-sdk"),
        default="policy",
        help="Render the raw policy, Responses API kwargs, or dependency-free Agents SDK spec.",
    )

    args = parser.parse_args(argv)
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
