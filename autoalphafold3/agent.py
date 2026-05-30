"""CLI entrypoint for local auto-AlphaFold3 agent operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoalphafold3.orchestrator import poll_trial, submit_trial
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
    return 2


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
