"""Local and Modal-aware orchestrator.

Dry-run mode validates a trial, runs local preflight gates, appends a canonical
ledger row, and returns a durable local call id. Modal mode is explicit and
fails as `INFRA_FAIL` when the Modal SDK is unavailable.
"""

from __future__ import annotations

from pathlib import Path

from autoalphafold3.ledger import append_ledger, latest_result_for_trial, read_ledger
from autoalphafold3.modal_app import APP_NAME
from autoalphafold3.preflight import run_preflight
from autoalphafold3.schema import AutoFoldResult, FoldCartographerReport, TrialStatus

LOCAL_CALL_PREFIX = "dryrun"
MODAL_CALL_PREFIX = "modal"


def submit_trial(
    trial_path: str | Path,
    *,
    repo_root: str | Path = ".",
    ledger_path: str | Path = "runs/ledger.jsonl",
    changed_paths: list[str] | None = None,
    manifest_paths: dict[str, str] | None = None,
    mode: str = "dry_run",
    enforce_git_diff: bool = False,
) -> str:
    """Run preflight and submit a trial in dry-run or explicit Modal mode.

    Dry-run return values use `dryrun:T###`. Modal return values use
    `modal:<function_call_id>` when submission succeeds.
    """

    preflight = run_preflight(
        trial_path,
        repo_root=repo_root,
        changed_paths=changed_paths,
        manifest_paths=manifest_paths,
        enforce_git_diff=enforce_git_diff,
    )
    if mode == "modal":
        return _submit_modal(preflight.trial.model_dump(mode="json"), repo_root=repo_root, ledger_path=ledger_path)
    if mode != "dry_run":
        raise ValueError(f"unsupported submit mode: {mode}")

    metrics_json = preflight.scorer_metrics
    result = AutoFoldResult(
        trial_id=preflight.trial.trial_id,
        status=TrialStatus.PREFLIGHT_PASSED,
        candidate_id="local_dry_run",
        metrics=metrics_json["metrics"],
        fold_cartographer=FoldCartographerReport.model_validate(metrics_json["fold_cartographer"]),
        artifacts={key: str(value) for key, value in metrics_json["artifacts"].items()},
        postmortem="Local dry-run preflight passed; Modal was not called.",
    )
    append_ledger(result, ledger_path=Path(repo_root) / ledger_path, dedupe=True)
    return f"{LOCAL_CALL_PREFIX}:{preflight.trial.trial_id}"


def _submit_modal(trial_payload: dict[str, object], *, repo_root: str | Path, ledger_path: str | Path) -> str:
    try:
        import modal
    except ModuleNotFoundError:
        result = AutoFoldResult(
            trial_id=str(trial_payload["trial_id"]),
            status=TrialStatus.INFRA_FAIL,
            candidate_id="modal_submission",
            metrics={},
            fold_cartographer=FoldCartographerReport(signature="modal_sdk_missing"),
            failure_signature="modal_sdk_missing",
            postmortem="Modal submission requested, but the Modal SDK is not installed.",
        )
        append_ledger(result, ledger_path=Path(repo_root) / ledger_path, dedupe=True)
        return f"{MODAL_CALL_PREFIX}:INFRA_FAIL:{trial_payload['trial_id']}"

    fn = modal.Function.from_name(APP_NAME, "run_trial")
    call = fn.spawn(trial_payload)
    return f"{MODAL_CALL_PREFIX}:{call.object_id}"


def poll_trial(
    call_id: str,
    *,
    repo_root: str | Path = ".",
    ledger_path: str | Path = "runs/ledger.jsonl",
) -> AutoFoldResult:
    """Return the ledger row for a local dry-run call id."""

    prefix, _, trial_id = call_id.partition(":")
    if prefix == MODAL_CALL_PREFIX:
        return _poll_modal(call_id, repo_root=repo_root, ledger_path=ledger_path)
    if prefix != LOCAL_CALL_PREFIX or not trial_id:
        raise ValueError(f"unsupported local call id: {call_id}")
    row = latest_result_for_trial(trial_id, ledger_path=Path(repo_root) / ledger_path)
    if row is not None:
        return row
    raise ValueError(f"no ledger row found for call id: {call_id}")


def _poll_modal(call_id: str, *, repo_root: str | Path, ledger_path: str | Path) -> AutoFoldResult:
    _, _, object_id = call_id.partition(":")
    if object_id.startswith("INFRA_FAIL:"):
        trial_id = object_id.split(":", 1)[1]
        for row in reversed(read_ledger(ledger_path=Path(repo_root) / ledger_path)):
            if row.trial_id == trial_id:
                return row
    try:
        import modal
    except ModuleNotFoundError:
        return AutoFoldResult(
            trial_id="UNKNOWN",
            status=TrialStatus.INFRA_FAIL,
            candidate_id="modal_poll",
            metrics={},
            fold_cartographer=FoldCartographerReport(signature="modal_sdk_missing"),
            failure_signature="modal_sdk_missing",
            postmortem="Modal poll requested, but the Modal SDK is not installed.",
        )
    call = modal.FunctionCall.from_id(object_id)
    payload = call.get(timeout=0)
    result = AutoFoldResult.model_validate(payload)
    append_ledger(result, ledger_path=Path(repo_root) / ledger_path, dedupe=True)
    return result


def record_trial_status(
    result: AutoFoldResult | dict[str, object],
    *,
    repo_root: str | Path = ".",
    ledger_path: str | Path = "runs/ledger.jsonl",
) -> AutoFoldResult:
    """Validate and append one lifecycle transition with duplicate protection."""

    row = result if isinstance(result, AutoFoldResult) else AutoFoldResult.model_validate(result)
    append_ledger(row, ledger_path=Path(repo_root) / ledger_path, dedupe=True)
    return row


def cancel_trial(call_id: str) -> AutoFoldResult:
    """Represent local cancellation without touching external infrastructure."""

    prefix, _, trial_id = call_id.partition(":")
    if prefix == MODAL_CALL_PREFIX:
        return AutoFoldResult(
            trial_id="UNKNOWN",
            status=TrialStatus.INFRA_FAIL,
            candidate_id="modal_cancel",
            metrics={},
            fold_cartographer=FoldCartographerReport(signature="modal_cancel_not_implemented_local"),
            failure_signature="modal_cancel_not_implemented_local",
            postmortem="Modal cancellation is reserved for deployed infrastructure mode.",
        )
    if prefix != LOCAL_CALL_PREFIX or not trial_id:
        raise ValueError(f"unsupported local call id: {call_id}")
    return AutoFoldResult(
        trial_id=trial_id,
        status=TrialStatus.INFRA_FAIL,
        candidate_id="local_dry_run",
        metrics={},
        fold_cartographer=FoldCartographerReport(signature="local_cancelled"),
        failure_signature="local_cancelled",
        postmortem="Local dry-run call was cancelled before any Modal submission.",
    )
