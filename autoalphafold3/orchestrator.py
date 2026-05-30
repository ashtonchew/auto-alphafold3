"""Local and Modal-aware orchestrator.

Dry-run mode validates a trial, runs local preflight gates, appends a canonical
ledger row, and returns a durable local call id. Modal mode is explicit and
fails as `INFRA_FAIL` when the Modal SDK is unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path

from autoalphafold3.baseline_readiness import (
    BaselineReadinessError,
    current_best_from_baseline_and_ledger,
)
from autoalphafold3.ledger import append_ledger, latest_result_for_trial, read_ledger
from autoalphafold3.modal_app import APP_NAME
from autoalphafold3.preflight import run_preflight
from autoalphafold3.schema import AutoFoldResult, DiscoveryStatus, FoldCartographerReport, PRIMARY_METRIC, TrialStatus

LOCAL_CALL_PREFIX = "dryrun"
MODAL_CALL_PREFIX = "modal"
DEFAULT_KEEP_DELTA = 0.001


def submit_trial(
    trial_path: str | Path,
    *,
    repo_root: str | Path = ".",
    ledger_path: str | Path = "runs/ledger.jsonl",
    changed_paths: list[str] | None = None,
    manifest_paths: dict[str, str] | None = None,
    mode: str = "dry_run",
    enforce_git_diff: bool = False,
    strict_nanofold_gates: bool = False,
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
        strict_nanofold_gates=strict_nanofold_gates,
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

    try:
        runner_cls = modal.Cls.from_name(APP_NAME, "TrialRunner")
        runner = runner_cls()
        call = runner.run.spawn(trial_payload)
    except Exception as exc:  # noqa: BLE001 - external Modal failures normalize to INFRA_FAIL.
        result = modal_infra_failure_result(
            trial_id=str(trial_payload["trial_id"]),
            candidate_id="modal_submission",
            exc=exc,
        )
        append_ledger(result, ledger_path=Path(repo_root) / ledger_path, dedupe=True, validate_lifecycle=True)
        return f"{MODAL_CALL_PREFIX}:INFRA_FAIL:{trial_payload['trial_id']}"
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
    try:
        call = modal.FunctionCall.from_id(object_id)
        payload = call.get(timeout=0)
        result = AutoFoldResult.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - polling failures are infrastructure failures.
        result = modal_infra_failure_result(
            trial_id="UNKNOWN",
            candidate_id="modal_poll",
            exc=exc,
        )
    append_ledger(result, ledger_path=Path(repo_root) / ledger_path, dedupe=True, validate_lifecycle=True)
    return result


def record_trial_status(
    result: AutoFoldResult | dict[str, object],
    *,
    repo_root: str | Path = ".",
    ledger_path: str | Path = "runs/ledger.jsonl",
) -> AutoFoldResult:
    """Validate and append one lifecycle transition with duplicate protection."""

    row = result if isinstance(result, AutoFoldResult) else AutoFoldResult.model_validate(result)
    append_ledger(row, ledger_path=Path(repo_root) / ledger_path, dedupe=True, validate_lifecycle=True)
    return row


def decide_stage_one_result(
    scored: AutoFoldResult | dict[str, object],
    *,
    repo_root: str | Path = ".",
    baseline_dir: str | Path = "runs/baseline",
    ledger_path: str | Path = "runs/ledger.jsonl",
    keep_delta: float = DEFAULT_KEEP_DELTA,
) -> AutoFoldResult:
    """Decide provisional KEEP/DISCARD/FAIL/INFRA_FAIL from one scored result.

    This does not run the Falsification Gate and never writes the Discovery
    Ledger. A KEEP returned here is provisional and gate-required.
    """

    row = scored if isinstance(scored, AutoFoldResult) else AutoFoldResult.model_validate(scored)
    if row.status == TrialStatus.INFRA_FAIL:
        return row.model_copy(update={"discovery": DiscoveryStatus.UNCONFIRMED})
    if row.status == TrialStatus.FAIL:
        return row.model_copy(update={"discovery": DiscoveryStatus.UNCONFIRMED})
    if row.status != TrialStatus.SCORED:
        return row.model_copy(
            update={
                "status": TrialStatus.FAIL,
                "discovery": DiscoveryStatus.UNCONFIRMED,
                "failure_signature": row.failure_signature or "stage_one_status_not_scored",
                "postmortem": row.postmortem or "Stage-one decision requires a SCORED result row.",
            }
        )

    score = _stage_one_score(row)
    if score is None:
        return row.model_copy(
            update={
                "status": TrialStatus.FAIL,
                "discovery": DiscoveryStatus.UNCONFIRMED,
                "failure_signature": row.failure_signature or "stage_one_score_missing",
                "postmortem": row.postmortem or "Stage-one decision requires best_val_calpha_lddt in [0, 1].",
            }
        )
    best = current_best_from_baseline_and_ledger(
        baseline_dir=Path(repo_root) / baseline_dir,
        ledger_path=Path(repo_root) / ledger_path,
    )
    if score > best.score + keep_delta:
        return row.model_copy(
            update={
                "status": TrialStatus.KEEP,
                "discovery": DiscoveryStatus.UNCONFIRMED,
                "falsification": None,
                "postmortem": row.postmortem or "Provisional KEEP; Falsification Gate evidence required before discovery.",
            }
        )
    return row.model_copy(
        update={
            "status": TrialStatus.DISCARD,
            "discovery": DiscoveryStatus.UNCONFIRMED,
            "falsification": None,
            "postmortem": row.postmortem or "Stage-one score did not clear current-best threshold.",
        }
    )


def record_stage_one_decision(
    scored: AutoFoldResult | dict[str, object],
    *,
    repo_root: str | Path = ".",
    baseline_dir: str | Path = "runs/baseline",
    ledger_path: str | Path = "runs/ledger.jsonl",
    keep_delta: float = DEFAULT_KEEP_DELTA,
) -> AutoFoldResult:
    """Decide and append a stage-one lifecycle row to the canonical ledger."""

    decision = decide_stage_one_result(
        scored,
        repo_root=repo_root,
        baseline_dir=baseline_dir,
        ledger_path=ledger_path,
        keep_delta=keep_delta,
    )
    append_ledger(decision, ledger_path=Path(repo_root) / ledger_path, dedupe=True, validate_lifecycle=True)
    return decision


def modal_infra_failure_result(*, trial_id: str, candidate_id: str, exc: BaseException) -> AutoFoldResult:
    """Normalize external Modal errors into the canonical INFRA_FAIL status."""

    signature = f"modal_{type(exc).__name__}"
    return AutoFoldResult(
        trial_id=trial_id,
        status=TrialStatus.INFRA_FAIL,
        candidate_id=candidate_id,
        metrics={},
        fold_cartographer=FoldCartographerReport(signature=signature),
        failure_signature=signature,
        postmortem=f"Modal infrastructure failure: {exc}",
    )


def _stage_one_score(row: AutoFoldResult) -> float | None:
    score = row.metrics.get(PRIMARY_METRIC)
    if isinstance(score, bool):
        return None
    if not isinstance(score, int | float):
        return None
    value = float(score)
    if not math.isfinite(value):
        return None
    if value < 0.0 or value > 1.0:
        return None
    return value


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
