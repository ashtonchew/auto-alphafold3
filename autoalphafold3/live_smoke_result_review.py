"""Offline review for completed bounded live-smoke evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

SCHEMA_VERSION = "autoaf3.live_smoke_result_review.v1"


class LiveSmokeResultReviewError(RuntimeError):
    """Raised when completed live-smoke evidence is missing or unsafe."""


@dataclass(frozen=True)
class LiveSmokeResultReview:
    """JSON-friendly decision from one completed bounded live smoke."""

    schema_version: str
    status: str
    decision: str
    reviewed_run_dir: str
    reviewed_trial_id: str
    candidate_id: str
    smoke_status: str
    result_status: str | None
    candidate_score: float | None
    global_baseline_delta: float | None
    provisional_keep: bool
    promotion_status: str | None
    failure_signature: str | None
    required_objectives: list[dict[str, object]]
    roadmap: list[dict[str, object]]
    may_start_live_candidate: bool
    may_start_open_ended_loop: bool
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def review_live_smoke_result(
    *,
    repo_root: str | Path = ".",
    live_smoke_run_dir: str | Path,
) -> LiveSmokeResultReview:
    """Review one completed bounded live smoke without authorizing new spend."""

    root = Path(repo_root)
    run_dir = _safe_run_dir(root=root, path=live_smoke_run_dir)
    summary = _read_json(run_dir / "summary.json", label="live smoke summary")
    _refuse_authority_claims(summary, label="live smoke summary")
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise LiveSmokeResultReviewError("live smoke summary must contain exactly one candidate")
    candidate = candidates[0]
    trial_id = _require_text(candidate.get("trial_id"), "trial_id")
    candidate_id = _require_text(candidate.get("candidate_id"), "candidate_id")
    candidate_dir = run_dir / "candidates" / trial_id
    if not candidate_dir.exists() or candidate_dir.is_symlink():
        raise LiveSmokeResultReviewError("live smoke candidate directory is missing or unsafe")
    decision_path = _candidate_relative_path(candidate, "decision_path", root=root)
    if decision_path != candidate_dir / "decision.json":
        raise LiveSmokeResultReviewError("live smoke decision_path does not match candidate directory")
    decision_payload = _read_json(decision_path, label="live smoke decision")
    _refuse_authority_claims(decision_payload, label="live smoke decision")
    if decision_payload.get("trial_id") != trial_id:
        raise LiveSmokeResultReviewError("live smoke decision trial_id mismatch")
    smoke_status = _require_text(decision_payload.get("status"), "decision status")
    metrics_payload = _optional_json(candidate_dir / "metrics.json")
    error_payload = _optional_json(candidate_dir / "error_report.json")
    if metrics_payload is not None:
        _refuse_authority_claims(metrics_payload, label="live smoke metrics")
    if error_payload is not None:
        _refuse_authority_claims(error_payload, label="live smoke error report")
    result_status = (
        str(metrics_payload.get("result_status"))
        if isinstance(metrics_payload, dict) and metrics_payload.get("result_status") is not None
        else None
    )
    candidate_score = _optional_float(
        metrics_payload.get("comparison", {}).get("candidate_score")
        if isinstance(metrics_payload, dict) and isinstance(metrics_payload.get("comparison"), dict)
        else None
    )
    global_delta = _optional_float(decision_payload.get("global_baseline_delta"))
    provisional_keep = bool(decision_payload.get("provisional_keep") is True)
    promotion_status = (
        str(decision_payload.get("promotion_status"))
        if decision_payload.get("promotion_status") is not None
        else None
    )
    failure_signature = _failure_signature(error_payload=error_payload, decision_payload=decision_payload)
    decision = _decision(
        smoke_status=smoke_status,
        provisional_keep=provisional_keep,
        result_status=result_status,
    )
    return LiveSmokeResultReview(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision=decision,
        reviewed_run_dir=str(Path(live_smoke_run_dir)),
        reviewed_trial_id=trial_id,
        candidate_id=candidate_id,
        smoke_status=smoke_status,
        result_status=result_status,
        candidate_score=candidate_score,
        global_baseline_delta=global_delta,
        provisional_keep=provisional_keep,
        promotion_status=promotion_status,
        failure_signature=failure_signature,
        required_objectives=_required_objectives(decision=decision),
        roadmap=_roadmap(decision=decision),
        may_start_live_candidate=False,
        may_start_open_ended_loop=False,
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _safe_run_dir(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise LiveSmokeResultReviewError("live smoke run dir must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise LiveSmokeResultReviewError("live smoke run dir must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise LiveSmokeResultReviewError(f"live smoke run dir must not be a symlink: {path}")
    if not full.exists():
        raise LiveSmokeResultReviewError(f"live smoke run dir does not exist: {path}")
    return full


def _candidate_relative_path(candidate: dict[str, object], key: str, *, root: Path) -> Path:
    value = candidate.get(key)
    if not isinstance(value, str) or not value:
        raise LiveSmokeResultReviewError(f"live smoke candidate missing {key}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise LiveSmokeResultReviewError(f"live smoke candidate {key} must be repo-relative")
    full = root / path
    if full.is_symlink():
        raise LiveSmokeResultReviewError(f"live smoke candidate {key} must not be a symlink")
    return full


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise LiveSmokeResultReviewError(f"{label} must not be a symlink")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveSmokeResultReviewError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise LiveSmokeResultReviewError(f"{label} must be a JSON object")
    return payload


def _optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return _read_json(path, label=path.name)


def _refuse_authority_claims(payload: dict[str, object], *, label: str) -> None:
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise LiveSmokeResultReviewError(f"{label} must not claim {key}=true")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LiveSmokeResultReviewError(f"live smoke evidence missing {name}")
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _failure_signature(*, error_payload: dict[str, object] | None, decision_payload: dict[str, object]) -> str | None:
    if error_payload is not None and isinstance(error_payload.get("error_report"), dict):
        signature = error_payload["error_report"].get("failure_signature")
        if isinstance(signature, str) and signature:
            return signature
    reason = decision_payload.get("reason")
    return str(reason) if isinstance(reason, str) and reason else None


def _decision(*, smoke_status: str, provisional_keep: bool, result_status: str | None) -> str:
    if provisional_keep:
        return "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_PROVISIONAL_KEEP"
    if smoke_status == "DISCARD" and result_status == "SCORED":
        return "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED"
    if smoke_status in {"FAIL", "INFRA_FAIL"}:
        return "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_FAILED"
    raise LiveSmokeResultReviewError(f"unsupported live smoke terminal status: {smoke_status}")


def _required_objectives(*, decision: str) -> list[dict[str, object]]:
    if decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_PROVISIONAL_KEEP":
        return [
            {
                "name": "run_falsification_gate",
                "status": "required",
                "objective": "Run Falsification Gate controls for the provisional live-smoke KEEP before any discovery or open-ended bench claim.",
                "evidence_required": "confirmed gate verdict written by trusted orchestrator; no Discovery Ledger entry before confirmation",
            }
        ]
    if decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED":
        return [
            {
                "name": "post_smoke_strategy_review",
                "status": "required",
                "objective": "Use the scored DISCARD evidence to define the next bounded strategy before more live spend.",
                "evidence_required": "new offline strategy artifact cites the scored smoke metrics, failure modes, candidate limit, and stop conditions",
            }
        ]
    return [
        {
            "name": "diagnose_live_smoke_failure",
            "status": "required",
            "objective": "Diagnose the live-smoke failure before retrying or spending on another live candidate.",
            "evidence_required": "failure diagnosis artifact with root cause, retry policy, and bounded next step",
        }
    ]


def _roadmap(*, decision: str) -> list[dict[str, object]]:
    if decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_PROVISIONAL_KEEP":
        return [
            {"step": "stop_open_ended_bench", "status": "required", "action": "Keep open-ended bench blocked."},
            {"step": "run_falsification_gate", "status": "required", "action": "Run controls for the provisional KEEP."},
            {"step": "confirm_or_discard", "status": "pending", "action": "Only confirmed mechanisms may enter the Discovery Ledger."},
        ]
    if decision == "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED":
        return [
            {"step": "stop_live_spend", "status": "required", "action": "Do not run another live candidate from the current gate."},
            {"step": "write_post_smoke_strategy", "status": "required", "action": "Choose the next bounded strategy from the scored DISCARD evidence."},
            {"step": "reopen_bench_gate", "status": "pending", "action": "Rerun bench readiness with the post-smoke strategy evidence."},
        ]
    return [
        {"step": "stop_live_spend", "status": "required", "action": "Do not retry live execution until the failure is diagnosed."},
        {"step": "diagnose_failure", "status": "required", "action": "Classify candidate-local versus infrastructure failure."},
        {"step": "reopen_live_gate", "status": "pending", "action": "Only a new gate may approve another bounded smoke."},
    ]
