"""Artifact-only autoresearch candidate comparisons.

This module computes matched-budget and global-current-best deltas for
autoresearch candidate envelopes. It updates only candidate evidence artifacts;
it never appends the canonical ledger or Discovery Ledger.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from autoalphafold3.autoresearch_candidates import (
    CandidateEnvelope,
    write_candidate_decision,
    write_candidate_evidence,
    write_candidate_promotion_plan,
)
from autoalphafold3.baseline_readiness import current_best_from_baseline_and_ledger
from autoalphafold3.orchestrator import DEFAULT_KEEP_DELTA
from autoalphafold3.schema import AutoFoldResult, PRIMARY_METRIC, TrialStatus


class AutoresearchComparisonError(RuntimeError):
    """Raised when comparison evidence is insufficient or invalid."""


@dataclass(frozen=True)
class AutoresearchComparisonResult:
    """JSON-friendly comparison result for one candidate decision."""

    status: str
    candidate_score: float
    matched_budget_score: float | None
    matched_budget_delta: float | None
    global_current_best_score: float
    global_baseline_delta: float
    keep_threshold_delta: float
    provisional_keep: bool
    discovery_status: str = "UNCONFIRMED"
    writes_ledger: bool = False
    writes_discovery_ledger: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_and_write_candidate_decision(
    envelope: CandidateEnvelope,
    *,
    candidate_result: AutoFoldResult | dict[str, object],
    matched_budget_result: AutoFoldResult | dict[str, object] | None,
    repo_root: str | Path = ".",
    baseline_dir: str | Path = "runs/baseline",
    ledger_path: str | Path = "runs/ledger.jsonl",
    keep_delta: float = DEFAULT_KEEP_DELTA,
) -> AutoresearchComparisonResult:
    """Compare a scored candidate and write artifact-only decision evidence."""

    candidate = _validate_result(candidate_result)
    if candidate.trial_id != envelope.trial_id:
        raise AutoresearchComparisonError(
            f"candidate result trial_id {candidate.trial_id!r} does not match envelope trial_id {envelope.trial_id!r}"
        )
    _validate_envelope_paths(envelope, Path(repo_root))
    comparison = compare_candidate_result(
        candidate_result=candidate,
        matched_budget_result=matched_budget_result,
        repo_root=repo_root,
        baseline_dir=baseline_dir,
        ledger_path=ledger_path,
        keep_delta=keep_delta,
    )
    write_candidate_evidence(envelope, metrics=_result_metrics_payload(candidate, comparison))
    promotion_plan_path = None
    if comparison.provisional_keep:
        write_candidate_promotion_plan(
            envelope,
            global_baseline_delta=comparison.global_baseline_delta,
            keep_threshold_delta=comparison.keep_threshold_delta,
            matched_budget_delta=comparison.matched_budget_delta,
        )
        promotion_plan_path = str(envelope.promotion_plan_path)
    write_candidate_decision(
        envelope,
        status=comparison.status,
        matched_budget_delta=comparison.matched_budget_delta,
        global_baseline_delta=comparison.global_baseline_delta,
        keep_threshold_delta=comparison.keep_threshold_delta,
        promotion_plan_path=promotion_plan_path,
        reason=_comparison_reason(comparison),
        postmortem=_comparison_postmortem(comparison),
    )
    return comparison


def compare_candidate_result(
    *,
    candidate_result: AutoFoldResult | dict[str, object],
    matched_budget_result: AutoFoldResult | dict[str, object] | None,
    repo_root: str | Path = ".",
    baseline_dir: str | Path = "runs/baseline",
    ledger_path: str | Path = "runs/ledger.jsonl",
    keep_delta: float = DEFAULT_KEEP_DELTA,
) -> AutoresearchComparisonResult:
    """Compute matched-budget and global-current-best comparison evidence."""

    if not math.isfinite(keep_delta) or keep_delta < 0:
        raise AutoresearchComparisonError("keep_delta must be finite and non-negative")
    candidate = _validate_result(candidate_result)
    if candidate.status != TrialStatus.SCORED:
        raise AutoresearchComparisonError("candidate comparison requires a SCORED result")
    candidate_score = _score(candidate)
    if candidate_score is None:
        raise AutoresearchComparisonError(f"candidate result is missing finite {PRIMARY_METRIC} in [0, 1]")
    matched_score = None
    if matched_budget_result is not None:
        matched = _validate_result(matched_budget_result)
        if matched.status != TrialStatus.SCORED:
            raise AutoresearchComparisonError("matched-budget comparison requires a SCORED result")
        matched_score = _score(matched)
        if matched_score is None:
            raise AutoresearchComparisonError(f"matched-budget result is missing finite {PRIMARY_METRIC} in [0, 1]")
    root = Path(repo_root)
    best = current_best_from_baseline_and_ledger(
        baseline_dir=_repo_relative_path(root, baseline_dir, "baseline_dir"),
        ledger_path=_repo_relative_path(root, ledger_path, "ledger_path"),
    )
    global_delta = candidate_score - best.score
    matched_delta = None if matched_score is None else candidate_score - matched_score
    status = "KEEP" if candidate_score > best.score + keep_delta else "DISCARD"
    return AutoresearchComparisonResult(
        status=status,
        candidate_score=candidate_score,
        matched_budget_score=matched_score,
        matched_budget_delta=matched_delta,
        global_current_best_score=best.score,
        global_baseline_delta=global_delta,
        keep_threshold_delta=keep_delta,
        provisional_keep=status == "KEEP",
    )


def _validate_result(result: AutoFoldResult | dict[str, object]) -> AutoFoldResult:
    return result if isinstance(result, AutoFoldResult) else AutoFoldResult.model_validate(result)


def _repo_relative_path(repo_root: Path, value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise AutoresearchComparisonError(f"{label} must be repo-relative")
    expected = Path("runs/baseline") if label == "baseline_dir" else Path("runs/ledger.jsonl")
    if path != expected:
        raise AutoresearchComparisonError(f"{label} must be {expected.as_posix()}")
    root = repo_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AutoresearchComparisonError(f"{label} must stay under repo root") from exc
    return resolved


def _validate_envelope_paths(envelope: CandidateEnvelope, repo_root: Path) -> None:
    root = repo_root.resolve()
    expected_run_root = root / "runs" / "autoresearch" / envelope.run_id
    expected_candidate_dir = expected_run_root / "candidates" / envelope.trial_id
    if envelope.root.resolve() != expected_run_root:
        raise AutoresearchComparisonError("candidate envelope root must be runs/autoresearch/<run_id>")
    if envelope.candidate_dir.resolve() != expected_candidate_dir:
        raise AutoresearchComparisonError(
            "candidate envelope directory must be runs/autoresearch/<run_id>/candidates/<trial_id>"
        )
    _refuse_symlink_components(root, expected_candidate_dir.relative_to(root))
    for path in (envelope.metrics_path, envelope.decision_path, envelope.postmortem_path):
        if path.is_symlink():
            raise AutoresearchComparisonError(f"candidate envelope path must not be a symlink: {path}")


def _refuse_symlink_components(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AutoresearchComparisonError(f"candidate envelope path must not contain symlinks: {current}")


def _score(result: AutoFoldResult) -> float | None:
    value = result.metrics.get(PRIMARY_METRIC)
    if not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not math.isfinite(score) or score < 0 or score > 1:
        return None
    return score


def _result_metrics_payload(
    result: AutoFoldResult | dict[str, object],
    comparison: AutoresearchComparisonResult,
) -> dict[str, object]:
    row = _validate_result(result)
    return {
        "schema_version": "autoaf3.autoresearch_comparison_metrics.v1",
        "trial_id": row.trial_id,
        "candidate_id": row.candidate_id,
        "official_benchmark_result": False,
        "primary_metric": PRIMARY_METRIC,
        "result_status": row.status.value,
        "metrics": dict(row.metrics),
        "fold_cartographer": row.fold_cartographer.model_dump(mode="json"),
        "candidate_artifacts": dict(row.artifacts),
        "comparison": comparison.to_dict(),
        "writes_ledger": False,
        "writes_discovery_ledger": False,
    }


def _comparison_reason(comparison: AutoresearchComparisonResult) -> str:
    if comparison.provisional_keep:
        return "candidate cleared global current-best threshold; provisional KEEP requires Falsification Gate"
    return "candidate did not clear global current-best threshold"


def _comparison_postmortem(comparison: AutoresearchComparisonResult) -> str:
    matched = (
        "matched-budget comparison unavailable"
        if comparison.matched_budget_delta is None
        else f"matched-budget delta {comparison.matched_budget_delta:+.6f}"
    )
    return (
        f"{matched}; global current-best delta {comparison.global_baseline_delta:+.6f}; "
        f"keep threshold {comparison.keep_threshold_delta:+.6f}. Discovery Ledger remains untouched."
    )
