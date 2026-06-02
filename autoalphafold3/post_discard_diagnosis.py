"""Offline post-discard evidence diagnosis for autoresearch candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from autoalphafold3.schema import PRIMARY_METRIC

SCHEMA_VERSION = "autoaf3.post_discard_diagnosis.v1"


class PostDiscardDiagnosisError(RuntimeError):
    """Raised when post-discard evidence cannot be diagnosed safely."""


@dataclass(frozen=True)
class PostDiscardDiagnosis:
    """JSON-friendly local diagnosis for repeated discarded candidates."""

    schema_version: str
    status: str
    verdict: str
    reference_trial_id: str | None
    candidate_trial_ids: list[str]
    exhausted_surfaces: list[str]
    score_summary: dict[str, object]
    artifact_summary: dict[str, object]
    recommendation: dict[str, object]
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def diagnose_post_discard_evidence(
    *,
    scorer_reports: list[str | Path],
    prediction_comparisons: list[str | Path],
    exhausted_surfaces: list[str] | None = None,
    repo_root: str | Path = ".",
) -> PostDiscardDiagnosis:
    """Classify repeated DISCARD evidence without starting search or scoring."""

    if not scorer_reports:
        raise PostDiscardDiagnosisError("at least one scorer report is required")
    if not prediction_comparisons:
        raise PostDiscardDiagnosisError("at least one prediction comparison is required")

    root = Path(repo_root)
    scorer_payloads = [_read_json_evidence(root=root, path=path, label="scorer report") for path in scorer_reports]
    comparison_payloads = [
        _read_json_evidence(root=root, path=path, label="prediction comparison")
        for path in prediction_comparisons
    ]
    _validate_scorer_reports(scorer_payloads)
    _validate_prediction_comparisons(comparison_payloads)

    reference_trial_id = _reference_trial_id(scorer_payloads)
    candidate_scores = _candidate_primary_scores(scorer_payloads, reference_trial_id=reference_trial_id)
    candidate_trial_ids = sorted(candidate_scores)
    if not candidate_trial_ids:
        raise PostDiscardDiagnosisError("scorer reports did not include candidate scores")
    per_target_summary = _per_target_delta_summary(scorer_payloads)
    artifact_summary = _artifact_summary(comparison_payloads)
    score_values = [candidate_scores[trial_id] for trial_id in candidate_trial_ids]
    candidate_scores_identical = _all_identical(score_values)
    all_candidate_deltas_negative = per_target_summary["positive_delta_count"] == 0 and per_target_summary["negative_delta_count"] > 0

    if artifact_summary["any_all_predictions_identical"]:
        verdict = "STALE_PREDICTION_ARTIFACTS"
        next_goal = (
            "Pause live trial-budget spend and diagnose stale prediction artifacts, sampler determinism, "
            "or candidate patch ineffectiveness before another Modal candidate."
        )
        stop_live_trial_budget = True
    elif candidate_scores_identical and all_candidate_deltas_negative and artifact_summary["all_comparisons_changed"]:
        verdict = "SHORT_TRAINING_FAMILY_SCORER_COLLAPSE"
        next_goal = (
            "Pause live trial-budget spend and design a new allowed surface around short-training "
            "initialization, artifact scale, or feature/curriculum handling before another candidate."
        )
        stop_live_trial_budget = True
    else:
        verdict = "MIXED_EVIDENCE_REVIEW_REQUIRED"
        next_goal = (
            "Do not run an open-ended loop; review the mixed scorer/artifact evidence and define one "
            "bounded dry-run candidate only after the next surface is explicit."
        )
        stop_live_trial_budget = True

    return PostDiscardDiagnosis(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        verdict=verdict,
        reference_trial_id=reference_trial_id,
        candidate_trial_ids=candidate_trial_ids,
        exhausted_surfaces=list(exhausted_surfaces or []),
        score_summary={
            "primary_metric": PRIMARY_METRIC,
            "candidate_scores": candidate_scores,
            "candidate_scores_identical": candidate_scores_identical,
            "all_candidate_per_target_deltas_negative": all_candidate_deltas_negative,
            "per_target_delta_summary": per_target_summary,
        },
        artifact_summary=artifact_summary,
        recommendation={
            "stop_live_trial_budget": stop_live_trial_budget,
            "do_not_start_open_ended_loop": True,
            "next_goal": next_goal,
            "allowed_next_step": (
                "offline design review or dry-run planner implementation; no canonical ledger, "
                "Discovery Ledger, baseline, manifest, fingerprint, scorer, or Modal resource edits"
            ),
        },
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _read_json_evidence(*, root: Path, path: str | Path, label: str) -> dict[str, object]:
    evidence_path = Path(path)
    if not evidence_path.is_absolute():
        evidence_path = root / evidence_path
    if evidence_path.is_symlink():
        raise PostDiscardDiagnosisError(f"{label} must not be a symlink: {path}")
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostDiscardDiagnosisError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PostDiscardDiagnosisError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_scorer_reports(reports: list[dict[str, object]]) -> None:
    for report in reports:
        if report.get("schema_version") != "autoaf3.scorer_sensitivity.v1":
            raise PostDiscardDiagnosisError("scorer report schema_version mismatch")
        if report.get("status") != "PASS":
            raise PostDiscardDiagnosisError("scorer report must have status=PASS")
        _refuse_true_flag(report, "starts_search", "scorer report")
        _refuse_true_flag(report, "writes_ledger", "scorer report")
        _refuse_true_flag(report, "writes_discovery_ledger", "scorer report")


def _validate_prediction_comparisons(comparisons: list[dict[str, object]]) -> None:
    for comparison in comparisons:
        if comparison.get("schema_version") != "autoaf3.prediction_artifact_comparison.v1":
            raise PostDiscardDiagnosisError("prediction comparison schema_version mismatch")
        if comparison.get("same_split") is not True:
            raise PostDiscardDiagnosisError("prediction comparisons must use the same split")
        if comparison.get("same_target_set") is not True:
            raise PostDiscardDiagnosisError("prediction comparisons must use the same target set")


def _refuse_true_flag(payload: dict[str, object], key: str, label: str) -> None:
    if payload.get(key) is True:
        raise PostDiscardDiagnosisError(f"{label} must not claim {key}=true")


def _reference_trial_id(reports: list[dict[str, object]]) -> str | None:
    references = {str(report.get("reference_trial_id")) for report in reports if report.get("reference_trial_id")}
    if len(references) > 1:
        raise PostDiscardDiagnosisError("scorer reports disagree on reference_trial_id")
    return next(iter(references)) if references else None


def _candidate_primary_scores(
    reports: list[dict[str, object]],
    *,
    reference_trial_id: str | None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for report in reports:
        scored_trials = report.get("scored_trials")
        if not isinstance(scored_trials, list):
            raise PostDiscardDiagnosisError("scorer report missing scored_trials")
        for item in scored_trials:
            if not isinstance(item, dict):
                continue
            trial_id = item.get("trial_id")
            if not isinstance(trial_id, str) or trial_id == reference_trial_id:
                continue
            metrics = item.get("metrics")
            score = metrics.get(PRIMARY_METRIC) if isinstance(metrics, dict) else item.get("score")
            if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(float(score)):
                raise PostDiscardDiagnosisError(f"candidate {trial_id} missing finite {PRIMARY_METRIC}")
            scores[trial_id] = float(score)
    return scores


def _per_target_delta_summary(reports: list[dict[str, object]]) -> dict[str, object]:
    negative = 0
    positive = 0
    zero = 0
    candidate_count = 0
    target_ids: set[str] = set()
    worst_target: tuple[str, float] | None = None
    for report in reports:
        deltas = report.get("per_target_score_deltas_vs_reference")
        if not isinstance(deltas, dict):
            raise PostDiscardDiagnosisError("scorer report missing per-target score deltas")
        for _trial_id, per_target in deltas.items():
            if not isinstance(per_target, dict):
                continue
            candidate_count += 1
            for target_id, raw_delta in per_target.items():
                if isinstance(raw_delta, bool) or not isinstance(raw_delta, int | float):
                    continue
                delta = float(raw_delta)
                if delta < 0.0:
                    negative += 1
                elif delta > 0.0:
                    positive += 1
                else:
                    zero += 1
                target_ids.add(str(target_id))
                if worst_target is None or delta < worst_target[1]:
                    worst_target = (str(target_id), delta)
    return {
        "candidate_delta_sets": candidate_count,
        "target_count": len(target_ids),
        "negative_delta_count": negative,
        "positive_delta_count": positive,
        "zero_delta_count": zero,
        "worst_target": worst_target[0] if worst_target else None,
        "worst_delta": worst_target[1] if worst_target else None,
    }


def _artifact_summary(comparisons: list[dict[str, object]]) -> dict[str, object]:
    changed_counts: dict[str, int] = {}
    identical_counts: dict[str, int] = {}
    distance_means: dict[str, float] = {}
    any_identical = False
    all_changed = True
    for comparison in comparisons:
        label = _comparison_label(comparison)
        changed = comparison.get("changed_targets")
        identical = comparison.get("identical_targets")
        if not isinstance(changed, list) or not isinstance(identical, list):
            raise PostDiscardDiagnosisError("prediction comparison missing target change lists")
        changed_counts[label] = len(changed)
        identical_counts[label] = len(identical)
        any_identical = any_identical or comparison.get("all_predictions_identical") is True
        all_changed = all_changed and comparison.get("all_predictions_identical") is False and len(changed) > 0
        distance_summary = comparison.get("distance_delta_summary")
        if isinstance(distance_summary, dict):
            mean_delta = distance_summary.get("mean_target_mean_abs_pair_distance_delta")
            if isinstance(mean_delta, (int, float)) and not isinstance(mean_delta, bool):
                distance_means[label] = float(mean_delta)
    return {
        "comparison_count": len(comparisons),
        "changed_target_counts": changed_counts,
        "identical_target_counts": identical_counts,
        "mean_pair_distance_delta_by_comparison": distance_means,
        "any_all_predictions_identical": any_identical,
        "all_comparisons_changed": all_changed,
    }


def _comparison_label(comparison: dict[str, object]) -> str:
    left = comparison.get("left") if isinstance(comparison.get("left"), dict) else {}
    right = comparison.get("right") if isinstance(comparison.get("right"), dict) else {}
    left_id = left.get("trial_id") if isinstance(left, dict) else None
    right_id = right.get("trial_id") if isinstance(right, dict) else None
    return f"{left_id or 'left'}-vs-{right_id or 'right'}"


def _all_identical(values: list[float]) -> bool:
    if not values:
        return False
    reference = values[0]
    return all(math.isclose(value, reference, rel_tol=0.0, abs_tol=0.0) for value in values)
