"""Offline next-surface review after mixed post-discard evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

SCHEMA_VERSION = "autoaf3.next_surface_review.v1"
POST_DISCARD_SCHEMA = "autoaf3.post_discard_diagnosis.v1"


class NextSurfaceReviewError(RuntimeError):
    """Raised when next-surface evidence cannot be reviewed safely."""


@dataclass(frozen=True)
class NextSurfaceReview:
    """JSON-friendly local next-surface decision report."""

    schema_version: str
    status: str
    source_diagnosis: str
    source_verdict: str
    decision: str
    approved_next_surface: str | None
    rejected_surfaces: list[str]
    evidence_summary: dict[str, object]
    required_next_pr: dict[str, object]
    allowed_next_step: str
    stop_live_trial_budget: bool
    do_not_start_open_ended_loop: bool
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def review_next_surface(
    *,
    diagnosis_path: str | Path,
    repo_root: str | Path = ".",
) -> NextSurfaceReview:
    """Review post-discard diagnosis evidence and define the next safe surface."""

    root = Path(repo_root)
    path = Path(diagnosis_path)
    if not path.is_absolute():
        path = root / path
    payload = _read_diagnosis(path=path, label=str(diagnosis_path))
    verdict = str(payload["verdict"])
    recommendation = payload.get("recommendation") if isinstance(payload.get("recommendation"), dict) else {}
    exhausted = _string_list(payload.get("exhausted_surfaces"))
    score_summary = payload.get("score_summary") if isinstance(payload.get("score_summary"), dict) else {}
    artifact_summary = payload.get("artifact_summary") if isinstance(payload.get("artifact_summary"), dict) else {}
    candidate_scores = score_summary.get("candidate_scores") if isinstance(score_summary.get("candidate_scores"), dict) else {}
    per_target = (
        score_summary.get("per_target_delta_summary")
        if isinstance(score_summary.get("per_target_delta_summary"), dict)
        else {}
    )
    stop_live = recommendation.get("stop_live_trial_budget") is True
    do_not_loop = recommendation.get("do_not_start_open_ended_loop") is True
    all_negative = score_summary.get("all_candidate_per_target_deltas_negative") is True
    all_changed = artifact_summary.get("all_comparisons_changed") is True
    exact_collapse_broken = score_summary.get("candidate_scores_identical") is False
    feature_curriculum_exhausted = "feature_curriculum" in exhausted
    mixed_after_feature_curriculum = (
        verdict == "MIXED_EVIDENCE_REVIEW_REQUIRED"
        and stop_live
        and do_not_loop
        and all_negative
        and all_changed
        and exact_collapse_broken
        and feature_curriculum_exhausted
    )
    sampler_scale_exhausted = {
        "sampler_coordinate_scale",
        "sampler_geometry_selection",
        "sampler_low_noise",
    }.issubset(set(exhausted))
    mixed_after_sampler_scale = (
        verdict == "MIXED_EVIDENCE_REVIEW_REQUIRED"
        and stop_live
        and do_not_loop
        and all_changed
        and exact_collapse_broken
        and sampler_scale_exhausted
        and per_target.get("negative_delta_count", 0) > 0
        and per_target.get("positive_delta_count", 0) > 0
    )
    if mixed_after_feature_curriculum:
        decision = "APPROVE_OFFLINE_PLANNER_PR_ONLY"
        approved_surface = "coordinate_scale_locality_diagnostic"
        allowed_next_step = (
            "Implement one dry-run-only planner for a coordinate scale/locality diagnostic; "
            "merge source behavior and rerun readiness before any live candidate."
        )
        required_next_pr = {
            "planner": "coordinate_scale_locality_diagnostic",
            "candidate_limit": 1,
            "mode_before_merge": "dry-run",
            "candidate_budget": "trial",
            "must_consume_review": True,
            "forbidden_edits": [
                "scorer",
                "benchmark_contract",
                "public_manifests",
                "fingerprints",
                "Modal resources",
                "templates",
                "baselines",
                "canonical_ledger",
                "Discovery Ledger",
            ],
            "stop_conditions": [
                "dry-run envelope plans more than one candidate",
                "review verdict is not MIXED_EVIDENCE_REVIEW_REQUIRED",
                "candidate writes ledger or Discovery Ledger",
                "readiness is not autonomous_search_ready=true after merge",
            ],
        }
    elif mixed_after_sampler_scale:
        decision = "APPROVE_OFFLINE_PLANNER_PR_ONLY"
        approved_surface = "diffusion_data_scale_diagnostic"
        allowed_next_step = (
            "Implement one dry-run-only planner for a diffusion data-scale diagnostic; "
            "merge source behavior and rerun readiness before any live candidate."
        )
        required_next_pr = {
            "planner": "diffusion_data_scale_diagnostic",
            "candidate_limit": 1,
            "mode_before_merge": "dry-run",
            "candidate_budget": "trial",
            "must_consume_review": True,
            "forbidden_edits": [
                "scorer",
                "benchmark_contract",
                "public_manifests",
                "fingerprints",
                "Modal resources",
                "templates",
                "baselines",
                "canonical_ledger",
                "Discovery Ledger",
            ],
            "stop_conditions": [
                "dry-run envelope plans more than one candidate",
                "review verdict is not MIXED_EVIDENCE_REVIEW_REQUIRED",
                "candidate writes ledger or Discovery Ledger",
                "readiness is not autonomous_search_ready=true after merge",
            ],
        }
    else:
        decision = "NO_NEXT_SURFACE_APPROVED"
        approved_surface = None
        allowed_next_step = (
            "Do offline human/agent design review only; do not implement a planner until the next "
            "surface is explicit and non-overlapping with exhausted evidence."
        )
        required_next_pr = {
            "planner": None,
            "candidate_limit": 0,
            "mode_before_merge": "none",
            "must_consume_review": False,
            "forbidden_edits": [],
            "stop_conditions": ["no approved next surface"],
        }
    return NextSurfaceReview(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        source_diagnosis=str(diagnosis_path),
        source_verdict=verdict,
        decision=decision,
        approved_next_surface=approved_surface,
        rejected_surfaces=exhausted,
        evidence_summary={
            "reference_trial_id": payload.get("reference_trial_id"),
            "candidate_trial_ids": _string_list(payload.get("candidate_trial_ids")),
            "candidate_scores": {str(key): value for key, value in candidate_scores.items()},
            "all_candidate_per_target_deltas_negative": all_negative,
            "negative_delta_count": per_target.get("negative_delta_count"),
            "positive_delta_count": per_target.get("positive_delta_count"),
            "worst_target": per_target.get("worst_target"),
            "worst_delta": per_target.get("worst_delta"),
            "all_comparisons_changed": all_changed,
            "comparison_count": artifact_summary.get("comparison_count"),
        },
        required_next_pr=required_next_pr,
        allowed_next_step=allowed_next_step,
        stop_live_trial_budget=True,
        do_not_start_open_ended_loop=True,
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _read_diagnosis(*, path: Path, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise NextSurfaceReviewError(f"diagnosis must not be a symlink: {label}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NextSurfaceReviewError(f"cannot read diagnosis: {label}") from exc
    if not isinstance(payload, dict):
        raise NextSurfaceReviewError(f"diagnosis must be a JSON object: {label}")
    if payload.get("schema_version") != POST_DISCARD_SCHEMA:
        raise NextSurfaceReviewError("diagnosis schema_version mismatch")
    if payload.get("status") != "PASS":
        raise NextSurfaceReviewError("diagnosis must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise NextSurfaceReviewError(f"diagnosis must not claim {key}=true")
    return payload


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
