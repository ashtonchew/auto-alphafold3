"""Offline strategy review for exhausted autoresearch surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

NEXT_SURFACE_SCHEMA = "autoaf3.next_surface_review.v1"
POST_DISCARD_SCHEMA = "autoaf3.post_discard_diagnosis.v1"
SCHEMA_VERSION = "autoaf3.surface_strategy_review.v1"


class SurfaceStrategyReviewError(RuntimeError):
    """Raised when strategy evidence cannot be reviewed safely."""


@dataclass(frozen=True)
class SurfaceStrategyReview:
    """JSON-friendly strategy decision after next-surface review."""

    schema_version: str
    status: str
    decision: str
    approved_next_surface: str | None
    approved_planner: str | None
    candidate_limit: int
    may_start_live_candidate: bool
    may_start_open_ended_loop: bool
    bench_blocked_reason: str | None
    consumed_next_surface_reviews: list[str]
    consumed_diagnoses: list[str]
    exhausted_surfaces: list[str]
    unimplemented_candidate_surfaces: list[str]
    required_next_step: str
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


IMPLEMENTED_PLANNER_SURFACES: dict[str, frozenset[str]] = {
    "targeted_diagnostic": frozenset({"local_geometry", "local_calpha_geometry_loss"}),
    "schedule_diagnostic": frozenset({"optimizer_schedule", "optimizer_scheduler"}),
    "capacity_diagnostic": frozenset({"width_depth", "model_capacity"}),
    "topology_recycling_diagnostic": frozenset({"recycling", "topology_recycling"}),
    "feature_curriculum_diagnostic": frozenset({"feature_curriculum", "reduced_crop_msa_curriculum"}),
    "coordinate_scale_locality_diagnostic": frozenset(
        {"coordinate_scale_locality", "coordinate_scale_locality_diagnostic"}
    ),
    "coordinate_normalized_sampler_diagnostic": frozenset(
        {"coordinate_normalized_sampler", "coordinate_normalized_sampler_diagnostic"}
    ),
    "calibrated_coordinate_normalized_sampler_diagnostic": frozenset(
        {
            "calibrated_coordinate_normalized_sampler_diagnostic",
            "sampler_coordinate_normalization",
        }
    ),
    "calibrated_sampler_locality_selection_diagnostic": frozenset(
        {
            "calibrated_sampler_locality_selection_diagnostic",
            "sampler_geometry_selection",
        }
    ),
    "calibrated_sampler_low_noise_diagnostic": frozenset(
        {"calibrated_sampler_low_noise_diagnostic", "sampler_low_noise"}
    ),
    "diffusion_data_scale_diagnostic": frozenset(
        {"diffusion_data_scale", "diffusion_data_scale_diagnostic"}
    ),
    "pairformer_attention_diagnostic": frozenset(
        {"pairformer_attention", "pairformer_attention_diagnostic"}
    ),
    "auxiliary_contact_loss_diagnostic": frozenset(
        {"auxiliary_loss", "auxiliary_contact_loss", "auxiliary_contact_loss_diagnostic"}
    ),
    "feature_ref_pos_scale_diagnostic": frozenset(
        {"feature_handling", "ref_pos_scale", "feature_ref_pos_scale_diagnostic"}
    ),
    "gradient_checkpointing_runtime_diagnostic": frozenset(
        {"memory_runtime", "gradient_checkpointing_runtime", "gradient_checkpointing_runtime_diagnostic"}
    ),
    "diffusion_initialization_scale_diagnostic": frozenset(
        {
            "diffusion_initialization_scale",
            "diffusion_initial_noise_scale",
            "diffusion_initialization_scale_diagnostic",
        }
    ),
}

UNIMPLEMENTED_ALLOWED_SURFACES = (
    "pairformer_attention",
    "auxiliary_loss",
    "feature_handling",
    "memory_runtime",
)


def review_surface_strategy(
    *,
    repo_root: str | Path = ".",
    next_surface_reviews: list[str | Path],
    diagnoses: list[str | Path] | None = None,
) -> SurfaceStrategyReview:
    """Decide whether a next planner is approved after offline evidence review."""

    if not next_surface_reviews:
        raise SurfaceStrategyReviewError("at least one next-surface review is required")
    root = Path(repo_root)
    review_payloads = [_read_json(root=root, path=item, schema=NEXT_SURFACE_SCHEMA) for item in next_surface_reviews]
    diagnosis_payloads = [
        _read_json(root=root, path=item, schema=POST_DISCARD_SCHEMA) for item in diagnoses or []
    ]
    exhausted = _exhausted_surfaces(review_payloads=review_payloads, diagnosis_payloads=diagnosis_payloads)
    latest = review_payloads[-1]
    latest_approved = latest.get("approved_next_surface")
    latest_required = latest.get("required_next_pr") if isinstance(latest.get("required_next_pr"), dict) else {}
    latest_planner = latest_required.get("planner") if isinstance(latest_required.get("planner"), str) else None
    latest_candidate_limit = latest_required.get("candidate_limit")
    if latest.get("decision") == "APPROVE_OFFLINE_PLANNER_PR_ONLY":
        if not isinstance(latest_approved, str) or not latest_planner:
            raise SurfaceStrategyReviewError("approved review must name approved_next_surface and planner")
        if _surface_exhausted(str(latest_approved), latest_planner, exhausted):
            return _blocked_report(
                next_surface_reviews=next_surface_reviews,
                diagnoses=diagnoses or [],
                exhausted=exhausted,
                reason=f"latest approved surface is already exhausted: {latest_approved}",
            )
        if latest_candidate_limit != 1:
            raise SurfaceStrategyReviewError("approved review must require candidate_limit=1")
        return SurfaceStrategyReview(
            schema_version=SCHEMA_VERSION,
            status="PASS",
            decision="APPROVE_OFFLINE_PLANNER_PR_ONLY",
            approved_next_surface=str(latest_approved),
            approved_planner=latest_planner,
            candidate_limit=1,
            may_start_live_candidate=False,
            may_start_open_ended_loop=False,
            bench_blocked_reason=(
                "offline planner PR must be implemented, merged, redeployed, and readiness-checked "
                "before one bounded live candidate"
            ),
            consumed_next_surface_reviews=[str(item) for item in next_surface_reviews],
            consumed_diagnoses=[str(item) for item in diagnoses or []],
            exhausted_surfaces=exhausted,
            unimplemented_candidate_surfaces=[],
            required_next_step="Implement the approved dry-run-only planner PR; do not start live search yet.",
            starts_search=False,
            writes_ledger=False,
            writes_discovery_ledger=False,
            official_benchmark_result=False,
        )
    return _blocked_report(
        next_surface_reviews=next_surface_reviews,
        diagnoses=diagnoses or [],
        exhausted=exhausted,
        reason="latest next-surface review did not approve a non-overlapping planner",
    )


def _blocked_report(
    *,
    next_surface_reviews: list[str | Path],
    diagnoses: list[str | Path],
    exhausted: list[str],
    reason: str,
) -> SurfaceStrategyReview:
    unimplemented = _unimplemented_candidate_surfaces(exhausted)
    return SurfaceStrategyReview(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision="NO_NON_OVERLAPPING_PLANNER_APPROVED",
        approved_next_surface=None,
        approved_planner=None,
        candidate_limit=0,
        may_start_live_candidate=False,
        may_start_open_ended_loop=False,
        bench_blocked_reason=reason,
        consumed_next_surface_reviews=[str(item) for item in next_surface_reviews],
        consumed_diagnoses=[str(item) for item in diagnoses],
        exhausted_surfaces=exhausted,
        unimplemented_candidate_surfaces=unimplemented,
        required_next_step=_blocked_required_next_step(unimplemented),
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _read_json(*, root: Path, path: str | Path, schema: str) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path)
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SurfaceStrategyReviewError(f"cannot read evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise SurfaceStrategyReviewError(f"evidence must be a JSON object: {path}")
    if payload.get("schema_version") != schema:
        raise SurfaceStrategyReviewError(f"evidence schema mismatch for {path}")
    if payload.get("status") != "PASS":
        raise SurfaceStrategyReviewError(f"evidence must have status=PASS: {path}")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise SurfaceStrategyReviewError(f"evidence must not claim {key}=true: {path}")
    return payload


def _safe_evidence_path(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SurfaceStrategyReviewError("evidence path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise SurfaceStrategyReviewError("evidence path must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise SurfaceStrategyReviewError(f"evidence must not be a symlink: {path}")
    return full


def _exhausted_surfaces(
    *,
    review_payloads: list[dict[str, object]],
    diagnosis_payloads: list[dict[str, object]],
) -> list[str]:
    exhausted: list[str] = []
    for payload in [*diagnosis_payloads, *review_payloads]:
        for key in ("exhausted_surfaces", "rejected_surfaces"):
            for item in _string_list(payload.get(key)):
                if item not in exhausted:
                    exhausted.append(item)
    return exhausted


def _surface_exhausted(approved_surface: str, planner: str, exhausted: list[str]) -> bool:
    exhausted_set = set(exhausted)
    aliases = set(IMPLEMENTED_PLANNER_SURFACES.get(planner, frozenset()))
    aliases.add(approved_surface)
    return any(alias in exhausted_set for alias in aliases)


def _unimplemented_candidate_surfaces(exhausted: list[str]) -> list[str]:
    exhausted_set = set(exhausted)
    implemented_aliases = {
        alias
        for aliases in IMPLEMENTED_PLANNER_SURFACES.values()
        for alias in aliases
    }
    return [
        surface
        for surface in UNIMPLEMENTED_ALLOWED_SURFACES
        if surface not in exhausted_set and surface not in implemented_aliases
    ]


def _blocked_required_next_step(unimplemented_candidate_surfaces: list[str]) -> str:
    if not unimplemented_candidate_surfaces:
        return (
            "No unimplemented allowed planner surfaces remain. Stop live trial-budget spend and open a "
            "broader offline strategy review before implementing another planner or starting the open-ended "
            "bench loop."
        )
    return (
        "Do offline design review for one unimplemented allowed surface; add a dry-run-only planner "
        "PR before any more live candidates or open-ended bench loop."
    )


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
