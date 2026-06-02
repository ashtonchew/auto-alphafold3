"""Offline design review for one new autoresearch surface planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

STRATEGY_SCHEMA = "autoaf3.surface_strategy_review.v1"
SCHEMA_VERSION = "autoaf3.surface_design_review.v1"

ALLOWED_DESIGN_SURFACES: dict[str, str] = {
    "pairformer_attention": "pairformer_attention_diagnostic",
}


class SurfaceDesignReviewError(RuntimeError):
    """Raised when a new surface cannot be approved safely."""


@dataclass(frozen=True)
class SurfaceDesignReview:
    """JSON-friendly design decision for an unimplemented planner."""

    schema_version: str
    status: str
    decision: str
    approved_next_surface: str | None
    approved_planner: str | None
    candidate_limit: int
    may_start_live_candidate: bool
    may_start_open_ended_loop: bool
    bench_blocked_reason: str | None
    consumed_strategy_review: str
    exhausted_surfaces: list[str]
    required_next_pr: dict[str, object]
    design_rationale: str
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def review_surface_design(
    *,
    repo_root: str | Path = ".",
    strategy_review: str | Path,
    proposed_surface: str,
) -> SurfaceDesignReview:
    """Approve one dry-run-only planner for an unexhausted, unimplemented surface."""

    if proposed_surface not in ALLOWED_DESIGN_SURFACES:
        raise SurfaceDesignReviewError(f"unsupported proposed surface: {proposed_surface}")
    root = Path(repo_root)
    payload = _read_strategy_review(root=root, path=strategy_review)
    exhausted = _string_list(payload.get("exhausted_surfaces"))
    unimplemented = _string_list(payload.get("unimplemented_candidate_surfaces"))
    if payload.get("decision") != "NO_NON_OVERLAPPING_PLANNER_APPROVED":
        raise SurfaceDesignReviewError("strategy review must first block implemented planners")
    if proposed_surface in set(exhausted):
        raise SurfaceDesignReviewError(f"proposed surface is already exhausted: {proposed_surface}")
    if proposed_surface not in set(unimplemented):
        raise SurfaceDesignReviewError("proposed surface must be listed as unimplemented and available")
    if payload.get("may_start_live_candidate") is True or payload.get("may_start_open_ended_loop") is True:
        raise SurfaceDesignReviewError("strategy review must not authorize live search")

    planner = ALLOWED_DESIGN_SURFACES[proposed_surface]
    return SurfaceDesignReview(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision="APPROVE_DRY_RUN_PLANNER_IMPLEMENTATION_ONLY",
        approved_next_surface=proposed_surface,
        approved_planner=planner,
        candidate_limit=1,
        may_start_live_candidate=False,
        may_start_open_ended_loop=False,
        bench_blocked_reason=(
            "new planner must be implemented, dry-run validated, merged, redeployed, and readiness-checked "
            "before one bounded live candidate"
        ),
        consumed_strategy_review=str(strategy_review),
        exhausted_surfaces=exhausted,
        required_next_pr={
            "planner": planner,
            "candidate_limit": 1,
            "mode_before_merge": "dry-run",
            "candidate_budget": "trial",
            "must_consume_review": True,
        },
        design_rationale=(
            "Pairformer triangular attention is a canonical allowed move family and remains non-overlapping "
            "with the exhausted sampler, locality, optimizer, capacity, recycling, curriculum, and diffusion "
            "data-scale surfaces."
        ),
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _read_strategy_review(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path)
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SurfaceDesignReviewError(f"cannot read strategy review: {path}") from exc
    if not isinstance(payload, dict):
        raise SurfaceDesignReviewError("strategy review must be a JSON object")
    if payload.get("schema_version") != STRATEGY_SCHEMA:
        raise SurfaceDesignReviewError("strategy review schema mismatch")
    if payload.get("status") != "PASS":
        raise SurfaceDesignReviewError("strategy review must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise SurfaceDesignReviewError(f"strategy review must not claim {key}=true")
    return payload


def _safe_evidence_path(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SurfaceDesignReviewError("evidence path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise SurfaceDesignReviewError("evidence path must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise SurfaceDesignReviewError(f"evidence must not be a symlink: {path}")
    return full


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
