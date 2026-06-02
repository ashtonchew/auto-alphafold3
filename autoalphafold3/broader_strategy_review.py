"""Broader offline strategy review after all planned surfaces are exhausted."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

SURFACE_STRATEGY_SCHEMA = "autoaf3.surface_strategy_review.v1"
BENCH_READINESS_SCHEMA = "autoaf3.bench_readiness_review.v1"
SCHEMA_VERSION = "autoaf3.broader_strategy_review.v1"


class BroaderStrategyReviewError(RuntimeError):
    """Raised when broader strategy evidence cannot be reviewed safely."""


@dataclass(frozen=True)
class BroaderStrategyReview:
    """JSON-friendly review for reopening a dry-run planner path."""

    schema_version: str
    status: str
    decision: str
    approved_next_surface: str | None
    approved_planner: str | None
    candidate_limit: int
    may_start_live_candidate: bool
    may_start_open_ended_loop: bool
    non_overlap_rationale: str | None
    forbidden_edits: list[str]
    stop_conditions: list[str]
    consumed_surface_strategy_review: str
    consumed_bench_readiness_review: str
    exhausted_surfaces: list[str]
    required_next_step: str
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EXHAUSTED_SURFACES_REQUIRED = frozenset({"auxiliary_loss", "feature_handling", "memory_runtime"})
APPROVED_SURFACE = "diffusion_initialization_scale"
APPROVED_PLANNER = "diffusion_initialization_scale_diagnostic"
FORBIDDEN_EDITS = [
    "scorer",
    "benchmark_contract",
    "public_manifests",
    "fingerprints",
    "Modal resources",
    "templates",
    "baselines",
    "canonical_ledger",
    "Discovery Ledger",
    "validation_labels",
    "full_msa_or_database_rebuild",
]
STOP_CONDITIONS = [
    "foundation readiness is not autonomous_search_ready=true",
    "surface strategy no longer reports all planned surfaces exhausted",
    "bench readiness does not report BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED",
    "dry-run planner creates more than one candidate",
    "candidate touches a forbidden edit surface",
    "candidate writes the canonical ledger or Discovery Ledger",
    "post-merge readiness is not green",
]


def review_broader_strategy(
    *,
    repo_root: str | Path = ".",
    surface_strategy_review: str | Path,
    bench_readiness_review: str | Path,
) -> BroaderStrategyReview:
    """Decide whether one new dry-run planner path is strategy-approved."""

    root = Path(repo_root)
    surface_strategy = _read_json(
        root=root,
        path=surface_strategy_review,
        expected_schema=SURFACE_STRATEGY_SCHEMA,
        label="surface strategy review",
    )
    bench_readiness = _read_json(
        root=root,
        path=bench_readiness_review,
        expected_schema=BENCH_READINESS_SCHEMA,
        label="bench readiness review",
    )
    exhausted = _unique_strings(surface_strategy.get("exhausted_surfaces"))
    exhausted_set = set(exhausted)
    planned_exhausted = EXHAUSTED_SURFACES_REQUIRED.issubset(exhausted_set)
    strategy_blocked = surface_strategy.get("decision") == "NO_NON_OVERLAPPING_PLANNER_APPROVED"
    no_unimplemented = _unique_strings(surface_strategy.get("unimplemented_candidate_surfaces")) == []
    bench_blocked = bench_readiness.get("decision") == "BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED"
    autonomous_ready = bench_readiness.get("autonomous_search_ready") is True
    if strategy_blocked and no_unimplemented and planned_exhausted and bench_blocked and autonomous_ready:
        return BroaderStrategyReview(
            schema_version=SCHEMA_VERSION,
            status="PASS",
            decision="APPROVE_DRY_RUN_PLANNER_PR_ONLY",
            approved_next_surface=APPROVED_SURFACE,
            approved_planner=APPROVED_PLANNER,
            candidate_limit=1,
            may_start_live_candidate=False,
            may_start_open_ended_loop=False,
            non_overlap_rationale=(
                "The exhausted evidence covers auxiliary loss shape, feature/ref-position scaling, and "
                "runtime/memory behavior. This surface targets the model-internal diffusion initial state "
                "scale before denoising, which is separate from post-hoc sampler normalization, "
                "diffusion data-scale training noise, feature handling, loss weighting, and Modal runtime."
            ),
            forbidden_edits=list(FORBIDDEN_EDITS),
            stop_conditions=list(STOP_CONDITIONS),
            consumed_surface_strategy_review=str(surface_strategy_review),
            consumed_bench_readiness_review=str(bench_readiness_review),
            exhausted_surfaces=exhausted,
            required_next_step=(
                "Implement one dry-run-only diffusion_initialization_scale_diagnostic planner PR; "
                "merge, rerun readiness, and do not start live Modal execution from this artifact alone."
            ),
            starts_search=False,
            writes_ledger=False,
            writes_discovery_ledger=False,
            official_benchmark_result=False,
        )
    return BroaderStrategyReview(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision="NO_BROADER_STRATEGY_APPROVED",
        approved_next_surface=None,
        approved_planner=None,
        candidate_limit=0,
        may_start_live_candidate=False,
        may_start_open_ended_loop=False,
        non_overlap_rationale=None,
        forbidden_edits=list(FORBIDDEN_EDITS),
        stop_conditions=["no broader non-overlapping dry-run planner was approved"],
        consumed_surface_strategy_review=str(surface_strategy_review),
        consumed_bench_readiness_review=str(bench_readiness_review),
        exhausted_surfaces=exhausted,
        required_next_step=(
            "Keep the open-ended bench blocked and produce stronger offline evidence for a "
            "non-overlapping strategy."
        ),
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _read_json(
    *,
    root: Path,
    path: str | Path,
    expected_schema: str,
    label: str,
) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path)
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BroaderStrategyReviewError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise BroaderStrategyReviewError(f"{label} must be a JSON object")
    if payload.get("schema_version") != expected_schema:
        raise BroaderStrategyReviewError(f"{label} schema mismatch")
    if payload.get("status") != "PASS":
        raise BroaderStrategyReviewError(f"{label} must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise BroaderStrategyReviewError(f"{label} must not claim {key}=true")
    return payload


def _safe_evidence_path(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BroaderStrategyReviewError("evidence path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise BroaderStrategyReviewError("evidence path must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise BroaderStrategyReviewError(f"evidence must not be a symlink: {path}")
    return full


def _unique_strings(value: object) -> list[str]:
    result: list[str] = []
    if not isinstance(value, list):
        return result
    for item in value:
        text = str(item)
        if text not in result:
            result.append(text)
    return result
