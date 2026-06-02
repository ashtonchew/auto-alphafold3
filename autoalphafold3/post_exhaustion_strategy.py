"""Offline PRD gate for strategy design after implemented planners are exhausted."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

BENCH_READINESS_SCHEMA = "autoaf3.bench_readiness_review.v1"
STRATEGY_EXHAUSTION_SCHEMA = "autoaf3.strategy_exhaustion_audit.v1"
SCHEMA_VERSION = "autoaf3.post_exhaustion_strategy_prd.v1"

APPROVED_STRATEGY_FAMILY = "evidence_guided_failure_mode_bridge"
APPROVED_PLANNER = "evidence_guided_failure_mode_bridge_diagnostic"
FORBIDDEN_EDITS = [
    "scorer",
    "benchmark_contract",
    "public_manifests",
    "fingerprints",
    "Modal resources",
    "templates",
    "baselines",
    "validation_labels",
    "canonical_ledger",
    "Discovery Ledger",
    "full_msa_or_database_rebuild",
]
ALLOWED_EDIT_AREAS = [
    "autoalphafold3/autoresearch_loop.py dry-run planner dispatch and candidate plan construction",
    "autoalphafold3/*review.py offline evidence review helpers",
    "configs/experiments/** generated candidate configs only",
    "tests/** local contract tests for the new dry-run planner and evidence gate",
    "docs/runbooks/** operating instructions",
]
STOP_CONDITIONS = [
    "bench readiness is not autonomous_search_ready=true",
    "strategy exhaustion audit does not report NO_IMPLEMENTED_PLANNER_REMAINING",
    "candidate_limit is not exactly 1",
    "dry-run artifact lacks scorer-sensitivity and prediction-geometry evidence inputs",
    "dry-run artifact repeats an exhausted implemented planner family",
    "dry-run artifact predicts only aggregate movement without target-level non-regression checks",
    "dry-run artifact touches scorer, benchmark, manifest, fingerprint, baseline, Modal resource, template, or ledger authority",
    "post-merge full local gate is not green",
]


class PostExhaustionStrategyError(RuntimeError):
    """Raised when post-exhaustion strategy evidence cannot be produced safely."""


@dataclass(frozen=True)
class PostExhaustionStrategyPrd:
    """JSON-friendly PRD for the first strategy after implemented planner exhaustion."""

    schema_version: str
    status: str
    decision: str
    approved_strategy_family: str | None
    approved_planner: str | None
    candidate_limit: int
    may_start_live_candidate: bool
    may_start_open_ended_loop: bool
    non_overlap_rationale: str | None
    allowed_edit_areas: list[str]
    forbidden_edits: list[str]
    dry_run_candidate_shape: dict[str, object]
    required_evidence_before_live_smoke: list[str]
    stop_conditions: list[str]
    consumed_bench_readiness_review: str
    consumed_strategy_exhaustion_audit: str
    exhausted_implemented_planners: list[str]
    remaining_implemented_planners: list[str]
    blocked_reason: str | None
    required_next_step: str
    ui_reporting_constraint: str
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def design_post_exhaustion_strategy(
    *,
    repo_root: str | Path = ".",
    bench_readiness_review: str | Path,
    strategy_exhaustion_audit: str | Path,
) -> PostExhaustionStrategyPrd:
    """Produce an offline strategy PRD after all implemented planners are exhausted."""

    root = Path(repo_root)
    bench = _read_json(
        root=root,
        path=bench_readiness_review,
        expected_schema=BENCH_READINESS_SCHEMA,
        label="bench readiness review",
    )
    audit = _read_json(
        root=root,
        path=strategy_exhaustion_audit,
        expected_schema=STRATEGY_EXHAUSTION_SCHEMA,
        label="strategy exhaustion audit",
    )
    autonomous_ready = bench.get("autonomous_search_ready") is True
    bench_blocked = bench.get("decision") == "BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED"
    can_start_bench = bench.get("can_start_open_ended_bench") is True
    no_planners = audit.get("decision") == "NO_IMPLEMENTED_PLANNER_REMAINING"
    remaining = _unique_strings(audit.get("remaining_implemented_planners"))
    exhausted = _unique_strings(audit.get("exhausted_implemented_planners"))
    if autonomous_ready and bench_blocked and not can_start_bench and no_planners and not remaining:
        return PostExhaustionStrategyPrd(
            schema_version=SCHEMA_VERSION,
            status="PASS",
            decision="APPROVE_DRY_RUN_STRATEGY_PRD_ONLY",
            approved_strategy_family=APPROVED_STRATEGY_FAMILY,
            approved_planner=APPROVED_PLANNER,
            candidate_limit=1,
            may_start_live_candidate=False,
            may_start_open_ended_loop=False,
            non_overlap_rationale=(
                "The exhausted catalog covers direct model, sampler, training, loss, runtime, feature, "
                "and diffusion knob families. This strategy is a pre-live evidence bridge: it changes "
                "candidate selection and falsification requirements by requiring target-level scorer "
                "sensitivity plus geometry failure-mode constraints before any new model or sampler "
                "candidate can spend Modal budget. It does not reapprove a new value for an exhausted "
                "knob family."
            ),
            allowed_edit_areas=list(ALLOWED_EDIT_AREAS),
            forbidden_edits=list(FORBIDDEN_EDITS),
            dry_run_candidate_shape=_dry_run_candidate_shape(),
            required_evidence_before_live_smoke=_required_evidence_before_live_smoke(),
            stop_conditions=list(STOP_CONDITIONS),
            consumed_bench_readiness_review=str(bench_readiness_review),
            consumed_strategy_exhaustion_audit=str(strategy_exhaustion_audit),
            exhausted_implemented_planners=exhausted,
            remaining_implemented_planners=remaining,
            blocked_reason=None,
            required_next_step=(
                "Implement one dry-run-only evidence_guided_failure_mode_bridge_diagnostic planner PR. "
                "It must consume scorer-sensitivity and geometry evidence, emit exactly one artifact-only "
                "candidate envelope, and keep live Modal execution blocked until a fresh bench-readiness "
                "review explicitly permits one bounded live smoke."
            ),
            ui_reporting_constraint=(
                "UI and reports may show this PRD as planning evidence only; they must not describe it as "
                "an official benchmark result, a discovery, or a scored candidate."
            ),
            starts_search=False,
            writes_ledger=False,
            writes_discovery_ledger=False,
            official_benchmark_result=False,
        )
    return PostExhaustionStrategyPrd(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision="NO_POST_EXHAUSTION_STRATEGY_APPROVED",
        approved_strategy_family=None,
        approved_planner=None,
        candidate_limit=0,
        may_start_live_candidate=False,
        may_start_open_ended_loop=False,
        non_overlap_rationale=None,
        allowed_edit_areas=[],
        forbidden_edits=list(FORBIDDEN_EDITS),
        dry_run_candidate_shape={},
        required_evidence_before_live_smoke=[],
        stop_conditions=["post-exhaustion strategy prerequisites are not satisfied"],
        consumed_bench_readiness_review=str(bench_readiness_review),
        consumed_strategy_exhaustion_audit=str(strategy_exhaustion_audit),
        exhausted_implemented_planners=exhausted,
        remaining_implemented_planners=remaining,
        blocked_reason=_blocked_reason(
            autonomous_ready=autonomous_ready,
            bench_blocked=bench_blocked,
            can_start_bench=can_start_bench,
            no_planners=no_planners,
            remaining=remaining,
        ),
        required_next_step="Keep the bench blocked and regenerate the prerequisite readiness/audit evidence.",
        ui_reporting_constraint="No planning claim is approved from this artifact.",
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _dry_run_candidate_shape() -> dict[str, object]:
    return {
        "artifact_only": True,
        "candidate_limit": 1,
        "requires_inputs": [
            "strategy_exhaustion_audit",
            "bench_readiness_review",
            "at_least_one_scorer_sensitivity_report",
            "at_least_one_prediction_geometry_audit",
        ],
        "must_emit": [
            "trial_id",
            "candidate_intent",
            "target_level_non_regression_expectations",
            "geometry_failure_modes_to_avoid",
            "forbidden_edit_attestation",
            "stop_before_live_modal",
        ],
        "must_not_emit": [
            "official_benchmark_result=true",
            "starts_search=true",
            "writes_ledger=true",
            "writes_discovery_ledger=true",
            "live_modal_execution=true",
        ],
    }


def _required_evidence_before_live_smoke() -> list[str]:
    return [
        "post-merge full local gate passes",
        "foundation readiness report remains autonomous_search_ready=true",
        "dry-run planner envelope contains exactly one candidate",
        "dry-run planner consumes real local scorer-sensitivity evidence",
        "dry-run planner consumes real local prediction-geometry evidence",
        "dry-run planner declares target-level non-regression expectations",
        "dry-run planner declares geometry failure modes it will kill before live execution",
        "fresh bench-readiness review still blocks open-ended bench and permits at most one bounded live smoke",
    ]


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
        raise PostExhaustionStrategyError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PostExhaustionStrategyError(f"{label} must be a JSON object")
    if payload.get("schema_version") != expected_schema:
        raise PostExhaustionStrategyError(f"{label} schema mismatch")
    if payload.get("status") != "PASS":
        raise PostExhaustionStrategyError(f"{label} must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise PostExhaustionStrategyError(f"{label} must not claim {key}=true")
    return payload


def _safe_evidence_path(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PostExhaustionStrategyError("evidence path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise PostExhaustionStrategyError("evidence path must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise PostExhaustionStrategyError(f"evidence must not be a symlink: {path}")
    return full


def _blocked_reason(
    *,
    autonomous_ready: bool,
    bench_blocked: bool,
    can_start_bench: bool,
    no_planners: bool,
    remaining: list[str],
) -> str:
    if not autonomous_ready:
        return "Foundation readiness is not green."
    if can_start_bench:
        return "Bench readiness already approves the open-ended bench; this PRD gate is not the authority."
    if not bench_blocked:
        return "Bench readiness is not in the expected strategy-exhausted blocked state."
    if not no_planners:
        return "Strategy exhaustion audit does not report all implemented planners exhausted."
    if remaining:
        return "Strategy exhaustion audit still reports remaining implemented planners."
    return "Post-exhaustion strategy prerequisites are not satisfied."


def _unique_strings(value: object) -> list[str]:
    result: list[str] = []
    if not isinstance(value, list):
        return result
    for item in value:
        text = str(item)
        if text not in result:
            result.append(text)
    return result
