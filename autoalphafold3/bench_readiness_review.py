"""Composite offline gate for starting the open-ended autoresearch bench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from autoalphafold3.readiness import build_readiness_report

SURFACE_STRATEGY_SCHEMA = "autoaf3.surface_strategy_review.v1"
BROADER_STRATEGY_SCHEMA = "autoaf3.broader_strategy_review.v1"
EVIDENCE_BRIDGE_SCHEMA = "autoaf3.evidence_bridge_review.v1"
CANDIDATE_IMPLEMENTATION_SCHEMA = "autoaf3.candidate_implementation_review.v1"
SCHEMA_VERSION = "autoaf3.bench_readiness_review.v1"


class BenchReadinessReviewError(RuntimeError):
    """Raised when bench readiness evidence cannot be reviewed safely."""


@dataclass(frozen=True)
class BenchReadinessReview:
    """JSON-friendly open-ended bench decision."""

    schema_version: str
    status: str
    decision: str
    can_start_open_ended_bench: bool
    autonomous_search_ready: bool
    surface_strategy_decision: str
    may_start_live_candidate: bool
    may_start_open_ended_loop: bool
    exhausted_surfaces: list[str]
    unimplemented_candidate_surfaces: list[str]
    broader_strategy_decision: str | None
    approved_broader_surface: str | None
    approved_broader_planner: str | None
    evidence_bridge_decision: str | None
    approved_evidence_bridge_planner: str | None
    candidate_implementation_decision: str | None
    approved_candidate_implementation: str | None
    required_objectives: list[dict[str, object]]
    roadmap: list[dict[str, object]]
    evidence: dict[str, object]
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def review_bench_readiness(
    *,
    repo_root: str | Path = ".",
    surface_strategy_review: str | Path,
    baseline_dir: str | Path = "runs/baseline",
    config_path: str | Path = "configs/nanofold_dev_cpu_smoke.json",
    calibration_path: str | Path = "runs/falsification_gate_calibration.json",
    modal_authority_path: str | Path = "runs/modal_event_authority.json",
    broader_strategy_review: str | Path | None = None,
    evidence_bridge_review: str | Path | None = None,
    candidate_implementation_review: str | Path | None = None,
) -> BenchReadinessReview:
    """Decide whether the actual open-ended autoresearch bench may start."""

    root = Path(repo_root)
    strategy = _read_surface_strategy(root=root, path=surface_strategy_review)
    broader_strategy = (
        _read_broader_strategy(root=root, path=broader_strategy_review)
        if broader_strategy_review is not None
        else None
    )
    evidence_bridge = (
        _read_evidence_bridge(root=root, path=evidence_bridge_review)
        if evidence_bridge_review is not None
        else None
    )
    candidate_implementation = (
        _read_candidate_implementation(root=root, path=candidate_implementation_review)
        if candidate_implementation_review is not None
        else None
    )
    readiness = build_readiness_report(
        repo_root=root,
        baseline_dir=baseline_dir,
        config_path=config_path,
        calibration_path=calibration_path,
        modal_authority_path=modal_authority_path,
    )
    readiness_payload = readiness.to_dict()
    autonomous_ready = readiness.autonomous_search_ready
    may_start_live = strategy.get("may_start_live_candidate") is True
    may_start_open_ended = strategy.get("may_start_open_ended_loop") is True
    exhausted = _string_list(strategy.get("exhausted_surfaces"))
    unimplemented = _string_list(strategy.get("unimplemented_candidate_surfaces"))
    strategy_decision = str(strategy.get("decision") or "")
    broader_decision = (
        str(broader_strategy.get("decision") or "")
        if broader_strategy is not None
        else None
    )
    broader_planner = (
        str(broader_strategy.get("approved_planner") or "")
        if broader_strategy is not None and broader_strategy.get("approved_planner") is not None
        else None
    )
    broader_surface = (
        str(broader_strategy.get("approved_next_surface") or "")
        if broader_strategy is not None and broader_strategy.get("approved_next_surface") is not None
        else None
    )
    evidence_bridge_decision = (
        str(evidence_bridge.get("decision") or "")
        if evidence_bridge is not None
        else None
    )
    evidence_bridge_planner = (
        str(evidence_bridge.get("approved_planner") or "")
        if evidence_bridge is not None and evidence_bridge.get("approved_planner") is not None
        else None
    )
    candidate_implementation_decision = (
        str(candidate_implementation.get("decision") or "")
        if candidate_implementation is not None
        else None
    )
    approved_candidate_implementation = (
        str(candidate_implementation.get("approved_candidate") or "")
        if candidate_implementation is not None and candidate_implementation.get("approved_candidate") is not None
        else None
    )
    if autonomous_ready and may_start_open_ended:
        decision = "APPROVE_OPEN_ENDED_BENCH"
        can_start = True
    elif autonomous_ready and candidate_implementation_decision == "APPROVE_LIVE_SMOKE_GATE_PR_ONLY":
        decision = "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_GATE_REQUIRED"
        can_start = False
    elif autonomous_ready and evidence_bridge_decision == "APPROVE_NEXT_CANDIDATE_IMPLEMENTATION_PR_ONLY":
        decision = "BLOCK_OPEN_ENDED_BENCH_CANDIDATE_IMPLEMENTATION_REQUIRED"
        can_start = False
    elif autonomous_ready and broader_decision == "APPROVE_DRY_RUN_PLANNER_PR_ONLY":
        decision = "BLOCK_OPEN_ENDED_BENCH_DRY_RUN_PLANNER_REQUIRED"
        can_start = False
    elif autonomous_ready and strategy_decision == "NO_NON_OVERLAPPING_PLANNER_APPROVED" and not unimplemented:
        decision = "BLOCK_OPEN_ENDED_BENCH_STRATEGY_EXHAUSTED"
        can_start = False
    elif not autonomous_ready:
        decision = "BLOCK_OPEN_ENDED_BENCH_READINESS_NOT_GREEN"
        can_start = False
    else:
        decision = "BLOCK_OPEN_ENDED_BENCH_STRATEGY_NOT_APPROVED"
        can_start = False
    return BenchReadinessReview(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision=decision,
        can_start_open_ended_bench=can_start,
        autonomous_search_ready=autonomous_ready,
        surface_strategy_decision=strategy_decision,
        may_start_live_candidate=may_start_live,
        may_start_open_ended_loop=may_start_open_ended,
        exhausted_surfaces=exhausted,
        unimplemented_candidate_surfaces=unimplemented,
        broader_strategy_decision=broader_decision,
        approved_broader_surface=broader_surface,
        approved_broader_planner=broader_planner,
        evidence_bridge_decision=evidence_bridge_decision,
        approved_evidence_bridge_planner=evidence_bridge_planner,
        candidate_implementation_decision=candidate_implementation_decision,
        approved_candidate_implementation=approved_candidate_implementation,
        required_objectives=_required_objectives(
            autonomous_ready=autonomous_ready,
            may_start_open_ended=may_start_open_ended,
            unimplemented=unimplemented,
            broader_decision=broader_decision,
            broader_surface=broader_surface,
            broader_planner=broader_planner,
            evidence_bridge_decision=evidence_bridge_decision,
            evidence_bridge_planner=evidence_bridge_planner,
            candidate_implementation_decision=candidate_implementation_decision,
            approved_candidate_implementation=approved_candidate_implementation,
        ),
        roadmap=_roadmap(
            autonomous_ready=autonomous_ready,
            may_start_open_ended=may_start_open_ended,
            unimplemented=unimplemented,
            broader_decision=broader_decision,
            broader_surface=broader_surface,
            broader_planner=broader_planner,
            evidence_bridge_decision=evidence_bridge_decision,
            evidence_bridge_planner=evidence_bridge_planner,
            candidate_implementation_decision=candidate_implementation_decision,
            approved_candidate_implementation=approved_candidate_implementation,
        ),
        evidence={
            "surface_strategy_review": str(surface_strategy_review),
            "broader_strategy_review": str(broader_strategy_review) if broader_strategy_review is not None else None,
            "evidence_bridge_review": str(evidence_bridge_review) if evidence_bridge_review is not None else None,
            "candidate_implementation_review": (
                str(candidate_implementation_review) if candidate_implementation_review is not None else None
            ),
            "readiness_mode": readiness_payload.get("mode"),
            "readiness_problems": readiness_payload.get("problems", []),
            "pending_human_actions": readiness_payload.get("pending_human_actions", []),
            "baseline_lock_status": _component_status(readiness_payload, "baseline_lock"),
            "local_gates_status": _component_status(readiness_payload, "local_gates"),
            "modal_event_authority_status": _component_status(readiness_payload, "modal_event_authority"),
            "gate_calibration_status": _component_status(readiness_payload, "gate_calibration"),
        },
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _read_surface_strategy(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path)
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchReadinessReviewError(f"cannot read surface strategy review: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchReadinessReviewError("surface strategy review must be a JSON object")
    if payload.get("schema_version") != SURFACE_STRATEGY_SCHEMA:
        raise BenchReadinessReviewError("surface strategy review schema mismatch")
    if payload.get("status") != "PASS":
        raise BenchReadinessReviewError("surface strategy review must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise BenchReadinessReviewError(f"surface strategy review must not claim {key}=true")
    return payload


def _read_broader_strategy(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path, label="broader strategy")
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchReadinessReviewError(f"cannot read broader strategy review: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchReadinessReviewError("broader strategy review must be a JSON object")
    if payload.get("schema_version") != BROADER_STRATEGY_SCHEMA:
        raise BenchReadinessReviewError("broader strategy review schema mismatch")
    if payload.get("status") != "PASS":
        raise BenchReadinessReviewError("broader strategy review must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise BenchReadinessReviewError(f"broader strategy review must not claim {key}=true")
    if payload.get("may_start_live_candidate") is True or payload.get("may_start_open_ended_loop") is True:
        raise BenchReadinessReviewError("broader strategy review must not authorize live or open-ended execution")
    return payload


def _read_evidence_bridge(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path, label="evidence bridge")
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchReadinessReviewError(f"cannot read evidence bridge review: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchReadinessReviewError("evidence bridge review must be a JSON object")
    if payload.get("schema_version") != EVIDENCE_BRIDGE_SCHEMA:
        raise BenchReadinessReviewError("evidence bridge review schema mismatch")
    if payload.get("status") != "PASS":
        raise BenchReadinessReviewError("evidence bridge review must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise BenchReadinessReviewError(f"evidence bridge review must not claim {key}=true")
    if payload.get("may_start_live_candidate") is True or payload.get("may_start_open_ended_loop") is True:
        raise BenchReadinessReviewError("evidence bridge review must not authorize live or open-ended execution")
    return payload


def _read_candidate_implementation(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path, label="candidate implementation")
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchReadinessReviewError(f"cannot read candidate implementation review: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchReadinessReviewError("candidate implementation review must be a JSON object")
    if payload.get("schema_version") != CANDIDATE_IMPLEMENTATION_SCHEMA:
        raise BenchReadinessReviewError("candidate implementation review schema mismatch")
    if payload.get("status") != "PASS":
        raise BenchReadinessReviewError("candidate implementation review must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise BenchReadinessReviewError(f"candidate implementation review must not claim {key}=true")
    if payload.get("may_start_live_candidate") is True or payload.get("may_start_open_ended_loop") is True:
        raise BenchReadinessReviewError("candidate implementation review must not authorize live or open-ended execution")
    return payload


def _safe_evidence_path(*, root: Path, path: str | Path, label: str = "surface strategy") -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BenchReadinessReviewError("evidence path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise BenchReadinessReviewError(f"{label} evidence must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise BenchReadinessReviewError(f"evidence must not be a symlink: {path}")
    return full


def _required_objectives(
    *,
    autonomous_ready: bool,
    may_start_open_ended: bool,
    unimplemented: list[str],
    broader_decision: str | None,
    broader_surface: str | None,
    broader_planner: str | None,
    evidence_bridge_decision: str | None,
    evidence_bridge_planner: str | None,
    candidate_implementation_decision: str | None,
    approved_candidate_implementation: str | None,
) -> list[dict[str, object]]:
    objectives: list[dict[str, object]] = []
    if not autonomous_ready:
        objectives.append(
            {
                "name": "restore_foundation_readiness",
                "status": "required",
                "objective": "Make baseline, scorer, manifests, Modal authority, preflight, and ledger gates pass.",
                "evidence_required": "readiness-report emits autonomous_search_ready=true",
            }
        )
    if not may_start_open_ended and unimplemented:
        objectives.append(
            {
                "name": "implement_remaining_dry_run_planner",
                "status": "required",
                "objective": "Consume offline design review for one unimplemented allowed surface before more live spend.",
                "evidence_required": "surface strategy approves a dry-run-only planner and post-merge readiness is green",
            }
        )
    if not may_start_open_ended and candidate_implementation_decision == "APPROVE_LIVE_SMOKE_GATE_PR_ONLY":
        objectives.append(
            {
                "name": "implement_live_smoke_approval_gate",
                "status": "required",
                "objective": (
                    f"Implement a one-candidate live-smoke approval gate for {approved_candidate_implementation}; "
                    "merge it and rerun readiness before any live Modal execution."
                ),
                "evidence_required": (
                    "live-smoke gate PR, full local gate, foundation readiness checks, explicit approval token, "
                    "and a fresh bench-readiness review"
                ),
            }
        )
    elif not may_start_open_ended and evidence_bridge_decision == "APPROVE_NEXT_CANDIDATE_IMPLEMENTATION_PR_ONLY":
        objectives.append(
            {
                "name": "implement_evidence_guided_candidate_pr",
                "status": "required",
                "objective": (
                    f"Implement one offline/local candidate PR from {evidence_bridge_planner}; "
                    "merge it and rerun readiness before live Modal execution."
                ),
                "evidence_required": (
                    "candidate PR, full local gate, foundation readiness checks, and a fresh bridge-aware "
                    "bench-readiness review"
                ),
            }
        )
    elif not may_start_open_ended and broader_decision == "APPROVE_DRY_RUN_PLANNER_PR_ONLY":
        objectives.append(
            {
                "name": "implement_broader_dry_run_planner",
                "status": "required",
                "objective": (
                    f"Implement one dry-run-only {broader_planner} candidate path for {broader_surface}; "
                    "merge it and rerun readiness before any live Modal execution."
                ),
                "evidence_required": (
                    "planner PR, dry-run candidate artifact with candidate_limit=1, and post-merge "
                    "readiness review"
                ),
            }
        )
    elif not may_start_open_ended and not unimplemented:
        objectives.append(
            {
                "name": "broader_offline_strategy_review",
                "status": "required",
                "objective": (
                    "Define a new explicit, non-overlapping search strategy before any new planner or "
                    "open-ended bench run."
                ),
                "evidence_required": (
                    "new strategy artifact names the surface, forbidden edits, candidate limit, stop "
                    "conditions, and why it is not exhausted"
                ),
            }
        )
    if may_start_open_ended:
        objectives.append(
            {
                "name": "start_open_ended_bench",
                "status": "approved",
                "objective": "Run the open-ended autoresearch bench through the trusted orchestrator.",
                "evidence_required": "surface strategy and readiness both approve open-ended execution",
            }
        )
    return objectives


def _roadmap(
    *,
    autonomous_ready: bool,
    may_start_open_ended: bool,
    unimplemented: list[str],
    broader_decision: str | None,
    broader_surface: str | None,
    broader_planner: str | None,
    evidence_bridge_decision: str | None,
    evidence_bridge_planner: str | None,
    candidate_implementation_decision: str | None,
    approved_candidate_implementation: str | None,
) -> list[dict[str, object]]:
    if may_start_open_ended:
        return [
            {
                "step": "run_open_ended_bench",
                "status": "ready",
                "action": "Start the approved open-ended Modal bench with explicit approval token.",
            }
        ]
    steps = [
        {
            "step": "stop_live_spend",
            "status": "required",
            "action": "Do not run another live candidate or the open-ended bench from the current evidence state.",
        }
    ]
    if not autonomous_ready:
        steps.append(
            {
                "step": "repair_foundation_readiness",
                "status": "required",
                "action": "Fix readiness problems before revisiting strategy.",
            }
        )
    elif candidate_implementation_decision == "APPROVE_LIVE_SMOKE_GATE_PR_ONLY":
        steps.append(
            {
                "step": "implement_live_smoke_approval_gate",
                "status": "required",
                "action": (
                    f"Implement a one-candidate live-smoke approval gate for {approved_candidate_implementation}; "
                    "continue blocking live Modal until that gate explicitly approves one bounded smoke."
                ),
            }
        )
    elif evidence_bridge_decision == "APPROVE_NEXT_CANDIDATE_IMPLEMENTATION_PR_ONLY":
        steps.append(
            {
                "step": "implement_evidence_guided_candidate_pr",
                "status": "required",
                "action": (
                    f"Implement one offline/local candidate PR from {evidence_bridge_planner}; keep live Modal "
                    "blocked until the next gate explicitly approves one bounded smoke."
                ),
            }
        )
    elif broader_decision == "APPROVE_DRY_RUN_PLANNER_PR_ONLY":
        steps.append(
            {
                "step": "implement_broader_dry_run_planner",
                "status": "required",
                "action": (
                    f"Implement {broader_planner} for {broader_surface} as one dry-run-only candidate "
                    "before any live Modal execution."
                ),
            }
        )
    elif unimplemented:
        steps.append(
            {
                "step": "consume_remaining_surface",
                "status": "required",
                "action": "Run offline design review and dry-run planner PR for the remaining allowed surface.",
            }
        )
    else:
        steps.append(
            {
                "step": "write_broader_strategy_prd",
                "status": "required",
                "action": (
                    "Create a new offline strategy/PRD that changes the search approach without touching locked "
                    "benchmark, scorer, Modal resources, manifests, fingerprints, or baselines."
                ),
            }
        )
    steps.append(
        {
            "step": "reopen_bench_gate",
            "status": "pending",
            "action": "Rerun bench-readiness-review after the new strategy evidence exists.",
        }
    )
    return steps


def _component_status(payload: dict[str, object], key: str) -> str | None:
    component = payload.get(key)
    if not isinstance(component, dict):
        return None
    value = component.get("status")
    return str(value) if value is not None else None


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
