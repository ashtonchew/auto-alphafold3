"""Offline gate for approving the actual open-ended autoresearch bench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from autoalphafold3.autoresearch_loop import APPROVAL_TEXT
from autoalphafold3.llm_policy import DEFAULT_LLM_MODEL
from autoalphafold3.readiness import build_readiness_report

SCHEMA_VERSION = "autoaf3.open_ended_bench_gate.v1"
STRATEGY_EXHAUSTION_SCHEMA = "autoaf3.strategy_exhaustion_audit.v1"
APPROVED_PLANNER = "llm"
APPROVED_MODE = "modal"
APPROVED_CANDIDATE_BUDGET = "smoke"
APPROVED_MAX_CANDIDATES = 3
APPROVED_FAILURE_STREAK_LIMIT = 2


class OpenEndedBenchGateError(RuntimeError):
    """Raised when the open-ended bench gate cannot approve safely."""


@dataclass(frozen=True)
class OpenEndedBenchGate:
    """JSON-friendly approval for one open-ended Modal bench run."""

    schema_version: str
    status: str
    decision: str
    approved_mode: str | None
    approved_planner: str | None
    approved_model: str | None
    approved_candidate_budget: str | None
    approved_max_candidates: int
    approved_failure_streak_limit: int
    required_approval_token: str | None
    consumed_strategy_exhaustion_audit: str
    readiness_evidence: dict[str, object]
    blocked_reasons: list[str]
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


def review_open_ended_bench_gate(
    *,
    repo_root: str | Path = ".",
    strategy_exhaustion_audit: str | Path,
    baseline_dir: str | Path = "runs/baseline",
    config_path: str | Path = "configs/nanofold_dev_cpu_smoke.json",
    calibration_path: str | Path = "runs/falsification_gate_calibration.json",
    modal_authority_path: str | Path = "runs/modal_event_authority.json",
    model: str = DEFAULT_LLM_MODEL,
) -> OpenEndedBenchGate:
    """Approve one LLM-planned Modal bench only after local strategy exhaustion."""

    root = Path(repo_root)
    audit = _read_strategy_exhaustion(root=root, path=strategy_exhaustion_audit)
    readiness = build_readiness_report(
        repo_root=root,
        baseline_dir=baseline_dir,
        config_path=config_path,
        calibration_path=calibration_path,
        modal_authority_path=modal_authority_path,
    )
    readiness_payload = readiness.to_dict()
    blocked: list[str] = []
    if audit.get("decision") != "NO_IMPLEMENTED_PLANNER_REMAINING":
        blocked.append("strategy exhaustion audit must report NO_IMPLEMENTED_PLANNER_REMAINING")
    if audit.get("remaining_implemented_planners") != []:
        blocked.append("strategy exhaustion audit must report no remaining implemented planners")
    if audit.get("can_start_open_ended_bench") is True:
        blocked.append("strategy exhaustion audit must not already claim bench can start")
    if not readiness.autonomous_search_ready:
        blocked.append("foundation readiness is not autonomous_search_ready=true")
    for component in ("baseline_lock", "local_gates", "gate_calibration", "modal_event_authority"):
        status = _component_status(readiness_payload, component)
        if status != "PASS":
            blocked.append(f"{component} status is not PASS")
    if readiness_payload.get("pending_human_actions"):
        blocked.append("readiness report still has pending human actions")
    decision = "APPROVE_OPEN_ENDED_BENCH_ONLY" if not blocked else "BLOCK_OPEN_ENDED_BENCH_GATE"
    approved = not blocked
    return OpenEndedBenchGate(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision=decision,
        approved_mode=APPROVED_MODE if approved else None,
        approved_planner=APPROVED_PLANNER if approved else None,
        approved_model=model if approved else None,
        approved_candidate_budget=APPROVED_CANDIDATE_BUDGET if approved else None,
        approved_max_candidates=APPROVED_MAX_CANDIDATES if approved else 0,
        approved_failure_streak_limit=APPROVED_FAILURE_STREAK_LIMIT if approved else 0,
        required_approval_token=APPROVAL_TEXT if approved else None,
        consumed_strategy_exhaustion_audit=str(strategy_exhaustion_audit),
        readiness_evidence={
            "autonomous_search_ready": readiness.autonomous_search_ready,
            "baseline_lock_status": _component_status(readiness_payload, "baseline_lock"),
            "local_gates_status": _component_status(readiness_payload, "local_gates"),
            "gate_calibration_status": _component_status(readiness_payload, "gate_calibration"),
            "modal_event_authority_status": _component_status(readiness_payload, "modal_event_authority"),
            "pending_human_actions": readiness_payload.get("pending_human_actions", []),
            "readiness_problems": readiness_payload.get("problems", []),
        },
        blocked_reasons=blocked,
        required_objectives=_required_objectives(approved=approved),
        roadmap=_roadmap(approved=approved),
        may_start_live_candidate=False,
        may_start_open_ended_loop=approved,
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _read_strategy_exhaustion(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path)
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenEndedBenchGateError(f"cannot read strategy exhaustion audit: {path}") from exc
    if not isinstance(payload, dict):
        raise OpenEndedBenchGateError("strategy exhaustion audit must be a JSON object")
    if payload.get("schema_version") != STRATEGY_EXHAUSTION_SCHEMA:
        raise OpenEndedBenchGateError("strategy exhaustion audit schema mismatch")
    if payload.get("status") != "PASS":
        raise OpenEndedBenchGateError("strategy exhaustion audit must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise OpenEndedBenchGateError(f"strategy exhaustion audit must not claim {key}=true")
    return payload


def _safe_evidence_path(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise OpenEndedBenchGateError("evidence path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/strategy_exhaustion_audit/"):
        raise OpenEndedBenchGateError("strategy exhaustion audit must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise OpenEndedBenchGateError(f"evidence must not be a symlink: {path}")
    return full


def _component_status(payload: dict[str, object], key: str) -> str | None:
    component = payload.get(key)
    if not isinstance(component, dict):
        return None
    value = component.get("status")
    return str(value) if value is not None else None


def _required_objectives(*, approved: bool) -> list[dict[str, object]]:
    if not approved:
        return [
            {
                "name": "repair_open_ended_bench_prerequisites",
                "status": "required",
                "objective": "Restore strategy exhaustion and foundation readiness before open-ended bench execution.",
                "evidence_required": "open-ended-bench-gate emits APPROVE_OPEN_ENDED_BENCH_ONLY",
            }
        ]
    return [
        {
            "name": "run_open_ended_bench",
            "status": "approved",
            "objective": "Run one LLM-planned Modal bench through the trusted orchestrator.",
            "evidence_required": f"mode=modal planner=llm max-candidates={APPROVED_MAX_CANDIDATES} with exact approval token {APPROVAL_TEXT}",
        }
    ]


def _roadmap(*, approved: bool) -> list[dict[str, object]]:
    if not approved:
        return [
            {
                "step": "stop_live_spend",
                "status": "required",
                "action": "Do not run live Modal or open-ended bench until this gate approves.",
            }
        ]
    return [
        {
            "step": "run_open_ended_bench",
            "status": "approved",
            "action": "Start one Modal LLM bench with max-candidates=3, smoke budget, and failure-streak-limit=2.",
        },
        {
            "step": "review_bench_results",
            "status": "pending",
            "action": "Run result review, Falsification Gate for any provisional KEEP, and bench-readiness review after completion.",
        },
    ]
