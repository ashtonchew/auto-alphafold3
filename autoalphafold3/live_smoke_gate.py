"""Offline gate for authorizing at most one bounded live Modal smoke."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from autoalphafold3.candidate_implementation_review import SCHEMA_VERSION as CANDIDATE_IMPLEMENTATION_SCHEMA
from autoalphafold3.post_smoke_strategy_review import SCHEMA_VERSION as POST_SMOKE_STRATEGY_SCHEMA
from autoalphafold3.readiness import build_readiness_report

SCHEMA_VERSION = "autoaf3.live_smoke_gate.v1"
APPROVAL_TOKEN = "I_APPROVE_AUTORESEARCH_LIVE_SEARCH"
APPROVED_CANDIDATE = "sampler_locality_guard"


class LiveSmokeGateError(RuntimeError):
    """Raised when a bounded live-smoke approval cannot be produced safely."""


@dataclass(frozen=True)
class LiveSmokeGate:
    """JSON-friendly bounded live-smoke decision."""

    schema_version: str
    status: str
    decision: str
    approved_candidate: str | None
    candidate_limit: int
    required_approval_token: str | None
    consumed_candidate_implementation_review: str
    consumed_post_smoke_strategy_review: str | None
    consumed_candidate_plan: str | None
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


def review_live_smoke_gate(
    *,
    repo_root: str | Path = ".",
    candidate_implementation_review: str | Path,
    baseline_dir: str | Path = "runs/baseline",
    config_path: str | Path = "configs/nanofold_dev_cpu_smoke.json",
    calibration_path: str | Path = "runs/falsification_gate_calibration.json",
    modal_authority_path: str | Path = "runs/modal_event_authority.json",
    post_smoke_strategy_review: str | Path | None = None,
    candidate_plan: str | Path | None = None,
) -> LiveSmokeGate:
    """Approve at most one bounded live smoke, never an open-ended bench."""

    root = Path(repo_root)
    candidate_review = _read_candidate_implementation(root=root, path=candidate_implementation_review)
    post_smoke_strategy = (
        _read_post_smoke_strategy(root=root, path=post_smoke_strategy_review)
        if post_smoke_strategy_review is not None
        else None
    )
    candidate_plan_payload = (
        _read_candidate_plan(root=root, path=candidate_plan)
        if candidate_plan is not None
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
    blocked: list[str] = []
    if candidate_review.get("decision") != "APPROVE_LIVE_SMOKE_GATE_PR_ONLY":
        blocked.append("candidate implementation review did not approve live-smoke gate PR only")
    if candidate_review.get("approved_candidate") != APPROVED_CANDIDATE:
        blocked.append("candidate implementation review did not approve sampler locality guard")
    if not readiness.autonomous_search_ready:
        blocked.append("foundation readiness is not autonomous_search_ready=true")
    for component in ("baseline_lock", "local_gates", "gate_calibration", "modal_event_authority"):
        status = _component_status(readiness_payload, component)
        if status != "PASS":
            blocked.append(f"{component} status is not PASS")
    if readiness_payload.get("pending_human_actions"):
        blocked.append("readiness report still has pending human actions")
    if (post_smoke_strategy is None) != (candidate_plan_payload is None):
        blocked.append("post-smoke live gate requires both post-smoke strategy review and candidate plan")
    if post_smoke_strategy is not None and candidate_plan_payload is not None:
        blocked.extend(
            _review_post_smoke_candidate_plan(
                post_smoke_strategy=post_smoke_strategy,
                candidate_plan=candidate_plan_payload,
            )
        )
    decision = "APPROVE_BOUNDED_LIVE_SMOKE_ONLY" if not blocked else "BLOCK_LIVE_SMOKE_GATE"
    return LiveSmokeGate(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision=decision,
        approved_candidate=APPROVED_CANDIDATE if not blocked else None,
        candidate_limit=1 if not blocked else 0,
        required_approval_token=APPROVAL_TOKEN if not blocked else None,
        consumed_candidate_implementation_review=str(candidate_implementation_review),
        consumed_post_smoke_strategy_review=(
            str(post_smoke_strategy_review) if post_smoke_strategy_review is not None else None
        ),
        consumed_candidate_plan=str(candidate_plan) if candidate_plan is not None else None,
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
        required_objectives=_required_objectives(blocked=blocked),
        roadmap=_roadmap(blocked=blocked),
        may_start_live_candidate=not blocked,
        may_start_open_ended_loop=False,
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _required_objectives(*, blocked: list[str]) -> list[dict[str, object]]:
    if blocked:
        return [
            {
                "name": "repair_live_smoke_prerequisites",
                "status": "required",
                "objective": "Repair candidate implementation or foundation readiness before live Modal spend.",
                "evidence_required": "live-smoke-gate emits APPROVE_BOUNDED_LIVE_SMOKE_ONLY",
            }
        ]
    return [
        {
            "name": "run_one_bounded_live_smoke",
            "status": "approved",
            "objective": "Run exactly one bounded live Modal smoke through the trusted orchestrator.",
            "evidence_required": f"operator supplies exact approval token {APPROVAL_TOKEN}",
        }
    ]


def _roadmap(*, blocked: list[str]) -> list[dict[str, object]]:
    if blocked:
        return [
            {
                "step": "stop_live_spend",
                "status": "required",
                "action": "Do not run live Modal until live-smoke-gate approves one bounded smoke.",
            }
        ]
    return [
        {
            "step": "run_one_bounded_live_smoke",
            "status": "approved",
            "action": "Run one Modal smoke with max-candidates=1 and the exact approval token.",
        },
        {
            "step": "rerun_bench_readiness",
            "status": "pending",
            "action": "Rerun bench-readiness-review after the smoke evidence exists.",
        },
    ]


def _read_candidate_implementation(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path)
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveSmokeGateError(f"cannot read candidate implementation review: {path}") from exc
    if not isinstance(payload, dict):
        raise LiveSmokeGateError("candidate implementation review must be a JSON object")
    if payload.get("schema_version") != CANDIDATE_IMPLEMENTATION_SCHEMA:
        raise LiveSmokeGateError("candidate implementation review schema mismatch")
    if payload.get("status") != "PASS":
        raise LiveSmokeGateError("candidate implementation review must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise LiveSmokeGateError(f"candidate implementation review must not claim {key}=true")
    if payload.get("may_start_live_candidate") is True or payload.get("may_start_open_ended_loop") is True:
        raise LiveSmokeGateError("candidate implementation review must not authorize live or open-ended execution")
    return payload


def _read_post_smoke_strategy(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path)
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveSmokeGateError(f"cannot read post-smoke strategy review: {path}") from exc
    if not isinstance(payload, dict):
        raise LiveSmokeGateError("post-smoke strategy review must be a JSON object")
    if payload.get("schema_version") != POST_SMOKE_STRATEGY_SCHEMA:
        raise LiveSmokeGateError("post-smoke strategy review schema mismatch")
    if payload.get("status") != "PASS":
        raise LiveSmokeGateError("post-smoke strategy review must have status=PASS")
    _refuse_authority_claims(payload, label="post-smoke strategy review")
    if payload.get("may_start_live_candidate") is True or payload.get("may_start_open_ended_loop") is True:
        raise LiveSmokeGateError("post-smoke strategy review must not authorize live or open-ended execution")
    return payload


def _read_candidate_plan(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_candidate_plan_path(root=root, path=path)
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveSmokeGateError(f"cannot read candidate plan: {path}") from exc
    if not isinstance(payload, dict):
        raise LiveSmokeGateError("candidate plan must be a JSON object")
    _refuse_authority_claims(payload, label="candidate plan")
    config = payload.get("config")
    if isinstance(config, dict):
        _refuse_authority_claims(config, label="candidate plan config")
    return payload


def _review_post_smoke_candidate_plan(
    *,
    post_smoke_strategy: dict[str, object],
    candidate_plan: dict[str, object],
) -> list[str]:
    blocked: list[str] = []
    if post_smoke_strategy.get("decision") != "APPROVE_NEXT_BOUNDED_CANDIDATE_PLAN_ONLY":
        blocked.append("post-smoke strategy review did not approve next bounded candidate plan only")
    if post_smoke_strategy.get("candidate_limit") != 1:
        blocked.append("post-smoke strategy review must require candidate_limit=1")
    next_plan = post_smoke_strategy.get("next_candidate_plan")
    if not isinstance(next_plan, dict):
        blocked.append("post-smoke strategy review must include next_candidate_plan")
        next_plan = {}
    required = next_plan.get("required_sampler_settings")
    if not isinstance(required, dict):
        blocked.append("post-smoke strategy review must include required sampler settings")
        required = {}
    trial = candidate_plan.get("trial")
    config = candidate_plan.get("config")
    if not isinstance(trial, dict):
        blocked.append("candidate plan must contain trial object")
        trial = {}
    if not isinstance(config, dict):
        blocked.append("candidate plan must contain config object")
        config = {}
    if trial.get("trial_kind") != "sampler":
        blocked.append("candidate plan must be a sampler trial")
    if config.get("candidate_limit") != 1:
        blocked.append("candidate plan config must require candidate_limit=1")
    if config.get("approved_strategy_family") != post_smoke_strategy.get("approved_strategy_family"):
        blocked.append("candidate plan strategy family must match post-smoke strategy")
    for key, value in required.items():
        if key != "max_templates" and trial.get(key) != value:
            blocked.append(f"candidate trial {key} does not match post-smoke strategy")
        if config.get(key) != value:
            blocked.append(f"candidate config {key} does not match post-smoke strategy")
    if trial.get("checkpoint_path") != "runs/trials/T010/checkpoint.pt":
        blocked.append("candidate plan must use the locked T010 checkpoint")
    if config.get("source_trial_id") != post_smoke_strategy.get("reviewed_trial_id"):
        blocked.append("candidate plan must cite the reviewed live-smoke source trial")
    return blocked


def _refuse_authority_claims(payload: dict[str, object], *, label: str) -> None:
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise LiveSmokeGateError(f"{label} must not claim {key}=true")


def _safe_evidence_path(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise LiveSmokeGateError("evidence path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise LiveSmokeGateError("evidence path must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise LiveSmokeGateError(f"evidence must not be a symlink: {path}")
    return full


def _safe_candidate_plan_path(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise LiveSmokeGateError("candidate plan path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("configs/experiments/"):
        raise LiveSmokeGateError("candidate plan must live under configs/experiments/")
    full = root / candidate
    if full.is_symlink():
        raise LiveSmokeGateError(f"candidate plan must not be a symlink: {path}")
    return full


def _component_status(payload: dict[str, object], key: str) -> str | None:
    component = payload.get(key)
    if not isinstance(component, dict):
        return None
    value = component.get("status")
    return str(value) if value is not None else None
