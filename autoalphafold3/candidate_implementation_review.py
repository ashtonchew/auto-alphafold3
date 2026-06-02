"""Offline review for candidate behavior implemented after the evidence bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from autoalphafold3.evidence_bridge_review import SCHEMA_VERSION as EVIDENCE_BRIDGE_SCHEMA
from autoalphafold3.sampler import (
    SAMPLER_LOCALITY_GUARDS,
    _ca_locality_flags,
    _normalize_ca_coordinates,
    _sampler_settings,
)
from autoalphafold3.schema import AutoFoldTrial

SCHEMA_VERSION = "autoaf3.candidate_implementation_review.v1"
APPROVED_CANDIDATE = "sampler_locality_guard"
APPROVED_GUARD = "reject_exploded"


class CandidateImplementationReviewError(RuntimeError):
    """Raised when implemented candidate behavior cannot be reviewed safely."""


@dataclass(frozen=True)
class CandidateImplementationReview:
    """JSON-friendly offline candidate implementation decision."""

    schema_version: str
    status: str
    decision: str
    approved_candidate: str | None
    approved_next_step: str
    consumed_evidence_bridge_review: str
    behavior_checks: list[dict[str, object]]
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


def review_candidate_implementation(
    *,
    repo_root: str | Path = ".",
    evidence_bridge_review: str | Path,
    candidate: str = APPROVED_CANDIDATE,
) -> CandidateImplementationReview:
    """Review the merged offline/local candidate behavior without running live Modal."""

    root = Path(repo_root)
    bridge = _read_bridge_review(root=root, path=evidence_bridge_review)
    blocked: list[str] = []
    checks: list[dict[str, object]] = []
    if bridge.get("decision") != "APPROVE_NEXT_CANDIDATE_IMPLEMENTATION_PR_ONLY":
        blocked.append("evidence bridge review did not approve the next candidate implementation PR")
    if bridge.get("may_start_live_candidate") is True or bridge.get("may_start_open_ended_loop") is True:
        blocked.append("evidence bridge review must not authorize live or open-ended execution")
    if candidate != APPROVED_CANDIDATE:
        blocked.append(f"unsupported candidate implementation review: {candidate}")
    checks.extend(_sampler_locality_guard_checks())
    blocked.extend(str(check["reason"]) for check in checks if check["status"] != "PASS")
    decision = "APPROVE_LIVE_SMOKE_GATE_PR_ONLY" if not blocked else "BLOCK_CANDIDATE_IMPLEMENTATION_REVIEW"
    return CandidateImplementationReview(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision=decision,
        approved_candidate=APPROVED_CANDIDATE if not blocked else None,
        approved_next_step=(
            "Implement a separate live-smoke approval gate. Do not run live Modal until that gate "
            "explicitly approves one bounded smoke."
            if not blocked
            else "Keep live Modal and the open-ended bench blocked; repair the candidate behavior and rerun review."
        ),
        consumed_evidence_bridge_review=str(evidence_bridge_review),
        behavior_checks=checks,
        blocked_reasons=blocked,
        required_objectives=_required_objectives(blocked=blocked),
        roadmap=_roadmap(blocked=blocked),
        may_start_live_candidate=False,
        may_start_open_ended_loop=False,
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _sampler_locality_guard_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    checks.append(
        _check(
            "guard_constant",
            APPROVED_GUARD in SAMPLER_LOCALITY_GUARDS,
            "sampler locality guard reject_exploded is not registered",
        )
    )
    settings = _sampler_settings({"sampler_steps": 1, "sampler_locality_guard": APPROVED_GUARD})
    checks.append(
        _check(
            "settings_parse",
            settings.get("sampler_locality_guard") == APPROVED_GUARD,
            "sampler settings do not preserve sampler_locality_guard",
        )
    )
    exploded = [[0.0, 0.0, 0.0], [600.0, 0.0, 0.0], [1200.0, 0.0, 0.0]]
    flags = _ca_locality_flags(exploded)
    checks.append(
        _check(
            "exploded_trace_flags",
            {"adjacent_ca_distance_exploded", "pair_distance_exploded"}.issubset(set(flags)),
            "locality flags do not detect exploded adjacent and pair distances",
        )
    )
    normalized = _normalize_ca_coordinates(exploded, policy="ca_bond")
    checks.append(
        _check(
            "normalized_trace_clear",
            _ca_locality_flags(normalized) == [],
            "ca_bond-normalized trace still reports locality flags",
        )
    )
    trial = _valid_sampler_trial()
    parsed = AutoFoldTrial.model_validate(trial)
    checks.append(
        _check(
            "trial_schema_accepts_sampler_guard",
            parsed.sampler_locality_guard == APPROVED_GUARD,
            "AutoFoldTrial schema does not accept sampler locality guard for sampler trials",
        )
    )
    invalid = dict(trial)
    invalid["trial_kind"] = "debug"
    invalid["max_steps"] = 1
    invalid.pop("sampler_steps")
    invalid.pop("checkpoint_path")
    try:
        AutoFoldTrial.model_validate(invalid)
    except ValueError:
        debug_rejected = True
    else:
        debug_rejected = False
    checks.append(
        _check(
            "trial_schema_rejects_debug_guard",
            debug_rejected,
            "AutoFoldTrial schema allows sampler locality guard on debug trials",
        )
    )
    return checks


def _check(name: str, passed: bool, reason: str) -> dict[str, object]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "reason": None if passed else reason}


def _required_objectives(*, blocked: list[str]) -> list[dict[str, object]]:
    if blocked:
        return [
            {
                "name": "repair_candidate_implementation",
                "status": "required",
                "objective": "Repair the offline/local sampler locality guard behavior.",
                "evidence_required": "candidate-implementation-review emits APPROVE_LIVE_SMOKE_GATE_PR_ONLY",
            }
        ]
    return [
        {
            "name": "implement_live_smoke_approval_gate",
            "status": "required",
            "objective": (
                "Add a separate gate that can approve at most one bounded live Modal smoke after local "
                "candidate behavior, readiness, and artifact path checks pass."
            ),
            "evidence_required": "live-smoke gate PR, full local gate, foundation readiness, explicit approval token",
        }
    ]


def _roadmap(*, blocked: list[str]) -> list[dict[str, object]]:
    steps = [
        {
            "step": "stop_live_spend",
            "status": "required",
            "action": "Do not run live Modal or the open-ended bench from this candidate review.",
        }
    ]
    if blocked:
        steps.append(
            {
                "step": "repair_candidate_implementation",
                "status": "required",
                "action": "Repair sampler locality guard behavior and rerun candidate-implementation-review.",
            }
        )
    else:
        steps.append(
            {
                "step": "implement_live_smoke_approval_gate",
                "status": "required",
                "action": "Create the explicit one-candidate live-smoke approval gate before any Modal spend.",
            }
        )
    return steps


def _read_bridge_review(*, root: Path, path: str | Path) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path)
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateImplementationReviewError(f"cannot read evidence bridge review: {path}") from exc
    if not isinstance(payload, dict):
        raise CandidateImplementationReviewError("evidence bridge review must be a JSON object")
    if payload.get("schema_version") != EVIDENCE_BRIDGE_SCHEMA:
        raise CandidateImplementationReviewError("evidence bridge review schema mismatch")
    if payload.get("status") != "PASS":
        raise CandidateImplementationReviewError("evidence bridge review must have status=PASS")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise CandidateImplementationReviewError(f"evidence bridge review must not claim {key}=true")
    return payload


def _safe_evidence_path(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CandidateImplementationReviewError("evidence path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise CandidateImplementationReviewError("evidence path must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise CandidateImplementationReviewError(f"evidence must not be a symlink: {path}")
    return full


def _valid_sampler_trial() -> dict[str, object]:
    return {
        "trial_id": "T178",
        "parent_commit": "abcdef0",
        "agent_session_id": "candidate-implementation-review",
        "trial_kind": "sampler",
        "hypothesis": "Reject label-free geometry collapse before scorer input.",
        "move_family": "diffusion_sampler_golf",
        "diagnostic_target": "stability_compute",
        "prediction": {
            "causal_component": "sampler_locality_guard",
            "predicted_axis": "stability_compute",
            "predicted_direction": "up",
            "expected_lddt_delta_band": [0.0, 0.0001],
        },
        "config_path": "configs/nanofold_dev_cpu_smoke.json",
        "budget": "sampler",
        "seed": 0,
        "sampler_steps": 1,
        "sampler_locality_guard": APPROVED_GUARD,
        "max_wall_minutes": 5,
        "param_cap": 1,
        "gpu_memory_cap": 0.0,
        "cost_cap": 0.0,
        "timeout_cap": 300,
        "checkpoint_path": "runs/trials/T010/checkpoint.pt",
    }
