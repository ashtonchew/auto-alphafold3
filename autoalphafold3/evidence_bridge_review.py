"""Offline review gate for post-exhaustion evidence bridge candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

POST_EXHAUSTION_SCHEMA = "autoaf3.post_exhaustion_strategy_prd.v1"
RUN_MANIFEST_SCHEMA = "autoaf3.autoresearch_run_manifest.v1"
SUMMARY_SCHEMA = "autoaf3.autoresearch_summary.v1"
PREFLIGHT_SCHEMA = "autoaf3.autoresearch_preflight_plan.v1"
BRIDGE_PLAN_SCHEMA = "autoaf3.evidence_guided_failure_mode_bridge_plan.v1"
SCORER_SENSITIVITY_SCHEMA = "autoaf3.scorer_sensitivity.v1"
PREDICTION_GEOMETRY_SCHEMA = "autoaf3.prediction_geometry_audit.v1"
SCHEMA_VERSION = "autoaf3.evidence_bridge_review.v1"

APPROVED_PLANNER = "evidence_guided_failure_mode_bridge_diagnostic"
APPROVED_STRATEGY_FAMILY = "evidence_guided_failure_mode_bridge"
MIN_TARGET_NON_REGRESSION_CHECKS = 1
MIN_GEOMETRY_FAILURE_MODES = 1


class EvidenceBridgeReviewError(RuntimeError):
    """Raised when bridge evidence cannot be reviewed safely."""


@dataclass(frozen=True)
class EvidenceBridgeReview:
    """JSON-friendly decision for the post-exhaustion bridge candidate."""

    schema_version: str
    status: str
    decision: str
    approved_planner: str | None
    approved_strategy_family: str | None
    approved_next_step: str
    candidate_limit: int
    reviewed_trial_id: str | None
    candidate_run_dir: str
    consumed_post_exhaustion_strategy_prd: str
    consumed_scorer_sensitivity: str | None
    consumed_prediction_geometry: str | None
    target_level_non_regression_count: int
    geometry_failure_mode_count: int
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


def review_evidence_bridge(
    *,
    repo_root: str | Path = ".",
    post_exhaustion_strategy: str | Path,
    candidate_run_dir: str | Path,
) -> EvidenceBridgeReview:
    """Review a dry-run bridge candidate without authorizing live execution."""

    root = Path(repo_root)
    prd = _read_json(
        root=root,
        path=post_exhaustion_strategy,
        expected_schema=POST_EXHAUSTION_SCHEMA,
        label="post-exhaustion strategy PRD",
    )
    run_dir = _safe_run_dir(root=root, path=candidate_run_dir)
    run_manifest = _read_run_file(
        run_dir=run_dir,
        filename="run_manifest.json",
        expected_schema=RUN_MANIFEST_SCHEMA,
    )
    summary = _read_run_file(run_dir=run_dir, filename="summary.json", expected_schema=SUMMARY_SCHEMA)
    blocked = _review_prd(prd)
    blocked.extend(_review_run_manifest(run_manifest))
    candidate = _single_candidate(summary)
    blocked.extend(candidate.blocked_reasons)
    trial_id = candidate.trial_id
    config_payload: dict[str, object] | None = None
    scorer_path: str | None = None
    geometry_path: str | None = None
    target_check_count = 0
    geometry_mode_count = 0
    if trial_id is not None:
        candidate_dir = run_dir / "candidates" / trial_id
        preflight = _read_candidate_file(
            candidate_dir=candidate_dir,
            filename="preflight.json",
            expected_schema=PREFLIGHT_SCHEMA,
        )
        config_payload = _read_candidate_file(
            candidate_dir=candidate_dir,
            filename="config.json",
            expected_schema=BRIDGE_PLAN_SCHEMA,
        )
        blocked.extend(_review_preflight(preflight, trial_id=trial_id))
        blocked.extend(
            _review_config(
                config_payload,
                trial_id=trial_id,
                post_exhaustion_strategy=str(post_exhaustion_strategy),
            )
        )
        scorer_path = _optional_string(config_payload.get("source_scorer_sensitivity"))
        geometry_path = _optional_string(config_payload.get("source_geometry_audit"))
        target_checks = config_payload.get("target_level_non_regression_expectations")
        if isinstance(target_checks, list):
            target_check_count = len(target_checks)
        modes = config_payload.get("geometry_failure_modes_to_avoid")
        if isinstance(modes, list):
            geometry_mode_count = len(modes)
        blocked.extend(
            _review_bound_evidence(
                root=root,
                scorer_path=scorer_path,
                geometry_path=geometry_path,
            )
        )
    decision = (
        "APPROVE_NEXT_CANDIDATE_IMPLEMENTATION_PR_ONLY"
        if not blocked
        else "BLOCK_EVIDENCE_BRIDGE_REVIEW"
    )
    return EvidenceBridgeReview(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision=decision,
        approved_planner=APPROVED_PLANNER if not blocked else None,
        approved_strategy_family=APPROVED_STRATEGY_FAMILY if not blocked else None,
        approved_next_step=(
            "Implement one candidate PR from the reviewed evidence bridge. Keep it offline/local until "
            "a later gate explicitly approves one bounded live smoke."
            if not blocked
            else "Keep live Modal and the open-ended bench blocked; fix the bridge evidence and rerun this review."
        ),
        candidate_limit=1 if not blocked else 0,
        reviewed_trial_id=trial_id,
        candidate_run_dir=str(candidate_run_dir),
        consumed_post_exhaustion_strategy_prd=str(post_exhaustion_strategy),
        consumed_scorer_sensitivity=scorer_path,
        consumed_prediction_geometry=geometry_path,
        target_level_non_regression_count=target_check_count,
        geometry_failure_mode_count=geometry_mode_count,
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


@dataclass(frozen=True)
class _CandidateReview:
    trial_id: str | None
    blocked_reasons: list[str]


def _review_prd(payload: dict[str, object]) -> list[str]:
    blocked: list[str] = []
    if payload.get("decision") != "APPROVE_DRY_RUN_STRATEGY_PRD_ONLY":
        blocked.append("post-exhaustion strategy PRD does not approve dry-run strategy PRD only")
    if payload.get("approved_planner") != APPROVED_PLANNER:
        blocked.append("post-exhaustion strategy PRD did not approve evidence bridge planner")
    if payload.get("approved_strategy_family") != APPROVED_STRATEGY_FAMILY:
        blocked.append("post-exhaustion strategy PRD did not approve evidence bridge strategy family")
    if payload.get("candidate_limit") != 1:
        blocked.append("post-exhaustion strategy PRD must require candidate_limit=1")
    if payload.get("may_start_live_candidate") is True or payload.get("may_start_open_ended_loop") is True:
        blocked.append("post-exhaustion strategy PRD must not authorize live or open-ended execution")
    return blocked


def _review_run_manifest(payload: dict[str, object]) -> list[str]:
    blocked: list[str] = []
    if payload.get("mode") != "dry-run":
        blocked.append("bridge candidate run must be dry-run mode")
    if payload.get("planner") != APPROVED_PLANNER:
        blocked.append("bridge candidate run must use the approved evidence bridge planner")
    if payload.get("candidate_count") != 1:
        blocked.append("bridge candidate run must contain exactly one candidate")
    if payload.get("live_modal_execution") is True:
        blocked.append("bridge candidate run must not execute live Modal")
    if payload.get("target") != "NanoFold-style AlphaFold3-lite":
        blocked.append("bridge candidate run target must remain NanoFold-style AlphaFold3-lite")
    return blocked


def _single_candidate(summary: dict[str, object]) -> _CandidateReview:
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        return _CandidateReview(trial_id=None, blocked_reasons=["bridge summary must contain exactly one candidate"])
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return _CandidateReview(trial_id=None, blocked_reasons=["bridge summary candidate must be an object"])
    blocked: list[str] = []
    trial_id = _optional_string(candidate.get("trial_id"))
    if not trial_id:
        blocked.append("bridge summary candidate must name a trial_id")
    if candidate.get("planning_status") != "PLANNED":
        blocked.append("bridge summary candidate must be planning_status=PLANNED")
    if candidate.get("official_benchmark_result") is True:
        blocked.append("bridge summary candidate must not claim official benchmark result")
    if candidate.get("writes_ledger") is True or candidate.get("writes_discovery_ledger") is True:
        blocked.append("bridge summary candidate must not write ledgers")
    if candidate.get("provisional_keep") is True:
        blocked.append("bridge summary candidate must not claim provisional KEEP")
    return _CandidateReview(trial_id=trial_id, blocked_reasons=blocked)


def _review_preflight(payload: dict[str, object], *, trial_id: str) -> list[str]:
    blocked: list[str] = []
    if payload.get("trial_id") != trial_id:
        blocked.append("preflight trial_id does not match reviewed candidate")
    if payload.get("mode") != "dry-run":
        blocked.append("preflight must be dry-run mode")
    if payload.get("planning_status") != "PLANNED":
        blocked.append("preflight must be planning_status=PLANNED")
    if payload.get("status") != "DRAFT":
        blocked.append("preflight must keep candidate in DRAFT")
    if payload.get("max_templates") != 0:
        blocked.append("preflight must keep max_templates=0")
    return blocked


def _review_config(payload: dict[str, object], *, trial_id: str, post_exhaustion_strategy: str) -> list[str]:
    blocked: list[str] = []
    if payload.get("approved_planner") != APPROVED_PLANNER:
        blocked.append("bridge config must use the approved evidence bridge planner")
    if payload.get("approved_strategy_family") != APPROVED_STRATEGY_FAMILY:
        blocked.append("bridge config must use the approved evidence bridge strategy family")
    if payload.get("candidate_limit") != 1:
        blocked.append("bridge config must require candidate_limit=1")
    if payload.get("artifact_only") is not True:
        blocked.append("bridge config must be artifact_only=true")
    if payload.get("stop_before_live_modal") is not True:
        blocked.append("bridge config must stop before live Modal")
    if payload.get("not_a_benchmark_claim") is not True:
        blocked.append("bridge config must mark not_a_benchmark_claim=true")
    if payload.get("official_benchmark_result") is True:
        blocked.append("bridge config must not claim official benchmark result")
    if payload.get("starts_search") is True:
        blocked.append("bridge config must not start search")
    if payload.get("writes_ledger") is True or payload.get("writes_discovery_ledger") is True:
        blocked.append("bridge config must not write ledgers")
    if payload.get("max_templates") != 0:
        blocked.append("bridge config must keep max_templates=0")
    if payload.get("source_post_exhaustion_strategy_prd") != post_exhaustion_strategy:
        blocked.append("bridge config must bind the reviewed post-exhaustion strategy PRD")
    target_checks = payload.get("target_level_non_regression_expectations")
    if not isinstance(target_checks, list) or len(target_checks) < MIN_TARGET_NON_REGRESSION_CHECKS:
        blocked.append("bridge config must include target-level non-regression expectations")
    else:
        for index, item in enumerate(target_checks):
            if not isinstance(item, dict) or not item.get("target_id") or item.get("kill_if_regresses") is not True:
                blocked.append(
                    f"target-level non-regression expectation {index} must name target_id and kill_if_regresses=true"
                )
                break
    modes = payload.get("geometry_failure_modes_to_avoid")
    if not isinstance(modes, list) or len(modes) < MIN_GEOMETRY_FAILURE_MODES:
        blocked.append("bridge config must include geometry failure modes to avoid")
    attestation = payload.get("forbidden_edit_attestation")
    if not isinstance(attestation, dict) or attestation.get("all_forbidden_edits_unchanged") is not True:
        blocked.append("bridge config must attest forbidden edits are unchanged")
    config_path = _optional_string(payload.get("config_path"))
    if not config_path or not config_path.startswith(f"configs/experiments/{trial_id}_"):
        blocked.append("bridge config must point at the reviewed trial experiment config path")
    return blocked


def _review_bound_evidence(
    *,
    root: Path,
    scorer_path: str | None,
    geometry_path: str | None,
) -> list[str]:
    blocked: list[str] = []
    if not scorer_path:
        blocked.append("bridge config must bind scorer sensitivity evidence")
    else:
        scorer = _read_json(
            root=root,
            path=scorer_path,
            expected_schema=SCORER_SENSITIVITY_SCHEMA,
            label="scorer sensitivity",
        )
        if scorer.get("all_primary_scores_identical") is True:
            blocked.append("scorer sensitivity evidence must show primary-score sensitivity")
    if not geometry_path:
        blocked.append("bridge config must bind prediction geometry evidence")
    else:
        geometry = _read_json(
            root=root,
            path=geometry_path,
            expected_schema=PREDICTION_GEOMETRY_SCHEMA,
            label="prediction geometry",
        )
        if not isinstance(geometry.get("artifacts"), list) or not isinstance(geometry.get("reference_deltas"), list):
            blocked.append("prediction geometry evidence must include artifact and reference-delta summaries")
    return blocked


def _required_objectives(*, blocked: list[str]) -> list[dict[str, object]]:
    if blocked:
        return [
            {
                "name": "repair_evidence_bridge_candidate",
                "status": "required",
                "objective": "Regenerate the dry-run bridge candidate with bound scorer and geometry evidence.",
                "evidence_required": "evidence-bridge-review emits APPROVE_NEXT_CANDIDATE_IMPLEMENTATION_PR_ONLY",
            }
        ]
    return [
        {
            "name": "implement_evidence_guided_candidate_pr",
            "status": "required",
            "objective": (
                "Implement one candidate PR from the reviewed evidence bridge while preserving locked scorer, "
                "manifest, fingerprint, baseline, Modal resource, template, and ledger boundaries."
            ),
            "evidence_required": "candidate PR, local tests, bridge-aware bench-readiness review",
        }
    ]


def _roadmap(*, blocked: list[str]) -> list[dict[str, object]]:
    steps = [
        {
            "step": "stop_live_spend",
            "status": "required",
            "action": "Do not run live Modal or the open-ended bench from this evidence bridge review.",
        }
    ]
    if blocked:
        steps.append(
            {
                "step": "repair_bridge_evidence",
                "status": "required",
                "action": "Regenerate or fix the dry-run bridge artifact and rerun evidence-bridge-review.",
            }
        )
    else:
        steps.extend(
            [
                {
                    "step": "implement_candidate_pr",
                    "status": "required",
                    "action": "Create one offline/local candidate implementation PR using the reviewed bridge constraints.",
                },
                {
                    "step": "rerun_bench_readiness",
                    "status": "pending",
                    "action": "Rerun bench-readiness-review with the evidence bridge review attached.",
                },
            ]
        )
    return steps


def _read_run_file(*, run_dir: Path, filename: str, expected_schema: str) -> dict[str, object]:
    return _read_json_file(path=run_dir / filename, expected_schema=expected_schema, label=filename)


def _read_candidate_file(*, candidate_dir: Path, filename: str, expected_schema: str) -> dict[str, object]:
    return _read_json_file(path=candidate_dir / filename, expected_schema=expected_schema, label=filename)


def _read_json(
    *,
    root: Path,
    path: str | Path,
    expected_schema: str,
    label: str,
) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path)
    return _read_json_file(path=checked, expected_schema=expected_schema, label=label)


def _read_json_file(*, path: Path, expected_schema: str, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise EvidenceBridgeReviewError(f"{label} evidence must not be a symlink")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBridgeReviewError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise EvidenceBridgeReviewError(f"{label} must be a JSON object")
    if payload.get("schema_version") != expected_schema:
        raise EvidenceBridgeReviewError(f"{label} schema mismatch")
    if payload.get("status") not in (None, "PASS", "DRAFT"):
        raise EvidenceBridgeReviewError(f"{label} has invalid status")
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise EvidenceBridgeReviewError(f"{label} must not claim {key}=true")
    return payload


def _safe_evidence_path(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvidenceBridgeReviewError("evidence path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise EvidenceBridgeReviewError("evidence path must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise EvidenceBridgeReviewError(f"evidence must not be a symlink: {path}")
    return full


def _safe_run_dir(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvidenceBridgeReviewError("candidate run dir must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise EvidenceBridgeReviewError("candidate run dir must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise EvidenceBridgeReviewError(f"candidate run dir must not be a symlink: {path}")
    if not full.is_dir():
        raise EvidenceBridgeReviewError(f"candidate run dir does not exist: {path}")
    return full


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
