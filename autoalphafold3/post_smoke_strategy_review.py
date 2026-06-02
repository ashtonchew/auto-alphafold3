"""Offline strategy gate after a completed bounded live smoke."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

BENCH_READINESS_SCHEMA = "autoaf3.bench_readiness_review.v1"
LIVE_SMOKE_RESULT_SCHEMA = "autoaf3.live_smoke_result_review.v1"
METRICS_SCHEMA = "autoaf3.autoresearch_comparison_metrics.v1"
SAMPLER_MANIFEST_SCHEMA = "autoaf3.sampler_manifest.v1"
SCHEMA_VERSION = "autoaf3.post_smoke_strategy_review.v1"

APPROVED_STRATEGY_FAMILY = "sampler_low_noise_locality_refinement"
APPROVED_NEXT_CANDIDATE = "sampler_low_noise_locality_refinement"
APPROVED_NEXT_PLANNER = "manual_sampler_low_noise_locality_refinement"
APPROVED_GUARD = "reject_exploded"
APPROVED_NORMALIZATION = "ca_bond"
NEXT_SAMPLER_NOISE_SCALE = 0.6
FINAL_SAMPLER_NOISE_SCALE = 0.3
NEXT_SAMPLER_NUM_SAMPLES = 4

FORBIDDEN_EDITS = [
    "autoalphafold3/scorer/**",
    "autoalphafold3/benchmark_contract.md",
    "autoalphafold3/modal_app.py",
    "public validation manifests",
    "fingerprints",
    "validation labels",
    "runs/baseline/**",
    "Modal GPU types, timeouts, max_containers, Volumes, or cost caps",
    "template database or max_templates>0",
    "canonical ledger or Discovery Ledger",
]


class PostSmokeStrategyReviewError(RuntimeError):
    """Raised when post-smoke strategy evidence is missing or unsafe."""


@dataclass(frozen=True)
class PostSmokeStrategyReview:
    """JSON-friendly next-strategy decision after one completed live smoke."""

    schema_version: str
    status: str
    decision: str
    reviewed_trial_id: str
    reviewed_run_dir: str
    live_smoke_result_review: str
    bench_readiness_review: str
    approved_strategy_family: str | None
    approved_next_candidate: str | None
    approved_next_planner: str | None
    candidate_limit: int
    candidate_score: float | None
    global_baseline_delta: float | None
    global_current_best_score: float | None
    num_scored_targets: int | None
    num_failed_targets: int | None
    sampler_manifest_summary: dict[str, object]
    next_candidate_plan: dict[str, object]
    blocked_reasons: list[str]
    required_objectives: list[dict[str, object]]
    roadmap: list[dict[str, object]]
    forbidden_edits: list[str]
    may_start_live_candidate: bool
    may_start_open_ended_loop: bool
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def review_post_smoke_strategy(
    *,
    repo_root: str | Path = ".",
    live_smoke_result_review: str | Path,
    bench_readiness_review: str | Path,
) -> PostSmokeStrategyReview:
    """Review one scored smoke and define the next bounded strategy objective."""

    root = Path(repo_root)
    live_result = _read_evidence(
        root=root,
        path=live_smoke_result_review,
        expected_schema=LIVE_SMOKE_RESULT_SCHEMA,
        label="live smoke result review",
    )
    bench = _read_evidence(
        root=root,
        path=bench_readiness_review,
        expected_schema=BENCH_READINESS_SCHEMA,
        label="bench readiness review",
    )
    run_dir = _safe_run_dir(root=root, path=_required_text(live_result.get("reviewed_run_dir"), "reviewed_run_dir"))
    trial_id = _required_text(live_result.get("reviewed_trial_id"), "reviewed_trial_id")
    candidate_dir = run_dir / "candidates" / trial_id
    metrics = _read_run_file(candidate_dir / "metrics.json", expected_schema=METRICS_SCHEMA, label="metrics")
    sampler = _read_run_file(
        candidate_dir / "sampler_manifest.json",
        expected_schema=SAMPLER_MANIFEST_SCHEMA,
        label="sampler manifest",
    )
    next_noise_scale = _next_sampler_noise_scale(sampler)
    blocked = _blocked_reasons(
        live_result=live_result,
        bench=bench,
        metrics=metrics,
        sampler=sampler,
        next_noise_scale=next_noise_scale,
    )
    approved = not blocked
    candidate_score = _optional_float(_comparison(metrics).get("candidate_score"))
    global_delta = _optional_float(_comparison(metrics).get("global_baseline_delta"))
    current_best = _optional_float(_comparison(metrics).get("global_current_best_score"))
    num_scored = _optional_int(_metrics(metrics).get("num_scored_targets"))
    num_failed = _optional_int(_metrics(metrics).get("num_failed_targets"))
    return PostSmokeStrategyReview(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision="APPROVE_NEXT_BOUNDED_CANDIDATE_PLAN_ONLY" if approved else "BLOCK_POST_SMOKE_STRATEGY_REVIEW",
        reviewed_trial_id=trial_id,
        reviewed_run_dir=str(live_result.get("reviewed_run_dir")),
        live_smoke_result_review=str(live_smoke_result_review),
        bench_readiness_review=str(bench_readiness_review),
        approved_strategy_family=APPROVED_STRATEGY_FAMILY if approved else None,
        approved_next_candidate=APPROVED_NEXT_CANDIDATE if approved else None,
        approved_next_planner=APPROVED_NEXT_PLANNER if approved else None,
        candidate_limit=1 if approved else 0,
        candidate_score=candidate_score,
        global_baseline_delta=global_delta,
        global_current_best_score=current_best,
        num_scored_targets=num_scored,
        num_failed_targets=num_failed,
        sampler_manifest_summary=_sampler_summary(sampler),
        next_candidate_plan=_next_candidate_plan(trial_id=trial_id, next_noise_scale=next_noise_scale) if approved else {},
        blocked_reasons=blocked,
        required_objectives=_required_objectives(approved=approved, trial_id=trial_id),
        roadmap=_roadmap(approved=approved, trial_id=trial_id),
        forbidden_edits=list(FORBIDDEN_EDITS),
        may_start_live_candidate=False,
        may_start_open_ended_loop=False,
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _read_evidence(
    *,
    root: Path,
    path: str | Path,
    expected_schema: str,
    label: str,
) -> dict[str, object]:
    checked = _safe_evidence_path(root=root, path=path, label=label)
    payload = _read_json(checked, label=label)
    if payload.get("schema_version") != expected_schema:
        raise PostSmokeStrategyReviewError(f"{label} schema mismatch")
    if payload.get("status") != "PASS":
        raise PostSmokeStrategyReviewError(f"{label} must have status=PASS")
    _refuse_authority_claims(payload, label=label)
    if payload.get("may_start_live_candidate") is True or payload.get("may_start_open_ended_loop") is True:
        raise PostSmokeStrategyReviewError(f"{label} must not authorize live or open-ended execution")
    return payload


def _safe_evidence_path(*, root: Path, path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PostSmokeStrategyReviewError(f"{label} path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise PostSmokeStrategyReviewError(f"{label} path must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink():
        raise PostSmokeStrategyReviewError(f"{label} path must not be a symlink: {path}")
    return full


def _safe_run_dir(*, root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PostSmokeStrategyReviewError("reviewed run dir must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/"):
        raise PostSmokeStrategyReviewError("reviewed run dir must live under runs/autoresearch/")
    full = root / candidate
    if full.is_symlink() or not full.exists():
        raise PostSmokeStrategyReviewError(f"reviewed run dir is missing or unsafe: {path}")
    return full


def _read_run_file(path: Path, *, expected_schema: str, label: str) -> dict[str, object]:
    payload = _read_json(path, label=label)
    if payload.get("schema_version") != expected_schema:
        raise PostSmokeStrategyReviewError(f"{label} schema mismatch")
    _refuse_authority_claims(payload, label=label)
    return payload


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise PostSmokeStrategyReviewError(f"{label} must not be a symlink")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostSmokeStrategyReviewError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PostSmokeStrategyReviewError(f"{label} must be a JSON object")
    return payload


def _refuse_authority_claims(payload: dict[str, object], *, label: str) -> None:
    for key in ("starts_search", "writes_ledger", "writes_discovery_ledger", "official_benchmark_result"):
        if payload.get(key) is True:
            raise PostSmokeStrategyReviewError(f"{label} must not claim {key}=true")


def _blocked_reasons(
    *,
    live_result: dict[str, object],
    bench: dict[str, object],
    metrics: dict[str, object],
    sampler: dict[str, object],
    next_noise_scale: float | None,
) -> list[str]:
    blocked: list[str] = []
    if live_result.get("decision") != "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED":
        blocked.append("live smoke result review must be a scored DISCARD")
    if live_result.get("smoke_status") != "DISCARD" or live_result.get("result_status") != "SCORED":
        blocked.append("live smoke result must have smoke_status=DISCARD and result_status=SCORED")
    if live_result.get("provisional_keep") is True:
        blocked.append("provisional KEEP requires falsification, not a next strategy")
    if bench.get("decision") != "BLOCK_OPEN_ENDED_BENCH_LIVE_SMOKE_DISCARDED":
        blocked.append("bench readiness must block on the completed live-smoke DISCARD")
    if bench.get("autonomous_search_ready") is not True or bench.get("can_start_open_ended_bench") is True:
        blocked.append("bench readiness must be foundation-green while keeping open-ended bench blocked")
    comparison = _comparison(metrics)
    if comparison.get("status") != "DISCARD" or comparison.get("provisional_keep") is True:
        blocked.append("metrics comparison must be terminal DISCARD without provisional KEEP")
    metric_values = _metrics(metrics)
    if _optional_int(metric_values.get("num_failed_targets")) not in (0, None):
        blocked.append("discarded smoke must not have failed scored targets before strategy refinement")
    if _optional_int(metric_values.get("num_scored_targets")) != _optional_int(metric_values.get("num_targets")):
        blocked.append("discarded smoke must score every target before strategy refinement")
    if sampler.get("status") != "SAMPLER_PREDICTED":
        blocked.append("sampler manifest must report SAMPLER_PREDICTED")
    if sampler.get("sampler_locality_guard") != APPROVED_GUARD:
        blocked.append("sampler manifest must preserve reject_exploded locality guard")
    if sampler.get("sampler_coordinate_normalization") != APPROVED_NORMALIZATION:
        blocked.append("sampler manifest must preserve ca_bond coordinate normalization")
    if _optional_int(sampler.get("sampler_num_samples")) is None or _optional_int(sampler.get("sampler_num_samples")) < 2:
        blocked.append("sampler manifest must show multi-sample geometry selection")
    if next_noise_scale is None:
        blocked.append("sampler noise refinement ladder is exhausted for this source smoke")
    if sampler.get("max_templates") != 0:
        blocked.append("sampler manifest must keep max_templates=0")
    return blocked


def _comparison(payload: dict[str, object]) -> dict[str, object]:
    comparison = payload.get("comparison")
    return comparison if isinstance(comparison, dict) else {}


def _metrics(payload: dict[str, object]) -> dict[str, object]:
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _sampler_summary(payload: dict[str, object]) -> dict[str, object]:
    keys = [
        "sampler_noise_scale",
        "sampler_num_samples",
        "sampler_selection_policy",
        "sampler_coordinate_normalization",
        "sampler_coordinate_scale",
        "sampler_locality_guard",
        "max_templates",
        "prediction_count",
    ]
    return {key: payload.get(key) for key in keys}


def _next_candidate_plan(*, trial_id: str, next_noise_scale: float | None) -> dict[str, object]:
    if next_noise_scale is None:
        return {}
    return {
        "source_trial_id": trial_id,
        "candidate_limit": 1,
        "approved_planner": APPROVED_NEXT_PLANNER,
        "approved_strategy_family": APPROVED_STRATEGY_FAMILY,
        "candidate_intent": (
            "Keep the locality guard and ca_bond normalization that prevented geometry collapse, "
            "but reduce sampler noise to test whether the scored DISCARD was over-noised rather than "
            "invalid geometry."
        ),
        "required_sampler_settings": {
            "sampler_locality_guard": APPROVED_GUARD,
            "sampler_coordinate_normalization": APPROVED_NORMALIZATION,
            "sampler_coordinate_scale": 1.0,
            "sampler_selection_policy": "geometry",
            "sampler_num_samples": NEXT_SAMPLER_NUM_SAMPLES,
            "sampler_noise_scale": next_noise_scale,
            "max_templates": 0,
        },
        "stop_conditions": [
            "any generated sample triggers adjacent_ca_distance_exploded or pair_distance_exploded",
            "candidate plan attempts scorer, baseline, Modal resource, manifest, fingerprint, template, or ledger edits",
            "dry-run planning does not produce exactly one candidate",
            "fresh live-smoke gate does not explicitly approve one bounded candidate",
        ],
    }


def _next_sampler_noise_scale(sampler: dict[str, object]) -> float | None:
    current = _optional_float(sampler.get("sampler_noise_scale"))
    if current is None:
        return None
    if current > NEXT_SAMPLER_NOISE_SCALE:
        return NEXT_SAMPLER_NOISE_SCALE
    if current > FINAL_SAMPLER_NOISE_SCALE:
        return FINAL_SAMPLER_NOISE_SCALE
    return None


def _required_objectives(*, approved: bool, trial_id: str) -> list[dict[str, object]]:
    if not approved:
        return [
            {
                "name": "repair_post_smoke_strategy_inputs",
                "status": "required",
                "objective": "Repair or regenerate post-smoke strategy inputs before planning another candidate.",
                "evidence_required": "post-smoke-strategy-review emits APPROVE_NEXT_BOUNDED_CANDIDATE_PLAN_ONLY",
            }
        ]
    return [
        {
            "name": "create_next_bounded_candidate_plan",
            "status": "required",
            "objective": f"Create one low-noise locality-refinement candidate plan after scored DISCARD {trial_id}.",
            "evidence_required": "candidate plan artifact with candidate_limit=1, no forbidden edits, max_templates=0, and no live authority",
        }
    ]


def _roadmap(*, approved: bool, trial_id: str) -> list[dict[str, object]]:
    steps = [
        {
            "step": "stop_live_spend",
            "status": "required",
            "action": "Do not run another live candidate or the open-ended bench from this strategy review.",
        }
    ]
    if approved:
        steps.append(
            {
                "step": "create_next_bounded_candidate_plan",
                "status": "required",
                "action": f"Create one low-noise locality-refinement plan using {trial_id} as the source smoke.",
            }
        )
        steps.append(
            {
                "step": "rerun_live_smoke_gate",
                "status": "pending",
                "action": "Only a fresh live-smoke gate may approve the next one-candidate Modal smoke.",
            }
        )
    else:
        steps.append(
            {
                "step": "repair_inputs",
                "status": "required",
                "action": "Fix the post-smoke evidence inputs and rerun this offline review.",
            }
        )
    return steps


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PostSmokeStrategyReviewError(f"post-smoke evidence missing {name}")
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None
