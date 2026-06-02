"""Global offline audit of implemented autoresearch strategy exhaustion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from autoalphafold3.surface_strategy_review import IMPLEMENTED_PLANNER_SURFACES

SCHEMA_VERSION = "autoaf3.strategy_exhaustion_audit.v1"
BENCH_READINESS_SCHEMA = "autoaf3.bench_readiness_review.v1"
FORBIDDEN_TRUE_FLAGS = (
    "starts_search",
    "writes_ledger",
    "writes_discovery_ledger",
    "official_benchmark_result",
)


class StrategyExhaustionAuditError(RuntimeError):
    """Raised when strategy evidence cannot be audited safely."""


@dataclass(frozen=True)
class StrategyExhaustionAudit:
    """JSON-friendly global implemented-strategy exhaustion report."""

    schema_version: str
    status: str
    decision: str
    implemented_planner_count: int
    exhausted_implemented_planners: list[str]
    remaining_implemented_planners: list[str]
    exhausted_surface_aliases: list[str]
    evidence_files: list[str]
    bench_readiness_review: str | None
    bench_decision: str | None
    can_start_open_ended_bench: bool | None
    required_next_step: str
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    official_benchmark_result: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_strategy_exhaustion(
    *,
    repo_root: str | Path = ".",
    evidence_root: str | Path = "runs/autoresearch",
    bench_readiness_review: str | Path | None = None,
) -> StrategyExhaustionAudit:
    """Audit all local autoresearch review/diagnosis evidence for exhausted planner families."""

    root = Path(repo_root)
    evidence_dir = _safe_evidence_root(root=root, path=evidence_root)
    evidence_files = _evidence_files(evidence_dir=evidence_dir, repo_root=root)
    exhausted_aliases: list[str] = []
    for relative_path in evidence_files:
        payload = _read_json(root=root, path=relative_path, label="strategy evidence")
        for key in ("exhausted_surfaces", "rejected_surfaces"):
            for item in _string_list(payload.get(key)):
                if item not in exhausted_aliases:
                    exhausted_aliases.append(item)
    exhausted_set = set(exhausted_aliases)
    exhausted_planners: list[str] = []
    remaining_planners: list[str] = []
    for planner, aliases in sorted(IMPLEMENTED_PLANNER_SURFACES.items()):
        if any(alias in exhausted_set for alias in aliases):
            exhausted_planners.append(planner)
        else:
            remaining_planners.append(planner)
    bench_payload = (
        _read_bench_readiness(root=root, path=bench_readiness_review)
        if bench_readiness_review is not None
        else None
    )
    can_start = (
        bool(bench_payload.get("can_start_open_ended_bench"))
        if bench_payload is not None
        else None
    )
    bench_decision = str(bench_payload.get("decision")) if bench_payload is not None else None
    if can_start is True:
        decision = "BENCH_GATE_ALREADY_APPROVES_OPEN_ENDED"
    elif remaining_planners:
        decision = "IMPLEMENTED_PLANNER_REMAINS_UNEXHAUSTED"
    else:
        decision = "NO_IMPLEMENTED_PLANNER_REMAINING"
    return StrategyExhaustionAudit(
        schema_version=SCHEMA_VERSION,
        status="PASS",
        decision=decision,
        implemented_planner_count=len(IMPLEMENTED_PLANNER_SURFACES),
        exhausted_implemented_planners=exhausted_planners,
        remaining_implemented_planners=remaining_planners,
        exhausted_surface_aliases=exhausted_aliases,
        evidence_files=[str(path) for path in evidence_files],
        bench_readiness_review=str(bench_readiness_review) if bench_readiness_review is not None else None,
        bench_decision=bench_decision,
        can_start_open_ended_bench=can_start,
        required_next_step=_required_next_step(decision),
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        official_benchmark_result=False,
    )


def _safe_evidence_root(*, root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise StrategyExhaustionAuditError("evidence root must be repo-relative without traversal")
    if candidate.as_posix() != "runs/autoresearch":
        raise StrategyExhaustionAuditError("strategy evidence root must be runs/autoresearch")
    full = root / candidate
    if full.is_symlink():
        raise StrategyExhaustionAuditError("strategy evidence root must not be a symlink")
    if not full.exists():
        raise StrategyExhaustionAuditError(f"strategy evidence root does not exist: {path}")
    return full


def _evidence_files(*, evidence_dir: Path, repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(evidence_dir.rglob("*.json")):
        if path.is_symlink():
            raise StrategyExhaustionAuditError(f"evidence file must not be a symlink: {path}")
        if any(part in {"candidates"} for part in path.parts):
            continue
        relative = path.relative_to(repo_root)
        if any(
            part
            in {
                "bench_readiness_review",
                "broader_strategy_review",
                "next_surface_review",
                "post_discard_diagnosis",
                "surface_design_review",
                "surface_strategy_review",
            }
            for part in relative.parts
        ):
            files.append(relative)
    if not files:
        raise StrategyExhaustionAuditError("no strategy evidence files found")
    return files


def _read_json(*, root: Path, path: Path, label: str) -> dict[str, object]:
    checked = root / path
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyExhaustionAuditError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise StrategyExhaustionAuditError(f"{label} must be a JSON object: {path}")
    if payload.get("status") not in {None, "PASS"}:
        raise StrategyExhaustionAuditError(f"{label} must have status=PASS when status is present: {path}")
    for key in FORBIDDEN_TRUE_FLAGS:
        if payload.get(key) is True:
            raise StrategyExhaustionAuditError(f"{label} must not claim {key}=true: {path}")
    return payload


def _read_bench_readiness(*, root: Path, path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise StrategyExhaustionAuditError("bench readiness path must be repo-relative without traversal")
    if not candidate.as_posix().startswith("runs/autoresearch/bench_readiness_review/"):
        raise StrategyExhaustionAuditError("bench readiness evidence must live under runs/autoresearch/")
    payload = _read_json(root=root, path=candidate, label="bench readiness review")
    if payload.get("schema_version") != BENCH_READINESS_SCHEMA:
        raise StrategyExhaustionAuditError("bench readiness review schema mismatch")
    return payload


def _required_next_step(decision: str) -> str:
    if decision == "BENCH_GATE_ALREADY_APPROVES_OPEN_ENDED":
        return "Use the composite bench-readiness-review gate to start the approved open-ended bench."
    if decision == "IMPLEMENTED_PLANNER_REMAINS_UNEXHAUSTED":
        return (
            "Do offline strategy review for one remaining implemented planner family before any live "
            "candidate or open-ended bench run."
        )
    return (
        "No implemented planner family remains unexhausted in local evidence. Keep the bench blocked "
        "until a genuinely new non-overlapping strategy is designed and merged."
    )


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
