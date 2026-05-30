"""Read-only pre-run readiness reporting for autonomous search start."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable

from autoalphafold3.baseline_readiness import BaselineReadinessReport, audit_baseline_readiness
from autoalphafold3.modal_assets import ModalAssetAudit, audit_modal_assets
from autoalphafold3.nanofold_checks import NanoFoldGateResult, run_nanofold_preflight_gates
from autoalphafold3.schema import FalsificationVerdict, PRIMARY_METRIC, SCORER_VERSION

PUBLIC_VAL_SPLIT = "public_val_small"
DEFAULT_CALIBRATION_PATH = Path("runs/falsification_gate_calibration.json")
HUMAN_ACTION_MARKER = "Human-approved live calibration:"
LIVE_SMOKE_MARKER = "Human-approved read-only live smoke:"


class ReadinessStatus(StrEnum):
    """Status values for readiness report sections."""

    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    NOT_REQUESTED = "NOT_REQUESTED"


@dataclass(frozen=True)
class ReadinessSection:
    """One report section with problems and pending actions."""

    status: ReadinessStatus
    problems: list[str] = field(default_factory=list)
    pending_human_action: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "problems": self.problems,
            "pending_human_action": self.pending_human_action,
            "details": self.details,
        }


@dataclass(frozen=True)
class ReadinessReport:
    """Top-level search readiness report."""

    mode: str
    autonomous_search_ready: bool
    baseline_lock: ReadinessSection
    local_gates: ReadinessSection
    gate_calibration: ReadinessSection
    live_smoke: ReadinessSection
    problems: list[str] = field(default_factory=list)
    pending_human_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "autonomous_search_ready": self.autonomous_search_ready,
            "baseline_lock": self.baseline_lock.to_dict(),
            "local_gates": self.local_gates.to_dict(),
            "gate_calibration": self.gate_calibration.to_dict(),
            "live_smoke": self.live_smoke.to_dict(),
            "problems": self.problems,
            "pending_human_actions": self.pending_human_actions,
        }


NanoFoldGateRunner = Callable[..., list[NanoFoldGateResult]]
ModalAuditRunner = Callable[[], ModalAssetAudit]


def build_readiness_report(
    *,
    repo_root: str | Path = ".",
    baseline_dir: str | Path = "runs/baseline",
    config_path: str | Path = "configs/nanofold_dev_cpu_smoke.json",
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    pending_human_calibration_action: str | None = None,
    include_live_smoke: bool = False,
    approved_live_smoke_action: str | None = None,
    nanofold_gates: list[NanoFoldGateResult] | None = None,
    modal_audit_runner: ModalAuditRunner | None = None,
) -> ReadinessReport:
    """Build a read-only report; this function never creates readiness evidence."""

    root = Path(repo_root)
    baseline = _baseline_section(root / baseline_dir)
    local_gates = _local_gates_section(
        repo_root=root,
        config_path=config_path,
        nanofold_gates=nanofold_gates,
    )
    calibration = _calibration_section(
        root / calibration_path,
        pending_human_calibration_action=pending_human_calibration_action,
    )
    live_smoke = _live_smoke_section(
        include_live_smoke=include_live_smoke,
        approved_live_smoke_action=approved_live_smoke_action,
        modal_audit_runner=modal_audit_runner,
    )

    sections = [baseline, local_gates, calibration]
    if include_live_smoke:
        sections.append(live_smoke)
    problems = [problem for section in sections for problem in section.problems]
    pending = [section.pending_human_action for section in sections if section.pending_human_action]
    autonomous_search_ready = all(section.status == ReadinessStatus.PASS for section in sections)
    return ReadinessReport(
        mode="live_smoke" if include_live_smoke else "offline",
        autonomous_search_ready=autonomous_search_ready,
        baseline_lock=baseline,
        local_gates=local_gates,
        gate_calibration=calibration,
        live_smoke=live_smoke,
        problems=problems,
        pending_human_actions=[item for item in pending if item is not None],
    )


def readiness_exit_code(report: ReadinessReport) -> int:
    """Return CLI exit code for a readiness report."""

    if report.autonomous_search_ready:
        return 0
    if report.pending_human_actions:
        return 2
    return 1


def _baseline_section(baseline_dir: Path) -> ReadinessSection:
    report = audit_baseline_readiness(baseline_dir=baseline_dir)
    details = report.to_dict()
    if report.status == "PASS":
        return ReadinessSection(status=ReadinessStatus.PASS, details=details)
    return ReadinessSection(
        status=ReadinessStatus.FAIL,
        problems=list(report.problems),
        pending_human_action=report.pending_human_action,
        details=details,
    )


def _local_gates_section(
    *,
    repo_root: Path,
    config_path: str | Path,
    nanofold_gates: list[NanoFoldGateResult] | None,
) -> ReadinessSection:
    gates = nanofold_gates
    if gates is None:
        try:
            gates = run_nanofold_preflight_gates(config_path=config_path, repo_root=repo_root)
        except Exception as exc:  # noqa: BLE001 - readiness reports failures instead of crashing.
            return ReadinessSection(
                status=ReadinessStatus.FAIL,
                problems=[f"NanoFold local gates could not run: {type(exc).__name__}: {exc}"],
                details={"config_path": str(config_path)},
            )
    problems: list[str] = []
    for gate in gates:
        if gate.status == "failed":
            problems.append(f"{gate.name} failed: {gate.reason}")
        if gate.name in {"tiny_forward", "finite_loss"} and gate.status != "passed":
            problems.append(f"{gate.name} blocks live readiness: {gate.status} ({gate.reason})")
    return ReadinessSection(
        status=ReadinessStatus.FAIL if problems else ReadinessStatus.PASS,
        problems=problems,
        details={"nanofold_gates": [gate.to_dict() for gate in gates]},
    )


def _calibration_section(
    calibration_path: Path,
    *,
    pending_human_calibration_action: str | None,
) -> ReadinessSection:
    if pending_human_calibration_action is not None:
        if not pending_human_calibration_action.startswith(HUMAN_ACTION_MARKER):
            return ReadinessSection(
                status=ReadinessStatus.FAIL,
                problems=["pending calibration action must name an exact human-approved live calibration action"],
                details={"calibration_path": str(calibration_path)},
            )
        return ReadinessSection(
            status=ReadinessStatus.PENDING,
            pending_human_action=pending_human_calibration_action,
            details={"calibration_path": str(calibration_path), "calibration_complete": False},
        )
    if not calibration_path.exists():
        return ReadinessSection(
            status=ReadinessStatus.FAIL,
            problems=["Falsification Gate calibration evidence is missing"],
            pending_human_action=(
                f"{HUMAN_ACTION_MARKER} run known-null and known-positive gate calibration with read-only/smoke "
                "Modal controls; do not write baseline, ledger, Discovery Ledger, benchmark, or metric artifacts."
            ),
            details={"calibration_path": str(calibration_path)},
        )
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ReadinessSection(
            status=ReadinessStatus.FAIL,
            problems=[f"Falsification Gate calibration evidence is unreadable: {exc}"],
            details={"calibration_path": str(calibration_path)},
        )
    problems = _validate_calibration_payload(payload)
    return ReadinessSection(
        status=ReadinessStatus.FAIL if problems else ReadinessStatus.PASS,
        problems=problems,
        details={"calibration_path": str(calibration_path), "calibration": payload},
    )


def _live_smoke_section(
    *,
    include_live_smoke: bool,
    approved_live_smoke_action: str | None,
    modal_audit_runner: ModalAuditRunner | None,
) -> ReadinessSection:
    if not include_live_smoke:
        return ReadinessSection(
            status=ReadinessStatus.NOT_REQUESTED,
            pending_human_action=(
                f"{LIVE_SMOKE_MARKER} run Modal asset audit only; no baseline, ledger, Discovery Ledger, "
                "benchmark, metric, Volume-write, or trial-run side effects."
            ),
        )
    if approved_live_smoke_action is None or not approved_live_smoke_action.startswith(LIVE_SMOKE_MARKER):
        return ReadinessSection(
            status=ReadinessStatus.PENDING,
            problems=["live readiness smoke requires an exact human-approved read-only action"],
            pending_human_action=(
                f"{LIVE_SMOKE_MARKER} run Modal asset audit only; no baseline, ledger, Discovery Ledger, "
                "benchmark, metric, Volume-write, or trial-run side effects."
            ),
        )
    audit = modal_audit_runner() if modal_audit_runner is not None else audit_modal_assets()
    details = {"approved_live_smoke_action": approved_live_smoke_action, "modal_assets": audit.to_dict()}
    if audit.status == "PASS":
        return ReadinessSection(status=ReadinessStatus.PASS, details=details)
    return ReadinessSection(status=ReadinessStatus.FAIL, problems=list(audit.problems), details=details)


def _validate_calibration_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["Falsification Gate calibration must be a JSON object"]
    problems: list[str] = []
    known_null = payload.get("known_null")
    known_positive = payload.get("known_positive")
    problems.extend(_validate_calibration_record("known_null", known_null, expected_positive=False))
    problems.extend(_validate_calibration_record("known_positive", known_positive, expected_positive=True))
    return problems


def _validate_calibration_record(name: str, value: object, *, expected_positive: bool) -> list[str]:
    if not isinstance(value, dict):
        return [f"{name} calibration record is missing"]
    if str(value.get("status", "")).lower() in {"placeholder", "pending", "todo", ""}:
        return [f"{name} calibration placeholder is not search-ready"]
    problems: list[str] = []
    verdict = value.get("verdict")
    if expected_positive:
        if verdict != FalsificationVerdict.CONFIRMED.value:
            problems.append("known_positive calibration must have CONFIRMED verdict")
    elif verdict not in {
        FalsificationVerdict.PLACEBO_KILL.value,
        FalsificationVerdict.KNOCKOUT_SURVIVES.value,
        FalsificationVerdict.AXIS_MISS.value,
        FalsificationVerdict.SEED_FRAGILE.value,
    }:
        problems.append("known_null calibration must be killed by the gate")
    if value.get("scorer_version") != SCORER_VERSION:
        problems.append(f"{name} calibration scorer_version must be {SCORER_VERSION}")
    if value.get("primary_metric") != PRIMARY_METRIC:
        problems.append(f"{name} calibration primary_metric must be {PRIMARY_METRIC}")
    if value.get("split") != PUBLIC_VAL_SPLIT:
        problems.append(f"{name} calibration split must be {PUBLIC_VAL_SPLIT}")
    for collection_name in ("manifest_hashes", "feature_fingerprints"):
        collection = value.get(collection_name)
        if not isinstance(collection, dict) or not collection:
            problems.append(f"{name} calibration {collection_name} must be non-empty")
    for field_name in ("baseline_id", "current_best_trial_id"):
        if not isinstance(value.get(field_name), str) or not value[field_name]:
            problems.append(f"{name} calibration {field_name} is required")
    control_ids = value.get("control_evidence_ids")
    if not isinstance(control_ids, list) or not control_ids or not all(isinstance(item, str) and item for item in control_ids):
        problems.append(f"{name} calibration requires control_evidence_ids")
    thresholds = value.get("gate_thresholds")
    if not isinstance(thresholds, dict):
        problems.append(f"{name} calibration gate_thresholds are required")
    else:
        for threshold_name in ("tau_attribution", "rho_placebo", "k_seed"):
            threshold = thresholds.get(threshold_name)
            if not isinstance(threshold, int | float) or not math.isfinite(float(threshold)):
                problems.append(f"{name} calibration gate_thresholds.{threshold_name} must be finite")
    if value.get("synthetic_fixture") is True:
        fixture_path = value.get("fixture_path")
        if not isinstance(fixture_path, str) or not fixture_path:
            problems.append(f"{name} synthetic calibration fixture_path is required")
        elif "/runs/" in fixture_path or fixture_path.startswith("runs/") or fixture_path.startswith("data/"):
            problems.append(f"{name} synthetic calibration fixture_path must not reference repo runs/ or data/")
    return problems
