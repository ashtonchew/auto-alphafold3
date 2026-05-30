"""Read-only pre-run readiness reporting for autonomous search start."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable

from autoalphafold3.baseline_readiness import BaselineReadinessReport, audit_baseline_readiness
from autoalphafold3.modal_assets import ModalAssetAudit, audit_modal_assets
from autoalphafold3.modal_app import event_search_readiness_contract
from autoalphafold3.nanofold_checks import NanoFoldGateResult, run_nanofold_preflight_gates
from autoalphafold3.schema import FalsificationVerdict, PRIMARY_METRIC, SCORER_VERSION

PUBLIC_VAL_SPLIT = "public_val_small"
DEFAULT_CALIBRATION_PATH = Path("runs/falsification_gate_calibration.json")
DEFAULT_MODAL_AUTHORITY_PATH = Path("runs/modal_event_authority.json")
HUMAN_ACTION_MARKER = "Human-approved live calibration:"
LIVE_SMOKE_MARKER = "Human-approved read-only live smoke:"
BASELINE_LOCK_MARKER = "Human-approved baseline lock:"
LOCAL_DEPENDENCY_MARKER = "Human-approved local dependency action:"
LOCAL_GATE_MARKER = "Human-approved local gate action:"
EVENT_AUTHORITY_MARKER = "Human-approved Modal event authority:"


class ReadinessStatus(StrEnum):
    """Status values for readiness report sections."""

    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    NOT_REQUESTED = "NOT_REQUESTED"


class CertificationStatus(StrEnum):
    """Canonical evidence classes used by the final readiness report."""

    PASS_LOCAL = "PASS_LOCAL"
    PASS_MOCKED_MODAL = "PASS_MOCKED_MODAL"
    PASS_LIVE = "PASS_LIVE"
    PENDING_HUMAN_LIVE_ACTION = "PENDING_HUMAN_LIVE_ACTION"
    BLOCKED = "BLOCKED"
    NOT_REQUESTED = "NOT_REQUESTED"


@dataclass(frozen=True)
class ReadinessSection:
    """One report section with problems and pending actions."""

    status: ReadinessStatus
    certification_status: CertificationStatus
    problems: list[str] = field(default_factory=list)
    pending_human_action: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "certification_status": self.certification_status.value,
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
    mocked_modal_contract: ReadinessSection
    modal_event_authority: ReadinessSection
    local_gates: ReadinessSection
    gate_calibration: ReadinessSection
    live_smoke: ReadinessSection
    problems: list[str] = field(default_factory=list)
    pending_human_actions: list[str] = field(default_factory=list)
    certification_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "autonomous_search_ready": self.autonomous_search_ready,
            "baseline_lock": self.baseline_lock.to_dict(),
            "mocked_modal_contract": self.mocked_modal_contract.to_dict(),
            "modal_event_authority": self.modal_event_authority.to_dict(),
            "local_gates": self.local_gates.to_dict(),
            "gate_calibration": self.gate_calibration.to_dict(),
            "live_smoke": self.live_smoke.to_dict(),
            "problems": self.problems,
            "pending_human_actions": self.pending_human_actions,
            "certification_counts": self.certification_counts,
        }


NanoFoldGateRunner = Callable[..., list[NanoFoldGateResult]]
ModalAuditRunner = Callable[[], ModalAssetAudit]


def build_readiness_report(
    *,
    repo_root: str | Path = ".",
    baseline_dir: str | Path = "runs/baseline",
    config_path: str | Path = "configs/nanofold_dev_cpu_smoke.json",
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    modal_authority_path: str | Path = DEFAULT_MODAL_AUTHORITY_PATH,
    pending_human_calibration_action: str | None = None,
    include_live_smoke: bool = False,
    approved_live_smoke_action: str | None = None,
    nanofold_gates: list[NanoFoldGateResult] | None = None,
    modal_audit_runner: ModalAuditRunner | None = None,
) -> ReadinessReport:
    """Build a read-only report; this function never creates readiness evidence."""

    root = Path(repo_root)
    baseline = _baseline_section(root / baseline_dir)
    mocked_modal_contract = _mocked_modal_contract_section()
    modal_event_authority = _modal_event_authority_section(
        mocked_modal_contract,
        authority_path=root / modal_authority_path,
    )
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

    sections = [baseline, mocked_modal_contract, modal_event_authority, local_gates, calibration]
    if include_live_smoke:
        sections.append(live_smoke)
    problems = [problem for section in sections for problem in section.problems]
    pending = [section.pending_human_action for section in sections if section.pending_human_action]
    autonomous_search_ready = all(section.status == ReadinessStatus.PASS for section in sections)
    certification_counts = _certification_counts(sections + ([] if include_live_smoke else [live_smoke]))
    return ReadinessReport(
        mode="live_smoke" if include_live_smoke else "offline",
        autonomous_search_ready=autonomous_search_ready,
        baseline_lock=baseline,
        mocked_modal_contract=mocked_modal_contract,
        modal_event_authority=modal_event_authority,
        local_gates=local_gates,
        gate_calibration=calibration,
        live_smoke=live_smoke,
        problems=problems,
        pending_human_actions=[item for item in pending if item is not None],
        certification_counts=certification_counts,
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
        return ReadinessSection(
            status=ReadinessStatus.PASS,
            certification_status=CertificationStatus.PASS_LOCAL,
            details=details,
        )
    if _baseline_requires_human_lock(report):
        return ReadinessSection(
            status=ReadinessStatus.PENDING,
            certification_status=CertificationStatus.PENDING_HUMAN_LIVE_ACTION,
            problems=list(report.problems),
            pending_human_action=(
                f"{BASELINE_LOCK_MARKER} provide real locked baseline metrics, error_report, "
                "feature_fingerprints, label hashes, and provenance under runs/baseline via the approved "
                "baseline-lock procedure; do not fabricate metrics or mutate locked labels."
            ),
            details=details,
        )
    return ReadinessSection(
        status=ReadinessStatus.FAIL,
        certification_status=CertificationStatus.BLOCKED,
        problems=list(report.problems),
        pending_human_action=report.pending_human_action,
        details=details,
    )


def _mocked_modal_contract_section() -> ReadinessSection:
    contract = event_search_readiness_contract()
    problems: list[str] = []
    if contract["event_search_ready_locally"] is not False:
        problems.append("local scaffold contract must not report event search ready")
    if contract["worker_contracts_valid"] is not True:
        problems.extend(str(item) for item in contract.get("worker_contract_errors", []))
    if contract["direct_modal_run_allowed"] is not False:
        problems.append("direct agent modal run must remain forbidden")
    if contract["arbitrary_agent_sandbox_allowed"] is not False:
        problems.append("arbitrary agent sandbox access must remain forbidden")
    return ReadinessSection(
        status=ReadinessStatus.FAIL if problems else ReadinessStatus.PASS,
        certification_status=CertificationStatus.BLOCKED if problems else CertificationStatus.PASS_MOCKED_MODAL,
        problems=problems,
        details=contract,
    )


def _modal_event_authority_section(mocked_modal_contract: ReadinessSection, *, authority_path: Path) -> ReadinessSection:
    contract = mocked_modal_contract.details
    if mocked_modal_contract.status != ReadinessStatus.PASS:
        return ReadinessSection(
            status=ReadinessStatus.FAIL,
            certification_status=CertificationStatus.BLOCKED,
            problems=["Modal event authority cannot be certified until mocked Modal contracts pass"],
            details={"mocked_modal_contract_status": mocked_modal_contract.status.value},
        )
    if authority_path.exists():
        try:
            payload = json.loads(authority_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ReadinessSection(
                status=ReadinessStatus.FAIL,
                certification_status=CertificationStatus.BLOCKED,
                problems=[f"Modal event authority proof is unreadable: {exc}"],
                details={"authority_path": str(authority_path)},
            )
        problems = _validate_modal_authority_payload(payload)
        return ReadinessSection(
            status=ReadinessStatus.FAIL if problems else ReadinessStatus.PASS,
            certification_status=CertificationStatus.BLOCKED if problems else CertificationStatus.PASS_LIVE,
            problems=problems,
            details={"authority_path": str(authority_path), "authority": payload},
        )
    pending_live_action = contract.get(
        "pending_live_action",
        "deploy and authenticate the Modal-hosted trusted orchestrator before event search",
    )
    return ReadinessSection(
        status=ReadinessStatus.PENDING,
        certification_status=CertificationStatus.PENDING_HUMAN_LIVE_ACTION,
        pending_human_action=f"{EVENT_AUTHORITY_MARKER} {pending_live_action}.",
        details={
            "required_event_authority": contract.get("required_event_authority"),
            "event_search_ready_locally": contract.get("event_search_ready_locally"),
            "local_scaffold_mode": contract.get("local_scaffold_mode"),
            "authority_path": str(authority_path),
        },
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
                certification_status=CertificationStatus.BLOCKED,
                problems=[f"NanoFold local gates could not run: {type(exc).__name__}: {exc}"],
                details={"config_path": str(config_path)},
            )
    problems: list[str] = []
    blocking_skips: list[NanoFoldGateResult] = []
    failed_gate = False
    for gate in gates:
        if gate.status == "failed":
            problems.append(f"{gate.name} failed: {gate.reason}")
            failed_gate = True
        if gate.name in {"tiny_forward", "finite_loss"} and gate.status != "passed":
            problems.append(f"{gate.name} blocks live readiness: {gate.status} ({gate.reason})")
            if gate.status == "skipped":
                blocking_skips.append(gate)
    if problems and not failed_gate:
        pending_action = _local_gate_pending_action(blocking_skips, config_path=config_path)
        if pending_action is not None:
            return ReadinessSection(
                status=ReadinessStatus.PENDING,
                certification_status=CertificationStatus.PENDING_HUMAN_LIVE_ACTION,
                problems=problems,
                pending_human_action=pending_action,
                details={"nanofold_gates": [gate.to_dict() for gate in gates]},
            )
    return ReadinessSection(
        status=ReadinessStatus.FAIL if problems else ReadinessStatus.PASS,
        certification_status=CertificationStatus.BLOCKED if problems else CertificationStatus.PASS_LOCAL,
        problems=problems,
        details={"nanofold_gates": [gate.to_dict() for gate in gates]},
    )


def _local_gate_pending_action(
    blocking_skips: list[NanoFoldGateResult],
    *,
    config_path: str | Path,
) -> str | None:
    if not blocking_skips:
        return None
    reasons = {gate.reason for gate in blocking_skips}
    supported_reasons = {"dependency_missing", "feature_fixture_not_available_without_cached_arrow"}
    if not reasons.issubset(supported_reasons):
        return None
    gate_names = _format_gate_names(gate.name for gate in blocking_skips)
    actions: list[str] = []
    if "dependency_missing" in reasons:
        actions.append("install the NanoFold runtime dependencies")
    if "feature_fixture_not_available_without_cached_arrow" in reasons:
        actions.append("provide the approved cached Arrow feature fixture needed by the local finite-loss gate")
    if reasons == {"dependency_missing"}:
        return (
            f"{LOCAL_DEPENDENCY_MARKER} install the NanoFold runtime dependencies needed for {gate_names}, "
            f"then rerun `python3 -m autoalphafold3.agent readiness-report --config-path {config_path}`."
        )
    return (
        f"{LOCAL_GATE_MARKER} {' and '.join(actions)} for {gate_names}, then rerun "
        f"`python3 -m autoalphafold3.agent readiness-report --config-path {config_path}`."
    )


def _format_gate_names(names: Iterable[str]) -> str:
    values = list(names)
    if len(values) <= 1:
        return str(values[0]) if values else "the blocking gates"
    return f"{', '.join(str(value) for value in values[:-1])} and {values[-1]}"


def _calibration_section(
    calibration_path: Path,
    *,
    pending_human_calibration_action: str | None,
) -> ReadinessSection:
    if pending_human_calibration_action is not None:
        if not pending_human_calibration_action.startswith(HUMAN_ACTION_MARKER):
            return ReadinessSection(
                status=ReadinessStatus.FAIL,
                certification_status=CertificationStatus.BLOCKED,
                problems=["pending calibration action must name an exact human-approved live calibration action"],
                details={"calibration_path": str(calibration_path)},
            )
        return ReadinessSection(
            status=ReadinessStatus.PENDING,
            certification_status=CertificationStatus.PENDING_HUMAN_LIVE_ACTION,
            pending_human_action=pending_human_calibration_action,
            details={"calibration_path": str(calibration_path), "calibration_complete": False},
        )
    if not calibration_path.exists():
        return ReadinessSection(
            status=ReadinessStatus.PENDING,
            certification_status=CertificationStatus.PENDING_HUMAN_LIVE_ACTION,
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
            certification_status=CertificationStatus.BLOCKED,
            problems=[f"Falsification Gate calibration evidence is unreadable: {exc}"],
            details={"calibration_path": str(calibration_path)},
        )
    problems = _validate_calibration_payload(payload)
    return ReadinessSection(
        status=ReadinessStatus.FAIL if problems else ReadinessStatus.PASS,
        certification_status=CertificationStatus.BLOCKED if problems else CertificationStatus.PASS_LOCAL,
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
            certification_status=CertificationStatus.NOT_REQUESTED,
            pending_human_action=(
                f"{LIVE_SMOKE_MARKER} run Modal asset audit only; no baseline, ledger, Discovery Ledger, "
                "benchmark, metric, Volume-write, or trial-run side effects."
            ),
        )
    if approved_live_smoke_action is None or not approved_live_smoke_action.startswith(LIVE_SMOKE_MARKER):
        return ReadinessSection(
            status=ReadinessStatus.PENDING,
            certification_status=CertificationStatus.PENDING_HUMAN_LIVE_ACTION,
            problems=["live readiness smoke requires an exact human-approved read-only action"],
            pending_human_action=(
                f"{LIVE_SMOKE_MARKER} run Modal asset audit only; no baseline, ledger, Discovery Ledger, "
                "benchmark, metric, Volume-write, or trial-run side effects."
            ),
        )
    audit = modal_audit_runner() if modal_audit_runner is not None else audit_modal_assets()
    details = {"approved_live_smoke_action": approved_live_smoke_action, "modal_assets": audit.to_dict()}
    if audit.status == "PASS":
        return ReadinessSection(
            status=ReadinessStatus.PASS,
            certification_status=CertificationStatus.PASS_LIVE,
            details=details,
        )
    return ReadinessSection(
        status=ReadinessStatus.FAIL,
        certification_status=CertificationStatus.BLOCKED,
        problems=list(audit.problems),
        details=details,
    )


def _certification_counts(sections: list[ReadinessSection]) -> dict[str, int]:
    counts = {status.value: 0 for status in CertificationStatus}
    for section in sections:
        counts[section.certification_status.value] += 1
    return counts


def _baseline_requires_human_lock(report: BaselineReadinessReport) -> bool:
    missing_only = {
        "baseline metrics.json is missing",
        "baseline error_report.json is missing",
        "baseline feature_fingerprints.json is missing",
    }
    return bool(report.problems) and set(report.problems).issubset(missing_only)


def _validate_calibration_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["Falsification Gate calibration must be a JSON object"]
    problems: list[str] = []
    known_null = payload.get("known_null")
    known_positive = payload.get("known_positive")
    problems.extend(_validate_calibration_record("known_null", known_null, expected_positive=False))
    problems.extend(_validate_calibration_record("known_positive", known_positive, expected_positive=True))
    return problems


def _validate_modal_authority_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["Modal event authority proof must be a JSON object"]
    required = {
        "status": "PASS",
        "app_name": "autoalphafold3-modal",
        "authority_class": "TrustedOrchestrator",
        "trusted_orchestrator": True,
        "can_submit_trials": True,
        "starts_search": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "direct_modal_run_allowed": False,
        "arbitrary_agent_sandbox_allowed": False,
        "required_event_authority": "modal_hosted_trusted_orchestrator",
    }
    problems = []
    for key, expected in required.items():
        if payload.get(key) != expected:
            problems.append(f"Modal event authority proof has invalid {key}")
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
