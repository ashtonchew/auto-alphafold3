"""Fakeable Modal adapter for orchestrator-owned Falsification Gate waves."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoalphafold3.modal_app import APP_NAME
from autoalphafold3.schema import (
    FalsificationPlan,
    FoldCartographerReport,
    TrialStatus,
)

DEFAULT_GATE_CLASS = "TrialRunner"
DEFAULT_GATE_METHOD = "run"
DEFAULT_GATE_TIMEOUT_SECONDS = 600
DEFAULT_MAX_GATE_VARIANTS = 8
DEFAULT_GATE_AGGREGATE_TIMEOUT_SECONDS = DEFAULT_GATE_TIMEOUT_SECONDS * DEFAULT_MAX_GATE_VARIANTS
MAX_GATE_SEEDS = 5
LOCKED_PATH_TOKENS = (
    "autoalphafold3-locked",
    "locked/labels",
    "public_val_labels",
    "hidden_val",
    "data/labels",
)


class GateWaveError(ValueError):
    """Raised when a gate wave is unsafe to submit."""


class GateControlKind(StrEnum):
    """Orchestrator-owned control variants for one provisional KEEP."""

    KNOCKOUT = "knockout"
    PLACEBO = "placebo"
    AXIS_CHECK = "axis_check"
    SEED_RERUN = "seed_rerun"


class GateControl(BaseModel):
    """One Modal gate-control input, authored only by the local orchestrator."""

    model_config = ConfigDict(extra="forbid")

    gate_id: str = Field(min_length=1)
    candidate_trial_id: str = Field(pattern=r"^T[0-9]{3,}$")
    authored_by: str = Field(default="orchestrator", pattern=r"^orchestrator$")
    control_kind: GateControlKind
    seed: int = Field(ge=0)
    timeout_seconds: int = Field(default=DEFAULT_GATE_TIMEOUT_SECONDS, ge=1)
    payload: dict[str, object]

    @model_validator(mode="after")
    def validate_payload_boundary(self) -> GateControl:
        _reject_locked_payload(self.payload)
        max_templates = self.payload.get("max_templates")
        if max_templates is not None and max_templates != 0:
            raise ValueError("gate controls must preserve max_templates=0")
        if self.payload.get("candidate_trial_id") != self.candidate_trial_id:
            raise ValueError("gate control payload candidate_trial_id must match control metadata")
        if self.payload.get("control_kind") != self.control_kind.value:
            raise ValueError("gate control payload control_kind must match control metadata")
        if self.payload.get("seed") != self.seed:
            raise ValueError("gate control payload seed must match control metadata")
        return self


class GateControlEvidence(BaseModel):
    """Structured evidence returned for one gate control."""

    model_config = ConfigDict(extra="forbid")

    gate_id: str = Field(min_length=1)
    candidate_trial_id: str = Field(pattern=r"^T[0-9]{3,}$")
    control_kind: GateControlKind
    seed: int = Field(ge=0)
    status: TrialStatus
    metrics: dict[str, object] = Field(default_factory=dict)
    fold_cartographer: FoldCartographerReport
    failure_signature: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class GateWaveReport(BaseModel):
    """Result of submitting a bounded Falsification Gate control wave."""

    model_config = ConfigDict(extra="forbid")

    candidate_trial_id: str = Field(pattern=r"^T[0-9]{3,}$")
    status: TrialStatus
    controls: list[GateControlEvidence] = Field(min_length=1)


class GateWaveFunction(Protocol):
    """Subset of Modal Function used by the adapter."""

    def starmap(
        self,
        input_iterator: Sequence[tuple[dict[str, object], int]],
        *,
        kwargs: dict[str, object] | None = None,
        order_outputs: bool = True,
        return_exceptions: bool = False,
        wrap_returned_exceptions: bool | None = None,
    ) -> object:
        """Run controls in parallel and return an iterable of control results."""


def build_gate_wave_controls(
    *,
    plan: FalsificationPlan,
    base_payload: dict[str, object],
    max_variants: int = DEFAULT_MAX_GATE_VARIANTS,
    timeout_seconds: int = DEFAULT_GATE_TIMEOUT_SECONDS,
) -> list[GateControl]:
    """Build bounded orchestrator-owned controls for one provisional KEEP."""

    if plan.authored_by != "orchestrator":
        raise GateWaveError("gate controls must be orchestrator-authored")
    if plan.n_seeds > MAX_GATE_SEEDS:
        raise GateWaveError(f"gate seed count must be <= {MAX_GATE_SEEDS}")
    if timeout_seconds <= 0:
        raise GateWaveError("gate timeout_seconds must be positive")

    controls: list[GateControl] = []
    fixed_controls = (
        (GateControlKind.KNOCKOUT, 0, {"knockout_patch": plan.knockout_patch}),
        (GateControlKind.PLACEBO, 0, {"placebo_family": plan.placebo_family.value}),
        (
            GateControlKind.AXIS_CHECK,
            0,
            {
                "tau_attribution": plan.tau_attribution,
                "rho_placebo": plan.rho_placebo,
                "k_seed": plan.k_seed,
            },
        ),
    )
    for kind, seed, extra in fixed_controls:
        controls.append(
            _control_from_payload(
                plan=plan,
                kind=kind,
                seed=seed,
                base_payload=base_payload,
                extra=extra,
                timeout_seconds=timeout_seconds,
            )
        )
    for seed in range(plan.n_seeds):
        controls.append(
            _control_from_payload(
                plan=plan,
                kind=GateControlKind.SEED_RERUN,
                seed=seed,
                base_payload=base_payload,
                extra={"seed_index": seed},
                timeout_seconds=timeout_seconds,
            )
        )

    if len(controls) > max_variants:
        raise GateWaveError(f"gate wave has {len(controls)} variants, max is {max_variants}")
    return controls


def run_gate_wave(function: GateWaveFunction, controls: Sequence[GateControl]) -> GateWaveReport:
    """Run a fake or Modal-like function with the required starmap contract."""

    return run_gate_wave_with_timeout(
        function,
        controls,
        aggregate_timeout_seconds=_aggregate_timeout_seconds(controls),
    )


def run_gate_wave_with_timeout(
    function: GateWaveFunction,
    controls: Sequence[GateControl],
    *,
    aggregate_timeout_seconds: int,
) -> GateWaveReport:
    """Run a bounded gate wave with an explicit aggregate timeout contract."""

    if aggregate_timeout_seconds <= 0:
        raise GateWaveError("gate aggregate_timeout_seconds must be positive")
    bounded = _validate_controls(controls)
    requested_timeout = sum(control.timeout_seconds for control in bounded)
    if requested_timeout > aggregate_timeout_seconds:
        raise GateWaveError(
            f"gate requested timeout {requested_timeout}s exceeds aggregate timeout {aggregate_timeout_seconds}s"
        )
    try:
        raw_results = list(
            function.starmap(
                [(control.payload, control.seed) for control in bounded],
                kwargs={"aggregate_timeout_seconds": aggregate_timeout_seconds},
                order_outputs=True,
                return_exceptions=True,
                wrap_returned_exceptions=False,
            )
        )
    except BaseException as exc:  # noqa: BLE001 - external adapter failures become infra evidence.
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise
        return _infra_report(bounded, "modal_starmap", exc)

    if len(raw_results) != len(bounded):
        return _infra_report(bounded, "modal_gate_wave_result_count", RuntimeError("result count mismatch"))
    evidence = [_evidence_from_result(control, raw) for control, raw in zip(bounded, raw_results, strict=True)]
    return GateWaveReport(candidate_trial_id=bounded[0].candidate_trial_id, status=_wave_status(evidence), controls=evidence)


def run_modal_gate_wave(
    controls: Sequence[GateControl],
    *,
    modal_module: object | None = None,
    class_name: str = DEFAULT_GATE_CLASS,
    method_name: str = DEFAULT_GATE_METHOD,
) -> GateWaveReport:
    """Look up the deployed Modal trial runner and submit a bounded wave."""

    bounded = _validate_controls(controls)
    if modal_module is None:
        try:
            import modal as modal_module
        except ModuleNotFoundError:
            return _infra_report(bounded, "modal_sdk_missing")
    try:
        runner_cls = modal_module.Cls.from_name(APP_NAME, class_name)  # type: ignore[attr-defined]
        runner = runner_cls()
        method = getattr(runner, method_name)
    except Exception as exc:  # noqa: BLE001 - deployed lookup failures are infrastructure failures.
        return _infra_report(bounded, "modal_lookup", exc)
    return run_gate_wave(method, bounded)


def require_scored_gate_wave(report: GateWaveReport) -> GateWaveReport:
    """Require complete scored evidence before Falsification Gate math runs."""

    if report.status != TrialStatus.SCORED:
        raise GateWaveError(f"gate wave must be SCORED before verdict math: {report.status.value}")
    controls = _validate_controls(
        [
            GateControl(
                gate_id=row.gate_id,
                candidate_trial_id=row.candidate_trial_id,
                control_kind=row.control_kind,
                seed=row.seed,
                payload=row.payload,
            )
            for row in report.controls
        ]
    )
    expected_ids = {control.gate_id for control in controls}
    observed_ids = {row.gate_id for row in report.controls}
    if observed_ids != expected_ids:
        raise GateWaveError("gate wave evidence does not match expected control ids")
    bad_statuses = [f"{row.gate_id}:{row.status.value}" for row in report.controls if row.status != TrialStatus.SCORED]
    if bad_statuses:
        raise GateWaveError(f"gate wave has unscored controls: {', '.join(bad_statuses)}")
    return report


def _control_from_payload(
    *,
    plan: FalsificationPlan,
    kind: GateControlKind,
    seed: int,
    base_payload: dict[str, object],
    extra: dict[str, object],
    timeout_seconds: int,
) -> GateControl:
    payload = dict(base_payload)
    payload.update(extra)
    payload.update(
        {
            "candidate_trial_id": plan.candidate_trial_id,
            "control_kind": kind.value,
            "seed": seed,
            "max_templates": 0,
            "timeout_seconds": timeout_seconds,
        }
    )
    return GateControl(
        gate_id=f"{plan.candidate_trial_id}:{kind.value}:{seed}",
        candidate_trial_id=plan.candidate_trial_id,
        control_kind=kind,
        seed=seed,
        timeout_seconds=timeout_seconds,
        payload=payload,
    )


def _aggregate_timeout_seconds(controls: Sequence[GateControl]) -> int:
    return min(sum(control.timeout_seconds for control in controls), DEFAULT_GATE_AGGREGATE_TIMEOUT_SECONDS)


def _validate_controls(controls: Sequence[GateControl]) -> list[GateControl]:
    if not controls:
        raise GateWaveError("gate wave requires at least one control")
    bounded = [control if isinstance(control, GateControl) else GateControl.model_validate(control) for control in controls]
    if len(bounded) > DEFAULT_MAX_GATE_VARIANTS:
        raise GateWaveError(f"gate wave has {len(bounded)} variants, max is {DEFAULT_MAX_GATE_VARIANTS}")
    trial_ids = {control.candidate_trial_id for control in bounded}
    if len(trial_ids) != 1:
        raise GateWaveError("gate wave controls must share one candidate_trial_id")
    kinds = {control.control_kind for control in bounded}
    required = {GateControlKind.KNOCKOUT, GateControlKind.PLACEBO, GateControlKind.AXIS_CHECK, GateControlKind.SEED_RERUN}
    missing = required - kinds
    if missing:
        missing_names = ", ".join(sorted(kind.value for kind in missing))
        raise GateWaveError(f"gate wave missing required controls: {missing_names}")
    return bounded


def _evidence_from_result(control: GateControl, raw: object) -> GateControlEvidence:
    if isinstance(raw, BaseException):
        return _infra_evidence(control, "modal", raw)
    if not isinstance(raw, dict):
        return _infra_evidence(control, "modal_gate_control_result", TypeError(type(raw).__name__))
    try:
        status = TrialStatus(raw.get("status", TrialStatus.SCORED))
        fold_cartographer = FoldCartographerReport.model_validate(
            raw.get("fold_cartographer", {"signature": "gate_control_returned", "summary": {}, "buckets": {}})
        )
        return GateControlEvidence(
            gate_id=control.gate_id,
            candidate_trial_id=control.candidate_trial_id,
            control_kind=control.control_kind,
            seed=control.seed,
            status=status,
            metrics=dict(raw.get("metrics", {})),
            fold_cartographer=fold_cartographer,
            failure_signature=raw.get("failure_signature") if isinstance(raw.get("failure_signature"), str) else None,
            payload=dict(raw.get("payload", control.payload)),
        )
    except Exception as exc:  # noqa: BLE001 - malformed control returns are infra evidence.
        return _infra_evidence(control, "modal_gate_control_result", exc)


def _infra_report(
    controls: Sequence[GateControl],
    prefix: str,
    exc: BaseException | None = None,
) -> GateWaveReport:
    evidence = [_infra_evidence(control, prefix, exc) for control in controls]
    return GateWaveReport(candidate_trial_id=controls[0].candidate_trial_id, status=TrialStatus.INFRA_FAIL, controls=evidence)


def _infra_evidence(control: GateControl, prefix: str, exc: BaseException | None) -> GateControlEvidence:
    signature = prefix if exc is None else f"{prefix}_{type(exc).__name__}"
    return GateControlEvidence(
        gate_id=control.gate_id,
        candidate_trial_id=control.candidate_trial_id,
        control_kind=control.control_kind,
        seed=control.seed,
        status=TrialStatus.INFRA_FAIL,
        fold_cartographer=FoldCartographerReport(signature=signature),
        failure_signature=signature,
        payload={"error": "" if exc is None else str(exc)},
    )


def _wave_status(evidence: Sequence[GateControlEvidence]) -> TrialStatus:
    if any(row.status == TrialStatus.INFRA_FAIL for row in evidence):
        return TrialStatus.INFRA_FAIL
    if any(row.status == TrialStatus.FAIL for row in evidence):
        return TrialStatus.FAIL
    return TrialStatus.SCORED


def _reject_locked_payload(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_locked_payload(key)
            _reject_locked_payload(nested)
        return
    if isinstance(value, list | tuple):
        for nested in value:
            _reject_locked_payload(nested)
        return
    if isinstance(value, str) and any(token in value.lower() for token in LOCKED_PATH_TOKENS):
        raise ValueError("gate controls must not reference locked labels or hidden validation")
