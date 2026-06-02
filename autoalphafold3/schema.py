"""Typed contracts for local auto-AlphaFold3 trial orchestration."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PRIMARY_METRIC = "best_val_calpha_lddt"
SCORER_VERSION = "calpha_lddt_v1"


class TrialKind(StrEnum):
    """Kinds of trials accepted by the local control plane."""

    TRAINING = "training"
    SAMPLER = "sampler"
    DEBUG = "debug"
    FINAL_VALIDATION = "final_validation"


class BudgetTier(StrEnum):
    """Fixed budget tiers. The dry-run tier is local-only and never calls Modal."""

    DRY_RUN = "dry_run"
    SMOKE = "smoke"
    TRIAL = "trial"
    SAMPLER = "sampler"
    DEBUG = "debug"
    FINAL = "final"


class MoveFamily(StrEnum):
    """Search move families from the handoff."""

    WIDTH_DEPTH = "width_depth"
    PAIRFORMER_ATTENTION = "pairformer_attention"
    DIFFUSION_SCHEDULE = "diffusion_schedule"
    DIFFUSION_SAMPLER_GOLF = "diffusion_sampler_golf"
    RECYCLING = "recycling"
    AUXILIARY_LOSS = "auxiliary_loss"
    GEOMETRY_LOSS = "geometry_loss"
    CURRICULUM = "curriculum"
    OPTIMIZER_SCHEDULER = "optimizer_scheduler"
    FEATURE_HANDLING = "feature_handling"
    MEMORY_RUNTIME = "memory_runtime"


class DiagnosticTarget(StrEnum):
    """Fold Cartographer diagnostic targets from `program.md`."""

    LOCAL_GEOMETRY_WEAK = "local_geometry_weak"
    LONG_RANGE_TOPOLOGY_WEAK = "long_range_topology_weak"
    DISTOGRAM_GOOD_LDDT_FLAT = "distogram_good_lddt_flat"
    STABILITY_COMPUTE = "stability_compute"


class FalsificationAxis(StrEnum):
    """Orthogonal axes that can be pre-registered for gate checks."""

    LOCAL_GEOMETRY = "local_geometry"
    LONG_RANGE_TOPOLOGY = "long_range_topology"
    DISTOGRAM_VS_3D = "distogram_vs_3d"
    STABILITY_COMPUTE = "stability_compute"


class PredictionDirection(StrEnum):
    """Expected direction for a pre-registered diagnostic-axis move."""

    UP = "up"
    DOWN = "down"


class FalsificationVerdict(StrEnum):
    """Final verdicts emitted by the Falsification Gate."""

    CONFIRMED = "CONFIRMED"
    KNOCKOUT_SURVIVES = "KNOCKOUT_SURVIVES"
    PLACEBO_KILL = "PLACEBO_KILL"
    AXIS_MISS = "AXIS_MISS"
    SEED_FRAGILE = "SEED_FRAGILE"


class DiscoveryStatus(StrEnum):
    """Discovery state carried by result rows."""

    UNCONFIRMED = "UNCONFIRMED"
    CONFIRMED = "CONFIRMED"
    KILLED = "KILLED"


class TrialStatus(StrEnum):
    """Lifecycle and outcome statuses for future AutoFold trials."""

    DRAFT = "DRAFT"
    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    SCORED = "SCORED"
    KEEP = "KEEP"
    DISCARD = "DISCARD"
    FAIL = "FAIL"
    INFRA_FAIL = "INFRA_FAIL"
    ARCHIVED = "ARCHIVED"


class RegisteredPrediction(BaseModel):
    """Falsifiable prediction authored before a trial runs."""

    model_config = ConfigDict(extra="forbid")

    causal_component: str = Field(min_length=1)
    predicted_axis: FalsificationAxis
    predicted_direction: PredictionDirection
    expected_lddt_delta_band: tuple[float, float]

    @field_validator("causal_component")
    @classmethod
    def validate_causal_component(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("causal_component must not be blank")
        return value

    @field_validator("expected_lddt_delta_band")
    @classmethod
    def validate_expected_lddt_delta_band(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("expected_lddt_delta_band values must be finite")
        if low < 0.0:
            raise ValueError("expected_lddt_delta_band lower bound must be non-negative")
        if high <= low:
            raise ValueError("expected_lddt_delta_band upper bound must be greater than lower bound")
        return value


class FalsificationPlan(BaseModel):
    """Orchestrator-authored gate plan for a provisional KEEP."""

    model_config = ConfigDict(extra="forbid")

    candidate_trial_id: str = Field(pattern=r"^T[0-9]{3,}$")
    authored_by: Literal["orchestrator"] = "orchestrator"
    knockout_patch: str = Field(min_length=1)
    placebo_family: MoveFamily
    n_seeds: int = Field(default=3, ge=1)
    tau_attribution: float = Field(default=0.5, gt=0.0, le=1.0)
    rho_placebo: float = Field(default=0.5, gt=0.0, le=1.0)
    k_seed: float = Field(default=2.0, gt=0.0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> FalsificationPlan:
        for name in ("tau_attribution", "rho_placebo", "k_seed"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self


class FalsificationResult(BaseModel):
    """Pure local result of applying the Falsification Gate verdict rule."""

    model_config = ConfigDict(extra="forbid")

    gain_full: float
    gain_knockout: float
    gain_placebo: float
    attributable_fraction: float = Field(ge=0.0)
    axis_delta_observed: float
    axis_prediction_held: bool
    seed_mean: float
    seed_std: float = Field(ge=0.0)
    verdict: FalsificationVerdict

    @model_validator(mode="after")
    def validate_finite_numbers(self) -> FalsificationResult:
        for name in (
            "gain_full",
            "gain_knockout",
            "gain_placebo",
            "attributable_fraction",
            "axis_delta_observed",
            "seed_mean",
            "seed_std",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self


class DiscoveryProvenance(BaseModel):
    """Provenance required for one confirmed Discovery Ledger record."""

    model_config = ConfigDict(extra="forbid")

    git_sha: str = Field(min_length=7)
    scorer_version: Literal["calpha_lddt_v1"] = SCORER_VERSION
    primary_metric: Literal["best_val_calpha_lddt"] = PRIMARY_METRIC
    manifest_hashes: dict[str, str] = Field(min_length=1)
    feature_fingerprints: dict[str, str] = Field(min_length=1)
    baseline_id: str = Field(min_length=1)
    current_best_trial_id: str = Field(min_length=1)
    causal_component: str = Field(min_length=1)
    predicted_axis: FalsificationAxis
    predicted_direction: PredictionDirection
    verdict_numbers: dict[str, float] = Field(min_length=1)
    gate_thresholds: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_provenance_numbers(self) -> DiscoveryProvenance:
        for collection_name in ("manifest_hashes", "feature_fingerprints"):
            collection = getattr(self, collection_name)
            for key, value in collection.items():
                if not key or not isinstance(value, str) or not value:
                    raise ValueError(f"{collection_name} must contain non-empty string hashes")
        for collection_name in ("verdict_numbers", "gate_thresholds"):
            collection = getattr(self, collection_name)
            for key, value in collection.items():
                if not key or not math.isfinite(value):
                    raise ValueError(f"{collection_name} values must be finite")
        return self


class DiscoveryRecord(BaseModel):
    """One confirmed mechanism in the orchestrator-written Discovery Ledger."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "autoaf3.discovery.v1"
    trial_id: str = Field(pattern=r"^T[0-9]{3,}$")
    candidate_id: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    axis_moved: FalsificationAxis
    design_rule: str = Field(min_length=1)
    falsification: FalsificationResult
    provenance: DiscoveryProvenance

    @model_validator(mode="after")
    def validate_confirmed_falsification(self) -> DiscoveryRecord:
        if self.falsification.verdict != FalsificationVerdict.CONFIRMED:
            raise ValueError("DiscoveryRecord requires a CONFIRMED falsification verdict")
        if self.axis_moved != self.provenance.predicted_axis:
            raise ValueError("DiscoveryRecord axis_moved must match provenance predicted_axis")
        return self


class FoldCartographerReport(BaseModel):
    """Aggregated diagnostics emitted beside the primary scalar."""

    model_config = ConfigDict(extra="forbid")

    signature: str
    summary: dict[str, object] = Field(default_factory=dict)
    buckets: dict[str, object] = Field(default_factory=dict)


class AutoFoldTrial(BaseModel):
    """Typed trial request crossing the agent/orchestrator trust boundary."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(pattern=r"^T[0-9]{3,}$")
    parent_commit: str = Field(min_length=7)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_session_id: str = Field(min_length=1)
    trial_kind: TrialKind
    hypothesis: str = Field(min_length=1)
    move_family: MoveFamily
    diagnostic_target: DiagnosticTarget
    prediction: RegisteredPrediction
    patch_path: str | None = None
    config_path: str
    config_payload: dict[str, object] | None = None
    budget: BudgetTier
    seed: int = Field(ge=0)
    n_res: int | None = Field(default=None, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    sampler_steps: int | None = Field(default=None, ge=1)
    sampler_noise_scale: float | None = Field(default=None, gt=0.0, le=2.0)
    sampler_step_scale: float | None = Field(default=None, gt=0.0, le=2.0)
    sampler_schedule_shape: Literal["linear", "cosine", "late_refine"] | None = None
    sampler_num_samples: int | None = Field(default=None, ge=1, le=4)
    sampler_selection_policy: Literal["first", "geometry", "compact_geometry"] | None = None
    sampler_coordinate_normalization: Literal["none", "ca_bond"] | None = None
    sampler_coordinate_scale: float | None = Field(default=None, gt=0.0, le=20.0)
    max_wall_minutes: int = Field(ge=1)
    manifest_hashes: dict[str, str] = Field(default_factory=dict)
    scorer_version: str = SCORER_VERSION
    primary_metric: Literal["best_val_calpha_lddt"] = PRIMARY_METRIC
    param_cap: int = Field(ge=1)
    gpu_memory_cap: float = Field(ge=0.0)
    cost_cap: float = Field(ge=0.0)
    timeout_cap: int = Field(ge=1)
    artifact_dir: str | None = None
    checkpoint_path: str | None = None

    @model_validator(mode="after")
    def validate_trial_shape(self) -> AutoFoldTrial:
        if self.scorer_version != SCORER_VERSION:
            raise ValueError(f"scorer_version must be {SCORER_VERSION}")
        target_axis = {
            DiagnosticTarget.LOCAL_GEOMETRY_WEAK: FalsificationAxis.LOCAL_GEOMETRY,
            DiagnosticTarget.LONG_RANGE_TOPOLOGY_WEAK: FalsificationAxis.LONG_RANGE_TOPOLOGY,
            DiagnosticTarget.DISTOGRAM_GOOD_LDDT_FLAT: FalsificationAxis.DISTOGRAM_VS_3D,
            DiagnosticTarget.STABILITY_COMPUTE: FalsificationAxis.STABILITY_COMPUTE,
        }[self.diagnostic_target]
        if self.prediction.predicted_axis != target_axis:
            raise ValueError(
                "prediction predicted_axis must match diagnostic_target "
                f"{self.diagnostic_target.value}: expected {target_axis.value}, "
                f"got {self.prediction.predicted_axis.value}"
            )
        if self.trial_kind == TrialKind.SAMPLER:
            if self.checkpoint_path is None:
                raise ValueError("sampler trials require checkpoint_path")
            if self.max_steps is not None:
                raise ValueError("sampler trials must not set training max_steps")
            if self.sampler_steps is None:
                raise ValueError("sampler trials require sampler_steps")
        elif self.trial_kind in {TrialKind.TRAINING, TrialKind.DEBUG, TrialKind.FINAL_VALIDATION}:
            if self.max_steps is None:
                raise ValueError(f"{self.trial_kind.value} trials require max_steps")
            if self.sampler_steps is not None:
                raise ValueError(f"{self.trial_kind.value} trials must not set sampler_steps")
            sampler_fields = {
                "sampler_noise_scale": self.sampler_noise_scale,
                "sampler_step_scale": self.sampler_step_scale,
                "sampler_schedule_shape": self.sampler_schedule_shape,
            }
            present = [name for name, value in sampler_fields.items() if value is not None]
            if present:
                raise ValueError(f"{self.trial_kind.value} trials must not set sampler-only fields: {present}")
        if (
            self.trial_kind not in {TrialKind.SAMPLER, TrialKind.TRAINING}
            and (self.sampler_num_samples is not None or self.sampler_selection_policy is not None)
        ):
            raise ValueError(f"{self.trial_kind.value} trials must not set post-training sampler selection fields")
        if self.sampler_coordinate_normalization is not None and self.trial_kind not in {
            TrialKind.SAMPLER,
            TrialKind.TRAINING,
        }:
            raise ValueError(
                f"{self.trial_kind.value} trials must not set sampler_coordinate_normalization"
            )
        if self.sampler_coordinate_scale is not None:
            if self.trial_kind not in {TrialKind.SAMPLER, TrialKind.TRAINING}:
                raise ValueError(f"{self.trial_kind.value} trials must not set sampler_coordinate_scale")
            if self.sampler_coordinate_normalization != "ca_bond":
                raise ValueError("sampler_coordinate_scale requires sampler_coordinate_normalization=ca_bond")
        if self.budget == BudgetTier.DRY_RUN and self.gpu_memory_cap != 0.0:
            raise ValueError("dry_run budget must use gpu_memory_cap=0.0")
        return self


class AutoFoldResult(BaseModel):
    """Canonical local result shape for dry-run and future Modal trials."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "autoaf3.result.v1"
    trial_id: str
    status: TrialStatus
    candidate_id: str
    primary_metric: Literal["best_val_calpha_lddt"] = PRIMARY_METRIC
    metrics: dict[str, object] = Field(default_factory=dict)
    fold_cartographer: FoldCartographerReport
    artifacts: dict[str, str] = Field(default_factory=dict)
    failure_signature: str | None = None
    discovery: DiscoveryStatus = DiscoveryStatus.UNCONFIRMED
    falsification: FalsificationResult | None = None
    postmortem: str = ""

    @model_validator(mode="after")
    def validate_discovery_status(self) -> AutoFoldResult:
        if self.falsification is None:
            if self.discovery != DiscoveryStatus.UNCONFIRMED:
                raise ValueError("confirmed or killed discovery statuses require falsification evidence")
            return self
        if self.falsification.verdict == FalsificationVerdict.CONFIRMED:
            if self.discovery != DiscoveryStatus.CONFIRMED:
                raise ValueError("CONFIRMED falsification verdict requires CONFIRMED discovery status")
            if self.status != TrialStatus.KEEP:
                raise ValueError("confirmed discoveries require KEEP status")
        elif self.discovery != DiscoveryStatus.KILLED:
            raise ValueError("non-CONFIRMED falsification verdicts require KILLED discovery status")
        return self
