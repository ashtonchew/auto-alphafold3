"""Manual and deterministic autoresearch planning loop."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoalphafold3.config_contract import validate_config_payload as validate_nanofold_config_payload
from autoalphafold3.llm_policy import DEFAULT_LLM_MODEL, AgentSearchPhase, default_llm_phase_policy
from autoalphafold3.patch_policy import PatchPolicyError, validate_patch_scope
from autoalphafold3.autoresearch_candidates import (
    create_candidate_envelope,
    create_run_manifest,
    validate_run_id,
    write_candidate_decision,
    write_candidate_evidence,
)
from autoalphafold3.autoresearch_comparisons import (
    AutoresearchComparisonError,
    compare_and_write_candidate_decision,
)
from autoalphafold3.modal_app import APP_NAME, DATA_MOUNT, TRUSTED_ORCHESTRATOR_CLASS
from autoalphafold3.schema import (
    AutoFoldResult,
    AutoFoldTrial,
    BudgetTier,
    DiagnosticTarget,
    FalsificationAxis,
    FoldCartographerReport,
    MoveFamily,
    PredictionDirection,
    RegisteredPrediction,
    TrialKind,
    TrialStatus,
)
from autoalphafold3.short_training import short_training_payload
from autoalphafold3.short_training_runner import DEFAULT_MODAL_FEATURES_PATH

APPROVAL_TEXT = "I_APPROVE_AUTORESEARCH_LIVE_SEARCH"
MODAL_WORKER_RESULT_TIMEOUT_S = 900
FORBIDDEN_TRUE_PLAN_FLAGS = {
    "official_benchmark_result",
    "writes_baseline",
    "writes_ledger",
    "writes_discovery_ledger",
    "starts_search",
    "live_modal_execution",
}
ALLOWED_CONFIG_EXACT = {"configs/nanofold_dev_cpu_smoke.json"}
ALLOWED_CONFIG_PREFIXES = ("configs/experiments/",)
LOCKED_READ_TOKENS = ("/locked", "locked/labels", "public_val_labels", "autoalphafold3-locked")
PATCH_FORBIDDEN_KEYS = FORBIDDEN_TRUE_PLAN_FLAGS | {"max_templates"}
_AUTORESEARCH_PLANNER_SYSTEM_PROMPT = """You are the NanoFold-style AlphaFold3-lite autoresearch planner.
Return exactly one JSON plan matching the provided schema.
Plan only bounded smoke-budget candidate changes on the approved experiment config surface.
Do not propose scorer, label, manifest, fingerprint, baseline, Modal, GPU, Volume, template database, or ledger changes.
Official NanoFold-style runs must keep max_templates=0.
Candidate plans must be falsifiable, pre-registered, artifact-only, and safe to discard before any ledger write."""


class AutoresearchLoopError(RuntimeError):
    """Raised when autoresearch planning cannot proceed safely."""


class AutoresearchPlanner(Protocol):
    """Injected planner seam for tests and future harness-owned LLM calls."""

    def plan(
        self,
        *,
        run_id: str,
        trial_id: str,
        candidate_index: int,
        model: str,
        policy: dict[str, dict[str, object]],
        base_commit: str,
        prior_plans: list[dict[str, object]],
        prior_outcomes: list[dict[str, object]],
        candidate_budget: str,
    ) -> "AutoresearchCandidatePlan":
        """Return one structured autoresearch candidate."""


class EmptyManifestHashes(BaseModel):
    """Strict empty object for early smoke candidates with no manifest hash claims."""

    model_config = ConfigDict(extra="forbid")


class PlannerConfigPayload(BaseModel):
    """Strict NanoFold config payload schema accepted from the LLM planner."""

    model_config = ConfigDict(extra="forbid")

    description: str
    device: str
    use_amp: bool
    detect_anomaly: bool
    compile_model: bool
    use_grad_checkpoint: bool
    train_split: float
    residue_crop_size: int
    num_recycle: int
    single_embedding_size: int
    pair_embedding_size: int
    input_atom_embedding_size: int
    input_atom_pair_embedding_size: int
    input_token_embedding_size: int
    position_bins: int
    num_atom_transformer_blocks: int
    num_atom_transformer_heads: int
    num_atom_transformer_queries: int
    num_atom_transformer_keys: int
    product_embedding_size: int
    num_msa: int
    num_msa_samples: int
    num_msa_blocks: int
    msa_embedding_size: int
    msa_averaging_embedding_size: int
    num_msa_heads: int
    msa_transition_multiplier: int
    num_triangular_update_channels: int
    num_triangular_attention_channels: int
    num_triangular_attention_heads: int
    num_template_blocks: int
    max_templates: int
    template_embedding_size: int
    num_pairformer_blocks: int
    num_pair_heads: int
    pairformer_transition_multiplier: int
    diffusion_steps: int
    diffusion_batch_size: int
    atom_embedding_size: int
    atom_pair_embedding_size: int
    token_embedding_size: int
    num_diffusion_transformer_blocks: int
    num_diffusion_transformer_heads: int
    fourier_embedding_size: int
    num_distogram_bins: int
    clip_norm: float
    learning_rate: float
    beta1: float
    beta2: float
    optimizer_eps: float
    lr_start_factor: float
    lr_warmup: int
    diffusion_loss_weight: float
    dist_loss_weight: float
    distogram_loss_weight: float
    local_calpha_geometry_loss_weight: float


class PlannerPrediction(BaseModel):
    """Strict model-facing prediction schema using repo-side semantic checks."""

    model_config = ConfigDict(extra="forbid")

    causal_component: str
    predicted_axis: str
    predicted_direction: str
    expected_lddt_delta_band: list[float]


class PlannerConfigSummary(BaseModel):
    """Strict candidate config summary emitted beside the executable trial."""

    model_config = ConfigDict(extra="forbid")

    config_path: str
    max_templates: int
    learning_rate: float
    local_calpha_geometry_loss_weight: float


class PlannerTrial(BaseModel):
    """Strict LLM-authored training trial shape for one smoke candidate."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    parent_commit: str
    agent_session_id: str
    trial_kind: str
    hypothesis: str
    move_family: str
    diagnostic_target: str
    prediction: PlannerPrediction
    patch_path: None
    config_path: str
    config_payload: PlannerConfigPayload
    budget: str
    seed: int
    n_res: int
    max_steps: int
    max_wall_minutes: int
    manifest_hashes: EmptyManifestHashes
    scorer_version: str
    primary_metric: str
    param_cap: int
    gpu_memory_cap: float
    cost_cap: float
    timeout_cap: int
    artifact_dir: str
    checkpoint_path: None


class TrustedAutoresearchClient(Protocol):
    """Injected trusted-orchestrator client seam for live Modal autoresearch."""

    def submit_and_poll_trial(self, trial: dict[str, object]) -> dict[str, object]:
        """Submit one trial through the trusted orchestrator and poll the spawned worker."""

    def score_trial(self, trial_id: str) -> dict[str, object]:
        """Score one trial through the scorer-only Modal worker."""


class AutoresearchCandidatePlan(BaseModel):
    """Strict LLM/recorded plan for exactly one autoresearch candidate."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    trial: PlannerTrial
    changed_paths: list[str]
    config: PlannerConfigSummary
    patch_text: str
    rationale: str

    @model_validator(mode="before")
    @classmethod
    def reject_multi_candidate_shape(cls, value: object) -> object:
        if isinstance(value, dict) and "candidates" in value:
            raise ValueError("LLM autoresearch planner must return exactly one candidate")
        return value

    @model_validator(mode="after")
    def validate_one_move_contract(self) -> "AutoresearchCandidatePlan":
        if not self.trial.trial_id:
            raise ValueError("LLM candidate trial must include trial_id")
        if not self.trial.move_family:
            raise ValueError("LLM candidate trial must include one move_family")
        if not self.trial.diagnostic_target:
            raise ValueError("LLM candidate trial must include one diagnostic_target")
        if self.config.max_templates != 0:
            raise ValueError("LLM candidate config must preserve max_templates=0")
        return self


class OpenAIAutoresearchPlanner:
    """Structured-output OpenAI planner for one autoresearch candidate."""

    def __init__(self, *, repo_root: str | Path = ".", model: str = DEFAULT_LLM_MODEL) -> None:
        self.repo_root = Path(repo_root)
        self.model = model
        self.policy = default_llm_phase_policy(AgentSearchPhase.PATCH_PLANNING, model=model)

    def plan(
        self,
        *,
        run_id: str,
        trial_id: str,
        candidate_index: int,
        model: str,
        policy: dict[str, dict[str, object]],
        base_commit: str,
        prior_plans: list[dict[str, object]] | None = None,
        prior_outcomes: list[dict[str, object]] | None = None,
        candidate_budget: str = BudgetTier.SMOKE.value,
    ) -> AutoresearchCandidatePlan:
        del model
        prompt = _autoresearch_planner_prompt(
            root=self.repo_root,
            run_id=run_id,
            trial_id=trial_id,
            candidate_index=candidate_index,
            base_commit=base_commit,
            policy=policy,
            prior_plans=prior_plans or [],
            prior_outcomes=prior_outcomes or [],
            candidate_budget=candidate_budget,
        )
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise AutoresearchLoopError("LLM planner requires the openai package") from exc

        try:
            client = OpenAI()
        except Exception as exc:  # noqa: BLE001 - missing local key should use Modal harness secret.
            if _is_missing_openai_credentials(exc):
                return _plan_autoresearch_with_modal_harness_secret(
                    prompt=prompt,
                    trial_id=trial_id,
                    candidate_index=candidate_index,
                    base_commit=base_commit,
                    policy=policy,
                    prior_plans=prior_plans or [],
                    prior_outcomes=prior_outcomes or [],
                    candidate_budget=candidate_budget,
                    model=self.model,
                )
            raise

        kwargs = self.policy.to_responses_create_kwargs()
        try:
            response = client.responses.parse(
                **kwargs,
                input=[
                    {"role": "system", "content": _AUTORESEARCH_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text_format=AutoresearchCandidatePlan,
            )
        except TypeError:
            response = client.responses.parse(
                **kwargs,
                input=[
                    {"role": "system", "content": _AUTORESEARCH_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text={"format": AutoresearchCandidatePlan},
            )
        except Exception as exc:  # noqa: BLE001 - allow harness-secret fallback only for missing local credentials.
            if _is_missing_openai_credentials(exc):
                return _plan_autoresearch_with_modal_harness_secret(
                    prompt=prompt,
                    trial_id=trial_id,
                    candidate_index=candidate_index,
                    base_commit=base_commit,
                    policy=policy,
                    prior_plans=prior_plans or [],
                    prior_outcomes=prior_outcomes or [],
                    candidate_budget=candidate_budget,
                    model=self.model,
                )
            raise
        return _extract_autoresearch_parsed_plan(response)


@dataclass(frozen=True)
class AutoresearchLoopResult:
    """JSON-friendly autoresearch planning result."""

    status: str
    mode: str
    planner: str
    run_id: str
    run_dir: str
    generated_trials: list[str]
    candidate_dirs: list[str]
    decisions: list[dict[str, object]]
    wrote_files: list[str]
    llm_policy: dict[str, dict[str, object]] | None
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    stopped_reason: str
    pending_live_action: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_autoresearch_loop(
    *,
    repo_root: str | Path = ".",
    run_id: str,
    mode: str = "dry-run",
    planner: str = "deterministic",
    start_trial_id: str = "T120",
    max_candidates: int = 6,
    candidate_plan: str | Path | None = None,
    approval: str | None = None,
    model: str = DEFAULT_LLM_MODEL,
    planner_client: AutoresearchPlanner | None = None,
    modal_env: str | None = None,
    modal_client: TrustedAutoresearchClient | None = None,
    failure_streak_limit: int = 2,
    prior_run_ids: list[str] | None = None,
    candidate_budget: str = BudgetTier.SMOKE.value,
    diagnostic_report: str | Path | None = None,
) -> AutoresearchLoopResult:
    """Plan autoresearch candidates and optionally run one approved Modal candidate."""

    if failure_streak_limit < 1:
        raise AutoresearchLoopError("failure_streak_limit must be at least 1")
    if candidate_budget not in {BudgetTier.SMOKE.value, BudgetTier.TRIAL.value}:
        raise AutoresearchLoopError("candidate_budget must be smoke or trial")
    if mode not in {"dry-run", "modal"}:
        raise AutoresearchLoopError(f"unsupported autoresearch mode: {mode}")
    if planner not in {"manual", "deterministic", "targeted_diagnostic", "llm"}:
        raise AutoresearchLoopError(f"unsupported autoresearch planner for this PR: {planner}")
    if mode == "modal":
        if approval != APPROVAL_TEXT:
            raise AutoresearchLoopError(f"live autoresearch requires --approve {APPROVAL_TEXT}")

    root = Path(repo_root)
    base_commit = _git_head(root)
    llm_policy = _llm_policy_specs(model) if planner == "llm" else None
    prior_outcomes = _prior_autoresearch_outcomes(root=root, prior_run_ids=prior_run_ids or [])
    planned = _planned_candidates(
        root=root,
        planner=planner,
        run_id=run_id,
        start_trial_id=start_trial_id,
        max_candidates=max_candidates,
        candidate_plan=candidate_plan,
        base_commit=base_commit,
        model=model,
        llm_policy=llm_policy,
        planner_client=planner_client,
        prior_outcomes=prior_outcomes,
        candidate_budget=candidate_budget,
        diagnostic_report=diagnostic_report,
    )
    for candidate in planned:
        trial = AutoFoldTrial.model_validate(candidate["trial"])
        _validate_trial_artifacts(trial.model_dump(mode="json"))
    _validate_unique_trial_ids(planned)
    run_manifest = create_run_manifest(
        repo_root=root,
        run_id=run_id,
        base_commit=base_commit,
        planner=planner,
        mode=mode,
        description="Autoresearch dry-run planning artifacts.",
    )
    generated_trials: list[str] = []
    candidate_dirs: list[str] = []
    decisions: list[dict[str, object]] = []
    wrote_files: list[str] = [
        str(root / "runs" / "autoresearch" / run_id / "run_manifest.json"),
        str(root / "runs" / "autoresearch" / run_id / "summary.json"),
        str(root / "runs" / "autoresearch" / run_id / "results.tsv"),
    ]
    matched_budget_results: dict[str, AutoFoldResult] = {}
    failure_streak = 0
    stopped_reason = "max_candidates_reached"
    for candidate in planned:
        trial = candidate["trial"]
        envelope = create_candidate_envelope(
            repo_root=root,
            run_id=run_id,
            trial_id=str(trial["trial_id"]),
            hypothesis=str(candidate["hypothesis"]),
            trial=trial,
            config=candidate.get("config"),
            patch_text=str(candidate.get("patch_text", "")),
        )
        generated_trials.append(str(trial["trial_id"]))
        candidate_dirs.append(str(envelope.candidate_dir))
        decisions.append(
            {
                "run_id": run_id,
                "trial_id": str(trial["trial_id"]),
                "candidate_id": envelope.candidate_id,
                "status": TrialStatus.DRAFT.value,
                "planning_status": "PLANNED",
                "writes_ledger": False,
                "writes_discovery_ledger": False,
                "official_benchmark_result": False,
            }
        )
        wrote_files.extend(
            [
                str(envelope.manifest_path),
                str(envelope.hypothesis_path),
                str(envelope.patch_path),
                str(envelope.trial_path),
            ]
        )
        if candidate.get("config") is not None:
            wrote_files.append(str(envelope.config_path))
        write_candidate_evidence(envelope, preflight=_planned_preflight(trial))
        wrote_files.append(str(envelope.preflight_path))
        if mode == "modal":
            live = _run_modal_candidate_smoke(
                root=root,
                run_id=run_id,
                envelope=envelope,
                trial=trial,
                modal_env=modal_env,
                modal_client=modal_client,
                matched_budget_result=matched_budget_results.get(str(trial["budget"])),
            )
            decisions[-1].update(live["decision"])
            wrote_files.extend(live["wrote_files"])
            result = live.get("result")
            if isinstance(result, AutoFoldResult) and result.status == TrialStatus.SCORED:
                matched_budget_results.setdefault(str(trial["budget"]), result)
                failure_streak = 0
            elif str(decisions[-1].get("status")) in {TrialStatus.FAIL.value, TrialStatus.INFRA_FAIL.value}:
                failure_streak += 1
                if failure_streak >= failure_streak_limit:
                    stopped_reason = f"failure_streak_limit:{decisions[-1].get('status')}"
                    break
            else:
                failure_streak = 0
    _write_planned_candidate_index(root=root, run_id=run_id, records=decisions)
    return AutoresearchLoopResult(
        status="PASS" if mode == "modal" else "PLANNED",
        mode=mode,
        planner=planner,
        run_id=str(run_manifest["run_id"]),
        run_dir=str(root / "runs" / "autoresearch" / run_id),
        generated_trials=generated_trials,
        candidate_dirs=candidate_dirs,
        decisions=decisions,
        wrote_files=wrote_files,
        llm_policy=llm_policy,
        starts_search=mode == "modal",
        writes_ledger=False,
        writes_discovery_ledger=False,
        stopped_reason=stopped_reason,
        pending_live_action=None if mode == "modal" else f"modal mode requires --approve {APPROVAL_TEXT}",
    )


def _run_modal_candidate_smoke(
    *,
    root: Path,
    run_id: str,
    envelope,
    trial: dict[str, object],
    modal_env: str | None,
    modal_client: TrustedAutoresearchClient | None,
    matched_budget_result: AutoFoldResult | None,
) -> dict[str, object]:
    checked = AutoFoldTrial.model_validate(trial)
    if checked.trial_kind not in {TrialKind.TRAINING, TrialKind.SAMPLER}:
        raise AutoresearchLoopError("live autoresearch currently supports training and sampler candidates only")
    client = modal_client if modal_client is not None else DeployedTrustedAutoresearchClient(environment_name=modal_env)
    try:
        payload = client.submit_and_poll_trial(_modal_trial_payload(checked))
    except Exception as exc:  # noqa: BLE001 - normalize delegated runner failures.
        raise AutoresearchLoopError(f"live autoresearch trusted-orchestrator trial failed: {exc}") from exc
    wrote_files: list[str] = []
    decision_overrides: dict[str, object] = {}
    if _is_short_training_manifest(payload) or _is_sampler_manifest(payload):
        if _is_short_training_manifest(payload):
            wrote_files.extend(_record_short_training_manifest(envelope=envelope, payload=payload))
            manifest_path = envelope.training_manifest_path
        else:
            wrote_files.extend(_record_sampler_manifest(envelope=envelope, payload=payload))
            manifest_path = envelope.sampler_manifest_path
        decision_overrides.update(
            {
                "execution_manifest_path": str(manifest_path),
                "worker_status": payload.get("status"),
            }
        )
        if _is_short_training_manifest(payload):
            decision_overrides.update(
                {
                    "training_manifest_path": str(envelope.training_manifest_path),
                    "training_status": payload.get("status"),
                }
            )
        else:
            decision_overrides.update(
                {
                    "sampler_manifest_path": str(envelope.sampler_manifest_path),
                    "sampler_status": payload.get("status"),
                }
            )
        try:
            payload = client.score_trial(checked.trial_id)
        except Exception as exc:  # noqa: BLE001 - normalize scorer failures.
            raise AutoresearchLoopError(f"live autoresearch scorer failed: {exc}") from exc
    scored = _record_modal_candidate_payload(
        root=root,
        run_id=run_id,
        envelope=envelope,
        payload=payload,
        decision_overrides=decision_overrides,
        matched_budget_result=matched_budget_result,
    )
    scored["wrote_files"] = [*wrote_files, *scored["wrote_files"]]
    return scored


def _prior_autoresearch_outcomes(*, root: Path, prior_run_ids: list[str]) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    for raw_run_id in prior_run_ids:
        run_id = validate_run_id(str(raw_run_id))
        run_dir = root / "runs" / "autoresearch" / run_id
        summary_path = run_dir / "summary.json"
        if summary_path.is_symlink():
            raise AutoresearchLoopError(f"prior autoresearch summary must not be a symlink: {summary_path}")
        if not summary_path.exists():
            raise AutoresearchLoopError(f"prior autoresearch run summary does not exist: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        candidates = summary.get("candidates")
        if not isinstance(candidates, list):
            raise AutoresearchLoopError(f"prior autoresearch summary has no candidates list: {summary_path}")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            trial_id = str(candidate.get("trial_id") or "")
            if not trial_id:
                continue
            candidate_dir = run_dir / "candidates" / trial_id
            trial_payload = _read_small_json(candidate_dir / "trial.json")
            metrics_payload = _read_small_json(candidate_dir / "metrics.json")
            comparison = metrics_payload.get("comparison") if isinstance(metrics_payload.get("comparison"), dict) else {}
            fold_cartographer = (
                metrics_payload.get("fold_cartographer") if isinstance(metrics_payload.get("fold_cartographer"), dict) else {}
            )
            outcomes.append(
                {
                    "run_id": run_id,
                    "trial_id": trial_id,
                    "status": candidate.get("status"),
                    "promotion_status": candidate.get("promotion_status"),
                    "provisional_keep": bool(candidate.get("provisional_keep", False)),
                    "matched_budget_delta": candidate.get("matched_budget_delta"),
                    "global_baseline_delta": candidate.get("global_baseline_delta"),
                    "candidate_score": comparison.get("candidate_score"),
                    "fold_cartographer_signature": fold_cartographer.get("signature"),
                    "candidate_artifacts": metrics_payload.get("candidate_artifacts")
                    if isinstance(metrics_payload.get("candidate_artifacts"), dict)
                    else {},
                    "hypothesis": _read_prior_hypothesis(candidate_dir / "hypothesis.md"),
                    "move_family": trial_payload.get("move_family"),
                    "diagnostic_target": trial_payload.get("diagnostic_target"),
                    "config_path": trial_payload.get("config_path"),
                    "budget": trial_payload.get("budget"),
                }
            )
    return outcomes


def _read_prior_hypothesis(path: Path) -> str | None:
    if path.is_symlink() or not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text[:500] if text else None


class DeployedTrustedAutoresearchClient:
    """Modal SDK client for one trusted-orchestrator trial submission."""

    def __init__(self, *, environment_name: str | None = None) -> None:
        self.environment_name = environment_name
        try:
            import modal
        except ModuleNotFoundError as exc:
            raise AutoresearchLoopError("Modal SDK is required for live Modal autoresearch") from exc
        self._modal = modal

    def submit_and_poll_trial(self, trial: dict[str, object]) -> dict[str, object]:
        orchestrator_cls = self._modal.Cls.from_name(
            APP_NAME,
            TRUSTED_ORCHESTRATOR_CLASS,
            environment_name=self.environment_name,
        )
        orchestrator = orchestrator_cls()
        submitted = orchestrator.submit_trial.remote(trial)
        if not isinstance(submitted, dict):
            raise AutoresearchLoopError("TrustedOrchestrator.submit_trial returned a non-object payload")
        worker_call_id = _worker_call_id(submitted)
        call = self._modal.FunctionCall.from_id(worker_call_id)
        payload = call.get(timeout=MODAL_WORKER_RESULT_TIMEOUT_S)
        if not isinstance(payload, dict):
            raise AutoresearchLoopError("trusted-orchestrator worker call returned a non-object payload")
        return payload

    def score_trial(self, trial_id: str) -> dict[str, object]:
        scorer_cls = self._modal.Cls.from_name(
            APP_NAME,
            "Scorer",
            environment_name=self.environment_name,
        )
        scorer = scorer_cls()
        payload = scorer.score.remote(trial_id)
        if not isinstance(payload, dict):
            raise AutoresearchLoopError("Scorer.score returned a non-object payload")
        return payload


def _record_modal_candidate_payload(
    *,
    root: Path,
    run_id: str,
    envelope,
    payload: dict[str, object],
    decision_overrides: dict[str, object] | None = None,
    matched_budget_result: AutoFoldResult | None = None,
) -> dict[str, object]:
    del run_id
    trial_id = envelope.trial_id
    status = str(payload.get("status") or "UNKNOWN")
    wrote_files: list[str] = []
    decision: dict[str, object] = {
        "execution_status": status,
        "trial_artifact_dir": str(root / "runs" / "trials" / trial_id),
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "official_benchmark_result": False,
    }
    decision.update(decision_overrides or {})
    if _is_short_training_manifest(payload):
        wrote_files.extend(_record_short_training_manifest(envelope=envelope, payload=payload))
        decision["training_manifest_path"] = str(envelope.training_manifest_path)
        decision["benchmark_decision"] = "NOT_SCORED"
        return {"decision": decision, "wrote_files": wrote_files}
    try:
        result = _score_payload_to_result(payload)
    except ValueError as exc:
        raise AutoresearchLoopError(f"live autoresearch returned invalid trial payload: {exc}") from exc
    if result.trial_id != trial_id:
        raise AutoresearchLoopError("live autoresearch result trial_id mismatch")
    if result.status == TrialStatus.SCORED:
        try:
            comparison = compare_and_write_candidate_decision(
                envelope,
                candidate_result=result,
                matched_budget_result=matched_budget_result,
                repo_root=root,
                baseline_dir="runs/baseline",
                ledger_path="runs/ledger.jsonl",
            )
        except AutoresearchComparisonError as exc:
            raise AutoresearchLoopError(f"live autoresearch comparison failed: {exc}") from exc
        wrote_files.extend([str(envelope.metrics_path), str(envelope.decision_path), str(envelope.postmortem_path)])
        if envelope.promotion_plan_path.exists():
            wrote_files.append(str(envelope.promotion_plan_path))
        decision.update(comparison.to_dict())
        decision["promotion_status"] = "FALSIFICATION_REQUIRED" if comparison.provisional_keep else "NOT_ELIGIBLE"
        decision["promotion_plan_path"] = str(envelope.promotion_plan_path) if comparison.provisional_keep else None
        decision["decision_path"] = str(envelope.decision_path)
        return {"decision": decision, "wrote_files": wrote_files, "result": result}
    if result.status in {TrialStatus.FAIL, TrialStatus.INFRA_FAIL}:
        wrote_files.extend(write_candidate_evidence(envelope, error_report=payload))
        write_candidate_decision(
            envelope,
            status=result.status.value,
            matched_budget_delta=None,
            global_baseline_delta=None,
            reason=result.failure_signature or f"modal trial returned {result.status.value}",
            postmortem=result.postmortem or f"Modal trial returned {result.status.value}.",
        )
        wrote_files.extend([str(envelope.decision_path), str(envelope.postmortem_path)])
        decision["status"] = result.status.value
        decision["promotion_status"] = "NOT_ELIGIBLE"
        decision["promotion_plan_path"] = None
        decision["decision_path"] = str(envelope.decision_path)
        return {"decision": decision, "wrote_files": wrote_files}
    raise AutoresearchLoopError(f"live autoresearch result status is not terminal: {result.status.value}")


def _worker_call_id(payload: dict[str, object]) -> str:
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict) and isinstance(artifacts.get("worker_call_id"), str):
        return artifacts["worker_call_id"]
    fold_cartographer = payload.get("fold_cartographer")
    if isinstance(fold_cartographer, dict):
        summary = fold_cartographer.get("summary")
        if isinstance(summary, dict) and isinstance(summary.get("worker_call_id"), str):
            return summary["worker_call_id"]
    raise AutoresearchLoopError("TrustedOrchestrator.submit_trial did not return worker_call_id")


def _is_short_training_manifest(payload: dict[str, object]) -> bool:
    return payload.get("schema_version") == "autoaf3.short_training_manifest.v1"


def _is_sampler_manifest(payload: dict[str, object]) -> bool:
    return payload.get("schema_version") == "autoaf3.sampler_manifest.v1"


def _record_short_training_manifest(*, envelope, payload: dict[str, object]) -> list[str]:
    if payload.get("trial_id") != envelope.trial_id:
        raise AutoresearchLoopError("live autoresearch short-training manifest trial_id mismatch")
    return write_candidate_evidence(envelope, training_manifest=payload)


def _record_sampler_manifest(*, envelope, payload: dict[str, object]) -> list[str]:
    if payload.get("trial_id") != envelope.trial_id:
        raise AutoresearchLoopError("live autoresearch sampler manifest trial_id mismatch")
    return write_candidate_evidence(envelope, sampler_manifest=payload)


def _score_payload_to_result(payload: dict[str, object]) -> AutoFoldResult:
    if payload.get("schema_version") == "autoaf3.result.v1":
        return AutoFoldResult.model_validate(payload)
    status = TrialStatus.SCORED if payload.get("status") == TrialStatus.SCORED.value else TrialStatus.FAIL
    error_report = payload.get("error_report") if isinstance(payload.get("error_report"), dict) else {}
    return AutoFoldResult(
        trial_id=str(payload.get("trial_id", "UNKNOWN")),
        status=status,
        candidate_id=str(payload.get("candidate_id", "autoresearch_score")),
        metrics=dict(payload.get("metrics") or {}),
        fold_cartographer=FoldCartographerReport.model_validate(payload.get("fold_cartographer") or {"signature": "missing"}),
        artifacts={key: str(value) for key, value in dict(payload.get("artifacts") or {}).items()},
        failure_signature=str(error_report.get("failure_signature") or payload.get("failure_signature") or "")
        if status != TrialStatus.SCORED
        else None,
        postmortem=str(error_report.get("reason") or payload.get("postmortem") or ""),
    )


def _modal_trial_payload(trial: AutoFoldTrial) -> dict[str, object]:
    if trial.trial_kind == TrialKind.SAMPLER:
        payload = trial.model_dump(mode="json")
        checkpoint_path = str(payload.get("checkpoint_path") or "")
        if checkpoint_path.startswith("runs/trials/"):
            payload["checkpoint_path"] = f"{DATA_MOUNT}/{checkpoint_path}"
        return payload
    return _modal_short_training_payload(trial)


def _modal_short_training_payload(trial: AutoFoldTrial) -> dict[str, object]:
    if trial.max_steps is None:
        raise AutoresearchLoopError("live autoresearch smoke training candidates require max_steps")
    return short_training_payload(
        trial_id=trial.trial_id,
        candidate_id=trial.trial_id,
        config_path=trial.config_path,
        features_path=DEFAULT_MODAL_FEATURES_PATH,
        max_steps=trial.max_steps,
        budget=trial.budget.value,
        seed=trial.seed,
        artifact_dir=trial.artifact_dir,
        local_only=False,
        predict_after_training=True,
        config_payload=trial.config_payload,
    )


def _planned_candidates(
    *,
    root: Path,
    planner: str,
    run_id: str,
    start_trial_id: str,
    max_candidates: int,
    candidate_plan: str | Path | None,
    base_commit: str,
    model: str,
    llm_policy: dict[str, dict[str, object]] | None,
    planner_client: AutoresearchPlanner | None,
    prior_outcomes: list[dict[str, object]],
    candidate_budget: str,
    diagnostic_report: str | Path | None,
) -> list[dict[str, object]]:
    if planner == "manual":
        return _manual_candidates(root=root, candidate_plan=candidate_plan)
    if planner == "deterministic":
        return _deterministic_candidates(start_trial_id=start_trial_id, max_candidates=max_candidates, base_commit=base_commit)
    if planner == "targeted_diagnostic":
        return _targeted_diagnostic_candidates(
            root=root,
            start_trial_id=start_trial_id,
            max_candidates=max_candidates,
            base_commit=base_commit,
            candidate_budget=candidate_budget,
            diagnostic_report=diagnostic_report,
        )
    if planner == "llm":
        return _llm_candidates(
            root=root,
            run_id=run_id,
            start_trial_id=start_trial_id,
            max_candidates=max_candidates,
            candidate_plan=candidate_plan,
            base_commit=base_commit,
            model=model,
            llm_policy=llm_policy or {},
            planner_client=planner_client,
            prior_outcomes=prior_outcomes,
            candidate_budget=candidate_budget,
        )
    raise AutoresearchLoopError(f"unsupported autoresearch planner: {planner}")


def _deterministic_candidates(*, start_trial_id: str, max_candidates: int, base_commit: str) -> list[dict[str, object]]:
    start = _trial_number(start_trial_id)
    if max_candidates < 1 or max_candidates > 6:
        raise AutoresearchLoopError("deterministic ladder max_candidates must be between 1 and 6")
    ladder = [
        ("short_train_baseline_smoke", TrialKind.TRAINING, BudgetTier.SMOKE, MoveFamily.CURRICULUM, 10, "configs/nanofold_dev_cpu_smoke.json"),
        ("first_geometry_patch_smoke", TrialKind.TRAINING, BudgetTier.SMOKE, MoveFamily.GEOMETRY_LOSS, 10, "configs/experiments/local_calpha_geometry_smoke.json"),
        ("short_train_baseline_trial", TrialKind.TRAINING, BudgetTier.TRIAL, MoveFamily.CURRICULUM, 250, "configs/nanofold_dev_cpu_smoke.json"),
        ("best_geometry_patch_trial", TrialKind.TRAINING, BudgetTier.TRIAL, MoveFamily.GEOMETRY_LOSS, 250, "configs/experiments/local_calpha_geometry_smoke.json"),
        ("no_geometry_aux_ablation", TrialKind.TRAINING, BudgetTier.TRIAL, MoveFamily.AUXILIARY_LOSS, 250, "configs/nanofold_dev_cpu_smoke.json"),
        ("sampler_after_best_checkpoint", TrialKind.SAMPLER, BudgetTier.SAMPLER, MoveFamily.DIFFUSION_SAMPLER_GOLF, None, "configs/experiments/local_calpha_geometry_smoke.json"),
    ]
    candidates: list[dict[str, object]] = []
    checkpoint_trial_id = f"T{start + 3:03d}"
    for offset, (name, kind, budget, move_family, max_steps, config_path) in enumerate(ladder[:max_candidates]):
        trial_id = f"T{start + offset:03d}"
        trial = _trial_payload(
            trial_id=trial_id,
            base_commit=base_commit,
            kind=kind,
            budget=budget,
            move_family=move_family,
            max_steps=max_steps,
            config_path=config_path,
            checkpoint_trial_id=checkpoint_trial_id if kind == TrialKind.SAMPLER else None,
            hypothesis=f"Deterministic ladder candidate {name} tests bounded local-geometry short-training behavior.",
        )
        candidates.append(
            {
                "hypothesis": trial["hypothesis"],
                "trial": trial,
                "config": {"config_path": config_path, "planned_candidate": name, "max_templates": 0},
                "patch_text": "",
            }
        )
    return candidates


def _targeted_diagnostic_candidates(
    *,
    root: Path,
    start_trial_id: str,
    max_candidates: int,
    base_commit: str,
    candidate_budget: str,
    diagnostic_report: str | Path | None,
) -> list[dict[str, object]]:
    if max_candidates != 1:
        raise AutoresearchLoopError("targeted_diagnostic planner requires max_candidates=1")
    if diagnostic_report is None:
        raise AutoresearchLoopError("targeted_diagnostic planner requires --diagnostic-report")
    target_summary = _targeted_diagnostic_summary(root=root, diagnostic_report=diagnostic_report)
    budget_shape = _candidate_budget_shape(candidate_budget)
    budget = BudgetTier(str(budget_shape["budget"]))
    trial_id = start_trial_id
    config_path = f"configs/experiments/{trial_id}_targeted_geometry_diagnostic.json"
    config_payload = _targeted_diagnostic_config_payload(root)
    worst_targets = target_summary["worst_targets"]
    trial = _trial_payload(
        trial_id=trial_id,
        base_commit=base_commit,
        kind=TrialKind.TRAINING,
        budget=budget,
        move_family=MoveFamily.GEOMETRY_LOSS,
        max_steps=int(budget_shape["max_steps"]),
        config_path=config_path,
        hypothesis=(
            "A bounded geometry-loss training candidate should test whether the recurring "
            f"reference-sweep loser targets ({', '.join(worst_targets)}) improve when the "
            "training objective emphasizes local C-alpha geometry stability without changing "
            "labels, manifests, scorer, templates, Modal resources, or ledger authority."
        ),
    )
    trial["agent_session_id"] = "targeted-diagnostic-planner"
    trial["seed"] = 90000 + _trial_number(trial_id)
    trial["max_wall_minutes"] = budget_shape["max_wall_minutes"]
    trial["timeout_cap"] = budget_shape["timeout_cap"]
    trial["config_payload"] = config_payload
    trial["prediction"] = RegisteredPrediction(
        causal_component="local_calpha_geometry_loss_weight",
        predicted_axis=FalsificationAxis.LOCAL_GEOMETRY,
        predicted_direction=PredictionDirection.UP,
        expected_lddt_delta_band=(0.001, 0.012),
    ).model_dump(mode="json")
    config = {
        "schema_version": "autoaf3.targeted_diagnostic_plan.v1",
        "config_path": config_path,
        "max_templates": 0,
        "source_diagnostic_report": str(diagnostic_report),
        "reference_trial_id": target_summary["reference_trial_id"],
        "candidate_trial_ids": target_summary["candidate_trial_ids"],
        "worst_targets": worst_targets,
        "target_loss_summary": target_summary["target_loss_summary"],
        "config_payload_overrides": {
            "learning_rate": config_payload["learning_rate"],
            "distogram_loss_weight": config_payload["distogram_loss_weight"],
            "local_calpha_geometry_loss_weight": config_payload["local_calpha_geometry_loss_weight"],
            "clip_norm": config_payload["clip_norm"],
        },
        "not_a_benchmark_claim": True,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
    }
    return [
        {
            "hypothesis": trial["hypothesis"],
            "trial": trial,
            "config": config,
            "patch_text": _targeted_diagnostic_patch_text(root=root, config_path=config_path, config=config),
        }
    ]


def _manual_candidates(*, root: Path, candidate_plan: str | Path | None) -> list[dict[str, object]]:
    if candidate_plan is None:
        raise AutoresearchLoopError("manual planner requires --candidate-plan")
    payload = _read_candidate_plan(root=root, candidate_plan=candidate_plan)
    candidates = payload.get("candidates", [payload])
    if not isinstance(candidates, list) or not candidates:
        raise AutoresearchLoopError("manual candidate plan must contain at least one candidate")
    checked = []
    for item in candidates:
        if not isinstance(item, dict) or not isinstance(item.get("trial"), dict):
            raise AutoresearchLoopError("manual candidate entries must contain a trial object")
        config = item.get("config")
        patch_text = str(item.get("patch_text", ""))
        if config is not None:
            if not isinstance(config, dict):
                raise AutoresearchLoopError("manual config must be an object")
            _refuse_plan_authority_claims(config, "manual config")
            _refuse_template_config(config)
        _refuse_unsafe_patch_text(root=root, patch_text=patch_text)
        checked.append(
            {
                "hypothesis": item.get("hypothesis") or item["trial"].get("hypothesis"),
                "trial": item["trial"],
                "config": config,
                "patch_text": patch_text,
            }
        )
    return checked


def _llm_candidates(
    *,
    root: Path,
    run_id: str,
    start_trial_id: str,
    max_candidates: int,
    candidate_plan: str | Path | None,
    base_commit: str,
    model: str,
    llm_policy: dict[str, dict[str, object]],
    planner_client: AutoresearchPlanner | None,
    prior_outcomes: list[dict[str, object]],
    candidate_budget: str,
) -> list[dict[str, object]]:
    if max_candidates < 1 or max_candidates > 3:
        raise AutoresearchLoopError("LLM autoresearch max_candidates must be between 1 and 3")
    if candidate_plan is not None and max_candidates != 1:
        raise AutoresearchLoopError("recorded LLM candidate plans can replay exactly one candidate")
    if candidate_plan is not None:
        raw_plan = _read_candidate_plan(root=root, candidate_plan=candidate_plan)
        return [
            _validate_llm_candidate_plan(
                root=root,
                raw_plan=raw_plan,
                expected_trial_id=start_trial_id,
                expected_budget=candidate_budget,
            )
        ]
    active_planner = planner_client or OpenAIAutoresearchPlanner(repo_root=root, model=model)
    planned: list[dict[str, object]] = []
    start = _trial_number(start_trial_id)
    for candidate_index in range(max_candidates):
        trial_id = f"T{start + candidate_index:03d}"
        try:
            raw_plan = active_planner.plan(
                run_id=run_id,
                trial_id=trial_id,
                candidate_index=candidate_index,
                model=model,
                policy=llm_policy,
                base_commit=base_commit,
                prior_plans=[
                    {
                        "trial_id": str(item["trial"]["trial_id"]),
                        "hypothesis": str(item["hypothesis"]),
                        "move_family": str(item["trial"].get("move_family")),
                        "diagnostic_target": str(item["trial"].get("diagnostic_target")),
                        "config": item.get("config"),
                    }
                    for item in planned
                ],
                prior_outcomes=prior_outcomes,
                candidate_budget=candidate_budget,
            )
        except Exception as exc:  # noqa: BLE001 - planner failures must stop before artifacts.
            raise AutoresearchLoopError(f"LLM autoresearch planner failed: {exc}") from exc
        planned.append(
            _validate_llm_candidate_plan(
                root=root,
                raw_plan=raw_plan,
                expected_trial_id=trial_id,
                expected_budget=candidate_budget,
            )
        )
    return planned


def _validate_llm_candidate_plan(
    *,
    root: Path,
    raw_plan: object,
    expected_trial_id: str,
    expected_budget: str,
) -> dict[str, object]:
    try:
        plan = AutoresearchCandidatePlan.model_validate(raw_plan)
        trial_payload = plan.trial.model_dump(mode="json")
        inline_config = trial_payload.get("config_payload")
        if isinstance(inline_config, dict):
            trial_payload["config_payload"] = {key: value for key, value in inline_config.items() if value is not None}
        AutoFoldTrial.model_validate(trial_payload)
        validate_patch_scope(plan.changed_paths, repo_root=root, allow_empty=True)
    except (ValueError, PatchPolicyError) as exc:
        raise AutoresearchLoopError(f"invalid LLM autoresearch plan: {exc}") from exc
    config_payload = plan.config.model_dump(mode="json")
    _refuse_plan_authority_claims(config_payload, "LLM config")
    _refuse_template_config(config_payload)
    patch_paths = _refuse_unsafe_patch_text(root=root, patch_text=plan.patch_text)
    if trial_payload["trial_id"] != expected_trial_id:
        raise AutoresearchLoopError("LLM candidate trial_id must match start_trial_id")
    expected_shape = _candidate_budget_shape(expected_budget)
    if trial_payload["budget"] != expected_shape["budget"]:
        raise AutoresearchLoopError(f"LLM candidate budget must be {expected_shape['budget']}")
    if trial_payload["max_steps"] != expected_shape["max_steps"]:
        raise AutoresearchLoopError(f"LLM candidate max_steps must be {expected_shape['max_steps']}")
    if trial_payload["max_wall_minutes"] != expected_shape["max_wall_minutes"]:
        raise AutoresearchLoopError(f"LLM candidate max_wall_minutes must be {expected_shape['max_wall_minutes']}")
    if trial_payload["timeout_cap"] != expected_shape["timeout_cap"]:
        raise AutoresearchLoopError(f"LLM candidate timeout_cap must be {expected_shape['timeout_cap']}")
    if patch_paths != set(plan.changed_paths):
        raise AutoresearchLoopError("LLM patch_text paths must match changed_paths")
    return {
        "hypothesis": plan.hypothesis,
        "trial": trial_payload,
        "config": config_payload,
        "patch_text": plan.patch_text,
    }


def _autoresearch_planner_prompt(
    *,
    root: Path,
    run_id: str,
    trial_id: str,
    candidate_index: int,
    base_commit: str,
    policy: dict[str, dict[str, object]],
    prior_plans: list[dict[str, object]],
    prior_outcomes: list[dict[str, object]],
    candidate_budget: str,
) -> str:
    base_config = _planner_reference_config(root / "configs" / "nanofold_dev_cpu_smoke.json")
    local_geometry_config = _planner_reference_config(root / "configs" / "experiments" / "local_calpha_geometry_smoke.json")
    budget_shape = _candidate_budget_shape(candidate_budget)
    payload = {
        "task": "Plan the next single bounded autoresearch candidate. Do not plan a batch.",
        "run_id": run_id,
        "trial_id": trial_id,
        "candidate_index": candidate_index,
        "base_commit": base_commit,
        "implementation_target": "NanoFold-style AlphaFold3-lite",
        "llm_policy": policy,
        "prior_planned_candidates": prior_plans[-5:],
        "prior_candidate_outcomes": prior_outcomes[-10:],
        "allowed_candidate_shape": {
            "trial_kind": "training",
            "budget": budget_shape["budget"],
            "max_steps": budget_shape["max_steps"],
            "max_wall_minutes": budget_shape["max_wall_minutes"],
            "artifact_dir": f"runs/trials/{trial_id}",
            "checkpoint_path": None,
            "primary_metric": "best_val_calpha_lddt",
            "scorer_version": "calpha_lddt_v1",
            "max_templates": 0,
        },
        "allowed_edit_surface": {
            "changed_paths": ["configs/experiments/<candidate-note>.json"],
            "config_path_prefix": "configs/experiments/",
            "allowed_existing_configs": sorted(ALLOWED_CONFIG_EXACT),
            "config_payload": "Use a full NanoFold config object derived from the supplied smoke configs for executable config changes.",
            "patch_text": "Use a harmless candidate-note JSON diff; executable config changes belong in trial.config_payload.",
        },
        "allowed_move_families": [item.value for item in MoveFamily],
        "allowed_diagnostic_targets": [item.value for item in DiagnosticTarget],
        "allowed_prediction": {
            "predicted_axis": [item.value for item in FalsificationAxis],
            "predicted_direction": [item.value for item in PredictionDirection],
            "expected_lddt_delta_band": "two non-negative floats, low <= high",
        },
        "required_trial_defaults": {
            "parent_commit": base_commit,
            "agent_session_id": "openai-autoresearch-planner",
            "seed": candidate_index,
            "n_res": 32,
            "manifest_hashes": {},
            "param_cap": 176514,
            "gpu_memory_cap": 80.0,
            "cost_cap": 2.0,
            "timeout_cap": budget_shape["timeout_cap"],
        },
        "hard_constraints": [
            "Return exactly one candidate object, not a candidates array.",
            "Choose a different candidate note path and hypothesis than prior_planned_candidates.",
            "Use prior_candidate_outcomes to avoid repeating discarded move families/config changes unless the new candidate isolates a different diagnostic axis.",
            "trial.trial_id must equal the requested trial_id.",
            "trial.artifact_dir must equal runs/trials/<trial_id>.",
            f"trial.budget, trial.max_steps, trial.max_wall_minutes, and trial.timeout_cap must exactly match {budget_shape}.",
            "trial.config_path must be repo-relative and under configs/experiments/ unless using an allowed existing config.",
            "If trial.config_payload is present, it must preserve max_templates=0 and include every required NanoFold config key.",
            "changed_paths must exactly match patch_text file paths.",
            "patch_text must be a real unified diff with hunk content.",
            "Do not include official_benchmark_result, writes_baseline, writes_ledger, writes_discovery_ledger, starts_search, or live_modal_execution true anywhere.",
            "Do not propose or mention reading locked labels or using the autoalphafold3-locked volume.",
            "Do not edit autoalphafold3/scorer, public manifests, fingerprints, runs/baseline, Modal app resources, GPU settings, Volumes, or cost caps.",
            "Do not add or use templates; max_templates remains 0.",
        ],
        "reference_configs": {
            "configs/nanofold_dev_cpu_smoke.json": base_config,
            "configs/experiments/local_calpha_geometry_smoke.json": local_geometry_config,
        },
    }
    return json.dumps(payload, allow_nan=False, sort_keys=True)


def _candidate_budget_shape(candidate_budget: str) -> dict[str, object]:
    if candidate_budget == BudgetTier.SMOKE.value:
        return {
            "budget": BudgetTier.SMOKE.value,
            "max_steps": 10,
            "max_wall_minutes": 5,
            "timeout_cap": 300,
        }
    if candidate_budget == BudgetTier.TRIAL.value:
        return {
            "budget": BudgetTier.TRIAL.value,
            "max_steps": 250,
            "max_wall_minutes": 45,
            "timeout_cap": 2700,
        }
    raise AutoresearchLoopError("candidate_budget must be smoke or trial")


def _targeted_diagnostic_summary(*, root: Path, diagnostic_report: str | Path) -> dict[str, object]:
    path = _diagnostic_report_path(root=root, diagnostic_report=diagnostic_report)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutoresearchLoopError(f"cannot read targeted diagnostic report: {diagnostic_report}") from exc
    if not isinstance(payload, dict):
        raise AutoresearchLoopError("targeted diagnostic report must be a JSON object")
    deltas = payload.get("per_target_score_deltas_vs_reference")
    if not isinstance(deltas, dict) or not deltas:
        raise AutoresearchLoopError("targeted diagnostic report missing per-target score deltas")
    target_stats: dict[str, dict[str, float | int]] = {}
    candidate_trial_ids: list[str] = []
    for trial_id, per_target in deltas.items():
        if not isinstance(per_target, dict):
            continue
        candidate_trial_ids.append(str(trial_id))
        for target_id, raw_delta in per_target.items():
            if not isinstance(raw_delta, int | float):
                continue
            delta = float(raw_delta)
            stats = target_stats.setdefault(
                str(target_id),
                {"negative_count": 0, "sum_negative_delta": 0.0, "min_delta": 0.0},
            )
            if delta < 0.0:
                stats["negative_count"] = int(stats["negative_count"]) + 1
                stats["sum_negative_delta"] = float(stats["sum_negative_delta"]) + delta
                stats["min_delta"] = min(float(stats["min_delta"]), delta)
    losers = [
        (target_id, stats)
        for target_id, stats in target_stats.items()
        if int(stats["negative_count"]) > 0
    ]
    if not losers:
        raise AutoresearchLoopError("targeted diagnostic report has no negative per-target deltas")
    losers.sort(
        key=lambda item: (
            -int(item[1]["negative_count"]),
            float(item[1]["sum_negative_delta"]),
            float(item[1]["min_delta"]),
            item[0],
        )
    )
    selected = losers[:4]
    return {
        "reference_trial_id": str(payload.get("reference_trial_id") or ""),
        "candidate_trial_ids": candidate_trial_ids,
        "worst_targets": [target_id for target_id, _stats in selected],
        "target_loss_summary": [
            {
                "target_id": target_id,
                "negative_count": int(stats["negative_count"]),
                "sum_negative_delta": float(stats["sum_negative_delta"]),
                "min_delta": float(stats["min_delta"]),
            }
            for target_id, stats in selected
        ],
    }


def _diagnostic_report_path(*, root: Path, diagnostic_report: str | Path) -> Path:
    path = Path(diagnostic_report)
    if path.is_absolute() or ".." in path.parts:
        raise AutoresearchLoopError("diagnostic report must be a repo-relative path without traversal")
    if not path.as_posix().startswith("runs/autoresearch/"):
        raise AutoresearchLoopError("diagnostic report must live under runs/autoresearch/")
    _refuse_plan_path_symlinks(root, path)
    return root / path


def _targeted_diagnostic_config_payload(root: Path) -> dict[str, object]:
    config_path = root / "configs" / "experiments" / "local_calpha_geometry_smoke.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutoresearchLoopError("targeted_diagnostic planner requires local_calpha_geometry_smoke config") from exc
    if not isinstance(payload, dict):
        raise AutoresearchLoopError("local_calpha_geometry_smoke config must be a JSON object")
    payload = dict(payload)
    payload["max_templates"] = 0
    payload["learning_rate"] = 0.0012
    payload["local_calpha_geometry_loss_weight"] = 0.4
    payload["distogram_loss_weight"] = 0.05
    payload["clip_norm"] = 5.0
    return payload


def _targeted_diagnostic_patch_text(*, root: Path, config_path: str, config: dict[str, object]) -> str:
    note = {
        "schema_version": config["schema_version"],
        "source_diagnostic_report": config["source_diagnostic_report"],
        "reference_trial_id": config["reference_trial_id"],
        "worst_targets": config["worst_targets"],
        "candidate_trial_ids": config["candidate_trial_ids"],
        "candidate_intent": "bounded local-geometry training diagnostic",
    }
    lines = json.dumps(note, allow_nan=False, indent=2, sort_keys=True).splitlines()
    patch = [
        f"diff --git a/{config_path} b/{config_path}",
        "--- /dev/null",
        f"+++ b/{config_path}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    patch.extend(f"+{line}" for line in lines)
    patch_text = "\n".join(patch) + "\n"
    _refuse_unsafe_patch_text(root=root, patch_text=patch_text)
    return patch_text


def _read_small_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _planner_reference_config(path: Path) -> dict[str, object]:
    payload = _read_small_json(path)
    payload.setdefault("diffusion_loss_weight", 4.0)
    payload.setdefault("dist_loss_weight", 0.0)
    payload.setdefault("distogram_loss_weight", 0.03)
    payload.setdefault("local_calpha_geometry_loss_weight", 0.0)
    return payload


def _extract_autoresearch_parsed_plan(response: object) -> AutoresearchCandidatePlan:
    for output in getattr(response, "output", []) or []:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []) or []:
            if getattr(item, "type", None) == "refusal":
                raise AutoresearchLoopError(f"LLM planner refused: {getattr(item, 'refusal', '')}")
            parsed = getattr(item, "parsed", None)
            if parsed is not None:
                return AutoresearchCandidatePlan.model_validate(parsed)
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return AutoresearchCandidatePlan.model_validate(parsed)
    raise AutoresearchLoopError("LLM planner returned no parsed autoresearch plan")


def _is_missing_openai_credentials(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "missing credentials" in message or ("api_key" in message and "environment variable" in message)


def _plan_autoresearch_with_modal_harness_secret(
    *,
    prompt: str,
    trial_id: str,
    candidate_index: int,
    base_commit: str,
    policy: dict[str, dict[str, object]],
    prior_plans: list[dict[str, object]],
    prior_outcomes: list[dict[str, object]],
    candidate_budget: str,
    model: str,
) -> AutoresearchCandidatePlan:
    try:
        import modal
    except ModuleNotFoundError as exc:
        raise AutoresearchLoopError("local OpenAI credentials are missing and Modal SDK is unavailable") from exc

    payload = {
        "prompt": prompt,
        "trial_id": trial_id,
        "candidate_index": candidate_index,
        "base_commit": base_commit,
        "policy": policy,
        "prior_plans": prior_plans[-5:],
        "prior_outcomes": prior_outcomes[-10:],
        "candidate_budget": candidate_budget,
        "model": model,
    }
    orchestrator = modal.Cls.from_name(APP_NAME, TRUSTED_ORCHESTRATOR_CLASS)()
    result = orchestrator.plan_autoresearch_candidate.remote(payload)
    return AutoresearchCandidatePlan.model_validate(result)


def _read_candidate_plan(*, root: Path, candidate_plan: str | Path | None) -> dict[str, object]:
    if candidate_plan is None:
        raise AutoresearchLoopError("candidate plan path is required")
    path = Path(candidate_plan)
    if path.is_absolute() or ".." in path.parts:
        raise AutoresearchLoopError("candidate plan must be a repo-relative path without traversal")
    if not path.as_posix().startswith("configs/experiments/"):
        raise AutoresearchLoopError("candidate plan must live under configs/experiments/")
    _refuse_plan_path_symlinks(root, path)
    path = root / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AutoresearchLoopError("candidate plan must be a JSON object")
    return payload


def _refuse_plan_path_symlinks(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AutoresearchLoopError(f"candidate plan path must not contain symlinks: {current}")


def _llm_policy_specs(model: str) -> dict[str, dict[str, object]]:
    policies = {
        AgentSearchPhase.HYPOTHESIS_GENERATION: default_llm_phase_policy(
            AgentSearchPhase.HYPOTHESIS_GENERATION,
            model=model,
        ),
        AgentSearchPhase.PATCH_PLANNING: default_llm_phase_policy(
            AgentSearchPhase.PATCH_PLANNING,
            model=model,
        ),
    }
    return {phase.value: policy.to_responses_create_kwargs() for phase, policy in policies.items()}


def _trial_payload(
    *,
    trial_id: str,
    base_commit: str,
    kind: TrialKind,
    budget: BudgetTier,
    move_family: MoveFamily,
    max_steps: int | None,
    config_path: str,
    checkpoint_trial_id: str | None = None,
    hypothesis: str,
) -> dict[str, object]:
    prediction = RegisteredPrediction(
        causal_component=move_family.value,
        predicted_axis=FalsificationAxis.LOCAL_GEOMETRY,
        predicted_direction=PredictionDirection.UP,
        expected_lddt_delta_band=(0.001, 0.01),
    ).model_dump(mode="json")
    payload: dict[str, object] = {
        "trial_id": trial_id,
        "parent_commit": base_commit,
        "agent_session_id": "deterministic-autoresearch-ladder",
        "trial_kind": kind.value,
        "hypothesis": hypothesis,
        "move_family": move_family.value,
        "diagnostic_target": DiagnosticTarget.LOCAL_GEOMETRY_WEAK.value,
        "prediction": prediction,
        "patch_path": None,
        "config_path": config_path,
        "budget": budget.value,
        "seed": 0,
        "n_res": 32,
        "max_wall_minutes": 5 if budget == BudgetTier.SMOKE else 45,
        "manifest_hashes": {},
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "param_cap": 176514,
        "gpu_memory_cap": 80.0,
        "cost_cap": 2.0,
        "timeout_cap": 2700,
        "artifact_dir": f"runs/trials/{trial_id}",
        "checkpoint_path": None,
    }
    if kind == TrialKind.SAMPLER:
        if checkpoint_trial_id is None:
            raise AutoresearchLoopError("sampler candidate requires a planned checkpoint trial")
        payload.update(
            {
                "checkpoint_path": f"runs/trials/{checkpoint_trial_id}/checkpoint.pt",
                "sampler_steps": 2,
                "sampler_noise_scale": 1.0,
                "sampler_step_scale": 1.0,
                "sampler_schedule_shape": "linear",
                "sampler_num_samples": 1,
                "sampler_selection_policy": "first",
                "max_wall_minutes": 5,
                "timeout_cap": 300,
            }
        )
    else:
        payload["max_steps"] = max_steps
    return payload


def _planned_preflight(trial: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "autoaf3.autoresearch_preflight_plan.v1",
        "trial_id": trial["trial_id"],
        "status": TrialStatus.DRAFT.value,
        "planning_status": "PLANNED",
        "mode": "dry-run",
        "max_templates": 0,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
    }


def _validate_trial_artifacts(trial: dict[str, object]) -> None:
    trial_id = str(trial["trial_id"])
    _validate_config_path(str(trial["config_path"]))
    _validate_config_payload(trial.get("config_payload"))
    expected_artifact_dir = f"runs/trials/{trial_id}"
    artifact_dir = trial.get("artifact_dir")
    if artifact_dir != expected_artifact_dir:
        raise AutoresearchLoopError(f"trial artifact_dir must be {expected_artifact_dir}")
    checkpoint_path = trial.get("checkpoint_path")
    if checkpoint_path is not None:
        path = Path(str(checkpoint_path))
        if path.is_absolute() or ".." in path.parts:
            raise AutoresearchLoopError("checkpoint_path must be repo-relative without traversal")
        if not str(checkpoint_path).startswith("runs/trials/"):
            raise AutoresearchLoopError("checkpoint_path must stay under runs/trials/")


def _validate_unique_trial_ids(planned: list[dict[str, object]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for candidate in planned:
        trial_id = str(candidate["trial"]["trial_id"])
        if trial_id in seen:
            duplicates.add(trial_id)
        seen.add(trial_id)
    if duplicates:
        raise AutoresearchLoopError(f"candidate plan contains duplicate trial_id values: {sorted(duplicates)}")


def _validate_config_path(config_path: str) -> None:
    path = Path(config_path)
    if path.is_absolute() or ".." in path.parts:
        raise AutoresearchLoopError("config_path must be repo-relative without traversal")
    normalized = path.as_posix()
    if normalized in ALLOWED_CONFIG_EXACT or any(normalized.startswith(prefix) for prefix in ALLOWED_CONFIG_PREFIXES):
        return
    raise AutoresearchLoopError(f"config_path is outside the planning config surface: {config_path}")


def _validate_config_payload(payload: object) -> None:
    if payload is None:
        return
    _refuse_plan_authority_claims(payload, "config_payload")
    if not isinstance(payload, dict):
        raise AutoresearchLoopError("config_payload must be an object")
    result = validate_nanofold_config_payload(payload, source="config_payload")
    if not result.valid:
        raise AutoresearchLoopError(f"config_payload is invalid: {result.missing_keys}")


def _write_planned_candidate_index(*, root: Path, run_id: str, records: list[dict[str, object]]) -> None:
    run_dir = root / "runs" / "autoresearch" / run_id
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = [
        _planned_summary_record(record)
        for record in records
    ]
    summary["candidates"] = candidates
    results_tmp = (run_dir / "results.tsv").with_suffix(".tsv.tmp")
    with results_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "trial_id",
                "candidate_id",
                "status",
                "primary_metric",
                "matched_budget_delta",
                "global_baseline_delta",
                "provisional_keep",
                "decision_path",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["trial_id"],
                    record["candidate_id"],
                    record.get("status", TrialStatus.DRAFT.value),
                    "best_val_calpha_lddt",
                    _tsv_optional(record.get("matched_budget_delta")),
                    _tsv_optional(record.get("global_baseline_delta")),
                    str(bool(record.get("provisional_keep", False))).lower(),
                    record.get("decision_path") or "",
                ]
            )
    results_tmp.replace(run_dir / "results.tsv")
    _atomic_write_json(summary_path, summary)


def _planned_summary_record(record: dict[str, object]) -> dict[str, object]:
    summary_record = {
        "trial_id": record["trial_id"],
        "candidate_id": record["candidate_id"],
        "status": record.get("status", TrialStatus.DRAFT.value),
        "planning_status": record["planning_status"],
        "decision_path": record.get("decision_path"),
        "postmortem_path": record.get("postmortem_path"),
        "matched_budget_delta": record.get("matched_budget_delta"),
        "global_baseline_delta": record.get("global_baseline_delta"),
        "provisional_keep": bool(record.get("provisional_keep", False)),
        "promotion_status": record.get("promotion_status"),
        "promotion_plan_path": record.get("promotion_plan_path"),
    }
    for key in (
        "execution_status",
        "training_status",
        "training_manifest_path",
        "trial_artifact_dir",
        "benchmark_decision",
        "writes_ledger",
        "writes_discovery_ledger",
        "official_benchmark_result",
    ):
        if key in record:
            summary_record[key] = record[key]
    return summary_record


def _tsv_optional(value: object) -> object:
    return "" if value is None else value


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _refuse_plan_authority_claims(payload: object, label: str) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in FORBIDDEN_TRUE_PLAN_FLAGS and value is not False:
                raise AutoresearchLoopError(f"{label} cannot claim {key}={value!r}")
            _refuse_plan_authority_claims(value, label)
    elif isinstance(payload, list):
        for item in payload:
            _refuse_plan_authority_claims(item, label)


def _refuse_template_config(config: dict[str, object]) -> None:
    if config.get("max_templates", 0) != 0:
        raise AutoresearchLoopError("manual config must pin max_templates=0")


def _refuse_unsafe_patch_text(*, root: Path, patch_text: str) -> set[str]:
    if not patch_text.strip():
        return set()
    paths: set[str] = set()
    added_lines: list[str] = []
    has_hunk_content = False
    in_hunk = False
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
            parts = line.split()
            if len(parts) < 4 or not parts[2].startswith("a/") or not parts[3].startswith("b/"):
                raise AutoresearchLoopError(f"unsupported manual patch header: {line}")
            paths.update(path[2:] for path in (parts[2], parts[3]))
        elif line.startswith(("--- ", "+++ ")):
            path = line[4:]
            if path == "/dev/null":
                continue
            if path.startswith(("a/", "b/")):
                paths.add(path[2:])
            else:
                raise AutoresearchLoopError(f"unsupported manual patch file header: {line}")
        elif line.startswith("@@"):
            in_hunk = True
        elif line.startswith("+") and not line.startswith("+++"):
            if not in_hunk:
                raise AutoresearchLoopError("manual patch_text hunk content must follow a hunk header")
            has_hunk_content = True
            if any(token in line for token in LOCKED_READ_TOKENS):
                raise AutoresearchLoopError("manual patch_text appears to read locked labels")
            added_lines.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            if not in_hunk:
                raise AutoresearchLoopError("manual patch_text hunk content must follow a hunk header")
            has_hunk_content = True
    if not paths:
        raise AutoresearchLoopError("manual patch_text must include file headers")
    try:
        validate_patch_scope(sorted(paths), repo_root=root, allow_empty=False)
    except PatchPolicyError as exc:
        raise AutoresearchLoopError(str(exc)) from exc
    if not has_hunk_content:
        raise AutoresearchLoopError("manual patch_text must include at least one patch hunk content line")
    _refuse_patch_authority_text("\n".join(added_lines))
    return paths


def _refuse_patch_authority_text(text: str) -> None:
    for key in PATCH_FORBIDDEN_KEYS:
        pattern = rf"['\"]?{re.escape(key)}['\"]?\s*:\s*([^,}}\s]+)"
        for match in re.finditer(pattern, text, flags=re.MULTILINE):
            value = match.group(1).strip().strip("'\"").lower()
            if key == "max_templates" and value in {"0", "0.0"}:
                continue
            if key != "max_templates" and value == "false":
                continue
            raise AutoresearchLoopError(f"manual patch_text cannot claim {key}")


def _trial_number(trial_id: str) -> int:
    if not trial_id.startswith("T"):
        raise AutoresearchLoopError(f"invalid start trial id: {trial_id}")
    try:
        return int(trial_id[1:])
    except ValueError as exc:
        raise AutoresearchLoopError(f"invalid start trial id: {trial_id}") from exc


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"
