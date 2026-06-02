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

from autoalphafold3.llm_policy import DEFAULT_LLM_MODEL, AgentSearchPhase, default_llm_phase_policy
from autoalphafold3.patch_policy import PatchPolicyError, validate_patch_scope
from autoalphafold3.autoresearch_candidates import (
    create_candidate_envelope,
    create_run_manifest,
    write_candidate_decision,
    write_candidate_evidence,
)
from autoalphafold3.autoresearch_comparisons import (
    AutoresearchComparisonError,
    compare_and_write_candidate_decision,
)
from autoalphafold3.modal_app import APP_NAME, TRUSTED_ORCHESTRATOR_CLASS
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
    ) -> "AutoresearchCandidatePlan":
        """Return one structured autoresearch candidate."""


class TrustedAutoresearchClient(Protocol):
    """Injected trusted-orchestrator client seam for live Modal autoresearch."""

    def submit_and_poll_trial(self, trial: dict[str, object]) -> dict[str, object]:
        """Submit one trial through the trusted orchestrator and poll the spawned worker."""

    def score_trial(self, trial_id: str) -> dict[str, object]:
        """Score one trial through the scorer-only Modal worker."""


class AutoresearchCandidatePlan(BaseModel):
    """Strict LLM/recorded plan for exactly one autoresearch candidate."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=20)
    trial: dict[str, object]
    changed_paths: list[str] = Field(min_length=1)
    config: dict[str, object] | None = None
    patch_text: str = ""
    rationale: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def reject_multi_candidate_shape(cls, value: object) -> object:
        if isinstance(value, dict) and "candidates" in value:
            raise ValueError("LLM autoresearch planner must return exactly one candidate")
        return value

    @model_validator(mode="after")
    def validate_one_move_contract(self) -> "AutoresearchCandidatePlan":
        if not isinstance(self.trial.get("trial_id"), str):
            raise ValueError("LLM candidate trial must include trial_id")
        if not isinstance(self.trial.get("move_family"), str):
            raise ValueError("LLM candidate trial must include one move_family")
        if not isinstance(self.trial.get("diagnostic_target"), str):
            raise ValueError("LLM candidate trial must include one diagnostic_target")
        if self.config is not None and self.config.get("max_templates", 0) != 0:
            raise ValueError("LLM candidate config must preserve max_templates=0")
        return self


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
) -> AutoresearchLoopResult:
    """Plan autoresearch candidates and optionally run one approved Modal candidate."""

    if mode not in {"dry-run", "modal"}:
        raise AutoresearchLoopError(f"unsupported autoresearch mode: {mode}")
    if planner not in {"manual", "deterministic", "llm"}:
        raise AutoresearchLoopError(f"unsupported autoresearch planner for this PR: {planner}")
    if mode == "modal":
        if approval != APPROVAL_TEXT:
            raise AutoresearchLoopError(f"live autoresearch requires --approve {APPROVAL_TEXT}")
        if max_candidates != 1:
            raise AutoresearchLoopError("live autoresearch currently supports exactly one candidate")

    root = Path(repo_root)
    base_commit = _git_head(root)
    llm_policy = _llm_policy_specs(model) if planner == "llm" else None
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
            )
            decisions[-1].update(live["decision"])
            wrote_files.extend(live["wrote_files"])
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
) -> dict[str, object]:
    checked = AutoFoldTrial.model_validate(trial)
    if checked.trial_kind != TrialKind.TRAINING:
        raise AutoresearchLoopError("live autoresearch smoke currently supports training candidates only")
    client = modal_client if modal_client is not None else DeployedTrustedAutoresearchClient(environment_name=modal_env)
    try:
        payload = client.submit_and_poll_trial(_modal_short_training_payload(checked))
    except Exception as exc:  # noqa: BLE001 - normalize delegated runner failures.
        raise AutoresearchLoopError(f"live autoresearch trusted-orchestrator trial failed: {exc}") from exc
    wrote_files: list[str] = []
    decision_overrides: dict[str, object] = {}
    if _is_short_training_manifest(payload):
        wrote_files.extend(_record_short_training_manifest(envelope=envelope, payload=payload))
        decision_overrides.update(
            {
                "training_manifest_path": str(envelope.training_manifest_path),
                "training_status": payload.get("status"),
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
    )
    scored["wrote_files"] = [*wrote_files, *scored["wrote_files"]]
    return scored


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
                matched_budget_result=None,
                repo_root=root,
                baseline_dir="runs/baseline",
                ledger_path="runs/ledger.jsonl",
            )
        except AutoresearchComparisonError as exc:
            raise AutoresearchLoopError(f"live autoresearch comparison failed: {exc}") from exc
        wrote_files.extend([str(envelope.metrics_path), str(envelope.decision_path), str(envelope.postmortem_path)])
        decision.update(comparison.to_dict())
        decision["decision_path"] = str(envelope.decision_path)
        return {"decision": decision, "wrote_files": wrote_files}
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


def _record_short_training_manifest(*, envelope, payload: dict[str, object]) -> list[str]:
    if payload.get("trial_id") != envelope.trial_id:
        raise AutoresearchLoopError("live autoresearch short-training manifest trial_id mismatch")
    return write_candidate_evidence(envelope, training_manifest=payload)


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
) -> list[dict[str, object]]:
    if planner == "manual":
        return _manual_candidates(root=root, candidate_plan=candidate_plan)
    if planner == "deterministic":
        return _deterministic_candidates(start_trial_id=start_trial_id, max_candidates=max_candidates, base_commit=base_commit)
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
) -> list[dict[str, object]]:
    if max_candidates != 1:
        raise AutoresearchLoopError("LLM autoresearch planner authors exactly one candidate per run")
    if planner_client is None and candidate_plan is None:
        raise AutoresearchLoopError("LLM autoresearch CLI planning requires --candidate-plan recorded output; live LLM planning is not enabled")
    if planner_client is not None:
        try:
            raw_plan = planner_client.plan(
                run_id=run_id,
                trial_id=start_trial_id,
                candidate_index=0,
                model=model,
                policy=llm_policy,
                base_commit=base_commit,
            )
        except Exception as exc:  # noqa: BLE001 - planner failures must stop before artifacts.
            raise AutoresearchLoopError(f"LLM autoresearch planner failed: {exc}") from exc
    else:
        raw_plan = _read_candidate_plan(root=root, candidate_plan=candidate_plan)
    try:
        plan = AutoresearchCandidatePlan.model_validate(raw_plan)
        AutoFoldTrial.model_validate(plan.trial)
        validate_patch_scope(plan.changed_paths, repo_root=root, allow_empty=True)
    except (ValueError, PatchPolicyError) as exc:
        raise AutoresearchLoopError(f"invalid LLM autoresearch plan: {exc}") from exc
    if plan.config is not None:
        _refuse_plan_authority_claims(plan.config, "LLM config")
        _refuse_template_config(plan.config)
    patch_paths = _refuse_unsafe_patch_text(root=root, patch_text=plan.patch_text)
    if plan.trial["trial_id"] != start_trial_id:
        raise AutoresearchLoopError("LLM candidate trial_id must match start_trial_id")
    if patch_paths != set(plan.changed_paths):
        raise AutoresearchLoopError("LLM patch_text paths must match changed_paths")
    return [
        {
            "hypothesis": plan.hypothesis,
            "trial": plan.trial,
            "config": plan.config,
            "patch_text": plan.patch_text,
        }
    ]


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
