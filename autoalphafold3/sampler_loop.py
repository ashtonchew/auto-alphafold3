"""Incremental sampler autoresearch loop."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoalphafold3.baseline_readiness import current_best_from_baseline_and_ledger
from autoalphafold3.ledger import LEDGER_WRITER_ROLE, append_ledger
from autoalphafold3.llm_policy import DEFAULT_LLM_MODEL, AgentSearchPhase, default_llm_phase_policy
from autoalphafold3.orchestrator import DEFAULT_KEEP_DELTA, decide_stage_one_result, submit_trial
from autoalphafold3.schema import AutoFoldResult, AutoFoldTrial, FoldCartographerReport, PRIMARY_METRIC, TrialStatus

APPROVAL_TEXT = "I_APPROVE_AUTONOMOUS_SAMPLER_LOOP"
SAMPLER_STRATEGY_REGRESSION_LIMIT = 3

_PLANNER_SYSTEM_PROMPT = """You are the NanoFold-style AlphaFold3-lite sampler planner.
Return exactly one JSON plan matching the provided schema.
You may only vary frozen-checkpoint inference-time sampler settings: sampler_steps, seed, noise/step scale, schedule shape, sample count, and label-free selection policy.
Do not propose scorer, label, manifest, Modal, GPU, Volume, template, checkpoint-training, or gate changes.
The objective is best_val_calpha_lddt; diagnostics only route hypotheses.
Candidate plans must be falsifiable and pre-registered before the run."""


class SamplerLoopError(RuntimeError):
    """Raised when the sampler loop cannot run safely."""


class SamplerLoopClient(Protocol):
    """Client seam for Modal and tests."""

    def submit(self, trial_path: Path) -> str:
        """Submit one trial and return the trusted-orchestrator call id."""

    def wait_for_sampler(self, call_id: str, *, timeout_s: int, poll_interval_s: float) -> dict[str, object]:
        """Wait until the sampler worker returns its manifest."""

    def score(self, trial_id: str) -> dict[str, object]:
        """Score one completed trial through the locked scorer."""


class SamplerCandidatePlan(BaseModel):
    """Structured LLM/deterministic plan for exactly one sampler candidate."""

    model_config = ConfigDict(extra="forbid")

    diagnostic_target: str = Field(min_length=1)
    hypothesis: str = Field(min_length=20)
    intervention: str = Field(min_length=1)
    predicted_direction: str = Field(pattern="^(up|down)$")
    expected_lddt_delta_band: list[float] = Field(min_length=2, max_length=2)
    sampler_steps: int = Field(ge=1, le=12)
    sampler_noise_scale: float = Field(ge=0.25, le=2.0)
    sampler_step_scale: float = Field(ge=0.25, le=2.0)
    sampler_schedule_shape: str = Field(pattern="^(linear|cosine|late_refine)$")
    sampler_num_samples: int = Field(ge=1, le=4)
    sampler_selection_policy: str = Field(pattern="^(first|geometry|compact_geometry)$")
    seed: int = Field(ge=0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sampler_only_plan(self) -> "SamplerCandidatePlan":
        if self.diagnostic_target not in {
            "local_geometry_weak",
            "long_range_topology_weak",
            "distogram_good_lddt_flat",
            "stability_compute",
        }:
            raise ValueError("diagnostic_target must be one canonical Fold Cartographer target")
        low, high = self.expected_lddt_delta_band
        if low > high:
            raise ValueError("expected_lddt_delta_band must be ordered")
        if low < 0 or high < 0:
            raise ValueError("expected_lddt_delta_band must be non-negative")
        text = f"{self.hypothesis} {self.intervention} {self.rationale}".lower()
        forbidden = (
            "change scorer",
            "edit scorer",
            "modify scorer",
            "change label",
            "edit label",
            "modify label",
            "read validation label",
            "change manifest",
            "edit manifest",
            "modify manifest",
            "change modal",
            "edit modal",
            "modify modal",
            "change gpu",
            "edit gpu",
            "modify gpu",
            "change volume",
            "edit volume",
            "modify volume",
            "checkpoint training",
            "train checkpoint",
            "use template",
            "add template",
        )
        if any(token in text for token in forbidden):
            raise ValueError("sampler plan must not touch scorer, labels, manifests, Modal policy, training, or templates")
        return self


class SamplerPlanner(Protocol):
    """One-candidate-at-a-time planner boundary."""

    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        global_current_best: dict[str, object],
        search_reference: dict[str, object],
        strategy_context: dict[str, object],
    ) -> SamplerCandidatePlan:
        """Plan the next sampler candidate from observed loop state."""


@dataclass(frozen=True)
class SamplerLoopResult:
    """Structured summary for one incremental sampler loop run."""

    status: str
    mode: str
    generated_trials: list[str]
    scored_trials: list[str]
    decisions: list[dict[str, object]]
    best_trial_id: str | None
    best_score: float | None
    planner: str
    stopped_reason: str
    wrote_files: list[str]
    starts_search: bool
    global_current_best: dict[str, object]
    search_reference: dict[str, object]
    writes_discovery_ledger: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_incremental_sampler_loop(
    *,
    seed_trial_path: str | Path,
    repo_root: str | Path = ".",
    output_dir: str | Path = "trials",
    ledger_path: str | Path = "runs/ledger.jsonl",
    baseline_dir: str | Path = "runs/baseline",
    mode: str = "dry-run",
    approval: str | None = None,
    max_candidates: int = 3,
    start_trial_id: str | None = None,
    poll_interval_s: float = 2.0,
    per_candidate_timeout_s: int = 180,
    failure_streak_limit: int = 2,
    planner: str = "deterministic",
    model: str = DEFAULT_LLM_MODEL,
    search_reference_trial_id: str | None = None,
    prior_decision_trial_ids: list[str] | None = None,
    client: SamplerLoopClient | None = None,
    planner_client: SamplerPlanner | None = None,
) -> SamplerLoopResult:
    """Run candidates sequentially, using prior scored results to choose the next."""

    if max_candidates < 1 or max_candidates > 20:
        raise SamplerLoopError("max_candidates must be between 1 and 20")
    if failure_streak_limit < 1:
        raise SamplerLoopError("failure_streak_limit must be at least 1")
    if mode not in {"dry-run", "modal"}:
        raise SamplerLoopError(f"unsupported mode: {mode}")
    if planner not in {"deterministic", "reference_sweep", "llm"}:
        raise SamplerLoopError(f"unsupported planner: {planner}")
    if mode == "modal" and approval != APPROVAL_TEXT:
        raise SamplerLoopError(f"modal sampler loop requires --approve {APPROVAL_TEXT}")

    root = Path(repo_root)
    seed_path = Path(seed_trial_path)
    if not seed_path.is_absolute():
        seed_path = root / seed_path
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if seed.get("trial_kind") != "sampler":
        raise SamplerLoopError("seed trial must be sampler trial_kind")

    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    scored_trials: list[str] = []
    decisions: list[dict[str, object]] = []
    wrote_files: list[str] = []
    best_trial_id: str | None = None
    best_score: float | None = None
    failure_streak = 0
    stopped_reason = "max_candidates_reached"
    next_id = _trial_number(start_trial_id or _next_trial_id(seed["trial_id"]))
    loop_client = client or ModalSamplerLoopClient(repo_root=root, ledger_path=ledger_path)
    active_planner = planner_client or _build_planner(planner, repo_root=root, baseline_dir=baseline_dir, ledger_path=ledger_path, model=model)
    global_current_best = _current_best_context(root=root, baseline_dir=baseline_dir, ledger_path=ledger_path)
    search_reference = _search_reference_context(
        root=root,
        ledger_path=ledger_path,
        trial_id=search_reference_trial_id,
    )
    if prior_decision_trial_ids:
        decisions = _prior_decisions_from_ledger(
            root=root,
            ledger_path=ledger_path,
            trial_ids=prior_decision_trial_ids,
            global_current_best=global_current_best,
            search_reference=search_reference,
        )

    for index in range(max_candidates):
        trial_id = f"T{next_id + index:03d}"
        strategy_context = _sampler_strategy_context(
            root=root,
            ledger_path=ledger_path,
            prior_decisions=decisions,
        )
        try:
            plan = active_planner.plan(
                seed_trial=seed,
                trial_id=trial_id,
                candidate_index=index,
                prior_decisions=decisions,
                global_current_best=global_current_best,
                search_reference=search_reference,
                strategy_context=strategy_context,
            )
        except Exception as exc:  # noqa: BLE001 - invalid autonomous plans must stop before submit.
            raise SamplerLoopError(f"planner failed for {trial_id}: {exc}") from exc
        _validate_strategy_gate(plan=plan, strategy_context=strategy_context)
        trial = _candidate_trial(seed, trial_id=trial_id, plan=plan)
        AutoFoldTrial.model_validate(trial)
        trial_path = out / f"{trial_id}.json"
        if trial_path.exists():
            raise SamplerLoopError(f"candidate trial already exists: {trial_path}")
        trial_path.write_text(json.dumps(trial, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        generated.append(trial_id)
        wrote_files.append(str(trial_path))

        if mode == "dry-run":
            decisions.append(
                {
                    "trial_id": trial_id,
                    "status": "PLANNED",
                    "sampler_steps": trial["sampler_steps"],
                    "seed": trial["seed"],
                    "planner": planner,
                    "hypothesis": plan.hypothesis,
                    "sampler_noise_scale": trial.get("sampler_noise_scale"),
                    "sampler_step_scale": trial.get("sampler_step_scale"),
                    "sampler_schedule_shape": trial.get("sampler_schedule_shape"),
                    "sampler_num_samples": trial.get("sampler_num_samples"),
                    "sampler_selection_policy": trial.get("sampler_selection_policy"),
                    "reason": "dry-run generated candidate only",
                    "global_current_best": global_current_best,
                    "search_reference": search_reference,
                    "strategy_context": strategy_context,
                }
            )
            continue

        try:
            call_id = loop_client.submit(trial_path)
            manifest = loop_client.wait_for_sampler(
                call_id,
                timeout_s=per_candidate_timeout_s,
                poll_interval_s=poll_interval_s,
            )
            _require_sampler_manifest(manifest, trial_id=trial_id)
            scored_payload = loop_client.score(trial_id)
            scored = _score_payload_to_result(scored_payload)
            append_ledger(
                scored,
                ledger_path=root / ledger_path,
                dedupe=True,
                validate_lifecycle=False,
                writer_role=LEDGER_WRITER_ROLE,
            )
            decision = decide_stage_one_result(
                scored,
                repo_root=root,
                baseline_dir=baseline_dir,
                ledger_path=ledger_path,
            )
            append_ledger(
                decision,
                ledger_path=root / ledger_path,
                dedupe=True,
                validate_lifecycle=True,
                writer_role=LEDGER_WRITER_ROLE,
            )
            score = _score(scored)
            if search_reference.get("status") != "available" and (
                search_reference_trial_id is None or scored.trial_id == search_reference_trial_id
            ):
                search_reference = _search_reference_from_result(scored)
            comparisons = _candidate_comparisons(
                score=score,
                global_current_best=global_current_best,
                search_reference=search_reference,
                global_keep=decision.status == TrialStatus.KEEP,
            )
            scored_trials.append(trial_id)
            decisions.append(
                {
                    "trial_id": trial_id,
                    "status": decision.status.value,
                    "score": score,
                    "sampler_steps": trial["sampler_steps"],
                    "seed": trial["seed"],
                    "planner": planner,
                    "hypothesis": plan.hypothesis,
                    "sampler_noise_scale": trial.get("sampler_noise_scale"),
                    "sampler_step_scale": trial.get("sampler_step_scale"),
                    "sampler_schedule_shape": trial.get("sampler_schedule_shape"),
                    "sampler_num_samples": trial.get("sampler_num_samples"),
                    "sampler_selection_policy": trial.get("sampler_selection_policy"),
                    "worker_status": manifest.get("status"),
                    "num_failed_targets": scored.metrics.get("num_failed_targets"),
                    "fold_cartographer": _compact_fold_cartographer(scored.fold_cartographer),
                    "strategy_context": strategy_context,
                    **comparisons,
                }
            )
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_trial_id = trial_id
            global_current_best = _current_best_context(root=root, baseline_dir=baseline_dir, ledger_path=ledger_path)
            failure_streak = 0
        except Exception as exc:  # noqa: BLE001 - loop must stop quickly on repeated live failures.
            failure_streak += 1
            decisions.append(
                {
                    "trial_id": trial_id,
                    "status": TrialStatus.INFRA_FAIL.value,
                    "failure_signature": type(exc).__name__,
                    "postmortem": str(exc),
                    "sampler_steps": trial["sampler_steps"],
                    "seed": trial["seed"],
                    "planner": planner,
                    "hypothesis": plan.hypothesis,
                    "sampler_noise_scale": trial.get("sampler_noise_scale"),
                    "sampler_step_scale": trial.get("sampler_step_scale"),
                    "sampler_schedule_shape": trial.get("sampler_schedule_shape"),
                    "sampler_num_samples": trial.get("sampler_num_samples"),
                    "sampler_selection_policy": trial.get("sampler_selection_policy"),
                    "global_current_best": global_current_best,
                    "search_reference": search_reference,
                }
            )
            if failure_streak >= failure_streak_limit:
                stopped_reason = f"failure_streak_limit:{type(exc).__name__}"
                break

    return SamplerLoopResult(
        status="PASS" if mode == "dry-run" or scored_trials else "FAIL",
        mode=mode,
        generated_trials=generated,
        scored_trials=scored_trials,
        decisions=decisions,
        best_trial_id=best_trial_id,
        best_score=best_score,
        planner=planner,
        stopped_reason=stopped_reason,
        wrote_files=wrote_files,
        starts_search=mode == "modal",
        global_current_best=global_current_best,
        search_reference=search_reference,
    )


class ModalSamplerLoopClient:
    """Modal implementation for the incremental sampler loop."""

    def __init__(self, *, repo_root: str | Path = ".", ledger_path: str | Path = "runs/ledger.jsonl") -> None:
        self.repo_root = Path(repo_root)
        self.ledger_path = ledger_path

    def submit(self, trial_path: Path) -> str:
        return submit_trial(
            trial_path,
            repo_root=self.repo_root,
            ledger_path=self.ledger_path,
            mode="modal",
            strict_nanofold_gates=False,
            enforce_git_diff=False,
        )

    def wait_for_sampler(self, call_id: str, *, timeout_s: int, poll_interval_s: float) -> dict[str, object]:
        import modal

        orchestrator_payload = _wait_modal_payload(call_id, timeout_s=timeout_s, poll_interval_s=poll_interval_s)
        worker_call_id = _worker_call_id(orchestrator_payload)
        return _wait_modal_payload(f"modal:{worker_call_id}", timeout_s=timeout_s, poll_interval_s=poll_interval_s)

    def score(self, trial_id: str) -> dict[str, object]:
        import modal

        from autoalphafold3.modal_app import APP_NAME

        scorer = modal.Cls.from_name(APP_NAME, "Scorer")()
        payload = scorer.score.remote(trial_id)
        if not isinstance(payload, dict):
            raise SamplerLoopError("Scorer.score returned a non-object payload")
        return payload


class DeterministicSamplerPlanner:
    """Reproducible fallback planner used for tests and no-key dry runs."""

    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        global_current_best: dict[str, object],
        search_reference: dict[str, object],
        strategy_context: dict[str, object],
    ) -> SamplerCandidatePlan:
        sampler_steps = _next_sampler_steps(candidate_index, prior_decisions)
        noise_scale, step_scale, schedule_shape, num_samples, selection_policy = _deterministic_sampler_knobs(
            candidate_index,
            prior_decisions,
        )
        return SamplerCandidatePlan(
            diagnostic_target=str(seed_trial.get("diagnostic_target") or "local_geometry_weak"),
            hypothesis=(
                "Incremental frozen-checkpoint sampler candidate selected after observing prior sampler scores; "
                "vary only bounded inference-time sampler schedule and target-blind selection settings."
            ),
            intervention=(
                f"Use sampler_steps={sampler_steps}, noise_scale={noise_scale}, "
                f"step_scale={step_scale}, schedule={schedule_shape}, samples={num_samples}, "
                f"selection={selection_policy} with deterministic seed {candidate_index}."
            ),
            predicted_direction="up",
            expected_lddt_delta_band=[0.001, 0.01],
            sampler_steps=sampler_steps,
            sampler_noise_scale=noise_scale,
            sampler_step_scale=step_scale,
            sampler_schedule_shape=schedule_shape,
            sampler_num_samples=num_samples,
            sampler_selection_policy=selection_policy,
            seed=candidate_index,
            rationale=(
                f"Global best is {global_current_best.get('score')}; "
                f"search reference is {search_reference.get('score')}; "
                f"prior decisions count is {len(prior_decisions)}; "
                f"strategy recommendation is {strategy_context.get('recommendation')}."
            ),
        )


class ReferenceSweepSamplerPlanner:
    """Deterministic T088-neighborhood sweep for bounded sampler follow-up."""

    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        global_current_best: dict[str, object],
        search_reference: dict[str, object],
        strategy_context: dict[str, object],
    ) -> SamplerCandidatePlan:
        sampler_steps, noise_scale, step_scale, schedule_shape, num_samples, selection_policy = (
            _reference_sweep_sampler_knobs(candidate_index)
        )
        reference_trial = search_reference.get("trial_id") or "unknown"
        reference_score = search_reference.get("score")
        return SamplerCandidatePlan(
            diagnostic_target=str(seed_trial.get("diagnostic_target") or "local_geometry_weak"),
            hypothesis=(
                "A T088-neighborhood frozen-checkpoint sampler sweep should test whether the known "
                "best sampler-family setting is locally improvable without changing training, scorer, "
                "labels, manifests, Modal resources, or templates."
            ),
            intervention=(
                f"Use sampler_steps={sampler_steps}, noise_scale={noise_scale}, "
                f"step_scale={step_scale}, schedule={schedule_shape}, samples={num_samples}, "
                f"selection={selection_policy}; this is a bounded neighborhood around the recorded "
                "T088 late-refine compact-geometry sampler."
            ),
            predicted_direction="up",
            expected_lddt_delta_band=[0.001, 0.02],
            sampler_steps=sampler_steps,
            sampler_noise_scale=noise_scale,
            sampler_step_scale=step_scale,
            sampler_schedule_shape=schedule_shape,
            sampler_num_samples=num_samples,
            sampler_selection_policy=selection_policy,
            seed=88000 + candidate_index,
            rationale=(
                f"Search reference {reference_trial} has score {reference_score}; "
                f"global best is {global_current_best.get('score')}; "
                f"prior decisions count is {len(prior_decisions)}. "
                f"Strategy recommendation is {strategy_context.get('recommendation')}. "
                "Run a deterministic local sweep before spending a full one-hour search window."
            ),
        )


class OpenAISamplerPlanner:
    """Structured-output OpenAI planner for one sampler candidate at a time."""

    def __init__(self, *, model: str = DEFAULT_LLM_MODEL) -> None:
        self.model = model
        self.policy = default_llm_phase_policy(AgentSearchPhase.HYPOTHESIS_GENERATION, model=model)

    def plan(
        self,
        *,
        seed_trial: dict[str, object],
        trial_id: str,
        candidate_index: int,
        prior_decisions: list[dict[str, object]],
        global_current_best: dict[str, object],
        search_reference: dict[str, object],
        strategy_context: dict[str, object],
    ) -> SamplerCandidatePlan:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise SamplerLoopError("LLM planner requires the openai package") from exc

        try:
            client = OpenAI()
        except Exception as exc:  # noqa: BLE001 - missing local key should use Modal harness secret.
            if _is_missing_openai_credentials(exc):
                return _plan_with_modal_harness_secret(
                    seed_trial=seed_trial,
                    trial_id=trial_id,
                    candidate_index=candidate_index,
                    prior_decisions=prior_decisions,
                    global_current_best=global_current_best,
                    search_reference=search_reference,
                    strategy_context=strategy_context,
                    model=self.model,
                )
            raise

        prompt = _planner_prompt(
            seed_trial=seed_trial,
            trial_id=trial_id,
            candidate_index=candidate_index,
            prior_decisions=prior_decisions,
            global_current_best=global_current_best,
            search_reference=search_reference,
            strategy_context=strategy_context,
        )
        kwargs = self.policy.to_responses_create_kwargs()
        try:
            response = client.responses.parse(
                **kwargs,
                input=[
                    {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text_format=SamplerCandidatePlan,
            )
        except TypeError:
            response = client.responses.parse(
                **kwargs,
                input=[
                    {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text={"format": SamplerCandidatePlan},
            )
        except Exception as exc:  # noqa: BLE001 - allow harness-secret fallback only for missing local credentials.
            if _is_missing_openai_credentials(exc):
                return _plan_with_modal_harness_secret(
                    seed_trial=seed_trial,
                    trial_id=trial_id,
                    candidate_index=candidate_index,
                    prior_decisions=prior_decisions,
                    global_current_best=global_current_best,
                    search_reference=search_reference,
                    strategy_context=strategy_context,
                    model=self.model,
                )
            raise
        return _extract_parsed_plan(response)


def _wait_modal_payload(call_id: str, *, timeout_s: int, poll_interval_s: float) -> dict[str, object]:
    import modal

    _, _, object_id = call_id.partition(":")
    if not object_id:
        raise SamplerLoopError(f"invalid Modal call id: {call_id}")
    call = modal.FunctionCall.from_id(object_id)
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            payload = call.get(timeout=0)
            if not isinstance(payload, dict):
                raise SamplerLoopError("Modal call returned a non-object payload")
            return payload
        except TimeoutError as exc:
            last_error = exc
            time.sleep(poll_interval_s)
        except Exception as exc:
            message = str(exc)
            if "Timeout" in type(exc).__name__ or "not ready" in message.lower():
                last_error = exc
                time.sleep(poll_interval_s)
                continue
            raise
    raise SamplerLoopError(f"Modal call timed out after {timeout_s}s: {last_error}")


def _candidate_trial(
    seed: dict[str, object],
    *,
    trial_id: str,
    plan: SamplerCandidatePlan,
) -> dict[str, object]:
    trial = copy.deepcopy(seed)
    trial["trial_id"] = trial_id
    trial["artifact_dir"] = f"runs/trials/{trial_id}"
    trial["agent_session_id"] = "autoaf3-incremental-sampler-loop"
    trial["diagnostic_target"] = plan.diagnostic_target
    trial["sampler_steps"] = plan.sampler_steps
    trial["sampler_noise_scale"] = plan.sampler_noise_scale
    trial["sampler_step_scale"] = plan.sampler_step_scale
    trial["sampler_schedule_shape"] = plan.sampler_schedule_shape
    trial["sampler_num_samples"] = plan.sampler_num_samples
    trial["sampler_selection_policy"] = plan.sampler_selection_policy
    trial["seed"] = plan.seed
    trial["hypothesis"] = plan.hypothesis
    trial["prediction"] = {
        "causal_component": "diffusion_sampler_step_scale",
        "predicted_axis": _predicted_axis(plan.diagnostic_target),
        "predicted_direction": plan.predicted_direction,
        "expected_lddt_delta_band": list(plan.expected_lddt_delta_band),
    }
    return trial


def _next_sampler_steps(candidate_index: int, prior_decisions: list[dict[str, object]]) -> int:
    if not prior_decisions:
        return 4
    scored = [row for row in prior_decisions if isinstance(row.get("score"), int | float)]
    if len(scored) >= 2 and float(scored[-1]["score"]) > float(scored[-2]["score"]):
        return min(12, int(scored[-1].get("sampler_steps", 4)) + 1)
    return [4, 6, 8, 5, 10, 3][candidate_index % 6]


def _deterministic_sampler_knobs(
    candidate_index: int,
    prior_decisions: list[dict[str, object]],
) -> tuple[float, float, str, int, str]:
    scored = [row for row in prior_decisions if isinstance(row.get("score"), int | float)]
    if len(scored) >= 2 and float(scored[-1]["score"]) > float(scored[-2]["score"]):
        return (
            float(scored[-1].get("sampler_noise_scale") or 1.0),
            min(2.0, float(scored[-1].get("sampler_step_scale") or 1.0) + 0.1),
            str(scored[-1].get("sampler_schedule_shape") or "linear"),
            min(4, int(scored[-1].get("sampler_num_samples") or 1) + 1),
            "geometry",
        )
    knob_grid = [
        (1.0, 1.0, "linear", 1, "first"),
        (0.8, 1.0, "cosine", 2, "geometry"),
        (1.2, 0.9, "late_refine", 2, "geometry"),
        (0.7, 1.1, "cosine", 3, "compact_geometry"),
        (1.4, 0.8, "late_refine", 3, "geometry"),
        (0.9, 1.2, "linear", 4, "compact_geometry"),
    ]
    return knob_grid[candidate_index % len(knob_grid)]


def _reference_sweep_sampler_knobs(candidate_index: int) -> tuple[int, float, float, str, int, str]:
    """Return bounded sampler settings near the recorded T088 best sampler candidate."""

    knob_grid = [
        (12, 0.6, 1.5, "late_refine", 4, "compact_geometry"),
        (12, 0.55, 1.6, "late_refine", 4, "compact_geometry"),
        (12, 0.65, 1.45, "late_refine", 4, "compact_geometry"),
        (11, 0.6, 1.5, "late_refine", 4, "compact_geometry"),
        (12, 0.5, 1.7, "late_refine", 4, "geometry"),
        (10, 0.6, 1.5, "late_refine", 4, "compact_geometry"),
        (12, 0.75, 1.35, "late_refine", 4, "compact_geometry"),
        (11, 0.55, 1.6, "late_refine", 4, "geometry"),
    ]
    return knob_grid[candidate_index % len(knob_grid)]


def _build_planner(
    planner: str,
    *,
    repo_root: Path,
    baseline_dir: str | Path,
    ledger_path: str | Path,
    model: str,
) -> SamplerPlanner:
    if planner == "deterministic":
        return DeterministicSamplerPlanner()
    if planner == "reference_sweep":
        return ReferenceSweepSamplerPlanner()
    if planner == "llm":
        return OpenAISamplerPlanner(model=model)
    raise SamplerLoopError(f"unsupported planner: {planner}")


def _current_best_context(*, root: Path, baseline_dir: str | Path, ledger_path: str | Path) -> dict[str, object]:
    try:
        best = current_best_from_baseline_and_ledger(
            baseline_dir=root / baseline_dir,
            ledger_path=root / ledger_path,
        )
    except Exception as exc:  # noqa: BLE001 - planner context should not hide readiness blockers.
        return {"status": "unavailable", "error": str(exc)}
    return {
        "status": "available",
        "trial_id": best.trial_id,
        "candidate_id": best.candidate_id,
        "score": best.score,
    }


def _search_reference_context(
    *,
    root: Path,
    ledger_path: str | Path,
    trial_id: str | None = None,
) -> dict[str, object]:
    ledger = root / ledger_path
    if not ledger.exists():
        return {
            "status": "unavailable",
            "reason": "ledger_missing",
            "interpretation": "same-family sampler comparison unavailable; global KEEP remains unchanged",
        }
    rows: list[AutoFoldResult] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = AutoFoldResult.model_validate(json.loads(line))
        except Exception:
            continue
        if row.status not in {TrialStatus.SCORED, TrialStatus.DISCARD, TrialStatus.KEEP}:
            continue
        score = _score(row)
        if score is None:
            continue
        if trial_id is not None and row.trial_id != trial_id:
            continue
        if trial_id is None and not str(row.candidate_id).endswith("_sampler"):
            continue
        rows.append(row)
        if trial_id is not None:
            break
    if not rows:
        return {
            "status": "unavailable",
            "trial_id": trial_id,
            "reason": "sampler_reference_not_found",
            "interpretation": "same-family sampler comparison unavailable; global KEEP remains unchanged",
        }
    reference = rows[0]
    return _search_reference_from_result(reference)


def _search_reference_from_result(reference: AutoFoldResult) -> dict[str, object]:
    return {
        "status": "available",
        "trial_id": reference.trial_id,
        "candidate_id": reference.candidate_id,
        "score": _score(reference),
        "fold_cartographer": _compact_fold_cartographer(reference.fold_cartographer),
        "interpretation": "same-family sampler reference; not a global discovery baseline",
    }


def _candidate_comparisons(
    *,
    score: float | None,
    global_current_best: dict[str, object],
    search_reference: dict[str, object],
    global_keep: bool,
) -> dict[str, object]:
    global_score = _maybe_float(global_current_best.get("score"))
    reference_score = _maybe_float(search_reference.get("score"))
    global_delta = None if score is None or global_score is None else score - global_score
    reference_delta = None if score is None or reference_score is None else score - reference_score
    beats_reference = reference_delta is not None and reference_delta > 0
    return {
        "global_delta": global_delta,
        "search_reference_delta": reference_delta,
        "beats_search_reference": beats_reference,
        "beats_global_current_best": global_keep,
        "sampler_search_status": "SAMPLER_IMPROVED" if beats_reference else "SAMPLER_NOT_IMPROVED",
        "global_current_best": global_current_best,
        "search_reference": search_reference,
        "keep_threshold_delta": DEFAULT_KEEP_DELTA,
    }


def _sampler_strategy_context(
    *,
    root: Path,
    ledger_path: str | Path,
    prior_decisions: list[dict[str, object]],
) -> dict[str, object]:
    ceiling = _sampler_family_ceiling(root=root, ledger_path=ledger_path)
    context: dict[str, object] = {
        "schema_version": "autoaf3.sampler_strategy_context.v1",
        "status": "active",
        "sampler_family_ceiling": ceiling,
        "regression_limit": SAMPLER_STRATEGY_REGRESSION_LIMIT,
        "recommendation": "continue_sampler_search",
        "blocked_neighborhood": None,
        "evidence": [],
    }
    ceiling_score = _maybe_float(ceiling.get("score"))
    ceiling_trial_id = ceiling.get("trial_id") if isinstance(ceiling.get("trial_id"), str) else None
    if ceiling_score is None or ceiling_trial_id is None:
        context["status"] = "insufficient_evidence"
        context["reason"] = "sampler_family_ceiling_unavailable"
        return context

    all_target_regressions = _all_target_regression_index(root=root, reference_trial_id=ceiling_trial_id)
    evidence: list[dict[str, object]] = []
    for row in prior_decisions:
        trial_id = row.get("trial_id")
        score = _maybe_float(row.get("score"))
        if not isinstance(trial_id, str) or score is None:
            continue
        delta = score - ceiling_score
        if delta >= 0 or not _is_t088_neighborhood_row(row):
            continue
        regression = all_target_regressions.get(trial_id)
        evidence.append(
            {
                "trial_id": trial_id,
                "score": score,
                "delta_vs_sampler_family_ceiling": delta,
                "sampler_steps": row.get("sampler_steps"),
                "sampler_noise_scale": row.get("sampler_noise_scale"),
                "sampler_step_scale": row.get("sampler_step_scale"),
                "sampler_schedule_shape": row.get("sampler_schedule_shape"),
                "sampler_num_samples": row.get("sampler_num_samples"),
                "sampler_selection_policy": row.get("sampler_selection_policy"),
                "all_target_regression_vs_ceiling": bool(regression),
                "scorer_sensitivity_report": regression.get("report_path") if isinstance(regression, dict) else None,
            }
        )

    context["evidence"] = evidence[-SAMPLER_STRATEGY_REGRESSION_LIMIT:]
    all_target_count = sum(1 for row in evidence if row.get("all_target_regression_vs_ceiling") is True)
    score_regression_count = len(evidence)
    context["score_regression_count"] = score_regression_count
    context["all_target_regression_count"] = all_target_count
    if all_target_count >= SAMPLER_STRATEGY_REGRESSION_LIMIT:
        context["recommendation"] = "stop_t088_neighborhood"
        context["blocked_neighborhood"] = "late_refine_compact_geometry_near_t088"
        context["reason"] = "repeated_all_target_regression_against_sampler_family_ceiling"
        context["required_pivot"] = (
            "Do not propose another late-refine compact/geometry T088-neighborhood sampler tweak. "
            "Choose a distinct sampler mechanism outside this neighborhood, or leave sampler-only "
            "search for a model-capacity/training-horizon diagnostic."
        )
    elif score_regression_count >= SAMPLER_STRATEGY_REGRESSION_LIMIT:
        context["recommendation"] = "avoid_t088_neighborhood"
        context["blocked_neighborhood"] = "late_refine_compact_geometry_near_t088"
        context["reason"] = "repeated_score_regression_against_sampler_family_ceiling"
        context["required_pivot"] = (
            "Avoid another local T088-neighborhood sampler tweak unless new scorer-sensitivity "
            "evidence contradicts the score regressions."
        )
    return context


def _sampler_family_ceiling(*, root: Path, ledger_path: str | Path) -> dict[str, object]:
    ledger = root / ledger_path
    if not ledger.exists():
        return {"status": "unavailable", "reason": "ledger_missing"}
    best: AutoFoldResult | None = None
    best_score: float | None = None
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = AutoFoldResult.model_validate(json.loads(line))
        except Exception:
            continue
        if not str(row.candidate_id).endswith("_sampler"):
            continue
        score = _score(row)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best = row
            best_score = score
    if best is None or best_score is None:
        return {"status": "unavailable", "reason": "sampler_rows_missing"}
    return {
        "status": "available",
        "trial_id": best.trial_id,
        "candidate_id": best.candidate_id,
        "score": best_score,
        "interpretation": "best sampler-family result in the local ledger; not a discovery baseline",
    }


def _all_target_regression_index(*, root: Path, reference_trial_id: str) -> dict[str, dict[str, object]]:
    reports_dir = root / "runs" / "autoresearch" / "scorer_sensitivity"
    if not reports_dir.exists():
        return {}
    index: dict[str, dict[str, object]] = {}
    for path in sorted(reports_dir.glob(f"{reference_trial_id}-vs-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("reference_trial_id") != reference_trial_id:
            continue
        deltas = payload.get("per_target_score_deltas_vs_reference")
        if not isinstance(deltas, dict):
            continue
        for trial_id, per_target in deltas.items():
            if not isinstance(trial_id, str) or not isinstance(per_target, dict) or not per_target:
                continue
            numeric = [float(value) for value in per_target.values() if isinstance(value, int | float)]
            if numeric and len(numeric) == len(per_target) and all(value < 0 for value in numeric):
                index[trial_id] = {
                    "report_path": str(path),
                    "target_count": len(numeric),
                    "min_delta": min(numeric),
                    "max_delta": max(numeric),
                }
    return index


def _is_t088_neighborhood_row(row: dict[str, object]) -> bool:
    steps = _maybe_float(row.get("sampler_steps"))
    noise = _maybe_float(row.get("sampler_noise_scale"))
    step_scale = _maybe_float(row.get("sampler_step_scale"))
    schedule = row.get("sampler_schedule_shape")
    selection = row.get("sampler_selection_policy")
    if steps is None or noise is None or step_scale is None:
        return False
    return (
        schedule == "late_refine"
        and selection in {"compact_geometry", "geometry"}
        and steps >= 8
        and noise <= 0.9
        and step_scale >= 1.2
    )


def _is_t088_neighborhood_plan(plan: SamplerCandidatePlan) -> bool:
    return (
        plan.sampler_schedule_shape == "late_refine"
        and plan.sampler_selection_policy in {"compact_geometry", "geometry"}
        and plan.sampler_steps >= 8
        and plan.sampler_noise_scale <= 0.9
        and plan.sampler_step_scale >= 1.2
    )


def _validate_strategy_gate(*, plan: SamplerCandidatePlan, strategy_context: dict[str, object]) -> None:
    if (
        strategy_context.get("recommendation") == "stop_t088_neighborhood"
        and _is_t088_neighborhood_plan(plan)
    ):
        ceiling = strategy_context.get("sampler_family_ceiling")
        ceiling_trial = ceiling.get("trial_id") if isinstance(ceiling, dict) else "UNKNOWN"
        raise SamplerLoopError(
            "sampler strategy gate blocks another late-refine compact/geometry T088-neighborhood "
            f"candidate after repeated all-target regressions against {ceiling_trial}"
        )


def _prior_decisions_from_ledger(
    *,
    root: Path,
    ledger_path: str | Path,
    trial_ids: list[str],
    global_current_best: dict[str, object],
    search_reference: dict[str, object],
) -> list[dict[str, object]]:
    if not trial_ids:
        return []
    wanted = [str(trial_id) for trial_id in trial_ids]
    ledger = root / ledger_path
    if not ledger.exists():
        raise SamplerLoopError(f"cannot continue from prior trials because ledger is missing: {ledger}")

    canonical = _canonical_sampler_smoke_index(root)
    latest: dict[str, AutoFoldResult] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = AutoFoldResult.model_validate(json.loads(line))
        except Exception:
            continue
        if row.trial_id not in wanted or row.status not in {TrialStatus.SCORED, TrialStatus.DISCARD, TrialStatus.KEEP}:
            continue
        if _score(row) is None:
            continue
        latest[row.trial_id] = row

    missing = [trial_id for trial_id in wanted if trial_id not in latest]
    if missing:
        raise SamplerLoopError(f"cannot continue from prior trials missing scored ledger rows: {', '.join(missing)}")

    decisions: list[dict[str, object]] = []
    for trial_id in wanted:
        row = latest[trial_id]
        score = _score(row)
        comparison = _candidate_comparisons(
            score=score,
            global_current_best=global_current_best,
            search_reference=search_reference,
            global_keep=row.status == TrialStatus.KEEP,
        )
        decision: dict[str, object] = {
            "trial_id": row.trial_id,
            "status": row.status.value,
            "score": score,
            "planner": "ledger_continuation",
            "num_failed_targets": row.metrics.get("num_failed_targets"),
            "fold_cartographer": _compact_fold_cartographer(row.fold_cartographer),
            "continuation_source": "ledger",
            **comparison,
        }
        if row.trial_id in canonical:
            decision.update(canonical[row.trial_id])
            decision["continuation_source"] = "ledger+canonical_sampler_smoke"
        decision.update(_trial_sampler_knobs(root=root, trial_id=row.trial_id))
        decisions.append(decision)
    return decisions


def _trial_sampler_knobs(*, root: Path, trial_id: str) -> dict[str, object]:
    path = root / "trials" / f"{trial_id}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    keys = {
        "sampler_steps",
        "sampler_noise_scale",
        "sampler_step_scale",
        "sampler_schedule_shape",
        "sampler_num_samples",
        "sampler_selection_policy",
        "seed",
    }
    return {key: payload[key] for key in keys if key in payload}


def _canonical_sampler_smoke_index(root: Path) -> dict[str, dict[str, object]]:
    path = root / "runs/canonical_sampler_smokes_2026-05-30.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("runs")
    if not isinstance(rows, list):
        return {}
    index: dict[str, dict[str, object]] = {}
    keys = {
        "hypothesis",
        "sampler_steps",
        "sampler_noise_scale",
        "sampler_step_scale",
        "sampler_schedule_shape",
        "sampler_num_samples",
        "sampler_selection_policy",
        "seed",
        "sampler_search_status",
        "worker_status",
        "global_delta",
        "search_reference_delta",
        "beats_search_reference",
        "beats_global_current_best",
    }
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("trial_id"), str):
            continue
        index[str(row["trial_id"])] = {key: row[key] for key in keys if key in row}
    return index


def _maybe_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _compact_fold_cartographer(report: FoldCartographerReport) -> dict[str, object]:
    summary = dict(report.summary)
    compact_summary = {
        key: summary[key]
        for key in (
            "canonical_target",
            "mean_target_calpha_lddt",
            "nan_prediction_residue_count",
            "num_scored_targets",
            "num_targets",
        )
        if key in summary
    }
    compact: dict[str, object] = {
        "signature": report.signature,
        "summary": compact_summary,
    }
    if "canonical_target" in compact_summary:
        compact["canonical_target"] = compact_summary["canonical_target"]
    if "mean_target_calpha_lddt" in compact_summary:
        compact["mean_target_calpha_lddt"] = compact_summary["mean_target_calpha_lddt"]
    bucket_summary = _compact_fold_cartographer_buckets(report.buckets)
    if bucket_summary:
        compact["buckets"] = bucket_summary
    return compact


def _compact_fold_cartographer_buckets(buckets: dict[str, object]) -> dict[str, object]:
    compact: dict[str, object] = {}
    for name, value in buckets.items():
        if not isinstance(value, dict):
            continue
        row: dict[str, object] = {}
        if "eligible_pair_count" in value:
            row["eligible_pair_count"] = value["eligible_pair_count"]
        target_ids = value.get("target_ids")
        if isinstance(target_ids, list):
            row["target_count"] = len(target_ids)
            row["target_ids_head"] = [str(target_id) for target_id in target_ids[:5]]
        if row:
            compact[str(name)] = row
    return compact


def _planner_prompt(
    *,
    seed_trial: dict[str, object],
    trial_id: str,
    candidate_index: int,
    prior_decisions: list[dict[str, object]],
    global_current_best: dict[str, object],
    search_reference: dict[str, object],
    strategy_context: dict[str, object] | None = None,
) -> str:
    payload = {
        "task": "Plan the next single sampler-only candidate. Do not plan a batch.",
        "trial_id": trial_id,
        "candidate_index": candidate_index,
        "global_current_best": global_current_best,
        "search_reference": search_reference,
        "strategy_context": strategy_context or {},
        "seed_trial": {
            "trial_kind": seed_trial.get("trial_kind"),
            "move_family": seed_trial.get("move_family"),
            "diagnostic_target": seed_trial.get("diagnostic_target"),
            "checkpoint_path": seed_trial.get("checkpoint_path"),
            "budget": seed_trial.get("budget"),
            "max_templates": 0,
        },
        "prior_decisions": prior_decisions[-20:],
        "allowed_knobs": {
            "sampler_steps": "integer 1..12",
            "seed": "integer >= 0",
            "sampler_noise_scale": "float in [0.25, 2.0]; scales diffusion schedule maximum noise",
            "sampler_step_scale": "float in [0.25, 2.0]; scales schedule point density",
            "sampler_schedule_shape": "one of linear, cosine, late_refine",
            "sampler_num_samples": "integer 1..4; repeated predictions per target",
            "sampler_selection_policy": "one of first, geometry, compact_geometry; label-free selection only",
        },
        "hard_constraints": [
            "sampler-only frozen checkpoint",
            "no training",
            "no patches",
            "no scorer/labels/manifests/fingerprints/baseline edits",
            "no Modal resource policy edits",
            "no Discovery Ledger writes",
            "official NanoFold runs stay max_templates=0",
        ],
    }
    return json.dumps(payload, allow_nan=False, sort_keys=True)


def _extract_parsed_plan(response: object) -> SamplerCandidatePlan:
    for output in getattr(response, "output", []) or []:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []) or []:
            if getattr(item, "type", None) == "refusal":
                raise SamplerLoopError(f"LLM planner refused: {getattr(item, 'refusal', '')}")
            parsed = getattr(item, "parsed", None)
            if parsed is not None:
                return SamplerCandidatePlan.model_validate(parsed)
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return SamplerCandidatePlan.model_validate(parsed)
    raise SamplerLoopError("LLM planner returned no parsed sampler plan")


def _is_missing_openai_credentials(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "missing credentials" in message or "api_key" in message and "environment variable" in message


def _plan_with_modal_harness_secret(
    *,
    seed_trial: dict[str, object],
    trial_id: str,
    candidate_index: int,
    prior_decisions: list[dict[str, object]],
    global_current_best: dict[str, object],
    search_reference: dict[str, object],
    strategy_context: dict[str, object],
    model: str,
) -> SamplerCandidatePlan:
    try:
        import modal
    except ModuleNotFoundError as exc:
        raise SamplerLoopError("local OpenAI credentials are missing and Modal SDK is unavailable") from exc

    from autoalphafold3.modal_app import APP_NAME, TRUSTED_ORCHESTRATOR_CLASS

    payload = {
        "seed_trial": seed_trial,
        "trial_id": trial_id,
        "candidate_index": candidate_index,
        "prior_decisions": prior_decisions[-20:],
        "global_current_best": global_current_best,
        "search_reference": search_reference,
        "strategy_context": strategy_context,
        "model": model,
    }
    orchestrator = modal.Cls.from_name(APP_NAME, TRUSTED_ORCHESTRATOR_CLASS)()
    result = orchestrator.plan_sampler_candidate.remote(payload)
    return SamplerCandidatePlan.model_validate(result)


def _predicted_axis(diagnostic_target: str) -> str:
    return {
        "local_geometry_weak": "local_geometry",
        "long_range_topology_weak": "long_range_topology",
        "distogram_good_lddt_flat": "distogram_vs_3d",
        "stability_compute": "stability_compute",
    }[diagnostic_target]


def _require_sampler_manifest(payload: dict[str, object], *, trial_id: str) -> None:
    if payload.get("schema_version") != "autoaf3.sampler_manifest.v1":
        raise SamplerLoopError("sampler worker did not return sampler manifest")
    if payload.get("trial_id") != trial_id:
        raise SamplerLoopError(f"sampler manifest trial_id mismatch: {payload.get('trial_id')} != {trial_id}")
    if payload.get("status") != "SAMPLER_PREDICTED":
        raise SamplerLoopError(f"sampler did not finish: {payload.get('status')}")
    if payload.get("real_training_performed") is not False or payload.get("inference_only") is not True:
        raise SamplerLoopError("sampler manifest must be inference-only with no training")
    if payload.get("writes_discovery_ledger") is not False:
        raise SamplerLoopError("sampler worker must not write Discovery Ledger")


def _score_payload_to_result(payload: dict[str, object]) -> AutoFoldResult:
    status = TrialStatus.SCORED if payload.get("status") == "SCORED" else TrialStatus.FAIL
    return AutoFoldResult(
        trial_id=str(payload.get("trial_id", "UNKNOWN")),
        status=status,
        candidate_id=str(payload.get("candidate_id", "sampler_score")),
        metrics=dict(payload.get("metrics") or {}),
        fold_cartographer=FoldCartographerReport.model_validate(payload.get("fold_cartographer") or {"signature": "missing"}),
        artifacts={key: str(value) for key, value in dict(payload.get("artifacts") or {}).items()},
        failure_signature=(
            str((payload.get("error_report") or {}).get("failure_signature"))
            if isinstance(payload.get("error_report"), dict) and (payload.get("error_report") or {}).get("failure_signature")
            else None
        ),
        postmortem="Locked scorer result recorded by incremental sampler loop.",
    )


def _worker_call_id(payload: dict[str, object]) -> str:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("worker_call_id"), str):
        raise SamplerLoopError("trusted orchestrator payload missing worker_call_id")
    return str(artifacts["worker_call_id"])


def _score(result: AutoFoldResult) -> float | None:
    value = result.metrics.get(PRIMARY_METRIC)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _next_trial_id(seed_trial_id: str) -> str:
    return f"T{_trial_number(seed_trial_id) + 1:03d}"


def _trial_number(trial_id: str) -> int:
    if not isinstance(trial_id, str) or not trial_id.startswith("T"):
        raise SamplerLoopError(f"invalid trial id: {trial_id}")
    return int(trial_id[1:])
