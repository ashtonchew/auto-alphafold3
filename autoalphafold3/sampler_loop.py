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
from autoalphafold3.orchestrator import decide_stage_one_result, submit_trial
from autoalphafold3.schema import AutoFoldResult, AutoFoldTrial, FoldCartographerReport, PRIMARY_METRIC, TrialStatus

APPROVAL_TEXT = "I_APPROVE_AUTONOMOUS_SAMPLER_LOOP"

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
        current_best: dict[str, object],
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
    if planner not in {"deterministic", "llm"}:
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
    current_best = _current_best_context(root=root, baseline_dir=baseline_dir, ledger_path=ledger_path)

    for index in range(max_candidates):
        trial_id = f"T{next_id + index:03d}"
        try:
            plan = active_planner.plan(
                seed_trial=seed,
                trial_id=trial_id,
                candidate_index=index,
                prior_decisions=decisions,
                current_best=current_best,
            )
        except Exception as exc:  # noqa: BLE001 - invalid autonomous plans must stop before submit.
            raise SamplerLoopError(f"planner failed for {trial_id}: {exc}") from exc
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
                }
            )
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_trial_id = trial_id
            current_best = _current_best_context(root=root, baseline_dir=baseline_dir, ledger_path=ledger_path)
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
        current_best: dict[str, object],
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
            rationale=f"Current best is {current_best.get('score')}; prior decisions count is {len(prior_decisions)}.",
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
        current_best: dict[str, object],
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
                    current_best=current_best,
                    model=self.model,
                )
            raise

        prompt = _planner_prompt(
            seed_trial=seed_trial,
            trial_id=trial_id,
            candidate_index=candidate_index,
            prior_decisions=prior_decisions,
            current_best=current_best,
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
                    current_best=current_best,
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


def _planner_prompt(
    *,
    seed_trial: dict[str, object],
    trial_id: str,
    candidate_index: int,
    prior_decisions: list[dict[str, object]],
    current_best: dict[str, object],
) -> str:
    payload = {
        "task": "Plan the next single sampler-only candidate. Do not plan a batch.",
        "trial_id": trial_id,
        "candidate_index": candidate_index,
        "current_best": current_best,
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
    current_best: dict[str, object],
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
        "current_best": current_best,
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
