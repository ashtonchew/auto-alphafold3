"""Manual and deterministic autoresearch planning loop."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autoalphafold3.autoresearch_candidates import (
    create_candidate_envelope,
    create_run_manifest,
    write_candidate_decision,
    write_candidate_evidence,
)
from autoalphafold3.schema import (
    AutoFoldTrial,
    BudgetTier,
    DiagnosticTarget,
    FalsificationAxis,
    MoveFamily,
    PredictionDirection,
    RegisteredPrediction,
    TrialKind,
)

APPROVAL_TEXT = "I_APPROVE_AUTORESEARCH_LIVE_SEARCH"


class AutoresearchLoopError(RuntimeError):
    """Raised when autoresearch planning cannot proceed safely."""


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
) -> AutoresearchLoopResult:
    """Plan manual or deterministic autoresearch candidates without live execution."""

    if mode not in {"dry-run", "modal"}:
        raise AutoresearchLoopError(f"unsupported autoresearch mode: {mode}")
    if planner not in {"manual", "deterministic"}:
        raise AutoresearchLoopError(f"unsupported autoresearch planner for this PR: {planner}")
    if mode == "modal":
        if approval != APPROVAL_TEXT:
            raise AutoresearchLoopError(f"live autoresearch requires --approve {APPROVAL_TEXT}")
        raise AutoresearchLoopError("live autoresearch execution is not implemented; use dry-run planning mode")

    root = Path(repo_root)
    base_commit = _git_head(root)
    run_manifest = create_run_manifest(
        repo_root=root,
        run_id=run_id,
        base_commit=base_commit,
        planner=planner,
        mode=mode,
        description="Autoresearch dry-run planning artifacts.",
    )
    planned = (
        _manual_candidates(root=root, candidate_plan=candidate_plan)
        if planner == "manual"
        else _deterministic_candidates(start_trial_id=start_trial_id, max_candidates=max_candidates, base_commit=base_commit)
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
        AutoFoldTrial.model_validate(trial)
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
        wrote_files.extend(
            [
                str(envelope.manifest_path),
                str(envelope.hypothesis_path),
                str(envelope.patch_path),
                str(envelope.trial_path),
            ]
        )
        write_candidate_evidence(envelope, preflight=_planned_preflight(trial))
        decision = write_candidate_decision(
            envelope,
            status="PLANNED",
            matched_budget_delta=None,
            global_baseline_delta=None,
            reason="dry-run planning only; no training, scoring, ledger, or Discovery Ledger write",
            postmortem="Candidate was planned only. Execute later through approved bounded runner paths.",
        )
        decisions.append(decision)
        wrote_files.extend([str(envelope.preflight_path), str(envelope.decision_path), str(envelope.postmortem_path)])
    return AutoresearchLoopResult(
        status="PLANNED",
        mode=mode,
        planner=planner,
        run_id=str(run_manifest["run_id"]),
        run_dir=str(root / "runs" / "autoresearch" / run_id),
        generated_trials=generated_trials,
        candidate_dirs=candidate_dirs,
        decisions=decisions,
        wrote_files=wrote_files,
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        pending_live_action=f"modal mode requires --approve {APPROVAL_TEXT}",
    )


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
    path = Path(candidate_plan)
    if not path.is_absolute():
        path = root / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [payload])
    if not isinstance(candidates, list) or not candidates:
        raise AutoresearchLoopError("manual candidate plan must contain at least one candidate")
    checked = []
    for item in candidates:
        if not isinstance(item, dict) or not isinstance(item.get("trial"), dict):
            raise AutoresearchLoopError("manual candidate entries must contain a trial object")
        checked.append(
            {
                "hypothesis": item.get("hypothesis") or item["trial"].get("hypothesis"),
                "trial": item["trial"],
                "config": item.get("config"),
                "patch_text": item.get("patch_text", ""),
            }
        )
    return checked


def _trial_payload(
    *,
    trial_id: str,
    base_commit: str,
    kind: TrialKind,
    budget: BudgetTier,
    move_family: MoveFamily,
    max_steps: int | None,
    config_path: str,
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
        payload.update(
            {
                "checkpoint_path": "/mnt/autoalphafold3/runs/trials/T123/checkpoint.pt",
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
        "status": "PLANNED",
        "mode": "dry-run",
        "max_templates": 0,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
    }


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
