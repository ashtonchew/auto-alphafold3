"""Local non-GPU preflight gates for AutoFold trials."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from autoalphafold3.config_contract import validate_config_file
from autoalphafold3.nanofold_checks import NanoFoldGateResult, run_nanofold_preflight_gates
from autoalphafold3.patch_policy import validate_patch_scope
from autoalphafold3.schema import AutoFoldTrial, BudgetTier, RegisteredPrediction, TrialStatus
from autoalphafold3.scorer import SCORER_VERSION, run_scorer_dry_run
from autoalphafold3.scorer.locked_dataset import manifest_hashes

BUDGET_RESOURCE_MAP = {
    BudgetTier.DRY_RUN: {"gpu": "none", "max_containers": 0, "timeout_s": 60, "max_steps": 1},
    BudgetTier.SMOKE: {"gpu": "A100-80GB", "max_containers": 1, "timeout_s": 900, "max_steps": 10},
    BudgetTier.TRIAL: {"gpu": "A100-80GB", "max_containers": 6, "timeout_s": 2700, "max_steps": 250},
    BudgetTier.SAMPLER: {"gpu": "A100", "max_containers": 50, "timeout_s": 300, "max_steps": 0},
    BudgetTier.DEBUG: {"gpu": "H100", "max_containers": 1, "timeout_s": 3600, "max_steps": 0},
    BudgetTier.FINAL: {"gpu": "H100", "max_containers": 5, "timeout_s": 5400, "max_steps": 0},
}


@dataclass(frozen=True)
class PreflightResult:
    """Successful local preflight output."""

    status: TrialStatus
    trial: AutoFoldTrial
    scorer_metrics: dict[str, object]
    budget_resources: dict[str, object]
    nanofold_gates: list[NanoFoldGateResult]


class PreflightError(ValueError):
    """Raised when a preflight gate fails."""


def load_trial(trial_path: str | Path) -> AutoFoldTrial:
    """Load and validate an AutoFoldTrial JSON file."""

    return AutoFoldTrial.model_validate_json(Path(trial_path).read_text(encoding="utf-8"))


def run_preflight(
    trial: AutoFoldTrial | str | Path,
    *,
    repo_root: str | Path = ".",
    changed_paths: list[str] | None = None,
    manifest_paths: dict[str, str] | None = None,
    strict_nanofold_gates: bool = False,
    enforce_git_diff: bool = False,
) -> PreflightResult:
    """Run local preflight gates that do not require GPU, Modal, or large data."""

    root = Path(repo_root)
    trial_model = load_trial(trial) if isinstance(trial, (str, Path)) else trial

    _require_parent_commit(trial_model.parent_commit, repo_root=root)
    paths_to_check = changed_paths
    if paths_to_check is None and enforce_git_diff:
        paths_to_check = changed_paths_from_parent(trial_model.parent_commit, repo_root=root)
    validate_patch_scope(paths_to_check or [], repo_root=root, allow_empty=True)
    _require_config_json(trial_model.config_path, repo_root=root)
    _require_registered_prediction(trial_model.prediction)
    _require_manifest_hashes(trial_model, manifest_paths or {}, repo_root=root)
    _require_scorer_version(trial_model.scorer_version)
    _require_empty_artifact_dir(trial_model.artifact_dir, repo_root=root)
    budget_resources = _budget_resources(trial_model.budget)
    _require_trial_budget_consistency(trial_model, budget_resources)
    nanofold_gates = run_nanofold_preflight_gates(
        config_path=trial_model.config_path,
        repo_root=root,
    )
    if strict_nanofold_gates:
        _require_nanofold_gates_pass(nanofold_gates)
    scorer_metrics = run_scorer_dry_run(repo_root=root, trial_id=trial_model.trial_id)
    _require_canonical_metrics(scorer_metrics)

    return PreflightResult(
        status=TrialStatus.PREFLIGHT_PASSED,
        trial=trial_model,
        scorer_metrics=scorer_metrics,
        budget_resources=budget_resources,
        nanofold_gates=nanofold_gates,
    )


def _require_parent_commit(parent_commit: str, *, repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{parent_commit}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PreflightError(f"parent commit does not exist: {parent_commit}")


def changed_paths_from_parent(parent_commit: str, *, repo_root: str | Path = ".") -> list[str]:
    """Return paths changed since the declared parent commit."""

    result = subprocess.run(
        ["git", "diff", "--name-only", parent_commit, "--"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PreflightError(f"could not compute git diff from parent commit: {parent_commit}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _require_config_json(config_path: str, *, repo_root: Path) -> None:
    try:
        result = validate_config_file(config_path, repo_root=repo_root)
    except FileNotFoundError as exc:
        raise PreflightError(f"config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise PreflightError(f"config is not valid JSON: {config_path}") from exc
    if not result.valid:
        raise PreflightError(
            f"config is missing required {result.config_kind} keys: {', '.join(result.missing_keys)}"
        )


def _require_registered_prediction(prediction: RegisteredPrediction) -> None:
    if not prediction.causal_component.strip():
        raise PreflightError("prediction causal_component is required")


def _require_manifest_hashes(
    trial: AutoFoldTrial,
    manifest_paths_arg: dict[str, str],
    *,
    repo_root: Path,
) -> None:
    if not trial.manifest_hashes:
        return
    if not manifest_paths_arg:
        raise PreflightError("trial declares manifest_hashes but no manifest_paths were supplied")
    actual = manifest_hashes(manifest_paths_arg, repo_root=repo_root)
    for name, expected in trial.manifest_hashes.items():
        if actual.get(name) != expected:
            raise PreflightError(f"manifest hash mismatch for {name}: expected {expected}, got {actual.get(name)}")


def _require_scorer_version(version: str) -> None:
    if version != SCORER_VERSION:
        raise PreflightError(f"scorer version mismatch: expected {SCORER_VERSION}, got {version}")


def _require_empty_artifact_dir(artifact_dir: str | None, *, repo_root: Path) -> None:
    if artifact_dir is None:
        return
    path = _safe_repo_path(repo_root, artifact_dir)
    if path.exists() and any(path.iterdir()):
        raise PreflightError(f"artifact directory is not empty: {artifact_dir}")


def _budget_resources(budget: BudgetTier) -> dict[str, object]:
    try:
        return dict(BUDGET_RESOURCE_MAP[budget])
    except KeyError as exc:
        raise PreflightError(f"budget tier has no resource mapping: {budget}") from exc


def _require_trial_budget_consistency(trial: AutoFoldTrial, budget_resources: dict[str, object]) -> None:
    max_steps = budget_resources.get("max_steps")
    if isinstance(max_steps, int) and trial.max_steps is not None and trial.max_steps > max_steps:
        raise PreflightError(
            f"trial max_steps {trial.max_steps} exceeds {trial.budget.value} cap {max_steps}"
        )
    timeout_s = budget_resources.get("timeout_s")
    if isinstance(timeout_s, int) and trial.max_wall_minutes * 60 > timeout_s:
        raise PreflightError(
            f"trial max_wall_minutes {trial.max_wall_minutes} exceeds {trial.budget.value} timeout"
        )
    if trial.budget == BudgetTier.DRY_RUN and trial.cost_cap != 0.0:
        raise PreflightError("dry_run budget must use cost_cap=0.0")


def _require_canonical_metrics(metrics: dict[str, object]) -> None:
    for key in ("schema_version", "scorer_version", "primary_metric", "metrics", "fold_cartographer"):
        if key not in metrics:
            raise PreflightError(f"scorer dry-run missing canonical key: {key}")


def _require_nanofold_gates_pass(gates: list[NanoFoldGateResult]) -> None:
    incomplete = [gate for gate in gates if gate.status != "passed"]
    if incomplete:
        rendered = ", ".join(f"{gate.name}:{gate.status}:{gate.reason}" for gate in incomplete)
        raise PreflightError(f"NanoFold-dependent gates did not pass: {rendered}")


def _safe_repo_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PreflightError(f"unsafe repo-relative path: {path}")
    return root / candidate
