"""Human-approved baseline source artifact runner.

This module coordinates an already-deployed Modal baseline run and scorer pass.
It writes trial-scoped source artifacts for ``lock-baseline`` only after the
scorer returns official, non-local evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from autoalphafold3.baseline_lock import DEFAULT_ERROR_REPORT, DEFAULT_METRICS
from autoalphafold3.modal_app import APP_NAME, validate_execution_payload
from autoalphafold3.runner import validate_trial_id
from autoalphafold3.schema import PRIMARY_METRIC, SCORER_VERSION

APPROVAL_TEXT = "I_APPROVE_BASELINE_RUN"
DEFAULT_BASELINE_TRIAL_ID = "T000"
DEFAULT_BASELINE_CANDIDATE_ID = "baseline_auto_tiny"


class BaselineRunError(RuntimeError):
    """Raised when the baseline runner refuses unsafe or incomplete evidence."""


class ModalBaselineClient(Protocol):
    """Small protocol for Modal-backed baseline execution."""

    def run_trial(self, trial_payload: dict[str, object]) -> dict[str, object]:
        """Run the baseline trial artifact worker."""

    def score_trial(self, trial_id: str) -> dict[str, object]:
        """Run the scorer-only worker for one trial."""


@dataclass(frozen=True)
class BaselineRunResult:
    """JSON-friendly baseline runner result."""

    status: str
    mode: str
    trial_id: str
    source_dir: str
    wrote_files: list[str]
    plan: dict[str, object]
    metrics: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "trial_id": self.trial_id,
            "source_dir": self.source_dir,
            "wrote_files": self.wrote_files,
            "plan": self.plan,
            "metrics": self.metrics,
        }


def run_baseline(
    *,
    repo_root: str | Path = ".",
    trial_id: str = DEFAULT_BASELINE_TRIAL_ID,
    source_dir: str | Path = "runs/trials/T000",
    approval: str | None = None,
    mode: str = "dry-run",
    modal_env: str | None = None,
    modal_client: ModalBaselineClient | None = None,
) -> BaselineRunResult:
    """Plan or execute the human-approved baseline source artifact run."""

    checked_trial_id = validate_trial_id(trial_id)
    root = Path(repo_root)
    source = root / source_dir
    _require_trial_source_dir(source, checked_trial_id)
    plan = baseline_run_plan(trial_id=checked_trial_id, source_dir=source)
    if mode == "dry-run":
        return BaselineRunResult(
            status="PLANNED",
            mode=mode,
            trial_id=checked_trial_id,
            source_dir=str(source),
            wrote_files=[],
            plan=plan,
        )
    if mode != "modal":
        raise BaselineRunError(f"unsupported baseline run mode: {mode}")
    if approval != APPROVAL_TEXT:
        raise BaselineRunError(f"baseline run requires --approve {APPROVAL_TEXT}")

    client = modal_client if modal_client is not None else DeployedModalBaselineClient(environment_name=modal_env)
    trial_payload = baseline_trial_payload(trial_id=checked_trial_id)
    run_payload = client.run_trial(trial_payload)
    _require_trial_worker_completed(run_payload)
    scored = client.score_trial(checked_trial_id)
    metrics, error_report = _require_lockable_baseline_score(scored)

    _require_empty_source_outputs(source)
    source.mkdir(parents=True, exist_ok=True)
    metrics_path = source / DEFAULT_METRICS
    error_report_path = source / DEFAULT_ERROR_REPORT
    _atomic_write_json(metrics_path, metrics)
    _atomic_write_json(error_report_path, error_report)
    return BaselineRunResult(
        status="PASS",
        mode=mode,
        trial_id=checked_trial_id,
        source_dir=str(source),
        wrote_files=[str(metrics_path), str(error_report_path)],
        plan=plan,
        metrics=metrics,
    )


def baseline_run_plan(*, trial_id: str = DEFAULT_BASELINE_TRIAL_ID, source_dir: str | Path = "runs/trials/T000") -> dict[str, object]:
    """Return the immutable baseline-run intent without touching Modal."""

    checked_trial_id = validate_trial_id(trial_id)
    return {
        "trial_id": checked_trial_id,
        "candidate_id": DEFAULT_BASELINE_CANDIDATE_ID,
        "source_dir": str(source_dir),
        "writes_baseline_dir": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "requires_approval": APPROVAL_TEXT,
        "requires_modal_deployment": APP_NAME,
        "trial_worker": "TrialRunner.run",
        "scorer_worker": "Scorer.score",
        "required_score": {
            "status": "SCORED",
            "official_benchmark_result": True,
            "local_only": False,
            "primary_metric": PRIMARY_METRIC,
            "scorer_version": SCORER_VERSION,
            "split": "public_val_small",
            "max_templates": 0,
        },
        "expected_lock_step": (
            "python3 -m autoalphafold3.agent lock-baseline --source-dir "
            f"{source_dir} --feature-fingerprints <approved feature_fingerprints.json> "
            "--approve I_APPROVE_BASELINE_LOCK --dry-run"
        ),
    }


def baseline_trial_payload(*, trial_id: str = DEFAULT_BASELINE_TRIAL_ID) -> dict[str, object]:
    """Return the baseline payload sent to the Modal trial worker."""

    checked_trial_id = validate_trial_id(trial_id)
    payload: dict[str, object] = {
        "trial_id": checked_trial_id,
        "candidate_id": DEFAULT_BASELINE_CANDIDATE_ID,
        "trial_kind": "training",
        "budget_tier": "trial",
        "config_path": "configs/nanofold_dev_cpu_smoke.json",
        "split": "public_val_small",
        "seed": 0,
        "max_templates": 0,
        "baseline": True,
        "description": "Human-approved NanoFold-style AlphaFold3-lite baseline run.",
    }
    validate_execution_payload(payload, role="trial")
    return payload


class DeployedModalBaselineClient:
    """Modal SDK client for the deployed baseline trial and scorer classes."""

    def __init__(self, *, environment_name: str | None = None) -> None:
        self.environment_name = environment_name
        try:
            import modal
        except ModuleNotFoundError as exc:
            raise BaselineRunError("Modal SDK is required for --mode modal baseline runs") from exc
        self._modal = modal

    def run_trial(self, trial_payload: dict[str, object]) -> dict[str, object]:
        runner_cls = self._modal.Cls.from_name(APP_NAME, "TrialRunner", environment_name=self.environment_name)
        runner = runner_cls()
        payload = runner.run.remote(trial_payload)
        if not isinstance(payload, dict):
            raise BaselineRunError("TrialRunner.run returned a non-object payload")
        return payload

    def score_trial(self, trial_id: str) -> dict[str, object]:
        scorer_cls = self._modal.Cls.from_name(APP_NAME, "Scorer", environment_name=self.environment_name)
        scorer = scorer_cls()
        payload = scorer.score.remote(trial_id)
        if not isinstance(payload, dict):
            raise BaselineRunError("Scorer.score returned a non-object payload")
        return payload


def _require_trial_worker_completed(payload: dict[str, object]) -> None:
    status = payload.get("status")
    if status in {"INFRA_FAIL", "FAIL"}:
        raise BaselineRunError(f"baseline trial worker failed: {payload.get('reason') or status}")


def _require_lockable_baseline_score(scored: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    if scored.get("status") != "SCORED":
        raise BaselineRunError(f"baseline scorer did not return SCORED: {scored.get('status')}")
    if scored.get("official_benchmark_result") is not True:
        raise BaselineRunError("baseline scorer result is not official_benchmark_result=true")
    if scored.get("local_only") is True:
        raise BaselineRunError("baseline scorer result must not be local_only")
    if scored.get("primary_metric") != PRIMARY_METRIC:
        raise BaselineRunError(f"baseline scorer primary_metric must be {PRIMARY_METRIC}")
    if scored.get("scorer_version") != SCORER_VERSION:
        raise BaselineRunError(f"baseline scorer_version must be {SCORER_VERSION}")
    if scored.get("split") != "public_val_small":
        raise BaselineRunError("baseline scorer split must be public_val_small")
    if scored.get("max_templates") != 0:
        raise BaselineRunError("baseline scorer result must record max_templates=0")
    error_report = scored.get("error_report")
    if not isinstance(error_report, dict) or error_report.get("scorer_only") is not True:
        raise BaselineRunError("baseline scorer result must include scorer_only error_report")
    metrics = {key: value for key, value in scored.items() if key != "error_report"}
    metrics["candidate_id"] = str(metrics.get("candidate_id") or DEFAULT_BASELINE_CANDIDATE_ID)
    return metrics, dict(error_report)


def _require_trial_source_dir(source: Path, trial_id: str) -> None:
    if source.name != trial_id:
        raise BaselineRunError(f"source_dir must be trial-scoped and end with {trial_id}")
    if "baseline" in source.parts:
        raise BaselineRunError("run-baseline must not write runs/baseline directly")


def _require_empty_source_outputs(source: Path) -> None:
    existing = [name for name in (DEFAULT_METRICS, DEFAULT_ERROR_REPORT) if (source / name).exists()]
    if existing:
        raise BaselineRunError(f"baseline source output already exists: {', '.join(existing)}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
