"""Read-only scorer sensitivity diagnostics for existing trial artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol

from autoalphafold3.modal_app import APP_NAME
from autoalphafold3.runner import RunnerError, validate_trial_id
from autoalphafold3.schema import PRIMARY_METRIC

APPROVAL_TEXT = "I_APPROVE_SCORER_SENSITIVITY_DIAGNOSTIC"


class ScorerSensitivityError(RuntimeError):
    """Raised when scorer sensitivity diagnostics cannot run."""


class ScorerSensitivityClient(Protocol):
    """Small client boundary for deployed scorer-only workers."""

    def score_trial(self, trial_id: str) -> dict[str, object]:
        """Return the scorer payload for an existing trial artifact."""


@dataclass(frozen=True)
class ScoredTrialSummary:
    """Compact summary of one scorer-only payload."""

    trial_id: str
    status: str
    score: float | None
    metrics: dict[str, object]
    fold_cartographer_signature: str | None
    official_benchmark_result: bool
    local_only: bool
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScorerSensitivityReport:
    """JSON-friendly read-only scorer sensitivity report."""

    schema_version: str
    status: str
    mode: str
    starts_search: bool
    writes_ledger: bool
    writes_discovery_ledger: bool
    trial_ids: list[str]
    reference_trial_id: str | None
    scored_trials: list[ScoredTrialSummary]
    metric_deltas_vs_reference: dict[str, dict[str, float]]
    all_primary_scores_identical: bool | None
    pending_live_action: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scored_trials"] = [trial.to_dict() for trial in self.scored_trials]
        return payload


def run_scorer_sensitivity(
    *,
    trial_ids: list[str],
    mode: str = "dry-run",
    approval: str | None = None,
    modal_env: str | None = None,
    client: ScorerSensitivityClient | None = None,
) -> ScorerSensitivityReport:
    """Run a read-only scorer sensitivity diagnostic for existing artifacts."""

    checked_trial_ids = _validate_trial_ids(trial_ids)
    if mode == "dry-run":
        return ScorerSensitivityReport(
            schema_version="autoaf3.scorer_sensitivity.v1",
            status="DRY_RUN",
            mode=mode,
            starts_search=False,
            writes_ledger=False,
            writes_discovery_ledger=False,
            trial_ids=checked_trial_ids,
            reference_trial_id=checked_trial_ids[0] if checked_trial_ids else None,
            scored_trials=[],
            metric_deltas_vs_reference={},
            all_primary_scores_identical=None,
            pending_live_action=(
                "Approve read-only live scorer sensitivity diagnostic with "
                f"{APPROVAL_TEXT}; no search, trial execution, ledger, or Discovery Ledger writes."
            ),
        )
    if mode != "modal":
        raise ScorerSensitivityError(f"unsupported scorer sensitivity mode: {mode}")
    if approval != APPROVAL_TEXT:
        raise ScorerSensitivityError(f"live scorer sensitivity diagnostic requires approval token {APPROVAL_TEXT}")
    scorer_client = client or DeployedScorerSensitivityClient(environment_name=modal_env)
    scored = [_summarize_score_payload(trial_id, scorer_client.score_trial(trial_id)) for trial_id in checked_trial_ids]
    return ScorerSensitivityReport(
        schema_version="autoaf3.scorer_sensitivity.v1",
        status="PASS",
        mode=mode,
        starts_search=False,
        writes_ledger=False,
        writes_discovery_ledger=False,
        trial_ids=checked_trial_ids,
        reference_trial_id=scored[0].trial_id if scored else None,
        scored_trials=scored,
        metric_deltas_vs_reference=_metric_deltas_vs_reference(scored),
        all_primary_scores_identical=_all_primary_scores_identical(scored),
    )


class DeployedScorerSensitivityClient:
    """Modal SDK client for read-only deployed scorer diagnostics."""

    def __init__(self, *, environment_name: str | None = None) -> None:
        self.environment_name = environment_name
        try:
            import modal
        except ModuleNotFoundError as exc:
            raise ScorerSensitivityError("Modal SDK is required for live scorer sensitivity diagnostics") from exc
        self._modal = modal

    def score_trial(self, trial_id: str) -> dict[str, object]:
        scorer_cls = self._modal.Cls.from_name(APP_NAME, "Scorer", environment_name=self.environment_name)
        scorer = scorer_cls()
        payload = scorer.score.remote(trial_id)
        if not isinstance(payload, dict):
            raise ScorerSensitivityError("Scorer.score returned a non-object payload")
        return payload


def _validate_trial_ids(trial_ids: list[str]) -> list[str]:
    if not trial_ids:
        raise ScorerSensitivityError("at least one --trial-id is required")
    checked: list[str] = []
    for trial_id in trial_ids:
        try:
            checked.append(validate_trial_id(trial_id))
        except RunnerError as exc:
            raise ScorerSensitivityError(str(exc)) from exc
    duplicates = sorted({trial_id for trial_id in checked if checked.count(trial_id) > 1})
    if duplicates:
        raise ScorerSensitivityError(f"duplicate trial ids are not allowed: {', '.join(duplicates)}")
    return checked


def _summarize_score_payload(trial_id: str, payload: dict[str, object]) -> ScoredTrialSummary:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    fold_cartographer = (
        payload.get("fold_cartographer") if isinstance(payload.get("fold_cartographer"), dict) else {}
    )
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    return ScoredTrialSummary(
        trial_id=str(payload.get("trial_id") or trial_id),
        status=str(payload.get("status") or "UNKNOWN"),
        score=_score(metrics),
        metrics=dict(metrics),
        fold_cartographer_signature=_optional_str(fold_cartographer.get("signature")),
        official_benchmark_result=payload.get("official_benchmark_result") is True,
        local_only=payload.get("local_only") is True,
        artifacts={str(key): str(value) for key, value in artifacts.items()},
    )


def _metric_deltas_vs_reference(scored: list[ScoredTrialSummary]) -> dict[str, dict[str, float]]:
    if not scored:
        return {}
    reference = scored[0]
    deltas: dict[str, dict[str, float]] = {}
    for trial in scored[1:]:
        trial_deltas: dict[str, float] = {}
        for key in sorted(set(reference.metrics) & set(trial.metrics)):
            left = reference.metrics.get(key)
            right = trial.metrics.get(key)
            if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
                trial_deltas[key] = float(right) - float(left)
        deltas[trial.trial_id] = trial_deltas
    return deltas


def _all_primary_scores_identical(scored: list[ScoredTrialSummary]) -> bool | None:
    scores = [trial.score for trial in scored]
    if not scores or any(score is None for score in scores):
        return None
    reference = float(scores[0])
    return all(math.isclose(float(score), reference, rel_tol=0.0, abs_tol=0.0) for score in scores)


def _score(metrics: dict[str, object]) -> float | None:
    value = metrics.get(PRIMARY_METRIC)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        return None
    return score


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
