from __future__ import annotations

import pytest

from autoalphafold3.scorer_sensitivity import (
    APPROVAL_TEXT,
    ScorerSensitivityError,
    run_scorer_sensitivity,
)


def test_scorer_sensitivity_dry_run_requires_no_live_client() -> None:
    report = run_scorer_sensitivity(trial_ids=["T150", "T157"], mode="dry-run")

    assert report.status == "DRY_RUN"
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.writes_discovery_ledger is False
    assert report.reference_trial_id == "T150"
    assert report.scored_trials == []
    assert APPROVAL_TEXT in str(report.pending_live_action)


def test_scorer_sensitivity_modal_compares_metric_deltas() -> None:
    report = run_scorer_sensitivity(
        trial_ids=["T150", "T157"],
        mode="modal",
        approval=APPROVAL_TEXT,
        client=FakeScorerClient(
            {
                "T150": _score_payload("T150", score=0.1, mean=0.08),
                "T157": _score_payload("T157", score=0.12, mean=0.08),
            }
        ),
    )

    assert report.status == "PASS"
    assert report.mode == "modal"
    assert report.starts_search is False
    assert report.writes_ledger is False
    assert report.scored_trials[0].official_benchmark_result is True
    assert report.scored_trials[0].local_only is False
    assert report.scored_trials[0].fold_cartographer_signature == "toy_geometry_failed"
    assert report.metric_deltas_vs_reference["T157"] == {
        "best_val_calpha_lddt": pytest.approx(0.02),
        "mean_val_calpha_lddt": pytest.approx(0.0),
        "num_scored_targets": pytest.approx(0.0),
    }
    assert report.all_primary_scores_identical is False


def test_scorer_sensitivity_detects_identical_primary_scores() -> None:
    report = run_scorer_sensitivity(
        trial_ids=["T150", "T157"],
        mode="modal",
        approval=APPROVAL_TEXT,
        client=FakeScorerClient(
            {
                "T150": _score_payload("T150", score=0.008),
                "T157": _score_payload("T157", score=0.008),
            }
        ),
    )

    assert report.all_primary_scores_identical is True
    assert report.metric_deltas_vs_reference["T157"]["best_val_calpha_lddt"] == pytest.approx(0.0)


def test_scorer_sensitivity_rejects_live_without_exact_approval() -> None:
    with pytest.raises(ScorerSensitivityError, match=APPROVAL_TEXT):
        run_scorer_sensitivity(
            trial_ids=["T150"],
            mode="modal",
            approval=None,
            client=FakeScorerClient({"T150": _score_payload("T150", score=0.1)}),
        )


def test_scorer_sensitivity_rejects_duplicate_or_bad_trial_ids() -> None:
    with pytest.raises(ScorerSensitivityError, match="duplicate"):
        run_scorer_sensitivity(trial_ids=["T150", "T150"], mode="dry-run")
    with pytest.raises(ScorerSensitivityError, match="invalid trial_id"):
        run_scorer_sensitivity(trial_ids=["../bad"], mode="dry-run")


class FakeScorerClient:
    def __init__(self, payloads: dict[str, dict[str, object]]) -> None:
        self.payloads = payloads

    def score_trial(self, trial_id: str) -> dict[str, object]:
        return self.payloads[trial_id]


def _score_payload(trial_id: str, *, score: float, mean: float | None = None) -> dict[str, object]:
    return {
        "schema_version": "autoaf3.metrics.v1",
        "status": "SCORED",
        "trial_id": trial_id,
        "candidate_id": trial_id,
        "official_benchmark_result": True,
        "local_only": False,
        "metrics": {
            "best_val_calpha_lddt": score,
            "mean_val_calpha_lddt": score if mean is None else mean,
            "num_scored_targets": 16,
        },
        "fold_cartographer": {"signature": "toy_geometry_failed", "summary": {}, "buckets": {}},
        "artifacts": {
            "predictions_json": f"/mnt/autoalphafold3/runs/trials/{trial_id}/predictions.json",
            "manifest": "manifests/public_val_small.json",
        },
    }
