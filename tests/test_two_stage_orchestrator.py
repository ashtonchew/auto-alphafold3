from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalphafold3.discovery_ledger import DiscoveryLedgerError, build_discovery_record, read_discovery_ledger
from autoalphafold3.ledger import LEDGER_WRITER_ROLE, append_ledger, read_ledger
from autoalphafold3.orchestrator import decide_stage_one_result, record_stage_one_decision
from autoalphafold3.schema import (
    AutoFoldResult,
    DiscoveryStatus,
    FalsificationResult,
    FoldCartographerReport,
    TrialStatus,
)

SHA = "a" * 64


def write_baseline_lock(tmp_path: Path, *, score: float = 0.42) -> Path:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    metrics = {
        "schema_version": "autoaf3.metrics.v1",
        "status": "SCORED",
        "trial_id": "baseline_auto_tiny",
        "candidate_id": "baseline_lock",
        "split": "public_val_small",
        "official_benchmark_result": True,
        "primary_metric": "best_val_calpha_lddt",
        "scorer_version": "calpha_lddt_v1",
        "max_templates": 0,
        "manifests": {"train_tiny": SHA, "public_val_small": SHA},
        "label_hashes": {"public_val_small": SHA},
        "metrics": {"best_val_calpha_lddt": score},
        "fold_cartographer": {"signature": "baseline_locked", "summary": {}, "buckets": {}},
        "artifacts": {"metrics_json": "runs/baseline/metrics.json"},
    }
    (baseline / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (baseline / "error_report.json").write_text(json.dumps({"scorer_only": True}), encoding="utf-8")
    (baseline / "feature_fingerprints.json").write_text(
        json.dumps(
            {
                "files": {
                    "features/train_tiny.arrow": SHA,
                    "features/public_val_small.arrow": SHA,
                },
                "max_templates": 0,
            }
        ),
        encoding="utf-8",
    )
    return baseline


def scored_result(trial_id: str = "T400", score: float = 0.45) -> AutoFoldResult:
    return AutoFoldResult(
        trial_id=trial_id,
        status=TrialStatus.SCORED,
        candidate_id=f"{trial_id}_candidate",
        metrics={"best_val_calpha_lddt": score},
        fold_cartographer=FoldCartographerReport(signature="synthetic_scored"),
    )


def test_stage_one_keep_when_score_beats_ready_current_best(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    ledger = tmp_path / "ledger.jsonl"
    discovery = tmp_path / "discovery.jsonl"

    decision = record_stage_one_decision(
        scored_result(score=0.43),
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path=ledger.relative_to(tmp_path),
        keep_delta=0.001,
    )

    assert decision.status == TrialStatus.KEEP
    assert decision.discovery == DiscoveryStatus.UNCONFIRMED
    assert decision.falsification is None
    assert "Falsification Gate" in decision.postmortem
    assert read_ledger(ledger_path=ledger)[0].status == TrialStatus.KEEP
    assert read_discovery_ledger(ledger_path=discovery) == []


def test_stage_one_discard_when_score_does_not_beat_current_best(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)

    decision = decide_stage_one_result(
        scored_result(score=0.4205),
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path="ledger.jsonl",
        keep_delta=0.001,
    )

    assert decision.status == TrialStatus.DISCARD
    assert decision.discovery == DiscoveryStatus.UNCONFIRMED


def test_stage_one_fail_when_candidate_score_is_invalid(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    bad = scored_result()
    bad.metrics.clear()

    decision = decide_stage_one_result(
        bad,
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path="ledger.jsonl",
    )

    assert decision.status == TrialStatus.FAIL
    assert decision.failure_signature == "stage_one_score_missing"


@pytest.mark.parametrize("score", [-0.01, 1.01, True])
def test_stage_one_fail_when_candidate_score_is_invalid_value(tmp_path: Path, score: object) -> None:
    baseline = write_baseline_lock(tmp_path)

    decision = decide_stage_one_result(
        scored_result(score=score),
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path="ledger.jsonl",
    )

    assert decision.status == TrialStatus.FAIL
    assert decision.failure_signature == "stage_one_score_missing"


def test_stage_one_fail_when_result_is_not_scored(tmp_path: Path) -> None:
    result = scored_result(score=0.99).model_copy(update={"status": TrialStatus.PREFLIGHT_PASSED})

    decision = decide_stage_one_result(result, repo_root=tmp_path, baseline_dir="missing")

    assert decision.status == TrialStatus.FAIL
    assert decision.discovery == DiscoveryStatus.UNCONFIRMED
    assert decision.failure_signature == "stage_one_status_not_scored"


def test_stage_one_preserves_infra_fail_without_baseline_lookup(tmp_path: Path) -> None:
    result = AutoFoldResult(
        trial_id="T401",
        status=TrialStatus.INFRA_FAIL,
        candidate_id="modal_poll",
        metrics={},
        fold_cartographer=FoldCartographerReport(signature="modal_timeout"),
        failure_signature="modal_timeout",
    )

    decision = decide_stage_one_result(result, repo_root=tmp_path, baseline_dir="missing")

    assert decision.status == TrialStatus.INFRA_FAIL
    assert decision.failure_signature == "modal_timeout"


def test_stage_one_refuses_without_ready_baseline(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="baseline is not ready"):
        decide_stage_one_result(scored_result(), repo_root=tmp_path, baseline_dir="missing")


def test_provisional_keep_cannot_enter_discovery_ledger(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path, score=0.42)
    decision = decide_stage_one_result(
        scored_result(score=0.43),
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        ledger_path="ledger.jsonl",
    )

    with pytest.raises(DiscoveryLedgerError, match="discovery=CONFIRMED"):
        build_discovery_record(
            decision,
            mechanism="Synthetic provisional claim.",
            design_rule="Never record provisional keeps.",
            provenance=_provenance(),
        )


def test_confirmed_gate_result_builds_discovery_only_through_helper() -> None:
    result = scored_result()
    confirmed = result.model_copy(
        update={
            "status": TrialStatus.KEEP,
            "discovery": DiscoveryStatus.CONFIRMED,
            "falsification": _falsification("CONFIRMED"),
        }
    )

    record = build_discovery_record(
        confirmed,
        mechanism="Synthetic confirmed mechanism.",
        design_rule="Use helper after confirmed gate evidence.",
        provenance=_provenance(),
    )

    assert record.falsification.verdict == "CONFIRMED"


@pytest.mark.parametrize("verdict", ["PLACEBO_KILL", "KNOCKOUT_SURVIVES", "AXIS_MISS", "SEED_FRAGILE"])
def test_stage_two_killed_gate_is_not_discovery(verdict: str, tmp_path: Path) -> None:
    killed = scored_result().model_copy(
        update={
            "status": TrialStatus.KEEP,
            "discovery": DiscoveryStatus.KILLED,
            "falsification": _falsification(verdict),
        }
    )
    ledger = tmp_path / "ledger.jsonl"
    append_ledger(killed, ledger_path=ledger, writer_role=LEDGER_WRITER_ROLE)

    with pytest.raises(DiscoveryLedgerError):
        build_discovery_record(
            killed,
            mechanism="Synthetic killed mechanism.",
            design_rule="Killed gate outcomes are not discoveries.",
            provenance=_provenance(),
        )


def _falsification(verdict: str) -> FalsificationResult:
    return FalsificationResult(
        gain_full=0.03,
        gain_knockout=0.0 if verdict != "KNOCKOUT_SURVIVES" else 0.025,
        gain_placebo=0.0 if verdict != "PLACEBO_KILL" else 0.02,
        attributable_fraction=1.0 if verdict != "KNOCKOUT_SURVIVES" else 0.1,
        axis_delta_observed=0.02,
        axis_prediction_held=verdict != "AXIS_MISS",
        seed_mean=0.45,
        seed_std=0.001 if verdict != "SEED_FRAGILE" else 0.02,
        verdict=verdict,
    )


def _provenance() -> dict[str, object]:
    return {
        "git_sha": "abcdef123",
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "manifest_hashes": {"train_tiny": SHA, "public_val_small": SHA},
        "feature_fingerprints": {"train_tiny.arrow": SHA},
        "baseline_id": "baseline_auto_tiny",
        "current_best_trial_id": "baseline_auto_tiny",
        "causal_component": "geometry loss ramp",
        "predicted_axis": "local_geometry",
        "predicted_direction": "up",
        "verdict_numbers": {
            "gain_full": 0.03,
            "gain_knockout": 0.0,
            "gain_placebo": 0.0,
            "attributable_fraction": 1.0,
            "axis_delta_observed": 0.02,
            "seed_mean": 0.45,
            "seed_std": 0.001,
        },
        "gate_thresholds": {"tau_attribution": 0.5, "rho_placebo": 0.5, "k_seed": 2.0},
    }
