from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoalphafold3.discovery_ledger import (
    DiscoveryLedgerError,
    DISCOVERY_LEDGER_WRITER_ROLE,
    append_discovery_record,
    build_discovery_record,
    read_discovery_ledger,
)
from autoalphafold3.patch_policy import PatchPolicyError, validate_patch_scope
from autoalphafold3.schema import (
    AutoFoldResult,
    DiscoveryStatus,
    FalsificationResult,
    FoldCartographerReport,
    TrialStatus,
)

SHA = "a" * 64


def confirmed_result(
    *,
    verdict: str = "CONFIRMED",
    discovery: DiscoveryStatus = DiscoveryStatus.CONFIRMED,
) -> AutoFoldResult:
    return AutoFoldResult(
        trial_id="T300",
        status=TrialStatus.KEEP,
        candidate_id="candidate_confirmed",
        metrics={"best_val_calpha_lddt": 0.55},
        fold_cartographer=FoldCartographerReport(signature="synthetic_confirmed_contract"),
        discovery=discovery,
        falsification=FalsificationResult(
            gain_full=0.03,
            gain_knockout=0.0,
            gain_placebo=0.0,
            attributable_fraction=1.0,
            axis_delta_observed=0.02,
            axis_prediction_held=True,
            seed_mean=0.548,
            seed_std=0.001,
            verdict=verdict,
        ),
    )


def provisional_keep_result() -> AutoFoldResult:
    return AutoFoldResult(
        trial_id="T300",
        status=TrialStatus.KEEP,
        candidate_id="candidate_provisional",
        metrics={"best_val_calpha_lddt": 0.55},
        fold_cartographer=FoldCartographerReport(signature="synthetic_provisional_contract"),
        discovery=DiscoveryStatus.UNCONFIRMED,
        falsification=None,
    )


def provenance(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "git_sha": "abcdef123",
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "manifest_hashes": {"train_tiny": SHA, "public_val_small": SHA},
        "feature_fingerprints": {"train_tiny.arrow": SHA, "public_val_small.arrow": SHA},
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
            "seed_mean": 0.548,
            "seed_std": 0.001,
        },
        "gate_thresholds": {
            "tau_attribution": 0.5,
            "rho_placebo": 0.5,
            "k_seed": 2.0,
        },
    }
    data.update(overrides)
    return data


def discovery_record(**provenance_overrides: object):
    return build_discovery_record(
        confirmed_result(),
        mechanism="Geometry loss ramp improved local backbone preservation.",
        design_rule="Ramp local geometry loss only when local lDDT is weak.",
        provenance=provenance(**provenance_overrides),
    )


def test_discovery_ledger_appends_and_reads_confirmed_record(tmp_path: Path) -> None:
    ledger = tmp_path / "discovery.jsonl"
    record = discovery_record()

    append_discovery_record(record, ledger_path=ledger, writer_role=DISCOVERY_LEDGER_WRITER_ROLE)
    rows = read_discovery_ledger(ledger_path=ledger)

    assert len(rows) == 1
    assert rows[0].trial_id == "T300"
    assert rows[0].falsification.verdict == "CONFIRMED"
    assert rows[0].provenance.git_sha == "abcdef123"


@pytest.mark.parametrize("verdict", ["KNOCKOUT_SURVIVES", "PLACEBO_KILL", "AXIS_MISS", "SEED_FRAGILE"])
def test_discovery_ledger_rejects_non_confirmed_verdicts(verdict: str, tmp_path: Path) -> None:
    ledger = tmp_path / "discovery.jsonl"

    with pytest.raises((DiscoveryLedgerError, ValidationError), match="CONFIRMED"):
        build_discovery_record(
            confirmed_result(verdict=verdict, discovery=DiscoveryStatus.KILLED),
            mechanism="Synthetic killed claim.",
            design_rule="Do not record killed claims.",
            provenance=provenance(),
        )

    assert not ledger.exists()


def test_discovery_ledger_rejects_provisional_keep_without_confirmed_discovery() -> None:
    with pytest.raises(DiscoveryLedgerError, match="discovery=CONFIRMED"):
        build_discovery_record(
            provisional_keep_result(),
            mechanism="Synthetic provisional keep.",
            design_rule="Provisional keeps are not discoveries.",
            provenance=provenance(),
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "git_sha",
        "manifest_hashes",
        "feature_fingerprints",
        "baseline_id",
        "current_best_trial_id",
        "causal_component",
        "predicted_axis",
        "predicted_direction",
        "verdict_numbers",
        "gate_thresholds",
    ],
)
def test_discovery_ledger_requires_full_provenance(missing_key: str) -> None:
    payload = provenance()
    payload.pop(missing_key)

    with pytest.raises(ValidationError):
        build_discovery_record(
            confirmed_result(),
            mechanism="Synthetic confirmed claim.",
            design_rule="A reusable rule.",
            provenance=payload,
        )


def test_discovery_ledger_requires_complete_verdict_numbers() -> None:
    payload = provenance()
    verdict_numbers = dict(payload["verdict_numbers"])
    verdict_numbers.pop("seed_mean")
    payload["verdict_numbers"] = verdict_numbers

    with pytest.raises(DiscoveryLedgerError, match="verdict_numbers missing required keys"):
        build_discovery_record(
            confirmed_result(),
            mechanism="Synthetic confirmed claim.",
            design_rule="A reusable rule.",
            provenance=payload,
        )


def test_discovery_ledger_recomputes_confirmed_verdict_from_evidence(tmp_path: Path) -> None:
    ledger = tmp_path / "discovery.jsonl"
    record = discovery_record()
    payload = record.model_dump(mode="json")
    payload["falsification"]["gain_placebo"] = 0.03
    payload["provenance"]["verdict_numbers"]["gain_placebo"] = 0.03

    with pytest.raises(DiscoveryLedgerError, match="PLACEBO_KILL"):
        append_discovery_record(payload, ledger_path=ledger, writer_role=DISCOVERY_LEDGER_WRITER_ROLE)

    assert not ledger.exists()


def test_discovery_ledger_rejects_axis_mismatch(tmp_path: Path) -> None:
    ledger = tmp_path / "discovery.jsonl"
    record = discovery_record()
    payload = record.model_dump(mode="json")
    payload["axis_moved"] = "long_range_topology"

    with pytest.raises((DiscoveryLedgerError, ValidationError), match="axis_moved"):
        append_discovery_record(payload, ledger_path=ledger, writer_role=DISCOVERY_LEDGER_WRITER_ROLE)

    assert not ledger.exists()


def test_discovery_ledger_jsonl_is_stable(tmp_path: Path) -> None:
    ledger = tmp_path / "discovery.jsonl"
    record = discovery_record()

    append_discovery_record(record, ledger_path=ledger, writer_role=DISCOVERY_LEDGER_WRITER_ROLE)
    raw = ledger.read_text(encoding="utf-8")
    parsed = json.loads(raw)

    assert raw.endswith("\n")
    assert raw.count("\n") == 1
    assert parsed["schema_version"] == "autoaf3.discovery.v1"
    assert read_discovery_ledger(ledger_path=ledger)[0] == record


def test_discovery_ledger_duplicate_policy(tmp_path: Path) -> None:
    ledger = tmp_path / "discovery.jsonl"
    record = discovery_record()

    append_discovery_record(record, ledger_path=ledger, writer_role=DISCOVERY_LEDGER_WRITER_ROLE)
    append_discovery_record(record, ledger_path=ledger, writer_role=DISCOVERY_LEDGER_WRITER_ROLE)
    assert len(read_discovery_ledger(ledger_path=ledger)) == 1

    conflicting = record.model_copy(update={"design_rule": "Conflicting rule."})
    with pytest.raises(DiscoveryLedgerError, match="conflicting"):
        append_discovery_record(conflicting, ledger_path=ledger, writer_role=DISCOVERY_LEDGER_WRITER_ROLE)


def test_discovery_ledger_rejects_non_orchestrator_writer(tmp_path: Path) -> None:
    ledger = tmp_path / "discovery.jsonl"

    with pytest.raises(DiscoveryLedgerError, match="writer_role=orchestrator"):
        append_discovery_record(discovery_record(), ledger_path=ledger, writer_role="trial_worker")
    with pytest.raises(TypeError, match="writer_role"):
        append_discovery_record(discovery_record(), ledger_path=ledger)  # type: ignore[call-arg]

    assert not ledger.exists()


def test_patch_policy_denies_discovery_ledger_write_paths() -> None:
    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["autoalphafold3/discovery_ledger.py"])
    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["autoalphafold3/ledger.py"])
    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["runs/discovery_ledger.jsonl"])
    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["runs/discovery/T300.json"])
