from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

from autoalphafold3.gate_wave import (
    DEFAULT_GATE_CLASS,
    DEFAULT_GATE_METHOD,
    GateControl,
    GateControlKind,
    GateWaveError,
    build_gate_wave_controls,
    require_scored_gate_wave,
    run_gate_wave,
    run_gate_wave_with_timeout,
    run_modal_gate_wave,
)
from autoalphafold3.patch_policy import PatchPolicyError, validate_patch_scope
from autoalphafold3.schema import FalsificationPlan, TrialStatus

REPO_ROOT = Path(__file__).resolve().parents[1]


def plan(n_seeds: int = 3) -> FalsificationPlan:
    return FalsificationPlan(
        candidate_trial_id="T900",
        knockout_patch="runs/trials/T900/falsification/knockout.patch",
        placebo_family="optimizer_scheduler",
        n_seeds=n_seeds,
    )


def base_payload() -> dict[str, object]:
    return {
        "trial_id": "T900",
        "config_path": "configs/auto_tiny.json",
        "max_templates": 0,
    }


def test_gate_wave_builds_orchestrator_owned_bounded_controls() -> None:
    controls = build_gate_wave_controls(plan=plan(n_seeds=2), base_payload=base_payload(), timeout_seconds=123)

    assert [control.control_kind for control in controls] == [
        GateControlKind.KNOCKOUT,
        GateControlKind.PLACEBO,
        GateControlKind.AXIS_CHECK,
        GateControlKind.SEED_RERUN,
        GateControlKind.SEED_RERUN,
    ]
    assert {control.authored_by for control in controls} == {"orchestrator"}
    assert all(control.payload["max_templates"] == 0 for control in controls)
    assert all(control.timeout_seconds == 123 for control in controls)
    assert all(control.payload["timeout_seconds"] == 123 for control in controls)


def test_gate_wave_rejects_unbounded_seed_count_before_modal_submission() -> None:
    with pytest.raises(GateWaveError, match="seed count"):
        build_gate_wave_controls(plan=plan(n_seeds=6), base_payload=base_payload())
    with pytest.raises(GateWaveError, match="max"):
        build_gate_wave_controls(plan=plan(n_seeds=5), base_payload=base_payload(), max_variants=4)


def test_gate_wave_rejects_locked_label_payloads() -> None:
    payload = base_payload()
    payload["label_path"] = "autoalphafold3-locked/public_val_labels.json"

    with pytest.raises(ValueError, match="locked labels"):
        build_gate_wave_controls(plan=plan(), base_payload=payload)


def test_gate_wave_rejects_locked_label_payload_keys() -> None:
    payload = base_payload()
    payload["public_val_labels"] = "indirect-reference"

    with pytest.raises(ValueError, match="locked labels"):
        build_gate_wave_controls(plan=plan(), base_payload=payload)


def test_gate_wave_rejects_payload_metadata_mismatch() -> None:
    payload = {
        **base_payload(),
        "candidate_trial_id": "T901",
        "control_kind": "knockout",
        "seed": 0,
    }

    with pytest.raises(ValueError, match="candidate_trial_id"):
        GateControl(
            gate_id="T900:knockout:0",
            candidate_trial_id="T900",
            control_kind="knockout",
            seed=0,
            payload=payload,
        )


def test_gate_wave_modal_adapter_uses_required_starmap_exception_contract() -> None:
    class FakeFunction:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def starmap(
            self,
            inputs: list[tuple[dict[str, object], int]],
            *,
            kwargs: dict[str, object],
            order_outputs: bool,
            return_exceptions: bool,
            wrap_returned_exceptions: bool | None,
        ) -> list[dict[str, object]]:
            self.calls.append(
                {
                    "inputs": inputs,
                    "kwargs": kwargs,
                    "order_outputs": order_outputs,
                    "return_exceptions": return_exceptions,
                    "wrap_returned_exceptions": wrap_returned_exceptions,
                }
            )
            return [
                {
                    "status": "SCORED",
                    "metrics": {"best_val_calpha_lddt": 0.5},
                    "fold_cartographer": {"signature": "gate_control_scored", "summary": {}, "buckets": {}},
                }
                for _payload, _seed in inputs
            ]

    function = FakeFunction()
    controls = build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload())

    report = run_gate_wave(function, controls)

    assert report.status == TrialStatus.SCORED
    assert function.calls[0]["kwargs"] == {"aggregate_timeout_seconds": 2400}
    assert function.calls[0]["order_outputs"] is True
    assert function.calls[0]["return_exceptions"] is True
    assert function.calls[0]["wrap_returned_exceptions"] is False
    assert function.calls[0]["inputs"][0][0]["control_kind"] == "knockout"
    assert require_scored_gate_wave(report) == report


def test_gate_wave_returned_exception_normalizes_to_control_infra_fail() -> None:
    class FakeFunction:
        def starmap(self, inputs: list[tuple[dict[str, object], int]], **kwargs: object) -> list[object]:
            return [
                {
                    "status": "SCORED",
                    "metrics": {"best_val_calpha_lddt": 0.5},
                    "fold_cartographer": {"signature": "gate_control_scored", "summary": {}, "buckets": {}},
                },
                RuntimeError("worker boom"),
                *[
                    {
                        "status": "SCORED",
                        "metrics": {"best_val_calpha_lddt": 0.5},
                        "fold_cartographer": {"signature": "gate_control_scored", "summary": {}, "buckets": {}},
                    }
                    for _payload, _seed in inputs[2:]
                ],
            ]

    report = run_gate_wave(FakeFunction(), build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload()))

    assert report.status == TrialStatus.INFRA_FAIL
    assert report.controls[1].status == TrialStatus.INFRA_FAIL
    assert report.controls[1].failure_signature == "modal_RuntimeError"
    with pytest.raises(GateWaveError, match="SCORED"):
        require_scored_gate_wave(report)


def test_gate_wave_requires_positive_aggregate_timeout_before_modal_submission() -> None:
    class FakeFunction:
        called = False

        def starmap(self, inputs: list[tuple[dict[str, object], int]], **kwargs: object) -> list[object]:
            self.called = True
            return []

    function = FakeFunction()

    with pytest.raises(GateWaveError, match="aggregate_timeout_seconds"):
        run_gate_wave_with_timeout(
            function,
            build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload()),
            aggregate_timeout_seconds=0,
        )

    assert function.called is False


def test_gate_wave_rejects_requested_timeout_over_aggregate_before_modal_submission() -> None:
    class FakeFunction:
        called = False

        def starmap(self, inputs: list[tuple[dict[str, object], int]], **kwargs: object) -> list[object]:
            self.called = True
            return []

    function = FakeFunction()
    controls = build_gate_wave_controls(
        plan=plan(n_seeds=1),
        base_payload=base_payload(),
        timeout_seconds=600,
    )

    with pytest.raises(GateWaveError, match="requested timeout"):
        run_gate_wave_with_timeout(function, controls, aggregate_timeout_seconds=599)

    assert function.called is False


def test_gate_wave_cancel_failure_normalizes_to_infra_fail() -> None:
    class ModalCancelled(BaseException):
        pass

    class FakeFunction:
        def starmap(self, inputs: list[tuple[dict[str, object], int]], **kwargs: object) -> list[object]:
            raise ModalCancelled("cancelled by remote platform")

    report = run_gate_wave(FakeFunction(), build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload()))

    assert report.status == TrialStatus.INFRA_FAIL
    assert {row.failure_signature for row in report.controls} == {"modal_starmap_ModalCancelled"}


def test_gate_wave_lookup_failure_normalizes_to_infra_fail() -> None:
    class FakeClsNamespace:
        @staticmethod
        def from_name(app_name: str, class_name: str) -> object:
            assert app_name == "autoalphafold3-modal"
            assert class_name == DEFAULT_GATE_CLASS
            raise RuntimeError("lookup unavailable")

    fake_modal = types.SimpleNamespace(Cls=FakeClsNamespace)

    report = run_modal_gate_wave(
        build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload()),
        modal_module=fake_modal,
    )

    assert report.status == TrialStatus.INFRA_FAIL
    assert {row.failure_signature for row in report.controls} == {"modal_lookup_RuntimeError"}


def test_gate_wave_modal_adapter_looks_up_trial_runner_method() -> None:
    class FakeRunMethod:
        def __init__(self) -> None:
            self.calls: list[list[tuple[dict[str, object], int]]] = []

        def starmap(
            self,
            inputs: list[tuple[dict[str, object], int]],
            *,
            kwargs: dict[str, object],
            order_outputs: bool,
            return_exceptions: bool,
            wrap_returned_exceptions: bool | None,
        ) -> list[dict[str, object]]:
            self.calls.append(inputs)
            assert kwargs == {"aggregate_timeout_seconds": 2400}
            assert order_outputs is True
            assert return_exceptions is True
            assert wrap_returned_exceptions is False
            return [
                {
                    "status": "SCORED",
                    "metrics": {"best_val_calpha_lddt": 0.5},
                    "fold_cartographer": {"signature": "gate_control_scored", "summary": {}, "buckets": {}},
                }
                for _payload, _seed in inputs
            ]

    fake_run = FakeRunMethod()

    class FakeRunner:
        run = fake_run

    class FakeClsNamespace:
        @staticmethod
        def from_name(app_name: str, class_name: str) -> type[FakeRunner]:
            assert app_name == "autoalphafold3-modal"
            assert class_name == DEFAULT_GATE_CLASS
            return FakeRunner

    fake_modal = types.SimpleNamespace(Cls=FakeClsNamespace)

    report = run_modal_gate_wave(
        build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload()),
        modal_module=fake_modal,
        method_name=DEFAULT_GATE_METHOD,
    )

    assert report.status == TrialStatus.SCORED
    assert fake_run.calls[0][0][0]["control_kind"] == "knockout"


def test_gate_wave_missing_sdk_normalizes_to_infra_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "modal":
            raise ModuleNotFoundError("No module named 'modal'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    report = run_modal_gate_wave(build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload()))

    assert report.status == TrialStatus.INFRA_FAIL
    assert {row.failure_signature for row in report.controls} == {"modal_sdk_missing"}


def test_gate_wave_starmap_failure_normalizes_without_writing_artifacts(tmp_path: Path) -> None:
    class FakeFunction:
        def starmap(self, inputs: list[tuple[dict[str, object], int]], **kwargs: object) -> list[object]:
            raise TimeoutError("gate wave timed out")

    report = run_gate_wave(FakeFunction(), build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload()))

    assert report.status == TrialStatus.INFRA_FAIL
    assert {row.failure_signature for row in report.controls} == {"modal_starmap_TimeoutError"}
    assert not (tmp_path / "baseline").exists()
    assert not (tmp_path / "discovery_ledger.jsonl").exists()


def test_gate_wave_requires_all_control_kinds() -> None:
    controls = build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload())
    missing_seed = [control for control in controls if control.control_kind != GateControlKind.SEED_RERUN]

    with pytest.raises(GateWaveError, match="seed_rerun"):
        run_gate_wave(object(), missing_seed)  # type: ignore[arg-type]


def test_gate_wave_scored_validator_rejects_missing_evidence() -> None:
    controls = build_gate_wave_controls(plan=plan(n_seeds=1), base_payload=base_payload())

    class FakeFunction:
        def starmap(self, inputs: list[tuple[dict[str, object], int]], **kwargs: object) -> list[dict[str, object]]:
            return [
                {
                    "status": "SCORED",
                    "metrics": {"best_val_calpha_lddt": 0.5},
                    "fold_cartographer": {"signature": "gate_control_scored", "summary": {}, "buckets": {}},
                }
                for _payload, _seed in inputs
            ]

    report = run_gate_wave(FakeFunction(), controls)
    incomplete = report.model_copy(update={"controls": report.controls[:-1]})

    with pytest.raises(GateWaveError, match="seed_rerun"):
        require_scored_gate_wave(incomplete)


def test_gate_wave_control_schema_rejects_agent_authoring() -> None:
    with pytest.raises(ValueError, match="authored_by"):
        GateControl(
            gate_id="T900:knockout:0",
            candidate_trial_id="T900",
            authored_by="agent",
            control_kind="knockout",
            seed=0,
            payload=base_payload(),
        )


def test_patch_policy_denies_gate_wave_control_paths() -> None:
    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["autoalphafold3/gate_wave.py"], repo_root=REPO_ROOT)
    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["runs/trials/T900/falsification/gate_wave.json"], repo_root=REPO_ROOT)
    with pytest.raises(PatchPolicyError, match="locked"):
        validate_patch_scope(["runs/gate_wave/T900.json"], repo_root=REPO_ROOT)


def test_patch_policy_denies_readiness_locked_artifact_paths() -> None:
    locked_paths = [
        "runs/ledger.jsonl",
        "runs/baseline/metrics.json",
        "runs/baseline/error_report.json",
        "runs/baseline/feature_fingerprints.json",
        "runs/benchmark/artifact_manifest.json",
        "runs/discovery_ledger.jsonl",
        "runs/discovery/T300.json",
    ]

    for path in locked_paths:
        with pytest.raises(PatchPolicyError, match="locked"):
            validate_patch_scope([path], repo_root=REPO_ROOT)
