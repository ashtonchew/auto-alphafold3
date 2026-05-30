from __future__ import annotations

import pytest

from autoalphafold3.falsification import (
    FalsificationError,
    axis_prediction_held,
    build_falsification_result,
    decide_falsification_verdict,
)
from autoalphafold3.schema import (
    FalsificationPlan,
    FalsificationVerdict,
    PredictionDirection,
    RegisteredPrediction,
)


def prediction(direction: str = "up") -> RegisteredPrediction:
    return RegisteredPrediction.model_validate(
        {
            "causal_component": "geometry loss ramp",
            "predicted_axis": "local_geometry",
            "predicted_direction": direction,
            "expected_lddt_delta_band": [0.01, 0.05],
        }
    )


def test_falsification_plan_has_orchestrator_owned_defaults() -> None:
    plan = FalsificationPlan(
        candidate_trial_id="T123",
        knockout_patch="runs/trials/T123/falsification/knockout.patch",
        placebo_family="optimizer_scheduler",
    )

    assert plan.n_seeds == 3
    assert plan.authored_by == "orchestrator"
    assert plan.tau_attribution == 0.5
    assert plan.rho_placebo == 0.5
    assert plan.k_seed == 2.0
    with pytest.raises(ValueError, match="authored_by"):
        FalsificationPlan(
            candidate_trial_id="T123",
            authored_by="agent",
            knockout_patch="runs/trials/T123/falsification/knockout.patch",
            placebo_family="optimizer_scheduler",
        )


def test_axis_prediction_direction_checks_up_and_down() -> None:
    assert axis_prediction_held(axis_delta_observed=0.02, predicted_direction=PredictionDirection.UP)
    assert axis_prediction_held(axis_delta_observed=-0.02, predicted_direction=PredictionDirection.DOWN)
    assert not axis_prediction_held(axis_delta_observed=-0.02, predicted_direction=PredictionDirection.UP)
    assert not axis_prediction_held(axis_delta_observed=0.02, predicted_direction=PredictionDirection.DOWN)


def test_confirmed_verdict_when_all_gate_conditions_hold() -> None:
    verdict = decide_falsification_verdict(
        gain_full=0.04,
        gain_knockout=0.01,
        gain_placebo=0.005,
        axis_prediction_held=True,
        seed_std=0.005,
    )

    assert verdict == FalsificationVerdict.CONFIRMED


def test_seed_fragile_takes_precedence() -> None:
    verdict = decide_falsification_verdict(
        gain_full=0.01,
        gain_knockout=0.0,
        gain_placebo=0.0,
        axis_prediction_held=True,
        seed_std=0.005,
    )

    assert verdict == FalsificationVerdict.SEED_FRAGILE


def test_placebo_kill_when_placebo_reproduces_gain() -> None:
    verdict = decide_falsification_verdict(
        gain_full=0.04,
        gain_knockout=0.0,
        gain_placebo=0.02,
        axis_prediction_held=True,
        seed_std=0.001,
    )

    assert verdict == FalsificationVerdict.PLACEBO_KILL


def test_knockout_survives_when_attribution_is_too_low() -> None:
    verdict = decide_falsification_verdict(
        gain_full=0.04,
        gain_knockout=0.03,
        gain_placebo=0.0,
        axis_prediction_held=True,
        seed_std=0.001,
    )

    assert verdict == FalsificationVerdict.KNOCKOUT_SURVIVES


def test_axis_miss_when_registered_axis_does_not_hold() -> None:
    verdict = decide_falsification_verdict(
        gain_full=0.04,
        gain_knockout=0.0,
        gain_placebo=0.0,
        axis_prediction_held=False,
        seed_std=0.001,
    )

    assert verdict == FalsificationVerdict.AXIS_MISS


def test_build_falsification_result_roundtrips_schema() -> None:
    result = build_falsification_result(
        parent_lddt=0.50,
        candidate_lddt=0.54,
        knockout_lddt=0.51,
        placebo_lddt=0.505,
        prediction=prediction(),
        axis_delta_observed=0.03,
        seed_mean=0.538,
        seed_std=0.004,
    )

    payload = result.model_dump(mode="json")
    assert payload["verdict"] == "CONFIRMED"
    assert result.attributable_fraction == pytest.approx(0.75)


def test_verdict_rejects_missing_or_nonfinite_controls() -> None:
    with pytest.raises(FalsificationError, match="gain_full"):
        decide_falsification_verdict(
            gain_full=float("nan"),
            gain_knockout=0.0,
            gain_placebo=0.0,
            axis_prediction_held=True,
            seed_std=0.0,
        )
    with pytest.raises(FalsificationError, match="gain_full must be positive"):
        decide_falsification_verdict(
            gain_full=0.0,
            gain_knockout=0.0,
            gain_placebo=0.0,
            axis_prediction_held=True,
            seed_std=0.0,
        )
    with pytest.raises(FalsificationError, match="seed_std"):
        build_falsification_result(
            parent_lddt=0.50,
            candidate_lddt=0.54,
            knockout_lddt=0.51,
            placebo_lddt=0.505,
            prediction=prediction(),
            axis_delta_observed=0.03,
            seed_mean=0.538,
            seed_std=float("nan"),
        )
