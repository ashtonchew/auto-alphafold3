"""Pure local Falsification Gate verdict logic."""

from __future__ import annotations

import math

from autoalphafold3.schema import (
    FalsificationResult,
    FalsificationVerdict,
    PredictionDirection,
    RegisteredPrediction,
)


class FalsificationError(ValueError):
    """Raised when gate evidence is incomplete or invalid."""


def attributable_fraction(*, gain_full: float, gain_knockout: float) -> float:
    """Return the share of full gain erased by the knock-out control."""

    _require_finite("gain_full", gain_full)
    _require_finite("gain_knockout", gain_knockout)
    if gain_full <= 0.0:
        raise FalsificationError("gain_full must be positive to compute attribution")
    return max(0.0, (gain_full - gain_knockout) / gain_full)


def axis_prediction_held(
    *,
    axis_delta_observed: float,
    predicted_direction: PredictionDirection,
) -> bool:
    """Return whether the observed diagnostic-axis delta moved as predicted."""

    _require_finite("axis_delta_observed", axis_delta_observed)
    if predicted_direction == PredictionDirection.UP:
        return axis_delta_observed > 0.0
    if predicted_direction == PredictionDirection.DOWN:
        return axis_delta_observed < 0.0
    raise FalsificationError(f"unsupported prediction direction: {predicted_direction}")


def decide_falsification_verdict(
    *,
    gain_full: float,
    gain_knockout: float,
    gain_placebo: float,
    axis_prediction_held: bool,
    seed_std: float,
    tau_attribution: float = 0.5,
    rho_placebo: float = 0.5,
    k_seed: float = 2.0,
) -> FalsificationVerdict:
    """Apply the canonical five-verdict rule to scored gate evidence."""

    for name, value in (
        ("gain_full", gain_full),
        ("gain_knockout", gain_knockout),
        ("gain_placebo", gain_placebo),
        ("seed_std", seed_std),
        ("tau_attribution", tau_attribution),
        ("rho_placebo", rho_placebo),
        ("k_seed", k_seed),
    ):
        _require_finite(name, value)
    if gain_full <= 0.0:
        raise FalsificationError("gain_full must be positive for a gate verdict")
    if seed_std < 0.0:
        raise FalsificationError("seed_std must be non-negative")
    if tau_attribution <= 0.0 or tau_attribution > 1.0:
        raise FalsificationError("tau_attribution must be in (0, 1]")
    if rho_placebo <= 0.0 or rho_placebo > 1.0:
        raise FalsificationError("rho_placebo must be in (0, 1]")
    if k_seed <= 0.0:
        raise FalsificationError("k_seed must be positive")

    if gain_full <= k_seed * seed_std:
        return FalsificationVerdict.SEED_FRAGILE
    if gain_placebo >= rho_placebo * gain_full:
        return FalsificationVerdict.PLACEBO_KILL
    if attributable_fraction(gain_full=gain_full, gain_knockout=gain_knockout) < tau_attribution:
        return FalsificationVerdict.KNOCKOUT_SURVIVES
    if not axis_prediction_held:
        return FalsificationVerdict.AXIS_MISS
    return FalsificationVerdict.CONFIRMED


def build_falsification_result(
    *,
    parent_lddt: float,
    candidate_lddt: float,
    knockout_lddt: float,
    placebo_lddt: float,
    prediction: RegisteredPrediction,
    axis_delta_observed: float,
    seed_mean: float,
    seed_std: float,
    tau_attribution: float = 0.5,
    rho_placebo: float = 0.5,
    k_seed: float = 2.0,
) -> FalsificationResult:
    """Build a serializable falsification result from complete scored controls."""

    for name, value in (
        ("parent_lddt", parent_lddt),
        ("candidate_lddt", candidate_lddt),
        ("knockout_lddt", knockout_lddt),
        ("placebo_lddt", placebo_lddt),
        ("axis_delta_observed", axis_delta_observed),
        ("seed_mean", seed_mean),
        ("seed_std", seed_std),
    ):
        _require_finite(name, value)

    gain_full = candidate_lddt - parent_lddt
    gain_knockout = knockout_lddt - parent_lddt
    gain_placebo = placebo_lddt - parent_lddt
    attribution = attributable_fraction(gain_full=gain_full, gain_knockout=gain_knockout)
    axis_held = axis_prediction_held(
        axis_delta_observed=axis_delta_observed,
        predicted_direction=prediction.predicted_direction,
    )
    verdict = decide_falsification_verdict(
        gain_full=gain_full,
        gain_knockout=gain_knockout,
        gain_placebo=gain_placebo,
        axis_prediction_held=axis_held,
        seed_std=seed_std,
        tau_attribution=tau_attribution,
        rho_placebo=rho_placebo,
        k_seed=k_seed,
    )
    return FalsificationResult(
        gain_full=gain_full,
        gain_knockout=gain_knockout,
        gain_placebo=gain_placebo,
        attributable_fraction=attribution,
        axis_delta_observed=axis_delta_observed,
        axis_prediction_held=axis_held,
        seed_mean=seed_mean,
        seed_std=seed_std,
        verdict=verdict,
    )


def _require_finite(name: str, value: float | None) -> None:
    if value is None or not math.isfinite(value):
        raise FalsificationError(f"{name} must be finite")
