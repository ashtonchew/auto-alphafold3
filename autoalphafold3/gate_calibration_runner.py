"""Approved Falsification Gate calibration evidence runner."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from autoalphafold3.baseline_readiness import audit_baseline_readiness
from autoalphafold3.falsification import decide_falsification_verdict
from autoalphafold3.gate_calibration import DEFAULT_CALIBRATION_PATH
from autoalphafold3.gate_wave import (
    GateControlEvidence,
    GateControlKind,
    build_gate_wave_controls,
    require_scored_gate_wave,
    run_modal_gate_wave,
)
from autoalphafold3.schema import (
    FalsificationPlan,
    FalsificationVerdict,
    MoveFamily,
    PRIMARY_METRIC,
    SCORER_VERSION,
)

APPROVAL_TEXT = "I_APPROVE_GATE_CALIBRATION_RUN"
DEFAULT_EVIDENCE_DIR = Path("runs/gate_calibration")
PUBLIC_VAL_SPLIT = "public_val_small"


class GateCalibrationRunError(RuntimeError):
    """Raised when calibration evidence cannot be produced honestly."""


@dataclass(frozen=True)
class GateCalibrationRunResult:
    """JSON-friendly result from producing calibration evidence."""

    status: str
    mode: str
    evidence_dir: str
    wrote_files: list[str]
    plan: dict[str, object]
    known_null: dict[str, object] | None = None
    known_positive: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "evidence_dir": self.evidence_dir,
            "wrote_files": self.wrote_files,
            "plan": self.plan,
            "known_null": self.known_null,
            "known_positive": self.known_positive,
        }


def run_gate_calibration(
    *,
    repo_root: str | Path = ".",
    evidence_dir: str | Path = DEFAULT_EVIDENCE_DIR,
    baseline_dir: str | Path = "runs/baseline",
    approval: str | None = None,
    mode: str = "dry-run",
    modal_env: str | None = None,
) -> GateCalibrationRunResult:
    """Plan or produce known-null and known-positive calibration evidence."""

    root = Path(repo_root)
    output_dir = root / evidence_dir
    plan = gate_calibration_run_plan(evidence_dir=evidence_dir, baseline_dir=baseline_dir)
    if mode == "dry-run":
        return GateCalibrationRunResult(
            status="PLANNED",
            mode=mode,
            evidence_dir=str(output_dir),
            wrote_files=[],
            plan=plan,
        )
    if mode != "modal":
        raise GateCalibrationRunError(f"unsupported calibration run mode: {mode}")
    if approval != APPROVAL_TEXT:
        raise GateCalibrationRunError(f"gate calibration run requires --approve {APPROVAL_TEXT}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GateCalibrationRunError(f"calibration evidence output already exists and is not empty: {output_dir}")

    baseline = audit_baseline_readiness(baseline_dir=root / baseline_dir)
    if baseline.status != "PASS":
        raise GateCalibrationRunError(f"baseline must be locked before gate calibration: {baseline.problems}")
    baseline_score = baseline.baseline_score
    if baseline_score is None or not math.isfinite(baseline_score):
        raise GateCalibrationRunError("locked baseline score is missing or non-finite")
    baseline_metrics = _read_json(root / baseline_dir / "metrics.json")
    feature_fingerprints = _read_json(root / baseline_dir / "feature_fingerprints.json")

    known_null = _run_one_case(
        case="known_null",
        candidate_trial_id="T900",
        baseline_score=baseline_score,
        baseline_metrics=baseline_metrics,
        feature_fingerprints=feature_fingerprints,
        modal_env=modal_env,
    )
    known_positive = _run_one_case(
        case="known_positive",
        candidate_trial_id="T901",
        baseline_score=baseline_score,
        baseline_metrics=baseline_metrics,
        feature_fingerprints=feature_fingerprints,
        modal_env=modal_env,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    null_path = output_dir / "known_null.json"
    positive_path = output_dir / "known_positive.json"
    _atomic_write_json(null_path, known_null)
    _atomic_write_json(positive_path, known_positive)
    return GateCalibrationRunResult(
        status="PASS",
        mode=mode,
        evidence_dir=str(output_dir),
        wrote_files=[str(null_path), str(positive_path)],
        plan=plan,
        known_null=known_null,
        known_positive=known_positive,
    )


def gate_calibration_run_plan(
    *,
    evidence_dir: str | Path = DEFAULT_EVIDENCE_DIR,
    baseline_dir: str | Path = "runs/baseline",
) -> dict[str, object]:
    """Return the approved calibration evidence production plan."""

    return {
        "evidence_dir": str(evidence_dir),
        "baseline_dir": str(baseline_dir),
        "requires_approval": APPROVAL_TEXT,
        "required_followup": (
            "python3 -m autoalphafold3.agent calibrate-gate --mode from-evidence "
            "--known-null-evidence runs/gate_calibration/known_null.json "
            "--known-positive-evidence runs/gate_calibration/known_positive.json "
            "--approve I_APPROVE_GATE_CALIBRATION"
        ),
        "writes_calibration_file": False,
        "calibration_file": str(DEFAULT_CALIBRATION_PATH),
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "starts_search": False,
        "cases": {
            "known_null": "calibration-only no-op/trivial control expected to be killed",
            "known_positive": "calibration-only strong positive control expected to be CONFIRMED",
        },
    }


def _run_one_case(
    *,
    case: str,
    candidate_trial_id: str,
    baseline_score: float,
    baseline_metrics: dict[str, object],
    feature_fingerprints: dict[str, object],
    modal_env: str | None,
) -> dict[str, object]:
    thresholds = {"tau_attribution": 0.5, "rho_placebo": 0.5, "k_seed": 2.0}
    plan = FalsificationPlan(
        candidate_trial_id=candidate_trial_id,
        knockout_patch=f"runs/gate_calibration/{case}/knockout.patch",
        placebo_family=MoveFamily.OPTIMIZER_SCHEDULER,
        n_seeds=3,
        **thresholds,
    )
    controls = build_gate_wave_controls(
        plan=plan,
        base_payload={
            "trial_id": candidate_trial_id,
            "config_path": "configs/nanofold_dev_cpu_smoke.json",
            "max_templates": 0,
            "calibration_control": True,
            "calibration_case": case,
            "baseline_score": baseline_score,
            "positive_gain": 0.1,
            "null_gain": 0.0001,
        },
    )
    report = require_scored_gate_wave(
        run_modal_gate_wave(controls, modal_module=None, environment_name=modal_env),
        expected_controls=controls,
    )
    by_kind = _controls_by_kind(report.controls)
    seed_scores = [_metric(row) for row in by_kind[GateControlKind.SEED_RERUN]]
    candidate_score = mean(seed_scores)
    seed_std = pstdev(seed_scores) if len(seed_scores) > 1 else 0.0
    knockout_score = _metric(by_kind[GateControlKind.KNOCKOUT][0])
    placebo_score = _metric(by_kind[GateControlKind.PLACEBO][0])
    axis_score = _metric(by_kind[GateControlKind.AXIS_CHECK][0])
    axis_delta = axis_score - baseline_score
    verdict = decide_falsification_verdict(
        gain_full=candidate_score - baseline_score,
        gain_knockout=knockout_score - baseline_score,
        gain_placebo=placebo_score - baseline_score,
        axis_prediction_held=axis_delta > 0.0,
        seed_std=seed_std,
        **thresholds,
    ).value
    if case == "known_positive" and verdict != FalsificationVerdict.CONFIRMED.value:
        raise GateCalibrationRunError(f"known_positive calibration did not confirm: {verdict}")
    if case == "known_null" and verdict == FalsificationVerdict.CONFIRMED.value:
        raise GateCalibrationRunError("known_null calibration unexpectedly confirmed")
    return {
        "status": "complete",
        "verdict": verdict,
        "scorer_version": SCORER_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "split": PUBLIC_VAL_SPLIT,
        "baseline_id": str(baseline_metrics.get("candidate_id", "baseline_auto_tiny")),
        "current_best_trial_id": str(baseline_metrics.get("trial_id", "T000")),
        "manifest_hashes": dict(baseline_metrics.get("manifests", {})),
        "feature_fingerprints": _feature_fingerprints_payload(feature_fingerprints),
        "gate_thresholds": thresholds,
        "control_evidence_ids": [row.gate_id for row in report.controls],
        "calibration_only": True,
        "starts_search": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "verdict_numbers": {
            "parent_lddt": baseline_score,
            "candidate_lddt": candidate_score,
            "knockout_lddt": knockout_score,
            "placebo_lddt": placebo_score,
            "axis_delta_observed": axis_delta,
            "seed_std": seed_std,
        },
    }


def _controls_by_kind(controls: list[GateControlEvidence]) -> dict[GateControlKind, list[GateControlEvidence]]:
    by_kind = {kind: [] for kind in GateControlKind}
    for row in controls:
        by_kind[row.control_kind].append(row)
    return by_kind


def _metric(row: GateControlEvidence) -> float:
    value = row.metrics.get(PRIMARY_METRIC)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise GateCalibrationRunError(f"gate control {row.gate_id} missing finite {PRIMARY_METRIC}")
    return float(value)


def _feature_fingerprints_payload(payload: dict[str, object]) -> dict[str, object]:
    files = payload.get("files")
    if isinstance(files, dict):
        return files
    data_files = payload.get("data_files")
    if isinstance(data_files, dict):
        return data_files
    return payload


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateCalibrationRunError(f"required calibration input is missing: {path}") from exc
    if not isinstance(payload, dict):
        raise GateCalibrationRunError(f"required calibration input must be a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
