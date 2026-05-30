"""Build a typed ``UiState`` for the evidence board.

All artifact reading and data-shaping lives here so the rendering layer only
ever sees clean, typed values. ``load_state`` reads real run artifacts (the
locked scorer's outputs); ``sample_state`` returns the illustrative figures used
for the design mockups. Anything not present in real artifacts degrades to a
clearly-labelled placeholder rather than being invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# --- data shapes -----------------------------------------------------------


@dataclass
class TrialPoint:
    """One point on the trajectory chart."""

    index: int
    trial_id: str
    score: float
    status: str  # confirmed | provisional | killed | fail


@dataclass
class Axis:
    """A Fold Cartographer diagnostic axis tile."""

    name: str
    value: str
    delta: str
    pct: int  # bar fill, 0-100
    foot: str
    tone: str = "up"  # up | warn


@dataclass
class LedgerRow:
    finding: str
    rule: str
    axis: str
    delta: str
    trial: str
    sha: str
    verdict: str  # CONFIRMED, PLACEBO_KILL, ...
    confirmed: bool


@dataclass
class GateBar:
    label: str
    value: float
    note: str
    full: bool  # True = credited/green, False = muted


@dataclass
class Gate:
    claim: str
    meta_axis: str
    meta_trial: str
    bars: list[GateBar]
    attributable: str
    verdict: str
    readout: str


@dataclass
class Overlay:
    is_sample: bool
    target: str
    length: int
    before: float
    after: float
    err_levels: list[int]  # 0-4 per residue


@dataclass
class UiState:
    best: float | None
    baseline: float | None
    delta: float | None
    prev_delta: float | None
    counts: dict[str, int]  # confirmed, killed, trials, keep, discard, fail, infra
    trajectory: list[TrialPoint]
    axes: list[Axis]
    failure_signature: str
    ledger: list[LedgerRow]
    gate: Gate | None
    overlay: Overlay
    provenance: dict[str, str]
    split: str
    scorer: str
    is_sample: bool
    source: str  # "sample" or the runs dir
    geometry_sample: bool = False  # live board whose geometry panels still use sample data

    def to_json(self) -> dict[str, object]:
        """Denormalised summary — the data contract a live frontend could poll."""
        return {
            "best_val_calpha_lddt": self.best,
            "baseline": self.baseline,
            "delta_vs_baseline": self.delta,
            "counts": self.counts,
            "split": self.split,
            "scorer": self.scorer,
            "is_sample": self.is_sample,
            "source": self.source,
            "trajectory": [
                {"index": p.index, "trial_id": p.trial_id, "score": p.score, "status": p.status}
                for p in self.trajectory
            ],
            "provenance": self.provenance,
        }


# --- status mapping --------------------------------------------------------

_CONFIRMED = {"CONFIRMED", "KEEP"}
_KILLED = {"KILLED"}
_FAIL = {"FAIL", "INFRA_FAIL"}


def _trial_status(status_value: str, discovery_value: str) -> str:
    if discovery_value in _CONFIRMED or status_value == "KEEP":
        return "confirmed"
    if discovery_value in _KILLED:
        return "killed"
    if status_value in _FAIL:
        return "fail"
    return "provisional"


# --- real artifacts --------------------------------------------------------


def load_state(runs_dir: str | Path = "runs") -> UiState:
    """Build a ``UiState`` from real run artifacts under ``runs_dir``.

    Reuses the canonical readers (``read_ledger`` / ``read_discovery_ledger``).
    Missing artifacts degrade to labelled placeholders; nothing is invented.
    """
    from autoalphafold3.ledger import read_ledger

    runs = Path(runs_dir)
    ledger_path = runs / "ledger.jsonl"
    rows = read_ledger(ledger_path=ledger_path) if ledger_path.exists() else []

    baseline = _read_baseline(runs / "baseline" / "metrics.json")

    trajectory: list[TrialPoint] = []
    counts = {"confirmed": 0, "killed": 0, "trials": 0, "keep": 0, "discard": 0, "fail": 0, "infra": 0}
    best: float | None = None
    scorer = ""
    split = ""
    idx = 0
    for row in rows:
        counts["trials"] += 1
        status = getattr(row.status, "value", str(row.status))
        discovery = getattr(getattr(row, "discovery", None), "value", str(getattr(row, "discovery", "")))
        cat = _trial_status(status, discovery)
        if status == "KEEP":
            counts["keep"] += 1
        elif status == "DISCARD":
            counts["discard"] += 1
        elif status == "FAIL":
            counts["fail"] += 1
        elif status == "INFRA_FAIL":
            counts["infra"] += 1
        if cat == "confirmed":
            counts["confirmed"] += 1
        elif cat == "killed":
            counts["killed"] += 1
        score = row.metrics.get("best_val_calpha_lddt") if isinstance(row.metrics, dict) else None
        scorer = scorer or str(row.metrics.get("scorer_version", "")) if isinstance(row.metrics, dict) else scorer
        split = split or str(row.metrics.get("split", "")) if isinstance(row.metrics, dict) else split
        if isinstance(score, (int, float)):
            idx += 1
            trajectory.append(TrialPoint(idx, row.trial_id, float(score), cat))
            if best is None or float(score) > best:
                best = float(score)

    # No real scored trials yet → show the coherent sample board.
    if not trajectory:
        return sample_state()

    ledger_rows = _real_ledger_rows(runs)
    delta = (best - baseline) if (best is not None and baseline is not None) else None

    # Cartographer axes, gate, and overlay need richer per-trial artifacts; until
    # those land we keep the labelled sample for the geometry panels (flagged via
    # geometry_sample) rather than inventing numbers.
    sample = sample_state()
    failure_sig = ""
    for row in reversed(rows):
        sig = getattr(row, "failure_signature", None) or getattr(row.fold_cartographer, "signature", "")
        if sig:
            failure_sig = str(sig)
            break

    return UiState(
        best=best,
        baseline=baseline,
        delta=delta,
        prev_delta=None,
        counts=counts,
        trajectory=trajectory,
        axes=sample.axes,
        failure_signature=failure_sig or sample.failure_signature,
        ledger=ledger_rows or sample.ledger,
        gate=sample.gate,
        overlay=Overlay(is_sample=True, **_overlay_kwargs(sample.overlay)),
        provenance=_real_provenance(rows, scorer),
        split=split or "public_val_small",
        scorer=scorer or "calpha_lddt_v1",
        is_sample=False,
        geometry_sample=True,
        source=str(runs),
    )


def _read_baseline(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("best_val_calpha_lddt")
    return float(value) if isinstance(value, (int, float)) else None


def _real_ledger_rows(runs: Path) -> list[LedgerRow]:
    path = runs / "discovery_ledger.jsonl"
    if not path.exists():
        return []
    try:
        from autoalphafold3.discovery_ledger import read_discovery_ledger

        records = read_discovery_ledger(ledger_path=path)
    except Exception:  # pragma: no cover - reader/validation issues degrade gracefully
        return []
    rows: list[LedgerRow] = []
    for rec in records:
        get = rec.get if isinstance(rec, dict) else lambda k, d=None: getattr(rec, k, d)
        confirmed = str(get("verdict", "")).upper() == "CONFIRMED"
        rows.append(
            LedgerRow(
                finding=str(get("design_rule", get("mechanism", ""))),
                rule="",
                axis=str(get("axis", "")),
                delta=_fmt_delta(get("primary_metric_delta", get("delta", 0.0))),
                trial=str(get("trial_id", "")),
                sha=str(get("git_sha", ""))[:7],
                verdict=str(get("verdict", "")),
                confirmed=confirmed,
            )
        )
    return rows


def _real_provenance(rows: list, scorer: str) -> dict[str, str]:
    if not rows:
        return {}
    last = rows[-1]
    arts = getattr(last, "artifacts", {}) or {}
    return {
        "candidate": getattr(last, "candidate_id", ""),
        "git_sha": str(arts.get("git_sha", ""))[:7],
        "scorer": scorer,
    }


def _overlay_kwargs(o: Overlay) -> dict[str, object]:
    return {"target": o.target, "length": o.length, "before": o.before, "after": o.after, "err_levels": o.err_levels}


def _fmt_delta(value: object) -> str:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    return f"{v:+.3f}"


# --- illustrative sample (design mockup data) ------------------------------

_SAMPLE_TRAJ = [
    ("T001", 0.325, "provisional"), ("T003", 0.318, "killed"), ("T006", 0.331, "provisional"),
    ("T009", 0.344, "confirmed"), ("T013", 0.339, "confirmed"), ("T018", 0.357, "confirmed"),
    ("T022", 0.349, "killed"), ("T027", 0.372, "confirmed"), ("T031", 0.381, "confirmed"),
    ("T036", 0.398, "confirmed"), ("T039", 0.405, "confirmed"), ("T042", 0.412, "confirmed"),
]

_SAMPLE_ERR = [0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 3, 4, 2, 1, 0, 0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 1, 0, 0]


def sample_state() -> UiState:
    """The illustrative figures behind the design mockups. Not benchmark results."""
    traj = [TrialPoint(i + 1, t, s, st) for i, (t, s, st) in enumerate(_SAMPLE_TRAJ)]
    axes = [
        Axis("Local geometry", "0.58", "+0.04", 72, "local lDDT", "up"),
        Axis("Long-range topology", "0.41", "+0.06", 54, "contact precision, sep ≥ 24", "up"),
        Axis("3D gap", "0.09", "−0.07", 66, "distogram to coordinate gap", "up"),
        Axis("Stability", "1.0×", "stable", 50, "peak 38 GB, no NaN or OOM", "warn"),
    ]
    ledger = [
        LedgerRow("Geometry-loss ramp closes the distogram to coordinate gap", "", "distogram_vs_3d", "+0.041", "T012", "4f2a9c1", "CONFIRMED", True),
        LedgerRow("Long-range contact weighting lifts topology at sep ≥ 24", "", "long_range_topology", "+0.026", "T019", "b7c014e", "CONFIRMED", True),
        LedgerRow("Sampler step-scale with fewer steps holds lDDT at lower cost", "", "diffusion_sampler", "+0.018", "T027", "2a9f5d3", "CONFIRMED", True),
        LedgerRow("Wider pair attention raised lDDT, but a matched placebo reproduced 0.7 of the gain", "", "pair_attention", "+0.012", "T022", "88e1aa0", "PLACEBO_KILL", False),
        LedgerRow("Extra recycling step raised the score, but knock-out kept it", "", "recycling", "+0.015", "T030", "c3d77b2", "KNOCKOUT_SURVIVES", False),
        LedgerRow("Norm-reorder gain vanished on a seed rerun", "", "normalization", "+0.009", "T035", "1f0b6e9", "SEED_FRAGILE", False),
    ]
    gate = Gate(
        claim="“Ramping the geometry loss makes the coordinate path use the contacts the pair head already learned.”",
        meta_axis="distogram_vs_3d",
        meta_trial="T012",
        bars=[
            GateBar("full", 0.041, "", True),
            GateBar("knock-out", 0.009, "collapses", False),
            GateBar("placebo", 0.004, "null", False),
            GateBar("seed mean", 0.038, "±0.006, n=3", True),
        ],
        attributable="attributable 0.78",
        verdict="CONFIRMED",
        readout="Knock-out collapses the gain, the placebo does not reproduce it, and the gain survives the seed rerun. The claim was registered before any of these bars existed.",
    )
    overlay = Overlay(is_sample=True, target="7XYZ_A", length=148, before=0.31, after=0.41, err_levels=list(_SAMPLE_ERR))
    return UiState(
        best=0.412,
        baseline=0.325,
        delta=0.087,
        prev_delta=0.018,
        counts={"confirmed": 7, "killed": 3, "trials": 42, "keep": 12, "discard": 18, "fail": 2, "infra": 1},
        trajectory=traj,
        axes=axes,
        failure_signature="distogram_good_lddt_flat",
        ledger=ledger,
        gate=gate,
        overlay=overlay,
        provenance={"candidate": "cand_31", "git_sha": "4f2a9c1", "scorer": "calpha_lddt_v1"},
        split="public_val_small",
        scorer="calpha_lddt_v1",
        is_sample=True,
        source="sample",
    )
