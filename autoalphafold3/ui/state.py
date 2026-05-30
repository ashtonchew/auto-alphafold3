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
class TrialRow:
    """One row in the Trials table (every submitted trial)."""

    trial_id: str
    move_family: str
    axis: str
    score: str
    delta: str
    runtime: str
    status: str  # display label
    tone: str  # spill tone: ok | bad | warn | info | muted
    cat: str  # filter category: confirmed | keep | discard | killed | fail


@dataclass
class LogEvent:
    """One line in the Logs feed."""

    time: str
    level: str  # info | ok | warn | err
    trial: str
    message: str


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
    trials: list = field(default_factory=list)  # TrialRow, for the Trials view
    logs: list = field(default_factory=list)  # LogEvent, for the Logs view

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


def _status_display(status_value: str, discovery_value: str, kill_reason: str = "") -> tuple[str, str, str]:
    """Return (label, spill tone, filter category) for the Trials table."""
    if discovery_value == "CONFIRMED":
        return "CONFIRMED", "ok", "confirmed"
    if status_value == "KEEP":
        return "KEEP", "ok", "keep"
    if discovery_value in _KILLED:
        return kill_reason or "KILLED", "bad", "killed"
    if status_value == "DISCARD":
        return "DISCARD", "muted", "discard"
    if status_value == "FAIL":
        return "FAIL", "warn", "fail"
    if status_value == "INFRA_FAIL":
        return "INFRA_FAIL", "info", "fail"
    return status_value, "muted", "other"


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
        trials=_build_trials(rows, baseline),
        logs=_build_logs(rows),
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


def _fmt_runtime(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def _enum_value(obj: object, *names: str) -> str:
    for name in names:
        val = getattr(obj, name, None)
        if val:
            return str(getattr(val, "value", val))
    return ""


def _build_trials(rows: list, baseline: float | None) -> list[TrialRow]:
    out: list[TrialRow] = []
    for row in rows:
        status = getattr(row.status, "value", str(row.status))
        discovery = _enum_value(row, "discovery")
        fals = getattr(row, "falsification", None)
        verdict = _enum_value(fals, "verdict") if fals is not None else ""
        axis = _enum_value(fals, "named_axis", "axis") if fals is not None else ""
        label, tone, cat = _status_display(status, discovery, verdict if discovery in _KILLED else "")
        metrics = row.metrics if isinstance(row.metrics, dict) else {}
        score = metrics.get("best_val_calpha_lddt")
        score_s = f"{float(score):.3f}" if isinstance(score, (int, float)) else "—"
        delta_s = f"{float(score) - baseline:+.3f}" if isinstance(score, (int, float)) and baseline is not None else "—"
        rt = metrics.get("runtime_seconds")
        out.append(
            TrialRow(
                trial_id=row.trial_id,
                move_family=str(metrics.get("move_family", "")) or "—",
                axis=axis or "—",
                score=score_s,
                delta=delta_s,
                runtime=_fmt_runtime(rt) if isinstance(rt, (int, float)) else "—",
                status=label,
                tone=tone,
                cat=cat,
            )
        )
    return out


def _build_logs(rows: list) -> list[LogEvent]:
    out: list[LogEvent] = []
    for row in rows:
        status = getattr(row.status, "value", str(row.status))
        discovery = _enum_value(row, "discovery")
        metrics = row.metrics if isinstance(row.metrics, dict) else {}
        score = metrics.get("best_val_calpha_lddt")
        out.append(LogEvent("", "info", row.trial_id, "submitted"))
        if isinstance(score, (int, float)):
            out.append(LogEvent("", "ok", row.trial_id, f"scored · best_val_calpha_lddt {float(score):.3f}"))
        if discovery == "CONFIRMED":
            out.append(LogEvent("", "ok", row.trial_id, "CONFIRMED at the gate"))
        elif discovery in _KILLED:
            out.append(LogEvent("", "warn", row.trial_id, "killed at the gate"))
        if status in _FAIL:
            out.append(LogEvent("", "err", row.trial_id, status))
    return out


# --- illustrative sample (design mockup data) ------------------------------

_SAMPLE_TRAJ = [
    ("T001", 0.325, "provisional"), ("T003", 0.318, "killed"), ("T006", 0.331, "provisional"),
    ("T009", 0.344, "confirmed"), ("T013", 0.339, "confirmed"), ("T018", 0.357, "confirmed"),
    ("T022", 0.349, "killed"), ("T027", 0.372, "confirmed"), ("T031", 0.381, "confirmed"),
    ("T036", 0.398, "confirmed"), ("T039", 0.405, "confirmed"), ("T042", 0.412, "confirmed"),
]

_SAMPLE_ERR = [0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 3, 4, 2, 1, 0, 0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 1, 0, 0]


def _sample_trials() -> list[TrialRow]:
    raw = [
        ("T042", "geometry_loss", "distogram_vs_3d", "0.412", "+0.041", "8m 05s", "CONFIRMED", "ok", "confirmed"),
        ("T019", "auxiliary_loss", "long_range_topology", "0.351", "+0.026", "7m 48s", "CONFIRMED", "ok", "confirmed"),
        ("T027", "diffusion_sampler", "distogram_vs_3d", "0.372", "+0.018", "5m 02s", "CONFIRMED", "ok", "confirmed"),
        ("T031", "pairformer", "local_geometry", "0.381", "+0.012", "9m 20s", "KEEP", "ok", "keep"),
        ("T040", "diffusion_sampler", "distogram_vs_3d", "0.405", "+0.006", "4m 58s", "KEEP", "ok", "keep"),
        ("T009", "recycling", "local_geometry", "0.344", "+0.008", "6m 11s", "KEEP", "ok", "keep"),
        ("T022", "pair_attention", "long_range_topology", "0.349", "+0.012", "8m 40s", "PLACEBO_KILL", "bad", "killed"),
        ("T030", "recycling", "local_geometry", "0.366", "+0.015", "7m 55s", "KNOCKOUT_SURVIVES", "bad", "killed"),
        ("T035", "normalization", "stability_compute", "0.358", "+0.009", "6m 30s", "SEED_FRAGILE", "bad", "killed"),
        ("T006", "optimizer", "—", "0.331", "+0.006", "5m 45s", "DISCARD", "muted", "discard"),
        ("T013", "curriculum", "—", "0.339", "+0.001", "6m 05s", "DISCARD", "muted", "discard"),
        ("T018", "msa_module", "—", "0.357", "—", "6m 22s", "DISCARD", "muted", "discard"),
        ("T003", "geometry_loss", "—", "—", "—", "2m 10s", "FAIL", "warn", "fail"),
        ("T017", "msa_module", "—", "—", "—", "0m 31s", "INFRA_FAIL", "info", "fail"),
    ]
    return [TrialRow(*r) for r in raw]


def _sample_logs() -> list[LogEvent]:
    raw = [
        ("09:18:02", "info", "T042", "submitted · geometry_loss · budget short-training"),
        ("09:18:03", "info", "T042", "preflight passed · locked mounts verified"),
        ("09:18:05", "info", "—", "Modal trial wave spawned · A100-80GB"),
        ("09:26:08", "ok", "T042", "scored · best_val_calpha_lddt 0.412"),
        ("09:26:09", "info", "—", "gate wave spawned · knock-out, placebo, seed×3"),
        ("09:27:11", "ok", "T042", "CONFIRMED · attributable 0.78 · gap closed"),
        ("09:27:12", "info", "—", "discovery ledger updated · 7 confirmed"),
        ("09:30:44", "info", "T043", "submitted · pair_attention"),
        ("09:38:01", "warn", "T043", "PLACEBO_KILL · matched placebo reproduced 0.7 of gain"),
        ("09:39:20", "info", "T044", "submitted · recycling"),
        ("09:46:55", "warn", "T044", "KNOCKOUT_SURVIVES · credit misattributed"),
        ("09:48:10", "info", "T017", "submitted · msa_module"),
        ("09:48:41", "err", "T017", "INFRA_FAIL · container OOM during polling"),
        ("09:52:03", "info", "T045", "submitted · diffusion_sampler"),
        ("09:57:39", "ok", "T045", "scored · best_val_calpha_lddt 0.405 · KEEP"),
        ("09:58:00", "info", "—", "scaled to zero · idle"),
    ]
    return [LogEvent(*r) for r in raw]


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
        trials=_sample_trials(),
        logs=_sample_logs(),
    )
