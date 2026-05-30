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


# --- illustrative sample: one-hour sampler-only run ------------------------
# 20 frozen-checkpoint sampler candidates; the best provisional KEEP (T012) is
# gated and confirmed. The rest stay provisional (ungated). Not benchmark results.

_SAMPLE_TRAJ = [
    ("T001", 0.321, "provisional"), ("T002", 0.318, "provisional"), ("T003", 0.326, "provisional"),
    ("T004", 0.331, "provisional"), ("T005", 0.316, "provisional"), ("T006", 0.337, "provisional"),
    ("T007", 0.323, "provisional"), ("T008", 0.333, "provisional"), ("T009", 0.319, "provisional"),
    ("T010", 0.339, "provisional"), ("T011", 0.314, "provisional"), ("T012", 0.343, "confirmed"),
    ("T013", 0.322, "provisional"), ("T014", 0.336, "provisional"), ("T015", 0.317, "provisional"),
    ("T016", 0.328, "provisional"), ("T017", 0.320, "provisional"), ("T018", 0.334, "provisional"),
    ("T019", 0.315, "provisional"), ("T020", 0.324, "provisional"),
]

_SAMPLE_ERR = [0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 3, 4, 2, 1, 0, 0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 1, 0, 0]


def _sample_trials() -> list[TrialRow]:
    raw = [
        ("T012", "sampler_step_scale", "distogram_vs_3d", "0.343", "+0.018", "1m 38s", "CONFIRMED", "ok", "confirmed"),
        ("T010", "sample_count", "distogram_vs_3d", "0.339", "+0.014", "2m 02s", "KEEP", "ok", "keep"),
        ("T006", "sampler_step_scale", "distogram_vs_3d", "0.337", "+0.012", "1m 41s", "KEEP", "ok", "keep"),
        ("T014", "sampler_step_scale", "distogram_vs_3d", "0.336", "+0.011", "1m 44s", "KEEP", "ok", "keep"),
        ("T018", "sample_count", "distogram_vs_3d", "0.334", "+0.009", "2m 05s", "KEEP", "ok", "keep"),
        ("T008", "noise_schedule", "distogram_vs_3d", "0.333", "+0.008", "1m 52s", "KEEP", "ok", "keep"),
        ("T004", "diffusion_steps", "distogram_vs_3d", "0.331", "+0.006", "2m 18s", "KEEP", "ok", "keep"),
        ("T016", "noise_schedule", "stability_compute", "0.328", "+0.003", "1m 49s", "KEEP", "ok", "keep"),
        ("T003", "diffusion_steps", "distogram_vs_3d", "0.326", "+0.001", "2m 21s", "KEEP", "ok", "keep"),
        ("T020", "sample_count", "—", "0.324", "−0.001", "2m 03s", "DISCARD", "muted", "discard"),
        ("T007", "sampler_step_scale", "—", "0.323", "−0.002", "1m 39s", "DISCARD", "muted", "discard"),
        ("T013", "diffusion_steps", "—", "0.322", "−0.003", "2m 15s", "DISCARD", "muted", "discard"),
        ("T001", "sampler_step_scale", "—", "0.321", "−0.004", "1m 36s", "DISCARD", "muted", "discard"),
        ("T017", "noise_schedule", "—", "0.320", "−0.005", "1m 47s", "DISCARD", "muted", "discard"),
        ("T009", "diffusion_steps", "—", "0.319", "−0.006", "2m 24s", "DISCARD", "muted", "discard"),
        ("T002", "sample_count", "—", "0.318", "−0.007", "2m 08s", "DISCARD", "muted", "discard"),
        ("T015", "sampler_step_scale", "—", "0.317", "−0.008", "1m 42s", "DISCARD", "muted", "discard"),
        ("T005", "noise_schedule", "—", "0.316", "−0.009", "1m 50s", "DISCARD", "muted", "discard"),
        ("T019", "diffusion_steps", "—", "0.315", "−0.010", "2m 19s", "DISCARD", "muted", "discard"),
        ("T011", "sample_count", "—", "0.314", "−0.011", "2m 06s", "DISCARD", "muted", "discard"),
    ]
    return [TrialRow(*r) for r in raw]


def _sample_logs() -> list[LogEvent]:
    raw = [
        ("09:08:02", "info", "—", "frozen checkpoint cand_31 loaded · step 6,000"),
        ("09:08:03", "info", "—", "preflight passed · locked mounts verified"),
        ("09:08:05", "info", "—", "sampler burst spawned · 20 candidates · A100-80GB"),
        ("09:08:06", "info", "—", "worker cap 6 · ~4 waves"),
        ("09:14:21", "ok", "T012", "scored · best_val_calpha_lddt 0.343"),
        ("09:16:40", "ok", "T010", "scored · best_val_calpha_lddt 0.339"),
        ("09:18:55", "warn", "T011", "below baseline · logged provisional"),
        ("09:19:55", "info", "—", "20 candidates scored · ranking provisional KEEPs"),
        ("09:20:02", "info", "T012", "selected best provisional KEEP"),
        ("09:20:05", "info", "—", "gate wave spawned · knock-out, placebo, seed×3"),
        ("09:22:47", "info", "—", "other improving candidates kept provisional, not gated"),
        ("09:27:38", "ok", "T012", "predicted axis distogram_vs_3d moved as registered"),
        ("09:28:11", "ok", "T012", "CONFIRMED · attributable 0.74"),
        ("09:28:12", "info", "—", "discovery ledger updated · 1 confirmed"),
        ("09:28:30", "info", "—", "scaled to zero · idle"),
    ]
    return [LogEvent(*r) for r in raw]


def sample_state() -> UiState:
    """Illustrative one-hour sampler-only run: 20 frozen-checkpoint candidates, the
    best provisional KEEP gated and confirmed. Not benchmark results."""
    traj = [TrialPoint(i + 1, t, s, st) for i, (t, s, st) in enumerate(_SAMPLE_TRAJ)]
    axes = [
        Axis("Local geometry", "0.56", "+0.01", 58, "local lDDT, roughly flat", "warn"),
        Axis("Long-range topology", "0.39", "±0.00", 45, "unchanged by the sampler", "warn"),
        Axis("3D gap", "0.06", "−0.05", 70, "distogram to coordinate gap closes", "up"),
        Axis("Stability", "0.7×", "faster", 55, "fewer steps, 0.7× runtime", "up"),
    ]
    ledger = [
        LedgerRow("Fewer diffusion steps with a lower step scale close the distogram to coordinate gap", "", "distogram_vs_3d", "+0.018", "T012", "4f2a9c1", "CONFIRMED", True),
    ]
    gate = Gate(
        claim="“Fewer diffusion steps with a lower step scale reduce sampler noise, so the coordinate path realizes the contacts the pair head already learned.”",
        meta_axis="distogram_vs_3d",
        meta_trial="T012",
        bars=[
            GateBar("full", 0.018, "", True),
            GateBar("knock-out", 0.004, "collapses", False),
            GateBar("placebo", 0.002, "null", False),
            GateBar("seed mean", 0.016, "±0.004, n=3", True),
        ],
        attributable="attributable 0.74",
        verdict="CONFIRMED",
        readout="Reverting the sampler change erases most of the gain, a matched sampler placebo does not reproduce it, and the gain survives the seed rerun. The claim was registered before any of these bars existed.",
    )
    overlay = Overlay(is_sample=True, target="7XYZ_A", length=148, before=0.325, after=0.343, err_levels=list(_SAMPLE_ERR))
    return UiState(
        best=0.343,
        baseline=0.325,
        delta=0.018,
        prev_delta=None,
        counts={"confirmed": 1, "killed": 0, "trials": 20, "keep": 8, "discard": 11, "fail": 0, "infra": 0},
        trajectory=traj,
        axes=axes,
        failure_signature="distogram_good_lddt_flat",
        ledger=ledger,
        gate=gate,
        overlay=overlay,
        provenance={"candidate": "cand_31 · sampler", "git_sha": "4f2a9c1", "scorer": "calpha_lddt_v1"},
        split="public_val_small",
        scorer="calpha_lddt_v1",
        is_sample=True,
        source="sample",
        trials=_sample_trials(),
        logs=_sample_logs(),
    )
