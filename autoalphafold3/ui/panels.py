"""Board sections.

Each function reads specific ``UiState`` fields and composes ``components`` into
one section of the evidence board. The data → markup binding for a section lives
entirely in its function (locality of behaviour). Functions return the *inner*
content; ``page.render_board`` wraps them in rows.
"""

from __future__ import annotations

from autoalphafold3.ui import components
from autoalphafold3.ui.components import esc, status_pill
from autoalphafold3.ui.state import UiState


def _geom_badge(s: UiState) -> str:
    """A 'sample' pill for geometry panels on a live board not yet backed by artifacts."""
    if s.geometry_sample and not s.is_sample:
        return ' <span class="spill warn" style="margin-left:8px">sample</span>'
    return ""


def metric_band(s: UiState) -> str:
    best = f"{s.best:.3f}" if s.best is not None else "—"
    delta = ""
    if s.delta is not None and s.baseline is not None:
        delta = (
            '<span class="metric-delta"><span>↑</span> '
            f'<span class="num">{s.delta:+.3f}</span> '
            f'<span class="vs num">vs baseline {s.baseline:.3f}</span></span>'
        )
    running: list[float] = []
    cur: float | None = None
    for p in s.trajectory:
        cur = p.score if cur is None else max(cur, p.score)
        running.append(cur)
    spark = components.sparkline(running) if running else ""
    chips = (
        components.chip(s.counts.get("confirmed", 0), "confirmed", "green")
        + components.chip(s.counts.get("killed", 0), "killed", "red")
        + components.chip(s.counts.get("trials", 0), "trials")
    )
    return (
        '<section class="metricband"><div class="metric">'
        '<div class="metric-lbl">Best validation Cα lDDT</div>'
        f'<div class="metric-row"><span class="metric-val num">{esc(best)}</span>{delta}{spark}</div>'
        f'</div><div class="chips">{chips}</div></section>'
    )


def trajectory_section(s: UiState) -> tuple[str, str]:
    svg, traj_json = components.trajectory_chart(s.trajectory, s.baseline)
    legend = (
        '<div class="legend">'
        '<span class="lg-item"><i class="sq conf"></i> confirmed</span>'
        '<span class="lg-item"><i class="sq prov"></i> provisional</span>'
        '<span class="lg-item"><i class="x-mark">✕</i> killed</span>'
        '<span class="spacer"></span>'
        '<label class="toggle"><input type="checkbox" class="kill-toggle" checked>'
        '<span class="track"><span class="knob"></span></span><span>Show killed</span></label>'
        '</div>'
    )
    inner = (
        '<h2 class="block-title">Trial trajectory</h2>'
        '<div class="block-sub">Validation Cα lDDT per trial. Failed and killed runs stay visible.</div>'
        f'{svg}{legend}'
    )
    return inner, traj_json


def cartographer_section(s: UiState) -> str:
    tiles = ""
    for a in s.axes:
        bar_cls = ' class="amber"' if a.tone == "warn" else ""
        tiles += (
            f'<div class="diag"><div class="diag-name">{esc(a.name)}</div>'
            f'<div class="diag-row"><span class="diag-val num">{esc(a.value)}</span>'
            f'<span class="diag-delta {esc(a.tone)} num">{esc(a.delta)}</span></div>'
            f'<div class="diag-bar"><i{bar_cls} style="width:{int(a.pct)}%"></i></div>'
            f'<div class="diag-foot">{esc(a.foot)}</div></div>'
        )
    return (
        f'<h2 class="block-title">Fold Cartographer{_geom_badge(s)}</h2>'
        '<div class="block-sub">Four geometry axes. Movement vs baseline, best candidate.</div>'
        f'<div class="diag-grid">{tiles}</div>'
    )


def ledger_section(s: UiState) -> str:
    rows = ""
    for r in s.ledger:
        if r.confirmed:
            pill = status_pill("CONFIRMED", "ok", "✓")
            dcls = "c-d num"
        else:
            pill = status_pill(r.verdict, "bad", "✕")
            dcls = "c-d muted num"
        rows += (
            f"<tr><td>{esc(r.finding)}</td>"
            f'<td class="c-axis">{esc(r.axis)}</td>'
            f'<td class="r {dcls}">{esc(r.delta)}</td>'
            f'<td class="r">{pill}</td></tr>'
        )
    cap = (
        "Killed rows show the apparent gain that did not survive a control. "
        f"{s.counts.get('confirmed', 0)} confirmed, {s.counts.get('killed', 0)} killed "
        f"across {s.counts.get('trials', 0)} trials."
    )
    return (
        '<h2 class="block-title">Discovery Ledger</h2>'
        '<div class="block-sub">Confirmed mechanisms and instructive kills, with the verdict that survived the gate.</div>'
        '<table class="dtable"><thead><tr><th>Finding</th><th>Axis</th>'
        '<th class="r">Δ lDDT</th><th class="r">Verdict</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
        f'<div class="led-cap">{esc(cap)}</div>'
    )


def gate_section(s: UiState) -> str:
    g = s.gate
    if g is None:
        return ""
    return (
        f'<h2 class="block-title">Headline discovery on trial{_geom_badge(s)} '
        f'<span class="right"><span class="badge">{esc(g.verdict)}</span></span></h2>'
        f'<p class="claim">{esc(g.claim)}</p>'
        f'<div class="claim-meta">pre-registered axis <span class="mono">{esc(g.meta_axis)}</span> '
        f'· trial <span class="mono">{esc(g.meta_trial)}</span></div>'
        f"{components.control_bars(g.bars)}"
        f'<div class="readout"><span class="mono">{esc(g.attributable)}</span>. {esc(g.readout)}</div>'
    )


def overlay_section(s: UiState) -> str:
    o = s.overlay
    return (
        f'<h2 class="block-title">Structure overlay{_geom_badge(s)}</h2>'
        '<div class="block-sub">Predicted vs true Cα backbone, best confirmed candidate.</div>'
        '<div class="ov-grid"><div>'
        f"{components.backbone_overlay()}"
        '<div class="ov-legend"><span><i style="background:#dcdcdc"></i> true</span>'
        '<span><i style="background:#5896f3"></i> baseline</span>'
        '<span><i style="background:#7fee64"></i> best</span></div>'
        '</div><div>'
        '<div class="err-title">Per-residue Cα error, best candidate</div>'
        f"{components.err_strip(o.err_levels)}"
        '<div class="err-scale"><span>low error, core</span><span>high, loops</span></div>'
        f'<div class="ov-meta">target <span class="mono">{esc(o.target)}</span> · L = {int(o.length)} '
        f'· before/after lDDT <span class="mono num">{o.before:.2f} → {o.after:.2f}</span></div>'
        '</div></div>'
    )
