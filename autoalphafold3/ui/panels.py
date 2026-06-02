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
    killed = s.counts.get("killed", 0)
    middle = (
        components.chip(killed, "killed", "red")
        if killed
        else components.chip(s.counts.get("keep", 0), "provisional KEEP")
    )
    chips = (
        components.chip(s.counts.get("confirmed", 0), "confirmed", "green")
        + middle
        + components.chip(s.counts.get("trials", 0), "candidates")
    )
    return (
        '<section class="metricband"><div class="metric">'
        '<div class="metric-lbl">Best validation Cα lDDT</div>'
        f'<div class="metric-row"><span class="metric-val num">{esc(best)}</span>{delta}{spark}</div>'
        f'</div><div class="chips">{chips}</div></section>'
    )


def autoresearch_section(s: UiState) -> str:
    rows = ""
    for run in s.autoresearch_runs:
        for c in run.candidates:
            is_planned = c.status == "PLANNED" or (c.status == "DRAFT" and c.planning_status == "PLANNED")
            tone = "ok" if c.provisional_keep else ("info" if is_planned else "muted")
            if c.status in {"FAIL", "INFRA_FAIL"}:
                tone = "warn"
            status = "PROVISIONAL KEEP" if c.provisional_keep else ("PLANNED" if is_planned else c.status)
            rows += (
                f"<tr><td>{esc(run.run_id)}</td>"
                f'<td class="c-mono">{esc(c.trial_id)}</td>'
                f"<td>{esc(run.planner)}</td>"
                f'<td class="r c-d muted num">{esc(c.matched_budget_delta)}</td>'
                f'<td class="r c-d muted num">{esc(c.global_baseline_delta)}</td>'
                f'<td class="r">{status_pill(status, tone)}</td></tr>'
            )
    official = any(run.official_benchmark_result for run in s.autoresearch_runs)
    note = (
        "Official benchmark result: false. These rows are autoresearch planning/evidence artifacts, "
        "not canonical scorer ledger entries or Discovery Ledger records."
        if not official
        else "Official benchmark flag present; verify scorer-owned evidence before making claims."
    )
    return (
        '<h2 class="block-title">Autoresearch evidence</h2>'
        f'<div class="block-sub">{esc(note)}</div>'
        '<table class="dtable"><thead><tr><th>Run</th><th>Trial</th><th>Planner</th>'
        '<th class="r">Matched Δ</th><th class="r">Global Δ</th><th class="r">Status</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


def trajectory_section(s: UiState) -> tuple[str, str]:
    svg, traj_json = components.trajectory_chart(s.trajectory, s.baseline)
    killed = s.counts.get("killed", 0)
    if killed:
        # Killed trials present: surface them with a toggle to show/hide.
        legend = (
            '<div class="legend">'
            '<span class="lg-item"><i class="sq conf"></i> confirmed best</span>'
            '<span class="lg-item"><i class="sq prov"></i> provisional candidate</span>'
            '<span class="lg-item"><i class="x-mark">✕</i> killed</span>'
            '<span class="spacer"></span>'
            '<label class="toggle"><input type="checkbox" class="kill-toggle" checked>'
            '<span class="track"><span class="knob"></span></span><span>Show killed</span></label>'
            '</div>'
        )
        sub = 'Validation Cα lDDT per candidate. Failed and killed runs stay visible.'
    else:
        # No kills this run: skip the dead toggle, name the baseline split instead.
        legend = (
            '<div class="legend">'
            '<span class="lg-item"><i class="sq conf"></i> confirmed best</span>'
            '<span class="lg-item"><i class="sq prov"></i> provisional candidate</span>'
            '<span class="spacer"></span>'
            '<span class="lg-note">above the baseline line is KEEP territory; below is DISCARD</span>'
            '</div>'
        )
        sub = 'Validation Cα lDDT per candidate. Every candidate stays visible, kept or discarded.'
    inner = (
        '<h2 class="block-title">Trial trajectory</h2>'
        f'<div class="block-sub">{sub}</div>'
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
        f'<h2 class="block-title">Fold Cartographer</h2>'
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
    killed = s.counts.get("killed", 0)
    cap = (
        "Confirmed-only: the gate runs on the best provisional KEEP. "
        f"{s.counts.get('confirmed', 0)} confirmed of {s.counts.get('trials', 0)} candidates"
        + (f"; {killed} killed at the gate." if killed else ".")
    )
    sub = (
        "Confirmed mechanisms and instructive kills, with the verdict that survived the gate."
        if killed
        else "Confirmed mechanisms only — the gated claim that survived knock-out, placebo, and a seed rerun."
    )
    return (
        f'<h2 class="block-title">Discovery Ledger</h2>'
        f'<div class="block-sub">{sub}</div>'
        '<table class="dtable"><thead><tr><th>Finding</th><th>Axis</th>'
        '<th class="r">Δ lDDT</th><th class="r">Verdict</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
        f'<div class="led-cap">{esc(cap)}</div>'
    )


def hypothesis_section(s: UiState) -> str:
    """Pre-registered hypothesis card, populated from a real ``trials/T*.json`` spec.

    Renders nothing when no spec is featured (keeps the board uncluttered when
    the only data on disk is scored output).
    """
    h = s.hypothesis
    if h is None:
        return ""
    band = f"+{h.expected_band_lo:.3f} to +{h.expected_band_hi:.3f}"
    direction = "↑" if h.predicted_direction == "up" else ("↓" if h.predicted_direction == "down" else h.predicted_direction)
    sampler = f" · sampler_steps {h.sampler_steps}" if h.sampler_steps is not None else ""
    return (
        '<h2 class="block-title">Pre-registered hypothesis '
        f'<span class="right"><span class="spill info">pending</span></span></h2>'
        f'<p class="claim">{esc(h.claim)}</p>'
        f'<div class="claim-meta">trial <span class="mono">{esc(h.trial_id)}</span> · '
        f'target <span class="mono">{esc(h.diagnostic_target)}</span> · '
        f'move family <span class="mono">{esc(h.move_family)}</span> · '
        f'predicted axis <span class="mono">{esc(h.predicted_axis)}</span> '
        f'<span class="mono">{esc(direction)}</span></div>'
        '<div class="claim-meta" style="margin-top:8px">'
        f'expected Δ best_val_calpha_lddt <span class="mono num">{esc(band)}</span> · '
        f'budget <span class="mono">{esc(h.budget)}</span>{esc(sampler)} · '
        f'component <span class="mono">{esc(h.causal_component)}</span></div>'
        '<div class="readout">Claim and predicted axis are committed before the run. '
        'The falsification gate will check it against knock-out, placebo, and seed reruns once the trial scores.</div>'
    )


def gate_section(s: UiState) -> str:
    g = s.gate
    if g is None:
        return ""
    return (
        f'<h2 class="block-title">Headline discovery on trial '
        f'<span class="right"><span class="badge">{esc(g.verdict)}</span></span></h2>'
        f'<p class="claim">{esc(g.claim)}</p>'
        f'<div class="claim-meta">pre-registered axis <span class="mono">{esc(g.meta_axis)}</span> '
        f'· trial <span class="mono">{esc(g.meta_trial)}</span></div>'
        f"{components.control_bars(g.bars)}"
        f'<div class="readout"><span class="mono">{esc(g.attributable)}</span>. {esc(g.readout)}</div>'
    )


def overlay_section(s: UiState) -> str:
    o = s.overlay
    if o is None:
        return ""
    return (
        f'<h2 class="block-title">Structure overlay</h2>'
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
        f'· before/after lDDT <span class="mono num">{o.before:.3f} → {o.after:.3f}</span></div>'
        '</div></div>'
    )


# --- Trials view -----------------------------------------------------------

_FILTERS = [
    ("all", "All"), ("confirmed", "Confirmed"), ("keep", "KEEP"),
    ("discard", "DISCARD"), ("killed", "Killed"), ("fail", "Failed"),
    ("pending", "Pending"),
]


def trials_table(s: UiState) -> str:
    all_trials = list(s.trials) + list(s.pending_trials)
    rows = ""
    for t in all_trials:
        if t.tone == "muted":
            pill = f'<span class="spill muted">{esc(t.status)}</span>'
        else:
            icon = "✓" if t.cat in ("confirmed", "keep") else ("✕" if t.cat == "killed" else None)
            pill = status_pill(t.status, t.tone, icon)
        dcls = "c-d num" if t.cat in ("confirmed", "keep") else "c-d muted num"
        rows += (
            f'<tr data-status="{esc(t.cat)}">'
            f'<td class="c-mono">{esc(t.trial_id)}</td>'
            f"<td>{esc(t.move_family)}</td>"
            f'<td class="c-axis">{esc(t.axis)}</td>'
            f'<td class="r {dcls}">{esc(t.delta)}</td>'
            f'<td class="r num">{esc(t.runtime)}</td>'
            f'<td class="r">{pill}</td></tr>'
        )

    def count(cat: str) -> int:
        return len(all_trials) if cat == "all" else sum(1 for t in all_trials if t.cat == cat)

    chips = "".join(
        f'<button class="fbtn{" is-active" if cat == "all" else ""}" data-filter="{cat}" type="button">'
        f'{esc(label)}<span class="n">{count(cat)}</span></button>'
        for cat, label in _FILTERS
    )
    total = len(all_trials)
    scored = sum(1 for t in all_trials if t.score != "—")
    pending = sum(1 for t in all_trials if t.cat == "pending")
    parts = []
    if scored:
        parts.append(f"{scored} scored")
    if pending:
        parts.append(f"{pending} pending")
    cap = f"{total} trials ({', '.join(parts)})." if parts else f"{total} trials."
    return (
        '<h2 class="block-title">Trials</h2>'
        '<div class="block-sub">Every submitted trial. Filter by status; failures and kills stay visible.</div>'
        f'<div class="filter-bar">{chips}</div>'
        '<table class="dtable" id="trialsTable"><thead><tr>'
        '<th>Trial</th><th>Move family</th><th>Axis</th>'
        '<th class="r">Δ lDDT</th><th class="r">Runtime</th><th class="r">Status</th>'
        f"</tr></thead><tbody>{rows}</tbody></table>"
        f'<div class="led-cap">{esc(cap)}</div>'
    )


# --- Logs view -------------------------------------------------------------

_SEARCH_ICON = (
    '<svg viewBox="0 0 16 16" fill="none"><circle cx="7" cy="7" r="4.5" stroke="currentColor" stroke-width="1.4"/>'
    '<path d="M11 11l3 3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>'
)


def logs_feed(s: UiState) -> str:
    rows = ""
    for ev in s.logs:
        msgcls = f" {ev.level}" if ev.level in ("warn", "err") else ""
        text = f"{ev.time} {ev.trial} {ev.message}".lower()
        rows += (
            f'<div class="logrow" data-text="{esc(text)}">'
            f'<span class="logdot {esc(ev.level)}"></span>'
            f'<span class="logtime">{esc(ev.time) or "—"}</span>'
            f'<span class="logtrial">{esc(ev.trial)}</span>'
            f'<span class="logmsg{msgcls}">{esc(ev.message)}</span></div>'
        )
    toolbar = (
        '<div class="log-toolbar"><div class="log-search">'
        f"{_SEARCH_ICON}"
        '<input type="text" id="logSearch" placeholder="Search logs" aria-label="Search logs"></div></div>'
    )
    return (
        '<h2 class="block-title">Logs</h2>'
        '<div class="block-sub">Orchestrator and trial events for the run.</div>'
        f"{toolbar}"
        f'<div class="logfeed" id="logFeed">{rows}</div>'
    )
