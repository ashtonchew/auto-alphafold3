"""Design-system primitives.

Each function returns a self-contained HTML fragment that uses the shared
``modal.css`` classes. This is where the *look* of each element lives — change a
status pill or a chart here and every panel updates. No state, no IO.
"""

from __future__ import annotations

import html as _html
import json
import math

GREEN = "#7fee64"
AMBER = "#f3cf58"


def esc(value: object) -> str:
    return _html.escape(str(value), quote=True)


def chip(value: object, label: str, dot: str | None = None) -> str:
    d = f'<span class="dot {esc(dot)}"></span>' if dot else ""
    return f'<span class="chip">{d}<span class="v num">{esc(value)}</span><span class="k">{esc(label)}</span></span>'


def stat(label: str, value: object, sub: str, tone: str = "") -> str:
    t = f" {esc(tone)}" if tone else ""
    return (
        f'<div class="stat"><div class="label">{esc(label)}</div>'
        f'<div class="value{t} num">{esc(value)}</div>'
        f'<div class="sub num">{esc(sub)}</div></div>'
    )


def status_pill(text: str, tone: str, icon: str | None = None) -> str:
    ic = f'<span class="pico">{esc(icon)}</span> ' if icon else ""
    return f'<span class="spill {esc(tone)}">{ic}{esc(text)}</span>'


def sparkline(values: list[float]) -> str:
    """Small best-so-far trend line for the metric band."""
    if not values:
        values = [0.0, 1.0]
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    n = len(values)
    pts = " ".join(
        f"{(3 + (i / (n - 1) * 181 if n > 1 else 0)):.0f},{(36 - (v - lo) / rng * 32):.0f}"
        for i, v in enumerate(values)
    )
    last_y = 36 - (values[-1] - lo) / rng * 32
    return (
        '<svg class="spark" viewBox="0 0 188 40" role="img" aria-label="Best-so-far score trend.">'
        f'<polyline class="trace" style="stroke-width:2" points="{pts}"/>'
        f'<circle cx="184" cy="{last_y:.0f}" r="3" fill="{GREEN}"/></svg>'
    )


def trajectory_chart(points: list, baseline: float | None) -> tuple[str, str]:
    """Return ``(svg, traj_json)``. ``traj_json`` feeds ``window.TRAJ_POINTS`` so
    ``board.js`` can attach hover/click to the same coordinates."""
    scores = [p.score for p in points] + ([baseline] if baseline is not None else [])
    smin, smax = (min(scores), max(scores)) if scores else (0.30, 0.42)
    # Auto-pick a tick step so the Y axis fits the data range tightly.
    raw_span = max(smax - smin, 0.001)
    for step in (0.002, 0.005, 0.01, 0.02, 0.04, 0.05, 0.1):
        if raw_span / step <= 5:
            break
    lo = math.floor((smin - step * 0.25) / step) * step
    hi = math.ceil((smax + step * 0.25) / step) * step
    if hi - lo < step * 3:
        hi = lo + step * 3
    span = hi - lo

    def y(s: float) -> float:
        return 300 - (s - lo) / span * 260

    n = len(points)

    def x(i: int) -> float:
        return 60 + (i / (n - 1) * 640 if n > 1 else 0)

    parts: list[str] = []
    n_ticks = max(3, round(span / step))
    tick_fmt = ".3f" if step < 0.01 else (".3f" if step < 0.04 else ".2f")
    for k in range(n_ticks + 1):
        val = lo + k * step
        yy = y(val)
        if 0 < k < n_ticks:
            parts.append(f'<line class="grid-line" x1="60" y1="{yy:.0f}" x2="700" y2="{yy:.0f}"/>')
        parts.append(f'<text class="ax-tick num" x="50" y="{yy + 4:.0f}" text-anchor="end">{val:{tick_fmt}}</text>')
    parts.append('<line class="ax-line" x1="60" y1="40" x2="60" y2="300"/>')
    parts.append('<line class="ax-line" x1="60" y1="300" x2="700" y2="300"/>')
    if baseline is not None:
        by = y(baseline)
        parts.append(f'<line class="base-line" x1="60" y1="{by:.0f}" x2="700" y2="{by:.0f}"/>')
        parts.append(f'<text class="ax-tick num" x="696" y="{by - 7:.0f}" text-anchor="end">baseline {baseline:.3f}</text>')
    if points:
        line = " ".join(f"{x(i):.0f},{y(p.score):.0f}" for i, p in enumerate(points))
        parts.append(f'<polyline class="trace" points="{line}"/>')
    traj: list[dict] = []
    for i, p in enumerate(points):
        px, py = x(i), y(p.score)
        traj.append({"x": round(px), "y": round(py), "t": p.trial_id, "s": round(p.score, 3), "st": p.status})
        if p.status == "confirmed":
            parts.append(f'<circle class="dot-confirmed" cx="{px:.0f}" cy="{py:.0f}" r="5.5"/>')
        elif p.status == "provisional":
            parts.append(f'<rect class="dot-prov" x="{px - 5:.0f}" y="{py - 5:.0f}" width="10" height="10" rx="2"/>')
        elif p.status == "killed":
            parts.append(f'<path class="kill" d="M{px - 5:.0f} {py - 5:.0f} L{px + 5:.0f} {py + 5:.0f} M{px + 5:.0f} {py - 5:.0f} L{px - 5:.0f} {py + 5:.0f}"/>')
        else:  # fail / infra
            parts.append(f'<path class="kill" style="stroke:{AMBER}" d="M{px - 5:.0f} {py - 5:.0f} L{px + 5:.0f} {py + 5:.0f} M{px + 5:.0f} {py - 5:.0f} L{px - 5:.0f} {py + 5:.0f}"/>')
    if points:
        parts.append(f'<text class="ax-label num" x="60" y="318" text-anchor="middle">{esc(points[0].trial_id)}</text>')
        parts.append('<text class="ax-label" x="380" y="318" text-anchor="middle">trial →</text>')
        parts.append(f'<text class="ax-label num" x="700" y="318" text-anchor="middle">{esc(points[-1].trial_id)}</text>')
    svg = (
        f'<svg id="trajChart" class="chart" viewBox="0 0 720 330" role="img" '
        f'aria-label="Validation C-alpha lDDT across {n} trials.">' + "".join(parts) + "</svg>"
    )
    return svg, json.dumps(traj)


def control_bars(bars: list) -> str:
    """Falsification control bars (full / knock-out / placebo / seed)."""
    if not bars:
        return ""
    maxv = max((b.value for b in bars), default=1.0) or 1.0
    parts = ['<line class="ax-line" x1="40" y1="150" x2="510" y2="150"/>']
    for i, b in enumerate(bars):
        bx = 58 + i * 120
        h = b.value / maxv * 120
        by = 150 - h
        cls = "bar-full" if b.full else "bar-mute"
        parts.append(f'<rect class="{cls}" x="{bx}" y="{by:.0f}" width="74" height="{h:.0f}"/>')
        parts.append(f'<text class="bar-val num" x="{bx + 37}" y="{by - 8:.0f}" text-anchor="middle">{b.value:+.3f}</text>')
        parts.append(f'<text class="bar-lbl" x="{bx + 37}" y="168" text-anchor="middle">{esc(b.label)}</text>')
        if b.note:
            parts.append(f'<text class="bar-note" x="{bx + 37}" y="182" text-anchor="middle">{esc(b.note)}</text>')
    return '<svg class="bars" viewBox="0 0 520 188" role="img" aria-label="Falsification control bars.">' + "".join(parts) + "</svg>"


def err_strip(levels: list[int]) -> str:
    cells = "".join(f'<i class="e{max(0, min(4, int(level)))}"></i>' for level in levels)
    return f'<div class="err-strip" aria-hidden="true">{cells}</div>'


def backbone_overlay() -> str:
    """Sample C-alpha backbone overlay (true / baseline / best)."""
    return (
        '<svg viewBox="0 0 520 150" role="img" width="100%" style="max-width:660px" '
        'aria-label="Predicted vs true C-alpha backbone overlay.">'
        '<path d="M16 110 C60 54, 96 54, 120 96 S176 138, 214 100 C250 66, 286 70, 318 106 S388 138, 430 88 C462 50, 492 62, 506 46" fill="none" stroke="#dcdcdc" stroke-width="2.4" stroke-linecap="round"/>'
        '<path d="M16 116 C58 66, 98 62, 124 102 S172 146, 210 112 C250 84, 290 84, 322 114 S386 144, 428 100 C460 68, 490 78, 506 60" fill="none" stroke="#5896f3" stroke-width="2" stroke-linecap="round" opacity="0.85"/>'
        '<path d="M16 111 C60 56, 96 56, 121 97 S176 139, 213 102 C250 69, 287 72, 319 107 S388 138, 430 90 C462 54, 492 67, 506 50" fill="none" stroke="#7fee64" stroke-width="2.2" stroke-linecap="round"/>'
        "</svg>"
    )
