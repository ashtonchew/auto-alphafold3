"""Render a small HTML trajectory view from the canonical JSONL ledger."""

from __future__ import annotations

import html
from pathlib import Path

from autoalphafold3.ledger import read_ledger
from autoalphafold3.schema import AutoFoldResult, TrialStatus

SAMPLE_NOTICE = "Sample/local dry-run data only. Not benchmark results."

STATUS_COLORS = {
    TrialStatus.PREFLIGHT_PASSED: "#5b7cfa",
    TrialStatus.SCORED: "#4d8b31",
    TrialStatus.KEEP: "#087f5b",
    TrialStatus.DISCARD: "#8a6d3b",
    TrialStatus.FAIL: "#b54708",
    TrialStatus.INFRA_FAIL: "#b42318",
    TrialStatus.ARCHIVED: "#667085",
}


def render_trajectory(
    ledger_path: str | Path,
    output_path: str | Path,
    *,
    sample: bool = False,
) -> Path:
    """Render `demo/trajectory.html` from a ledger or marked sample rows."""

    rows = _sample_rows() if sample else read_ledger(ledger_path=ledger_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_trajectory_html(rows, sample=sample), encoding="utf-8")
    return output


def _trajectory_html(rows: list[AutoFoldResult], *, sample: bool) -> str:
    points = _svg_points(rows)
    table_rows = "\n".join(_table_row(index, row) for index, row in enumerate(rows, start=1))
    latest_signature = rows[-1].fold_cartographer.signature if rows else "none"
    notice = SAMPLE_NOTICE if sample else "Ledger-backed trajectory."
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>auto-AlphaFold3 trajectory</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #172033; }}
    .notice {{ border: 1px solid #d0d5dd; background: #f8fafc; padding: 10px 12px; margin-bottom: 16px; }}
    svg {{ width: 100%; max-width: 880px; height: 320px; border: 1px solid #d0d5dd; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 980px; margin-top: 16px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #eaecf0; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <main>
    <h1>auto-AlphaFold3 trajectory</h1>
    <div class="notice">{html.escape(notice)} Latest Fold Cartographer signature: <strong>{html.escape(latest_signature)}</strong>.</div>
    <svg viewBox="0 0 880 320" role="img" aria-label="Trial trajectory plot">
      <line x1="48" y1="268" x2="836" y2="268" stroke="#98a2b3" />
      <line x1="48" y1="32" x2="48" y2="268" stroke="#98a2b3" />
      <text x="48" y="292" font-size="12">trial</text>
      <text x="8" y="36" font-size="12">lDDT</text>
      {points}
    </svg>
    <table>
      <thead><tr><th>#</th><th>Trial</th><th>Status</th><th>best_val_calpha_lddt</th><th>Signature</th><th>Postmortem</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </main>
</body>
</html>
"""


def _svg_points(rows: list[AutoFoldResult]) -> str:
    if not rows:
        return '<text x="72" y="152" font-size="14">No ledger rows available.</text>'
    max_score = max(float(row.metrics.get("best_val_calpha_lddt", 0.0) or 0.0) for row in rows) or 1.0
    step = 760 / max(1, len(rows) - 1)
    parts = []
    for index, row in enumerate(rows):
        score = float(row.metrics.get("best_val_calpha_lddt", 0.0) or 0.0)
        x = 62 + index * step
        y = 268 - (score / max_score) * 220
        color = STATUS_COLORS.get(row.status, "#344054")
        label = f"{row.trial_id} {row.status.value} {score:.3f}"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"><title>{html.escape(label)}</title></circle>'
        )
        parts.append(f'<text x="{x - 10:.1f}" y="286" font-size="10">{html.escape(row.trial_id)}</text>')
    return "\n      ".join(parts)


def _table_row(index: int, row: AutoFoldResult) -> str:
    score = row.metrics.get("best_val_calpha_lddt", "")
    return (
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{html.escape(row.trial_id)}</td>"
        f"<td>{html.escape(row.status.value)}</td>"
        f"<td>{html.escape(str(score))}</td>"
        f"<td>{html.escape(row.fold_cartographer.signature)}</td>"
        f"<td>{html.escape(row.postmortem)}</td>"
        "</tr>"
    )


def _sample_rows() -> list[AutoFoldResult]:
    return [
        AutoFoldResult(
            trial_id="T001",
            status=TrialStatus.PREFLIGHT_PASSED,
            candidate_id="sample_dry_run",
            metrics={"best_val_calpha_lddt": 0.0},
            fold_cartographer={"signature": "preflight_only", "summary": {}, "buckets": {}},
            postmortem="Sample preflight row; no benchmark run was performed.",
        ),
        AutoFoldResult(
            trial_id="T002",
            status=TrialStatus.FAIL,
            candidate_id="sample_candidate",
            metrics={"best_val_calpha_lddt": 0.18},
            fold_cartographer={"signature": "toy_geometry_failed", "summary": {}, "buckets": {}},
            postmortem="Sample failure row for trajectory rendering only.",
        ),
        AutoFoldResult(
            trial_id="T003",
            status=TrialStatus.KEEP,
            candidate_id="sample_candidate",
            metrics={"best_val_calpha_lddt": 0.42},
            fold_cartographer={"signature": "toy_geometry_preserved", "summary": {}, "buckets": {}},
            postmortem="Sample KEEP row for renderer smoke; not a real result.",
        ),
    ]
