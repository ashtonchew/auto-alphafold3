"""Render C-alpha trace overlays as PDB-like traces and HTML."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

Coordinate = tuple[float, float, float]
SAMPLE_NOTICE = "Sample/local C-alpha traces only. Not benchmark results."


def ca_trace_to_pdb(coordinates: Iterable[Coordinate], *, chain_id: str = "A") -> str:
    """Convert C-alpha coordinates into a minimal PDB string."""

    lines = []
    for index, (x, y, z) in enumerate(coordinates, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  ALA {chain_id}{index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def render_overlay(
    output_path: str | Path,
    *,
    true_ca: Iterable[Coordinate],
    baseline_ca: Iterable[Coordinate],
    best_ca: Iterable[Coordinate],
    sample: bool = False,
) -> Path:
    """Render an HTML overlay page with embedded C-alpha PDB traces."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _overlay_html(
            true_pdb=ca_trace_to_pdb(true_ca, chain_id="A"),
            baseline_pdb=ca_trace_to_pdb(baseline_ca, chain_id="B"),
            best_pdb=ca_trace_to_pdb(best_ca, chain_id="C"),
            sample=sample,
        ),
        encoding="utf-8",
    )
    return output


def render_sample_overlay(output_path: str | Path) -> Path:
    """Render a tiny sample overlay for local demo smoke checks."""

    true_ca = [(0.0, 0.0, 0.0), (1.2, 0.1, 0.0), (2.4, 0.0, 0.2), (3.6, -0.1, 0.0)]
    baseline_ca = [(0.0, 0.0, 0.0), (1.0, 0.6, 0.0), (2.2, 0.8, 0.5), (3.1, 0.9, 0.2)]
    best_ca = [(0.0, 0.0, 0.0), (1.1, 0.2, 0.0), (2.35, 0.15, 0.2), (3.5, 0.0, 0.05)]
    return render_overlay(
        output_path,
        true_ca=true_ca,
        baseline_ca=baseline_ca,
        best_ca=best_ca,
        sample=True,
    )


def _overlay_html(*, true_pdb: str, baseline_pdb: str, best_pdb: str, sample: bool) -> str:
    notice = SAMPLE_NOTICE if sample else "Ledger-backed structure overlay."
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>auto-AlphaFold3 C-alpha overlay</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #172033; }}
    .notice {{ border: 1px solid #d0d5dd; background: #f8fafc; padding: 10px 12px; margin-bottom: 16px; }}
    #viewer {{ width: min(960px, 100%); height: 560px; border: 1px solid #d0d5dd; }}
    pre {{ max-width: 960px; overflow: auto; background: #101828; color: #f9fafb; padding: 12px; }}
  </style>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
  <main>
    <h1>auto-AlphaFold3 C-alpha overlay</h1>
    <div class="notice">{html.escape(notice)}</div>
    <div id="viewer"></div>
    <p>True trace: green. Baseline trace: amber. Best/sample candidate: blue.</p>
    <script>
      const truePdb = `{_js_string(true_pdb)}`;
      const baselinePdb = `{_js_string(baseline_pdb)}`;
      const bestPdb = `{_js_string(best_pdb)}`;
      const viewer = $3Dmol.createViewer('viewer', {{ backgroundColor: 'white' }});
      viewer.addModel(truePdb, 'pdb');
      viewer.setStyle({{model: 0}}, {{cartoon: {{color: 'green'}}}});
      viewer.addModel(baselinePdb, 'pdb');
      viewer.setStyle({{model: 1}}, {{cartoon: {{color: 'orange'}}}});
      viewer.addModel(bestPdb, 'pdb');
      viewer.setStyle({{model: 2}}, {{cartoon: {{color: 'royalblue'}}}});
      viewer.zoomTo();
      viewer.render();
    </script>
    <details><summary>Embedded true C-alpha PDB trace</summary><pre>{html.escape(true_pdb)}</pre></details>
  </main>
</body>
</html>
"""


def _js_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("`", "\\`")
