"""Tests for the demo UI renderer (autoalphafold3.ui).

Local and offline: no Modal, no GPU, no hidden validation. Builds from the
illustrative sample and from a synthetic ledger, and checks that real-derived
values reach the HTML.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoalphafold3.ui import load_state, sample_state
from autoalphafold3.ui.build import build
from autoalphafold3.ui.page import render_board


def _traj_points(html: str) -> list[dict]:
    marker = "window.TRAJ_POINTS = "
    start = html.index(marker) + len(marker)
    end = html.index(";</script>", start)
    return json.loads(html[start:end])


def test_sample_board_renders_key_values() -> None:
    state = sample_state()
    html = render_board(state)
    for needle in ("0.412", "+0.087", "baseline 0.325", "CONFIRMED", "PLACEBO_KILL", 'id="trajChart"', "Sample"):
        assert needle in html, needle
    assert len(_traj_points(html)) == len(state.trajectory) == 12


def test_ui_state_json_contract() -> None:
    payload = sample_state().to_json()
    assert payload["best_val_calpha_lddt"] == 0.412
    assert payload["is_sample"] is True
    assert payload["counts"]["confirmed"] == 7
    assert len(payload["trajectory"]) == 12


def test_load_state_from_real_ledger(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    rows = [
        {
            "trial_id": "T001", "status": "SCORED", "candidate_id": "c1",
            "fold_cartographer": {"signature": "baseline"},
            "metrics": {"best_val_calpha_lddt": 0.331, "scorer_version": "calpha_lddt_v1", "split": "public_val_small"},
        },
        {
            "trial_id": "T002", "status": "KEEP", "candidate_id": "c2",
            "fold_cartographer": {"signature": "geometry_loss"},
            "metrics": {"best_val_calpha_lddt": 0.392},
        },
    ]
    (runs / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (runs / "baseline").mkdir()
    (runs / "baseline" / "metrics.json").write_text(json.dumps({"best_val_calpha_lddt": 0.325}), encoding="utf-8")

    state = load_state(runs)
    assert state.is_sample is False
    assert state.geometry_sample is True  # geometry panels still sample until artifacts land
    assert state.best == 0.392
    assert state.baseline == 0.325
    assert state.counts["trials"] == 2
    assert state.counts["confirmed"] == 1  # the KEEP row
    assert [p.trial_id for p in state.trajectory] == ["T001", "T002"]

    html = render_board(state)
    assert "0.392" in html
    assert "baseline 0.325" in html
    assert "sample" in html  # geometry badge present on the live board


def test_no_ledger_falls_back_to_sample(tmp_path: Path) -> None:
    state = load_state(tmp_path / "empty-runs")
    assert state.is_sample is True
    assert state.best == 0.412


def test_build_writes_outputs(tmp_path: Path) -> None:
    out = build(tmp_path / "ui", sample=True)
    assert (out / "index.html").exists()
    payload = json.loads((out / "ui_state.json").read_text(encoding="utf-8"))
    assert payload["best_val_calpha_lddt"] == 0.412
    # design system is copied from the repo if present
    assert (out / "assets" / "modal.css").exists()
