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
from autoalphafold3.ui.page import render_board, render_logs, render_trials


def _traj_points(html: str) -> list[dict]:
    marker = "window.TRAJ_POINTS = "
    start = html.index(marker) + len(marker)
    end = html.index(";</script>", start)
    return json.loads(html[start:end])


def test_sample_board_renders_key_values() -> None:
    state = sample_state()
    html = render_board(state)
    for needle in ("0.343", "+0.018", "baseline 0.325", "CONFIRMED", "sampler", 'id="trajChart"', "Demo board"):
        assert needle in html, needle
    assert len(_traj_points(html)) == len(state.trajectory) == 20


def test_ui_state_json_contract() -> None:
    payload = sample_state().to_json()
    assert payload["best_val_calpha_lddt"] == 0.343
    assert payload["is_sample"] is True
    assert payload["counts"]["confirmed"] == 1
    assert len(payload["trajectory"]) == 20


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
    assert state.best == 0.392
    assert state.baseline == 0.325
    assert state.counts["trials"] == 2
    assert state.counts["confirmed"] == 1  # the KEEP row
    assert [p.trial_id for p in state.trajectory] == ["T001", "T002"]
    # No real falsification, discovery_ledger, or predictions → those sections hide.
    assert state.gate is None
    assert state.overlay is None
    assert state.show_ledger is False

    html = render_board(state)
    assert "0.392" in html
    assert "baseline 0.325" in html
    assert "sample" not in html.lower() or "sampler" in html.lower()  # no 'sample' badge


def test_no_ledger_falls_back_to_sample(tmp_path: Path) -> None:
    state = load_state(tmp_path / "empty-runs")
    assert state.is_sample is True
    assert state.best == 0.343


def test_real_falsification_populates_gate(tmp_path: Path) -> None:
    """When a ledger row carries falsification evidence, the gate panel must
    show real bars (gain_full / knock-out / placebo / seed mean) — not sample."""
    runs = tmp_path / "runs"
    runs.mkdir()
    row = {
        "trial_id": "T012",
        "status": "KEEP",
        "candidate_id": "cand_31",
        "fold_cartographer": {"signature": "distogram_good_lddt_flat"},
        "metrics": {"best_val_calpha_lddt": 0.343, "scorer_version": "calpha_lddt_v1"},
        "discovery": "CONFIRMED",
        "falsification": {
            "gain_full": 0.018,
            "gain_knockout": 0.004,
            "gain_placebo": 0.002,
            "attributable_fraction": 0.74,
            "axis_delta_observed": 0.05,
            "axis_prediction_held": True,
            "seed_mean": 0.016,
            "seed_std": 0.004,
            "verdict": "CONFIRMED",
        },
    }
    (runs / "ledger.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    state = load_state(runs)
    assert state.gate is not None
    assert state.gate.verdict == "CONFIRMED"
    assert state.gate.meta_trial == "T012"
    # bar values came from the real falsification record
    assert state.gate.bars[0].value == 0.018  # full
    assert state.gate.bars[1].value == 0.004  # knock-out
    assert state.gate.bars[2].value == 0.002  # placebo
    assert state.gate.bars[3].value == 0.016  # seed mean


def test_per_trial_metrics_fallback_when_no_ledger(tmp_path: Path) -> None:
    """If runs/ledger.jsonl is missing, the UI should still surface real trials
    from per-trial metrics.json files instead of silently falling back to the sample."""
    runs = tmp_path / "runs"
    (runs / "trials" / "T000").mkdir(parents=True)
    (runs / "trials" / "T001").mkdir(parents=True)
    (runs / "trials" / "T000" / "metrics.json").write_text(json.dumps({
        "trial_id": "T000",
        "status": "SCORED",
        "candidate_id": "baseline_auto_tiny",
        "fold_cartographer": {"signature": "toy_geometry_failed"},
        "metrics": {"best_val_calpha_lddt": 0.0794, "scorer_version": "calpha_lddt_v1",
                    "split": "public_val_small"},
    }), encoding="utf-8")
    (runs / "trials" / "T001" / "metrics.json").write_text(json.dumps({
        "trial_id": "T001", "status": "SCORED", "candidate_id": "c1",
        "fold_cartographer": {"signature": "geometry_loss"},
        "metrics": {"best_val_calpha_lddt": 0.12},
    }), encoding="utf-8")
    # baseline file in the real AutoFoldResult-shape (metric nested under "metrics")
    (runs / "baseline").mkdir()
    (runs / "baseline" / "metrics.json").write_text(json.dumps({
        "metrics": {"best_val_calpha_lddt": 0.08}}), encoding="utf-8")

    state = load_state(runs)
    assert state.is_sample is False
    assert "per-trial metrics" in state.source
    assert state.best == 0.12
    assert state.baseline == 0.08
    assert [p.trial_id for p in state.trajectory] == ["T000", "T001"]
    assert state.counts["trials"] == 2


def test_build_writes_outputs(tmp_path: Path) -> None:
    out = build(tmp_path / "ui", sample=True)
    payload = json.loads((out / "ui_state.json").read_text(encoding="utf-8"))
    assert payload["best_val_calpha_lddt"] == 0.343
    # all pages + design system
    for name in ("index.html", "trials.html", "logs.html", "assets/modal.css"):
        assert (out / name).exists(), name


def test_trials_view_renders() -> None:
    html = render_trials(sample_state())
    for needle in ("Trials", 'id="trialsTable"', "sampler_step_scale", "diffusion_steps", 'data-filter="killed"', 'href="index.html"'):
        assert needle in html, needle


def test_logs_view_renders() -> None:
    html = render_logs(sample_state())
    for needle in ('id="logFeed"', "best_val_calpha_lddt 0.343", "sampler burst", 'id="logSearch"', 'href="logs.html"'):
        assert needle in html, needle


def test_real_ledger_populates_trials_and_logs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    rows = [
        {"trial_id": "T001", "status": "SCORED", "candidate_id": "c1",
         "fold_cartographer": {"signature": "baseline"},
         "metrics": {"best_val_calpha_lddt": 0.33}},
        {"trial_id": "T002", "status": "KEEP", "candidate_id": "c2",
         "fold_cartographer": {"signature": "good"},
         "metrics": {"best_val_calpha_lddt": 0.39, "runtime_seconds": 492}},
    ]
    (runs / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    state = load_state(runs)
    assert [t.trial_id for t in state.trials] == ["T001", "T002"]
    assert any(t.runtime == "8m 12s" for t in state.trials)  # 492s formatted
    assert any(e.message.startswith("scored") for e in state.logs)


def test_load_state_from_autoresearch_summary_without_fake_scores(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "autoresearch" / "ui-smoke"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.autoresearch_run_manifest.v1",
                "run_id": "ui-smoke",
                "planner": "deterministic",
                "mode": "dry-run",
                "official_benchmark_result": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "autoaf3.autoresearch_summary.v1",
                "run_id": "ui-smoke",
                "official_benchmark_result": False,
                "candidates": [
                    {
                        "trial_id": "T120",
                        "status": "DRAFT",
                        "planning_status": "PLANNED",
                        "candidate_id": "T120",
                        "decision_path": None,
                        "postmortem_path": None,
                        "matched_budget_delta": None,
                        "global_baseline_delta": None,
                        "provisional_keep": False,
                    },
                    {
                        "trial_id": "T121",
                        "status": "KEEP",
                        "candidate_id": "T121",
                        "decision_path": str(run / "candidates/T121/decision.json"),
                        "postmortem_path": str(run / "candidates/T121/postmortem.md"),
                        "matched_budget_delta": 0.004,
                        "global_baseline_delta": None,
                        "provisional_keep": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "results.tsv").write_text("trial_id\tcandidate_id\tstatus\nT120\tT120\tPLANNED\n", encoding="utf-8")

    state = load_state(runs)
    assert state.is_sample is False
    assert state.best is None
    assert state.baseline is None
    assert state.counts["trials"] == 2
    assert state.counts["keep"] == 1
    assert state.show_ledger is False
    assert state.autoresearch_runs[0].run_id == "ui-smoke"
    assert [t.trial_id for t in state.trials] == ["T120", "T121"]
    assert state.trials[0].status == "PLANNED"
    assert state.trials[0].cat == "pending"
    assert any(t.status == "PROVISIONAL KEEP" for t in state.trials)

    html = render_board(state)
    assert "Autoresearch evidence" in html
    assert "Official benchmark result: false" in html
    assert "T120" in html and "T121" in html
    assert "0.343" not in html
    assert "baseline 0.325" not in html
    assert '<h2 class="block-title">Discovery Ledger</h2>' not in html


def test_build_outputs_autoresearch_ui_state(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "autoresearch" / "build-smoke"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"run_id": "build-smoke", "planner": "llm", "mode": "dry-run", "official_benchmark_result": False}),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "build-smoke",
                "official_benchmark_result": False,
                "candidates": [
                    {
                        "trial_id": "T200",
                        "status": "DRAFT",
                        "planning_status": "PLANNED",
                        "candidate_id": "T200",
                        "decision_path": None,
                        "postmortem_path": None,
                        "matched_budget_delta": None,
                        "global_baseline_delta": None,
                        "provisional_keep": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out = build(tmp_path / "ui", runs_dir=runs)
    payload = json.loads((out / "ui_state.json").read_text(encoding="utf-8"))
    assert payload["is_sample"] is False
    assert payload["best_val_calpha_lddt"] is None
    assert payload["autoresearch_runs"][0]["run_id"] == "build-smoke"
    assert payload["autoresearch_runs"][0]["official_benchmark_result"] is False
    assert payload["autoresearch_runs"][0]["candidates"][0]["status"] == "DRAFT"
    assert payload["autoresearch_runs"][0]["candidates"][0]["planning_status"] == "PLANNED"
    assert "build-smoke" in (out / "index.html").read_text(encoding="utf-8")
    assert "T200" in (out / "trials.html").read_text(encoding="utf-8")


def test_autoresearch_rows_append_to_scored_trials_and_logs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "ledger.jsonl").write_text(
        json.dumps(
            {
                "trial_id": "T001",
                "status": "KEEP",
                "candidate_id": "scored",
                "fold_cartographer": {"signature": "good"},
                "metrics": {"best_val_calpha_lddt": 0.39},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run = runs / "autoresearch" / "mixed"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"run_id": "mixed", "planner": "deterministic", "mode": "dry-run"}),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "mixed",
                "candidates": [
                    {
                        "trial_id": "T130",
                        "status": "DRAFT",
                        "planning_status": "PLANNED",
                        "candidate_id": "T130",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state = load_state(runs)
    assert state.best == 0.39
    assert [t.trial_id for t in state.trials] == ["T001", "T130"]
    assert state.pending_trials == []
    assert state.counts["trials"] == 2
    assert state.counts["keep"] == 1
    assert any(event.trial == "T130" and event.message.startswith("PLANNED") for event in state.logs)
    assert '<span class="v num">2</span><span class="k">candidates</span>' in render_board(state)
    assert "2 trials (1 scored, 1 pending)." in render_trials(state)


def test_autoresearch_summary_rejects_symlinked_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run_manifest.json").write_text(json.dumps({"run_id": "outside"}), encoding="utf-8")
    (outside / "summary.json").write_text(
        json.dumps({"run_id": "outside", "candidates": [{"trial_id": "T999", "status": "KEEP"}]}),
        encoding="utf-8",
    )
    autoresearch = runs / "autoresearch"
    autoresearch.mkdir(parents=True)
    (autoresearch / "linked").symlink_to(outside, target_is_directory=True)

    state = load_state(runs)
    assert not state.autoresearch_runs
    assert all(t.trial_id != "T999" for t in state.trials)


def test_autoresearch_summary_ignores_symlinked_manifest(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "autoresearch" / "local-run"
    run.mkdir(parents=True)
    outside = tmp_path / "outside-manifest.json"
    outside.write_text(json.dumps({"planner": "leaked", "mode": "external"}), encoding="utf-8")
    (run / "run_manifest.json").symlink_to(outside)
    (run / "summary.json").write_text(
        json.dumps({"run_id": "local-run", "candidates": [{"trial_id": "T220", "status": "DRAFT", "planning_status": "PLANNED"}]}),
        encoding="utf-8",
    )

    state = load_state(runs)
    assert state.autoresearch_runs[0].planner == "unknown"
    assert state.autoresearch_runs[0].mode == "unknown"
    assert "leaked" not in render_board(state)


def test_autoresearch_summary_cannot_promote_official_claims(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "autoresearch" / "bad-authority"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"run_id": "bad-authority", "planner": "llm", "mode": "dry-run", "official_benchmark_result": True}),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "bad-authority",
                "official_benchmark_result": True,
                "candidates": [{"trial_id": "T210", "status": "DRAFT", "planning_status": "PLANNED"}],
            }
        ),
        encoding="utf-8",
    )

    state = load_state(runs)
    assert state.autoresearch_runs[0].official_benchmark_result is False
    assert state.to_json()["autoresearch_runs"][0]["official_benchmark_result"] is False
    assert "Official benchmark result: false" in render_board(state)


def test_autoresearch_summary_escapes_rendered_fields(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "autoresearch" / "xss"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"run_id": "xss", "planner": "<script>alert(1)</script>", "mode": "dry-run"}),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "xss",
                "candidates": [
                    {
                        "trial_id": "<img src=x onerror=alert(1)>",
                        "status": "<script>alert(2)</script>",
                        "candidate_id": "bad",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = render_board(load_state(runs))
    assert "<script>alert" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;alert" in html
