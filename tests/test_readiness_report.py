from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autoalphafold3.modal_assets import ModalAssetAudit
from autoalphafold3.nanofold_checks import NanoFoldGateResult
from autoalphafold3.readiness import (
    CertificationStatus,
    HUMAN_ACTION_MARKER,
    LIVE_SMOKE_MARKER,
    build_readiness_report,
    readiness_exit_code,
)

SHA = "a" * 64
REPO_ROOT = Path(__file__).resolve().parents[1]


def write_baseline_lock(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    metrics = {
        "schema_version": "autoaf3.metrics.v1",
        "status": "SCORED",
        "trial_id": "baseline_auto_tiny",
        "candidate_id": "baseline_lock",
        "split": "public_val_small",
        "official_benchmark_result": True,
        "primary_metric": "best_val_calpha_lddt",
        "scorer_version": "calpha_lddt_v1",
        "max_templates": 0,
        "manifests": {"train_tiny": SHA, "public_val_small": SHA},
        "label_hashes": {"public_val_small": SHA},
        "metrics": {"best_val_calpha_lddt": 0.42},
        "fold_cartographer": {"signature": "baseline_locked", "summary": {}, "buckets": {}},
        "artifacts": {"metrics_json": "runs/baseline/metrics.json"},
    }
    (baseline / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (baseline / "error_report.json").write_text(json.dumps({"scorer_only": True}), encoding="utf-8")
    (baseline / "feature_fingerprints.json").write_text(
        json.dumps(
            {
                "files": {
                    "features/train_tiny.arrow": SHA,
                    "features/public_val_small.arrow": SHA,
                },
                "max_templates": 0,
            }
        ),
        encoding="utf-8",
    )
    return baseline


def write_calibration(tmp_path: Path) -> Path:
    path = tmp_path / "gate_calibration.json"
    path.write_text(
        json.dumps(
            {
                "known_null": calibration_record(
                    verdict="PLACEBO_KILL",
                    fixture_path=str(tmp_path / "known_null.json"),
                ),
                "known_positive": calibration_record(
                    verdict="CONFIRMED",
                    fixture_path=str(tmp_path / "known_positive.json"),
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


def calibration_record(*, verdict: str, fixture_path: str) -> dict[str, object]:
    return {
        "status": "complete",
        "verdict": verdict,
        "scorer_version": "calpha_lddt_v1",
        "primary_metric": "best_val_calpha_lddt",
        "split": "public_val_small",
        "baseline_id": "baseline_auto_tiny",
        "current_best_trial_id": "baseline_auto_tiny",
        "manifest_hashes": {"train_tiny": SHA, "public_val_small": SHA},
        "feature_fingerprints": {"train_tiny.arrow": SHA},
        "gate_thresholds": {"tau_attribution": 0.5, "rho_placebo": 0.5, "k_seed": 2.0},
        "control_evidence_ids": ["knockout", "placebo", "axis", "seed"],
        "synthetic_fixture": True,
        "fixture_path": fixture_path,
    }


def passed_gates() -> list[NanoFoldGateResult]:
    return [
        NanoFoldGateResult("parameter_count", "passed", "counted", {"parameter_count": 1}),
        NanoFoldGateResult("tiny_forward", "passed", "finite", {}),
        NanoFoldGateResult("finite_loss", "passed", "finite", {}),
    ]


def skipped_gates() -> list[NanoFoldGateResult]:
    return [
        NanoFoldGateResult("parameter_count", "passed", "counted", {"parameter_count": 1}),
        NanoFoldGateResult("tiny_forward", "skipped", "dependency_missing", {}),
        NanoFoldGateResult("finite_loss", "skipped", "feature_fixture_not_available_without_cached_arrow", {}),
    ]


def dependency_skipped_gates() -> list[NanoFoldGateResult]:
    return [
        NanoFoldGateResult("parameter_count", "passed", "counted", {"parameter_count": 1}),
        NanoFoldGateResult("tiny_forward", "skipped", "dependency_missing", {}),
        NanoFoldGateResult("finite_loss", "skipped", "dependency_missing", {}),
    ]


def unknown_skipped_gates() -> list[NanoFoldGateResult]:
    return [
        NanoFoldGateResult("parameter_count", "passed", "counted", {"parameter_count": 1}),
        NanoFoldGateResult("tiny_forward", "skipped", "unexpected_preflight_gap", {}),
        NanoFoldGateResult("finite_loss", "passed", "finite", {}),
    ]


def test_readiness_report_distinguishes_offline_and_live_actions(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    live_calls = 0

    def fake_live() -> ModalAssetAudit:
        nonlocal live_calls
        live_calls += 1
        return ModalAssetAudit(status="PASS", locked_asset_layout="separate_locked_volume", official_lock_boundary=True, target_layout="two_volume")

    offline = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path="missing.json",
        nanofold_gates=passed_gates(),
        modal_audit_runner=fake_live,
    )

    assert offline.mode == "offline"
    assert offline.live_smoke.status == "NOT_REQUESTED"
    assert offline.modal_event_authority.status == "PENDING"
    assert offline.modal_event_authority.certification_status == "PENDING_HUMAN_LIVE_ACTION"
    assert offline.autonomous_search_ready is False
    assert live_calls == 0

    live = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path=write_calibration(tmp_path).relative_to(tmp_path),
        nanofold_gates=passed_gates(),
        include_live_smoke=True,
        approved_live_smoke_action=f"{LIVE_SMOKE_MARKER} inspect Modal assets only",
        modal_audit_runner=fake_live,
    )

    assert live.mode == "live_smoke"
    assert live.live_smoke.status == "PASS"
    assert live.modal_event_authority.status == "PENDING"
    assert live.autonomous_search_ready is False
    assert live_calls == 1


def test_readiness_report_fails_when_baseline_missing(tmp_path: Path) -> None:
    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir="missing-baseline",
        calibration_path=write_calibration(tmp_path).relative_to(tmp_path),
        nanofold_gates=passed_gates(),
    )

    assert report.baseline_lock.status == "PENDING"
    assert report.baseline_lock.certification_status == "PENDING_HUMAN_LIVE_ACTION"
    assert report.autonomous_search_ready is False
    assert "baseline metrics.json is missing" in report.problems
    assert report.baseline_lock.details["baseline_score"] is None


def test_skipped_tiny_forward_and_finite_loss_block_live_readiness(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path=write_calibration(tmp_path).relative_to(tmp_path),
        nanofold_gates=skipped_gates(),
    )

    assert report.local_gates.status == "PENDING"
    assert report.local_gates.certification_status == "PENDING_HUMAN_LIVE_ACTION"
    assert report.local_gates.pending_human_action is not None
    assert report.local_gates.pending_human_action.startswith("Human-approved local gate action:")
    assert "runtime dependencies" in report.local_gates.pending_human_action
    assert "cached Arrow feature fixture" in report.local_gates.pending_human_action
    assert report.autonomous_search_ready is False
    assert "tiny_forward blocks live readiness" in " ".join(report.problems)
    assert "finite_loss blocks live readiness" in " ".join(report.problems)


def test_dependency_only_skipped_gates_report_dependency_action(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path=write_calibration(tmp_path).relative_to(tmp_path),
        nanofold_gates=dependency_skipped_gates(),
    )

    assert report.local_gates.status == "PENDING"
    assert report.local_gates.pending_human_action is not None
    assert report.local_gates.pending_human_action.startswith("Human-approved local dependency action:")
    assert "tiny_forward and finite_loss" in report.local_gates.pending_human_action


def test_unknown_skipped_gate_remains_blocked_until_action_is_defined(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path=write_calibration(tmp_path).relative_to(tmp_path),
        nanofold_gates=unknown_skipped_gates(),
    )

    assert report.local_gates.status == "FAIL"
    assert report.local_gates.certification_status == "BLOCKED"
    assert report.local_gates.pending_human_action is None
    assert "unexpected_preflight_gap" in " ".join(report.local_gates.problems)


def test_gate_calibration_placeholders_are_not_search_ready(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    calibration = tmp_path / "placeholder_calibration.json"
    calibration.write_text(
        json.dumps({"known_null": {"status": "placeholder"}, "known_positive": {"status": "placeholder"}}),
        encoding="utf-8",
    )

    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path=calibration.relative_to(tmp_path),
        nanofold_gates=passed_gates(),
    )

    assert report.gate_calibration.status == "FAIL"
    assert report.autonomous_search_ready is False
    assert "placeholder is not search-ready" in " ".join(report.problems)


def test_known_null_and_known_positive_complete_passes_from_tmp_fixtures_only(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    calibration = write_calibration(tmp_path)

    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path=calibration.relative_to(tmp_path),
        nanofold_gates=passed_gates(),
    )

    assert report.gate_calibration.status == "PASS"
    assert report.modal_event_authority.status == "PENDING"
    assert report.autonomous_search_ready is False
    assert "/runs/" not in json.dumps(report.gate_calibration.details)
    assert '"data/' not in json.dumps(report.gate_calibration.details)


def test_pending_exact_human_approved_calibration_action_is_allowed_but_blocks_search(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    action = f"{HUMAN_ACTION_MARKER} run known-null T000 and known-positive T001 via run_gate_control starmap"

    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path="missing.json",
        pending_human_calibration_action=action,
        nanofold_gates=passed_gates(),
    )

    assert report.gate_calibration.status == "PENDING"
    assert report.gate_calibration.pending_human_action == action
    assert report.autonomous_search_ready is False
    assert readiness_exit_code(report) == 2


def test_vague_pending_calibration_action_fails(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path="missing.json",
        pending_human_calibration_action="needs calibration",
        nanofold_gates=passed_gates(),
    )

    assert report.gate_calibration.status == "FAIL"
    assert "exact human-approved" in " ".join(report.problems)


def test_readiness_cli_json_reports_not_ready_without_side_effects(tmp_path: Path) -> None:
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "readiness-report",
            "--repo-root",
            str(tmp_path),
            "--baseline-dir",
            "missing-baseline",
            "--calibration-path",
            "missing-calibration.json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["autonomous_search_ready"] is False
    assert "live Modal" not in result.stderr
    assert before == after


def test_readiness_cli_forbids_live_without_explicit_approval(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "readiness-report",
            "--repo-root",
            str(tmp_path),
            "--include-live-smoke",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode != 0
    assert payload["live_smoke"]["status"] == "PENDING"
    assert "human-approved" in " ".join(payload["problems"])


def test_readiness_report_does_not_write_locked_or_canonical_artifacts(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    calibration = write_calibration(tmp_path)
    watched = [
        tmp_path / "runs" / "baseline" / "metrics.json",
        tmp_path / "runs" / "ledger.jsonl",
        tmp_path / "runs" / "discovery_ledger.jsonl",
        tmp_path / "runs" / "discovery" / "T300.json",
        tmp_path / "runs" / "gate_wave" / "T300.json",
        tmp_path / "runs" / "benchmark" / "artifact_manifest.json",
    ]

    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path=calibration.relative_to(tmp_path),
        nanofold_gates=passed_gates(),
    )

    assert report.autonomous_search_ready is False
    assert all(not path.exists() for path in watched)


def test_readiness_report_serializes_canonical_certification_statuses(tmp_path: Path) -> None:
    baseline = write_baseline_lock(tmp_path)
    action = f"{HUMAN_ACTION_MARKER} run known-null T000 and known-positive T001 via run_gate_control starmap"

    report = build_readiness_report(
        repo_root=tmp_path,
        baseline_dir=baseline.relative_to(tmp_path),
        calibration_path="missing.json",
        pending_human_calibration_action=action,
        nanofold_gates=passed_gates(),
    )
    payload = report.to_dict()
    allowed = {status.value for status in CertificationStatus}
    observed = {
        payload["baseline_lock"]["certification_status"],
        payload["mocked_modal_contract"]["certification_status"],
        payload["modal_event_authority"]["certification_status"],
        payload["local_gates"]["certification_status"],
        payload["gate_calibration"]["certification_status"],
        payload["live_smoke"]["certification_status"],
    }

    assert observed <= allowed
    assert payload["gate_calibration"]["certification_status"] == "PENDING_HUMAN_LIVE_ACTION"
    assert payload["modal_event_authority"]["certification_status"] == "PENDING_HUMAN_LIVE_ACTION"
    assert payload["baseline_lock"]["certification_status"] == "PASS_LOCAL"
    assert payload["mocked_modal_contract"]["certification_status"] == "PASS_MOCKED_MODAL"
    assert payload["local_gates"]["certification_status"] == "PASS_LOCAL"
    assert payload["live_smoke"]["certification_status"] == "NOT_REQUESTED"
    assert payload["certification_counts"]["PENDING_HUMAN_LIVE_ACTION"] == 2
    assert payload["certification_counts"]["PASS_MOCKED_MODAL"] == 1
    assert report.autonomous_search_ready is False


def test_readiness_cli_serializes_exact_human_approval_commands(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "readiness-report",
            "--repo-root",
            str(tmp_path),
            "--baseline-dir",
            "missing-baseline",
            "--calibration-path",
            "missing-calibration.json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["modal_event_authority"]["pending_human_action"].startswith("Human-approved Modal event authority:")
    assert payload["gate_calibration"]["pending_human_action"].startswith(HUMAN_ACTION_MARKER)
    assert payload["live_smoke"]["pending_human_action"].startswith(LIVE_SMOKE_MARKER)
    assert "do not write baseline, ledger, Discovery Ledger, benchmark, or metric artifacts" in payload[
        "gate_calibration"
    ]["pending_human_action"]
    assert "Volume-write" in payload["live_smoke"]["pending_human_action"]
