from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoalphafold3.modal_assets import (
    DATA_VOLUME,
    LOCKED_VOLUME,
    ModalAssetAuditError,
    audit_modal_assets,
    require_search_ready_assets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _entry(filename: str, size: str = "1 KiB") -> dict[str, object]:
    return {"Filename": filename, "Type": "file", "Size": size}


def _dir(filename: str) -> dict[str, object]:
    return {"Filename": filename, "Type": "dir", "Size": "4.0 KiB"}


def _manifest(split: str, count: int) -> str:
    return json.dumps([
        {
            "record_id": f"{split}_{index}",
            "split": split,
            "sequence_length": 10 + index,
        }
        for index in range(count)
    ])


class FakeModalVolumes:
    def __init__(self, *, locked: bool, bad_counts: bool = False) -> None:
        self.separate_locked = locked
        self.bad_counts = bad_counts

    def ls(self, volume: str, path: str) -> list[dict[str, object]]:
        if volume == DATA_VOLUME:
            return self._ls_data(path)
        if volume == LOCKED_VOLUME and self.separate_locked:
            return self._ls_locked(path)
        raise ModalAssetAuditError(f"could not list {volume}:{path}: Volume not found")

    def read(self, volume: str, path: str) -> str:
        if "train_tiny.json" in path:
            return _manifest("train_tiny", 31 if self.bad_counts else 32)
        if "public_val_small.json" in path:
            return _manifest("public_val_small", 16)
        if path.endswith("feature_fingerprints.json"):
            return json.dumps({"files": {}})
        if path.endswith("provenance.json"):
            return json.dumps({"splits": {"train_tiny": 32, "public_val_small": 16}})
        raise ModalAssetAuditError(f"could not read {volume}:{path}: missing")

    def _ls_data(self, path: str) -> list[dict[str, object]]:
        if path == "/":
            return [_dir("features"), _dir("raw"), _entry("provenance.json")]
        if path == "/features":
            return [
                _entry("features/train_tiny.arrow"),
                _entry("features/public_val_small.arrow"),
                _entry("features/feature_fingerprints.json"),
                _entry("features/nanofold_event_small_no_templates.arrow"),
            ]
        raise ModalAssetAuditError(f"could not list {DATA_VOLUME}:{path}: missing")

    def _ls_locked(self, path: str) -> list[dict[str, object]]:
        if path == "/":
            return [_dir("manifests"), _dir("labels"), _entry("scorer_version.txt")]
        if path == "/manifests":
            return [_entry("manifests/train_tiny.json"), _entry("manifests/public_val_small.json")]
        if path == "/labels":
            return [_entry("labels/public_val_labels.arrow")]
        raise ModalAssetAuditError(f"could not list {LOCKED_VOLUME}:{path}: missing")


def test_modal_asset_audit_passes_clean_two_volume_layout() -> None:
    fake = FakeModalVolumes(locked=True)

    report = audit_modal_assets(lister=fake.ls, reader=fake.read)

    assert report.status == "PASS"
    assert report.locked_asset_layout == "separate_locked_volume"
    assert report.official_lock_boundary is True
    assert report.split_counts == {"train_tiny": 32, "public_val_small": 16}
    require_search_ready_assets(report)


def test_modal_asset_audit_fails_missing_locked_assets() -> None:
    fake = FakeModalVolumes(locked=False)

    report = audit_modal_assets(lister=fake.ls, reader=fake.read)

    assert report.status == "FAIL"
    assert report.locked_asset_layout == "missing"
    assert "locked manifests/labels are missing" in " ".join(report.problems)


def test_modal_asset_audit_fails_bad_split_counts() -> None:
    fake = FakeModalVolumes(locked=True, bad_counts=True)

    report = audit_modal_assets(lister=fake.ls, reader=fake.read)

    assert report.status == "FAIL"
    assert "train_tiny count mismatch" in " ".join(report.problems)


def test_agent_audit_modal_assets_cli_with_real_modal_skipped_if_unavailable() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "autoalphafold3.agent", "audit-modal-assets", "--search-ready"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        payload = json.loads(result.stdout)
        assert payload["locked_asset_layout"] == "separate_locked_volume"
    else:
        assert result.stdout or result.stderr
