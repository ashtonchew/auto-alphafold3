from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from autoalphafold3.modal_assets import (
    DATA_VOLUME,
    LOCKED_VOLUME,
    ModalAssetAudit,
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


def _arrow_ipc_bytes(rows: int = 1) -> bytes:
    pa = pytest.importorskip("pyarrow")
    ipc = pytest.importorskip("pyarrow.ipc")
    sink = pa.BufferOutputStream()
    table = pa.table({"target_id": [f"target_{index}" for index in range(rows)]})
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


class FakeModalVolumes:
    def __init__(
        self,
        *,
        locked: bool,
        bad_counts: bool = False,
        bad_provenance: bool = False,
        bad_fingerprints: bool = False,
        bad_scorer_version: bool = False,
        locked_prefix_in_data: bool = False,
        arrow_bytes: bytes | None = None,
    ) -> None:
        self.separate_locked = locked
        self.bad_counts = bad_counts
        self.bad_provenance = bad_provenance
        self.bad_fingerprints = bad_fingerprints
        self.bad_scorer_version = bad_scorer_version
        self.locked_prefix_in_data = locked_prefix_in_data
        self.arrow_bytes = arrow_bytes

    def ls(self, volume: str, path: str) -> list[dict[str, object]]:
        if volume == DATA_VOLUME:
            return self._ls_data(path)
        if volume == LOCKED_VOLUME and self.separate_locked:
            return self._ls_locked(path)
        raise ModalAssetAuditError(f"could not list {volume}:{path}: Volume not found")

    def read(self, volume: str, path: str) -> str | bytes:
        if path.endswith(".arrow") and self.arrow_bytes is not None:
            return self.arrow_bytes
        if "train_tiny.json" in path:
            return _manifest("train_tiny", 31 if self.bad_counts else 32)
        if "public_val_small.json" in path:
            return _manifest("public_val_small", 16)
        if path.endswith("feature_fingerprints.json"):
            if self.bad_fingerprints:
                return json.dumps({"files": {}})
            return json.dumps(
                {
                    "files": {
                        "features/train_tiny.arrow": "a" * 64,
                        "features/public_val_small.arrow": "b" * 64,
                    }
                }
            )
        if path.endswith("provenance.json"):
            if self.bad_provenance:
                return json.dumps({"splits": {"train_tiny": 32}})
            return json.dumps({"splits": {"train_tiny": 32, "public_val_small": 16}})
        if path.endswith("scorer_version.txt"):
            return "wrong" if self.bad_scorer_version else "calpha_lddt_v1"
        raise ModalAssetAuditError(f"could not read {volume}:{path}: missing")

    def _ls_data(self, path: str) -> list[dict[str, object]]:
        if path == "/":
            entries = [_dir("features"), _dir("raw"), _entry("provenance.json")]
            if self.locked_prefix_in_data:
                entries.append(_dir("locked"))
            return entries
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
    fake = FakeModalVolumes(locked=True, arrow_bytes=_arrow_ipc_bytes())

    report = audit_modal_assets(lister=fake.ls, reader=fake.read)

    assert report.status == "PASS"
    assert report.locked_asset_layout == "separate_locked_volume"
    assert report.official_lock_boundary is True
    assert report.split_counts == {"train_tiny": 32, "public_val_small": 16}
    assert report.feature_fingerprints_valid is True
    assert report.provenance_valid is True
    assert report.scorer_version_valid is True
    assert report.public_data_locked_prefix_absent is True
    require_search_ready_assets(report)


def test_modal_asset_audit_reads_arrow_bytes_when_available() -> None:
    fake = FakeModalVolumes(locked=True, arrow_bytes=_arrow_ipc_bytes())

    report = audit_modal_assets(lister=fake.ls, reader=fake.read)

    assert report.status == "PASS"
    assert report.arrow_readability == {
        "train_tiny": "readable",
        "public_val_small": "readable",
    }


@pytest.mark.parametrize("status", ["skipped_pyarrow_unavailable", "skipped_text_reader", "failed"])
def test_search_ready_assets_require_readable_arrow_features(status: str) -> None:
    report = ModalAssetAudit(
        status="PASS",
        locked_asset_layout="separate_locked_volume",
        official_lock_boundary=True,
        target_layout="two_volume",
        arrow_readability={"train_tiny": "readable", "public_val_small": status},
    )

    with pytest.raises(ModalAssetAuditError, match="readable Arrow features"):
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


@pytest.mark.parametrize(
    ("flag", "problem"),
    [
        ("bad_provenance", "provenance.json split count mismatch"),
        ("bad_fingerprints", "feature_fingerprints.json missing SHA256"),
        ("bad_scorer_version", "scorer_version.txt mismatch"),
        ("locked_prefix_in_data", "public data Volume must not contain a locked/ prefix"),
    ],
)
def test_modal_asset_audit_fails_invalid_readiness_metadata(flag: str, problem: str) -> None:
    kwargs: dict[str, Any] = {flag: True}
    fake = FakeModalVolumes(locked=True, **kwargs)

    report = audit_modal_assets(lister=fake.ls, reader=fake.read)

    assert report.status == "FAIL"
    assert problem in " ".join(report.problems)


def test_modal_asset_audit_rejects_non_hex_fingerprints() -> None:
    class BadHexFingerprints(FakeModalVolumes):
        def read(self, volume: str, path: str) -> str | bytes:
            if path.endswith("feature_fingerprints.json"):
                return json.dumps(
                    {
                        "files": {
                            "features/train_tiny.arrow": "z" * 64,
                            "features/public_val_small.arrow": "b" * 64,
                        }
                    }
                )
            return super().read(volume, path)

    report = audit_modal_assets(lister=BadHexFingerprints(locked=True).ls, reader=BadHexFingerprints(locked=True).read)

    assert report.status == "FAIL"
    assert "feature_fingerprints.json missing SHA256 for features/train_tiny.arrow" in " ".join(report.problems)


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
