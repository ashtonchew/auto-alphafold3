"""Modal Volume asset audit for the two-Volume benchmark contract.

The target event contract uses two Volumes:

- ``autoalphafold3-data`` for public features, raw inputs, runs, and renders.
- ``autoalphafold3-locked`` for scorer-only manifests, labels, and metadata.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Literal

DATA_VOLUME = "autoalphafold3-data"
LOCKED_VOLUME = "autoalphafold3-locked"

TRAIN_SPLIT = "train_tiny"
PUBLIC_VAL_SPLIT = "public_val_small"
EXPECTED_SPLIT_COUNTS = {
    TRAIN_SPLIT: 32,
    PUBLIC_VAL_SPLIT: 16,
}

REQUIRED_DATA_FILES = (
    "features/train_tiny.arrow",
    "features/public_val_small.arrow",
    "features/feature_fingerprints.json",
    "provenance.json",
)
OPTIONAL_DATA_FILES = (
    "features/nanofold_event_small_no_templates.arrow",
)
REQUIRED_LOCKED_FILES = (
    "manifests/train_tiny.json",
    "manifests/public_val_small.json",
    "labels/public_val_labels.arrow",
)
OPTIONAL_LOCKED_FILES = (
    "scorer_version.txt",
)

LockedAssetLayout = Literal["separate_locked_volume", "missing"]
AuditStatus = Literal["PASS", "FAIL"]
VolumeLister = Callable[[str, str], list[dict[str, object]]]
VolumeReader = Callable[[str, str], str]


class ModalAssetAuditError(RuntimeError):
    """Raised when the Modal CLI cannot inspect assets."""


@dataclass(frozen=True)
class FileEvidence:
    """Evidence for one expected Modal Volume path."""

    path: str
    present: bool
    volume: str | None = None
    size: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "present": self.present,
            "volume": self.volume,
            "size": self.size,
        }


@dataclass(frozen=True)
class ModalAssetAudit:
    """JSON-friendly report for Modal Volume readiness."""

    status: AuditStatus
    locked_asset_layout: LockedAssetLayout
    official_lock_boundary: bool
    target_layout: str
    data_volume: str = DATA_VOLUME
    locked_volume: str = LOCKED_VOLUME
    data_files: list[FileEvidence] = field(default_factory=list)
    locked_files: list[FileEvidence] = field(default_factory=list)
    optional_files: list[FileEvidence] = field(default_factory=list)
    split_counts: dict[str, int | None] = field(default_factory=dict)
    expected_split_counts: dict[str, int] = field(default_factory=lambda: dict(EXPECTED_SPLIT_COUNTS))
    feature_fingerprints_present: bool = False
    provenance_present: bool = False
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "locked_asset_layout": self.locked_asset_layout,
            "official_lock_boundary": self.official_lock_boundary,
            "target_layout": self.target_layout,
            "data_volume": self.data_volume,
            "locked_volume": self.locked_volume,
            "data_files": [item.to_dict() for item in self.data_files],
            "locked_files": [item.to_dict() for item in self.locked_files],
            "optional_files": [item.to_dict() for item in self.optional_files],
            "split_counts": self.split_counts,
            "expected_split_counts": self.expected_split_counts,
            "feature_fingerprints_present": self.feature_fingerprints_present,
            "provenance_present": self.provenance_present,
            "problems": self.problems,
            "notes": self.notes,
        }


def audit_modal_assets(
    *,
    data_volume: str = DATA_VOLUME,
    locked_volume: str = LOCKED_VOLUME,
    env: str | None = None,
    lister: VolumeLister | None = None,
    reader: VolumeReader | None = None,
) -> ModalAssetAudit:
    """Audit Modal Volume assets for May 30, 2026 search-loop readiness."""

    lister = lister or _modal_volume_ls(env=env)
    reader = reader or _modal_volume_get(env=env)
    problems: list[str] = []
    notes: list[str] = []

    data_index = _safe_index_volume(lister, data_volume, "/", problems=problems)
    data_features = _safe_index_volume(lister, data_volume, "/features", problems=problems)
    locked_root = _safe_index_volume(lister, locked_volume, "/", problems=[])
    locked_manifests = _safe_index_volume(lister, locked_volume, "/manifests", problems=[])
    locked_labels = _safe_index_volume(lister, locked_volume, "/labels", problems=[])

    data_files = [
        _evidence(path, data_volume, _index_for_path(path, data_index, data_features))
        for path in REQUIRED_DATA_FILES
    ]
    data_missing = [item.path for item in data_files if not item.present]
    if data_missing:
        problems.append(f"missing required data files: {', '.join(data_missing)}")

    locked_files = [
        _evidence(path, locked_volume, _index_for_path(path, locked_root, locked_manifests, locked_labels))
        for path in REQUIRED_LOCKED_FILES
    ]
    locked_complete = all(item.present for item in locked_files)
    if locked_complete:
        layout: LockedAssetLayout = "separate_locked_volume"
        official_lock_boundary = True
    else:
        layout = "missing"
        official_lock_boundary = False
        problems.append("locked manifests/labels are missing from autoalphafold3-locked")

    split_counts = _split_counts(reader, layout=layout, locked_volume=locked_volume)
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        actual = split_counts.get(split)
        if actual is not None and actual != expected:
            problems.append(f"{split} count mismatch: expected {expected}, got {actual}")

    optional_files = [
        _evidence(path, data_volume, _index_for_path(path, data_index, data_features))
        for path in OPTIONAL_DATA_FILES
    ]
    optional_files.extend(
        _evidence(path, locked_volume, _index_for_path(path, locked_root))
        for path in OPTIONAL_LOCKED_FILES
    )

    feature_fingerprints_present = any(item.path == "features/feature_fingerprints.json" and item.present for item in data_files)
    provenance_present = any(item.path == "provenance.json" and item.present for item in data_files)
    if not feature_fingerprints_present:
        problems.append("feature_fingerprints.json is missing")
    if not provenance_present:
        problems.append("provenance.json is missing")

    if problems:
        status: AuditStatus = "FAIL"
    else:
        status = "PASS"

    return ModalAssetAudit(
        status=status,
        locked_asset_layout=layout,
        official_lock_boundary=official_lock_boundary,
        target_layout="two_volume",
        data_volume=data_volume,
        locked_volume=locked_volume,
        data_files=data_files,
        locked_files=locked_files,
        optional_files=optional_files,
        split_counts=split_counts,
        feature_fingerprints_present=feature_fingerprints_present,
        provenance_present=provenance_present,
        problems=problems,
        notes=notes,
    )


def require_search_ready_assets(report: ModalAssetAudit) -> ModalAssetAudit:
    """Require the final two-Volume lock boundary before search-loop execution."""

    if report.status != "PASS":
        raise ModalAssetAuditError(f"Modal assets are not ready: {report.problems}")
    if report.locked_asset_layout != "separate_locked_volume" or not report.official_lock_boundary:
        raise ModalAssetAuditError("search readiness requires autoalphafold3-locked as a separate Volume")
    return report


def audit_modal_assets_json(**kwargs: object) -> str:
    """Return the audit report as stable JSON."""

    report = audit_modal_assets(**kwargs)
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def _safe_index_volume(
    lister: VolumeLister,
    volume: str,
    path: str,
    *,
    problems: list[str],
) -> dict[str, dict[str, object]]:
    try:
        entries = lister(volume, path)
    except ModalAssetAuditError as exc:
        if path == "/":
            problems.append(str(exc))
        return {}
    return {str(entry.get("Filename", "")): entry for entry in entries}


def _index_for_path(
    path: str,
    *indexes: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    for index in indexes:
        if path in index:
            return index[path]
    return None


def _evidence(path: str, volume: str, entry: dict[str, object] | None) -> FileEvidence:
    return FileEvidence(
        path=path,
        present=entry is not None,
        volume=volume if entry is not None else None,
        size=str(entry.get("Size")) if entry and entry.get("Size") is not None else None,
    )


def _split_counts(
    reader: VolumeReader,
    *,
    layout: LockedAssetLayout,
    locked_volume: str,
) -> dict[str, int | None]:
    if layout == "separate_locked_volume":
        volume = locked_volume
    else:
        return {TRAIN_SPLIT: None, PUBLIC_VAL_SPLIT: None}
    return {
        TRAIN_SPLIT: _manifest_count(reader, volume, "/manifests/train_tiny.json"),
        PUBLIC_VAL_SPLIT: _manifest_count(reader, volume, "/manifests/public_val_small.json"),
    }


def _manifest_count(reader: VolumeReader, volume: str, path: str) -> int | None:
    try:
        payload = json.loads(reader(volume, path))
    except (ModalAssetAuditError, json.JSONDecodeError):
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return len(payload["entries"])
    return None


def _modal_volume_ls(*, env: str | None = None) -> VolumeLister:
    def run(volume: str, path: str) -> list[dict[str, object]]:
        cmd = ["modal", "volume", "ls", volume, path, "--json"]
        if env:
            cmd.extend(["--env", env])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ModalAssetAuditError(f"could not list {volume}:{path}: {detail}")
        return json.loads(result.stdout)

    return run


def _modal_volume_get(*, env: str | None = None) -> VolumeReader:
    def run(volume: str, path: str) -> str:
        cmd = ["modal", "volume", "get", volume, path, "-"]
        if env:
            cmd.extend(["--env", env])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ModalAssetAuditError(f"could not read {volume}:{path}: {detail}")
        return _strip_modal_get_footer(result.stdout)

    return run


def _strip_modal_get_footer(text: str) -> str:
    marker = "\n✓ Finished downloading files to local!"
    if marker in text:
        return text.split(marker, 1)[0].strip()
    return text.strip()
