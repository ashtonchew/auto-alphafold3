"""Locked manifest loading and hash verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SHA256_HEX_LEN = 64
OfficialSplit = Literal["train", "train_tiny", "public_val", "public_val_small"]
SmokeSplit = Literal["smoke"]


class ManifestEntry(BaseModel):
    """One explicit target entry in a locked or smoke manifest."""

    model_config = ConfigDict(extra="forbid")

    target_id: str
    pdb_id: str
    chain_id: str
    sequence_sha256: str = Field(min_length=SHA256_HEX_LEN, max_length=SHA256_HEX_LEN)
    feature_sha256: str = Field(min_length=SHA256_HEX_LEN, max_length=SHA256_HEX_LEN)
    label_sha256: str = Field(min_length=SHA256_HEX_LEN, max_length=SHA256_HEX_LEN)
    length: int = Field(ge=1)
    msa_depth_bucket: str
    length_bucket: str
    split: OfficialSplit | SmokeSplit
    feature_path: str
    label_path: str


class LockedManifest(BaseModel):
    """Manifest envelope with explicit entries."""

    model_config = ConfigDict(extra="forbid")

    manifest_kind: str = "locked_manifest"
    schema_version: str = "autoaf3.manifest.v1"
    description: str = ""
    entries: list[ManifestEntry]


@dataclass(frozen=True)
class VerifiedManifest:
    """Loaded manifest plus the manifest file hash."""

    path: Path
    sha256: str
    manifest: LockedManifest


@dataclass(frozen=True)
class ManifestValidationReport:
    """JSON-friendly validation report for one manifest file."""

    path: str
    sha256: str
    schema_version: str
    manifest_kind: str
    entry_count: int
    splits: dict[str, int]
    verified_assets: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "manifest_kind": self.manifest_kind,
            "entry_count": self.entry_count,
            "splits": self.splits,
            "verified_assets": self.verified_assets,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA256 hex digest for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_locked_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path = ".",
    verify_assets: bool = True,
) -> VerifiedManifest:
    """Load a manifest and verify referenced feature/label hashes when present."""

    root = Path(repo_root)
    path = _safe_repo_path(root, manifest_path)
    data = json.loads(path.read_text())
    data = _normalize_manifest_data(data)
    manifest = LockedManifest.model_validate(data)

    if verify_assets:
        verify_manifest_assets(manifest, repo_root=root)

    return VerifiedManifest(path=path, sha256=sha256_file(path), manifest=manifest)


def validate_manifest_file(
    manifest_path: str | Path,
    *,
    repo_root: str | Path = ".",
    verify_assets: bool = True,
    allow_empty: bool = False,
) -> ManifestValidationReport:
    """Validate one manifest file and return a compact report."""

    verified = load_locked_manifest(manifest_path, repo_root=repo_root, verify_assets=verify_assets)
    entries = verified.manifest.entries
    if not entries and not allow_empty:
        raise ValueError(f"manifest has no entries: {manifest_path}")
    splits: dict[str, int] = {}
    for entry in entries:
        splits[entry.split] = splits.get(entry.split, 0) + 1
    return ManifestValidationReport(
        path=str(manifest_path),
        sha256=verified.sha256,
        schema_version=verified.manifest.schema_version,
        manifest_kind=verified.manifest.manifest_kind,
        entry_count=len(entries),
        splits=splits,
        verified_assets=verify_assets,
    )


def validate_manifest_files(
    manifest_paths: list[str | Path],
    *,
    repo_root: str | Path = ".",
    verify_assets: bool = True,
    allow_empty: bool = False,
) -> list[ManifestValidationReport]:
    """Validate multiple manifest files."""

    return [
        validate_manifest_file(
            path,
            repo_root=repo_root,
            verify_assets=verify_assets,
            allow_empty=allow_empty,
        )
        for path in manifest_paths
    ]


def verify_manifest_assets(manifest: LockedManifest, *, repo_root: str | Path = ".") -> None:
    """Verify feature and label hashes for all entries in a manifest."""

    root = Path(repo_root)
    for entry in manifest.entries:
        feature_path = _safe_repo_path(root, entry.feature_path)
        label_path = _safe_repo_path(root, entry.label_path)
        _require_hash("feature", feature_path, entry.feature_sha256)
        _require_hash("label", label_path, entry.label_sha256)


def manifest_hashes(paths: dict[str, str | Path], *, repo_root: str | Path = ".") -> dict[str, str]:
    """Return `{name: sha256}` for manifest files."""

    root = Path(repo_root)
    return {name: sha256_file(_safe_repo_path(root, path)) for name, path in paths.items()}


def _normalize_manifest_data(data: object) -> object:
    if not isinstance(data, list):
        return data
    return {
        "manifest_kind": "locked_manifest",
        "schema_version": "autoaf3.manifest.v1",
        "description": "list manifest normalized at load time",
        "entries": [_normalize_manifest_entry(entry) for entry in data],
    }


def _normalize_manifest_entry(entry: object) -> object:
    if not isinstance(entry, dict) or "record_id" not in entry:
        return entry
    split = str(entry["split"])
    source_hash = str(entry.get("source_mmcif_sha256") or "0" * SHA256_HEX_LEN)
    return {
        "target_id": str(entry["record_id"]),
        "pdb_id": str(entry["pdb_id"]),
        "chain_id": str(entry["chain_id"]),
        "sequence_sha256": source_hash,
        "feature_sha256": source_hash,
        "label_sha256": source_hash,
        "length": int(entry["sequence_length"]),
        "msa_depth_bucket": "event_small",
        "length_bucket": "event_small",
        "split": split,
        "feature_path": f"features/{split}.arrow",
        "label_path": f"labels/{split.replace('_small', '')}_labels.arrow",
    }


def label_path_for_entry(entry: ManifestEntry, *, access_mode: Literal["training", "scorer"]) -> Path:
    """Return a label path only for the mode allowed by the benchmark contract."""

    if access_mode == "training" and entry.split not in {"train", "train_tiny"}:
        raise PermissionError(f"training code may not access labels for split {entry.split!r}")
    return Path(entry.label_path)


def refuse_random_split(official_mode: bool, *, random_split: bool) -> None:
    """Refuse random split behavior in official mode."""

    if official_mode and random_split:
        raise ValueError("official trials must use explicit locked manifests, not random splits")


def _require_hash(kind: str, path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{kind} hash mismatch for {path}: expected {expected}, got {actual}")


def _safe_repo_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError(f"absolute paths are not allowed in manifests: {path}")
    if ".." in candidate.parts:
        raise ValueError(f"path traversal is not allowed in manifests: {path}")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if root_resolved not in {resolved, *resolved.parents}:
        raise ValueError(f"manifest path escapes repo root: {path}")
    return resolved
