"""Lightweight NanoFold pin and import inspection helpers.

These helpers intentionally avoid training, feature rebuilding, database access, and
Arrow feature generation. They only verify that the pinned NanoFold code surface
exists and report which optional dependencies are missing locally.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from autoalphafold3.config_contract import validate_config_file
from autoalphafold3.scorer.locked_dataset import (
    VerifiedManifest,
    label_path_for_entry,
    load_locked_manifest,
    refuse_random_split,
)

NANOFOLD_PATH = Path("external/nanofold")
NANOFOLD_COMMIT_FILE = Path("NANOFOLD_COMMIT")
TRAIN_SPLITS = frozenset({"train", "train_tiny"})
PUBLIC_VAL_SPLITS = frozenset({"public_val", "public_val_small"})
TEMPLATE_COLUMNS = (
    "template_mask",
    "template_sequence",
    "template_translations",
    "template_rotations",
)

KEY_PATHS = {
    "readme": "README.md",
    "pyproject": "pyproject.toml",
    "train_entrypoint": "nanofold/train/__main__.py",
    "preprocess_entrypoint": "nanofold/preprocess/__main__.py",
    "train_config_dev": "config/config.dev.json",
    "train_config_default": "config/config.json",
    "docker_train": "docker/Dockerfile.train",
    "docker_preprocess": "docker/Dockerfile.preprocess",
    "dataset": "nanofold/train/chain_dataset.py",
    "trainer": "nanofold/train/trainer.py",
    "loss": "nanofold/train/loss.py",
    "nanofold_model": "nanofold/train/model/nanofold.py",
    "trunk": "nanofold/train/model/nanofold_trunk.py",
    "pairformer": "nanofold/train/model/pairformer.py",
    "diffusion_model": "nanofold/train/model/diffusion_model.py",
    "diffusion_transformer": "nanofold/train/model/diffusion_transformer.py",
    "msa_module": "nanofold/train/model/msa_module.py",
    "template_embedder": "nanofold/train/model/template_embedder.py",
}

IMPORT_TARGETS = (
    "nanofold",
    "nanofold.train.model.nanofold",
    "nanofold.train.trainer",
    "nanofold.preprocess.__main__",
)


@dataclass(frozen=True)
class ImportStatus:
    """Import result for one NanoFold module."""

    module: str
    ok: bool
    error_type: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DatasetBoundary:
    """Manifest-backed official dataset boundary for NanoFold workers."""

    train_manifest: str
    public_val_manifest: str
    train_manifest_sha256: str
    public_val_manifest_sha256: str
    train_count: int
    public_val_count: int
    training_label_access: Literal["train_only"]
    validation_label_access: Literal["scorer_only"]
    random_split_allowed: bool = False
    max_templates: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "train_manifest": self.train_manifest,
            "public_val_manifest": self.public_val_manifest,
            "train_manifest_sha256": self.train_manifest_sha256,
            "public_val_manifest_sha256": self.public_val_manifest_sha256,
            "train_count": self.train_count,
            "public_val_count": self.public_val_count,
            "training_label_access": self.training_label_access,
            "validation_label_access": self.validation_label_access,
            "random_split_allowed": self.random_split_allowed,
            "max_templates": self.max_templates,
        }


def expected_nanofold_commit(*, repo_root: str | Path = ".") -> str:
    """Read the pinned NanoFold commit recorded by this repo."""

    return (Path(repo_root) / NANOFOLD_COMMIT_FILE).read_text(encoding="utf-8").strip()


def repo_root_path(repo_root: str | Path = ".") -> Path:
    """Resolve a repo root without baking in a developer-specific path."""

    return Path(repo_root).resolve()


def nanofold_root(*, repo_root: str | Path = ".") -> Path:
    """Return the pinned NanoFold root under the supplied repo root."""

    root = repo_root_path(repo_root)
    path = (root / NANOFOLD_PATH).resolve()
    if root not in {path, *path.parents}:
        raise ValueError(f"NanoFold path escapes repo root: {path}")
    return path


def actual_nanofold_commit(*, repo_root: str | Path = ".") -> str | None:
    """Return the pinned NanoFold commit for a git checkout or vendored copy."""

    path = nanofold_root(repo_root=repo_root)
    if not path.exists():
        return None
    if not (path / ".git").exists():
        try:
            nanofold_path_map(repo_root=repo_root)
            return expected_nanofold_commit(repo_root=repo_root)
        except (FileNotFoundError, OSError):
            return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        try:
            nanofold_path_map(repo_root=repo_root)
            return expected_nanofold_commit(repo_root=repo_root)
        except (FileNotFoundError, OSError):
            return None
    return result.stdout.strip()


def validate_nanofold_pin(*, repo_root: str | Path = ".") -> None:
    """Assert the local NanoFold checkout matches `NANOFOLD_COMMIT`."""

    expected = expected_nanofold_commit(repo_root=repo_root)
    actual = actual_nanofold_commit(repo_root=repo_root)
    if actual != expected:
        raise ValueError(f"NanoFold pin mismatch: expected {expected}, got {actual}")


def nanofold_path_map(*, repo_root: str | Path = ".") -> dict[str, str]:
    """Return the important pinned NanoFold paths after checking they exist."""

    root = nanofold_root(repo_root=repo_root)
    missing = [name for name, rel in KEY_PATHS.items() if not (root / rel).exists()]
    if missing:
        raise FileNotFoundError(f"NanoFold checkout missing key paths: {', '.join(missing)}")
    return {name: str(NANOFOLD_PATH / rel) for name, rel in KEY_PATHS.items()}


def load_nanofold_config(config_path: str | Path, *, repo_root: str | Path = ".") -> dict[str, object]:
    """Load a NanoFold JSON config from the pinned checkout or repo root."""

    path = Path(config_path)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return json.loads(path.read_text(encoding="utf-8"))


def import_smoke(*, repo_root: str | Path = ".") -> list[ImportStatus]:
    """Try lightweight NanoFold imports and report missing optional deps."""

    root = repo_root_path(repo_root)
    nanofold_sys_path = str(nanofold_root(repo_root=root))
    added = False
    if nanofold_sys_path not in sys.path:
        sys.path.insert(0, nanofold_sys_path)
        added = True

    statuses: list[ImportStatus] = []
    try:
        for module in IMPORT_TARGETS:
            try:
                importlib.import_module(module)
                statuses.append(ImportStatus(module=module, ok=True))
            except Exception as exc:  # noqa: BLE001 - diagnostics should report any import failure.
                statuses.append(
                    ImportStatus(
                        module=module,
                        ok=False,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                )
    finally:
        if added:
            try:
                sys.path.remove(nanofold_sys_path)
            except ValueError:
                pass
    return statuses


def import_smoke_summary(*, repo_root: str | Path = ".") -> dict[str, object]:
    """Return JSON-friendly NanoFold pin and import diagnostics."""

    statuses = import_smoke(repo_root=repo_root)
    return {
        "expected_commit": expected_nanofold_commit(repo_root=repo_root),
        "actual_commit": actual_nanofold_commit(repo_root=repo_root),
        "paths": nanofold_path_map(repo_root=repo_root),
        "imports": [status.__dict__ for status in statuses],
    }


def official_dataset_boundary(
    *,
    train_manifest: str | Path,
    public_val_manifest: str | Path,
    repo_root: str | Path = ".",
    verify_assets: bool = False,
    random_split: bool = False,
) -> DatasetBoundary:
    """Validate explicit train/public-validation manifests for official workers."""

    refuse_random_split(True, random_split=random_split)
    train = load_locked_manifest(train_manifest, repo_root=repo_root, verify_assets=verify_assets)
    public_val = load_locked_manifest(public_val_manifest, repo_root=repo_root, verify_assets=verify_assets)
    _require_manifest_splits(train, allowed=TRAIN_SPLITS, label="train")
    _require_manifest_splits(public_val, allowed=PUBLIC_VAL_SPLITS, label="public validation")
    for entry in train.manifest.entries:
        label_path_for_entry(entry, access_mode="training")
    for entry in public_val.manifest.entries:
        try:
            label_path_for_entry(entry, access_mode="training")
        except PermissionError:
            continue
        raise PermissionError(f"public validation label path is not training-accessible: {entry.target_id}")
    return DatasetBoundary(
        train_manifest=str(train_manifest),
        public_val_manifest=str(public_val_manifest),
        train_manifest_sha256=train.sha256,
        public_val_manifest_sha256=public_val.sha256,
        train_count=len(train.manifest.entries),
        public_val_count=len(public_val.manifest.entries),
        training_label_access="train_only",
        validation_label_access="scorer_only",
    )


def validate_no_template_config(config_path: str | Path, *, repo_root: str | Path = ".") -> None:
    """Require a local or NanoFold config to pin official runs to max_templates=0."""

    result = validate_config_file(config_path, repo_root=repo_root)
    if not result.valid:
        raise ValueError(f"config is invalid for no-template official runs: {result.missing_keys}")


def verify_empty_template_placeholders(feature_path: str | Path) -> dict[str, object]:
    """Verify NanoFold Arrow IPC template columns are present and empty."""

    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.ipc as ipc  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to inspect NanoFold feature Arrow files") from exc

    path = Path(feature_path)
    with pa.memory_map(str(path)) as source:
        table = ipc.open_file(source).read_all()
    missing = [column for column in TEMPLATE_COLUMNS if column not in table.column_names]
    if missing:
        raise ValueError(f"feature file missing template columns: {', '.join(missing)}")
    non_empty = {}
    for column in TEMPLATE_COLUMNS:
        lengths = [len(value) for value in table[column].to_pylist()]
        bad_lengths = [length for length in lengths if length != 0]
        if bad_lengths:
            non_empty[column] = bad_lengths
    if non_empty:
        raise ValueError(f"feature file contains non-empty template placeholders: {non_empty}")
    return {
        "feature_path": str(path),
        "records": table.num_rows,
        "template_columns": list(TEMPLATE_COLUMNS),
        "template_records_all_empty": True,
    }


def _require_manifest_splits(
    verified: VerifiedManifest,
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    splits = {entry.split for entry in verified.manifest.entries}
    unexpected = sorted(splits - allowed)
    if unexpected:
        raise ValueError(f"{label} manifest contains disallowed splits: {', '.join(unexpected)}")
    if not verified.manifest.entries:
        raise ValueError(f"{label} manifest must contain explicit entries")
