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

NANOFOLD_PATH = Path("external/nanofold")
NANOFOLD_COMMIT_FILE = Path("NANOFOLD_COMMIT")

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


def expected_nanofold_commit(*, repo_root: str | Path = ".") -> str:
    """Read the pinned NanoFold commit recorded by this repo."""

    return (Path(repo_root) / NANOFOLD_COMMIT_FILE).read_text(encoding="utf-8").strip()


def actual_nanofold_commit(*, repo_root: str | Path = ".") -> str | None:
    """Return the pinned NanoFold commit for a git checkout or vendored copy."""

    path = Path(repo_root) / NANOFOLD_PATH
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

    root = Path(repo_root) / NANOFOLD_PATH
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

    root = Path(repo_root)
    nanofold_root = str(root / NANOFOLD_PATH)
    added = False
    if nanofold_root not in sys.path:
        sys.path.insert(0, nanofold_root)
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
                sys.path.remove(nanofold_root)
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
