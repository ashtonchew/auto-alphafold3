"""Safe git helpers for autoresearch candidate keep/revert decisions."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from autoalphafold3.patch_policy import PatchPolicyError, validate_patch_scope


class SafeGitError(RuntimeError):
    """Raised when a git operation would touch unsafe or unrelated state."""


@dataclass(frozen=True)
class CandidateSnapshot:
    """Patch snapshot for one candidate."""

    changed_paths: list[str]
    patch_path: str
    patch_text: str


def candidate_diff_snapshot(
    *,
    repo_root: str | Path,
    changed_paths: list[str],
    patch_path: str | Path,
) -> CandidateSnapshot:
    """Validate candidate paths and write their current git diff to disk."""

    root = Path(repo_root)
    normalized = _validate_candidate_paths(root, changed_paths)
    patch_text = _git(root, ["diff", "--", *normalized]).stdout
    if not patch_text.strip():
        raise SafeGitError("candidate snapshot requires a non-empty diff")
    destination = Path(patch_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(patch_text, encoding="utf-8")
    return CandidateSnapshot(changed_paths=normalized, patch_path=str(destination), patch_text=patch_text)


def keep_candidate_changes(
    *,
    repo_root: str | Path,
    changed_paths: list[str],
) -> list[str]:
    """Stage only validated candidate paths for a KEEP commit."""

    root = Path(repo_root)
    normalized = _validate_candidate_paths(root, changed_paths)
    _git(root, ["add", "--", *normalized])
    return normalized


def revert_candidate_snapshot(
    *,
    repo_root: str | Path,
    patch_path: str | Path,
    changed_paths: list[str],
) -> list[str]:
    """Reverse one candidate patch without deleting untracked files."""

    root = Path(repo_root)
    normalized = _validate_candidate_paths(root, changed_paths)
    patch = Path(patch_path)
    if not patch.exists():
        raise SafeGitError(f"candidate patch is missing: {patch}")
    _refuse_untracked_candidate_paths(root, normalized)
    _git(root, ["apply", "--check", "-R", str(patch)])
    _git(root, ["apply", "-R", str(patch)])
    return normalized


def _validate_candidate_paths(root: Path, paths: list[str]) -> list[str]:
    try:
        normalized = validate_patch_scope(paths, repo_root=root, allow_empty=False)
    except PatchPolicyError as exc:
        raise SafeGitError(str(exc)) from exc
    for path in normalized:
        _refuse_generated_or_locked_commit_path(path)
    return normalized


def _refuse_generated_or_locked_commit_path(path: str) -> None:
    forbidden_prefixes = (
        "runs/",
        "data/",
        "locked/",
        "autoalphafold3/scorer/",
    )
    forbidden_exact = {
        "runs/ledger.jsonl",
        "runs/discovery_ledger.jsonl",
        "autoalphafold3/modal_app.py",
        "autoalphafold3/benchmark_contract.md",
    }
    if path in forbidden_exact or any(path.startswith(prefix) for prefix in forbidden_prefixes):
        raise SafeGitError(f"candidate git operation refuses locked/generated path: {path}")


def _refuse_untracked_candidate_paths(root: Path, paths: list[str]) -> None:
    untracked = set(_git(root, ["ls-files", "--others", "--exclude-standard", "--", *paths]).stdout.splitlines())
    if untracked:
        raise SafeGitError(f"refusing to delete or overwrite untracked candidate paths: {sorted(untracked)}")


def _git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "git command failed"
        raise SafeGitError(message) from exc
