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
    destination = _validate_patch_destination(root, patch_path)
    _refuse_staged_candidate_paths(root, normalized)
    patch_text = _candidate_diff(root, normalized)
    if not patch_text.strip():
        raise SafeGitError("candidate snapshot requires a non-empty diff")
    patch_paths = _patch_changed_paths(patch_text)
    if patch_paths != set(normalized):
        raise SafeGitError(f"candidate patch paths do not match changed_paths: {sorted(patch_paths)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(patch_text, encoding="utf-8")
    return CandidateSnapshot(changed_paths=normalized, patch_path=str(destination), patch_text=patch_text)


def keep_candidate_changes(
    *,
    repo_root: str | Path,
    changed_paths: list[str],
    patch_path: str | Path,
) -> list[str]:
    """Stage only validated candidate paths for a KEEP commit."""

    root = Path(repo_root)
    normalized = _validate_candidate_paths(root, changed_paths)
    patch = _validate_patch_destination(root, patch_path)
    if not patch.exists():
        raise SafeGitError(f"candidate patch is missing: {patch}")
    patch_text = patch.read_text(encoding="utf-8")
    patch_paths = _validated_patch_file_paths(root, patch)
    if patch_paths != set(normalized):
        raise SafeGitError(
            "candidate patch paths do not match changed_paths: "
            f"patch={sorted(patch_paths)} changed_paths={normalized}"
        )
    _refuse_staged_candidate_paths(root, normalized)
    current_patch = _candidate_diff(root, normalized)
    if current_patch != patch_text:
        raise SafeGitError("candidate working tree diff no longer matches captured patch")
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
    patch = _validate_patch_destination(root, patch_path)
    if not patch.exists():
        raise SafeGitError(f"candidate patch is missing: {patch}")
    patch_paths = _validated_patch_file_paths(root, patch)
    if patch_paths != set(normalized):
        raise SafeGitError(
            "candidate patch paths do not match changed_paths: "
            f"patch={sorted(patch_paths)} changed_paths={normalized}"
        )
    _refuse_staged_candidate_paths(root, normalized)
    if _candidate_diff(root, normalized) != patch.read_text(encoding="utf-8"):
        raise SafeGitError("candidate working tree diff no longer matches captured patch")
    _git(root, ["apply", "--check", "-R", str(patch)])
    _git(root, ["apply", "-R", str(patch)])
    return normalized


def _candidate_diff(root: Path, paths: list[str]) -> str:
    staged_and_unstaged = _git(root, ["diff", "HEAD", "--", *paths]).stdout
    untracked = _untracked_patch(root, paths)
    return staged_and_unstaged + untracked


def _untracked_patch(root: Path, paths: list[str]) -> str:
    untracked = _git(root, ["ls-files", "--others", "--exclude-standard", "--", *paths]).stdout.splitlines()
    patches: list[str] = []
    for path in untracked:
        _refuse_generated_or_locked_commit_path(path)
        content_path = root / path
        if not content_path.is_file():
            continue
        try:
            content = content_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SafeGitError(f"candidate snapshot refuses non-text untracked file: {path}") from exc
        patches.append(_new_file_patch(path, content))
    return "".join(patches)


def _new_file_patch(path: str, content: str) -> str:
    lines = content.splitlines(keepends=True)
    body_lines = [f"+{line}" for line in lines]
    if content and not content.endswith("\n"):
        body_lines[-1] = f"{body_lines[-1]}\n"
        body_lines.append("\\ No newline at end of file\n")
    body = "".join(body_lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..0000000\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(content.splitlines())} @@\n"
        f"{body}"
    )


def _validate_patch_destination(root: Path, patch_path: str | Path) -> Path:
    resolved_root = root.resolve()
    candidate = Path(patch_path)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise SafeGitError(f"candidate patch path must stay inside repo: {candidate}") from exc
        if ".." in relative.parts:
            raise SafeGitError(f"candidate patch path must not contain traversal: {candidate}")
    else:
        if ".." in candidate.parts:
            raise SafeGitError(f"candidate patch path must not contain traversal: {candidate}")
        relative = candidate
        candidate = root / relative
    parts = relative.parts
    if (
        len(parts) != 6
        or parts[0] != "runs"
        or parts[1] != "autoresearch"
        or parts[3] != "candidates"
        or parts[5] != "patch.diff"
    ):
        raise SafeGitError(
            "candidate patch path must be runs/autoresearch/<run_id>/candidates/<trial_id>/patch.diff"
        )
    _refuse_symlink_components(root, relative)
    artifact_root = (root / "runs" / "autoresearch").resolve()
    try:
        candidate.parent.resolve().relative_to(artifact_root)
    except ValueError as exc:
        raise SafeGitError(f"candidate patch path must stay inside autoresearch artifacts: {candidate}") from exc
    if candidate.is_symlink():
        raise SafeGitError(f"candidate patch path must not be a symlink: {candidate}")
    return candidate


def _refuse_symlink_components(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SafeGitError(f"candidate patch path must not contain symlinks: {current}")


def _validated_patch_file_paths(root: Path, patch: Path) -> set[str]:
    parsed = _git(root, ["apply", "--numstat", str(patch)]).stdout
    paths: set[str] = set()
    for line in parsed.splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            raise SafeGitError(f"unsupported patch stat line: {line}")
        path = fields[-1]
        if " => " in path or path.startswith("{"):
            raise SafeGitError(f"candidate patch refuses rename/copy path: {path}")
        _refuse_generated_or_locked_commit_path(path)
        paths.add(path)
    if not paths:
        raise SafeGitError("candidate patch has no file headers")
    return paths


def _patch_changed_paths(patch_text: str) -> set[str]:
    paths: set[str] = set()
    for line in patch_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4 or not parts[2].startswith("a/") or not parts[3].startswith("b/"):
            raise SafeGitError(f"unsupported patch header: {line}")
        for path in (parts[2][2:], parts[3][2:]):
            if path != "/dev/null":
                _refuse_generated_or_locked_commit_path(path)
                paths.add(path)
    if not paths:
        raise SafeGitError("candidate patch has no file headers")
    return paths


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


def _refuse_staged_candidate_paths(root: Path, paths: list[str]) -> None:
    staged = _git(root, ["diff", "--cached", "--name-only", "--", *paths]).stdout.splitlines()
    if staged:
        raise SafeGitError(f"candidate git operation refuses already-staged paths: {sorted(staged)}")


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
