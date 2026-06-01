from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from autoalphafold3.safe_git import (
    SafeGitError,
    candidate_diff_snapshot,
    keep_candidate_changes,
    revert_candidate_snapshot,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)
    return result.stdout


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "agent@example.com")
    git(repo, "config", "user.name", "Agent")
    allowed = repo / "configs/experiments"
    allowed.mkdir(parents=True)
    (allowed / "candidate.json").write_text('{"value": 1}\n', encoding="utf-8")
    nanofold = repo / "external/nanofold/nanofold/train/model"
    nanofold.mkdir(parents=True)
    (nanofold / "nanofold.py").write_text("LOSS = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("user docs\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    return repo


def test_snapshot_records_allowed_candidate_diff(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    path = repo / "configs/experiments/candidate.json"
    path.write_text('{"value": 2}\n', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"

    snapshot = candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/candidate.json"],
        patch_path=patch,
    )

    assert snapshot.changed_paths == ["configs/experiments/candidate.json"]
    assert "value" in snapshot.patch_text
    assert patch.exists()


def test_snapshot_records_untracked_allowed_candidate_file(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    new_config = repo / "configs/experiments/new_candidate.json"
    new_config.write_text('{"value": 7}\n', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"

    snapshot = candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/new_candidate.json"],
        patch_path=patch,
    )

    assert snapshot.changed_paths == ["configs/experiments/new_candidate.json"]
    assert "new file mode" in snapshot.patch_text
    assert "+{\"value\": 7}" in snapshot.patch_text


def test_snapshot_refuses_patch_destination_outside_candidate_artifacts(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    path = repo / "configs/experiments/candidate.json"
    path.write_text('{"value": 2}\n', encoding="utf-8")

    with pytest.raises(SafeGitError, match="runs/autoresearch"):
        candidate_diff_snapshot(
            repo_root=repo,
            changed_paths=["configs/experiments/candidate.json"],
            patch_path=repo / "configs/experiments/patch.diff",
        )
    with pytest.raises(SafeGitError, match="inside repo"):
        candidate_diff_snapshot(
            repo_root=repo,
            changed_paths=["configs/experiments/candidate.json"],
            patch_path=tmp_path / "outside.diff",
        )


def test_snapshot_refuses_symlinked_patch_destination(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    path = repo / "configs/experiments/candidate.json"
    path.write_text('{"value": 2}\n', encoding="utf-8")
    candidate_root = repo / "runs/autoresearch/run1/candidates"
    candidate_root.mkdir(parents=True)
    (candidate_root / "T123").symlink_to(tmp_path)

    with pytest.raises(SafeGitError, match="inside repo|autoresearch artifacts|symlink"):
        candidate_diff_snapshot(
            repo_root=repo,
            changed_paths=["configs/experiments/candidate.json"],
            patch_path=repo / "runs/autoresearch/run1/candidates/T123/patch.diff",
        )


def test_snapshot_refuses_symlinked_candidate_dir_inside_artifacts(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    path = repo / "configs/experiments/candidate.json"
    path.write_text('{"value": 2}\n', encoding="utf-8")
    candidate_root = repo / "runs/autoresearch/run1/candidates"
    redirected = repo / "runs/autoresearch/run2/candidates/T999"
    candidate_root.mkdir(parents=True)
    redirected.mkdir(parents=True)
    (candidate_root / "T123").symlink_to("../../run2/candidates/T999")

    with pytest.raises(SafeGitError, match="symlink"):
        candidate_diff_snapshot(
            repo_root=repo,
            changed_paths=["configs/experiments/candidate.json"],
            patch_path=repo / "runs/autoresearch/run1/candidates/T123/patch.diff",
        )


def test_snapshot_refuses_symlinked_patch_file(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    path = repo / "configs/experiments/candidate.json"
    path.write_text('{"value": 2}\n', encoding="utf-8")
    patch_dir = repo / "runs/autoresearch/run1/candidates/T123"
    patch_dir.mkdir(parents=True)
    (patch_dir / "patch.diff").symlink_to(tmp_path / "outside.diff")

    with pytest.raises(SafeGitError, match="symlink"):
        candidate_diff_snapshot(
            repo_root=repo,
            changed_paths=["configs/experiments/candidate.json"],
            patch_path=patch_dir / "patch.diff",
        )


def test_snapshot_refuses_already_staged_candidate_paths(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    path = repo / "configs/experiments/candidate.json"
    path.write_text('{"value": 2}\n', encoding="utf-8")
    git(repo, "add", "configs/experiments/candidate.json")

    with pytest.raises(SafeGitError, match="staged"):
        candidate_diff_snapshot(
            repo_root=repo,
            changed_paths=["configs/experiments/candidate.json"],
            patch_path=repo / "runs/autoresearch/run1/candidates/T123/patch.diff",
        )


def test_snapshot_refuses_locked_and_generated_paths(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    locked = repo / "autoalphafold3/scorer"
    locked.mkdir(parents=True)
    (locked / "calpha_lddt.py").write_text("# locked\n", encoding="utf-8")
    generated = repo / "runs/trials/T123"
    generated.mkdir(parents=True)
    (generated / "checkpoint.pt").write_bytes(b"fake")

    with pytest.raises(SafeGitError, match="locked"):
        candidate_diff_snapshot(
            repo_root=repo,
            changed_paths=["autoalphafold3/scorer/calpha_lddt.py"],
            patch_path=repo / "patch.diff",
        )
    with pytest.raises(SafeGitError, match="locked|generated|binary"):
        candidate_diff_snapshot(
            repo_root=repo,
            changed_paths=["runs/trials/T123/checkpoint.pt"],
            patch_path=repo / "patch.diff",
        )


def test_keep_stages_only_allowed_candidate_paths(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    (repo / "configs/experiments/candidate.json").write_text('{"value": 3}\n', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"
    candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/candidate.json"],
        patch_path=patch,
    )
    (repo / "README.md").write_text("unrelated user change\n", encoding="utf-8")

    staged = keep_candidate_changes(
        repo_root=repo,
        changed_paths=["configs/experiments/candidate.json"],
        patch_path=patch,
    )

    assert staged == ["configs/experiments/candidate.json"]
    assert git(repo, "diff", "--cached", "--name-only").splitlines() == ["configs/experiments/candidate.json"]
    assert git(repo, "diff", "--name-only").splitlines() == ["README.md"]
    assert (repo / "README.md").read_text(encoding="utf-8") == "unrelated user change\n"


def test_keep_refuses_same_path_change_after_snapshot(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    candidate = repo / "configs/experiments/candidate.json"
    candidate.write_text('{"value": 3}\n', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"
    candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/candidate.json"],
        patch_path=patch,
    )
    candidate.write_text('{"value": 4}\n', encoding="utf-8")

    with pytest.raises(SafeGitError, match="no longer matches"):
        keep_candidate_changes(
            repo_root=repo,
            changed_paths=["configs/experiments/candidate.json"],
            patch_path=patch,
        )


def test_keep_refuses_staged_path_after_snapshot(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    candidate = repo / "configs/experiments/candidate.json"
    candidate.write_text('{"value": 3}\n', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"
    candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/candidate.json"],
        patch_path=patch,
    )
    git(repo, "add", "configs/experiments/candidate.json")

    with pytest.raises(SafeGitError, match="staged"):
        keep_candidate_changes(
            repo_root=repo,
            changed_paths=["configs/experiments/candidate.json"],
            patch_path=patch,
        )


def test_revert_snapshot_preserves_unrelated_user_changes(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    candidate = repo / "configs/experiments/candidate.json"
    candidate.write_text('{"value": 4}\n', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"
    candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/candidate.json"],
        patch_path=patch,
    )
    (repo / "README.md").write_text("unrelated user change\n", encoding="utf-8")

    reverted = revert_candidate_snapshot(
        repo_root=repo,
        patch_path=patch,
        changed_paths=["configs/experiments/candidate.json"],
    )

    assert reverted == ["configs/experiments/candidate.json"]
    assert candidate.read_text(encoding="utf-8") == '{"value": 1}\n'
    assert (repo / "README.md").read_text(encoding="utf-8") == "unrelated user change\n"


def test_revert_deletes_captured_untracked_candidate_paths(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    candidate = repo / "configs/experiments/new_candidate.json"
    candidate.write_text('{"value": 9}\n', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"
    candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/new_candidate.json"],
        patch_path=patch,
    )

    reverted = revert_candidate_snapshot(
        repo_root=repo,
        patch_path=patch,
        changed_paths=["configs/experiments/new_candidate.json"],
    )

    assert reverted == ["configs/experiments/new_candidate.json"]
    assert not candidate.exists()


def test_revert_deletes_captured_untracked_file_without_final_newline(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    candidate = repo / "configs/experiments/new_candidate.json"
    candidate.write_text('{"value": 9}', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"
    candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/new_candidate.json"],
        patch_path=patch,
    )

    reverted = revert_candidate_snapshot(
        repo_root=repo,
        patch_path=patch,
        changed_paths=["configs/experiments/new_candidate.json"],
    )

    assert reverted == ["configs/experiments/new_candidate.json"]
    assert not candidate.exists()


def test_revert_refuses_untracked_candidate_path_drift(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    candidate = repo / "configs/experiments/new_candidate.json"
    candidate.write_text('{"value": 9}\n', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"
    candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/new_candidate.json"],
        patch_path=patch,
    )
    candidate.write_text('{"value": 10}\n', encoding="utf-8")

    with pytest.raises(SafeGitError, match="no longer matches"):
        revert_candidate_snapshot(
            repo_root=repo,
            patch_path=patch,
            changed_paths=["configs/experiments/new_candidate.json"],
        )


def test_revert_refuses_patch_with_unrelated_or_locked_paths(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    candidate = repo / "configs/experiments/candidate.json"
    candidate.write_text('{"value": 6}\n', encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"
    candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/candidate.json"],
        patch_path=patch,
    )
    patch.write_text(
        patch.read_text(encoding="utf-8")
        + "diff --git a/autoalphafold3/modal_app.py b/autoalphafold3/modal_app.py\n"
        + "--- a/autoalphafold3/modal_app.py\n"
        + "+++ b/autoalphafold3/modal_app.py\n"
        + "@@ -0,0 +1 @@\n"
        + "+tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(SafeGitError, match="locked|generated"):
        revert_candidate_snapshot(
            repo_root=repo,
            patch_path=patch,
            changed_paths=["configs/experiments/candidate.json"],
        )


def test_revert_refuses_headerless_patch_sections(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    candidate = repo / "configs/experiments/candidate.json"
    candidate.write_text('{"value": 6}\n', encoding="utf-8")
    locked = repo / "autoalphafold3/modal_app.py"
    locked.parent.mkdir(exist_ok=True)
    locked.write_text("tampered\n", encoding="utf-8")
    git(repo, "add", "autoalphafold3/modal_app.py")
    git(repo, "commit", "-m", "add locked file")
    candidate.write_text('{"value": 7}\n', encoding="utf-8")
    locked.write_text("safe\n", encoding="utf-8")
    patch = repo / "runs/autoresearch/run1/candidates/T123/patch.diff"
    candidate_diff_snapshot(
        repo_root=repo,
        changed_paths=["configs/experiments/candidate.json"],
        patch_path=patch,
    )
    patch.write_text(
        patch.read_text(encoding="utf-8")
        + "--- a/autoalphafold3/modal_app.py\n"
        + "+++ b/autoalphafold3/modal_app.py\n"
        + "@@ -1 +1 @@\n"
        + "-tampered\n"
        + "+safe\n",
        encoding="utf-8",
    )

    with pytest.raises(SafeGitError, match="modal_app|patch does not apply|corrupt"):
        revert_candidate_snapshot(
            repo_root=repo,
            patch_path=patch,
            changed_paths=["configs/experiments/candidate.json"],
        )


def test_keep_refuses_artifact_paths(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    artifact = repo / "runs/autoresearch/run1/candidates/T123"
    artifact.mkdir(parents=True)
    (artifact / "decision.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(SafeGitError, match="outside|locked|generated"):
        keep_candidate_changes(
            repo_root=repo,
            changed_paths=["runs/autoresearch/run1/candidates/T123/decision.json"],
            patch_path=artifact / "patch.diff",
        )
