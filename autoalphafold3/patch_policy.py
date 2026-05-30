"""Patch-policy validation for local preflight gates."""

from __future__ import annotations

from pathlib import Path

ALLOWED_PREFIXES = (
    "autoalphafold3/patches/",
    "configs/experiments/",
)
ALLOWED_EXACT = {
    "external/nanofold/nanofold/train/model/nanofold.py",
    "external/nanofold/nanofold/train/model/nanofold_trunk.py",
    "external/nanofold/nanofold/train/model/pairformer.py",
    "external/nanofold/nanofold/train/model/diffusion_model.py",
    "external/nanofold/nanofold/train/model/diffusion_transformer.py",
    "external/nanofold/nanofold/train/model/msa_module.py",
    "external/nanofold/nanofold/train/model/template_embedder.py",
    "external/nanofold/nanofold/train/loss.py",
    "external/nanofold/nanofold/train/trainer.py",
    "external/nanofold/nanofold/train/chain_dataset.py",
}
DENIED_PREFIXES = (
    "autoalphafold3/scorer/",
    "data/manifests/",
    "data/fingerprints/",
    "data/labels/",
    "data/features/",
    "runs/discovery/",
    "runs/discovery_ledger.jsonl",
    "runs/gate_wave/",
    "runs/trials/",
    "runs/benchmark/",
    "runs/baseline/",
    "locked/",
    "external/nanofold/nanofold/preprocess/",
    "external/nanofold/docker/",
    "external/nanofold/requirements/",
)
DENIED_EXACT = {
    "NANOFOLD_COMMIT",
    "external/nanofold/scripts/download_pdb.sh",
    "autoalphafold3/benchmark_contract.md",
    "autoalphafold3/discovery_ledger.py",
    "autoalphafold3/falsification.py",
    "autoalphafold3/gate_wave.py",
    "autoalphafold3/ledger.py",
    "autoalphafold3/baseline_readiness.py",
    "autoalphafold3/locked_scorer.py",
    "autoalphafold3/modal_assets.py",
    "runs/ledger.jsonl",
    "autoalphafold3/modal_app.py",
    "autoalphafold3/orchestrator.py",
    "autoalphafold3/preflight.py",
    "autoalphafold3/patch_policy.py",
}
# Files that must not be modified by agent patches. _tracing.py is build-time
# developer tooling; if the agent could rewrite it, it could disable trace
# capture and mask its own behavior from human review.
TRACING_LOCKED_FILES = frozenset({
    "autoalphafold3/_tracing.py",
})

# Modules that may import the raindrop or opentelemetry SDK. Only _tracing.py
# is allowed; any other import path bypasses the locked tracing module and
# is rejected.
TRACING_SDK_IMPORTERS = frozenset({
    "autoalphafold3/_tracing.py",
})

TRACING_FORBIDDEN_IMPORTS = (
    "import raindrop",
    "from raindrop",
    "import opentelemetry",
    "from opentelemetry",
)

BINARY_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".png",
    ".pt",
    ".safetensors",
}


class PatchPolicyError(ValueError):
    """Raised when an agent patch violates the editable surface."""


def validate_patch_scope(
    changed_paths: list[str] | tuple[str, ...],
    *,
    repo_root: str | Path = ".",
    allow_empty: bool = True,
) -> list[str]:
    """Validate changed paths against the allowed local search surface."""

    if not changed_paths and allow_empty:
        return []
    if not changed_paths:
        raise PatchPolicyError("patch must contain at least one changed path")

    root = Path(repo_root)
    normalized = [_normalize_repo_path(path) for path in changed_paths]
    validate_tracing_lockout(normalized, "")
    for path in normalized:
        _reject_forbidden_path(path)
        _reject_binary_path(path)
        _reject_symlink(root, path)
        _reject_locked_label_reads(root, path)
        if not _is_allowed_path(path):
            raise PatchPolicyError(f"path is outside the editable surface: {path}")
    return normalized


def _normalize_repo_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        raise PatchPolicyError(f"absolute paths are not allowed: {path}")
    if ".." in candidate.parts:
        raise PatchPolicyError(f"path traversal is not allowed: {path}")
    return candidate.as_posix()


def _is_allowed_path(path: str) -> bool:
    return path in ALLOWED_EXACT or any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _reject_forbidden_path(path: str) -> None:
    if path in DENIED_EXACT or any(path.startswith(prefix) for prefix in DENIED_PREFIXES):
        raise PatchPolicyError(f"path is locked during search: {path}")


def _reject_binary_path(path: str) -> None:
    if Path(path).suffix.lower() in BINARY_SUFFIXES:
        raise PatchPolicyError(f"generated binary patches are not allowed: {path}")


def _reject_symlink(root: Path, path: str) -> None:
    candidate = root / path
    if candidate.exists() and candidate.is_symlink():
        raise PatchPolicyError(f"symlinks are not allowed in patches: {path}")


def validate_tracing_lockout(patch_paths: list[str], patch_diff_text: str) -> None:
    """Reject patches that touch _tracing.py or introduce raindrop/opentelemetry
    imports outside _tracing.py.

    Raises PatchPolicyError on violation.
    """
    # Block direct edits to the tracing module.
    for path in patch_paths:
        if path in TRACING_LOCKED_FILES:
            raise PatchPolicyError(
                f"patch touches locked tracing module: {path}. "
                f"_tracing.py is locked during search per editable_surface.md."
            )

    # Block raindrop/opentelemetry imports added outside _tracing.py.
    # We do this by scanning the patch text for added lines that introduce
    # the forbidden imports. A perfect implementation would parse the diff;
    # for the hackathon we accept the heuristic that any added line
    # starting with "+" and matching the forbidden imports is a violation.
    added_lines = [
        line[1:].lstrip()
        for line in patch_diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    for added in added_lines:
        for forbidden in TRACING_FORBIDDEN_IMPORTS:
            if added.startswith(forbidden):
                raise PatchPolicyError(
                    f"patch adds forbidden import outside _tracing.py: '{added}'. "
                    f"Only autoalphafold3/_tracing.py may import raindrop or opentelemetry."
                )


def _reject_locked_label_reads(root: Path, path: str) -> None:
    candidate = root / path
    if not candidate.exists() or candidate.is_dir():
        return
    try:
        content = candidate.read_text(errors="ignore")
    except OSError:
        return
    suspicious = ("/locked", "locked/labels", "public_val_labels")
    if any(token in content for token in suspicious):
        raise PatchPolicyError(f"editable code appears to read locked labels directly: {path}")
