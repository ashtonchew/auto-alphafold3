"""Modal control-plane definition for auto-AlphaFold3.

This module encodes the deploy-once/call-many contract from the handoff. It is
safe to import without the Modal SDK installed; actual deployment requires Modal
and is intentionally separate from local dry-run tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

APP_NAME = "autoalphafold3-modal"

DATA_VOLUME = "autoalphafold3-data"
LOCKED_VOLUME = "autoalphafold3-locked"
STATUS_DICT = "autoalphafold3-status"

DATA_MOUNT = "/mnt/autoalphafold3"
FEATURES_MOUNT = "/mnt/autoalphafold3-features"
RUNS_MOUNT = "/mnt/autoalphafold3-runs"
LOCKED_MOUNT = "/mnt/autoalphafold3-locked"

TRIAL_WORKER_MOUNTS = {
    FEATURES_MOUNT: f"{DATA_VOLUME}:/features:ro",
    RUNS_MOUNT: f"{DATA_VOLUME}:/runs:rw",
}
SCORER_WORKER_MOUNTS = {
    RUNS_MOUNT: f"{DATA_VOLUME}:/runs:ro",
    LOCKED_MOUNT: f"{LOCKED_VOLUME}:/:ro",
}
PREPROCESS_MOUNTS = {
    DATA_MOUNT: f"{DATA_VOLUME}:/:rw",
}

FUNCTION_CONTRACTS = {
    "run_trial": {
        "tier": "trial",
        "mounts": "trial_workers",
        "writes_ledger": False,
        "reads_locked_labels": False,
        "description": "Fixed-budget artifact-generation trial; no scoring labels mounted.",
    },
    "score_trial": {
        "tier": "score_trial",
        "mounts": "scorer_workers",
        "writes_ledger": False,
        "reads_locked_labels": True,
        "description": "Scorer-only public validation worker.",
    },
    "sample_once": {
        "tier": "sampler",
        "mounts": "trial_workers",
        "writes_ledger": False,
        "reads_locked_labels": False,
        "description": "Inference-only sampler burst over a frozen checkpoint.",
    },
    "final_validate_seed": {
        "tier": "final_validation",
        "mounts": "trial_workers",
        "writes_ledger": False,
        "reads_locked_labels": False,
        "description": "Finalist seed artifact generation.",
    },
    "score_final_seed": {
        "tier": "score_trial",
        "mounts": "scorer_workers",
        "writes_ledger": False,
        "reads_locked_labels": True,
        "description": "Scorer-only final public-validation scoring.",
    },
    "debug_sandbox_entry": {
        "tier": "debug",
        "mounts": "trial_workers",
        "writes_ledger": False,
        "reads_locked_labels": False,
        "description": "Manual debug path; not agent-callable during search.",
    },
}


@dataclass(frozen=True)
class ModalResourceTier:
    """Fixed Modal resource policy for one budget tier."""

    gpu: str | None
    timeout_s: int
    startup_timeout_s: int
    max_containers: int
    min_containers: int = 0
    scaledown_window: int | None = None
    retries: int = 0


RESOURCE_TIERS = {
    "dry_run": ModalResourceTier(gpu=None, timeout_s=60, startup_timeout_s=60, max_containers=0),
    "trial": ModalResourceTier(
        gpu="A100-80GB",
        timeout_s=2700,
        startup_timeout_s=600,
        max_containers=6,
        min_containers=1,
        scaledown_window=300,
    ),
    "sampler": ModalResourceTier(gpu="A100", timeout_s=300, startup_timeout_s=300, max_containers=50, scaledown_window=20),
    "score_trial": ModalResourceTier(
        gpu=None,
        timeout_s=600,
        startup_timeout_s=60,
        max_containers=10,
        min_containers=1,
        scaledown_window=600,
    ),
    "final_validation": ModalResourceTier(gpu="H100", timeout_s=5400, startup_timeout_s=900, max_containers=5),
}

MODAL_OBJECT_CONTRACTS = {
    "Scorer": {
        "kind": "cls",
        "tier": "score_trial",
        "enable_memory_snapshot": True,
        "cpu": 2.0,
        "min_containers": 1,
        "scaledown_window": 600,
        "concurrent": {"max_inputs": 4, "target_inputs": 2},
        "mounts": "scorer_workers",
        "reads_locked_labels": True,
    },
    "TrialRunner": {
        "kind": "cls",
        "tier": "trial",
        "enable_memory_snapshot": True,
        "gpu": "A100-80GB",
        "min_containers": 1,
        "scaledown_window": 300,
        "max_containers": 6,
        "mounts": "trial_workers",
        "reads_locked_labels": False,
    },
}


def modal_sdk_available() -> bool:
    """Return whether the Modal SDK is importable in this environment."""

    try:
        import modal  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def healthcheck() -> dict[str, Any]:
    """Return static control-plane health without touching GPUs or Volumes."""

    return {
        "status": "configured_not_deployed",
        "app_name": APP_NAME,
        "modal_sdk_available": modal_sdk_available(),
        "volumes": {
            "data": DATA_VOLUME,
            "locked": LOCKED_VOLUME,
            "status": STATUS_DICT,
        },
        "mounts": {
            "trial_workers": TRIAL_WORKER_MOUNTS,
            "scorer_workers": SCORER_WORKER_MOUNTS,
            "preprocess": PREPROCESS_MOUNTS,
        },
        "resource_tiers": {name: tier.__dict__ for name, tier in RESOURCE_TIERS.items()},
        "function_contracts": FUNCTION_CONTRACTS,
        "modal_object_contracts": MODAL_OBJECT_CONTRACTS,
        "contract": (
            "trial/sampler/debug workers do not mount locked labels; scorer-only "
            "workers mount autoalphafold3-locked and return metrics"
        ),
        "template_policy": "official runs pin max_templates=0 and use empty template placeholders",
        "locked_asset_policy": {
            "target_layout": "two_volume",
            "official_locked_volume": LOCKED_VOLUME,
            "search_ready_requires_locked_volume": True,
        },
    }


def modal_deploy_plan() -> dict[str, Any]:
    """Return the deployable control-plane contract without importing Modal."""

    return {
        "app_name": APP_NAME,
        "deploy_command": "modal deploy autoalphafold3/modal_app.py",
        "sdk_required_for_deploy": True,
        "local_import_safe_without_sdk": True,
        "volumes": {
            "data": DATA_VOLUME,
            "locked": LOCKED_VOLUME,
        },
        "mounts": {
            "trial_workers": TRIAL_WORKER_MOUNTS,
            "scorer_workers": SCORER_WORKER_MOUNTS,
        },
        "resource_tiers": {name: tier.__dict__ for name, tier in RESOURCE_TIERS.items()},
        "function_contracts": FUNCTION_CONTRACTS,
        "modal_object_contracts": MODAL_OBJECT_CONTRACTS,
        "official_training_function": "run_trial",
        "official_training_gpu": RESOURCE_TIERS["trial"].gpu,
        "official_validation_split": "public_val_small",
        "benchmark_result_produced_locally": False,
        "locked_asset_policy": {
            "target_layout": "two_volume",
            "official_locked_volume": LOCKED_VOLUME,
            "search_ready_requires_locked_volume": True,
        },
    }


def trial_dir(trial_id: str) -> PurePosixPath:
    """Return the only Volume directory a worker may write for a trial."""

    if "/" in trial_id or ".." in trial_id:
        raise ValueError(f"unsafe trial_id: {trial_id}")
    return PurePosixPath(RUNS_MOUNT) / "trials" / trial_id


def trial_artifact_dir(trial_id: str) -> str:
    """Return the trial artifact directory as a POSIX path string."""

    return str(trial_dir(trial_id))


def worker_artifact_paths(trial_id: str) -> dict[str, str]:
    """Return canonical per-trial worker artifact paths."""

    root = trial_dir(trial_id)
    return {
        "artifact_manifest_json": str(root / "artifact_manifest.json"),
        "training_log_json": str(root / "training_log.json"),
        "stdout_log": str(root / "stdout.log"),
        "stderr_log": str(root / "stderr.log"),
        "patch_diff": str(root / "patch.diff"),
        "checkpoint": str(root / "checkpoint.pt"),
        "done_marker": str(root / "DONE"),
    }


def _trial_artifact_placeholder(trial_payload: dict[str, Any], *, function_name: str) -> dict[str, Any]:
    """Structural placeholder for the future Modal worker function.

    The local implementation refuses to compute so callers cannot mistake this
    for a real GPU trial. The deployed Modal version will replace this body with
    fixed-budget NanoFold training/evaluation that writes artifacts only.
    """

    trial_id = str(trial_payload.get("trial_id", "UNKNOWN"))
    return {
        "status": "INFRA_FAIL",
        "reason": "modal_worker_not_deployed_in_local_environment",
        "trial_id": trial_id,
        "function_name": function_name,
        "artifacts": worker_artifact_paths(trial_id),
    }


def run_trial(trial_payload: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for the one pinned-A100 official trial Function."""

    return _trial_artifact_placeholder(trial_payload, function_name="run_trial")


def sample_once(sample_payload: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for one future inference-only sampler job."""

    return _trial_artifact_placeholder({**sample_payload, "trial_kind": "sampler"}, function_name="sample_once")


def run_sampler_grid(trial_payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for future synchronous sampler bursts."""

    return sample_once(trial_payload)


def score_trial(trial_id: str) -> dict[str, Any]:
    """Placeholder for scorer-only public-validation scoring."""

    return {
        "status": "INFRA_FAIL",
        "reason": "scorer_worker_not_deployed_in_local_environment",
        "trial_id": trial_id,
        "scoring_mounts": SCORER_WORKER_MOUNTS,
    }


def run_final_validation(trial_payload: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for finalist multi-seed artifact generation."""

    return final_validate_seed(trial_payload, seed=int(trial_payload.get("seed", 0)))


def final_validate_seed(trial_payload: dict[str, Any], seed: int) -> dict[str, Any]:
    """Placeholder for one finalist seed artifact-generation job."""

    return _trial_artifact_placeholder(
        {**trial_payload, "trial_kind": "final_validation", "seed": seed},
        function_name="final_validate_seed",
    )


def score_final_seed(trial_id: str, seed: int, split: str = "public_val_small") -> dict[str, Any]:
    """Placeholder for scorer-only final public-validation scoring."""

    if split != "public_val_small":
        raise PermissionError(f"unsupported final scoring split: {split}")
    return {
        "status": "INFRA_FAIL",
        "reason": "final_scorer_worker_not_deployed_in_local_environment",
        "trial_id": trial_id,
        "seed": seed,
        "split": split,
        "scoring_mounts": SCORER_WORKER_MOUNTS,
    }


def debug_sandbox_entry(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Placeholder for a manually triggered debug path."""

    return {
        "status": "INFRA_FAIL",
        "reason": "debug_sandbox_not_deployed_in_local_environment",
        "payload_keys": sorted((payload or {}).keys()),
    }


try:
    import modal
except ModuleNotFoundError:
    modal = None  # type: ignore[assignment]


if modal is not None:
    data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=False)
    locked_volume = modal.Volume.from_name(LOCKED_VOLUME, create_if_missing=False)
    runs_ro = data_volume.with_mount_options(read_only=True, sub_path="/runs")
    runs_rw = data_volume.with_mount_options(sub_path="/runs")
    features_ro = data_volume.with_mount_options(read_only=True, sub_path="/features")
    locked_ro = locked_volume.with_mount_options(read_only=True)
    scorer_image = modal.Image.debian_slim().pip_install("numpy").add_local_python_source(
        "autoalphafold3",
        copy=False,
    )
    train_image = modal.Image.debian_slim().pip_install("numpy").add_local_python_source(
        "autoalphafold3",
        copy=False,
    ).add_local_dir(
        "external/nanofold",
        remote_path="/root/external/nanofold",
        copy=False,
    )
    app = modal.App(APP_NAME)

    @app.cls(
        image=scorer_image,
        cpu=2.0,
        enable_memory_snapshot=True,
        min_containers=1,
        scaledown_window=600,
        timeout=RESOURCE_TIERS["score_trial"].timeout_s,
        max_containers=RESOURCE_TIERS["score_trial"].max_containers,
        volumes={RUNS_MOUNT: runs_ro, LOCKED_MOUNT: locked_ro},
    )
    @modal.concurrent(max_inputs=4, target_inputs=2)
    class Scorer:
        """Scorer-only Modal class; locked labels are mounted only here."""

        @modal.enter(snap=True)
        def load_locked_state(self) -> None:
            runs_ro.reload()
            locked_ro.reload()
            from autoalphafold3.locked_scorer import load_locked_state

            self._locked = load_locked_state(LOCKED_MOUNT)

        @modal.method()
        def score(self, trial_id: str) -> dict[str, Any]:
            from autoalphafold3.locked_scorer import score_trial_artifacts

            return score_trial_artifacts(
                artifact_dir=trial_artifact_dir(trial_id),
                split="public_val_small",
                locked=self._locked,
            )

    @app.cls(
        image=train_image,
        gpu=RESOURCE_TIERS["trial"].gpu,
        enable_memory_snapshot=True,
        min_containers=1,
        scaledown_window=300,
        timeout=RESOURCE_TIERS["trial"].timeout_s,
        max_containers=RESOURCE_TIERS["trial"].max_containers,
        volumes={FEATURES_MOUNT: features_ro, RUNS_MOUNT: runs_rw},
    )
    class TrialRunner:
        """Trial worker class; public features and trial runs only."""

        @modal.enter(snap=True)
        def cpu_init(self) -> None:
            self.runner_ready = True

        @modal.method()
        def run(self, trial_json: dict[str, Any]) -> dict[str, Any]:
            from autoalphafold3.runner import run_fixed_budget_trial

            return run_fixed_budget_trial(
                trial_json,
                features_dir=FEATURES_MOUNT,
                output_dir=trial_artifact_dir(str(trial_json["trial_id"])),
            )
else:
    app = None
