"""Modal control-plane definition for auto-AlphaFold3.

This module encodes the deploy-once/call-many contract from the handoff. It is
safe to import without the Modal SDK installed; actual deployment requires Modal
and is intentionally separate from local dry-run tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

APP_NAME = "autoalphafold3-modal"
TRUSTED_ORCHESTRATOR_CLASS = "TrustedOrchestrator"
TRIAL_RUNNER_CLASS = "TrialRunner"

DATA_VOLUME = "autoalphafold3-data"
LOCKED_VOLUME = "autoalphafold3-locked"
STATUS_DICT = "autoalphafold3-status"

DATA_MOUNT = "/mnt/autoalphafold3"
FEATURES_MOUNT = f"{DATA_MOUNT}/features"
RUNS_MOUNT = f"{DATA_MOUNT}/runs"
TRIALS_MOUNT = f"{RUNS_MOUNT}/trials"
LOCKED_MOUNT = "/mnt/autoalphafold3-locked"

HARNESS_SECRET_NAMES = (
    "openai-api-key",
    "modal-token",
    "github-token",
    "autoalphafold3-dashboard",
    "autoalphafold3-judge",
)
FORBIDDEN_EXECUTION_SECRET_ENV = (
    "OPENAI_API_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "GITHUB_TOKEN",
    "DASHBOARD_TOKEN",
    "JUDGE_API_KEY",
    "EVALUATOR_API_KEY",
)

TRIAL_WORKER_MOUNTS = {
    DATA_MOUNT: f"{DATA_VOLUME}:/:rw",
}
SCORER_WORKER_MOUNTS = {
    DATA_MOUNT: f"{DATA_VOLUME}:/:ro",
    LOCKED_MOUNT: f"{LOCKED_VOLUME}:/:ro",
}
HARNESS_MOUNTS = {
    RUNS_MOUNT: f"{DATA_VOLUME}:/runs:rw",
}
PREPROCESS_MOUNTS = {
    DATA_MOUNT: f"{DATA_VOLUME}:/:rw",
}


class WorkerRole(StrEnum):
    """Execution-plane worker roles owned by the Modal deployment."""

    TRIAL = "trial"
    SAMPLER = "sampler"
    SCORER = "scorer"
    DEBUG = "debug"
    FINAL_VALIDATION = "final_validation"


WORKER_ROLE_CONTRACTS = {
    WorkerRole.TRIAL.value: {
        "plane": "execution",
        "mounts": "trial_workers",
        "may_read_locked_labels": False,
        "may_write_canonical_ledger": False,
        "may_write_discovery_ledger": False,
        "allowed_secret_names": (),
        "forbidden_secret_env": FORBIDDEN_EXECUTION_SECRET_ENV,
        "artifact_scope": "/runs/trials/<trial_id>/",
    },
    WorkerRole.SAMPLER.value: {
        "plane": "execution",
        "mounts": "trial_workers",
        "may_read_locked_labels": False,
        "may_write_canonical_ledger": False,
        "may_write_discovery_ledger": False,
        "allowed_secret_names": (),
        "forbidden_secret_env": FORBIDDEN_EXECUTION_SECRET_ENV,
        "artifact_scope": "/runs/trials/<trial_id>/sampler/",
    },
    WorkerRole.SCORER.value: {
        "plane": "execution",
        "mounts": "scorer_workers",
        "may_read_locked_labels": True,
        "may_write_canonical_ledger": False,
        "may_write_discovery_ledger": False,
        "allowed_secret_names": (),
        "forbidden_secret_env": FORBIDDEN_EXECUTION_SECRET_ENV,
        "artifact_scope": "read-only trial artifacts plus returned metrics payload",
    },
    WorkerRole.DEBUG.value: {
        "plane": "execution",
        "mounts": "trial_workers",
        "may_read_locked_labels": False,
        "may_write_canonical_ledger": False,
        "may_write_discovery_ledger": False,
        "allowed_secret_names": (),
        "forbidden_secret_env": FORBIDDEN_EXECUTION_SECRET_ENV,
        "artifact_scope": "manual debug workspace only after explicit trigger",
    },
    WorkerRole.FINAL_VALIDATION.value: {
        "plane": "execution",
        "mounts": "trial_workers",
        "may_read_locked_labels": False,
        "may_write_canonical_ledger": False,
        "may_write_discovery_ledger": False,
        "allowed_secret_names": (),
        "forbidden_secret_env": FORBIDDEN_EXECUTION_SECRET_ENV,
        "artifact_scope": "/runs/trials/<trial_id>/final_validation/<seed>/",
    },
}

TRUSTED_HARNESS_CONTRACT = {
    "plane": "harness",
    "event_search_authority": "modal_hosted_trusted_orchestrator",
    "local_scaffold_mode": "smoke_only_not_event_search_ready",
    "cpu_only": True,
    "may_hold_secret_names": HARNESS_SECRET_NAMES,
    "may_write_canonical_ledger": True,
    "may_write_discovery_ledger": True,
    "authors_falsification_controls": True,
    "direct_agent_modal_run_allowed": False,
    "arbitrary_agent_sandbox_allowed": False,
    "deployed_lookup_pattern": {
        "trial_submit": (
            "modal.Cls.from_name(APP_NAME, 'TrustedOrchestrator')()"
            ".submit_trial.spawn(trial_json)"
        ),
        "poll": "modal.FunctionCall.from_id(object_id).get(timeout=0)",
        "control_wave": "TrialRunner.run_gate_control.starmap over orchestrator-authored control tuples",
    },
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
            "harness": HARNESS_MOUNTS,
            "preprocess": PREPROCESS_MOUNTS,
        },
        "resource_tiers": {name: tier.__dict__ for name, tier in RESOURCE_TIERS.items()},
        "function_contracts": FUNCTION_CONTRACTS,
        "modal_object_contracts": MODAL_OBJECT_CONTRACTS,
        "trusted_harness_contract": TRUSTED_HARNESS_CONTRACT,
        "worker_role_contracts": WORKER_ROLE_CONTRACTS,
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
            "harness": HARNESS_MOUNTS,
        },
        "resource_tiers": {name: tier.__dict__ for name, tier in RESOURCE_TIERS.items()},
        "function_contracts": FUNCTION_CONTRACTS,
        "modal_object_contracts": MODAL_OBJECT_CONTRACTS,
        "trusted_harness_contract": TRUSTED_HARNESS_CONTRACT,
        "worker_role_contracts": WORKER_ROLE_CONTRACTS,
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


def validate_worker_role_contracts() -> dict[str, Any]:
    """Validate execution-plane worker contracts without importing Modal."""

    errors: list[str] = []
    for role, contract in WORKER_ROLE_CONTRACTS.items():
        mounts = contract["mounts"]
        if contract["plane"] != "execution":
            errors.append(f"{role}: worker role must stay on execution plane")
        if contract["allowed_secret_names"]:
            errors.append(f"{role}: execution worker may not receive harness secrets")
        if contract["may_write_canonical_ledger"] or contract["may_write_discovery_ledger"]:
            errors.append(f"{role}: execution worker may not write ledgers")
        if role != WorkerRole.SCORER.value and contract["may_read_locked_labels"]:
            errors.append(f"{role}: only scorer role may read locked labels")
        if role != WorkerRole.SCORER.value and mounts != "trial_workers":
            errors.append(f"{role}: non-scorer role must use trial worker mounts")
        if role == WorkerRole.SCORER.value and mounts != "scorer_workers":
            errors.append("scorer: scorer role must use scorer worker mounts")
    if LOCKED_MOUNT in TRIAL_WORKER_MOUNTS or any(LOCKED_VOLUME in spec for spec in TRIAL_WORKER_MOUNTS.values()):
        errors.append("trial_workers: execution workers must not mount locked assets")
    feature_spec = TRIAL_WORKER_MOUNTS.get(FEATURES_MOUNT, "")
    if feature_spec and not feature_spec.endswith(":ro"):
        errors.append("trial_workers: feature mount must be read-only")
    writable_trial_mounts = [mount for mount, spec in TRIAL_WORKER_MOUNTS.items() if spec.endswith(":rw")]
    if writable_trial_mounts != [DATA_MOUNT]:
        errors.append("trial_workers: Modal-supported writable mount must be the single data volume mount")
    for spec in TRIAL_WORKER_MOUNTS.values():
        if ":/runs:rw" in spec:
            errors.append("trial_workers: must not mount whole runs tree writable")
    return {"ok": not errors, "errors": errors}


def validate_execution_payload(payload: dict[str, Any], *, role: str) -> dict[str, Any]:
    """Reject worker payloads that try to serialize harness secrets."""

    if role not in WORKER_ROLE_CONTRACTS:
        raise ValueError(f"unknown worker role: {role}")
    leaked_keys = sorted(_secret_leak_paths(payload))
    if leaked_keys:
        raise PermissionError(f"execution payload contains harness secret keys: {', '.join(leaked_keys)}")
    return payload


def _secret_leak_paths(value: Any, *, path: str = "$") -> set[str]:
    forbidden = set(FORBIDDEN_EXECUTION_SECRET_ENV) | set(HARNESS_SECRET_NAMES)
    leaked: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if (
                key_text in forbidden
                or "SECRET" in key_text
                or "TOKEN" in key_text
                or "API_KEY" in key_text
                or key_text in HARNESS_SECRET_NAMES
            ):
                leaked.add(child_path)
            leaked.update(_secret_leak_paths(child, path=child_path))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            leaked.update(_secret_leak_paths(child, path=f"{path}[{index}]"))
    elif isinstance(value, str) and value in forbidden:
        leaked.add(f"{path}={value}")
    return leaked


def event_search_readiness_contract() -> dict[str, Any]:
    """Return the static event-readiness contract for the Modal harness."""

    worker_validation = validate_worker_role_contracts()
    return {
        "event_search_ready_locally": False,
        "local_scaffold_mode": TRUSTED_HARNESS_CONTRACT["local_scaffold_mode"],
        "required_event_authority": TRUSTED_HARNESS_CONTRACT["event_search_authority"],
        "direct_modal_run_allowed": TRUSTED_HARNESS_CONTRACT["direct_agent_modal_run_allowed"],
        "arbitrary_agent_sandbox_allowed": TRUSTED_HARNESS_CONTRACT["arbitrary_agent_sandbox_allowed"],
        "worker_contracts_valid": worker_validation["ok"],
        "worker_contract_errors": worker_validation["errors"],
        "pending_live_action": "deploy and authenticate the Modal-hosted trusted orchestrator before event search",
    }


def trial_dir(trial_id: str) -> PurePosixPath:
    """Return the only Volume directory a worker may write for a trial."""

    if "/" in trial_id or ".." in trial_id:
        raise ValueError(f"unsafe trial_id: {trial_id}")
    from autoalphafold3.runner import validate_trial_id

    return PurePosixPath(TRIALS_MOUNT) / validate_trial_id(trial_id)


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

    role = {
        "run_trial": WorkerRole.TRIAL.value,
        "sample_once": WorkerRole.SAMPLER.value,
        "final_validate_seed": WorkerRole.FINAL_VALIDATION.value,
    }.get(function_name, WorkerRole.TRIAL.value)
    validate_execution_payload(trial_payload, role=role)
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

    from autoalphafold3.runner import validate_trial_id

    checked_trial_id = validate_trial_id(trial_id)
    return {
        "status": "INFRA_FAIL",
        "reason": "scorer_worker_not_deployed_in_local_environment",
        "trial_id": checked_trial_id,
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

    from autoalphafold3.runner import validate_trial_id

    if split != "public_val_small":
        raise PermissionError(f"unsupported final scoring split: {split}")
    checked_trial_id = validate_trial_id(trial_id)
    return {
        "status": "INFRA_FAIL",
        "reason": "final_scorer_worker_not_deployed_in_local_environment",
        "trial_id": checked_trial_id,
        "seed": seed,
        "split": split,
        "scoring_mounts": SCORER_WORKER_MOUNTS,
    }


def debug_sandbox_entry(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Placeholder for a manually triggered debug path."""

    validate_execution_payload(payload or {}, role=WorkerRole.DEBUG.value)
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
    data_rw = data_volume.with_mount_options()
    data_ro = data_volume.with_mount_options(read_only=True)
    runs_rw = data_volume.with_mount_options(sub_path="/runs")
    locked_ro = locked_volume.with_mount_options(read_only=True)
    scorer_image = modal.Image.debian_slim().pip_install("numpy", "pyarrow", "pydantic").add_local_python_source(
        "autoalphafold3",
        copy=True,
    )
    train_image = modal.Image.debian_slim().pip_install("numpy", "pyarrow", "pydantic", "torch").add_local_python_source(
        "autoalphafold3",
        copy=True,
    ).add_local_dir(
        "configs",
        remote_path="/root/configs",
        copy=True,
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
        volumes={DATA_MOUNT: data_ro, LOCKED_MOUNT: locked_ro},
    )
    @modal.concurrent(max_inputs=4, target_inputs=2)
    class Scorer:
        """Scorer-only Modal class; locked labels are mounted only here."""

        @modal.enter(snap=True)
        def load_locked_state(self) -> None:
            data_ro.reload()
            locked_ro.reload()
            from autoalphafold3.locked_scorer import load_locked_state

            self._locked = load_locked_state(LOCKED_MOUNT)

        @modal.method()
        def score(self, trial_id: str) -> dict[str, Any]:
            from autoalphafold3.locked_scorer import score_trial_artifacts

            data_ro.reload()
            return score_trial_artifacts(
                artifact_dir=trial_artifact_dir(trial_id),
                split="public_val_small",
                locked=self._locked,
                write_outputs=False,
            )

    @app.cls(
        image=train_image,
        gpu=RESOURCE_TIERS["trial"].gpu,
        enable_memory_snapshot=True,
        min_containers=1,
        scaledown_window=300,
        timeout=RESOURCE_TIERS["trial"].timeout_s,
        max_containers=RESOURCE_TIERS["trial"].max_containers,
        volumes={DATA_MOUNT: data_rw},
    )
    class TrialRunner:
        """Trial worker class; public features and trial runs only."""

        @modal.enter(snap=True)
        def cpu_init(self) -> None:
            self.runner_ready = True

        @modal.method()
        def run(self, trial_json: dict[str, Any]) -> dict[str, Any]:
            from autoalphafold3.runner import run_fixed_budget_trial

            validate_execution_payload(trial_json, role=WorkerRole.TRIAL.value)
            result = run_fixed_budget_trial(
                trial_json,
                features_dir=FEATURES_MOUNT,
                output_dir=trial_artifact_dir(str(trial_json["trial_id"])),
            )
            data_volume.commit()
            return result

        @modal.method()
        def run_checkpoint(self, trial_json: dict[str, Any]) -> dict[str, Any]:
            from autoalphafold3.checkpoint_training import run_one_batch_nanofold_checkpoint

            validate_execution_payload(trial_json, role=WorkerRole.TRIAL.value)
            result = run_one_batch_nanofold_checkpoint(
                trial_json,
                features_dir=FEATURES_MOUNT,
                output_dir=trial_artifact_dir(str(trial_json["trial_id"])),
            )
            data_volume.commit()
            return result

        @modal.method()
        def run_sampler(self, trial_json: dict[str, Any]) -> dict[str, Any]:
            from autoalphafold3.sampler import run_sampler_trial

            data_volume.reload()
            validate_execution_payload(trial_json, role=WorkerRole.SAMPLER.value)
            result = run_sampler_trial(
                trial_json,
                features_dir=FEATURES_MOUNT,
                output_dir=trial_artifact_dir(str(trial_json["trial_id"])),
            )
            data_volume.commit()
            return result

        @modal.method()
        def run_gate_control(self, control_payload: dict[str, Any], seed: int) -> dict[str, Any]:
            validate_execution_payload(control_payload, role=WorkerRole.TRIAL.value)
            if control_payload.get("calibration_control") is True:
                return calibration_gate_control_result(control_payload, seed=seed)
            return self.run({**control_payload, "seed": seed})

    @app.cls(
        image=scorer_image,
        cpu=1.0,
        volumes={RUNS_MOUNT: runs_rw},
    )
    class TrustedOrchestrator:
        """CPU-only trusted harness entrypoint; execution workers do not call it."""

        @modal.method()
        def authority_health(self) -> dict[str, Any]:
            return modal_event_authority_health()

        @modal.method()
        def submit_trial(self, trial_json: dict[str, Any]) -> dict[str, Any]:
            role = WorkerRole.SAMPLER.value if trial_json.get("trial_kind") == "sampler" else WorkerRole.TRIAL.value
            validate_execution_payload(trial_json, role=role)
            runner = TrialRunner()
            if trial_json.get("trial_kind") == "sampler":
                call = runner.run_sampler.spawn(trial_json)
            else:
                call = runner.run.spawn(trial_json)
            return {
                "trial_id": str(trial_json["trial_id"]),
                "status": "SUBMITTED",
                "candidate_id": "modal_trusted_orchestrator",
                "metrics": {},
                "fold_cartographer": {
                    "signature": "trusted_orchestrator_submitted",
                    "summary": {"worker_call_id": call.object_id},
                    "buckets": {},
                },
                "artifacts": {"worker_call_id": call.object_id},
                "postmortem": "Trusted orchestrator accepted and spawned the trial worker.",
            }
else:
    app = None


def calibration_gate_control_result(control_payload: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Return calibration-only scored control evidence without training or labels."""

    case = str(control_payload.get("calibration_case", ""))
    kind = str(control_payload.get("control_kind", ""))
    baseline_score = float(control_payload.get("baseline_score", 0.0))
    positive_gain = float(control_payload.get("positive_gain", 0.1))
    null_gain = float(control_payload.get("null_gain", 0.0001))
    if case == "known_positive":
        gain_by_kind = {
            "knockout": positive_gain * 0.1,
            "placebo": positive_gain * 0.1,
            "axis_check": positive_gain,
            "seed_rerun": positive_gain + (seed * 0.0001),
        }
    elif case == "known_null":
        gain_by_kind = {
            "knockout": 0.0,
            "placebo": null_gain,
            "axis_check": null_gain,
            "seed_rerun": null_gain + (seed * 0.001),
        }
    else:
        return {
            "status": "FAIL",
            "failure_signature": "unknown_calibration_case",
            "metrics": {},
            "fold_cartographer": {
                "signature": "unknown_calibration_case",
                "summary": {"calibration_case": case},
                "buckets": {},
            },
            "payload": dict(control_payload),
        }
    score = max(0.0, min(1.0, baseline_score + gain_by_kind.get(kind, 0.0)))
    return {
        "status": "SCORED",
        "metrics": {
            "best_val_calpha_lddt": score,
            "calibration_baseline_score": baseline_score,
            "calibration_gain": score - baseline_score,
        },
        "fold_cartographer": {
            "signature": "gate_calibration_control_scored",
            "summary": {
                "calibration_case": case,
                "control_kind": kind,
                "seed": seed,
            },
            "buckets": {},
        },
        "payload": dict(control_payload),
    }


def modal_event_authority_health() -> dict[str, Any]:
    """No-side-effect proof payload for deployed Modal event authority."""

    contract = event_search_readiness_contract()
    return {
        "status": "PASS",
        "app_name": APP_NAME,
        "authority_class": TRUSTED_ORCHESTRATOR_CLASS,
        "trusted_orchestrator": True,
        "can_submit_trials": True,
        "starts_search": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "direct_modal_run_allowed": contract["direct_modal_run_allowed"],
        "arbitrary_agent_sandbox_allowed": contract["arbitrary_agent_sandbox_allowed"],
        "required_event_authority": contract["required_event_authority"],
    }
