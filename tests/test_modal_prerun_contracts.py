from __future__ import annotations

from pathlib import Path

import pytest

from autoalphafold3.modal_app import (
    FORBIDDEN_EXECUTION_SECRET_ENV,
    HARNESS_SECRET_NAMES,
    TRIALS_MOUNT,
    TRIAL_WORKER_MOUNTS,
    TRUSTED_ORCHESTRATOR_CLASS,
    WORKER_ROLE_CONTRACTS,
    WorkerRole,
    debug_sandbox_entry,
    event_search_readiness_contract,
    final_validate_seed,
    healthcheck,
    modal_deploy_plan,
    run_trial,
    sample_once,
    score_trial,
    trial_artifact_dir,
    validate_execution_payload,
    validate_worker_role_contracts,
    worker_artifact_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_modal_harness_and_worker_role_contracts_enforce_secret_boundary() -> None:
    status = healthcheck()
    plan = modal_deploy_plan()
    validation = validate_worker_role_contracts()

    assert validation == {"ok": True, "errors": []}
    assert status["trusted_harness_contract"]["cpu_only"] is True
    assert status["trusted_harness_contract"]["may_hold_secret_names"] == HARNESS_SECRET_NAMES
    assert status["trusted_harness_contract"] == plan["trusted_harness_contract"]
    assert status["trusted_harness_contract"]["direct_agent_modal_run_allowed"] is False
    assert status["trusted_harness_contract"]["arbitrary_agent_sandbox_allowed"] is False
    trial_submit = status["trusted_harness_contract"]["deployed_lookup_pattern"]["trial_submit"]
    assert TRUSTED_ORCHESTRATOR_CLASS in trial_submit
    assert "submit_trial.spawn" in trial_submit
    for role, contract in WORKER_ROLE_CONTRACTS.items():
        assert contract["plane"] == "execution"
        assert contract["allowed_secret_names"] == ()
        assert contract["may_write_canonical_ledger"] is False
        assert contract["may_write_discovery_ledger"] is False
        assert contract["forbidden_secret_env"] == FORBIDDEN_EXECUTION_SECRET_ENV
        if role == WorkerRole.SCORER:
            assert contract["mounts"] == "scorer_workers"
            assert contract["may_read_locked_labels"] is True
        else:
            assert contract["mounts"] == "trial_workers"
            assert contract["may_read_locked_labels"] is False


def test_execution_payload_rejects_serialized_harness_secret_keys_recursively() -> None:
    assert validate_execution_payload({"trial_id": "T123"}, role="trial") == {"trial_id": "T123"}
    with pytest.raises(PermissionError, match="OPENAI_API_KEY"):
        validate_execution_payload({"trial_id": "T123", "OPENAI_API_KEY": "secret"}, role="trial")
    with pytest.raises(PermissionError, match="CUSTOM_TOKEN"):
        validate_execution_payload({"trial_id": "T123", "CUSTOM_TOKEN": "secret"}, role="sampler")
    with pytest.raises(PermissionError, match=r"env.OPENAI_API_KEY"):
        validate_execution_payload({"trial_id": "T123", "env": {"OPENAI_API_KEY": "secret"}}, role="trial")
    with pytest.raises(PermissionError, match=r"secrets\[0\]"):
        validate_execution_payload({"trial_id": "T123", "secrets": ["github-token"]}, role="debug")
    with pytest.raises(ValueError, match="unknown worker role"):
        validate_execution_payload({"trial_id": "T123"}, role="harness")


def test_worker_boundaries_reject_serialized_harness_secret_keys() -> None:
    with pytest.raises(PermissionError, match="OPENAI_API_KEY"):
        run_trial({"trial_id": "T123", "OPENAI_API_KEY": "secret"})
    with pytest.raises(PermissionError, match="GITHUB_TOKEN"):
        sample_once({"trial_id": "T123", "env": {"GITHUB_TOKEN": "secret"}})
    with pytest.raises(PermissionError, match="MODAL_TOKEN_SECRET"):
        final_validate_seed({"trial_id": "T123", "MODAL_TOKEN_SECRET": "secret"}, seed=1)
    with pytest.raises(PermissionError, match="DASHBOARD_TOKEN"):
        debug_sandbox_entry({"DASHBOARD_TOKEN": "secret"})


def test_trial_worker_mounts_are_limited_to_trial_artifacts() -> None:
    assert TRIAL_WORKER_MOUNTS[TRIALS_MOUNT] == "autoalphafold3-data:/runs/trials:rw"
    assert all("autoalphafold3-locked" not in spec for spec in TRIAL_WORKER_MOUNTS.values())
    assert all(":/runs:rw" not in spec for spec in TRIAL_WORKER_MOUNTS.values())
    assert trial_artifact_dir("T123") == f"{TRIALS_MOUNT}/T123"
    assert all(path.startswith(f"{TRIALS_MOUNT}/T123/") for path in worker_artifact_paths("T123").values())
    with pytest.raises(ValueError, match="unsafe trial_id"):
        worker_artifact_paths("../ledger")


def test_local_scaffold_contract_is_not_event_search_ready() -> None:
    contract = event_search_readiness_contract()

    assert contract["event_search_ready_locally"] is False
    assert contract["local_scaffold_mode"] == "smoke_only_not_event_search_ready"
    assert contract["required_event_authority"] == "modal_hosted_trusted_orchestrator"
    assert contract["direct_modal_run_allowed"] is False
    assert contract["arbitrary_agent_sandbox_allowed"] is False
    assert contract["worker_contracts_valid"] is True
    assert "Modal-hosted trusted orchestrator" in contract["pending_live_action"]


def test_local_modal_placeholders_never_return_benchmark_ready_evidence() -> None:
    placeholders = [
        run_trial({"trial_id": "T123"}),
        sample_once({"trial_id": "T123"}),
        score_trial("T123"),
        final_validate_seed({"trial_id": "T123"}, seed=1),
        debug_sandbox_entry({"trial_id": "T123"}),
    ]

    for payload in placeholders:
        assert payload["status"] == "INFRA_FAIL"
        assert payload["status"] not in {"SCORED", "KEEP"}
        assert "not_deployed_in_local_environment" in payload["reason"]


def test_modal_source_does_not_expose_forbidden_agent_triggers() -> None:
    source = (REPO_ROOT / "autoalphafold3" / "modal_app.py").read_text(encoding="utf-8")

    assert "app.run(" not in source
    assert "modal.Sandbox.create" not in source
    assert "modal run" not in source
    assert ".with_options(" not in source
