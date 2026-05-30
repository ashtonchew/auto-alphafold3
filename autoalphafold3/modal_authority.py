"""Approved Modal event authority proof writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from autoalphafold3.modal_app import APP_NAME, TRUSTED_ORCHESTRATOR_CLASS

APPROVAL_TEXT = "I_APPROVE_MODAL_EVENT_AUTHORITY"
DEFAULT_AUTHORITY_PATH = Path("runs/modal_event_authority.json")


class ModalAuthorityError(RuntimeError):
    """Raised when Modal event authority cannot be proven safely."""


class ModalAuthorityClient(Protocol):
    """Small protocol for deployed Modal authority lookup."""

    def authority_health(self) -> dict[str, object]:
        """Return no-side-effect health evidence from the trusted orchestrator."""


@dataclass(frozen=True)
class ModalAuthorityResult:
    """JSON-friendly Modal authority audit result."""

    status: str
    mode: str
    authority_path: str
    wrote_files: list[str]
    plan: dict[str, object]
    authority: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "authority_path": self.authority_path,
            "wrote_files": self.wrote_files,
            "plan": self.plan,
            "authority": self.authority,
        }


def audit_modal_event_authority(
    *,
    repo_root: str | Path = ".",
    authority_path: str | Path = DEFAULT_AUTHORITY_PATH,
    approval: str | None = None,
    mode: str = "dry-run",
    modal_env: str | None = None,
    client: ModalAuthorityClient | None = None,
) -> ModalAuthorityResult:
    """Plan or write a live proof that the trusted Modal authority is deployed."""

    root = Path(repo_root)
    output_path = root / authority_path
    plan = modal_authority_plan(authority_path=authority_path)
    if mode == "dry-run":
        return ModalAuthorityResult(
            status="PLANNED",
            mode=mode,
            authority_path=str(output_path),
            wrote_files=[],
            plan=plan,
        )
    if mode != "modal":
        raise ModalAuthorityError(f"unsupported Modal authority audit mode: {mode}")
    if approval != APPROVAL_TEXT:
        raise ModalAuthorityError(f"Modal authority audit requires --approve {APPROVAL_TEXT}")
    if output_path.exists():
        raise ModalAuthorityError(f"Modal authority output already exists: {output_path}")
    modal_client = client if client is not None else DeployedModalAuthorityClient(environment_name=modal_env)
    payload = modal_client.authority_health()
    _require_authority_payload(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, payload)
    return ModalAuthorityResult(
        status="PASS",
        mode=mode,
        authority_path=str(output_path),
        wrote_files=[str(output_path)],
        plan=plan,
        authority=payload,
    )


def modal_authority_plan(*, authority_path: str | Path = DEFAULT_AUTHORITY_PATH) -> dict[str, object]:
    """Return the no-side-effect Modal event-authority proof plan."""

    return {
        "authority_path": str(authority_path),
        "requires_approval": APPROVAL_TEXT,
        "app_name": APP_NAME,
        "authority_class": TRUSTED_ORCHESTRATOR_CLASS,
        "method": "authority_health",
        "starts_search": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
    }


class DeployedModalAuthorityClient:
    """Modal SDK client for the deployed trusted-orchestrator health method."""

    def __init__(self, *, environment_name: str | None = None) -> None:
        self.environment_name = environment_name
        try:
            import modal
        except ModuleNotFoundError as exc:
            raise ModalAuthorityError("Modal SDK is required for live Modal authority audits") from exc
        self._modal = modal

    def authority_health(self) -> dict[str, object]:
        orchestrator_cls = self._modal.Cls.from_name(
            APP_NAME,
            TRUSTED_ORCHESTRATOR_CLASS,
            environment_name=self.environment_name,
        )
        orchestrator = orchestrator_cls()
        payload = orchestrator.authority_health.remote()
        if not isinstance(payload, dict):
            raise ModalAuthorityError("TrustedOrchestrator.authority_health returned a non-object payload")
        return payload


def _require_authority_payload(payload: dict[str, object]) -> None:
    required = {
        "status": "PASS",
        "app_name": APP_NAME,
        "authority_class": TRUSTED_ORCHESTRATOR_CLASS,
        "trusted_orchestrator": True,
        "can_submit_trials": True,
        "starts_search": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "direct_modal_run_allowed": False,
        "arbitrary_agent_sandbox_allowed": False,
        "required_event_authority": "modal_hosted_trusted_orchestrator",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ModalAuthorityError(f"Modal authority proof has invalid {key}: {payload.get(key)!r}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
