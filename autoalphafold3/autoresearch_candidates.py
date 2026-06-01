"""Candidate artifact envelopes for bounded autoresearch runs."""

from __future__ import annotations

import csv
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoalphafold3.runner import validate_trial_id
from autoalphafold3.schema import PRIMARY_METRIC, SCORER_VERSION, TrialStatus

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
RUN_MANIFEST_SCHEMA = "autoaf3.autoresearch_run_manifest.v1"
CANDIDATE_MANIFEST_SCHEMA = "autoaf3.autoresearch_candidate_manifest.v1"
DECISION_SCHEMA = "autoaf3.autoresearch_decision.v1"
DECISION_STATUSES = {
    TrialStatus.KEEP.value,
    TrialStatus.DISCARD.value,
    TrialStatus.FAIL.value,
    TrialStatus.INFRA_FAIL.value,
}
FORBIDDEN_TRUE_EVIDENCE_FLAGS = {
    "official_benchmark_result",
    "writes_baseline",
    "writes_ledger",
    "writes_discovery_ledger",
    "starts_search",
}


class CandidateArtifactError(RuntimeError):
    """Raised when candidate artifacts would violate the run contract."""


@dataclass(frozen=True)
class CandidateEnvelope:
    """Paths for one candidate artifact envelope."""

    run_id: str
    trial_id: str
    root: Path
    candidate_dir: Path
    candidate_id: str

    @property
    def hypothesis_path(self) -> Path:
        return self.candidate_dir / "hypothesis.md"

    @property
    def patch_path(self) -> Path:
        return self.candidate_dir / "patch.diff"

    @property
    def config_path(self) -> Path:
        return self.candidate_dir / "config.json"

    @property
    def trial_path(self) -> Path:
        return self.candidate_dir / "trial.json"

    @property
    def preflight_path(self) -> Path:
        return self.candidate_dir / "preflight.json"

    @property
    def training_manifest_path(self) -> Path:
        return self.candidate_dir / "training_manifest.json"

    @property
    def loss_history_path(self) -> Path:
        return self.candidate_dir / "loss_history.json"

    @property
    def metrics_path(self) -> Path:
        return self.candidate_dir / "metrics.json"

    @property
    def error_report_path(self) -> Path:
        return self.candidate_dir / "error_report.json"

    @property
    def decision_path(self) -> Path:
        return self.candidate_dir / "decision.json"

    @property
    def postmortem_path(self) -> Path:
        return self.candidate_dir / "postmortem.md"

    @property
    def manifest_path(self) -> Path:
        return self.candidate_dir / "candidate_manifest.json"


def validate_run_id(run_id: str) -> str:
    """Validate a filesystem-safe autoresearch run id."""

    if not RUN_ID_RE.fullmatch(run_id):
        raise CandidateArtifactError(f"invalid autoresearch run_id: {run_id}")
    return run_id


def run_root(repo_root: str | Path, run_id: str) -> Path:
    """Return the source-local evidence root for one autoresearch run."""

    return Path(repo_root) / "runs" / "autoresearch" / validate_run_id(run_id)


def create_run_manifest(
    *,
    repo_root: str | Path,
    run_id: str,
    base_commit: str,
    planner: str,
    mode: str,
    description: str,
) -> dict[str, object]:
    """Create the run-level manifest and empty summary/results files."""

    root = run_root(repo_root, run_id)
    _require_safe_run_root(root, run_id)
    if (root / "run_manifest.json").exists():
        raise CandidateArtifactError(f"run manifest already exists: {root / 'run_manifest.json'}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidates").mkdir(exist_ok=True)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": validate_run_id(run_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": base_commit,
        "branch": _current_branch(repo_root),
        "planner": planner,
        "mode": mode,
        "target": "NanoFold-style AlphaFold3-lite",
        "description": description,
        "primary_metric": PRIMARY_METRIC,
        "scorer_version": SCORER_VERSION,
        "official_benchmark_result": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "live_modal_execution": mode == "modal",
        "starts_search": mode == "modal",
        "candidate_count": 0,
    }
    _atomic_write_json(root / "run_manifest.json", manifest)
    _atomic_write_json(
        root / "summary.json",
        {
            "schema_version": "autoaf3.autoresearch_summary.v1",
            "run_id": run_id,
            "candidates": [],
            "official_benchmark_result": False,
        },
    )
    (root / "results.tsv").write_text(
        "trial_id\tcandidate_id\tstatus\tprimary_metric\tmatched_budget_delta\tglobal_baseline_delta\tprovisional_keep\tdecision_path\n",
        encoding="utf-8",
    )
    return manifest


def create_candidate_envelope(
    *,
    repo_root: str | Path,
    run_id: str,
    trial_id: str,
    hypothesis: str,
    trial: dict[str, object],
    config: dict[str, object] | None = None,
    patch_text: str = "",
) -> CandidateEnvelope:
    """Create one candidate artifact envelope with preregistered inputs."""

    checked_trial_id = validate_trial_id(trial_id)
    if trial.get("trial_id") != checked_trial_id:
        raise CandidateArtifactError(f"trial payload trial_id must match envelope trial_id: {checked_trial_id}")
    root = run_root(repo_root, run_id)
    if not (root / "run_manifest.json").exists():
        raise CandidateArtifactError(f"run manifest is missing: {root / 'run_manifest.json'}")
    candidate_dir = root / "candidates" / checked_trial_id
    temp_candidate_dir = root / "candidates" / f".{checked_trial_id}.{uuid.uuid4().hex}.tmp"
    _require_missing_dir(candidate_dir)
    _require_missing_dir(temp_candidate_dir)
    temp_candidate_dir.mkdir(parents=True)
    envelope = CandidateEnvelope(
        run_id=validate_run_id(run_id),
        trial_id=checked_trial_id,
        root=root,
        candidate_dir=candidate_dir,
        candidate_id=str(trial.get("candidate_id", checked_trial_id)),
    )
    temp_envelope = CandidateEnvelope(
        run_id=envelope.run_id,
        trial_id=envelope.trial_id,
        root=envelope.root,
        candidate_dir=temp_candidate_dir,
        candidate_id=envelope.candidate_id,
    )
    _atomic_write_text(temp_envelope.hypothesis_path, _require_text(hypothesis, "hypothesis"))
    _atomic_write_text(temp_envelope.patch_path, patch_text)
    _atomic_write_json(temp_envelope.trial_path, trial)
    if config is not None:
        _atomic_write_json(temp_envelope.config_path, config)
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA,
        "run_id": envelope.run_id,
        "trial_id": envelope.trial_id,
        "candidate_id": envelope.candidate_id,
        "artifact_dir": str(envelope.candidate_dir),
        "trial_artifact_dir": str(Path("runs") / "trials" / envelope.trial_id),
        "primary_metric": PRIMARY_METRIC,
        "scorer_version": SCORER_VERSION,
        "official_benchmark_result": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "artifacts": _candidate_artifact_paths(envelope),
    }
    _atomic_write_json(temp_envelope.manifest_path, manifest)
    temp_candidate_dir.rename(candidate_dir)
    _update_run_candidate_count(root)
    return envelope


def write_candidate_evidence(
    envelope: CandidateEnvelope,
    *,
    preflight: dict[str, object] | None = None,
    training_manifest: dict[str, object] | None = None,
    loss_history: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    error_report: dict[str, object] | None = None,
) -> list[str]:
    """Write optional evidence payloads for a candidate."""

    wrote: list[str] = []
    for path, payload in (
        (envelope.preflight_path, preflight),
        (envelope.training_manifest_path, training_manifest),
        (envelope.loss_history_path, loss_history),
        (envelope.metrics_path, metrics),
        (envelope.error_report_path, error_report),
    ):
        if payload is not None:
            _refuse_authority_claims(payload, path.name)
            _atomic_write_json(path, payload)
            wrote.append(str(path))
    return wrote


def write_candidate_decision(
    envelope: CandidateEnvelope,
    *,
    status: str,
    matched_budget_delta: float | None,
    global_baseline_delta: float | None,
    reason: str,
    postmortem: str,
) -> dict[str, object]:
    """Write decision and postmortem evidence, then update summary/results."""

    if status not in DECISION_STATUSES:
        raise CandidateArtifactError(f"invalid candidate decision status: {status}")
    decision = {
        "schema_version": DECISION_SCHEMA,
        "run_id": envelope.run_id,
        "trial_id": envelope.trial_id,
        "candidate_id": envelope.candidate_id,
        "status": status,
        "primary_metric": PRIMARY_METRIC,
        "matched_budget_delta": matched_budget_delta,
        "global_baseline_delta": global_baseline_delta,
        "keep_threshold_delta": None,
        "reason": _require_text(reason, "reason"),
        "provisional_keep": status == "KEEP",
        "discovery_status": "UNCONFIRMED",
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "official_benchmark_result": False,
        "artifacts": _candidate_artifact_paths(envelope),
    }
    _atomic_write_json(envelope.decision_path, decision)
    _atomic_write_text(envelope.postmortem_path, _require_text(postmortem, "postmortem"))
    _update_summary_and_results(envelope, decision)
    return decision


def _candidate_artifact_paths(envelope: CandidateEnvelope) -> dict[str, str]:
    return {
        "hypothesis": str(envelope.hypothesis_path),
        "patch": str(envelope.patch_path),
        "config": str(envelope.config_path),
        "trial": str(envelope.trial_path),
        "preflight": str(envelope.preflight_path),
        "training_manifest": str(envelope.training_manifest_path),
        "loss_history": str(envelope.loss_history_path),
        "metrics": str(envelope.metrics_path),
        "error_report": str(envelope.error_report_path),
        "decision": str(envelope.decision_path),
        "postmortem": str(envelope.postmortem_path),
    }


def _require_safe_run_root(root: Path, run_id: str) -> None:
    if root.name != validate_run_id(run_id) or root.parent.name != "autoresearch" or root.parent.parent.name != "runs":
        raise CandidateArtifactError(f"autoresearch run root must be runs/autoresearch/{run_id}: {root}")
    if "runs/baseline" in root.as_posix() or "runs/trials" in root.as_posix():
        raise CandidateArtifactError(f"autoresearch artifacts must not use locked/generated run roots: {root}")


def _require_missing_dir(path: Path) -> None:
    if path.exists():
        raise CandidateArtifactError(f"candidate artifact directory already exists: {path}")


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateArtifactError(f"{label} must not be blank")
    return value


def _update_run_candidate_count(root: Path) -> None:
    manifest_path = root / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["candidate_count"] = len([path for path in (root / "candidates").iterdir() if not path.name.startswith(".")])
    _atomic_write_json(manifest_path, payload)


def _write_results(envelope: CandidateEnvelope, candidates: list[dict[str, object]]) -> None:
    tmp = (envelope.root / "results.tsv").with_suffix(".tsv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "trial_id",
                "candidate_id",
                "status",
                "primary_metric",
                "matched_budget_delta",
                "global_baseline_delta",
                "provisional_keep",
                "decision_path",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate["trial_id"],
                    candidate["candidate_id"],
                    candidate["status"],
                    PRIMARY_METRIC,
                    "" if candidate["matched_budget_delta"] is None else candidate["matched_budget_delta"],
                    "" if candidate["global_baseline_delta"] is None else candidate["global_baseline_delta"],
                    str(candidate["provisional_keep"]).lower(),
                    candidate["decision_path"],
                ]
            )
    tmp.replace(envelope.root / "results.tsv")


def _update_summary_and_results(envelope: CandidateEnvelope, decision: dict[str, object]) -> None:
    summary_path = envelope.root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = [item for item in summary.get("candidates", []) if item.get("trial_id") != envelope.trial_id]
    candidates.append(
        {
            "trial_id": envelope.trial_id,
            "status": decision["status"],
            "candidate_id": decision["candidate_id"],
            "decision_path": str(envelope.decision_path),
            "postmortem_path": str(envelope.postmortem_path),
            "matched_budget_delta": decision["matched_budget_delta"],
            "global_baseline_delta": decision["global_baseline_delta"],
            "provisional_keep": decision["provisional_keep"],
        }
    )
    _write_results(envelope, candidates)
    summary["candidates"] = candidates
    _atomic_write_json(summary_path, summary)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _refuse_authority_claims(payload: object, label: str) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in FORBIDDEN_TRUE_EVIDENCE_FLAGS and value is not False:
                raise CandidateArtifactError(f"{label} cannot claim {key}={value!r}")
            if key == "live_modal_execution" and value is not False:
                raise CandidateArtifactError(f"{label} cannot claim live_modal_execution={value!r}")
            _refuse_authority_claims(value, label)
    elif isinstance(payload, list):
        for item in payload:
            _refuse_authority_claims(item, label)


def _current_branch(repo_root: str | Path) -> str:
    head = Path(repo_root) / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref: refs/heads/"):
        return content.removeprefix("ref: refs/heads/")
    return "detached"
