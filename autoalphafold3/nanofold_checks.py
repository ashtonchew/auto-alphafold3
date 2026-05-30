"""Dependency-aware NanoFold preflight checks.

These checks do not download data, rebuild feature artifacts, or require GPUs. When
optional NanoFold dependencies are unavailable, they return explicit skipped
gate results so preflight evidence remains honest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from autoalphafold3.config_contract import validate_config_file
from autoalphafold3.local_fixtures import default_fixture_path, validate_local_nanofold_fixture
from autoalphafold3.nanofold_adapter import NANOFOLD_PATH, import_smoke_summary, load_nanofold_config

GateStatus = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True)
class NanoFoldGateResult:
    """Result for one NanoFold-aware preflight gate."""

    name: str
    status: GateStatus
    reason: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_nanofold_preflight_gates(
    *,
    config_path: str | Path,
    repo_root: str | Path = ".",
) -> list[NanoFoldGateResult]:
    """Run NanoFold-aware checks without downloading data or running training."""

    import_summary = import_smoke_summary(repo_root=repo_root)
    return [
        parameter_count_gate(config_path=config_path, repo_root=repo_root, import_summary=import_summary),
        tiny_forward_gate(config_path=config_path, repo_root=repo_root, import_summary=import_summary),
        finite_loss_gate(config_path=config_path, repo_root=repo_root, import_summary=import_summary),
    ]


def parameter_count_gate(
    *,
    config_path: str | Path,
    repo_root: str | Path = ".",
    import_summary: dict[str, object] | None = None,
) -> NanoFoldGateResult:
    """Count parameters if PyTorch and NanoFold training modules are importable."""

    config_result = validate_config_file(config_path, repo_root=repo_root)
    if not config_result.valid:
        return NanoFoldGateResult(
            name="parameter_count",
            status="failed",
            reason="config_invalid",
            details={"missing_keys": config_result.missing_keys},
        )
    if config_result.config_kind != "nanofold_training":
        return NanoFoldGateResult(
            name="parameter_count",
            status="skipped",
            reason="not_nanofold_training_config",
            details={"config_kind": config_result.config_kind},
        )

    if import_summary is None:
        import_summary = import_smoke_summary(repo_root=repo_root)
    model_import = _module_status(import_summary, "nanofold.train.model.nanofold")
    if not model_import.get("ok"):
        return NanoFoldGateResult(
            name="parameter_count",
            status="skipped",
            reason="dependency_missing",
            details={"module": model_import},
        )

    root = Path(repo_root)
    try:
        import sys

        nanofold_root = str(root / NANOFOLD_PATH)
        if nanofold_root not in sys.path:
            sys.path.insert(0, nanofold_root)
        from nanofold.train.model.nanofold import Nanofold

        config = load_nanofold_config(config_path, repo_root=repo_root)
        model = Nanofold.from_config(config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    except Exception as exc:  # noqa: BLE001 - preflight must turn import/runtime failures into evidence.
        return NanoFoldGateResult(
            name="parameter_count",
            status="failed",
            reason="parameter_count_error",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )

    return NanoFoldGateResult(
        name="parameter_count",
        status="passed",
        reason="counted",
        details={"parameter_count": int(parameter_count)},
    )


def tiny_forward_gate(
    *,
    config_path: str | Path = "configs/nanofold_dev_cpu_smoke.json",
    repo_root: str | Path = ".",
    import_summary: dict[str, object] | None = None,
) -> NanoFoldGateResult:
    """Run a tiny NanoFold forward pass when an approved local fixture exists."""

    if import_summary is None:
        import_summary = import_smoke_summary(repo_root=repo_root)
    model_import = _module_status(import_summary, "nanofold.train.model.nanofold")
    if not model_import.get("ok"):
        return NanoFoldGateResult(
            name="tiny_forward",
            status="skipped",
            reason="dependency_missing",
            details={"module": model_import},
        )
    fixture_report = validate_local_nanofold_fixture(
        fixture_path=default_fixture_path(repo_root),
        repo_root=repo_root,
    )
    if fixture_report.status != "PASS":
        return NanoFoldGateResult(
            name="tiny_forward",
            status="skipped",
            reason="feature_fixture_not_available_without_cached_arrow",
            details={"fixture": fixture_report.to_dict()},
        )
    try:
        loss_report = _run_tiny_nanofold_loss(config_path=config_path, repo_root=repo_root)
    except Exception as exc:  # noqa: BLE001 - preflight must report runtime failures as evidence.
        return NanoFoldGateResult(
            name="tiny_forward",
            status="failed",
            reason="tiny_forward_error",
            details={"error_type": type(exc).__name__, "error": str(exc), "fixture": fixture_report.to_dict()},
        )
    return NanoFoldGateResult(
        name="tiny_forward",
        status="passed",
        reason="forward_loss_finite",
        details={"fixture": fixture_report.to_dict(), "losses": loss_report},
    )


def finite_loss_gate(
    *,
    config_path: str | Path = "configs/nanofold_dev_cpu_smoke.json",
    repo_root: str | Path = ".",
    import_summary: dict[str, object] | None = None,
) -> NanoFoldGateResult:
    """Run a one-batch finite-loss gate when an approved local fixture exists."""

    if import_summary is None:
        import_summary = import_smoke_summary(repo_root=repo_root)
    trainer_import = _module_status(import_summary, "nanofold.train.trainer")
    if not trainer_import.get("ok"):
        return NanoFoldGateResult(
            name="finite_loss",
            status="skipped",
            reason="dependency_missing",
            details={"module": trainer_import},
        )
    fixture_report = validate_local_nanofold_fixture(
        fixture_path=default_fixture_path(repo_root),
        repo_root=repo_root,
    )
    if fixture_report.status != "PASS":
        return NanoFoldGateResult(
            name="finite_loss",
            status="skipped",
            reason="feature_fixture_not_available_without_cached_arrow",
            details={"fixture": fixture_report.to_dict()},
        )
    try:
        loss_report = _run_tiny_nanofold_loss(config_path=config_path, repo_root=repo_root)
    except Exception as exc:  # noqa: BLE001 - preflight must report runtime failures as evidence.
        return NanoFoldGateResult(
            name="finite_loss",
            status="failed",
            reason="finite_loss_error",
            details={"error_type": type(exc).__name__, "error": str(exc), "fixture": fixture_report.to_dict()},
        )
    return NanoFoldGateResult(
        name="finite_loss",
        status="passed",
        reason="total_loss_finite",
        details={"fixture": fixture_report.to_dict(), "losses": loss_report},
    )


def _module_status(import_summary: dict[str, object], module: str) -> dict[str, object]:
    imports = import_summary.get("imports", [])
    for row in imports:
        if isinstance(row, dict) and row.get("module") == module:
            return row
    return {"module": module, "ok": False, "error_type": "NotFound", "error": "module status missing"}


def _run_tiny_nanofold_loss(*, config_path: str | Path, repo_root: str | Path) -> dict[str, float]:
    import math
    import sys

    import torch

    root = Path(repo_root)
    nanofold_root = str(root / NANOFOLD_PATH)
    if nanofold_root not in sys.path:
        sys.path.insert(0, nanofold_root)

    from nanofold.train.chain_dataset import ChainDataset
    from nanofold.train.model.nanofold import Nanofold

    torch.manual_seed(0)
    config = load_nanofold_config(config_path, repo_root=root)
    train, _held_out = ChainDataset.construct_datasets(
        default_fixture_path(root),
        0.5,
        config["residue_crop_size"],
        config["num_msa_samples"],
    )
    features = next(iter(train))
    model = Nanofold.from_config(config)
    model.train()
    losses = model(features)
    report: dict[str, float] = {}
    for key in ("mse_loss", "lddt_loss", "diffusion_loss", "dist_loss", "total_loss"):
        value = float(losses[key].detach().cpu())
        if not math.isfinite(value):
            raise ValueError(f"{key} is not finite")
        report[key] = value
    return report
