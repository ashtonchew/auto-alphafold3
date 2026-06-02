"""Config validation for local auto-AlphaFold3 and pinned NanoFold configs."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

AUTO_TINY_SCHEMA_VERSION = "autoaf3.config.scaffold.v1"

NANOFOLD_REQUIRED_KEYS = frozenset(
    {
        "device",
        "use_amp",
        "detect_anomaly",
        "compile_model",
        "use_grad_checkpoint",
        "train_split",
        "residue_crop_size",
        "num_recycle",
        "single_embedding_size",
        "pair_embedding_size",
        "num_msa",
        "num_msa_samples",
        "num_msa_blocks",
        "max_templates",
        "num_pairformer_blocks",
        "diffusion_steps",
        "diffusion_batch_size",
        "num_distogram_bins",
        "clip_norm",
        "learning_rate",
        "beta1",
        "beta2",
        "optimizer_eps",
        "lr_start_factor",
        "lr_warmup",
    }
)
NANOFOLD_OPTIONAL_NONNEGATIVE_FLOAT_KEYS = frozenset(
    {
        "diffusion_loss_weight",
        "dist_loss_weight",
        "distogram_loss_weight",
        "local_calpha_geometry_loss_weight",
    }
)


class AutoTinyConfig(BaseModel):
    """Local scaffold config used before real NanoFold training is available."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = AUTO_TINY_SCHEMA_VERSION
    status: str
    description: str
    benchmark: dict[str, object]

    @property
    def max_templates(self) -> int:
        value = self.benchmark.get("max_templates")
        if value != 0:
            raise ValueError("auto_tiny scaffold must pin benchmark.max_templates=0")
        return value


class ConfigValidationResult(BaseModel):
    """Result of lightweight config validation."""

    config_kind: str
    path: str
    valid: bool
    missing_keys: list[str] = Field(default_factory=list)


def validate_config_file(config_path: str | Path, *, repo_root: str | Path = ".") -> ConfigValidationResult:
    """Validate either the local scaffold config or a NanoFold training config."""

    path = Path(config_path)
    if not path.is_absolute():
        path = Path(repo_root) / path
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_config_payload(data, source=str(config_path))


def validate_config_payload(data: object, *, source: str = "<inline-config>") -> ConfigValidationResult:
    """Validate an already-loaded scaffold or NanoFold training config payload."""

    if not isinstance(data, dict):
        return ConfigValidationResult(
            config_kind="unknown",
            path=source,
            valid=False,
            missing_keys=["json_object"],
        )
    if data.get("schema_version") == AUTO_TINY_SCHEMA_VERSION:
        config = AutoTinyConfig.model_validate(data)
        config.max_templates
        return ConfigValidationResult(config_kind="auto_tiny_scaffold", path=source, valid=True)

    missing = sorted(NANOFOLD_REQUIRED_KEYS - data.keys())
    if missing:
        return ConfigValidationResult(
            config_kind="nanofold_training",
            path=source,
            valid=False,
            missing_keys=missing,
        )
    if data.get("max_templates") != 0:
        return ConfigValidationResult(
            config_kind="nanofold_training",
            path=source,
            valid=False,
            missing_keys=["max_templates=0"],
        )
    invalid_optional = [
        key
        for key in sorted(NANOFOLD_OPTIONAL_NONNEGATIVE_FLOAT_KEYS)
        if not _is_nonnegative_finite_number(data.get(key, 0.0))
    ]
    if invalid_optional:
        return ConfigValidationResult(
            config_kind="nanofold_training",
            path=source,
            valid=False,
            missing_keys=invalid_optional,
        )
    return ConfigValidationResult(config_kind="nanofold_training", path=source, valid=True)


def _is_nonnegative_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0
