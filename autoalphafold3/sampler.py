"""Frozen-checkpoint NanoFold sampler worker helpers."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from autoalphafold3.checkpoint_training import DEFAULT_CHECKPOINT_MANIFEST, validate_checkpoint_manifest
from autoalphafold3.nanofold_adapter import NANOFOLD_PATH
from autoalphafold3.runner import (
    DONE_FILENAME,
    PREDICTIONS_FILENAME,
    artifact_manifest_shape,
    validate_trial_id,
    write_prediction_artifact,
)
from autoalphafold3.schema import PRIMARY_METRIC, SCORER_VERSION

SAMPLER_MANIFEST_SCHEMA = "autoaf3.sampler_manifest.v1"
DEFAULT_SAMPLER_NOISE_SCALE = 1.0
DEFAULT_SAMPLER_STEP_SCALE = 1.0
DEFAULT_SAMPLER_SCHEDULE_SHAPE = "linear"
DEFAULT_SAMPLER_NUM_SAMPLES = 1
DEFAULT_SAMPLER_SELECTION_POLICY = "first"
SAMPLER_SCHEDULE_SHAPES = {"linear", "cosine", "late_refine"}
SAMPLER_SELECTION_POLICIES = {"first", "geometry", "compact_geometry"}
SAMPLER_COORDINATE_NORMALIZATION_POLICIES = {"none", "ca_bond"}
SAMPLER_LOCALITY_GUARDS = {"none", "reject_exploded"}


class SamplerError(RuntimeError):
    """Raised when a sampler trial cannot honestly produce predictions."""


def run_sampler_trial(
    trial_json: dict[str, Any],
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    split: str = "public_val_small",
) -> dict[str, object]:
    """Run a bounded inference-only NanoFold sample from a frozen checkpoint."""

    import torch

    root = Path(repo_root)
    trial_id = validate_trial_id(str(trial_json.get("trial_id", "")))
    if trial_json.get("trial_kind") != "sampler":
        raise SamplerError("run_sampler_trial only accepts sampler trials")
    if trial_json.get("max_steps") is not None:
        raise SamplerError("sampler trials must not set max_steps or train")

    output = Path(output_dir)
    _require_trial_output_dir(output, trial_id)
    if output.exists() and any(output.iterdir()):
        raise SamplerError(f"sampler output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _checkpoint_path(trial_json)
    checkpoint_manifest = _load_and_validate_checkpoint_source_manifest(checkpoint_path)
    _require_checkpoint_sha(checkpoint_path, str(checkpoint_manifest["checkpoint_sha256"]))
    _require_checkpoint_matches_trial(checkpoint_manifest, checkpoint_path)

    _ensure_nanofold_import_path(root)
    from nanofold.train.chain_dataset import ChainDataset
    from nanofold.train.trainer import Trainer

    started = time.time()
    torch.manual_seed(int(trial_json.get("seed", 0)))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config") or {})
    config["device"] = _sampler_device(config)
    config["max_templates"] = 0
    _require_checkpoint_config(config)

    feature_file = _sampler_feature_file(
        features_dir=features_dir,
        checkpoint_manifest=checkpoint_manifest,
    )
    trainer = Trainer(config, loggers=[], checkpoint_save_freq=0, checkpoint=checkpoint)
    sampler_settings = _sampler_settings(trial_json)
    predictions = _sample_public_predictions(
        trainer=trainer,
        feature_file=feature_file,
        public_feature_file=Path(features_dir) / f"{split}.arrow",
        residue_crop_size=int(config["residue_crop_size"]),
        num_msa=int(config["num_msa_samples"]),
        sampler_settings=sampler_settings,
    )
    prediction_payload = write_prediction_artifact(
        trial_id=trial_id,
        split=split,
        output_dir=output,
        source="frozen_checkpoint_nanofold_sampler",
        predictions=predictions,
    )
    prediction_payload["candidate_id"] = str(trial_json.get("candidate_id", f"{trial_id}_sampler"))
    prediction_payload["max_templates"] = 0
    _atomic_write_json(output / "predictions.json", prediction_payload)

    artifact_manifest = artifact_manifest_shape(
        trial_id=trial_id,
        output_dir=output,
        features_dir=features_dir,
        split=split,
        status="REAL_MODE_UNAVAILABLE",
    )
    artifact_manifest.update(
        {
            "status": "SAMPLER_PREDICTED",
            "real_training_performed": False,
            "runner_mode": "frozen_checkpoint_sampler",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": str(checkpoint_manifest["checkpoint_sha256"]),
            "max_templates": 0,
            **sampler_settings,
            "primary_metric": PRIMARY_METRIC,
            "scorer_version": SCORER_VERSION,
            "disclaimer": (
                "Inference-only frozen-checkpoint sampler artifact. This file is "
                "not a benchmark result; scorer-only metrics determine trial status."
            ),
        }
    )
    _atomic_write_json(output / "artifact_manifest.json", artifact_manifest)
    sampler_manifest = {
        "schema_version": SAMPLER_MANIFEST_SCHEMA,
        "status": "SAMPLER_PREDICTED",
        "trial_id": trial_id,
        "candidate_id": prediction_payload["candidate_id"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint_manifest["checkpoint_sha256"]),
        "checkpoint_source_trial_id": str(checkpoint_manifest["trial_id"]),
        "feature_file": str(feature_file),
        "target_ids": [str(prediction["target_id"]) for prediction in predictions],
        "prediction_count": len(predictions),
        "real_training_performed": False,
        "inference_only": True,
        "max_templates": 0,
        **sampler_settings,
        "starts_search": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "runtime_s": time.time() - started,
    }
    _atomic_write_json(output / "sampler_manifest.json", sampler_manifest)
    _atomic_write_json(
        output / "training_log.json",
        {
            "schema_version": "autoaf3.training_log.v1",
            "trial_id": trial_id,
            "status": "SAMPLER_PREDICTED",
            "real_training_performed": False,
            "inference_only": True,
            "events": [
                {
                    "event": "frozen_checkpoint_sampler_predicted",
                    "timestamp_unix": int(started),
                    "message": "Loaded frozen NanoFold checkpoint and wrote sampler predictions.",
                }
            ],
        },
    )
    (output / "stdout.log").write_text("", encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")
    (output / "patch.diff").write_text("", encoding="utf-8")
    (output / DONE_FILENAME).write_text("frozen_checkpoint_sampler_completed\n", encoding="utf-8")
    return sampler_manifest


def run_checkpoint_prediction_artifacts(
    trial_json: dict[str, Any],
    *,
    features_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    split: str = "public_val_small",
) -> dict[str, object]:
    """Write scorer-compatible predictions from a trial-scoped checkpoint.

    This is the post-training inference step for bounded short-training trials.
    It reuses the sampler implementation but does not create a second trial or
    overwrite training artifacts.
    """

    import torch

    root = Path(repo_root)
    trial_id = validate_trial_id(str(trial_json.get("trial_id", "")))
    output = Path(output_dir)
    _require_trial_output_dir(output, trial_id)
    if not output.exists():
        raise SamplerError(f"checkpoint prediction output does not exist: {output}")

    checkpoint_path = _checkpoint_path(
        {
            **trial_json,
            "checkpoint_path": str(trial_json.get("checkpoint_path") or output / "checkpoint.pt"),
        }
    )
    checkpoint_manifest = _load_and_validate_checkpoint_source_manifest(checkpoint_path)
    _require_checkpoint_sha(checkpoint_path, str(checkpoint_manifest["checkpoint_sha256"]))
    _require_checkpoint_matches_trial(checkpoint_manifest, checkpoint_path)
    if str(checkpoint_manifest["trial_id"]) != trial_id:
        raise SamplerError("checkpoint prediction trial_id must match checkpoint manifest")

    _ensure_nanofold_import_path(root)
    from nanofold.train.trainer import Trainer

    started = time.time()
    torch.manual_seed(int(trial_json.get("seed", checkpoint_manifest.get("seed", 0))))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config") or {})
    config["device"] = _sampler_device(config)
    config["max_templates"] = 0
    _require_checkpoint_config(config)

    feature_file = _sampler_feature_file(
        features_dir=features_dir,
        checkpoint_manifest=checkpoint_manifest,
    )
    trainer = Trainer(config, loggers=[], checkpoint_save_freq=0, checkpoint=checkpoint)
    sampler_settings = _sampler_settings(trial_json)
    predictions = _sample_public_predictions(
        trainer=trainer,
        feature_file=feature_file,
        public_feature_file=Path(features_dir) / f"{split}.arrow",
        residue_crop_size=int(config["residue_crop_size"]),
        num_msa=int(config["num_msa_samples"]),
        sampler_settings=sampler_settings,
    )
    prediction_payload = write_prediction_artifact(
        trial_id=trial_id,
        split=split,
        output_dir=output,
        source="short_training_checkpoint_nanofold_sampler",
        predictions=predictions,
    )
    prediction_payload["candidate_id"] = str(
        trial_json.get("candidate_id", checkpoint_manifest.get("candidate_id", trial_id))
    )
    prediction_payload["max_templates"] = 0
    _atomic_write_json(output / PREDICTIONS_FILENAME, prediction_payload)

    artifact_manifest_path = output / "artifact_manifest.json"
    if artifact_manifest_path.exists():
        artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    else:
        artifact_manifest = artifact_manifest_shape(
            trial_id=trial_id,
            output_dir=output,
            features_dir=features_dir,
            split=split,
            status="SAMPLER_PREDICTED",
        )
    artifacts = dict(artifact_manifest.get("artifacts") or {})
    artifacts["predictions_json"] = str(output / PREDICTIONS_FILENAME)
    artifacts["sampler_manifest_json"] = str(output / "sampler_manifest.json")
    artifact_manifest.update(
        {
            "status": "SHORT_TRAINING_PREDICTED",
            "runner_mode": "short_training_checkpoint_sampler",
            "split": split,
            "artifacts": artifacts,
            "predictions_ready": True,
            "prediction_source": "short_training_checkpoint_nanofold_sampler",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": str(checkpoint_manifest["checkpoint_sha256"]),
            "max_templates": 0,
            **sampler_settings,
            "primary_metric": PRIMARY_METRIC,
            "scorer_version": SCORER_VERSION,
        }
    )
    lifecycle = dict(artifact_manifest.get("lifecycle") or {})
    lifecycle["predictions_ready"] = True
    artifact_manifest["lifecycle"] = lifecycle
    _atomic_write_json(artifact_manifest_path, artifact_manifest)

    sampler_manifest = {
        "schema_version": SAMPLER_MANIFEST_SCHEMA,
        "status": "SHORT_TRAINING_PREDICTED",
        "trial_id": trial_id,
        "candidate_id": prediction_payload["candidate_id"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint_manifest["checkpoint_sha256"]),
        "checkpoint_source_trial_id": str(checkpoint_manifest["trial_id"]),
        "checkpoint_manifest_schema": str(checkpoint_manifest["schema_version"]),
        "feature_file": str(feature_file),
        "target_ids": [str(prediction["target_id"]) for prediction in predictions],
        "prediction_count": len(predictions),
        "real_training_performed": bool(checkpoint_manifest.get("real_training_performed") is True),
        "inference_only": True,
        "max_templates": 0,
        **sampler_settings,
        "starts_search": False,
        "writes_baseline": False,
        "writes_ledger": False,
        "writes_discovery_ledger": False,
        "runtime_s": time.time() - started,
    }
    _atomic_write_json(output / "sampler_manifest.json", sampler_manifest)
    return sampler_manifest


def _sample_public_predictions(
    *,
    trainer: Any,
    feature_file: Path,
    public_feature_file: Path,
    residue_crop_size: int,
    num_msa: int,
    sampler_settings: dict[str, object],
) -> list[dict[str, object]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc
    from nanofold.train.chain_dataset import ChainDataset

    with pa.memory_map(str(feature_file)) as source:
        with ipc.open_file(source) as reader:
            nanofold_table = reader.read_all()
    nanofold_table = nanofold_table.append_column(
        "length",
        pc.list_value_length(nanofold_table["positions"]),
    )
    if public_feature_file.exists():
        with pa.memory_map(str(public_feature_file)) as source:
            with ipc.open_file(source) as reader:
                public_rows = reader.read_all().to_pylist()
    else:
        public_rows = _fallback_public_rows(nanofold_table)

    index = _nanofold_feature_index(nanofold_table)
    predictions: list[dict[str, object]] = []
    for fallback_index, public_row in enumerate(public_rows):
        target_id = str(public_row.get("record_id") or f"{public_row.get('pdb_id')}_{public_row.get('chain_id')}")
        sequence = str(public_row.get("sequence", ""))
        feature_index = _select_nanofold_row(index, public_row, fallback_index=fallback_index)
        dataset = ChainDataset(nanofold_table, [feature_index], residue_crop_size, num_msa)
        batch = trainer.load_batch(next(iter(dataset)))
        predicted_ca, selection = _sample_selected_ca_coordinates(
            trainer.model,
            batch,
            sampler_settings=sampler_settings,
        )
        target_len = int(public_row.get("sequence_length") or len(sequence) or len(predicted_ca))
        predictions.append(
            {
                "target_id": target_id,
                "predicted_ca": _resize_ca_trace(predicted_ca, target_len),
                "selection": selection,
            }
        )
    return predictions


def _sample_selected_ca_coordinates(
    model: Any,
    features: dict[str, Any],
    *,
    sampler_settings: dict[str, object],
) -> tuple[list[list[float]], dict[str, object]]:
    num_samples = int(sampler_settings["sampler_num_samples"])
    policy = str(sampler_settings["sampler_selection_policy"])
    coordinate_normalization = str(sampler_settings["sampler_coordinate_normalization"])
    locality_guard = str(sampler_settings["sampler_locality_guard"])
    candidates: list[tuple[float, int, list[list[float]], list[str]]] = []
    rejected: list[dict[str, object]] = []
    for sample_index in range(num_samples):
        raw_ca = _sample_ca_coordinates(model, features, sampler_settings=sampler_settings)
        ca = _normalize_ca_coordinates(
            raw_ca,
            policy=coordinate_normalization,
            coordinate_scale=float(sampler_settings["sampler_coordinate_scale"]),
        )
        locality_flags = _ca_locality_flags(ca)
        if locality_guard == "reject_exploded" and locality_flags:
            rejected.append({"sample_index": sample_index, "locality_flags": locality_flags})
            continue
        quality = _label_free_ca_quality(ca, policy=policy)
        candidates.append((quality, sample_index, ca, locality_flags))
        if policy == "first":
            break
    if not candidates:
        raise SamplerError(
            "sampler_locality_guard rejected all samples for label-free geometry collapse: "
            + json.dumps(rejected, sort_keys=True)
        )
    quality, selected_index, selected, locality_flags = min(candidates, key=lambda row: (row[0], row[1]))
    return selected, {
        "policy": policy,
        "num_samples": num_samples,
        "selected_index": selected_index,
        "label_free_quality": quality,
        "coordinate_normalization": coordinate_normalization,
        "locality_guard": locality_guard,
        "locality_flags": locality_flags,
        "rejected_sample_count": len(rejected),
    }


def _sample_ca_coordinates(
    model: Any,
    features: dict[str, Any],
    *,
    sampler_settings: dict[str, object],
) -> list[list[float]]:
    import torch

    sampler_steps = int(sampler_settings["sampler_steps"])
    if sampler_steps < 1:
        raise SamplerError("sampler_steps must be at least 1")
    model.eval()
    with torch.no_grad():
        model.diffusion_model.inference = True
        _install_sampler_schedule(
            model.diffusion_model,
            sampler_steps=sampler_steps,
            noise_scale=float(sampler_settings["sampler_noise_scale"]),
            step_scale=float(sampler_settings["sampler_step_scale"]),
            schedule_shape=str(sampler_settings["sampler_schedule_shape"]),
        )
        input_embedding, single_rep_init, pair_rep_init = model.nanofold_input(features)
        single_rep_prev = torch.zeros_like(single_rep_init)
        pair_rep_prev = torch.zeros_like(pair_rep_init)
        num_recycle = int(model.num_recycle)
        for _ in range(num_recycle):
            single_rep, pair_rep = model.nanofold_trunk(
                features,
                input_embedding,
                pair_rep_init,
                single_rep_init,
                pair_rep_prev,
                single_rep_prev,
            )
            single_rep_prev, pair_rep_prev = single_rep, pair_rep
        sampled = model.diffusion_model.sample_diffusion(features, input_embedding, single_rep, pair_rep)
    if sampled.ndim != 2 or sampled.shape[-1] != 3:
        raise SamplerError(f"sampler returned unexpected coordinate shape: {tuple(sampled.shape)}")
    local_coords = features["local_coords"]
    residue_count = int(local_coords.shape[0])
    atoms_per_residue = int(local_coords.shape[1])
    if atoms_per_residue < 2:
        raise SamplerError("sampler features do not contain C-alpha atom positions")
    expected_atoms = residue_count * atoms_per_residue
    if int(sampled.shape[0]) != expected_atoms:
        raise SamplerError(f"sampler returned {sampled.shape[0]} atoms for expected {expected_atoms}")
    ca = sampled.reshape(residue_count, atoms_per_residue, 3)[:, 1, :]
    values = ca.detach().cpu().to(torch.float64).tolist()
    for row in values:
        if len(row) != 3 or not all(math.isfinite(float(coord)) for coord in row):
            raise SamplerError("sampler produced non-finite C-alpha coordinates")
    return [[float(coord) for coord in row] for row in values]


def _install_sampler_schedule(
    diffusion_model: Any,
    *,
    sampler_steps: int,
    noise_scale: float = DEFAULT_SAMPLER_NOISE_SCALE,
    step_scale: float = DEFAULT_SAMPLER_STEP_SCALE,
    schedule_shape: str = DEFAULT_SAMPLER_SCHEDULE_SHAPE,
) -> None:
    import torch

    if sampler_steps < 1:
        raise SamplerError("sampler_steps must be at least 1")
    if not 0.25 <= noise_scale <= 2.0:
        raise SamplerError("sampler_noise_scale must be in [0.25, 2.0]")
    if not 0.25 <= step_scale <= 2.0:
        raise SamplerError("sampler_step_scale must be in [0.25, 2.0]")
    if schedule_shape not in SAMPLER_SCHEDULE_SHAPES:
        raise SamplerError(f"unsupported sampler_schedule_shape: {schedule_shape}")

    s_max = 160 * noise_scale
    s_min = 0.0004
    p = 7
    schedule_points = max(2, int(round((sampler_steps + 1) * step_scale)))
    raw_steps = torch.arange(schedule_points, device=next(diffusion_model.parameters()).device) / (schedule_points - 1)
    if schedule_shape == "cosine":
        steps = 0.5 - 0.5 * torch.cos(raw_steps * math.pi)
    elif schedule_shape == "late_refine":
        steps = raw_steps**1.5
    else:
        steps = raw_steps
    schedule = diffusion_model.data_std_dev * (
        s_max ** (1 / p)
        + steps * (s_min ** (1 / p) - s_max ** (1 / p))
    ) ** p
    diffusion_model.schedule = torch.cat([schedule, torch.zeros_like(schedule[:1])])


def _sampler_settings(trial_json: dict[str, Any]) -> dict[str, object]:
    settings = {
        "sampler_steps": int(trial_json.get("sampler_steps", 1)),
        "sampler_noise_scale": float(trial_json.get("sampler_noise_scale", DEFAULT_SAMPLER_NOISE_SCALE)),
        "sampler_step_scale": float(trial_json.get("sampler_step_scale", DEFAULT_SAMPLER_STEP_SCALE)),
        "sampler_schedule_shape": str(trial_json.get("sampler_schedule_shape", DEFAULT_SAMPLER_SCHEDULE_SHAPE)),
        "sampler_num_samples": int(trial_json.get("sampler_num_samples", DEFAULT_SAMPLER_NUM_SAMPLES)),
        "sampler_selection_policy": str(
            trial_json.get("sampler_selection_policy", DEFAULT_SAMPLER_SELECTION_POLICY)
        ),
        "sampler_coordinate_normalization": str(trial_json.get("sampler_coordinate_normalization", "none")),
        "sampler_coordinate_scale": float(trial_json.get("sampler_coordinate_scale", 1.0)),
        "sampler_locality_guard": str(trial_json.get("sampler_locality_guard", "none")),
    }
    _validate_sampler_settings(settings)
    return settings


def _validate_sampler_settings(settings: dict[str, object]) -> None:
    if not 1 <= int(settings["sampler_steps"]) <= 12:
        raise SamplerError("sampler_steps must be in [1, 12]")
    if not 0.25 <= float(settings["sampler_noise_scale"]) <= 2.0:
        raise SamplerError("sampler_noise_scale must be in [0.25, 2.0]")
    if not 0.25 <= float(settings["sampler_step_scale"]) <= 2.0:
        raise SamplerError("sampler_step_scale must be in [0.25, 2.0]")
    if str(settings["sampler_schedule_shape"]) not in SAMPLER_SCHEDULE_SHAPES:
        raise SamplerError("sampler_schedule_shape must be linear, cosine, or late_refine")
    if not 1 <= int(settings["sampler_num_samples"]) <= 4:
        raise SamplerError("sampler_num_samples must be in [1, 4]")
    if str(settings["sampler_selection_policy"]) not in SAMPLER_SELECTION_POLICIES:
        raise SamplerError("sampler_selection_policy must be first, geometry, or compact_geometry")
    if str(settings["sampler_coordinate_normalization"]) not in SAMPLER_COORDINATE_NORMALIZATION_POLICIES:
        raise SamplerError("sampler_coordinate_normalization must be none or ca_bond")
    if str(settings["sampler_locality_guard"]) not in SAMPLER_LOCALITY_GUARDS:
        raise SamplerError("sampler_locality_guard must be none or reject_exploded")
    if not 0.0 < float(settings["sampler_coordinate_scale"]) <= 20.0:
        raise SamplerError("sampler_coordinate_scale must be in (0, 20]")
    if (
        float(settings["sampler_coordinate_scale"]) != 1.0
        and str(settings["sampler_coordinate_normalization"]) != "ca_bond"
    ):
        raise SamplerError("sampler_coordinate_scale requires sampler_coordinate_normalization=ca_bond")


def _normalize_ca_coordinates(
    ca: list[list[float]],
    *,
    policy: str,
    coordinate_scale: float = 1.0,
) -> list[list[float]]:
    if not 0.0 < coordinate_scale <= 20.0:
        raise SamplerError("sampler_coordinate_scale must be in (0, 20]")
    if policy == "none":
        if coordinate_scale != 1.0:
            raise SamplerError("sampler_coordinate_scale requires sampler_coordinate_normalization=ca_bond")
        return ca
    if policy != "ca_bond":
        raise SamplerError("unsupported sampler coordinate normalization policy")
    if len(ca) < 2:
        return [[float(coord) * coordinate_scale for coord in row] for row in ca]
    center = [sum(row[axis] for row in ca) / len(ca) for axis in range(3)]
    centered = [[float(row[axis]) - center[axis] for axis in range(3)] for row in ca]
    distances = [_distance(centered[index], centered[index - 1]) for index in range(1, len(centered))]
    positive_distances = [distance for distance in distances if math.isfinite(distance) and distance > 0.0]
    if not positive_distances:
        return [[coord * coordinate_scale for coord in row] for row in centered]
    scale = 3.8 / (sum(positive_distances) / len(positive_distances))
    return [[coord * scale * coordinate_scale for coord in row] for row in centered]


def _label_free_ca_quality(ca: list[list[float]], *, policy: str) -> float:
    if policy == "first":
        return 0.0
    if len(ca) < 2:
        return float("inf")
    distances = [_distance(ca[index], ca[index - 1]) for index in range(1, len(ca))]
    bond_penalty = sum((distance - 3.8) ** 2 for distance in distances) / len(distances)
    jump_penalty = sum(max(0.0, distance - 8.0) ** 2 for distance in distances) / len(distances)
    smooth_penalty = 0.0
    if len(ca) >= 3:
        angles = []
        for index in range(2, len(ca)):
            prev_vec = _vector(ca[index - 1], ca[index - 2])
            next_vec = _vector(ca[index], ca[index - 1])
            angles.append((_norm(next_vec) - _norm(prev_vec)) ** 2)
        smooth_penalty = sum(angles) / len(angles)
    compact_penalty = 0.0
    if policy == "compact_geometry":
        center = [sum(row[axis] for row in ca) / len(ca) for axis in range(3)]
        radius = sum(_distance(row, center) for row in ca) / len(ca)
        compact_penalty = max(0.0, radius - 25.0) / 25.0
    return bond_penalty + jump_penalty + 0.25 * smooth_penalty + compact_penalty


def _ca_locality_flags(ca: list[list[float]]) -> list[str]:
    if not ca:
        return ["empty_ca_trace"]
    flags: list[str] = []
    adjacent = [_distance(ca[index], ca[index - 1]) for index in range(1, len(ca))]
    if any(not math.isfinite(distance) for distance in adjacent):
        flags.append("non_finite_adjacent_ca_distance")
    finite_adjacent = [distance for distance in adjacent if math.isfinite(distance)]
    if finite_adjacent and max(finite_adjacent) > 30.0:
        flags.append("adjacent_ca_distance_outlier_gt_30A")
    if finite_adjacent and (sum(finite_adjacent) / len(finite_adjacent)) > 30.0:
        flags.append("adjacent_ca_distance_exploded")
    pair_distances = [
        _distance(ca[right], ca[left])
        for left in range(len(ca))
        for right in range(left + 1, len(ca))
    ]
    if any(not math.isfinite(distance) for distance in pair_distances):
        flags.append("non_finite_pair_distance")
    finite_pairs = [distance for distance in pair_distances if math.isfinite(distance)]
    if finite_pairs and max(finite_pairs) > 500.0:
        flags.append("pair_distance_outlier_gt_500A")
    if finite_pairs and (sum(finite_pairs) / len(finite_pairs)) > 500.0:
        flags.append("pair_distance_exploded")
    return flags


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(left[axis]) - float(right[axis])) ** 2 for axis in range(3)))


def _vector(left: list[float], right: list[float]) -> list[float]:
    return [float(left[axis]) - float(right[axis]) for axis in range(3)]


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _sampler_device(config: dict[str, Any]) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard.
        raise SamplerError("torch is required for sampler execution") from exc
    configured = str(config.get("device", "cpu"))
    if configured == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return configured


def _checkpoint_path(trial_json: dict[str, Any]) -> Path:
    value = trial_json.get("checkpoint_path")
    if not isinstance(value, str) or not value:
        raise SamplerError("sampler trials require checkpoint_path")
    path = Path(value)
    if path.name != "checkpoint.pt" or "runs" not in path.parts or "trials" not in path.parts:
        raise SamplerError("checkpoint_path must point at a trial-scoped checkpoint.pt")
    if not path.exists():
        raise SamplerError(f"checkpoint_path does not exist: {path}")
    return path


def _load_and_validate_checkpoint_manifest(checkpoint_path: Path) -> dict[str, object]:
    manifest_path = checkpoint_path.parent / DEFAULT_CHECKPOINT_MANIFEST
    if not manifest_path.exists():
        raise SamplerError(f"checkpoint manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        return validate_checkpoint_manifest(payload)
    except Exception as exc:  # noqa: BLE001 - normalize checkpoint validation failures.
        raise SamplerError(f"checkpoint manifest is invalid: {exc}") from exc


def _load_and_validate_checkpoint_source_manifest(checkpoint_path: Path) -> dict[str, object]:
    checkpoint_manifest_path = checkpoint_path.parent / DEFAULT_CHECKPOINT_MANIFEST
    if checkpoint_manifest_path.exists():
        return _load_and_validate_checkpoint_manifest(checkpoint_path)

    from autoalphafold3.short_training import DEFAULT_SHORT_TRAINING_MANIFEST, validate_short_training_manifest

    short_manifest_path = checkpoint_path.parent / DEFAULT_SHORT_TRAINING_MANIFEST
    if not short_manifest_path.exists():
        raise SamplerError(f"checkpoint source manifest is missing: {checkpoint_manifest_path}")
    payload = json.loads(short_manifest_path.read_text(encoding="utf-8"))
    try:
        return validate_short_training_manifest(payload)
    except Exception as exc:  # noqa: BLE001 - normalize checkpoint validation failures.
        raise SamplerError(f"short-training checkpoint manifest is invalid: {exc}") from exc


def _require_checkpoint_sha(path: Path, expected: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise SamplerError(f"checkpoint sha256 mismatch: expected {expected}, got {actual}")


def _require_checkpoint_matches_trial(manifest: dict[str, object], checkpoint_path: Path) -> None:
    manifest_path = Path(str(manifest["checkpoint_path"]))
    if manifest_path != checkpoint_path:
        raise SamplerError(f"checkpoint_path does not match manifest: {checkpoint_path} != {manifest_path}")


def _require_checkpoint_config(config: dict[str, Any]) -> None:
    if config.get("max_templates") != 0:
        raise SamplerError("sampler checkpoint config must preserve max_templates=0")
    if not config:
        raise SamplerError("checkpoint is missing NanoFold config")


def _sampler_feature_file(*, features_dir: str | Path, checkpoint_manifest: dict[str, object]) -> Path:
    manifest_feature = Path(str(checkpoint_manifest["features_path"]))
    feature_file = manifest_feature if manifest_feature.is_absolute() else Path(features_dir) / manifest_feature
    if not feature_file.exists():
        feature_file = Path(features_dir) / manifest_feature.name
    if not feature_file.exists():
        raise SamplerError(f"sampler feature file is missing: {feature_file}")
    return feature_file


def _fallback_public_rows(table: Any) -> list[dict[str, object]]:
    rows = []
    for index, row in enumerate(table.to_pylist()):
        structure_id = str(row.get("structure_id") or f"target_{index}")
        chain_id = str(row.get("chain_id") or "A")
        sequence = str(row.get("sequence") or "")
        rows.append(
            {
                "record_id": f"{structure_id.upper()}_{chain_id}",
                "pdb_id": structure_id.upper(),
                "chain_id": chain_id,
                "sequence": sequence,
                "sequence_length": len(sequence),
            }
        )
    return rows


def _nanofold_feature_index(table: Any) -> dict[str, dict[object, int]]:
    by_sequence: dict[object, int] = {}
    by_structure: dict[object, int] = {}
    for index, row in enumerate(table.to_pylist()):
        sequence = str(row.get("sequence") or "")
        if sequence and sequence not in by_sequence:
            by_sequence[sequence] = index
        structure_id = str(row.get("structure_id") or "").lower()
        chain_id = str(row.get("chain_id") or "")
        if structure_id and chain_id:
            by_structure[(structure_id, chain_id)] = index
    return {"sequence": by_sequence, "structure": by_structure}


def _select_nanofold_row(index: dict[str, dict[object, int]], public_row: dict[str, object], *, fallback_index: int) -> int:
    sequence = str(public_row.get("sequence") or "")
    by_sequence = index["sequence"]
    if sequence in by_sequence:
        return by_sequence[sequence]
    pdb_id = str(public_row.get("pdb_id") or str(public_row.get("record_id", "")).split("_", 1)[0]).lower()
    chain_id = str(public_row.get("chain_id") or "A")
    by_structure = index["structure"]
    if (pdb_id, chain_id) in by_structure:
        return by_structure[(pdb_id, chain_id)]
    if not by_sequence:
        raise SamplerError("NanoFold feature file has no usable rows")
    return fallback_index % len(by_sequence)


def _resize_ca_trace(predicted_ca: list[list[float]], target_len: int) -> list[list[float]]:
    if target_len <= 0:
        raise SamplerError("target sequence length must be positive")
    if len(predicted_ca) == target_len:
        return predicted_ca
    if len(predicted_ca) == 1:
        return [predicted_ca[0] for _ in range(target_len)]
    resized = []
    for idx in range(target_len):
        source_idx = round(idx * (len(predicted_ca) - 1) / max(1, target_len - 1))
        resized.append(predicted_ca[source_idx])
    return resized


def _require_trial_output_dir(path: Path, trial_id: str) -> None:
    as_posix = path.as_posix()
    if "runs/baseline" in as_posix:
        raise SamplerError("sampler must not write runs/baseline")
    if path.name != trial_id or "runs/trials" not in as_posix:
        raise SamplerError(f"sampler output must be trial-scoped under runs/trials/{trial_id}: {path}")


def _ensure_nanofold_import_path(repo_root: Path) -> None:
    nanofold_root = str(repo_root / NANOFOLD_PATH)
    if nanofold_root not in sys.path:
        sys.path.insert(0, nanofold_root)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
