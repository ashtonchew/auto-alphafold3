from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from autoalphafold3.config_contract import validate_config_file
from autoalphafold3.local_fixtures import APPROVAL_TOKEN, materialize_local_nanofold_fixture
from autoalphafold3.patch_policy import validate_patch_scope
from autoalphafold3.short_training import run_short_nanofold_training, short_training_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
NANOFOLD_ROOT = REPO_ROOT / "external/nanofold"
if str(NANOFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(NANOFOLD_ROOT))

import torch  # noqa: E402
from nanofold.train.loss import compute_diffusion_loss, compute_local_calpha_geometry_loss, extract_calpha_coords  # noqa: E402
from nanofold.train.model.nanofold import Nanofold  # noqa: E402

try:
    sys.path.remove(str(NANOFOLD_ROOT))
except ValueError:
    pass


def nanofold_loss_stub(
    *,
    diffusion_loss_weight: float = 4.0,
    distogram_loss_weight: float = 0.03,
    local_calpha_geometry_loss_weight: float = 0.0,
) -> Nanofold:
    model = object.__new__(Nanofold)
    model.diffusion_loss_weight = diffusion_loss_weight
    model.distogram_loss_weight = distogram_loss_weight
    model.local_calpha_geometry_loss_weight = local_calpha_geometry_loss_weight
    return model


def test_default_total_loss_matches_original_formula() -> None:
    model = nanofold_loss_stub()
    diffusion_loss = torch.tensor(2.0)
    dist_loss = torch.tensor(5.0)
    geometry_loss = torch.tensor(100.0)

    total = model.get_total_loss(diffusion_loss, dist_loss, geometry_loss)

    assert torch.allclose(total, 4 * diffusion_loss + 0.03 * dist_loss)


def test_custom_loss_weights_change_total_loss() -> None:
    model = nanofold_loss_stub(
        diffusion_loss_weight=2.0,
        distogram_loss_weight=0.5,
        local_calpha_geometry_loss_weight=0.25,
    )

    total = model.get_total_loss(torch.tensor(3.0), torch.tensor(4.0), torch.tensor(8.0))

    assert torch.allclose(total, torch.tensor(10.0))


def test_nanofold_get_args_defaults_preserve_zero_geometry_weight() -> None:
    config = json.loads((REPO_ROOT / "configs/nanofold_dev_cpu_smoke.json").read_text(encoding="utf-8"))

    args = Nanofold.get_args(config)

    assert args["diffusion_loss_weight"] == 4.0
    assert args["distogram_loss_weight"] == 0.03
    assert args["local_calpha_geometry_loss_weight"] == 0.0


def test_nanofold_get_args_reads_experiment_loss_weights() -> None:
    config = json.loads(
        (REPO_ROOT / "configs/experiments/local_calpha_geometry_smoke.json").read_text(encoding="utf-8")
    )

    args = Nanofold.get_args(config)

    assert args["diffusion_loss_weight"] == 4.0
    assert args["distogram_loss_weight"] == 0.03
    assert args["local_calpha_geometry_loss_weight"] == 0.25


def test_nanofold_config_rejects_invalid_loss_weights(tmp_path: Path) -> None:
    config = json.loads((REPO_ROOT / "configs/nanofold_dev_cpu_smoke.json").read_text(encoding="utf-8"))
    config["local_calpha_geometry_loss_weight"] = -0.1
    config_path = tmp_path / "bad_loss_weight.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = validate_config_file(config_path)

    assert not result.valid
    assert result.missing_keys == ["local_calpha_geometry_loss_weight"]


def test_nanofold_config_rejects_invalid_legacy_dist_loss_weight(tmp_path: Path) -> None:
    config = json.loads((REPO_ROOT / "configs/nanofold_dev_cpu_smoke.json").read_text(encoding="utf-8"))
    config["dist_loss_weight"] = -1.0
    config_path = tmp_path / "bad_legacy_loss_weight.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = validate_config_file(config_path)

    assert not result.valid
    assert result.missing_keys == ["dist_loss_weight"]


def test_diffusion_loss_skips_geometry_when_disabled_for_defaults() -> None:
    predicted = torch.zeros(1, 4, 3)
    target = torch.ones(1, 4, 3)
    t = torch.ones(1, 1, 1)

    losses = compute_diffusion_loss(predicted, target, t, data_std_dev=16, compute_local_geometry=False)

    assert losses["local_calpha_geometry_loss"].item() == 0.0


def test_local_calpha_geometry_loss_is_differentiable() -> None:
    predicted = torch.tensor(
        [
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [4.0, -1.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
                [8.5, -1.0, 0.0],
                [8.5, 0.0, 0.0],
                [8.5, 1.0, 0.0],
                [12.0, -1.0, 0.0],
                [12.0, 0.0, 0.0],
                [12.0, 1.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [3.8, -1.0, 0.0],
                [3.8, 0.0, 0.0],
                [3.8, 1.0, 0.0],
                [7.6, -1.0, 0.0],
                [7.6, 0.0, 0.0],
                [7.6, 1.0, 0.0],
                [11.4, -1.0, 0.0],
                [11.4, 0.0, 0.0],
                [11.4, 1.0, 0.0],
            ]
        ],
    )

    loss = compute_local_calpha_geometry_loss(predicted, target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_local_calpha_geometry_loss_known_cutoff_value() -> None:
    target = torch.tensor(
        [
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, -1.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
                [30.0, -1.0, 0.0],
                [30.0, 0.0, 0.0],
                [30.0, 1.0, 0.0],
            ]
        ]
    )
    predicted = target.clone()
    predicted[:, 4, 0] = 12.0
    predicted[:, 7, 0] = 60.0

    loss = compute_local_calpha_geometry_loss(predicted, target)

    assert torch.allclose(loss, torch.tensor(1.5))


def test_local_calpha_geometry_loss_handles_empty_local_mask() -> None:
    predicted = torch.tensor([[[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    loss = compute_local_calpha_geometry_loss(predicted, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_local_calpha_geometry_loss_ignores_non_calpha_perturbations() -> None:
    target = torch.tensor(
        [
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [3.8, -1.0, 0.0],
                [3.8, 0.0, 0.0],
                [3.8, 1.0, 0.0],
            ]
        ]
    )
    predicted = target.clone()
    predicted[:, 0, :] += 5.0
    predicted[:, 2, :] -= 5.0
    predicted[:, 3, :] += 5.0
    predicted[:, 5, :] -= 5.0

    assert torch.allclose(compute_local_calpha_geometry_loss(predicted, target), torch.tensor(0.0))


def test_extract_calpha_coords_rejects_invalid_backbone_shape() -> None:
    with pytest.raises(ValueError, match="divisible"):
        extract_calpha_coords(torch.zeros(1, 4, 3))


def test_geometry_loss_paths_are_patch_policy_allowed() -> None:
    changed = validate_patch_scope(
        [
            "external/nanofold/nanofold/train/loss.py",
            "external/nanofold/nanofold/train/model/nanofold.py",
            "configs/experiments/local_calpha_geometry_smoke.json",
        ],
        repo_root=REPO_ROOT,
        allow_empty=False,
    )

    assert changed == [
        "external/nanofold/nanofold/train/loss.py",
        "external/nanofold/nanofold/train/model/nanofold.py",
        "configs/experiments/local_calpha_geometry_smoke.json",
    ]


def test_fixture_short_training_records_finite_geometry_loss(tmp_path: Path) -> None:
    materialize_local_nanofold_fixture(
        repo_root=tmp_path,
        output_dir="features",
        approval=APPROVAL_TOKEN,
    )

    manifest = run_short_nanofold_training(
        short_training_payload(
            trial_id="T121",
            candidate_id="T121",
            config_path="configs/experiments/local_calpha_geometry_smoke.json",
            features_path="tiny_features.arrow",
            max_steps=1,
            budget="smoke",
            seed=0,
            local_only=True,
        ),
        features_dir=tmp_path / "features",
        output_dir=tmp_path / "runs/trials/T121",
        repo_root=REPO_ROOT,
        local_only=True,
    )

    assert manifest["official_benchmark_result"] is False
    assert manifest["final_losses"]["local_calpha_geometry_loss"] >= 0.0
    assert "local_calpha_geometry_loss" in json.loads(
        (tmp_path / "runs/trials/T121/loss_history.json").read_text(encoding="utf-8")
    )["losses"][0]["losses"]
