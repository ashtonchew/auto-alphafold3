from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_nanofold_data_test", REPO_ROOT / "scripts" / "prepare_nanofold_data.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBatch:
    def __init__(self, volume: "FakeVolume") -> None:
        self.volume = volume

    def __enter__(self) -> "FakeBatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def put_file(self, local_path: str, remote_path: str) -> None:
        self.volume.uploads.append((Path(local_path).name, remote_path))


class FakeVolume:
    volumes: dict[str, "FakeVolume"] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self.uploads: list[tuple[str, str]] = []

    @classmethod
    def from_name(cls, name: str, create_if_missing: bool = False) -> "FakeVolume":
        del create_if_missing
        volume = cls(name)
        cls.volumes[name] = volume
        return volume

    def batch_upload(self, force: bool = False) -> FakeBatch:
        assert force is True
        return FakeBatch(self)


def test_selected_splits_rejects_inactive_prereq_splits() -> None:
    module = load_prepare_module()

    with pytest.raises(SystemExit, match="Unsupported active prereq splits"):
        module.selected_splits({"counts": {"train_small": 1}, "smoke_counts": {}}, smoke=False)


def test_modal_upload_splits_public_and_locked_volumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_prepare_module()
    FakeVolume.volumes = {}
    fake_modal = ModuleType("modal")
    fake_modal.Volume = FakeVolume
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "train_tiny.arrow").write_text("public")
    (tmp_path / "locked" / "manifests").mkdir(parents=True)
    (tmp_path / "locked" / "manifests" / "train_tiny.json").write_text("locked")
    (tmp_path / "locked" / "labels").mkdir()
    (tmp_path / "locked" / "labels" / "public_val_labels.arrow").write_text("labels")

    module.modal_upload(
        SimpleNamespace(
            out=tmp_path,
            data_volume_name="autoalphafold3-data",
            locked_volume_name="autoalphafold3-locked",
            data_remote_path="/",
            locked_remote_path="/",
        )
    )

    data_uploads = FakeVolume.volumes["autoalphafold3-data"].uploads
    locked_uploads = FakeVolume.volumes["autoalphafold3-locked"].uploads

    assert data_uploads == [("train_tiny.arrow", "/features/train_tiny.arrow")]
    assert locked_uploads == [
        ("public_val_labels.arrow", "/labels/public_val_labels.arrow"),
        ("train_tiny.json", "/manifests/train_tiny.json"),
    ]
