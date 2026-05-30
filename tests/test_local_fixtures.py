from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoalphafold3.local_fixtures import (
    APPROVAL_TOKEN,
    FIXTURE_ARROW_NAME,
    FIXTURE_PROVENANCE_NAME,
    LocalFixtureError,
    materialize_local_nanofold_fixture,
    validate_local_nanofold_fixture,
)
from autoalphafold3.nanofold_checks import finite_loss_gate, tiny_forward_gate

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_local_fixture_requires_exact_approval(tmp_path: Path) -> None:
    with pytest.raises(LocalFixtureError, match="approval"):
        materialize_local_nanofold_fixture(repo_root=tmp_path, approval="yes")

    assert not list(tmp_path.rglob("*"))


def test_materialize_local_fixture_writes_local_only_arrow_and_provenance(tmp_path: Path) -> None:
    report = materialize_local_nanofold_fixture(repo_root=tmp_path, approval=APPROVAL_TOKEN)

    fixture_path = tmp_path / "data/toy/nanofold_fixture" / FIXTURE_ARROW_NAME
    provenance_path = tmp_path / "data/toy/nanofold_fixture" / FIXTURE_PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert report.status == "PASS"
    assert fixture_path.exists()
    assert provenance["local_only"] is True
    assert provenance["official_benchmark_result"] is False
    assert provenance["max_templates"] == 0
    assert provenance["writes_baseline"] is False
    assert provenance["writes_ledger"] is False
    assert provenance["starts_search"] is False
    assert validate_local_nanofold_fixture(fixture_path=fixture_path, repo_root=tmp_path).status == "PASS"


def test_materialized_fixture_parses_through_nanofold_chain_dataset(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("torch")
    materialize_local_nanofold_fixture(repo_root=tmp_path, approval=APPROVAL_TOKEN)

    import sys

    nanofold_root = str(REPO_ROOT / "external/nanofold")
    if nanofold_root not in sys.path:
        sys.path.insert(0, nanofold_root)
    from nanofold.train.chain_dataset import ChainDataset

    train, _held_out = ChainDataset.construct_datasets(
        tmp_path / "data/toy/nanofold_fixture" / FIXTURE_ARROW_NAME,
        0.5,
        32,
        4,
    )
    features = next(iter(train))

    assert features["msa"].shape[-1] == 22
    assert features["has_deletion"].shape[-1] == 1
    assert features["deletion_value"].shape[-1] == 1
    assert features["template_restype"].shape[0] == 0


def test_materialize_local_fixture_cli_has_no_baseline_or_ledger_side_effects(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoalphafold3.agent",
            "materialize-local-fixture",
            "--repo-root",
            str(tmp_path),
            "--approve",
            APPROVAL_TOKEN,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert not (tmp_path / "runs").exists()
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == [
        "data",
        "data/toy",
        "data/toy/nanofold_fixture",
        "data/toy/nanofold_fixture/fixture_provenance.json",
        "data/toy/nanofold_fixture/tiny_features.arrow",
    ]


def test_local_gates_pass_with_approved_fixture_and_finite_loss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    materialize_local_nanofold_fixture(repo_root=tmp_path, approval=APPROVAL_TOKEN)

    import autoalphafold3.nanofold_checks as checks

    monkeypatch.setattr(
        checks,
        "_run_tiny_nanofold_loss",
        lambda *, config_path, repo_root: {
            "mse_loss": 1.0,
            "lddt_loss": 0.5,
            "diffusion_loss": 1.5,
            "dist_loss": 2.0,
            "total_loss": 6.06,
        },
    )
    import_summary = {
        "imports": [
            {"module": "nanofold.train.model.nanofold", "ok": True},
            {"module": "nanofold.train.trainer", "ok": True},
        ]
    }

    tiny = tiny_forward_gate(repo_root=tmp_path, import_summary=import_summary)
    finite = finite_loss_gate(repo_root=tmp_path, import_summary=import_summary)

    assert tiny.status == "passed"
    assert tiny.reason == "forward_loss_finite"
    assert finite.status == "passed"
    assert finite.reason == "total_loss_finite"
    assert tiny.details["fixture"]["local_only"] is True
