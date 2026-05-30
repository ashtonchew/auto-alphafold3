"""Approved local NanoFold fixture materialization for preflight gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

APPROVAL_TOKEN = "I_APPROVE_LOCAL_NANOFOLD_FIXTURE"
DEFAULT_FIXTURE_DIR = Path("data/toy/nanofold_fixture")
FIXTURE_ARROW_NAME = "tiny_features.arrow"
FIXTURE_PROVENANCE_NAME = "fixture_provenance.json"
FIXTURE_SCHEMA_VERSION = "autoaf3.local_nanofold_fixture.v1"


class LocalFixtureError(ValueError):
    """Raised when local fixture materialization or validation fails."""


@dataclass(frozen=True)
class LocalFixtureReport:
    """Local-only cached fixture report."""

    status: str
    fixture_path: str
    provenance_path: str
    sha256: str | None
    rows: int
    local_only: bool
    max_templates: int
    problems: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "fixture_path": self.fixture_path,
            "provenance_path": self.provenance_path,
            "sha256": self.sha256,
            "rows": self.rows,
            "local_only": self.local_only,
            "max_templates": self.max_templates,
            "problems": self.problems,
        }


def default_fixture_path(repo_root: str | Path = ".") -> Path:
    return Path(repo_root) / DEFAULT_FIXTURE_DIR / FIXTURE_ARROW_NAME


def materialize_local_nanofold_fixture(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_FIXTURE_DIR,
    approval: str,
    overwrite: bool = False,
) -> LocalFixtureReport:
    """Write a deterministic local-only Arrow fixture after explicit approval."""

    if approval != APPROVAL_TOKEN:
        raise LocalFixtureError(f"approval must be exactly {APPROVAL_TOKEN}")

    root = Path(repo_root)
    fixture_dir = root / output_dir
    fixture_path = fixture_dir / FIXTURE_ARROW_NAME
    provenance_path = fixture_dir / FIXTURE_PROVENANCE_NAME
    if (fixture_path.exists() or provenance_path.exists()) and not overwrite:
        raise LocalFixtureError("local fixture already exists; pass overwrite=True to replace it")

    pa, ipc = _require_pyarrow()
    fixture_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(_fixture_columns())
    with pa.OSFile(str(fixture_path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    digest = _sha256(fixture_path)
    provenance = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "local_only": True,
        "official_benchmark_result": False,
        "source": "deterministic synthetic NanoFold parser fixture for local preflight only",
        "fixture_path": fixture_path.relative_to(root).as_posix(),
        "sha256": digest,
        "rows": table.num_rows,
        "max_templates": 0,
        "approval": APPROVAL_TOKEN,
        "writes_baseline": False,
        "writes_ledger": False,
        "starts_search": False,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validate_local_nanofold_fixture(fixture_path=fixture_path, repo_root=root)


def validate_local_nanofold_fixture(
    *,
    fixture_path: str | Path,
    repo_root: str | Path = ".",
) -> LocalFixtureReport:
    """Validate the local-only Arrow fixture and provenance."""

    root = Path(repo_root)
    path = Path(fixture_path)
    if not path.is_absolute():
        path = root / path
    provenance_path = path.with_name(FIXTURE_PROVENANCE_NAME)
    problems: list[str] = []
    if not path.exists():
        problems.append("fixture Arrow file is missing")
    if not provenance_path.exists():
        problems.append("fixture provenance is missing")
    payload: dict[str, object] = {}
    if provenance_path.exists():
        try:
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"fixture provenance is invalid JSON: {exc}")
    rows = 0
    digest = _sha256(path) if path.exists() else None
    if payload:
        if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
            problems.append("fixture provenance schema_version is invalid")
        if payload.get("local_only") is not True:
            problems.append("fixture provenance must be local_only=true")
        if payload.get("official_benchmark_result") is not False:
            problems.append("fixture must not claim official_benchmark_result")
        if payload.get("max_templates") != 0:
            problems.append("fixture max_templates must be 0")
        if payload.get("sha256") != digest:
            problems.append("fixture sha256 does not match provenance")
    if path.exists():
        try:
            pa, ipc = _require_pyarrow()
            with pa.memory_map(str(path)) as source:
                with ipc.open_file(source) as reader:
                    table = reader.read_all()
            rows = table.num_rows
            missing_columns = sorted(set(_fixture_columns()) - set(table.column_names))
            if missing_columns:
                problems.append(f"fixture missing columns: {missing_columns}")
            if table.num_rows < 2:
                problems.append("fixture must contain at least two rows for train/held-out split")
        except Exception as exc:  # noqa: BLE001 - validation reports actionable fixture problems.
            problems.append(f"fixture Arrow file is unreadable: {type(exc).__name__}: {exc}")
    return LocalFixtureReport(
        status="PASS" if not problems else "FAIL",
        fixture_path=str(path),
        provenance_path=str(provenance_path),
        sha256=digest,
        rows=rows,
        local_only=payload.get("local_only") is True,
        max_templates=int(payload.get("max_templates", -1)) if payload else -1,
        problems=problems,
    )


def _fixture_columns() -> dict[str, object]:
    length = 40
    sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    positions = list(range(length))
    translations = [[float(i), 0.1 * float(i % 3), 0.0] for i in range(length)]
    rotations = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for _ in range(length)
    ]
    profile = [[1.0 / 32.0 for _ in range(32)] for _ in range(length)]
    deletion_mean = [0.0 for _ in range(length)]
    rows = []
    for offset in (0, 1):
        rows.append(
            {
                "positions": [item + offset for item in positions],
                "translations": translations,
                "rotations": rotations,
                "sequence": sequence,
                "template_mask": [],
                "template_sequence": [],
                "template_translations": [],
                "template_rotations": [],
                "profile": profile,
                "deletion_mean": deletion_mean,
                "msa_coords": [[i, 0, 0] for i in range(length)],
                "msa_data": [True for _ in range(length)],
                "msa_shape": [length, 1, 32],
                "has_deletion_coords": [[i, 0, 0] for i in range(length)],
                "has_deletion_data": [False for _ in range(length)],
                "has_deletion_shape": [length, 1, 1],
                "deletion_value_coords": [[i, 0, 0] for i in range(length)],
                "deletion_value_data": [0.0 for _ in range(length)],
                "deletion_value_shape": [length, 1, 1],
            }
        )
    return {key: [row[key] for row in rows] for key in rows[0]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - depends on optional local env.
        raise LocalFixtureError("pyarrow is required for local NanoFold fixture materialization") from exc
    return pa, ipc
