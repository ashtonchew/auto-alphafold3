#!/usr/bin/env python3
"""Verify no-template NanoFold Arrow IPC feature files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for candidate in (Path.cwd(), Path("/app"), Path("/Users/naveenramasamy/src/nanofold")):
    if (candidate / "nanofold").exists():
        sys.path.insert(0, str(candidate))

import pyarrow as pa

from nanofold.common.msa_metadata import COMPRESSED_MSA_FIELDS
from nanofold.preprocess.ipc import SCHEMA

try:
    from nanofold.train.chain_dataset import ChainDataset
except ModuleNotFoundError:
    ChainDataset = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--expected-records", required=True, type=int)
    parser.add_argument("--crop-size", default=64, type=int)
    parser.add_argument("--num-msa", default=16, type=int)
    return parser.parse_args()


def load_manifest_records(paths: list[Path]) -> set[str]:
    records: set[str] = set()
    for path in paths:
        with path.open() as f:
            for entry in json.load(f):
                records.add(f"{entry['pdb_id'].upper()}_{entry['chain_id']}")
    return records


def main() -> None:
    args = parse_args()
    if not args.features.exists():
        raise SystemExit(f"missing features file: {args.features}")

    with pa.memory_map(str(args.features)) as source:
        table = pa.ipc.open_file(source).read_all()

    if table.schema != SCHEMA:
        raise SystemExit("Arrow schema does not match nanofold.preprocess.ipc.SCHEMA")
    if table.num_rows != args.expected_records:
        raise SystemExit(f"expected {args.expected_records} records, found {table.num_rows}")

    records = {
        f"{structure_id.upper()}_{chain_id}"
        for structure_id, chain_id in zip(
            table["structure_id"].to_pylist(), table["chain_id"].to_pylist()
        )
    }
    expected_records = load_manifest_records(args.manifest)
    if records != expected_records:
        raise SystemExit(
            f"feature records do not match manifests; missing={sorted(expected_records - records)} "
            f"extra={sorted(records - expected_records)}"
        )

    for column in [
        "template_mask",
        "template_sequence",
        "template_translations",
        "template_rotations",
    ]:
        lengths = [len(value) for value in table[column].to_pylist()]
        if any(length != 0 for length in lengths):
            raise SystemExit(f"{column} is not empty for every record: {lengths}")

    for field in COMPRESSED_MSA_FIELDS:
        for suffix in ["shape", "data", "coords"]:
            column = f"{field}_{suffix}"
            if column not in table.column_names:
                raise SystemExit(f"missing MSA column: {column}")
        if any(len(value) == 0 for value in table[f"{field}_data"].to_pylist()):
            raise SystemExit(f"MSA column {field}_data has an empty record")

    if ChainDataset is None:
        chain_dataset_path = next(
            (
                candidate / "nanofold/train/chain_dataset.py"
                for candidate in (Path.cwd(), Path("/app"), Path("/Users/naveenramasamy/src/nanofold"))
                if (candidate / "nanofold/train/chain_dataset.py").exists()
            ),
            None,
        )
        if chain_dataset_path is None:
            raise SystemExit("ChainDataset unavailable and source file not found for inspection")
        chain_dataset_source = chain_dataset_path.read_text()
        required_snippets = [
            'if len(row["template_sequence"]) == 0:',
            "torch.empty(0, length",
            '"template_restype": template_restype',
        ]
        missing_snippets = [
            snippet for snippet in required_snippets if snippet not in chain_dataset_source
        ]
        if missing_snippets:
            raise SystemExit(
                f"ChainDataset empty-template source inspection failed: {missing_snippets}"
            )
        chain_dataset_status = "source-empty-template-branch-ok"
    else:
        train, _ = ChainDataset.construct_datasets(
            args.features,
            0.8,
            residue_crop_size=args.crop_size,
            num_msa=args.num_msa,
        )
        batch = next(iter(train))
        required_loaded_keys = {
            "msa",
            "has_deletion",
            "deletion_value",
            "template_restype",
            "template_backbone_frame_mask",
            "template_distogram",
            "template_unit_vector",
        }
        missing_keys = sorted(required_loaded_keys - set(batch))
        if missing_keys:
            raise SystemExit(f"ChainDataset output missing keys: {missing_keys}")
        if batch["template_restype"].shape[0] != 0:
            raise SystemExit("ChainDataset produced non-empty template_restype")
        chain_dataset_status = "load-ok"

    print(
        json.dumps(
            {
                "features": str(args.features),
                "records": table.num_rows,
                "msa_fields": sorted(COMPRESSED_MSA_FIELDS.keys()),
                "template_records_all_empty": True,
                "chain_dataset": chain_dataset_status,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
