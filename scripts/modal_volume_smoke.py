#!/usr/bin/env python3
"""Smoke-test the uploaded auto-AlphaFold3 Modal Volume artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("autoalphafold3-data-smoke")
data_volume = modal.Volume.from_name("autoalphafold3-data")
locked_volume = modal.Volume.from_name("autoalphafold3-locked")
image = modal.Image.debian_slim().pip_install("pyarrow")


@app.function(
    image=image,
    volumes={
        "/mnt/autoalphafold3": data_volume.with_mount_options(read_only=True),
        "/mnt/autoalphafold3-locked": locked_volume.with_mount_options(read_only=True),
    },
)
def smoke() -> dict:
    import pyarrow as pa

    root = Path("/mnt/autoalphafold3")
    locked_root = Path("/mnt/autoalphafold3-locked")
    event_features = root / "features/nanofold_event_small_no_templates.arrow"
    train_features = root / "features/train_tiny.arrow"
    public_val_features = root / "features/public_val_small.arrow"
    fingerprint = root / "features/feature_fingerprints.json"
    train_manifest = locked_root / "manifests/train_tiny.json"
    public_val_manifest = locked_root / "manifests/public_val_small.json"
    public_val_labels = locked_root / "labels/public_val_labels.arrow"
    scorer_version = locked_root / "scorer_version.txt"

    required = [
        event_features,
        train_features,
        public_val_features,
        fingerprint,
        train_manifest,
        public_val_manifest,
        public_val_labels,
        scorer_version,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing uploaded artifacts: {missing}")

    with pa.memory_map(str(event_features)) as source:
        event_table = pa.ipc.open_file(source).read_all()
    with pa.memory_map(str(train_features)) as source:
        train_table = pa.ipc.open_file(source).read_all()
    with pa.memory_map(str(public_val_features)) as source:
        val_table = pa.ipc.open_file(source).read_all()

    if event_table.num_rows != 48:
        raise RuntimeError(f"Expected 48 event records, found {event_table.num_rows}")
    if train_table.num_rows != 32:
        raise RuntimeError(f"Expected 32 train records, found {train_table.num_rows}")
    if val_table.num_rows != 16:
        raise RuntimeError(f"Expected 16 public val records, found {val_table.num_rows}")

    for column in [
        "template_mask",
        "template_sequence",
        "template_translations",
        "template_rotations",
    ]:
        lengths = [len(value) for value in event_table[column].to_pylist()]
        if any(lengths):
            raise RuntimeError(f"{column} has non-empty template records")

    raw_count = len(list((root / "raw/mmcif").glob("*.cif")))
    if raw_count != 48:
        raise RuntimeError(f"Expected 48 raw mmCIF files, found {raw_count}")
    if (root / "locked").exists():
        raise RuntimeError("Public data Volume must not contain /locked")

    fingerprint_payload = json.loads(fingerprint.read_text())
    if fingerprint_payload.get("locked_volume") != "autoalphafold3-locked":
        raise RuntimeError("feature_fingerprints.json does not reference locked Volume")

    return {
        "event_records": event_table.num_rows,
        "train_records": train_table.num_rows,
        "public_val_records": val_table.num_rows,
        "raw_mmcif": raw_count,
        "data_fingerprint_files": len(fingerprint_payload["data_files"]),
        "locked_fingerprint_files": len(fingerprint_payload["locked_files"]),
        "templates_empty": True,
        "locked_volume": "autoalphafold3-locked",
    }


@app.local_entrypoint()
def main():
    print(smoke.remote())
