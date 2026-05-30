#!/usr/bin/env python3
"""Prepare a small locked public-data NanoFold-style dataset.

The script deliberately keeps the event contract simple:
- raw public mmCIF files come from RCSB
- derived features are Arrow IPC files
- validation C-alpha labels live under locked/labels while staged locally
- split manifests and fingerprints are stable JSON files

This is a lightweight local feature packer. It does not download the full
AlphaFold database bundle or run proprietary AlphaFold3 assets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}

ACTIVE_SPLITS = {"train_tiny", "public_val_small"}

# Curated public monomer-oriented candidates. The selector still validates
# lengths from downloaded mmCIF during feature packing; these IDs are intentionally
# ordinary public RCSB structures so the first run does not depend on a complex
# RCSB query API.
DEFAULT_CANDIDATES = [
    "1UBQ",
    "1ENH",
    "1BDD",
    "1PGB",
    "2CI2",
    "1SHF",
    "1TEN",
    "1CSP",
    "2LZM",
    "1AKI",
    "1HEL",
    "3CHY",
    "1CRN",
    "1VII",
    "1L2Y",
    "1HZ6",
    "1FME",
    "1SRL",
    "1E0L",
    "1BTA",
    "1I6C",
    "1A3A",
    "1MJC",
    "2PTL",
    "1RIS",
    "1BRS",
    "1MJC",
    "1NUN",
    "1KSR",
    "1MUG",
    "1PLC",
    "1POH",
    "1A2P",
    "1FNA",
    "1HRC",
    "1YCC",
    "2LZM",
    "1AKI",
    "1HEL",
    "3CHY",
    "1R69",
    "1IFC",
    "1ROP",
    "1G6P",
    "1TIT",
    "1KDX",
    "1CSE",
    "1POA",
    "1NKR",
    "1PGA",
    "1A6M",
    "1BPI",
    "1C9O",
    "1EJG",
    "1HYP",
    "1JOO",
    "1KTE",
    "1LMB",
    "1MJC",
    "1NOA",
    "1PHT",
    "1QAU",
    "1RNB",
    "1SNC",
    "1TUP",
    "1VIK",
    "1WIT",
    "1XNB",
    "1YPI",
    "2ACY",
    "2BPA",
    "2CRO",
    "2ERL",
    "2GB1",
    "2HBA",
    "2IHB",
    "2KNT",
    "2MHR",
    "2OVO",
    "2PDD",
    "2RN2",
    "2SNI",
    "2TRX",
    "2VIK",
    "2WRP",
    "3B5C",
    "3CI2",
    "3EBX",
    "3F3Z",
    "3GB1",
    "3LZM",
    "3MBP",
    "3PGK",
    "3SSI",
    "4AKE",
    "4HHB",
    "4ICB",
    "4LYZ",
    "4PTI",
    "5PTI",
    "6LYZ",
]


def fetch_rcsb_candidates(limit: int) -> list[str]:
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                "operator": "exact_match",
                "value": "Protein (only)",
            },
        },
        "request_options": {
            "paginate": {"start": 0, "rows": max(limit, 256)},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}],
            "results_content_type": ["experimental"],
        },
        "return_type": "entry",
    }
    body = json.dumps(query).encode()
    request = urllib.request.Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "autoalphafold3-prep/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode())
    return [row["identifier"] for row in payload.get("result_set", [])]


def fetch_rcsb_entity_metadata(pdb_id: str, entity_id: str = "1") -> dict[str, Any] | None:
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "autoalphafold3-prep/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    entity_poly = payload.get("entity_poly", {})
    identifiers = payload.get("rcsb_polymer_entity_container_identifiers", {})
    auth_asym_ids = identifiers.get("auth_asym_ids") or identifiers.get("asym_ids") or []
    return {
        "pdb_id": pdb_id,
        "chain_id": auth_asym_ids[0] if auth_asym_ids else "A",
        "sequence_length": entity_poly.get("rcsb_sample_sequence_length"),
        "polymer_type": entity_poly.get("rcsb_entity_polymer_type") or entity_poly.get("type"),
        "nonstandard_monomer_count": entity_poly.get("rcsb_non_std_monomer_count", 0),
        "auth_asym_ids": auth_asym_ids,
    }


def filter_candidate_metadata(
    candidates: list[str], config: dict[str, Any], total_needed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for pdb_id in candidates:
        try:
            meta = fetch_rcsb_entity_metadata(pdb_id)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue
        length = meta.get("sequence_length")
        if not isinstance(length, int):
            continue
        if not (config["length_min"] <= length <= config["length_max"]):
            continue
        if len(meta.get("auth_asym_ids", [])) != 1:
            continue
        if int(meta.get("nonstandard_monomer_count") or 0) != 0:
            continue
        selected.append(meta)
        if len(selected) >= total_needed:
            break
    return selected


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for Arrow IPC output. Install with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return pa, ipc


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_dirs(out: Path) -> dict[str, Path]:
    return {
        "raw": out / "raw" / "mmcif",
        "features": out / "features",
        "manifests": out / "locked" / "manifests",
        "labels": out / "locked" / "labels",
        "locked": out / "locked",
    }


def normalize_pdb_id(value: str) -> str:
    return value.strip().upper()


def unique_candidates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        pdb_id = normalize_pdb_id(value)
        if pdb_id and pdb_id not in seen:
            seen.add(pdb_id)
            result.append(pdb_id)
    return result


def selected_splits(config: dict[str, Any], smoke: bool) -> dict[str, int]:
    if smoke:
        counts = dict(config["smoke_counts"])
    else:
        counts = dict(config["counts"])
    unsupported = sorted(set(counts) - ACTIVE_SPLITS)
    if unsupported:
        raise SystemExit(f"Unsupported active prereq splits: {unsupported}")
    return counts


def require_no_template_policy(config: dict[str, Any]) -> None:
    """Require dataset configs to preserve the official no-template contract."""

    template_search = config.get("template_search")
    if not isinstance(template_search, dict):
        raise SystemExit("Config missing template_search policy")
    if template_search.get("max_templates") != 0:
        raise SystemExit("Official NanoFold-style AlphaFold3-lite runs require max_templates=0")
    if template_search.get("enabled_for_full") is not False:
        raise SystemExit("Template search must be disabled for full official runs")
    if template_search.get("database") is not None:
        raise SystemExit("Template database must be null for the no-template benchmark")


def select(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    require_no_template_policy(config)
    dirs = dataset_dirs(args.out)
    counts = selected_splits(config, args.smoke)
    total_needed = sum(counts.values())
    selected_metadata: list[dict[str, Any]] = []
    if args.candidate_file:
        candidates = unique_candidates(args.candidate_file.read_text().splitlines())
        selected_metadata = [{"pdb_id": pdb_id, "chain_id": "A"} for pdb_id in candidates]
    else:
        try:
            candidates = unique_candidates(fetch_rcsb_candidates(total_needed * 3))
            print(f"fetched {len(candidates)} RCSB candidate entries")
            selected_metadata = filter_candidate_metadata(candidates, config, total_needed)
            print(f"selected {len(selected_metadata)} length-filtered monomer entities")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"RCSB search failed, using curated fallback candidates: {exc}", file=sys.stderr)
            candidates = unique_candidates(DEFAULT_CANDIDATES)
            selected_metadata = [{"pdb_id": pdb_id, "chain_id": "A"} for pdb_id in candidates]
    if len(selected_metadata) < total_needed:
        raise SystemExit(
            f"Need {total_needed} selected candidate chains, only have {len(selected_metadata)}. "
            "Provide --candidate-file with one PDB ID per line, or run with network access."
        )

    cursor = 0
    all_entries: dict[str, list[dict[str, Any]]] = {}
    for split, count in counts.items():
        entries = []
        for meta in selected_metadata[cursor : cursor + count]:
            pdb_id = meta["pdb_id"]
            entries.append(
                {
                    "pdb_id": pdb_id,
                    "chain_id": meta["chain_id"],
                    "split": split,
                    "source_url": config["rcsb_mmcif_url_template"].format(pdb_id=pdb_id),
                    "selection_note": "curated public RCSB candidate; validated during feature packing",
                    "sequence_length": meta.get("sequence_length"),
                    "polymer_type": meta.get("polymer_type"),
                }
            )
        cursor += count
        all_entries[split] = entries
        write_json(dirs["manifests"] / f"{split}.json", entries)

    provenance = {
        "dataset_name": config["dataset_name"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "event-small",
        "nanofold_repo": config["nanofold_repo"],
        "nanofold_commit": config["nanofold_commit"],
        "rcsb_mmcif_url_template": config["rcsb_mmcif_url_template"],
        "msa_search": config["msa_search"],
        "template_search": config["template_search"],
        "splits": {split: len(entries) for split, entries in all_entries.items()},
    }
    write_json(args.out / "provenance.json", provenance)
    print(f"wrote manifests under {dirs['manifests']}")


def download_url(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "autoalphafold3-prep/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        tmp.write_bytes(response.read())
    tmp.replace(path)


def manifest_entries(manifests_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(manifests_dir.glob("*.json")):
        entries.extend(read_json(path))
    return entries


def download(args: argparse.Namespace) -> None:
    dirs = dataset_dirs(args.out)
    entries = manifest_entries(dirs["manifests"])
    if not entries:
        raise SystemExit("No manifests found. Run select first.")

    for entry in entries:
        pdb_id = entry["pdb_id"]
        target = dirs["raw"] / f"{pdb_id}.cif"
        if target.exists() and not args.force:
            print(f"exists {target}")
            continue
        try:
            download_url(entry["source_url"], target)
            print(f"downloaded {pdb_id} -> {target}")
        except urllib.error.URLError as exc:
            raise SystemExit(f"Failed to download {entry['source_url']}: {exc}") from exc


def parse_loop(lines: list[str], start_index: int) -> tuple[list[str], list[list[str]], int]:
    tags: list[str] = []
    rows: list[list[str]] = []
    i = start_index + 1
    while i < len(lines) and lines[i].strip().startswith("_"):
        tags.append(lines[i].strip())
        i += 1
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "#" or line.startswith("loop_") or line.startswith("_"):
            break
        rows.append(line.split())
        i += 1
    return tags, rows, i


def parse_atom_site_ca(mmcif: str, chain_id: str) -> tuple[str, list[list[float]]]:
    lines = mmcif.splitlines()
    residues: list[str] = []
    coords: list[list[float]] = []

    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        tags, rows, next_i = parse_loop(lines, i)
        if "_atom_site.group_PDB" not in tags:
            i = next_i
            continue
        tag_index = {tag: idx for idx, tag in enumerate(tags)}
        required = [
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        ]
        if not all(tag in tag_index for tag in required):
            i = next_i
            continue

        chain_tags = [
            tag
            for tag in ("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
            if tag in tag_index
        ]
        seq_tag = "_atom_site.label_seq_id" if "_atom_site.label_seq_id" in tag_index else None
        seen_residues: set[str] = set()
        for row in rows:
            if len(row) < len(tags):
                continue
            if row[tag_index["_atom_site.label_atom_id"]].strip("\"'") != "CA":
                continue
            if chain_tags and all(row[tag_index[tag]].strip("\"'") != chain_id for tag in chain_tags):
                continue
            residue_key = row[tag_index[seq_tag]] if seq_tag else str(len(coords))
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)
            comp = row[tag_index["_atom_site.label_comp_id"]].strip("\"'").upper()
            aa = AA3_TO_1.get(comp, "X")
            try:
                xyz = [
                    float(row[tag_index["_atom_site.Cartn_x"]]),
                    float(row[tag_index["_atom_site.Cartn_y"]]),
                    float(row[tag_index["_atom_site.Cartn_z"]]),
                ]
            except ValueError:
                continue
            residues.append(aa)
            coords.append(xyz)
        break
    return "".join(residues), coords


def feature_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("record_id", pa.string()),
            ("pdb_id", pa.string()),
            ("chain_id", pa.string()),
            ("split", pa.string()),
            ("sequence", pa.string()),
            ("sequence_length", pa.int32()),
            ("ca_positions", pa.list_(pa.list_(pa.float32()))),
            ("ca_mask", pa.list_(pa.bool_())),
            ("source_mmcif_sha256", pa.string()),
            ("msa_database", pa.string()),
            ("template_database", pa.string()),
            ("template_search_enabled", pa.bool_()),
        ]
    )


def labels_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("record_id", pa.string()),
            ("pdb_id", pa.string()),
            ("chain_id", pa.string()),
            ("sequence_length", pa.int32()),
            ("ca_positions", pa.list_(pa.list_(pa.float32()))),
            ("ca_mask", pa.list_(pa.bool_())),
        ]
    )


def write_arrow(path: Path, rows: list[dict[str, Any]], schema: Any, pa: Any, ipc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    with path.open("wb") as f:
        with ipc.new_file(f, schema) as writer:
            writer.write_table(table)


def read_arrow(path: Path, ipc: Any) -> Any:
    with path.open("rb") as f:
        return ipc.open_file(f).read_all()


def split_feature_rows(
    split: str,
    entries: list[dict[str, Any]],
    raw_dir: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    features: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    updated_manifest: list[dict[str, Any]] = []
    for entry in entries:
        pdb_id = entry["pdb_id"]
        chain_id = entry["chain_id"]
        raw_path = raw_dir / f"{pdb_id}.cif"
        if not raw_path.exists():
            raise SystemExit(f"Missing raw mmCIF for {pdb_id}: {raw_path}")
        mmcif = raw_path.read_text(errors="replace")
        sequence, coords = parse_atom_site_ca(mmcif, chain_id)
        if not sequence or len(sequence) != len(coords):
            raise SystemExit(f"Could not extract C-alpha sequence/coords for {pdb_id}_{chain_id}")
        if not (config["length_min"] <= len(sequence) <= config["length_max"]):
            raise SystemExit(
                f"{pdb_id}_{chain_id} length {len(sequence)} outside "
                f"{config['length_min']}..{config['length_max']}"
            )

        raw_hash = sha256_file(raw_path)
        record_id = f"{pdb_id}_{chain_id}"
        feature = {
            "record_id": record_id,
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "split": split,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "ca_positions": coords,
            "ca_mask": [True] * len(coords),
            "source_mmcif_sha256": raw_hash,
            "msa_database": config["msa_search"]["database"],
            "template_database": config["template_search"]["database"],
            "template_search_enabled": bool(config["template_search"]["enabled_for_full"]),
        }
        label = {
            "record_id": record_id,
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "sequence_length": len(sequence),
            "ca_positions": coords,
            "ca_mask": [True] * len(coords),
        }
        manifest_entry = dict(entry)
        manifest_entry.update(
            {
                "record_id": record_id,
                "sequence_length": len(sequence),
                "source_mmcif_sha256": raw_hash,
            }
        )
        features.append(feature)
        labels.append(label)
        updated_manifest.append(manifest_entry)
    return features, labels, updated_manifest


def preprocess(args: argparse.Namespace) -> None:
    pa, ipc = require_pyarrow()
    config = read_json(args.config)
    require_no_template_policy(config)
    dirs = dataset_dirs(args.out)
    manifests = sorted(dirs["manifests"].glob("*.json"))
    if not manifests:
        raise SystemExit("No manifests found. Run select first.")

    fingerprints: dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_name": config["dataset_name"],
        "nanofold_commit": config["nanofold_commit"],
        "files": {},
    }

    for manifest_path in manifests:
        split = manifest_path.stem
        entries = read_json(manifest_path)
        features, labels, updated_manifest = split_feature_rows(split, entries, dirs["raw"], config)

        if split.startswith("public_val"):
            label_path = dirs["labels"] / f"{split.replace('_small', '')}_labels.arrow"
            write_arrow(label_path, labels, labels_schema(pa), pa, ipc)
            for row in features:
                row["ca_positions"] = []
                row["ca_mask"] = []

        feature_path = dirs["features"] / f"{split}.arrow"
        write_arrow(feature_path, features, feature_schema(pa), pa, ipc)
        write_json(manifest_path, updated_manifest)
        fingerprints["files"][str(feature_path.relative_to(args.out))] = sha256_file(feature_path)
        fingerprints["files"][str(manifest_path.relative_to(args.out))] = sha256_file(manifest_path)
        if split.startswith("public_val"):
            fingerprints["files"][str(label_path.relative_to(args.out))] = sha256_file(label_path)
        print(f"wrote {feature_path}")

    for raw_path in sorted(dirs["raw"].glob("*.cif")):
        fingerprints["files"][str(raw_path.relative_to(args.out))] = sha256_file(raw_path)

    write_json(dirs["features"] / "feature_fingerprints.json", fingerprints)
    print(f"wrote {dirs['features'] / 'feature_fingerprints.json'}")


def validate(args: argparse.Namespace) -> None:
    pa, ipc = require_pyarrow()
    del pa
    dirs = dataset_dirs(args.out)
    manifests = {path.stem: read_json(path) for path in sorted(dirs["manifests"].glob("*.json"))}
    if not manifests:
        raise SystemExit("No manifests found.")

    record_to_split: dict[str, str] = {}
    for split, entries in manifests.items():
        for entry in entries:
            record_id = entry.get("record_id", f"{entry['pdb_id']}_{entry['chain_id']}")
            if record_id in record_to_split:
                raise SystemExit(f"Record {record_id} appears in both {record_to_split[record_id]} and {split}")
            record_to_split[record_id] = split

    for split, entries in manifests.items():
        feature_path = dirs["features"] / f"{split}.arrow"
        if not feature_path.exists():
            raise SystemExit(f"Missing feature file {feature_path}")
        table = read_arrow(feature_path, ipc)
        if table.num_rows != len(entries):
            raise SystemExit(f"{feature_path} has {table.num_rows} rows, expected {len(entries)}")
        names = set(table.schema.names)
        required = {
            "record_id",
            "pdb_id",
            "chain_id",
            "split",
            "sequence",
            "sequence_length",
            "ca_positions",
            "ca_mask",
            "source_mmcif_sha256",
        }
        missing = required - names
        if missing:
            raise SystemExit(f"{feature_path} missing required columns: {sorted(missing)}")
        rows = table.to_pylist()
        for row in rows:
            if row["sequence_length"] != len(row["sequence"]):
                raise SystemExit(f"{row['record_id']} sequence_length mismatch")
            if row["split"].startswith("train") and len(row["ca_positions"]) != row["sequence_length"]:
                raise SystemExit(f"{row['record_id']} missing train labels")
            if row["split"].startswith("public_val") and row["ca_positions"]:
                raise SystemExit(f"{row['record_id']} validation labels leaked into features")

    for label_path in sorted(dirs["labels"].glob("*_labels.arrow")):
        table = read_arrow(label_path, ipc)
        for row in table.to_pylist():
            if len(row["ca_positions"]) != row["sequence_length"]:
                raise SystemExit(f"{row['record_id']} label length mismatch in {label_path}")

    fingerprint_path = dirs["features"] / "feature_fingerprints.json"
    fingerprints = read_json(fingerprint_path)
    for rel_path, expected_hash in fingerprints["files"].items():
        actual = sha256_file(args.out / rel_path)
        if actual != expected_hash:
            raise SystemExit(f"Hash mismatch for {rel_path}: {actual} != {expected_hash}")
    print("validation passed")


def modal_upload(args: argparse.Namespace) -> None:
    try:
        import modal
    except ImportError as exc:
        raise SystemExit("modal is required for upload. Install and authenticate Modal first.") from exc

    if not args.out.exists():
        raise SystemExit(f"Missing dataset directory: {args.out}")
    data_volume = modal.Volume.from_name(args.data_volume_name, create_if_missing=True)
    locked_volume = modal.Volume.from_name(args.locked_volume_name, create_if_missing=True)
    data_remote_root = args.data_remote_path.rstrip("/") or "/"
    locked_remote_root = args.locked_remote_path.rstrip("/") or "/"

    data_files: list[Path] = []
    locked_files: list[Path] = []
    for local_path in sorted(args.out.rglob("*")):
        if not local_path.is_file():
            continue
        rel = local_path.relative_to(args.out)
        if rel.parts and rel.parts[0] == "locked":
            locked_files.append(local_path)
        else:
            data_files.append(local_path)

    with data_volume.batch_upload(force=True) as batch:
        for local_path in data_files:
            rel = local_path.relative_to(args.out).as_posix()
            remote = f"{data_remote_root}/{rel}".replace("//", "/")
            batch.put_file(str(local_path), remote)
            print(f"uploaded {local_path} -> {remote}")

    with locked_volume.batch_upload(force=True) as batch:
        for local_path in locked_files:
            rel = local_path.relative_to(args.out / "locked").as_posix()
            remote = f"{locked_remote_root}/{rel}".replace("//", "/")
            batch.put_file(str(local_path), remote)
            print(f"uploaded {local_path} -> {remote}")
    print(f"committed Modal volumes {args.data_volume_name} and {args.locked_volume_name}")


def nanofold_command(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    require_no_template_policy(config)
    db_paths = config["database_paths"]
    modal_volume = config.get("modal_volume", {})
    locked_volume = config.get("locked_volume", {})
    if modal_volume.get("canonical"):
        root = modal_volume.get("mount_path", "/mnt/autoalphafold3")
        locked_root = locked_volume.get("mount_path", "/mnt/autoalphafold3-locked")
        cache = f"{root}/cache"
        output = f"{root}/features/nanofold_event_small_no_templates.arrow"
        mmcif = f"{root}/raw/mmcif"
        manifests = [
            f"{locked_root}/manifests/train_tiny.json",
            f"{locked_root}/manifests/public_val_small.json",
        ]
    else:
        cache = str(args.out / "cache")
        output = str(args.out / "features" / "nanofold_event_small_no_templates.arrow")
        mmcif = str(args.out / "raw" / "mmcif")
        manifests = [
            str(args.out / "locked" / "manifests" / "train_tiny.json"),
            str(args.out / "locked" / "manifests" / "public_val_small.json"),
        ]
    command = [
        "python",
        "scripts/nanofold_preprocess_no_templates.py",
        "-m",
        mmcif,
        "-c",
        cache,
        "-o",
        output,
        "--small_bfd",
        db_paths["small_bfd"],
        "--uniclust30",
        db_paths["uniclust30"],
        "--manifest",
        manifests[0],
        "--manifest",
        manifests[1],
        "--reset-db",
    ]
    print(" ".join(command))


def clean(args: argparse.Namespace) -> None:
    if not args.out.exists():
        return
    if args.yes != "DELETE":
        raise SystemExit("Refusing to delete without --yes DELETE")
    shutil.rmtree(args.out)
    print(f"deleted {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser, config: bool = True) -> None:
        if config:
            subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--out", type=Path, required=True)

    select_parser = subparsers.add_parser("select", help="write locked split manifests")
    add_common(select_parser)
    select_parser.add_argument("--smoke", action="store_true")
    select_parser.add_argument("--candidate-file", type=Path)
    select_parser.set_defaults(func=select)

    download_parser = subparsers.add_parser("download", help="download selected RCSB mmCIF files")
    add_common(download_parser)
    download_parser.add_argument("--force", action="store_true")
    download_parser.set_defaults(func=download)

    preprocess_parser = subparsers.add_parser("preprocess", help="write Arrow features and labels")
    add_common(preprocess_parser)
    preprocess_parser.set_defaults(func=preprocess)

    validate_parser = subparsers.add_parser("validate", help="validate manifests, Arrow files, and hashes")
    add_common(validate_parser)
    validate_parser.set_defaults(func=validate)

    upload_parser = subparsers.add_parser("modal-upload", help="upload validated artifacts to Modal Volumes")
    add_common(upload_parser, config=False)
    upload_parser.add_argument("--data-volume-name", default="autoalphafold3-data")
    upload_parser.add_argument("--locked-volume-name", default="autoalphafold3-locked")
    upload_parser.add_argument("--data-remote-path", default="/")
    upload_parser.add_argument("--locked-remote-path", default="/")
    upload_parser.set_defaults(func=modal_upload)

    nanofold_parser = subparsers.add_parser(
        "nanofold-command", help="print the upstream NanoFold feature-generation command"
    )
    add_common(nanofold_parser)
    nanofold_parser.set_defaults(func=nanofold_command)

    clean_parser = subparsers.add_parser("clean", help="delete generated dataset artifacts")
    add_common(clean_parser, config=False)
    clean_parser.add_argument("--yes", required=True)
    clean_parser.set_defaults(func=clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    try:
        args.func(args)
    except BrokenPipeError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
