#!/usr/bin/env python3
"""Run NanoFold feature generation with MSA features and empty template placeholders.

This is a local wrapper around upstream NanoFold's feature-generation modules for the
hackathon no-template contract. It intentionally never constructs or searches a
template database. Chains become IPC-ready by writing the same `templates` field
that NanoFold's IPC dumper expects, but with empty arrays.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from autoalphafold3.nanofold_adapter import nanofold_root

REPO_ROOT = Path(__file__).resolve().parents[1]


def nanofold_source_candidates() -> list[Path]:
    """Return possible NanoFold source roots without personal absolute paths."""

    candidates: list[Path] = []
    for repo_root in (Path.cwd(), REPO_ROOT, Path("/app")):
        try:
            candidates.append(nanofold_root(repo_root=repo_root))
        except ValueError:
            continue
    candidates.extend([Path.cwd(), Path("/app")])
    return candidates


def add_nanofold_to_path() -> None:
    """Add the pinned NanoFold checkout to sys.path for upstream imports."""

    for candidate in nanofold_source_candidates():
        if (candidate / "nanofold").exists():
            sys.path.insert(0, str(candidate))
            return


add_nanofold_to_path()

from nanofold.preprocess.db import DBManager
from nanofold.preprocess.hhblits import HHblitsRunner
from nanofold.preprocess.ipc import dump_to_ipc
from nanofold.preprocess.mmcif_processor import process_mmcif_files
from nanofold.preprocess.msa_builder import build_msa, prefetch_msa
from nanofold.preprocess.msa_runner import MSARunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-b", "--batch", default=1000, type=int)
    parser.add_argument("-m", "--mmcif", required=True, type=Path)
    parser.add_argument("-c", "--cache", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-s", "--small_bfd", required=True, type=Path)
    parser.add_argument("-u", "--uniclust30", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help="Locked manifest JSON. Repeat for train and public-validation splits.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Drop NanoFold's Mongo feature database before running.",
    )
    parser.add_argument(
        "--hhblits-cpu",
        default=1,
        type=int,
        help="CPU threads per HHblits process. Keep low for local Docker Desktop memory.",
    )
    parser.add_argument(
        "--dump-only",
        action="store_true",
        help="Skip mmCIF/MSA work and only stamp empty templates then dump IPC.",
    )
    parser.add_argument("-l", "--logging", default="INFO")
    return parser.parse_args()


def load_allowed_ids(manifest_paths: list[Path]) -> set[tuple[str, str]]:
    if not manifest_paths:
        raise SystemExit("explicit --manifest is required for no-template NanoFold preprocessing")
    allowed: set[tuple[str, str]] = set()
    for path in manifest_paths:
        with path.open() as f:
            payload = json.load(f)
            entries = payload.get("entries", []) if isinstance(payload, dict) else payload
        for entry in entries:
            allowed.add((entry["pdb_id"].lower(), entry["chain_id"]))
    return allowed


def restrict_to_manifest_chains(db_manager: DBManager, allowed_ids: set[tuple[str, str]] | None) -> None:
    if allowed_ids is None:
        return
    all_ids = {
        (doc["_id"]["structure_id"], doc["_id"]["chain_id"])
        for doc in db_manager.chains().find({}, {"_id": 1})
    }
    remove_ids = [
        {"structure_id": structure_id, "chain_id": chain_id}
        for structure_id, chain_id in sorted(all_ids - allowed_ids)
    ]
    if remove_ids:
        result = db_manager.chains().delete_many({"_id": {"$in": remove_ids}})
        logging.info("Removed %d chains outside locked manifests", result.deleted_count)


def manifest_mmcif_dir(mmcif_dir: Path, cache_dir: Path, allowed_ids: set[tuple[str, str]] | None) -> Path:
    if allowed_ids is None:
        return mmcif_dir
    selected_dir = cache_dir / "manifest_mmcif"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True)
    for structure_id in sorted({structure_id for structure_id, _ in allowed_ids}):
        source = mmcif_dir / f"{structure_id.upper()}.cif"
        if not source.exists():
            source = mmcif_dir / f"{structure_id.lower()}.cif"
        if not source.exists():
            raise FileNotFoundError(f"Missing mmCIF for manifest structure {structure_id}")
        os.symlink(source, selected_dir / source.name)
    return selected_dir


def stamp_empty_template_features(db_manager: DBManager, msa_output_dir: Path) -> int:
    updated = 0
    for msa_file in sorted(msa_output_dir.glob("*.pkl.gz")):
        stem = msa_file.name.removesuffix(".pkl.gz")
        structure_id, chain_id = stem.split("_", 1)
        result = db_manager.chains().update_one(
            {"_id": {"structure_id": structure_id, "chain_id": chain_id}},
            {
                "$set": {
                    "templates": {
                        "mask": [],
                        "sequence": [],
                        "translations": [],
                        "rotations": [],
                    }
                }
            },
        )
        updated += result.modified_count
    logging.info("Stamped empty template placeholders on %d MSA-ready chains", updated)
    return updated


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=getattr(logging, args.logging.upper()),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    msa_output_dir = args.cache / "msa"
    msa_output_dir.mkdir(exist_ok=True)
    jackhmmer_results_path = args.cache / "small_bfd_cache"
    jackhmmer_results_path.mkdir(exist_ok=True)
    uniclust30_cache_dir = args.cache / "uniclust30_cache"
    uniclust30_cache_dir.mkdir(exist_ok=True)

    db_manager = DBManager(uri=os.getenv("MONGODB_URI"))
    if args.reset_db:
        db_manager.client.drop_database(db_manager.db.name)
        logging.info("Dropped Mongo database %s", db_manager.db.name)

    allowed_ids = load_allowed_ids(args.manifest)
    mmcif_dir = manifest_mmcif_dir(args.mmcif, args.cache, allowed_ids)
    small_bfd_msa_search = MSARunner(
        shutil.which("jackhmmer"),
        args.small_bfd,
        jackhmmer_results_path,
        num_cpus=1,
        max_sequences=5000,
    )
    uniclust30_msa_search = HHblitsRunner(
        shutil.which("hhblits"),
        args.uniclust30,
        uniclust30_cache_dir,
        num_iterations=3,
        num_cpu=args.hhblits_cpu,
        output_format="a3m",
    )

    if not args.dump_only:
        with ProcessPoolExecutor() as executor:
            process_mmcif_files(db_manager, executor, mmcif_dir, args.batch)
        restrict_to_manifest_chains(db_manager, allowed_ids)

        with ThreadPoolExecutor() as executor:
            logging.info("Prefetching MSA from small BFD")
            prefetch_msa(small_bfd_msa_search, db_manager, executor, jackhmmer_results_path)

        with ThreadPoolExecutor(max_workers=1) as executor:
            logging.info("Prefetching MSA from Uniclust30")
            prefetch_msa(uniclust30_msa_search, db_manager, executor, uniclust30_cache_dir)

        with ProcessPoolExecutor(max_workers=1) as executor:
            build_msa(
                small_bfd_msa_search,
                uniclust30_msa_search,
                db_manager,
                executor,
                msa_output_dir,
                include_dirs=[jackhmmer_results_path, uniclust30_cache_dir],
            )

    with ThreadPoolExecutor() as executor:
        stamp_empty_template_features(db_manager, msa_output_dir)
        dump_to_ipc(db_manager, msa_output_dir, args.output, executor)


if __name__ == "__main__":
    main()
