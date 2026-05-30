"""Local scorer dry-run using smoke fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autoalphafold3.schema import PRIMARY_METRIC
from autoalphafold3.scorer.calpha_lddt import SCORER_VERSION, aggregate_calpha_lddt, score_calpha_lddt
from autoalphafold3.scorer.fold_cartographer import summarize_fold_cartographer
from autoalphafold3.scorer.locked_dataset import load_locked_manifest, manifest_hashes


def run_scorer_dry_run(
    *,
    repo_root: str | Path = ".",
    manifest_path: str | Path = "data/manifests/smoke.json",
    trial_id: str = "T000",
    status: str = "DISCARD",
) -> dict[str, object]:
    """Emit canonical metrics JSON from local toy coordinates.

    This intentionally uses a tiny smoke manifest and must not be interpreted as
    an official benchmark result.
    """

    root = Path(repo_root)
    verified = load_locked_manifest(manifest_path, repo_root=root, verify_assets=True)
    results = []

    for entry in verified.manifest.entries:
        label = json.loads((root / entry.label_path).read_text())
        target_ca = np.asarray(label["target_ca"], dtype=np.float64)
        target_mask = np.asarray(label["target_mask"], dtype=bool)
        predicted_ca = target_ca.copy()
        results.append(
            score_calpha_lddt(
                predicted_ca,
                target_ca,
                target_mask,
                target_id=entry.target_id,
            )
        )

    aggregate = aggregate_calpha_lddt(results)
    metrics = dict(aggregate["metrics"])
    return {
        "schema_version": "autoaf3.metrics.v1",
        "scorer_version": SCORER_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "status": status,
        "trial_id": trial_id,
        "candidate_id": "local_dry_run",
        "seed": 0,
        "split": "smoke",
        "manifests": manifest_hashes({"smoke": manifest_path}, repo_root=root),
        "metrics": metrics,
        "quality_gates": {
            "nan_detected": any(result.nan_prediction_residue_count > 0 for result in results),
            "oom_detected": False,
            "timeout_detected": False,
            "max_runtime_s": 0,
            "runtime_s": 0.0,
            "peak_gpu_mem_gb": 0.0,
            "parameter_count": 0,
        },
        "fold_cartographer": summarize_fold_cartographer(results),
        "artifacts": {
            "metrics_json": "local_dry_run_only",
            "manifest": str(manifest_path),
        },
    }
