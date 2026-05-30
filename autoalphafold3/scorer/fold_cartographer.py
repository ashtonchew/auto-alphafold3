"""Small Fold Cartographer summary helpers for local dry-runs."""

from __future__ import annotations

from collections.abc import Iterable

from autoalphafold3.scorer.calpha_lddt import CalphaLddtResult


def summarize_fold_cartographer(results: Iterable[CalphaLddtResult]) -> dict[str, object]:
    """Summarize toy C-alpha lDDT results into diagnostic buckets."""

    result_list = list(results)
    scored = [result for result in result_list if result.eligible_pair_count > 0]
    nan_residues = sum(result.nan_prediction_residue_count for result in result_list)
    mean_score = sum(result.score for result in scored) / len(scored) if scored else 0.0

    if nan_residues:
        signature = "nan_prediction_instability"
        canonical_target = "stability_compute"
    elif mean_score >= 0.95:
        signature = "toy_geometry_preserved"
        canonical_target = "local_geometry_weak"
    elif mean_score >= 0.5:
        signature = "toy_geometry_degraded"
        canonical_target = "local_geometry_weak"
    else:
        signature = "toy_geometry_failed"
        canonical_target = "local_geometry_weak"

    return {
        "signature": signature,
        "summary": {
            "canonical_target": canonical_target,
            "mean_target_calpha_lddt": mean_score,
            "nan_prediction_residue_count": nan_residues,
            "num_targets": len(result_list),
            "num_scored_targets": len(scored),
        },
        "buckets": {
            "toy_all": {
                "eligible_pair_count": sum(result.eligible_pair_count for result in result_list),
                "target_ids": [result.target_id for result in result_list],
            }
        },
    }
