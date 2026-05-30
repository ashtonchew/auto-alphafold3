# Editable Surface

This file defines the intended search surface for future autoresearch trials. NanoFold is pinned as a vendored working-tree copy at `external/nanofold` with commit recorded in `NANOFOLD_COMMIT`.

## Agent-Editable During Search

- `autoalphafold3/patches/**` when added
- `configs/experiments/**`
- `external/nanofold/nanofold/train/model/nanofold.py`
- `external/nanofold/nanofold/train/model/nanofold_trunk.py`
- `external/nanofold/nanofold/train/model/pairformer.py`
- `external/nanofold/nanofold/train/model/diffusion_model.py`
- `external/nanofold/nanofold/train/model/diffusion_transformer.py`
- `external/nanofold/nanofold/train/model/msa_module.py`
- `external/nanofold/nanofold/train/model/template_embedder.py`
- `external/nanofold/nanofold/train/loss.py`
- `external/nanofold/nanofold/train/trainer.py`
- `external/nanofold/nanofold/train/chain_dataset.py`
- `external/nanofold/config/config.dev.json` for reference only; experiment changes should be copied into `configs/experiments/**`
- sampler wrappers that do not read validation labels
- loss-weight schedules
- optimizer and scheduler configs
- curriculum settings
- diffusion/noising schedules in the active path
- recycling settings
- Pairformer or attention block variants
- auxiliary heads
- geometry-loss experiments
- memory and runtime optimizations inside the approved model/training surface

## Locked During Search

- `NANOFOLD_COMMIT` and nested git metadata
- `external/nanofold/nanofold/preprocess/**`
- `external/nanofold/scripts/download_pdb.sh`
- `external/nanofold/docker/**` unless a human explicitly unlocks infrastructure mode
- `external/nanofold/requirements/**` unless a human explicitly unlocks dependency mode
- `autoalphafold3/_tracing.py`
- `autoalphafold3/scorer/**`
- `autoalphafold3/benchmark_contract.md`
- `autoalphafold3/modal_app.py`
- `autoalphafold3/orchestrator.py`
- `autoalphafold3/preflight.py`
- public validation manifests
- validation labels
- data fingerprints
- cached feature outputs
- result parser
- leaderboard renderer
- `runs/baseline/**`
- Modal GPU type, timeout, Volume, retry, and `max_containers` settings

## Required Before First Real Search Trial

1. Clone or vendor NanoFold in a clearly pinned location. Done: `external/nanofold`.
2. Record the exact NanoFold commit. Done: `NANOFOLD_COMMIT`.
3. Map model, trainer, loss, sampler, dataset, and config files. Done in this file and `docs/nanofold_map.md`.
4. Add patch-policy tests that reject edits outside this surface.

## Rule

The agent may edit model/training/sampler hypotheses. It may not edit the benchmark contract or Modal control plane.
