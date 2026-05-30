# NanoFold Pin and Implementation Map

NanoFold is pinned as a vendored working-tree copy at `external/nanofold`.

- Upstream: `https://github.com/ogchen/nanofold.git`
- Pinned commit: `f670440e4fa533592e7b411408dfbffc5e1bf113`
- Local record: `NANOFOLD_COMMIT`

This map is for implementation planning only. It does not download MSA databases, provision template databases, rebuild feature artifacts, generate Arrow files, train a baseline, or create benchmark metrics.

## Upstream Runtime Shape

NanoFold has two major runtime paths:

- Feature generation: `external/nanofold/nanofold/preprocess/__main__.py`
- Training: `external/nanofold/nanofold/train/__main__.py`

The upstream feature-generation path can parse mmCIF files, search small BFD and Uniclust30 for MSA construction, search PDB70 for templates, and write Arrow IPC features. The hackathon contract disables templates with `max_templates=0`, so any approved data-owner rebuild must produce empty template placeholders rather than provision or search a template database. That path is intentionally not run in this slice.

The training path loads Arrow IPC features through `ChainDataset`, constructs train/test splits from `train_split`, creates `Trainer`, instantiates `Nanofold`, and logs metrics/checkpoints through stdout and upstream optional logging hooks. This project treats the local JSONL ledger as canonical.

## Key Files Read

- `external/nanofold/README.md`
- `external/nanofold/config/config.dev.json`
- `external/nanofold/nanofold/train/__main__.py`
- `external/nanofold/nanofold/train/trainer.py`
- `external/nanofold/nanofold/train/model/nanofold.py`
- `external/nanofold/docker/Dockerfile.preprocess`
- `external/nanofold/docker/Dockerfile.train`

## Editable Research Surface

Future agent research patches may target these pinned files after preflight and patch-policy approval:

- model wrapper: `external/nanofold/nanofold/train/model/nanofold.py`
- trunk: `external/nanofold/nanofold/train/model/nanofold_trunk.py`
- Pairformer blocks: `external/nanofold/nanofold/train/model/pairformer.py`
- diffusion model: `external/nanofold/nanofold/train/model/diffusion_model.py`
- diffusion transformer: `external/nanofold/nanofold/train/model/diffusion_transformer.py`
- MSA module: `external/nanofold/nanofold/train/model/msa_module.py`
- template embedder: `external/nanofold/nanofold/train/model/template_embedder.py`
- losses: `external/nanofold/nanofold/train/loss.py`
- trainer loop: `external/nanofold/nanofold/train/trainer.py`
- dataset wrapper: `external/nanofold/nanofold/train/chain_dataset.py`
- experiment configs copied under `configs/experiments/**`

## Locked Or Infrastructure Surface

Do not edit these during search:

- feature-generation code: `external/nanofold/nanofold/preprocess/**`
- upstream download script: `external/nanofold/scripts/download_pdb.sh`
- Docker and requirement files unless infrastructure mode is explicitly unlocked
- `NANOFOLD_COMMIT` and nested git metadata
- local scorer, manifests, baseline ledger, Modal caps, and orchestrator guardrails

## Import Smoke Findings

Local import smoke is implemented in `autoalphafold3/nanofold_adapter.py`.

Expected behavior after creating a repo-local venv and installing
`requirements.txt`:

- `nanofold` base package imports.
- training modules import with PyTorch available.
- feature-generation entrypoint imports with PyMongo available.

If `torch`, `pymongo`, or `modal` are missing from the active interpreter, create
or refresh `.venv` from the root `requirements.txt`. The cached Arrow features
and Modal Volumes remove the need for event-time feature rebuilding; they do not
install local Python packages into the active environment.
