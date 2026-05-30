# Manifest Locking Runbook

The official benchmark manifests are explicit JSON files. They must not be created from random split behavior during official trials.

These are split-manifest templates only. They are unrelated to NanoFold protein templates, which are disabled for the official benchmark with `max_templates=0`.

## Current Files

- `data/manifests/smoke.json`: local toy manifest for contract tests only.
- `data/manifests/train_tiny.template.json`: empty future official split-manifest template.
- `data/manifests/public_val_small.template.json`: empty future official split-manifest template.

The template files contain no official targets and must not be treated as benchmark data.

## Validate Current Smoke Manifest

```bash
python -m autoalphafold3.agent validate-manifest data/manifests/smoke.json
```

This verifies referenced local toy feature and label hashes.

## Validate Empty Manifest Templates

```bash
python -m autoalphafold3.agent validate-manifest \
  data/manifests/train_tiny.template.json \
  data/manifests/public_val_small.template.json \
  --allow-empty --no-verify-assets
```

`--allow-empty` is only for empty split-manifest templates. Real official manifests must contain entries and use asset verification.

## Real Manifest Fill-In Rules

Every real entry needs:

- `target_id`
- `pdb_id`
- `chain_id`
- `sequence_sha256`
- `feature_sha256`
- `label_sha256`
- `length`
- `msa_depth_bucket`
- `length_bucket`
- `split`
- `feature_path`
- `label_path`

Training code may access train labels only. Public validation labels are scorer-stage only.
