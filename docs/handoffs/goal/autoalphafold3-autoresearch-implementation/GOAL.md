# Autoresearch Implementation Goal

Status: active
Base verified: PR #50 merged into `origin/main` at `cf5844e43fb80562eebf31c741b2bd0f119b90f0` on 2026-06-01.
Implementation target: NanoFold-style AlphaFold3-lite.

## Objective

Implement the SimplexFold/Karpathy-style autoresearch loop described in
`docs/spec/autoalphafold3-autoresearch-spec.md` without weakening the locked
benchmark, scorer, data, Modal, baseline, ledger, or Discovery Ledger
boundaries.

The loop must support bounded short training, honest candidate artifacts,
safe keep/revert behavior, deterministic planning before LLM planning, and UI
evidence rendering. Live Modal execution and open-ended search remain
`PENDING_HUMAN_LIVE_ACTION` until a human supplies the exact approval token.

## Stack

1. `feat/autoresearch-contract-docs`
2. `feat/short-training-runner`
3. `feat/nanofold-geometry-loss`
4. `feat/autoresearch-candidates`
5. `feat/deterministic-autoresearch-ladder`
6. `feat/autoresearch-llm-planner`
7. `feat/autoresearch-ui-evidence`

## Locked Boundary

Do not edit scorer math, public validation membership, validation labels,
manifest fingerprints, cached features, baseline artifacts, Modal GPU/resource
policy, canonical ledger authority, or Discovery Ledger rules. Official runs
must keep `max_templates=0`.

## Completion Evidence

Done requires all implementation PRs open or merged, checklist completion
except exact `PENDING_HUMAN_LIVE_ACTION` items, passing final tests, fixture
short-training artifacts with honest stamps, deterministic ladder dry-run
evidence, UI rendering for real or labelled sample autoresearch artifacts, and
no unapproved live search or fabricated evidence.
