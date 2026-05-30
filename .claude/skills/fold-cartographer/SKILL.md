---
name: fold-cartographer
description: Use when interpreting auto-AlphaFold3 scorer diagnostics into exactly one Fold Cartographer failure target, preserving best_val_calpha_lddt as the primary objective and mapping local geometry, long-range topology, distogram-vs-3D, or stability/compute evidence to safe move-family suggestions.
---

# Fold Cartographer

Use this skill to convert scorer diagnostics into one routing target for the
next AlphaFold3-lite hypothesis. The output guides research; it does not change
the objective.

## Workflow

1. Read `references/routing.md`.
2. Inspect primary score and diagnostics.
3. Choose exactly one diagnostic target.
4. Name two or three compatible move families.
5. Explain the evidence briefly.

## Hard Rules

- The only primary objective is `best_val_calpha_lddt`.
- Do not optimize diagnostics directly.
- Do not use hidden validation for search routing.
- Do not ask to inspect labels, scorer code, validation splits, or locked data.
- Do not call `modal run`, Modal APIs, or submit trials.

## Output Contract

```text
TARGET: <one target>
EVIDENCE: <metrics that justify target>
MOVE_FAMILY_CANDIDATES: <comma-separated families>
AVOID: <moves that would violate the contract or chase diagnostics directly>
```

## References

- Read `references/routing.md` for target definitions and tie-break behavior.
