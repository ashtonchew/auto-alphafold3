# AutoAlphaFold3 Modal Pre-Run Readiness Goal

Implement the post-PR-21/PR-22 Modal pre-run readiness layer for the
NanoFold-style AlphaFold3-lite system without starting autonomous search,
creating fake benchmark artifacts, or mutating locked data.

Accepted base:

- `main` fast-forwarded to `origin/main` at `f057aab`.
- PR #21 merged at `fab93c8383e893bc6488236f7a17b82eac873206`.
- PR #22 merged at `f057aab5589d03d2432dc5541d7c795779c296f6`.

Branch plan:

1. `feat/modal-prerun-contracts`
2. `feat/falsification-and-ledgers`
3. `feat/baseline-scorer-readiness`
4. `feat/prerun-readiness-report`

Live Modal, baseline locking, gate calibration, and autonomous search remain
separate human-approved actions.
