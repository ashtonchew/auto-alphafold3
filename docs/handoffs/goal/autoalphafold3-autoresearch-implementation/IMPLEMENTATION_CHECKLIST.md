# Implementation Checklist

- [x] PR #50 base state verified.
- [x] Required docs read in order.
- [x] `modal-docs` skill invoked before Modal work.
- [x] Goal progress files created.
- [x] Autoresearch runbook created.
- [x] Agent program prompt created.
- [ ] Short-training manifest schema implemented.
- [ ] Short-training runner writes trial-scoped artifacts only.
- [ ] Short-training runner rejects fake training claims.
- [ ] Short-training runner rejects `max_templates != 0`.
- [ ] Short-training runner rejects unsafe feature paths.
- [ ] Short-training runner rejects non-empty output directories.
- [ ] Fixture-backed 2-3 step short-training test passes.
- [ ] Local scaffold mode cannot stamp official benchmark evidence.
- [ ] Config-driven NanoFold loss weights implemented.
- [ ] Defaults preserve current loss behavior when new weights are zero.
- [ ] Differentiable local C-alpha geometry loss implemented.
- [ ] Local C-alpha geometry loss has finite-value and masking tests.
- [ ] Candidate artifact envelope implemented.
- [ ] Candidate patch snapshots implemented.
- [ ] Safe git keep/revert wrapper implemented.
- [ ] Safe git wrapper preserves unrelated user changes.
- [ ] Safe git wrapper refuses locked/generated artifacts.
- [ ] Manual planner mode implemented.
- [ ] Deterministic ladder planning implemented.
- [ ] Deterministic ladder dry-run/planning mode passes.
- [ ] Matched-budget baseline comparison implemented.
- [ ] Global-baseline provisional KEEP comparison preserved.
- [ ] Provisional KEEP does not write Discovery Ledger.
- [ ] LLM planner mode implemented only after deterministic mode passes.
- [ ] LLM planner emits exactly one hypothesis, move family, diagnostic target,
  and candidate patch.
- [ ] LLM planner cannot bypass patch policy.
- [ ] Web search is allowed only for hypothesis generation, not patch planning.
- [ ] Execution workers receive no OpenAI/GitHub/Modal/dashboard/judge secrets.
- [ ] `TrialRunner.run(...)` remains the official training entrypoint.
- [ ] No dynamic Modal `.with_options(...)` resource escalation appears.
- [ ] Training artifacts cannot write baseline, canonical ledger, or Discovery
  Ledger records.
- [ ] Live Modal commands refuse absent or wrong approval tokens without
  calling Modal, writing ledgers, or promoting run artifacts.
- [ ] Worker handoffs commit and reload Modal Volume state before
  cross-container reads.
- [ ] Canonical ledger, results TSV, and run summary writes are serialized.
- [ ] Repeated failure/OOM/NaN/gate-kill patterns hit explicit stop rules.
- [ ] UI reads autoresearch summary artifacts.
- [ ] UI labels sample fallback data honestly.
- [ ] Full tests pass or failures are recorded with exact blockers.
- [ ] Any live Modal/search step is marked `PENDING_HUMAN_LIVE_ACTION` with
  exact command and approval token.
