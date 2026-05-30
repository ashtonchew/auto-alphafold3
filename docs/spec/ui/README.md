# Demo UI mockups

Modal-styled, high-fidelity HTML mockups of the demo surfaces described in
[`docs/spec/demo-ui-plan.html`](../demo-ui-plan.html) §3. These are **illustrative
design references** — every value shown is a placeholder, not a benchmark result.

## How to view

Open any file in a browser (no build step, no server needed). Start at
[`index.html`](index.html), which links to the combined board and every panel.

## Layout

```
ui/
  index.html              gallery linking the board + all panels
  evidence-board.html     the combined stage dashboard (§2)
  assets/
    modal.css             shared Modal design system (tokens + all components)
    board.js              trajectory hover/click interactivity
  panels/                 one focused mockup per §3 surface
    01-overview.html
    02-trial-trajectory.html
    03-hypothesis-card.html
    04-fold-cartographer.html
    05-falsification-gate.html
    06-discovery-ledger.html
    07-modal-execution.html
    08-structure-overlay.html
    09-sampler-burst.html
```

## Design system

Every page links `assets/modal.css` — one source of truth for the Modal look
(dark `#181818` base, `#7FEE64` green accent, Inter + Fira Mono, status pills,
data tables, stat grids, chips). Tokens were taken from `modal.com`. To restyle
all mockups, edit that one file.

## Panels ↔ plan surfaces

| File | §3 surface | Purpose |
| --- | --- | --- |
| `evidence-board.html` | §2 | Combined stage dashboard |
| `panels/01-overview.html` | 01 Overview | Run state, provenance, Modal readiness |
| `panels/02-trial-trajectory.html` | 02 Trajectory | Interactive score-over-trials + selected trial |
| `panels/03-hypothesis-card.html` | 03 Hypothesis | Pre-registered falsifiable claim |
| `panels/04-fold-cartographer.html` | 04 Cartographer | Four geometry axes + context buckets |
| `panels/05-falsification-gate.html` | 05 Gate | Knock-out / placebo / seed controls |
| `panels/06-discovery-ledger.html` | 06 Discovery | Confirmed mechanisms + instructive kills |
| `panels/07-modal-execution.html` | 07 Modal | Execution mode, burst, mount policy, cost |
| `panels/08-structure-overlay.html` | 08 Overlay | Backbone overlay + per-residue error |
| `panels/09-sampler-burst.html` | 09 Sampler | Frozen-checkpoint fan-out + label-free selection |
