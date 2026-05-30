# Demo UI mockups

Modal-styled, high-fidelity HTML mockups of the demo surfaces described in
[`docs/spec/demo-ui-plan.html`](../demo-ui-plan.html) §3. These are **illustrative
design references** — every value shown is a placeholder, not a benchmark result.

## Running the frontend

There are two ways to run it: open the pre-rendered mockups, or build the board
from a real run. Both produce **static HTML** — no server, no framework, no
build toolchain. Just Python's standard library and a browser.

### 1. Static mockups (instant)

The HTML files already committed in this folder are pre-rendered. Open any of
them directly — start at [`index.html`](index.html), which links to the combined
board and every panel. Nothing to install.

```bash
open index.html            # macOS  (Linux: xdg-open index.html)
```

These are illustrative; every number is a placeholder.

### 2. Live board (rendered from run artifacts)

`autoalphafold3.ui.build` renders the same surfaces from a real run — or from
the illustrative sample — into an output directory:

```bash
# Illustrative sample (badged "Sample" in the UI)
python -m autoalphafold3.ui.build --sample --out demo/ui

# Real run artifacts
python -m autoalphafold3.ui.build --runs runs --out demo/ui

open demo/ui/index.html
```

It writes `index.html`, `trials.html`, `logs.html`, and `ui_state.json`, and
copies the shared `assets/` (`modal.css` + `board.js`) next to them. The default
`--out` is `demo/ui`, which is gitignored — it is build output, not source.

**What `--runs` reads** (each input is optional; missing ones degrade honestly):

| Path under `--runs` | Provides |
| --- | --- |
| `ledger.jsonl` | per-trial scores → trajectory, trials table, metric band |
| `baseline/metrics.json` | baseline Cα lDDT for the delta |
| `discovery_ledger.jsonl` | confirmed mechanisms → Discovery Ledger |

**Honest fallback.** If `runs/` is missing or partial, the build does not crash
or invent numbers: it badges the board **Sample** and uses clearly-flagged
`geometry_sample` placeholders for any panel that has no real data yet. A live
build only ever shows real metrics or labelled samples — never a fabricated
benchmark.

### Regenerating the committed mockups

The pre-rendered files in this folder are generated, not hand-edited. After
changing the renderer (`autoalphafold3/ui/`) or `assets/`, regenerate them so
the committed HTML matches the code:

```bash
python -c "from autoalphafold3.ui.build import build; \
  build('docs/spec/ui', sample=True, board_name='evidence-board.html', write_state=False)"
```

## Layout

```
ui/
  index.html              gallery linking the stage views + all panels
  evidence-board.html     the combined stage dashboard (§2)
  trials.html             all-trials table, filterable by status
  logs.html               timestamped orchestrator/trial event feed, searchable
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
| `panels/06-discovery-ledger.html` | 06 Discovery | Confirmed mechanisms + provisional KEEPs |
| `panels/07-modal-execution.html` | 07 Modal | Execution mode, burst, mount policy, cost |
| `panels/08-structure-overlay.html` | 08 Overlay | Backbone overlay + per-residue error |
| `panels/09-sampler-burst.html` | 09 Sampler | Frozen-checkpoint fan-out + label-free selection |
