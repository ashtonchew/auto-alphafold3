"""Build the evidence board from run artifacts (or the illustrative sample).

    python -m autoalphafold3.ui.build --sample            # design mockup
    python -m autoalphafold3.ui.build --runs runs --out demo/ui   # live

Writes ``index.html`` + ``ui_state.json`` and copies the shared design system
(``modal.css`` + ``board.js``) next to it. Never writes benchmark data; sample
builds are badged and real builds carry their source/provenance.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from autoalphafold3.ui.page import render_board
from autoalphafold3.ui.state import load_state, sample_state

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_SRC = REPO_ROOT / "docs" / "spec" / "ui" / "assets"
ASSET_FILES = ("modal.css", "board.js")


def build(out_dir: str | Path, runs_dir: str | Path = "runs", *, sample: bool = False) -> Path:
    state = sample_state() if sample else load_state(runs_dir)
    out = Path(out_dir)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render_board(state), encoding="utf-8")
    (out / "ui_state.json").write_text(json.dumps(state.to_json(), indent=2), encoding="utf-8")
    for name in ASSET_FILES:
        src = ASSETS_SRC / name
        if src.exists():
            shutil.copyfile(src, out / "assets" / name)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autoalphafold3.ui.build",
        description="Render the Modal-styled evidence board from run artifacts.",
    )
    parser.add_argument("--runs", default="runs", help="run artifacts directory (default: runs)")
    parser.add_argument("--out", default="demo/ui", help="output directory (default: demo/ui)")
    parser.add_argument("--sample", action="store_true", help="use illustrative sample data instead of real artifacts")
    args = parser.parse_args(argv)
    out = build(args.out, args.runs, sample=args.sample)
    source = "sample" if args.sample else f"runs:{args.runs}"
    print(f"wrote {out / 'index.html'} and {out / 'ui_state.json'} ({source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
