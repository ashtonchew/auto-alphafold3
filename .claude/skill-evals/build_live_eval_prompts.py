#!/usr/bin/env python3
"""Render self-contained sampled live eval prompts.

The Codex skill attachment path can vary across worker environments. This
script builds prompts that inline SKILL.md plus directly linked references, then
adds one synthetic eval case without leaking expected outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = ROOT / ".claude" / "skills"
DEFAULT_SKILLS = [
    "autoalphafold3-researcher",
    "autoalphafold3-trial-submit",
    "fold-cartographer",
    "autoalphafold3-subagent-worker",
]


def linked_references(skill_md: str) -> list[str]:
    links = re.findall(r"\((references/[^)]+)\)", skill_md)
    inline = re.findall(r"`(references/[^`]+)`", skill_md)
    return sorted(set(links + inline))


def render_prompt(skill_name: str, case_id: str | None = None) -> dict:
    skill_dir = SKILLS_ROOT / skill_name
    skill_md = (skill_dir / "SKILL.md").read_text()
    evals = json.loads((skill_dir / "evals" / "evals.json").read_text())
    cases = evals["cases"]
    if case_id:
        case = next(item for item in cases if item["id"] == case_id)
    else:
        case = cases[0]

    reference_blocks = []
    for ref in linked_references(skill_md):
        ref_path = skill_dir / ref
        reference_blocks.append(f"## {ref}\n\n{ref_path.read_text()}")

    fixture = case.get("fixture")
    fixture_text = ""
    if fixture is not None:
        fixture_text = "\n\nSynthetic fixture:\n" + json.dumps(fixture, indent=2)

    prompt = f"""Sampled live skill eval for `{skill_name}`.

Do not edit files. Do not call Modal, GPUs, hidden validation, or real trial
submission. Use only the inlined skill instructions and references below. Do not
open evals/evals.json.

Return the skill-appropriate output contract, then a short self-check.

# SKILL.md

{skill_md}

# Linked References

{chr(10).join(reference_blocks)}

# Eval Prompt

{case["prompt"]}{fixture_text}
"""
    return {
        "skill": skill_name,
        "case_id": case["id"],
        "model": evals.get("model_policy", {}).get("default_live_model", "gpt-5.3-codex-spark"),
        "parallel": evals.get("model_policy", {}).get("parallel", True),
        "prompt": prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=DEFAULT_SKILLS)
    parser.add_argument("--case-id")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        prompts = [render_prompt(skill) for skill in DEFAULT_SKILLS]
        print(json.dumps(prompts, indent=2))
    else:
        skill = args.skill or DEFAULT_SKILLS[0]
        print(json.dumps(render_prompt(skill, args.case_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
