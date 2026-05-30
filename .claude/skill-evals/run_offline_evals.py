#!/usr/bin/env python3
"""Fast offline checks for auto-AlphaFold3 skills.

This runner uses only local files and synthetic expected outputs. It performs
cheap checks before any optional live model/subagent evals:

- skill frontmatter and required resources
- linked reference existence
- basic guardrail scans
- fixture assertion grading from per-skill evals/evals.json

It never calls Modal, launches GPUs, submits trials, or runs model inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = ROOT / ".claude" / "skills"
WORKSPACE_ROOT = ROOT / ".claude" / "skill-eval-workspace"

EXPECTED_SKILLS = [
    "autoalphafold3-researcher",
    "autoalphafold3-trial-submit",
    "fold-cartographer",
    "autoalphafold3-subagent-worker",
]

DIAGNOSTIC_TARGETS = [
    "local_geometry_weak",
    "long_range_topology_weak",
    "distogram_good_lddt_flat",
    "stability_compute",
]

REQUIRED_GUARDRAILS = [
    "Do not call `modal run`",
    "Do not",
    "hidden validation",
]

DANGEROUS_POSITIVE_PATTERNS = [
    re.compile(r"(?im)^\s*(run|call|use|invoke|spawn|create)\s+`?modal run"),
    re.compile(r"(?im)^\s*(run|call|use|invoke|spawn|create)\s+`?modal\.Sandbox\.create"),
    re.compile(r"(?im)^\s*(run|call|use|invoke|spawn|create)\s+`?modal\.Function\.from_name"),
    re.compile(r"(?im)^\s*(edit|change|modify)\s+`?autoalphafold3/modal_app\.py"),
    re.compile(r"(?im)^\s*(edit|change|modify)\s+.*hidden validation"),
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def add(checks: list[Check], name: str, ok: bool, detail: str) -> None:
    checks.append(Check(name=name, ok=ok, detail=detail))


def parse_frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise ValueError("missing YAML frontmatter")
    data: dict[str, str] = {}
    current_key: str | None = None
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            data[current_key] = value.strip()
        elif current_key:
            data[current_key] = (data[current_key] + " " + line.strip()).strip()
    return data


def linked_references(skill_md: str) -> list[str]:
    links = re.findall(r"\((references/[^)]+)\)", skill_md)
    inline = re.findall(r"`(references/[^`]+)`", skill_md)
    return sorted(set(links + inline))


def assertion_ok(assertion: dict, output: str) -> tuple[bool, str]:
    kind = assertion["kind"]
    if kind == "contains":
        value = assertion["value"]
        return value in output, f"contains {value!r}"
    if kind == "not_contains":
        value = assertion["value"]
        return value not in output, f"does not contain {value!r}"
    if kind == "regex":
        value = assertion["value"]
        return re.search(value, output, re.MULTILINE) is not None, f"matches {value!r}"
    if kind == "exactly_one_of":
        values = assertion["values"]
        count = sum(1 for value in values if value in output)
        return count == 1, f"exactly one of {values!r} present; found {count}"
    raise ValueError(f"unknown assertion kind: {kind}")


def validate_skill(skill_name: str, checks: list[Check]) -> None:
    skill_dir = SKILLS_ROOT / skill_name
    skill_md_path = skill_dir / "SKILL.md"
    evals_path = skill_dir / "evals" / "evals.json"
    agent_path = skill_dir / "agents" / "openai.yaml"

    add(checks, f"{skill_name}:directory", skill_dir.is_dir(), str(skill_dir))
    if not skill_md_path.exists():
        add(checks, f"{skill_name}:skill-md", False, "missing SKILL.md")
        return

    text = skill_md_path.read_text()
    try:
        frontmatter = parse_frontmatter(text)
        add(checks, f"{skill_name}:frontmatter", True, "frontmatter parsed")
    except Exception as exc:
        add(checks, f"{skill_name}:frontmatter", False, str(exc))
        frontmatter = {}

    add(
        checks,
        f"{skill_name}:name",
        frontmatter.get("name") == skill_name,
        f"name={frontmatter.get('name')!r}",
    )
    description = frontmatter.get("description", "")
    add(
        checks,
        f"{skill_name}:description",
        40 <= len(description) <= 1024 and "<" not in description and ">" not in description,
        f"description length={len(description)}",
    )

    add(checks, f"{skill_name}:agent-yaml", agent_path.exists(), str(agent_path))
    if agent_path.exists():
        agent_text = agent_path.read_text()
        add(
            checks,
            f"{skill_name}:agent-default-prompt",
            f"${skill_name}" in agent_text,
            "default prompt references skill name",
        )

    refs = linked_references(text)
    add(checks, f"{skill_name}:references-linked", bool(refs), f"refs={refs}")
    for ref in refs:
        add(checks, f"{skill_name}:reference:{ref}", (skill_dir / ref).exists(), ref)

    guardrail_ok = all(fragment in text for fragment in REQUIRED_GUARDRAILS)
    add(checks, f"{skill_name}:required-guardrails", guardrail_ok, "core guardrails present")

    positive_hits = []
    for pattern in DANGEROUS_POSITIVE_PATTERNS:
        positive_hits.extend(match.group(0) for match in pattern.finditer(text))
    add(
        checks,
        f"{skill_name}:no-dangerous-positive-language",
        not positive_hits,
        "; ".join(positive_hits) if positive_hits else "no hits",
    )

    if not evals_path.exists():
        add(checks, f"{skill_name}:evals-json", False, "missing evals/evals.json")
        return

    try:
        eval_data = json.loads(evals_path.read_text())
        add(checks, f"{skill_name}:evals-json", True, "parsed")
    except Exception as exc:
        add(checks, f"{skill_name}:evals-json", False, str(exc))
        return

    cases = eval_data.get("cases", [])
    add(checks, f"{skill_name}:eval-count", len(cases) >= 3, f"cases={len(cases)}")
    types = {case.get("type") for case in cases}
    add(
        checks,
        f"{skill_name}:eval-types",
        {"positive", "negative", "edge"}.issubset(types),
        f"types={sorted(types)}",
    )

    policy = eval_data.get("model_policy", {})
    model = policy.get("default_live_model", "")
    add(
        checks,
        f"{skill_name}:fast-model-policy",
        "mini" in model or "spark" in model,
        f"default_live_model={model!r}",
    )
    add(
        checks,
        f"{skill_name}:parallel-policy",
        policy.get("parallel") is True,
        f"parallel={policy.get('parallel')!r}",
    )

    for case in cases:
        output = case.get("expected_output", "")
        assertions = case.get("assertions", [])
        add(
            checks,
            f"{skill_name}:{case.get('id')}:assertions-present",
            bool(assertions),
            f"assertions={len(assertions)}",
        )
        for index, assertion in enumerate(assertions):
            try:
                ok, detail = assertion_ok(assertion, output)
            except Exception as exc:
                ok, detail = False, str(exc)
            add(checks, f"{skill_name}:{case.get('id')}:assertion-{index}", ok, detail)


def validate_live_prompt_support(checks: list[Check]) -> None:
    plan_path = ROOT / ".claude" / "skill-evals" / "live_eval_plan.md"
    builder_path = ROOT / ".claude" / "skill-evals" / "build_live_eval_prompts.py"

    add(checks, "live-evals:plan-exists", plan_path.exists(), str(plan_path))
    if plan_path.exists():
        plan = plan_path.read_text()
        plan_lower = plan.lower()
        add(checks, "live-evals:plan-fanout", "fanout" in plan, "mentions fanout")
        add(
            checks,
            "live-evals:plan-fast-model",
            "mini/spark-class" in plan or "gpt-5.3-codex-spark" in plan,
            "mentions fast model policy",
        )
        add(
            checks,
            "live-evals:plan-no-modal",
            "never call modal" in plan_lower and "hidden" in plan_lower and "validation" in plan_lower,
            "documents synthetic no-Modal/no-hidden-validation rule",
        )

    add(checks, "live-evals:prompt-builder-exists", builder_path.exists(), str(builder_path))
    if not builder_path.exists():
        return

    spec = importlib.util.spec_from_file_location("build_live_eval_prompts", builder_path)
    if not spec or not spec.loader:
        add(checks, "live-evals:prompt-builder-import", False, "could not create import spec")
        return

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        add(checks, "live-evals:prompt-builder-import", True, "imported")
    except Exception as exc:
        add(checks, "live-evals:prompt-builder-import", False, str(exc))
        return

    for skill_name in EXPECTED_SKILLS:
        try:
            rendered = module.render_prompt(skill_name)
        except Exception as exc:
            add(checks, f"live-evals:{skill_name}:render", False, str(exc))
            continue
        prompt = rendered.get("prompt", "")
        add(checks, f"live-evals:{skill_name}:render", True, rendered.get("case_id", ""))
        add(
            checks,
            f"live-evals:{skill_name}:inlines-skill",
            "# SKILL.md" in prompt and f"name: {skill_name}" in prompt,
            "inlines SKILL.md",
        )
        add(
            checks,
            f"live-evals:{skill_name}:inlines-reference",
            "# Linked References" in prompt and "references/" in prompt,
            "inlines direct references",
        )
        add(
            checks,
            f"live-evals:{skill_name}:no-expected-output-leak",
            "expected_output" not in prompt and "assertions" not in prompt,
            "does not leak grader answers",
        )
        add(
            checks,
            f"live-evals:{skill_name}:fast-model",
            rendered.get("model") and ("mini" in rendered["model"] or "spark" in rendered["model"]),
            f"model={rendered.get('model')!r}",
        )
        add(
            checks,
            f"live-evals:{skill_name}:parallel",
            rendered.get("parallel") is True,
            f"parallel={rendered.get('parallel')!r}",
        )


def write_artifacts(checks: list[Check]) -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(WORKSPACE_ROOT.glob("iteration-*"))
    next_index = len(existing) + 1
    out_dir = WORKSPACE_ROOT / f"iteration-{next_index:03d}"
    out_dir.mkdir()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "all_passed": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
    }
    (out_dir / "offline-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-artifacts", action="store_true")
    args = parser.parse_args()

    checks: list[Check] = []
    for skill in EXPECTED_SKILLS:
        validate_skill(skill, checks)
    validate_live_prompt_support(checks)

    failed = [check for check in checks if not check.ok]
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status} {check.name}: {check.detail}")

    if args.write_artifacts:
        out_dir = write_artifacts(checks)
        print(f"wrote artifacts: {out_dir}")

    if failed:
        print(f"{len(failed)} check(s) failed", file=sys.stderr)
        return 1
    print(f"all {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
