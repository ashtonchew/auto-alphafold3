from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from autoalphafold3 import agent
from autoalphafold3.llm_policy import DEFAULT_LLM_MODEL, AgentSearchPhase, LLMPhasePolicy, default_llm_phase_policy


def test_hypothesis_generation_policy_uses_web_search_low_reasoning_and_priority() -> None:
    policy = default_llm_phase_policy(AgentSearchPhase.HYPOTHESIS_GENERATION)

    assert policy.model == DEFAULT_LLM_MODEL
    assert policy.service_tier == "priority"
    assert policy.reasoning_effort == "low"
    assert policy.web_search_enabled is True

    responses_kwargs = policy.to_responses_create_kwargs()
    assert responses_kwargs["reasoning"] == {"effort": "low"}
    assert responses_kwargs["service_tier"] == "priority"
    assert responses_kwargs["tools"] == [{"type": "web_search", "search_context_size": "medium"}]

    agents_spec = policy.to_agents_sdk_spec()
    assert agents_spec["model_settings"]["extra_args"] == {"service_tier": "priority"}
    assert agents_spec["model_settings"]["reasoning"] == {"effort": "low"}
    assert agents_spec["tools"] == [{"class": "WebSearchTool", "kwargs": {"search_context_size": "medium"}}]


def test_patch_planning_policy_disables_web_search_and_uses_low_reasoning() -> None:
    policy = default_llm_phase_policy("patch_planning")

    assert policy.service_tier == "priority"
    assert policy.reasoning_effort == "low"
    assert policy.web_search_enabled is False
    assert policy.to_responses_create_kwargs()["tools"] == []
    assert policy.to_agents_sdk_spec()["tools"] == []


def test_phase_policy_rejects_contract_drift() -> None:
    with pytest.raises(ValidationError, match="hypothesis generation requires web search"):
        LLMPhasePolicy(
            phase="hypothesis_generation",
            reasoning_effort="low",
            web_search_enabled=False,
        )

    with pytest.raises(ValidationError, match="patch planning uses low reasoning"):
        LLMPhasePolicy(
            phase="patch_planning",
            reasoning_effort="medium",
            web_search_enabled=False,
        )


def test_agent_llm_policy_cli_outputs_agents_sdk_spec(capsys: pytest.CaptureFixture[str]) -> None:
    assert agent.main(["llm-policy", "--phase", "patch_planning", "--format", "agents-sdk"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == DEFAULT_LLM_MODEL
    assert payload["model_settings"]["reasoning"] == {"effort": "low"}
    assert payload["model_settings"]["extra_args"] == {"service_tier": "priority"}
    assert payload["tools"] == []
