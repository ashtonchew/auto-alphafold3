"""LLM phase policy for the future autonomous search harness.

This module intentionally avoids importing the OpenAI Agents SDK. The scaffold
can validate policy locally, while the Modal-hosted harness can translate these
typed settings into real Agent, ModelSettings, and WebSearchTool objects.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ReasoningEffort = Literal["none", "low", "medium", "high"]
ServiceTier = Literal["auto", "default", "priority", "flex"]
TextVerbosity = Literal["low", "medium", "high"]
WebSearchContextSize = Literal["low", "medium", "high"]
DEFAULT_LLM_MODEL = "gpt-5.4-mini"


class AgentSearchPhase(StrEnum):
    """Autonomous-search LLM phases with different tool and reasoning budgets."""

    HYPOTHESIS_GENERATION = "hypothesis_generation"
    PATCH_PLANNING = "patch_planning"


class LLMPhasePolicy(BaseModel):
    """Serializable policy for one LLM phase in the search loop."""

    model_config = ConfigDict(extra="forbid")

    phase: AgentSearchPhase
    model: str = Field(default=DEFAULT_LLM_MODEL, min_length=1)
    service_tier: ServiceTier = "priority"
    reasoning_effort: ReasoningEffort
    text_verbosity: TextVerbosity = "low"
    web_search_enabled: bool
    web_search_context_size: WebSearchContextSize = "medium"

    @model_validator(mode="after")
    def validate_phase_contract(self) -> LLMPhasePolicy:
        if self.phase == AgentSearchPhase.HYPOTHESIS_GENERATION:
            if not self.web_search_enabled:
                raise ValueError("hypothesis generation requires web search")
            if self.reasoning_effort != "low":
                raise ValueError("hypothesis generation uses low reasoning")
        if self.phase == AgentSearchPhase.PATCH_PLANNING:
            if self.web_search_enabled:
                raise ValueError("patch planning must not use web search")
            if self.reasoning_effort != "low":
                raise ValueError("patch planning uses low reasoning")
        return self

    def to_responses_create_kwargs(self) -> dict[str, Any]:
        """Return OpenAI Responses API kwargs for this phase policy."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "service_tier": self.service_tier,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {"verbosity": self.text_verbosity},
        }
        if self.web_search_enabled:
            kwargs["tools"] = [
                {
                    "type": "web_search",
                    "search_context_size": self.web_search_context_size,
                }
            ]
        else:
            kwargs["tools"] = []
        return kwargs

    def to_agents_sdk_spec(self) -> dict[str, Any]:
        """Return a dependency-free spec for constructing an Agents SDK Agent."""

        tools: list[dict[str, Any]] = []
        if self.web_search_enabled:
            tools.append(
                {
                    "class": "WebSearchTool",
                    "kwargs": {"search_context_size": self.web_search_context_size},
                }
            )
        return {
            "model": self.model,
            "model_settings": {
                "reasoning": {"effort": self.reasoning_effort},
                "verbosity": self.text_verbosity,
                "extra_args": {"service_tier": self.service_tier},
            },
            "tools": tools,
        }


def default_llm_phase_policies(model: str = DEFAULT_LLM_MODEL) -> dict[AgentSearchPhase, LLMPhasePolicy]:
    """Return the project default LLM settings for autonomous search phases."""

    return {
        AgentSearchPhase.HYPOTHESIS_GENERATION: LLMPhasePolicy(
            phase=AgentSearchPhase.HYPOTHESIS_GENERATION,
            model=model,
            service_tier="priority",
            reasoning_effort="low",
            text_verbosity="low",
            web_search_enabled=True,
            web_search_context_size="medium",
        ),
        AgentSearchPhase.PATCH_PLANNING: LLMPhasePolicy(
            phase=AgentSearchPhase.PATCH_PLANNING,
            model=model,
            service_tier="priority",
            reasoning_effort="low",
            text_verbosity="low",
            web_search_enabled=False,
        ),
    }


def default_llm_phase_policy(phase: AgentSearchPhase | str, model: str = DEFAULT_LLM_MODEL) -> LLMPhasePolicy:
    """Return the default policy for one phase."""

    normalized = AgentSearchPhase(phase)
    return default_llm_phase_policies(model=model)[normalized]
