"""Pydantic models for LongHorizon-Bench data schema."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class MatchType(str, Enum):
    EXACT = "exact"
    CONTAINS = "contains"
    ENUM = "enum"
    KEYWORD_COVERAGE = "keyword_coverage"


class ParamRule(BaseModel):
    expected: Any = None
    required_keywords: list[str] | None = None
    match: MatchType


class ScoringRule(BaseModel):
    tool: ParamRule
    params: dict[str, ParamRule] = Field(default_factory=dict)


class Evidence(BaseModel):
    required_facts: list[str]
    forbidden_actions: list[str]
    acceptable_actions: list[dict[str, str]]


class CheckpointQuery(BaseModel):
    query: str
    expected_keywords: list[str] | None = None
    dimension: str
    match: str | None = None
    consistency_rule: str | None = None


class ActionEvent(BaseModel):
    event_id: str
    type: str
    input: str
    depends_on: list[str]
    node_type: Literal["action"] = "action"
    is_checkpoint: Literal[False] = False
    is_critical: bool = False
    dimensions: list[str]
    scoring_rule: ScoringRule
    evidence: Evidence


class CheckpointEvent(BaseModel):
    event_id: str
    type: str
    input: str | None = None
    node_type: Literal["checkpoint"] = "checkpoint"
    is_checkpoint: Literal[True] = True
    checkpoint_queries: list[CheckpointQuery]


Event = Union[ActionEvent, CheckpointEvent]


class ToolCall(BaseModel):
    tool: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    tool_calls: list[ToolCall]


class CheckpointResponse(BaseModel):
    answers: dict[str, str]


class Scenario(BaseModel):
    scenario_id: str
    domain: str
    role: str
    difficulty: int
    background_docs: list[str]
    background_tokens: int
    total_events: int
    total_checkpoints: int
    annotator: str
    annotator_agreement: float | None = None
    generation_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[ActionEvent | CheckpointEvent]

    @property
    def action_events(self) -> list[ActionEvent]:
        return [e for e in self.events if e.node_type == "action"]

    @property
    def checkpoint_events(self) -> list[CheckpointEvent]:
        return [e for e in self.events if e.node_type == "checkpoint"]
