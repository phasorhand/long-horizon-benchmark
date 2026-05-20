"""Event-level scoring engine."""
from __future__ import annotations
from dataclasses import dataclass
from longhorizon_bench.schema import (
    ActionEvent, AgentAction, CheckpointEvent, CheckpointResponse, MatchType, ParamRule,
)


@dataclass
class ActionScore:
    tool_score: float
    param_scores: dict[str, float]
    total: float


@dataclass
class CheckpointScore:
    query_scores: list[float]
    total: float


class EventScorer:
    def score_action(self, event: ActionEvent, action: AgentAction) -> ActionScore:
        rule = event.scoring_rule
        matching_call = None
        for call in action.tool_calls:
            if call.tool == rule.tool.expected:
                matching_call = call
                break
        if matching_call is None:
            tool_score = 0.0
            param_scores = {name: 0.0 for name in rule.params}
        else:
            tool_score = 1.0
            param_scores = {}
            for name, param_rule in rule.params.items():
                value = matching_call.kwargs.get(name)
                if value is None:
                    param_scores[name] = 0.0
                else:
                    param_scores[name] = self._match_param(param_rule, value)
        if not rule.params:
            total = tool_score
        else:
            param_avg = sum(param_scores.values()) / len(param_scores) if param_scores else 0.0
            total = tool_score * 0.4 + param_avg * 0.6
        return ActionScore(tool_score=tool_score, param_scores=param_scores, total=total)

    def score_checkpoint(self, event: CheckpointEvent, response: CheckpointResponse) -> CheckpointScore:
        query_scores: list[float] = []
        for q in event.checkpoint_queries:
            answer = response.answers.get(q.query, "")
            if q.match == "keyword_coverage" and q.expected_keywords:
                hits = sum(1 for kw in q.expected_keywords if kw in answer)
                query_scores.append(hits / len(q.expected_keywords))
            else:
                query_scores.append(0.0)
        total = sum(query_scores) / len(query_scores) if query_scores else 0.0
        return CheckpointScore(query_scores=query_scores, total=total)

    def _match_param(self, rule: ParamRule, value: str) -> float:
        match rule.match:
            case MatchType.EXACT:
                return 1.0 if str(value) == str(rule.expected) else 0.0
            case MatchType.CONTAINS:
                return 1.0 if str(rule.expected) in str(value) else 0.0
            case MatchType.ENUM:
                allowed = rule.expected if isinstance(rule.expected, list) else [rule.expected]
                return 1.0 if value in allowed else 0.0
            case MatchType.KEYWORD_COVERAGE:
                if not rule.required_keywords:
                    return 1.0
                hits = sum(1 for kw in rule.required_keywords if kw in str(value))
                return hits / len(rule.required_keywords)
            case _:
                return 0.0
