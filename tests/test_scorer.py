import pytest
from longhorizon_bench.schema import (
    ActionEvent, AgentAction, ToolCall, ScoringRule, ParamRule, Evidence,
    CheckpointEvent, CheckpointQuery, CheckpointResponse,
)


def _make_event(**overrides) -> ActionEvent:
    defaults = dict(
        event_id="E01", type="routine_inspection", input="测试", depends_on=[],
        is_critical=False, dimensions=["domain_knowledge"],
        scoring_rule=ScoringRule(
            tool=ParamRule(expected="submit_inspection_report", match="exact"),
            params={
                "target": ParamRule(expected="3号车间", match="contains"),
                "risk_level": ParamRule(expected=["medium", "high"], match="enum"),
                "findings": ParamRule(required_keywords=["压力容器", "检验周期"], match="keyword_coverage"),
            },
        ),
        evidence=Evidence(required_facts=[], forbidden_actions=[], acceptable_actions=[]),
    )
    defaults.update(overrides)
    return ActionEvent(**defaults)


def test_perfect_score():
    from longhorizon_bench.evaluation.scorer import EventScorer
    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={
            "target": "3号车间设备", "risk_level": "high",
            "findings": "发现压力容器接近检验周期，需要排查",
        }),
    ])
    result = scorer.score_action(event, action)
    assert result.tool_score == 1.0
    assert result.param_scores["target"] == 1.0
    assert result.param_scores["risk_level"] == 1.0
    assert result.param_scores["findings"] == 1.0
    assert result.total == 1.0


def test_wrong_tool():
    from longhorizon_bench.evaluation.scorer import EventScorer
    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "无需操作"})])
    result = scorer.score_action(event, action)
    assert result.tool_score == 0.0
    assert result.total < 0.5


def test_exact_match_wrong_value():
    from longhorizon_bench.evaluation.scorer import EventScorer
    event = _make_event(
        scoring_rule=ScoringRule(
            tool=ParamRule(expected="no_action", match="exact"),
            params={"reason": ParamRule(expected="无需操作", match="exact")},
        ),
    )
    scorer = EventScorer()
    action = AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "不需要做什么"})])
    result = scorer.score_action(event, action)
    assert result.tool_score == 1.0
    assert result.param_scores["reason"] == 0.0


def test_contains_match():
    from longhorizon_bench.evaluation.scorer import EventScorer
    scorer = EventScorer()
    event = _make_event(
        scoring_rule=ScoringRule(
            tool=ParamRule(expected="respond_to_query", match="exact"),
            params={"content": ParamRule(expected="安全隐患", match="contains")},
        ),
    )
    action = AgentAction(tool_calls=[
        ToolCall(tool="respond_to_query", kwargs={"content": "经排查，存在安全隐患需要整改"}),
    ])
    result = scorer.score_action(event, action)
    assert result.param_scores["content"] == 1.0


def test_enum_match_accepts_any_in_list():
    from longhorizon_bench.evaluation.scorer import EventScorer
    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={
            "target": "3号车间", "risk_level": "medium",
            "findings": "压力容器检验周期到了",
        }),
    ])
    result = scorer.score_action(event, action)
    assert result.param_scores["risk_level"] == 1.0


def test_keyword_coverage_partial():
    from longhorizon_bench.evaluation.scorer import EventScorer
    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={
            "target": "3号车间", "risk_level": "high",
            "findings": "发现压力容器有问题",
        }),
    ])
    result = scorer.score_action(event, action)
    assert result.param_scores["findings"] == 0.5


def test_missing_param_scores_zero():
    from longhorizon_bench.evaluation.scorer import EventScorer
    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"}),
    ])
    result = scorer.score_action(event, action)
    assert result.param_scores["risk_level"] == 0.0
    assert result.param_scores["findings"] == 0.0


def test_multi_tool_action_scores_first_matching():
    from longhorizon_bench.evaluation.scorer import EventScorer
    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="update_safety_ledger", kwargs={"item_id": "x", "status": "open", "remarks": "y"}),
        ToolCall(tool="submit_inspection_report", kwargs={
            "target": "3号车间", "risk_level": "high", "findings": "压力容器检验周期到了",
        }),
    ])
    result = scorer.score_action(event, action)
    assert result.tool_score == 1.0


def test_checkpoint_scoring():
    from longhorizon_bench.evaluation.scorer import EventScorer
    scorer = EventScorer()
    event = CheckpointEvent(
        event_id="CP01", type="checkpoint",
        checkpoint_queries=[
            CheckpointQuery(
                query="E02中发现的隐患当前状态？",
                expected_keywords=["整改中", "未完成", "3号车间"],
                dimension="long_term_memory", match="keyword_coverage",
            ),
        ],
    )
    response = CheckpointResponse(
        answers={"E02中发现的隐患当前状态？": "3号车间的隐患目前整改中，尚未完成"}
    )
    result = scorer.score_checkpoint(event, response)
    assert result.query_scores[0] == 1.0
