import pytest
from datetime import date


def test_match_type_enum_values():
    from longhorizon_bench.schema import MatchType
    assert MatchType.EXACT == "exact"
    assert MatchType.CONTAINS == "contains"
    assert MatchType.ENUM == "enum"
    assert MatchType.KEYWORD_COVERAGE == "keyword_coverage"


def test_param_rule_exact_match():
    from longhorizon_bench.schema import ParamRule
    rule = ParamRule(expected="3号车间", match="exact")
    assert rule.expected == "3号车间"
    assert rule.match == "exact"


def test_param_rule_enum_match_accepts_list():
    from longhorizon_bench.schema import ParamRule
    rule = ParamRule(expected=["medium", "high"], match="enum")
    assert rule.expected == ["medium", "high"]


def test_param_rule_keyword_coverage():
    from longhorizon_bench.schema import ParamRule
    rule = ParamRule(required_keywords=["压力容器", "检验周期"], match="keyword_coverage")
    assert rule.required_keywords == ["压力容器", "检验周期"]


def test_scoring_rule_structure():
    from longhorizon_bench.schema import ScoringRule, ParamRule
    rule = ScoringRule(
        tool=ParamRule(expected="submit_inspection_report", match="exact"),
        params={
            "target": ParamRule(expected="3号车间", match="contains"),
            "risk_level": ParamRule(expected=["medium", "high"], match="enum"),
        },
    )
    assert rule.tool.expected == "submit_inspection_report"
    assert "target" in rule.params


def test_evidence_structure():
    from longhorizon_bench.schema import Evidence
    ev = Evidence(
        required_facts=["E01: 压力容器接近检验周期"],
        forbidden_actions=["approve_work_permit"],
        acceptable_actions=[{"tool": "submit_inspection_report"}],
    )
    assert len(ev.required_facts) == 1
    assert len(ev.forbidden_actions) == 1


def test_action_event_creation():
    from longhorizon_bench.schema import ActionEvent, ScoringRule, ParamRule, Evidence
    event = ActionEvent(
        event_id="E01",
        type="routine_inspection",
        input="收到设备科提交的3号车间季度巡检报告",
        depends_on=[],
        is_critical=True,
        dimensions=["domain_knowledge"],
        scoring_rule=ScoringRule(
            tool=ParamRule(expected="submit_inspection_report", match="exact"),
            params={},
        ),
        evidence=Evidence(required_facts=[], forbidden_actions=[], acceptable_actions=[]),
    )
    assert event.node_type == "action"
    assert event.is_checkpoint is False
    assert event.is_critical is True


def test_checkpoint_event_creation():
    from longhorizon_bench.schema import CheckpointEvent, CheckpointQuery
    event = CheckpointEvent(
        event_id="E05",
        type="checkpoint",
        checkpoint_queries=[
            CheckpointQuery(
                query="E02中发现的隐患当前状态？",
                expected_keywords=["整改中", "未完成"],
                dimension="long_term_memory",
                match="keyword_coverage",
            ),
        ],
    )
    assert event.node_type == "checkpoint"
    assert event.is_checkpoint is True
    assert len(event.checkpoint_queries) == 1


def test_scenario_creation():
    from longhorizon_bench.schema import (
        Scenario, ActionEvent, CheckpointEvent, CheckpointQuery,
        ScoringRule, ParamRule, Evidence,
    )
    scenario = Scenario(
        scenario_id="DEMO-001",
        domain="industrial",
        role="测试角色",
        difficulty=1,
        background_docs=["doc_001.txt"],
        background_tokens=1000,
        total_events=1,
        total_checkpoints=1,
        annotator="test",
        annotator_agreement=0.85,
        generation_model="test-model",
        metadata={"source": "test"},
        events=[
            ActionEvent(
                event_id="E01",
                type="routine_inspection",
                input="测试事件",
                depends_on=[],
                is_critical=False,
                dimensions=["domain_knowledge"],
                scoring_rule=ScoringRule(
                    tool=ParamRule(expected="no_action", match="exact"),
                    params={},
                ),
                evidence=Evidence(required_facts=[], forbidden_actions=[], acceptable_actions=[]),
            ),
            CheckpointEvent(
                event_id="CP01",
                type="checkpoint",
                checkpoint_queries=[
                    CheckpointQuery(
                        query="测试问题",
                        expected_keywords=["测试"],
                        dimension="long_term_memory",
                        match="keyword_coverage",
                    ),
                ],
            ),
        ],
    )
    assert scenario.scenario_id == "DEMO-001"
    assert len(scenario.events) == 2
    assert scenario.action_events[0].event_id == "E01"
    assert scenario.checkpoint_events[0].event_id == "CP01"


def test_agent_action_single_tool():
    from longhorizon_bench.schema import AgentAction, ToolCall
    action = AgentAction(
        tool_calls=[ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})]
    )
    assert len(action.tool_calls) == 1


def test_agent_action_multiple_tools():
    from longhorizon_bench.schema import AgentAction, ToolCall
    action = AgentAction(
        tool_calls=[
            ToolCall(tool="file_incident_report", kwargs={"severity": "major"}),
            ToolCall(tool="escalate_to_management", kwargs={"urgency": "emergency"}),
        ]
    )
    assert len(action.tool_calls) == 2
