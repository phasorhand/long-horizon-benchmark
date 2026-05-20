"""End-to-end integration test: load scenario → run agent → evaluate."""

import json
import pytest
from pathlib import Path

from longhorizon_bench.env import LongHorizonEnv
from longhorizon_bench.schema import AgentAction, CheckpointResponse, ToolCall
from longhorizon_bench.runners.base import BaseRunner


class PerfectAgent(BaseRunner):
    """Agent that gives the 'correct' answers for DEMO-001."""

    def act(self, observation):
        event = observation["current_event"]

        if event["node_type"] == "checkpoint":
            queries = event.get("queries", [])
            return CheckpointResponse(answers={
                q: "E01中提到1台压力容器接近检验周期，需要安排检验" for q in queries
            })

        if event["event_id"] == "E01":
            return AgentAction(tool_calls=[
                ToolCall(tool="submit_inspection_report", kwargs={
                    "target": "3号车间设备",
                    "findings": "压力容器接近检验周期，需要安排定期检验",
                    "risk_level": "medium",
                }),
            ])

        return AgentAction(tool_calls=[
            ToolCall(tool="no_action", kwargs={"reason": "无需操作，继续观察"}),
        ])


class BadAgent(BaseRunner):
    """Agent that always picks the wrong tool."""

    def act(self, observation):
        event = observation["current_event"]

        if event["node_type"] == "checkpoint":
            queries = event.get("queries", [])
            return CheckpointResponse(answers={q: "不记得了" for q in queries})

        return AgentAction(tool_calls=[
            ToolCall(tool="approve_work_permit", kwargs={
                "permit_type": "hot_work",
                "conditions": [],
                "approved": True,
            }),
        ])


@pytest.fixture
def demo_data_dir(tmp_path):
    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    bg_dir = tmp_path / "background_docs" / "DEMO-001"
    bg_dir.mkdir(parents=True)
    (bg_dir / "doc_001.txt").write_text(
        "压力容器定期检验周期为3年。", encoding="utf-8"
    )

    scenario = {
        "scenario_id": "DEMO-001",
        "domain": "industrial",
        "role": "化工厂安全生产管理员",
        "difficulty": 1,
        "background_docs": ["doc_001.txt"],
        "background_tokens": 50,
        "total_events": 2,
        "total_checkpoints": 1,
        "annotator": "test",
        "generation_model": "test",
        "metadata": {},
        "events": [
            {
                "event_id": "E01",
                "type": "routine_inspection",
                "input": "收到设备科提交的3号车间季度巡检报告，其中提到1台压力容器接近检验周期。",
                "depends_on": [],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": True,
                "dimensions": ["domain_knowledge"],
                "scoring_rule": {
                    "tool": {"expected": "submit_inspection_report", "match": "exact"},
                    "params": {
                        "target": {"expected": "3号车间", "match": "contains"},
                        "findings": {"required_keywords": ["压力容器", "检验周期"], "match": "keyword_coverage"},
                        "risk_level": {"expected": ["medium", "high"], "match": "enum"},
                    },
                },
                "evidence": {
                    "required_facts": ["背景文档: 压力容器定期检验周期为3年"],
                    "forbidden_actions": ["approve_work_permit"],
                    "acceptable_actions": [{"tool": "submit_inspection_report"}],
                },
            },
            {
                "event_id": "E02",
                "type": "routine_report",
                "input": "本月安全生产月报需要编制。",
                "depends_on": ["E01"],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": False,
                "dimensions": ["multi_step_reasoning"],
                "scoring_rule": {
                    "tool": {"expected": "no_action", "match": "exact"},
                    "params": {"reason": {"expected": "无需", "match": "contains"}},
                },
                "evidence": {
                    "required_facts": [],
                    "forbidden_actions": [],
                    "acceptable_actions": [],
                },
            },
            {
                "event_id": "CP01",
                "type": "checkpoint",
                "node_type": "checkpoint",
                "is_checkpoint": True,
                "checkpoint_queries": [
                    {
                        "query": "E01中提到的压力容器情况是什么？",
                        "expected_keywords": ["压力容器", "检验周期"],
                        "dimension": "long_term_memory",
                        "match": "keyword_coverage",
                    },
                ],
            },
        ],
    }
    (scenario_dir / "DEMO-001.json").write_text(
        json.dumps(scenario, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def test_perfect_agent_scores_high(demo_data_dir):
    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    agent = PerfectAgent()
    results = agent.run(env)

    assert results["chain_score"] > 0.8
    assert results["chain_pass"] is True
    assert len(results["consistency_violations"]) == 0
    assert results["dimension_scores"]["long_term_memory"] == 1.0


def test_bad_agent_scores_low(demo_data_dir):
    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    agent = BadAgent()
    results = agent.run(env)

    assert results["chain_score"] < 0.5
    assert results["chain_pass"] is False


def test_rolling_window_mode_works(demo_data_dir):
    env = LongHorizonEnv("industrial/DEMO-001", mode="rolling_window", data_dir=demo_data_dir)
    agent = PerfectAgent()
    results = agent.run(env)

    assert results["chain_score"] > 0.8


def test_memory_only_mode_works(demo_data_dir):
    env = LongHorizonEnv("industrial/DEMO-001", mode="memory_only", data_dir=demo_data_dir)
    agent = PerfectAgent()
    results = agent.run(env)

    assert "chain_score" in results


def test_mode_comparison_shows_delta(demo_data_dir):
    """The core thesis: Mode A vs Mode B delta measures memory reliance."""
    agent = PerfectAgent()

    env_a = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    results_a = agent.run(env_a)

    env_b = LongHorizonEnv("industrial/DEMO-001", mode="rolling_window", data_dir=demo_data_dir)
    results_b = agent.run(env_b)

    assert results_a["chain_score"] > 0
    assert results_b["chain_score"] > 0
