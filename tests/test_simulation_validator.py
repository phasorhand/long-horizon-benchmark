import json
import pytest
from pathlib import Path
from longhorizon_bench.schema import AgentAction, CheckpointResponse, ToolCall
from longhorizon_bench.runners.base import BaseRunner


def test_perfect_agent_from_scenario():
    from longhorizon_bench.pipeline.simulation_validator import build_perfect_agent
    scenario_dict = {
        "events": [
            {
                "event_id": "E01", "node_type": "action",
                "scoring_rule": {
                    "tool": {"expected": "no_action", "match": "exact"},
                    "params": {"reason": {"expected": "test", "match": "contains"}},
                },
            },
            {
                "event_id": "CP01", "node_type": "checkpoint",
                "checkpoint_queries": [
                    {"query": "test?", "expected_keywords": ["answer"], "dimension": "long_term_memory", "match": "keyword_coverage"},
                ],
            },
        ]
    }
    agent = build_perfect_agent(scenario_dict)
    obs_action = {"current_event": {"event_id": "E01", "node_type": "action", "input": "test"}}
    result = agent.act(obs_action)
    assert isinstance(result, AgentAction)
    assert result.tool_calls[0].tool == "no_action"

    obs_cp = {"current_event": {"event_id": "CP01", "node_type": "checkpoint", "queries": ["test?"]}}
    result_cp = agent.act(obs_cp)
    assert isinstance(result_cp, CheckpointResponse)
    assert "answer" in result_cp.answers["test?"]


def test_bad_agent():
    from longhorizon_bench.pipeline.simulation_validator import BadSimAgent
    agent = BadSimAgent()
    obs = {"current_event": {"event_id": "E01", "node_type": "action", "input": "test"}}
    result = agent.act(obs)
    assert isinstance(result, AgentAction)
    assert result.tool_calls[0].tool == "approve_work_permit"


def test_run_simulation():
    from longhorizon_bench.pipeline.simulation_validator import run_simulation
    scenario_dict = {
        "scenario_id": "TEST",
        "domain": "industrial",
        "role": "测试",
        "difficulty": 1,
        "background_docs": ["bg.txt"],
        "background_tokens": 100,
        "total_events": 1,
        "total_checkpoints": 1,
        "annotator": "test",
        "metadata": {},
        "events": [
            {
                "event_id": "E01", "type": "test", "input": "测试",
                "depends_on": [], "node_type": "action", "is_checkpoint": False,
                "is_critical": False, "dimensions": ["domain_knowledge"],
                "scoring_rule": {"tool": {"expected": "no_action", "match": "exact"}, "params": {"reason": {"expected": "ok", "match": "contains"}}},
                "evidence": {"required_facts": ["test"], "forbidden_actions": [], "acceptable_actions": []},
            },
            {
                "event_id": "CP01", "type": "checkpoint", "node_type": "checkpoint",
                "is_checkpoint": True,
                "checkpoint_queries": [{"query": "测试?", "expected_keywords": ["测试"], "dimension": "long_term_memory", "match": "keyword_coverage"}],
            },
        ],
    }
    result = run_simulation(scenario_dict, tmp_data_dir=None)
    assert "perfect_agent_score" in result
    assert "bad_agent_score" in result
    assert "delta" in result
    assert result["perfect_agent_score"] > result["bad_agent_score"]
