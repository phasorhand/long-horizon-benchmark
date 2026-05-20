# tests/test_runner.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from longhorizon_bench.schema import AgentAction, ToolCall, CheckpointResponse


@pytest.fixture
def demo_data_dir(tmp_path):
    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    bg_dir = tmp_path / "background_docs" / "DEMO-001"
    bg_dir.mkdir(parents=True)
    (bg_dir / "doc_001.txt").write_text("背景文档", encoding="utf-8")

    scenario = {
        "scenario_id": "DEMO-001",
        "domain": "industrial",
        "role": "测试角色",
        "difficulty": 1,
        "background_docs": ["doc_001.txt"],
        "background_tokens": 10,
        "total_events": 1,
        "total_checkpoints": 1,
        "annotator": "test",
        "generation_model": "test",
        "metadata": {},
        "events": [
            {
                "event_id": "E01",
                "type": "test",
                "input": "测试",
                "depends_on": [],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": False,
                "dimensions": ["domain_knowledge"],
                "scoring_rule": {
                    "tool": {"expected": "no_action", "match": "exact"},
                    "params": {},
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
                        "query": "测试",
                        "expected_keywords": ["测试"],
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


def test_base_runner_is_abstract():
    from longhorizon_bench.runners.base import BaseRunner

    with pytest.raises(TypeError):
        BaseRunner()


def test_base_runner_subclass_must_implement_act():
    from longhorizon_bench.runners.base import BaseRunner

    class BadRunner(BaseRunner):
        pass

    with pytest.raises(TypeError):
        BadRunner()


def test_base_runner_subclass_works():
    from longhorizon_bench.runners.base import BaseRunner

    class SimpleRunner(BaseRunner):
        def act(self, observation: dict) -> AgentAction | CheckpointResponse:
            if observation["current_event"]["node_type"] == "checkpoint":
                queries = observation["current_event"].get("queries", [])
                return CheckpointResponse(answers={q: "测试回答" for q in queries})
            return AgentAction(tool_calls=[
                ToolCall(tool="no_action", kwargs={"reason": "test"}),
            ])

    runner = SimpleRunner()
    obs = {"current_event": {"event_id": "E01", "node_type": "action", "input": "test"}}
    result = runner.act(obs)
    assert isinstance(result, AgentAction)


def test_base_runner_run_scenario(demo_data_dir):
    from longhorizon_bench.runners.base import BaseRunner
    from longhorizon_bench.env import LongHorizonEnv

    class DummyRunner(BaseRunner):
        def act(self, observation: dict) -> AgentAction | CheckpointResponse:
            if observation["current_event"]["node_type"] == "checkpoint":
                queries = observation["current_event"].get("queries", [])
                return CheckpointResponse(answers={q: "巡检报告" for q in queries})
            return AgentAction(tool_calls=[
                ToolCall(tool="no_action", kwargs={"reason": "test"}),
            ])

    runner = DummyRunner()
    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    results = runner.run(env)
    assert "chain_score" in results
