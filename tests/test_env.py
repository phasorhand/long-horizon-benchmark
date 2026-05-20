# tests/test_env.py
import json
import pytest
from pathlib import Path
from longhorizon_bench.schema import AgentAction, ToolCall, CheckpointResponse


@pytest.fixture
def demo_data_dir(tmp_path: Path) -> Path:
    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    bg_dir = tmp_path / "background_docs" / "DEMO-001"
    bg_dir.mkdir(parents=True)
    (bg_dir / "doc_001.txt").write_text("压力容器定期检验周期为3年。", encoding="utf-8")

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
                "input": "收到巡检报告",
                "depends_on": [],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": True,
                "dimensions": ["domain_knowledge"],
                "scoring_rule": {
                    "tool": {"expected": "submit_inspection_report", "match": "exact"},
                    "params": {"target": {"expected": "3号车间", "match": "contains"}},
                },
                "evidence": {
                    "required_facts": [],
                    "forbidden_actions": [],
                    "acceptable_actions": [],
                },
            },
            {
                "event_id": "E02",
                "type": "routine_inspection",
                "input": "收到第二份报告",
                "depends_on": ["E01"],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": False,
                "dimensions": ["multi_step_reasoning"],
                "scoring_rule": {
                    "tool": {"expected": "no_action", "match": "exact"},
                    "params": {"reason": {"expected": "无需操作", "match": "contains"}},
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
                "input": None,
                "node_type": "checkpoint",
                "is_checkpoint": True,
                "checkpoint_queries": [
                    {
                        "query": "E01的内容是什么？",
                        "expected_keywords": ["巡检"],
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


def test_env_reset_returns_observation(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    obs = env.reset()
    assert "role" in obs
    assert "current_event" in obs
    assert obs["current_event"]["event_id"] == "E01"
    assert "background_docs" in obs


def test_env_step_advances_event(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    env.reset()

    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"}),
    ])
    obs, reward, done, info = env.step(action)
    assert obs["current_event"]["event_id"] == "E02"
    assert done is False


def test_env_checkpoint_step(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    env.reset()

    # Step through E01
    env.step(AgentAction(tool_calls=[ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})]))
    # Step through E02
    env.step(AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "无需操作"})]))

    # Now at checkpoint — obs should indicate checkpoint
    obs = env._current_observation()
    assert obs["current_event"]["node_type"] == "checkpoint"


def test_env_done_after_last_event(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    env.reset()

    env.step(AgentAction(tool_calls=[ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})]))
    env.step(AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "无需操作"})]))
    # Checkpoint
    _, _, done, _ = env.step(CheckpointResponse(answers={"E01的内容是什么？": "收到巡检报告"}))
    assert done is True


def test_env_evaluate_returns_results(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    env.reset()

    env.step(AgentAction(tool_calls=[ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})]))
    env.step(AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "无需操作"})]))
    env.step(CheckpointResponse(answers={"E01的内容是什么？": "收到巡检报告"}))

    results = env.evaluate()
    assert "chain_score" in results
    assert "chain_pass" in results
    assert "event_scores" in results
    assert "dimension_scores" in results
    assert "consistency_violations" in results


def test_env_rolling_window_limits_history(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="rolling_window", window_size=1, data_dir=demo_data_dir)
    env.reset()

    env.step(AgentAction(tool_calls=[ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})]))
    obs, _, _, _ = env.step(AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "无需操作"})]))

    # In rolling window with size 1, only E02 should have full text in history
    full_history = [h for h in obs.get("history", []) if "input" in h and h["input"] is not None]
    summary_history = [h for h in obs.get("history", []) if h.get("summary_only")]
    assert len(full_history) <= 1


def test_env_memory_only_no_background(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="memory_only", data_dir=demo_data_dir)
    obs = env.reset()
    assert obs.get("background_docs") is None or obs.get("background_summary") is not None
