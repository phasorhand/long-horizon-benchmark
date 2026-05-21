import json
import pytest
from pathlib import Path


def _make_chain() -> dict:
    return {
        "scenario_id": "IND-TEST",
        "domain": "industrial",
        "subdomain": "safety_production",
        "role": "化工厂安全生产管理员",
        "difficulty": 3,
        "events": [
            {"id": "E01", "atom_ref": "ATOM-001", "depends_on": [], "is_critical": True, "dimensions": ["domain_knowledge"]},
            {"id": "E02", "atom_ref": "ATOM-002", "depends_on": ["E01"], "is_critical": False, "dimensions": ["multi_step_reasoning"]},
        ],
        "checkpoints": [
            {"id": "CP01", "after": "E02", "queries_target_dimensions": ["long_term_memory"]},
        ],
    }


def _make_atoms() -> dict:
    return {
        "ATOM-001": {
            "atom_id": "ATOM-001",
            "type": "routine_inspection",
            "expected_tool": "submit_inspection_report",
            "params": {
                "target": {"value": "3号车间", "match": "contains"},
                "risk_level": {"value": ["medium", "high"], "match": "enum"},
            },
            "evidence": {
                "required_facts": ["压力容器检验周期为3年"],
                "forbidden_actions": ["approve_work_permit"],
                "acceptable_actions": [{"tool": "submit_inspection_report"}],
            },
        },
        "ATOM-002": {
            "atom_id": "ATOM-002",
            "type": "routine_report",
            "expected_tool": "no_action",
            "params": {"reason": {"value": "无需", "match": "contains"}},
            "evidence": {
                "required_facts": ["日常报告"],
                "forbidden_actions": [],
                "acceptable_actions": [{"tool": "no_action"}],
            },
        },
    }


def test_assemble_action_event():
    from longhorizon_bench.pipeline.assembler import assemble_action_event
    atom = _make_atoms()["ATOM-001"]
    chain_event = {"id": "E01", "atom_ref": "ATOM-001", "depends_on": [], "is_critical": True, "dimensions": ["domain_knowledge"]}
    event_input = "收到巡检报告，发现压力容器接近检验周期"
    action_event = assemble_action_event(chain_event, atom, event_input)
    assert action_event["event_id"] == "E01"
    assert action_event["node_type"] == "action"
    assert action_event["scoring_rule"]["tool"]["expected"] == "submit_inspection_report"
    assert "target" in action_event["scoring_rule"]["params"]


def test_assemble_checkpoint_event():
    from longhorizon_bench.pipeline.assembler import assemble_checkpoint_event
    queries = [
        {"query": "压力容器情况如何？", "expected_keywords": ["压力容器"], "dimension": "long_term_memory", "match": "keyword_coverage"},
    ]
    cp_event = assemble_checkpoint_event("CP01", queries)
    assert cp_event["event_id"] == "CP01"
    assert cp_event["node_type"] == "checkpoint"
    assert cp_event["is_checkpoint"] is True
    assert len(cp_event["checkpoint_queries"]) == 1


def test_assemble_scenario():
    from longhorizon_bench.pipeline.assembler import assemble_scenario
    from longhorizon_bench.schema import Scenario
    chain = _make_chain()
    atoms = _make_atoms()
    event_inputs = {"E01": "巡检报告内容", "E02": "月报编制"}
    checkpoint_queries = {"CP01": [{"query": "test", "expected_keywords": ["test"], "dimension": "long_term_memory", "match": "keyword_coverage"}]}
    bg_text = "背景文档" * 100
    scenario_dict = assemble_scenario(chain, atoms, event_inputs, checkpoint_queries, bg_text)
    scenario = Scenario.model_validate(scenario_dict)
    assert scenario.scenario_id == "IND-TEST"
    assert len(scenario.action_events) == 2
    assert len(scenario.checkpoint_events) == 1


def test_save_scenario(tmp_path):
    from longhorizon_bench.pipeline.assembler import save_scenario
    scenario_dict = {"scenario_id": "IND-TEST", "domain": "industrial"}
    save_scenario(scenario_dict, tmp_path)
    path = tmp_path / "scenarios" / "industrial" / "IND-TEST.json"
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["scenario_id"] == "IND-TEST"
