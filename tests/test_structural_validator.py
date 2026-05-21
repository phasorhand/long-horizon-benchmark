import json
import pytest
from pathlib import Path


def _make_valid_scenario() -> dict:
    return {
        "scenario_id": "TEST-001",
        "domain": "industrial",
        "role": "测试",
        "difficulty": 1,
        "background_docs": ["background.txt"],
        "background_tokens": 20000,
        "total_events": 2,
        "total_checkpoints": 1,
        "annotator": "test",
        "metadata": {},
        "events": [
            {
                "event_id": "E01", "type": "test", "input": "压力容器检验周期为3年的设备需要安排检查",
                "depends_on": [], "node_type": "action", "is_checkpoint": False, "is_critical": True,
                "dimensions": ["domain_knowledge"],
                "scoring_rule": {"tool": {"expected": "submit_inspection_report", "match": "exact"}, "params": {"target": {"expected": "车间", "match": "contains"}}},
                "evidence": {"required_facts": ["压力容器检验周期为3年"], "forbidden_actions": [], "acceptable_actions": [{"tool": "submit_inspection_report"}]},
            },
            {
                "event_id": "E02", "type": "test", "input": "无需操作的报告",
                "depends_on": ["E01"], "node_type": "action", "is_checkpoint": False, "is_critical": False,
                "dimensions": ["multi_step_reasoning"],
                "scoring_rule": {"tool": {"expected": "no_action", "match": "exact"}, "params": {"reason": {"expected": "无需", "match": "contains"}}},
                "evidence": {"required_facts": ["日常报告"], "forbidden_actions": [], "acceptable_actions": [{"tool": "no_action"}]},
            },
            {
                "event_id": "CP01", "type": "checkpoint", "node_type": "checkpoint", "is_checkpoint": True,
                "checkpoint_queries": [{"query": "test", "expected_keywords": ["压力容器"], "dimension": "long_term_memory", "match": "keyword_coverage"}],
            },
        ],
    }


def test_check_evidence_traceability_pass():
    from longhorizon_bench.pipeline.structural_validator import check_evidence_traceability
    scenario = _make_valid_scenario()
    bg_text = "压力容器检验周期为3年，到期前应安排检查。日常报告管理制度。"
    errors = check_evidence_traceability(scenario, bg_text)
    assert errors == []


def test_check_evidence_traceability_fail():
    from longhorizon_bench.pipeline.structural_validator import check_evidence_traceability
    scenario = _make_valid_scenario()
    bg_text = "完全无关的内容"
    errors = check_evidence_traceability(scenario, bg_text)
    assert len(errors) > 0


def test_check_background_length_pass():
    from longhorizon_bench.pipeline.structural_validator import check_background_length
    errors = check_background_length("内容" * 8000)
    assert errors == []


def test_check_background_length_fail():
    from longhorizon_bench.pipeline.structural_validator import check_background_length
    errors = check_background_length("短")
    assert len(errors) > 0


def test_run_structural_checks():
    from longhorizon_bench.pipeline.structural_validator import run_structural_checks
    scenario = _make_valid_scenario()
    bg_text = "压力容器检验周期为3年。日常报告管理流程。" * 500
    result = run_structural_checks(scenario, bg_text)
    assert "passed" in result
    assert "failed" in result
    assert isinstance(result["passed"], int)
