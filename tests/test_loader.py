# tests/test_loader.py
import json
import pytest
from pathlib import Path


@pytest.fixture
def demo_scenario_path(tmp_path: Path) -> Path:
    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    bg_dir = tmp_path / "background_docs" / "DEMO-001"
    bg_dir.mkdir(parents=True)
    (bg_dir / "doc_001.txt").write_text("这是背景文档内容，压力容器定期检验周期为3年。", encoding="utf-8")

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
        "annotator_agreement": 0.85,
        "generation_model": "test-model",
        "metadata": {"source": "test"},
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
                "event_id": "CP01",
                "type": "checkpoint",
                "input": None,
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
    (scenario_dir / "DEMO-001.json").write_text(json.dumps(scenario, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def test_load_scenario_from_path(demo_scenario_path):
    from longhorizon_bench.loader import load_scenario

    scenario = load_scenario("industrial/DEMO-001", data_dir=demo_scenario_path)
    assert scenario.scenario_id == "DEMO-001"
    assert len(scenario.events) == 2


def test_load_background_docs(demo_scenario_path):
    from longhorizon_bench.loader import load_background_docs

    docs = load_background_docs("DEMO-001", ["doc_001.txt"], data_dir=demo_scenario_path)
    assert "压力容器" in docs["doc_001.txt"]


def test_load_scenario_file_not_found():
    from longhorizon_bench.loader import load_scenario

    with pytest.raises(FileNotFoundError):
        load_scenario("industrial/NONEXISTENT", data_dir=Path("/nonexistent"))


def test_load_scenario_validation_error(tmp_path):
    from longhorizon_bench.loader import load_scenario

    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "BAD-001.json").write_text('{"scenario_id": "BAD"}', encoding="utf-8")

    with pytest.raises(Exception):  # Pydantic ValidationError
        load_scenario("industrial/BAD-001", data_dir=tmp_path)
