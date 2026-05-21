import pytest
from unittest.mock import MagicMock, patch


SAMPLE_LLM_RESPONSE = """```yaml
- atom_id: ATOM-test-001
  source_cluster: test_cluster
  type: routine_inspection
  trigger: "设备科提交巡检报告，发现压力容器接近检验周期"
  expected_tool: submit_inspection_report
  params:
    target: {value: "3号车间", match: contains}
    risk_level: {value: [medium, high], match: enum}
    findings: {keywords: [压力容器, 检验周期], match: keyword_coverage}
  evidence:
    required_facts: ["压力容器定期检验周期为3年"]
    forbidden_actions: [approve_work_permit]
  dimensions: [domain_knowledge]
  is_critical: true
- atom_id: ATOM-test-002
  source_cluster: test_cluster
  type: incident
  trigger: "车间报告设备异常振动"
  expected_tool: file_incident_report
  params:
    incident_type: {value: equipment_failure, match: exact}
    severity: {value: major, match: exact}
    description: {value: "设备异常振动", match: contains}
  evidence:
    required_facts: ["设备运行记录"]
    forbidden_actions: []
  dimensions: [domain_knowledge, multi_step_reasoning]
  is_critical: false
```"""


def test_parse_atoms_from_llm_response():
    from longhorizon_bench.pipeline.atom_generator import parse_atoms_from_response
    atoms = parse_atoms_from_response(SAMPLE_LLM_RESPONSE)
    assert len(atoms) == 2
    assert atoms[0]["atom_id"] == "ATOM-test-001"
    assert atoms[0]["expected_tool"] == "submit_inspection_report"
    assert atoms[1]["type"] == "incident"


def test_build_atom_prompt():
    from longhorizon_bench.pipeline.atom_generator import build_atom_prompt
    topic_pack = {
        "cluster_id": 0,
        "docs": [{"text": "压力容器检验周期为3年", "doc_id": "d1"}],
        "regulations": [{"title": "安全生产法", "text": "安全生产管理条例"}],
    }
    system, user = build_atom_prompt(topic_pack, n_atoms=3)
    assert "3" in user
    assert "压力容器" in user
    assert "submit_inspection_report" in system or "ToolRegistry" in system


def test_generate_atoms_for_pack():
    from longhorizon_bench.pipeline.atom_generator import generate_atoms_for_pack
    mock_client = MagicMock()
    mock_client.chat.return_value = SAMPLE_LLM_RESPONSE
    topic_pack = {
        "cluster_id": 0,
        "docs": [{"text": "压力容器", "doc_id": "d1"}],
        "regulations": [],
    }
    atoms = generate_atoms_for_pack(mock_client, topic_pack, n_atoms=3)
    assert len(atoms) == 2
    mock_client.chat.assert_called_once()
