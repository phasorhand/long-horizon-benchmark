import pytest
from unittest.mock import MagicMock


def test_fill_action_event():
    from longhorizon_bench.pipeline.event_filler import fill_action_event
    mock_client = MagicMock()
    mock_client.chat.return_value = "收到设备科提交的3号车间季度巡检报告，其中提到1台压力容器接近检验周期，根据《压力容器安全技术监察规程》第34条，应在到期前3个月启动检验申请。"
    atom = {
        "atom_id": "ATOM-001",
        "type": "routine_inspection",
        "trigger": "设备老化报告",
        "expected_tool": "submit_inspection_report",
        "params": {
            "target": {"value": "3号车间", "match": "contains"},
            "findings": {"keywords": ["压力容器", "检验周期"], "match": "keyword_coverage"},
        },
        "evidence": {"required_facts": ["压力容器检验周期为3年"], "forbidden_actions": []},
        "dimensions": ["domain_knowledge"],
        "is_critical": True,
    }
    event_input = fill_action_event(mock_client, atom, "背景文档内容", [])
    assert isinstance(event_input, str)
    assert len(event_input) > 0


def test_fill_checkpoint_queries():
    from longhorizon_bench.pipeline.event_filler import fill_checkpoint_queries
    mock_client = MagicMock()
    mock_client.chat.return_value = """```yaml
- query: "E01中提到的压力容器情况是什么？"
  expected_keywords: ["压力容器", "检验周期"]
  dimension: long_term_memory
  match: keyword_coverage
- query: "前序事件中安全隐患的处理结果如何？"
  expected_keywords: ["整改", "隐患"]
  dimension: consistency
  match: keyword_coverage
```"""
    prior_events = [
        {"event_id": "E01", "input": "压力容器接近检验周期"},
        {"event_id": "E02", "input": "安全隐患排查整改"},
    ]
    queries = fill_checkpoint_queries(
        mock_client, prior_events, target_dimensions=["long_term_memory", "consistency"]
    )
    assert len(queries) == 2
    assert queries[0]["dimension"] == "long_term_memory"
