import pytest


def test_common_tools_registered():
    from longhorizon_bench.tools.registry import ToolRegistry
    from longhorizon_bench.tools.common import COMMON_TOOLS
    reg = ToolRegistry(COMMON_TOOLS)
    assert "retrieve_policy" in reg
    assert "inspect_history" in reg
    assert "request_clarification" in reg
    assert "no_action" in reg
    assert "respond_to_query" in reg


def test_industrial_tools_registered():
    from longhorizon_bench.tools.industrial import INDUSTRIAL_TOOLS
    assert "submit_inspection_report" in INDUSTRIAL_TOOLS
    assert "file_incident_report" in INDUSTRIAL_TOOLS
    assert "approve_work_permit" in INDUSTRIAL_TOOLS


def test_registry_merge():
    from longhorizon_bench.tools.registry import ToolRegistry
    from longhorizon_bench.tools.common import COMMON_TOOLS
    from longhorizon_bench.tools.industrial import INDUSTRIAL_TOOLS
    reg = ToolRegistry({**COMMON_TOOLS, **INDUSTRIAL_TOOLS})
    assert "retrieve_policy" in reg
    assert "submit_inspection_report" in reg


def test_validate_action_valid():
    from longhorizon_bench.tools.registry import ToolRegistry, ToolDef
    from longhorizon_bench.schema import ToolCall
    reg = ToolRegistry({"no_action": ToolDef(params={"reason": {"type": "str"}})})
    errors = reg.validate_tool_call(ToolCall(tool="no_action", kwargs={"reason": "无需操作"}))
    assert errors == []


def test_validate_action_unknown_tool():
    from longhorizon_bench.tools.registry import ToolRegistry
    from longhorizon_bench.schema import ToolCall
    reg = ToolRegistry({})
    errors = reg.validate_tool_call(ToolCall(tool="nonexistent", kwargs={}))
    assert len(errors) == 1
    assert "unknown tool" in errors[0].lower()


def test_validate_action_missing_required_param():
    from longhorizon_bench.tools.registry import ToolRegistry, ToolDef
    from longhorizon_bench.schema import ToolCall
    reg = ToolRegistry({
        "submit_inspection_report": ToolDef(params={
            "target": {"type": "str"},
            "findings": {"type": "str"},
            "risk_level": {"type": "enum", "values": ["low", "medium", "high", "critical"]},
        }),
    })
    errors = reg.validate_tool_call(
        ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})
    )
    assert len(errors) == 2


def test_validate_enum_param_invalid_value():
    from longhorizon_bench.tools.registry import ToolRegistry, ToolDef
    from longhorizon_bench.schema import ToolCall
    reg = ToolRegistry({
        "file_incident_report": ToolDef(params={
            "incident_type": {"type": "enum", "values": ["injury", "leak", "fire"]},
            "severity": {"type": "enum", "values": ["minor", "major", "critical"]},
            "description": {"type": "str"},
        }),
    })
    errors = reg.validate_tool_call(
        ToolCall(tool="file_incident_report",
                 kwargs={"incident_type": "explosion", "severity": "major", "description": "test"})
    )
    assert len(errors) == 1
    assert "incident_type" in errors[0]
