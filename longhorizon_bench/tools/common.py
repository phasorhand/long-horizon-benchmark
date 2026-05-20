"""Common tools shared across all domains."""
from longhorizon_bench.tools.registry import ToolDef

COMMON_TOOLS: dict[str, ToolDef] = {
    "retrieve_policy": ToolDef(params={"query": {"type": "str"}}, note="查询背景文档中的相关条款"),
    "inspect_history": ToolDef(params={"event_id": {"type": "str"}}, note="查看指定历史事件的详情"),
    "request_clarification": ToolDef(params={"question": {"type": "str"}}, note="向上级/同事请求补充信息"),
    "no_action": ToolDef(params={"reason": {"type": "str"}}, note="判断当前事件无需采取行动"),
    "respond_to_query": ToolDef(params={"recipient": {"type": "str"}, "content": {"type": "str"}}),
}
