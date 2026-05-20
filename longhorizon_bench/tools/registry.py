"""Tool registry: stores definitions, validates agent actions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from longhorizon_bench.schema import ToolCall

@dataclass
class ToolDef:
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    note: str = ""

class ToolRegistry:
    def __init__(self, tools: dict[str, ToolDef | dict]) -> None:
        self._tools: dict[str, ToolDef] = {}
        for name, defn in tools.items():
            if isinstance(defn, dict):
                defn = ToolDef(**defn)
            self._tools[name] = defn

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def all_names(self) -> list[str]:
        return list(self._tools.keys())

    def validate_tool_call(self, call: ToolCall) -> list[str]:
        errors: list[str] = []
        defn = self._tools.get(call.tool)
        if defn is None:
            errors.append(f"Unknown tool: {call.tool}")
            return errors
        for param_name, param_spec in defn.params.items():
            if param_name not in call.kwargs:
                errors.append(f"Missing required param: {param_name}")
                continue
            value = call.kwargs[param_name]
            if param_spec.get("type") == "enum":
                allowed = param_spec.get("values", [])
                if value not in allowed:
                    errors.append(f"Invalid value for {param_name}: '{value}' not in {allowed}")
        return errors
