"""Tool definitions and registry."""
from longhorizon_bench.tools.registry import ToolDef, ToolRegistry
from longhorizon_bench.tools.common import COMMON_TOOLS
from longhorizon_bench.tools.industrial import INDUSTRIAL_TOOLS

def build_registry(domain: str) -> ToolRegistry:
    domain_tools: dict[str, ToolDef] = {"industrial": INDUSTRIAL_TOOLS}
    tools = {**COMMON_TOOLS, **(domain_tools.get(domain, {}))}
    return ToolRegistry(tools)
