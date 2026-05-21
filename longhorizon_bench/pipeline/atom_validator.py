"""Stage 2c: Validate atomic events against ToolRegistry and schema rules."""

from __future__ import annotations

from typing import Any

from longhorizon_bench.tools import build_registry

VALID_MATCH_TYPES = {"exact", "contains", "enum", "keyword_coverage"}

_registry = build_registry("industrial")


def validate_atom(atom: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tool_name = atom.get("expected_tool", "")
    if tool_name not in _registry:
        errors.append(f"Tool '{tool_name}' not in ToolRegistry")
    else:
        tool_def = _registry.get(tool_name)
        if tool_def:
            for param_key in atom.get("params", {}):
                if param_key not in tool_def.params:
                    errors.append(f"Param '{param_key}' not defined for tool '{tool_name}'")

    for param_key, param_spec in atom.get("params", {}).items():
        match_type = param_spec.get("match", "")
        if match_type not in VALID_MATCH_TYPES:
            errors.append(f"Invalid match type '{match_type}' for param '{param_key}'. Valid: {VALID_MATCH_TYPES}")

    evidence = atom.get("evidence", {})
    required_facts = evidence.get("required_facts", [])
    if not required_facts:
        errors.append("evidence.required_facts must have at least 1 entry")

    return errors


def validate_batch(
    atoms: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for atom in atoms:
        errors = validate_atom(atom)
        if errors:
            atom["_validation_errors"] = errors
            failed.append(atom)
        else:
            passed.append(atom)
    return passed, failed
