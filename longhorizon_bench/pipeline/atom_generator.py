"""Stage 2b: Generate atomic events from topic packs via LLM."""

from __future__ import annotations

from typing import Any

import yaml

from longhorizon_bench.tools import COMMON_TOOLS, INDUSTRIAL_TOOLS


def _tool_list_text() -> str:
    lines: list[str] = []
    for name, tdef in {**COMMON_TOOLS, **INDUSTRIAL_TOOLS}.items():
        params_desc = ", ".join(
            f"{k}: {v.get('type', 'str')}" + (f" ({v.get('values', '')})" if v.get("values") else "")
            for k, v in tdef.params.items()
        )
        lines.append(f"- {name}({params_desc})")
    return "\n".join(lines)


def build_atom_prompt(
    topic_pack: dict[str, Any], n_atoms: int = 5
) -> tuple[str, str]:
    tool_text = _tool_list_text()

    system = f"""你是一名中国工业安全领域的资深专家，正在为benchmark构造原子事件。

可用工具列表（每个事件必须从中选择一个 expected_tool）：
{tool_text}

输出要求：
- 输出 YAML 列表，每个元素包含：atom_id, source_cluster, type, trigger, expected_tool, params, evidence, dimensions, is_critical
- params 中的 match 类型只能是：exact, contains, enum, keyword_coverage
- evidence 必须包含 required_facts（至少1条）和 forbidden_actions
- dimensions 从以下选择：domain_knowledge, multi_step_reasoning, long_term_memory, consistency, priority_judgment, information_integration
- 用 ```yaml ``` 包裹输出"""

    doc_texts = "\n\n".join(
        f"[{d.get('doc_id', '')}] {d.get('text', '')[:1000]}"
        for d in topic_pack.get("docs", [])
    )
    reg_texts = "\n\n".join(
        f"[{r.get('title', '')}] {r.get('text', '')[:500]}"
        for r in topic_pack.get("regulations", [])
    )

    user = f"""基于以下文档和法规，生成 {n_atoms} 个独立的原子事件：

## 文档素材
{doc_texts}

## 相关法规
{reg_texts}

请生成 {n_atoms} 个原子事件，覆盖不同的事件类型（巡检、事故、政策变更、整改等）。"""

    return system, user


def parse_atoms_from_response(response: str) -> list[dict[str, Any]]:
    start = response.find("```yaml")
    end = response.find("```", start + 7) if start >= 0 else -1
    if start >= 0 and end >= 0:
        yaml_text = response[start + 7:end].strip()
    else:
        yaml_text = response.strip()
    parsed = yaml.safe_load(yaml_text)
    if isinstance(parsed, list):
        return parsed
    return []


def generate_atoms_for_pack(
    llm_client: Any,
    topic_pack: dict[str, Any],
    n_atoms: int = 5,
) -> list[dict[str, Any]]:
    system, user = build_atom_prompt(topic_pack, n_atoms)
    response = llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=4096,
    )
    return parse_atoms_from_response(response)
