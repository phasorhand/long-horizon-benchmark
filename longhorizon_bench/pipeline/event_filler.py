"""Stage 3b: Fill event details using LLM."""

from __future__ import annotations

from typing import Any

import yaml


def fill_action_event(
    llm_client: Any,
    atom: dict[str, Any],
    background_text: str,
    prior_inputs: list[dict[str, Any]],
) -> str:
    param_hints = []
    for name, spec in atom.get("params", {}).items():
        if spec.get("keywords"):
            param_hints.append(f"必须包含关键词: {spec['keywords']}")
        elif spec.get("value"):
            param_hints.append(f"必须提及: {spec['value']}")

    prior_text = "\n".join(
        f"[{p.get('event_id', '')}] {p.get('input', '')[:200]}"
        for p in prior_inputs[-5:]
    )

    system = """你是一名工业安全事件描述撰写专家。生成一段200-500字的事件描述。

要求：
1. 引用背景文档中的具体条款或数据
2. 包含后续评分所需的关键信息
3. 语言风格为正式工作通知/报告
4. 只输出事件描述文本，不要标题或格式标记"""

    user = f"""事件类型: {atom.get('type', '')}
触发: {atom.get('trigger', '')}

背景文档（节选）:
{background_text[:3000]}

前序事件:
{prior_text}

评分关键信息（必须包含在描述中）:
{chr(10).join(param_hints)}

请生成事件描述文本。"""

    return llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=1024,
    )


def fill_checkpoint_queries(
    llm_client: Any,
    prior_events: list[dict[str, Any]],
    target_dimensions: list[str] | None = None,
) -> list[dict[str, Any]]:
    dims = target_dimensions or ["long_term_memory"]
    events_text = "\n".join(
        f"[{e.get('event_id', '')}] {e.get('input', '')[:200]}"
        for e in prior_events
    )

    system = """你是一名benchmark检查点设计专家。设计回溯问题来测试模型的长期记忆和一致性。

输出 YAML 列表，用 ```yaml ``` 包裹。每个问题包含:
- query: 问题文本
- expected_keywords: 正确答案应包含的关键词列表
- dimension: 评测维度
- match: keyword_coverage"""

    user = f"""基于以下前序事件，设计 {len(dims)} 个检查点问题。

前序事件:
{events_text}

要求覆盖的维度: {dims}
问题的答案必须依赖3个以上事件之前的信息。"""

    response = llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=2048,
    )

    start = response.find("```yaml")
    end = response.find("```", start + 7) if start >= 0 else -1
    if start >= 0 and end >= 0:
        yaml_text = response[start + 7:end].strip()
    else:
        yaml_text = response.strip()

    parsed = yaml.safe_load(yaml_text)
    return parsed if isinstance(parsed, list) else []
