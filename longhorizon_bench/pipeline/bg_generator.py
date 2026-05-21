"""Stage 3a: Generate layered background documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_regulation_core(
    regulations: list[dict[str, Any]], max_chars: int = 5000
) -> str:
    parts: list[str] = []
    total = 0
    for reg in regulations:
        title = reg.get("title", "")
        text = reg.get("text", "")
        chunk = f"【{title}】\n{text}"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def generate_background(
    llm_client: Any,
    chain: dict[str, Any],
    regulation_core: str,
) -> str:
    events_desc = "\n".join(
        f"- {e.get('id', '')}: {e.get('trigger', '')}"
        for e in chain.get("events", [])
    )
    system = """你是一名工业安全领域的技术文档撰写专家。基于提供的法规核心内容，扩展生成一份完整的企业背景文档。

要求：
1. 保留法规中的条款编号和核心数据（检验周期、间距标准等）
2. 改写为虚构企业的背景材料（公司名、人名均为虚构）
3. 包括：公司概况、组织架构、设备台账、检验记录、历史事故、整改情况
4. 每个数据点都应当能被事件链中的 evidence 引用
5. 总字数控制在15000-20000字"""

    user = f"""场景角色：{chain.get('role', '')}
场景ID：{chain.get('scenario_id', '')}

## 法规核心内容（底层，必须保留的数据点）
{regulation_core}

## 事件链概览（背景文档需支撑这些事件）
{events_desc}

请生成完整的企业背景文档。"""

    return llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=16384,
    )


def save_background_docs(
    scenario_id: str, bg_text: str, data_dir: Path
) -> Path:
    doc_dir = data_dir / "background_docs" / scenario_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / "background.txt"
    path.write_text(bg_text, encoding="utf-8")
    return path
