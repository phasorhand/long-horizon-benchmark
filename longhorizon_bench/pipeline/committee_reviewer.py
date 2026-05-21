"""Stage 4 Layer 2: LLM committee review with dual-model voting."""

from __future__ import annotations

import json
from typing import Any

REVIEW_DIMENSIONS = ["因果连贯性", "证据可追溯", "难度梯度", "答案区分度", "专业准确性"]


def build_review_prompt(scenario_dict: dict[str, Any]) -> tuple[str, str]:
    system = """你是一名benchmark质量评审专家。评审一个工业安全事件链场景的质量。

对以下5个维度各打1-5分：
1. 因果连贯性：事件间 depends_on 关系是否合理
2. 证据可追溯：required_facts 是否真的能从背景文档找到
3. 难度梯度：事件链是否从简单到复杂递进
4. 答案区分度：scoring_rule 是否能区分好坏agent
5. 专业准确性：法规引用、行业术语是否正确

用 ```json ``` 输出，格式如 {"因果连贯性": 4, ...}"""

    events_summary = "\n".join(
        f"- {e.get('event_id', '')}: {e.get('type', '')} - {str(e.get('input', ''))[:100]}"
        for e in scenario_dict.get("events", [])[:20]
    )

    user = f"""场景ID: {scenario_dict.get('scenario_id', '')}
角色: {scenario_dict.get('role', '')}
事件数: {scenario_dict.get('total_events', 0)} actions + {scenario_dict.get('total_checkpoints', 0)} checkpoints

事件链:
{events_summary}

请评审并打分。"""

    return system, user


def parse_review_scores(response: str) -> dict[str, int]:
    start = response.find("```json")
    end = response.find("```", start + 7) if start >= 0 else -1
    if start >= 0 and end >= 0:
        json_text = response[start + 7:end].strip()
    else:
        json_text = response.strip()
    return json.loads(json_text)


def check_committee_agreement(
    scores_a: dict[str, int], scores_b: dict[str, int], max_diff: int = 1
) -> tuple[list[str], list[str]]:
    agreed: list[str] = []
    disagreed: list[str] = []
    for dim in REVIEW_DIMENSIONS:
        sa = scores_a.get(dim, 0)
        sb = scores_b.get(dim, 0)
        if abs(sa - sb) <= max_diff:
            agreed.append(dim)
        else:
            disagreed.append(dim)
    return agreed, disagreed


def compute_verdict(
    scores_a: dict[str, int],
    scores_b: dict[str, int],
    min_avg: float = 3.5,
) -> str:
    _, disagreed = check_committee_agreement(scores_a, scores_b)
    if disagreed:
        return "NEEDS_HUMAN_REVIEW"

    for dim in REVIEW_DIMENSIONS:
        avg = (scores_a.get(dim, 0) + scores_b.get(dim, 0)) / 2
        if avg < min_avg:
            return "FAIL"

    return "PASS"


def review_scenario(
    claude_client: Any,
    deepseek_client: Any,
    scenario_dict: dict[str, Any],
) -> dict[str, Any]:
    system, user = build_review_prompt(scenario_dict)
    msgs = [{"role": "user", "content": user}]

    response_a = claude_client.chat(system=system, messages=msgs)
    response_b = deepseek_client.chat(system=system, messages=msgs)

    scores_a = parse_review_scores(response_a)
    scores_b = parse_review_scores(response_b)
    verdict = compute_verdict(scores_a, scores_b)

    return {
        "claude": scores_a,
        "deepseek": scores_b,
        "verdict": verdict,
    }
