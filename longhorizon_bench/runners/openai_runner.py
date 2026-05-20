# longhorizon_bench/runners/openai_runner.py
"""OpenAI-compatible runner for LongHorizon-Bench."""

from __future__ import annotations

from typing import Any

from longhorizon_bench.runners.base import BaseRunner
from longhorizon_bench.schema import AgentAction, CheckpointResponse, ToolCall


class OpenAIRunner(BaseRunner):
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: "
                    "pip install longhorizon-bench[openai]"
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def act(self, observation: dict[str, Any]) -> AgentAction | CheckpointResponse:
        event = observation["current_event"]

        if event["node_type"] == "checkpoint":
            return self._handle_checkpoint(observation)

        return self._handle_action(observation)

    def _handle_action(self, observation: dict[str, Any]) -> AgentAction:
        client = self._get_client()
        messages = self._build_messages(observation)

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            tools=self._build_tool_defs(),
            tool_choice="required",
        )

        tool_calls: list[ToolCall] = []
        for tc in response.choices[0].message.tool_calls or []:
            import json
            tool_calls.append(ToolCall(
                tool=tc.function.name,
                kwargs=json.loads(tc.function.arguments),
            ))

        return AgentAction(tool_calls=tool_calls) if tool_calls else AgentAction(
            tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "no tool calls returned"})]
        )

    def _handle_checkpoint(self, observation: dict[str, Any]) -> CheckpointResponse:
        client = self._get_client()
        queries = observation["current_event"].get("queries", [])
        answers: dict[str, str] = {}

        for query in queries:
            messages = self._build_messages(observation)
            messages.append({"role": "user", "content": query})

            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
            answers[query] = response.choices[0].message.content or ""

        return CheckpointResponse(answers=answers)

    def _build_messages(self, observation: dict[str, Any]) -> list[dict]:
        system = f"你是一名{observation['role']}。根据背景文档和历史事件，对当前事件做出决策。"

        bg = observation.get("background_docs") or observation.get("background_summary", "")
        if isinstance(bg, dict):
            bg = "\n\n".join(bg.values())

        history_text = ""
        for h in observation.get("history", []):
            if h.get("summary_only"):
                history_text += f"\n[{h['event_id']}] {h['type']}（摘要）"
            else:
                history_text += f"\n[{h['event_id']}] {h.get('type', '')}: {h.get('input', '')}"

        event = observation["current_event"]
        current = f"当前事件 [{event['event_id']}]: {event.get('input', '')}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"背景文档:\n{bg}\n\n历史事件:{history_text}\n\n{current}"},
        ]

    def _build_tool_defs(self) -> list[dict]:
        from longhorizon_bench.tools import COMMON_TOOLS, INDUSTRIAL_TOOLS

        defs: list[dict] = []
        for name, tool_def in {**COMMON_TOOLS, **INDUSTRIAL_TOOLS}.items():
            properties = {}
            for param_name, param_spec in tool_def.params.items():
                ptype = param_spec.get("type", "str")
                if ptype == "enum":
                    properties[param_name] = {
                        "type": "string",
                        "enum": param_spec.get("values", []),
                    }
                elif ptype == "bool":
                    properties[param_name] = {"type": "boolean"}
                elif ptype == "float":
                    properties[param_name] = {"type": "number"}
                elif ptype.startswith("list"):
                    properties[param_name] = {"type": "array", "items": {"type": "string"}}
                else:
                    properties[param_name] = {"type": "string"}

            defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties.keys()),
                    },
                },
            })
        return defs
