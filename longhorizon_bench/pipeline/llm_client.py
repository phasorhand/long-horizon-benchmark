"""Unified LLM client supporting Claude and DeepSeek."""

from __future__ import annotations

from typing import Any


_DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "deepseek": "deepseek-chat",
}

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class LLMClient:
    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        if provider not in _DEFAULT_MODELS:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'deepseek'.")
        self.provider = provider
        self.model = model or _DEFAULT_MODELS[provider]
        self.temperature = temperature
        self._api_key = api_key
        self._claude_client: Any = None
        self._openai_client: Any = None

        if provider == "claude":
            self._init_claude()
        else:
            self._init_deepseek()

    def _init_claude(self) -> None:
        from anthropic import Anthropic
        self._claude_client = Anthropic(api_key=self._api_key)

    def _init_deepseek(self) -> None:
        from openai import OpenAI
        self._openai_client = OpenAI(
            api_key=self._api_key,
            base_url=_DEEPSEEK_BASE_URL,
        )

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        if self.provider == "claude":
            return self._chat_claude(system, messages, max_tokens)
        return self._chat_deepseek(system, messages, max_tokens)

    def _chat_claude(
        self, system: str, messages: list[dict[str, str]], max_tokens: int
    ) -> str:
        response = self._claude_client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.content[0].text

    def _chat_deepseek(
        self, system: str, messages: list[dict[str, str]], max_tokens: int
    ) -> str:
        all_messages = [{"role": "system", "content": system}] + messages
        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
