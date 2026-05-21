import pytest
from unittest.mock import patch, MagicMock


def test_llm_client_claude_provider():
    from longhorizon_bench.pipeline.llm_client import LLMClient
    client = LLMClient(provider="claude", api_key="test-key")
    assert client.provider == "claude"
    assert client.model == "claude-sonnet-4-20250514"


def test_llm_client_deepseek_provider():
    from longhorizon_bench.pipeline.llm_client import LLMClient
    client = LLMClient(provider="deepseek", api_key="test-key")
    assert client.provider == "deepseek"
    assert client.model == "deepseek-chat"


def test_llm_client_custom_model():
    from longhorizon_bench.pipeline.llm_client import LLMClient
    client = LLMClient(provider="claude", api_key="test-key", model="claude-opus-4-20250514")
    assert client.model == "claude-opus-4-20250514"


def test_llm_client_invalid_provider():
    from longhorizon_bench.pipeline.llm_client import LLMClient
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMClient(provider="invalid", api_key="test")


def test_llm_client_chat_claude():
    from longhorizon_bench.pipeline.llm_client import LLMClient
    client = LLMClient(provider="claude", api_key="test-key")
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="response text")]
    with patch.object(client, "_claude_client") as mock_client:
        mock_client.messages.create.return_value = mock_response
        result = client.chat(
            system="You are a helper.",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert result == "response text"


def test_llm_client_chat_deepseek():
    from longhorizon_bench.pipeline.llm_client import LLMClient
    client = LLMClient(provider="deepseek", api_key="test-key")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="ds response"))]
    with patch.object(client, "_openai_client") as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        result = client.chat(
            system="You are a helper.",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert result == "ds response"
