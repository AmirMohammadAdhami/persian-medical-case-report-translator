"""
Unit tests for OpenCodeTranslator.
"""

from unittest.mock import MagicMock, patch
import pytest

from case_translator.translators import get_translator
from case_translator.translators.opencode_translator import (
    DEFAULT_OPENCODE_BASE_URL,
    DEFAULT_OPENCODE_MODEL,
    OpenCodeTranslator,
)


def test_opencode_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENCODE_API_KEY not found"):
        OpenCodeTranslator(api_key=None)


def test_opencode_initialization_defaults():
    translator = OpenCodeTranslator(api_key="sk-test-opencode-key")
    assert translator.api_key == "sk-test-opencode-key"
    assert translator.base_url == DEFAULT_OPENCODE_BASE_URL
    assert translator.model_name == DEFAULT_OPENCODE_MODEL


def test_opencode_custom_model_and_base_url():
    custom_url = "https://custom.opencode.ai/v1"
    custom_model = "deepseek-reasoner"
    translator = OpenCodeTranslator(
        api_key="sk-test-key",
        model_name=custom_model,
        base_url=custom_url
    )
    assert translator.base_url == custom_url
    assert translator.model_name == custom_model


def test_opencode_factory():
    translator = get_translator(
        provider="opencode",
        api_key="sk-test-key",
        model_name="qwen/qwen-2.5-72b-instruct",
        base_url="https://opencode.ai/zen/go/v1"
    )
    assert isinstance(translator, OpenCodeTranslator)
    assert translator.model_name == "qwen/qwen-2.5-72b-instruct"


def test_opencode_translation_mocked():
    translator = OpenCodeTranslator(
        api_key="sk-test-key",
        model_name="deepseek-chat"
    )

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "ترجمه فارسی دقیق مورد بالینی دندانپزشکی"
    mock_response.choices = [mock_choice]

    with patch.object(translator.client.chat.completions, "create", return_value=mock_response) as mock_create:
        result = translator.translate_text("Clinical dental case report.")
        assert result == "ترجمه فارسی دقیق مورد بالینی دندانپزشکی"
        assert mock_create.called
        kwargs = mock_create.call_args[1]
        assert kwargs["model"] == "deepseek-chat"
