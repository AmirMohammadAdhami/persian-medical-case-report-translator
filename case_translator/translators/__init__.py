"""
Translators package with factory method for selecting providers.
"""

import os
from typing import Optional

from .base import (
    NonRetryableTranslationError,
    RetryableTranslationError,
    TranslationError,
    Translator,
)
from .gemini_translator import DEFAULT_GEMINI_MODEL, GeminiTranslator
from .mock_translator import MockTranslator
from .openai_translator import DEFAULT_OPENAI_MODEL, OpenAITranslator
from .opencode_translator import (
    OpenCodeTranslator,
    DEFAULT_OPENCODE_BASE_URL,
    DEFAULT_OPENCODE_MODEL,
    ZEN_FREE_MODELS,
)


def list_opencode_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> list:
    """
    Lists the model IDs available on an OpenCode gateway (Zen or Go).
    Checks for an API key first so the failure mode is a clear message
    rather than a 401 from the remote endpoint.
    """
    resolved_key = api_key or os.getenv("OPENCODE_API_KEY")
    if not resolved_key:
        raise ValueError(
            "OPENCODE_API_KEY not found. Set it in your .env file or pass --api-key before listing models."
        )
    translator = OpenCodeTranslator(api_key=resolved_key, base_url=base_url)
    return translator.list_models()


def get_translator(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_vision: bool = False
) -> Translator:
    """
    Factory function to instantiate the appropriate translator.
    Supported providers: 'gemini', 'openai', 'opencode', 'mock'
    """
    provider_clean = (provider or "gemini").lower().strip()

    if provider_clean in ("opencode", "zen"):
        return OpenCodeTranslator(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            enable_vision=enable_vision
        )
    elif provider_clean == "gemini":
        resolved_model = model_name or DEFAULT_GEMINI_MODEL
        return GeminiTranslator(
            api_key=api_key,
            model_name=resolved_model,
            enable_vision=enable_vision,
            base_url=base_url
        )
    elif provider_clean == "openai":
        resolved_model = model_name or DEFAULT_OPENAI_MODEL
        return OpenAITranslator(
            api_key=api_key,
            model_name=resolved_model,
            enable_vision=enable_vision,
            base_url=base_url
        )
    elif provider_clean in ("mock", "offline", "test"):
        return MockTranslator(enable_vision=enable_vision)
    else:
        raise ValueError(
            f"Unsupported translation provider: '{provider}'. Supported providers are: 'opencode', 'gemini', 'openai', 'mock'."
        )


__all__ = [
    "Translator",
    "TranslationError",
    "NonRetryableTranslationError",
    "RetryableTranslationError",
    "GeminiTranslator",
    "OpenAITranslator",
    "OpenCodeTranslator",
    "MockTranslator",
    "get_translator",
    "list_opencode_models",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_OPENCODE_MODEL",
    "DEFAULT_OPENCODE_BASE_URL",
    "ZEN_FREE_MODELS",
]
