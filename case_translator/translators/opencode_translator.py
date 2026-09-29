"""
OpenCode (Zen / Go) implementation of the Translator interface.
Connects to OpenCode's OpenAI-compatible API gateway, allowing usage of
various models (DeepSeek, GLM, Kimi, MiMo, Qwen, etc.).

Two distinct gateways exist and they do NOT share a model catalogue:

  * Zen (pay-as-you-go)   -> https://opencode.ai/zen/v1      (default here)
  * Go  (subscription)    -> https://opencode.ai/zen/go/v1

A model valid on one gateway may be rejected with HTTP 400 / 'inference_failed'
on the other (e.g. the free '*--free' IDs are Zen-only). Always pair the base
URL with a model ID from that gateway's own /models listing.
"""

import base64
import os
import time
from pathlib import Path
from typing import List, Optional

from .base import (
    CAPTION_TRANSLATOR_SYSTEM_PROMPT,
    MEDICAL_TRANSLATOR_SYSTEM_PROMPT,
    VISION_ANALYSIS_SYSTEM_PROMPT,
    Translator,
)


# Zen is the pay-as-you-go gateway and the one most keys are provisioned for.
DEFAULT_OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"

# NOTE: model IDs rotate frequently. This default is a long-lived, non-free
# Zen chat model. Verify with `python main.py --list-models` if it ever 404s.
DEFAULT_OPENCODE_MODEL = "deepseek-v4.1-flash"

# Model IDs that are free of charge on the Zen gateway (no billing required).
# These are Zen-only and will fail with 400 on the Go gateway.
ZEN_FREE_MODELS = [
    "deepseek-v4-flash-free",
    "mimo-v2.6-flash-free",
    "mimo-v2.5-free",
    "longcat-2.5-preview-free",
    "ling-3.0-flash-fin-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "space-bunny-free",
    "jev-1.13-free",
]


class OpenCodeTranslator(Translator):
    """
    Translator powered by OpenCode models (such as DeepSeek, Qwen, etc.)
    via OpenCode's OpenAI-compatible API gateway.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        enable_vision: bool = False
    ):
        self.api_key = api_key or os.getenv("OPENCODE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENCODE_API_KEY not found. Please provide --api-key or set OPENCODE_API_KEY in your environment or .env file."
            )

        self.base_url = (
            base_url
            or os.getenv("OPENCODE_BASE_URL")
            or DEFAULT_OPENCODE_BASE_URL
        ).rstrip("/")

        self.model_name = (
            model_name
            or os.getenv("OPENCODE_MODEL")
            or DEFAULT_OPENCODE_MODEL
        )

        self.max_retries = max_retries
        self.enable_vision = enable_vision

        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _chat_completion(self, system_prompt: str, user_content: any) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                )
                choice = response.choices[0]
                text = choice.message.content or ""
                return text.strip()
            except Exception as e:
                # A model-level rejection (bad model ID for this gateway) will
                # never succeed on retry, so fail fast with an actionable hint.
                if self._is_model_rejection(e):
                    raise RuntimeError(self._model_hint(e)) from e

                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"OpenCode API request failed on model '{self.model_name}' (endpoint: {self.base_url}) after {self.max_retries} attempts: {e}"
                    ) from e
                time.sleep(2 ** attempt)
        return ""

    @staticmethod
    def _is_model_rejection(error: Exception) -> bool:
        """Detects a flat '400 model not accepted by this gateway' rejection."""
        status = getattr(error, "status_code", None)
        if status != 400:
            return False
        text = str(error).lower()
        return (
            "inference_failed" in text
            or "model" in text
            or "not found" in text
        )

    def _model_hint(self, error: Exception) -> str:
        """Builds an actionable error message for a model rejection."""
        lines = [
            f"OpenCode rejected the model ID '{self.model_name}' on endpoint '{self.base_url}' (HTTP 400).",
            "",
            "This usually means the model ID does not exist on THAT gateway.",
            "Zen (opencode.ai/zen/v1) and Go (opencode.ai/zen/go/v1) have separate catalogues.",
            "Free '*--free' models exist on Zen only.",
            "",
            f"Run this to list valid IDs for your key:\n  python main.py --list-models --base-url {self.base_url}",
        ]
        if self.model_name.endswith("-free") and "/go/" in self.base_url:
            lines.insert(
                4,
                "Your model ID ends in '-free' but your endpoint is the Go subscription "
                "gateway. Either switch to https://opencode.ai/zen/v1 or drop the '-free' suffix.",
            )
        lines.append("")
        lines.append(f"Original error: {error}")
        return "\n".join(lines)

    def list_models(self, timeout: int = 25) -> List[str]:
        """
        Fetches the model IDs available on the configured gateway.
        Useful because model IDs rotate and the two gateways differ.
        """
        import requests

        url = f"{self.base_url}/models"
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("data") or payload.get("models") or []
        return sorted(
            entry.get("id") for entry in entries if entry.get("id")
        )

    def translate_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        return self._chat_completion(MEDICAL_TRANSLATOR_SYSTEM_PROMPT, text)

    def translate_caption(self, caption: str) -> str:
        if not caption or not caption.strip():
            return ""
        return self._chat_completion(CAPTION_TRANSLATOR_SYSTEM_PROMPT, caption)

    def describe_image(self, image_path: str) -> Optional[str]:
        if not self.enable_vision:
            return None

        path = Path(image_path)
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            ext = path.suffix.lower().replace(".", "")
            if ext == "jpg":
                ext = "jpeg"
            mime = f"image/{ext}" if ext in ("png", "jpeg", "webp", "gif") else "image/png"

            user_content = [
                {"type": "text", "text": "Describe the key medical / radiographic / clinical findings in this image in Persian."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
            ]
            return self._chat_completion(VISION_ANALYSIS_SYSTEM_PROMPT, user_content)
        except Exception:
            # If the selected model is text-only or vision fails, return None gracefully
            return None
