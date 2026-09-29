"""
Google Gemini implementation of the Translator interface using google-genai.
"""

import os
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from .base import (
    CAPTION_TRANSLATOR_SYSTEM_PROMPT,
    MEDICAL_TRANSLATOR_SYSTEM_PROMPT,
    VISION_ANALYSIS_SYSTEM_PROMPT,
    Translator,
)


# Default Gemini model. `gemini-2.5-flash` was retired for new API users
# (returns HTTP 404 NOT_FOUND), which used to break the default Gemini path.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class GeminiTranslator(Translator):
    """
    Translator powered by Google's Gemini models.
    Supports high-quality Persian medical translation and optional vision analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        max_retries: int = 3,
        enable_vision: bool = False,
        base_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please provide api_key or set GEMINI_API_KEY in your environment or .env file."
            )

        # Optional custom endpoint (e.g. a proxy or the OpenCode gateway).
        # Previously --base-url was silently ignored for this provider.
        self.base_url = (base_url or os.getenv("GEMINI_BASE_URL") or "").rstrip("/") or None

        from google import genai

        if self.base_url:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options={"base_url": self.base_url},
            )
        else:
            self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.enable_vision = enable_vision

    @staticmethod
    def _generation_config(system_instruction: str) -> dict:
        """
        Builds the generation config.

        We never pass Python callables/tools to the SDK, so Automatic Function
        Calling (AFC) is irrelevant here. Disabling it explicitly silences the
        SDK's 'Direct use of automatic function calling (AFC) is not
        recommended' warning, which otherwise appears on every translation call.
        """
        return {
            "system_instruction": system_instruction,
            "temperature": 0.2,
            "automatic_function_calling": {"disable": True},
        }

    def _generate_with_retry(self, prompt: str, system_instruction: str) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._generation_config(system_instruction)
                )
                text = response.text or ""
                return text.strip()
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Gemini API request failed after {self.max_retries} attempts: {e}") from e
                time.sleep(2 ** attempt)
        return ""

    def translate_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        return self._generate_with_retry(
            prompt=text,
            system_instruction=MEDICAL_TRANSLATOR_SYSTEM_PROMPT
        )

    def translate_caption(self, caption: str) -> str:
        if not caption or not caption.strip():
            return ""
        return self._generate_with_retry(
            prompt=caption,
            system_instruction=CAPTION_TRANSLATOR_SYSTEM_PROMPT
        )

    def describe_image(self, image_path: str) -> Optional[str]:
        if not self.enable_vision:
            return None

        path = Path(image_path)
        if not path.exists():
            return None

        for attempt in range(1, self.max_retries + 1):
            try:
                img = Image.open(path)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[img, "Describe the key medical / radiographic / clinical findings in this image in Persian."],
                    config=self._generation_config(VISION_ANALYSIS_SYSTEM_PROMPT)
                )
                desc = (response.text or "").strip()
                return desc if desc else None
            except Exception:
                if attempt == self.max_retries:
                    return None
                time.sleep(2 ** attempt)

        return None
