"""
Google Gemini implementation of the Translator interface using google-genai.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from .base import (
    CAPTION_TRANSLATOR_SYSTEM_PROMPT,
    MEDICAL_TRANSLATOR_SYSTEM_PROMPT,
    VISION_ANALYSIS_SYSTEM_PROMPT,
    NonRetryableTranslationError,
    RetryableTranslationError,
    Translator,
    classify_translation_error,
    retry_delay,
)

logger = logging.getLogger(__name__)


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
        # Base class owns the glossary cache and stats; see Translator.__init__.
        super().__init__()

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

    def _generate_with_retry(self, prompt, system_instruction: str) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            self.stats["requests"] += 1
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._generation_config(system_instruction)
                )
                text = response.text or ""
                return text.strip()
            except Exception as e:
                classified = classify_translation_error(e)
                last_error = classified
                if isinstance(classified, NonRetryableTranslationError):
                    self.stats["failures"] += 1
                    raise NonRetryableTranslationError(
                        f"Gemini request failed (non-retryable) on model "
                        f"'{self.model_name}': {e}"
                    ) from e
                if attempt == self.max_retries:
                    break
                self.stats["retries"] += 1
                delay = retry_delay(attempt)
                logger.warning(
                    "Gemini request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, self.max_retries, delay, e,
                )
                time.sleep(delay)

        self.stats["failures"] += 1
        raise RetryableTranslationError(
            f"Gemini API request failed after {self.max_retries} attempts: {last_error}"
        )

    # -- Translator interface ---------------------------------------------

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

    def _translate_with_system(self, system_prompt: str, text: str) -> str:
        if not text or not text.strip():
            return ""
        return self._generate_with_retry(prompt=text, system_instruction=system_prompt)

    def describe_image(self, image_path: str) -> Optional[str]:
        if not self.enable_vision:
            return None

        path = Path(image_path)
        if not path.exists():
            logger.warning("Vision analysis skipped: image not found at %s", image_path)
            return None

        last_error: Optional[Exception] = None
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
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(retry_delay(attempt))

        logger.warning("Vision analysis failed for %s: %s", image_path, last_error)
        return None
