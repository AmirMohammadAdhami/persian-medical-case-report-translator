"""
OpenAI implementation of the Translator interface.
"""

import base64
import os
import time
from pathlib import Path
from typing import Optional

from .base import (
    CAPTION_TRANSLATOR_SYSTEM_PROMPT,
    MEDICAL_TRANSLATOR_SYSTEM_PROMPT,
    VISION_ANALYSIS_SYSTEM_PROMPT,
    Translator,
)


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAITranslator(Translator):
    """
    Translator powered by OpenAI models (e.g. gpt-4o, gpt-4o-mini).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3,
        enable_vision: bool = False,
        base_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please provide api_key or set OPENAI_API_KEY in your environment or .env file."
            )

        # Optional custom endpoint (proxy, Azure-compatible gateway, etc.).
        # Previously --base-url was silently ignored for this provider.
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").rstrip("/") or None
        self.model_name = model_name or DEFAULT_OPENAI_MODEL

        from openai import OpenAI
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        self.max_retries = max_retries
        self.enable_vision = enable_vision

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
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenAI API request failed after {self.max_retries} attempts: {e}") from e
                time.sleep(2 ** attempt)
        return ""

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
            return None
