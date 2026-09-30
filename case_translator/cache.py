"""
On-disk translation cache.

Why this exists
---------------
A medical PDF is re-translated from scratch on every run, even when nothing but
the output directory changed. During development that means paying for the same
hundreds of model calls over and over, and waiting for them again. The cache is
keyed on everything that can change the answer — model, system prompt and the
exact source text — so a hit is always semantically identical to a miss.

The cache is deliberately dumb: one small file per entry, no index, no
eviction. A run of a 40-block article produces ~50 tiny files, which is far
below the point where an index would pay for itself, and a deleted directory
simply means "translate again".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".translation-cache"


class TranslationCache:
    """Content-addressed store for translated strings."""

    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Translation cache disabled (cannot create %s): %s", self.cache_dir, exc)
                self.enabled = False

    @staticmethod
    def make_key(model: str, system_prompt: str, text: str, extra: str = "") -> str:
        payload = "\x00".join([model or "", system_prompt or "", extra, text or ""])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        path = self._path_for(key)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            value = payload.get("value")
            if isinstance(value, str):
                self.hits += 1
                return value
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Ignoring unreadable cache entry %s: %s", path, exc)
        self.misses += 1
        return None

    def set(self, key: str, value: str, model: str = "", source_text: str = "") -> None:
        if not self.enabled or value is None:
            return
        path = self._path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a crash mid-write cannot leave a half file
            # that a later run would read back as a corrupt translation.
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
            )
            try:
                json.dump(
                    {"model": model, "source": source_text, "value": value},
                    handle,
                    ensure_ascii=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            os.replace(handle.name, path)
        except OSError as exc:
            logger.debug("Could not write cache entry %s: %s", path, exc)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "enabled": self.enabled}
