"""
Base translator abstraction and medical translation guidelines.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


MEDICAL_TRANSLATOR_SYSTEM_PROMPT = """You are a professional medical translator specializing in clinical case reports and dentistry/medicine.

Translate the provided English medical text into fluent, natural, professional Persian.

Do not translate word-for-word. Preserve the exact meaning, medical accuracy, numbers, units, drug names, anatomical terms, citations, abbreviations, and scientific relationships.

Do not summarize or omit information.

ANATOMICAL NAMES — NEVER TRANSLATE:
Dental and oral anatomical names, and the tooth types they describe, MUST stay in the
original English, untouched. This includes tooth type names (incisor, lateral incisor,
central incisor, canine, premolar, molar, wisdom tooth) and positional qualifiers
(maxillary, mandibular, mesial, distal, buccal, lingual, palatal, occlusal, apical,
coronal, interproximal, right, left). Examples that must NOT be translated:
"maxillary right lateral incisor", "mandibular left first molar", "maxillary central incisor".
Writing a Persian translation of these risks an anatomical error (lateral vs central,
maxillary vs mandibular), so the English form is always kept as-is. Do not add a Persian
equivalent for them, not even in parentheses.

SPECIALIZED CLINICAL TERMS — PERSIAN FIRST, ENGLISH IN PARENTHESES:
For advanced clinical/procedural terms, write the fluent Persian equivalent FIRST and then
the English term in parentheses: [Persian] ([English]).
Examples of the required pattern:
  "شکستگی پیچیده تاج-ریشه (Complicated Crown-Root Fracture)"
  "روش اتصال مجدد (Reattachment Procedure)"
  "پست فایبر (Fiber Post)"
  "عرض بیولوژیک (Biologic Width)"
  "اکستروژن ارتودنتیک (Orthodontic Extrusion)"
  "ایزولاسیون (Isolation)"
  "الگوی رویش پروتروزیو (Protrusive Eruptive Pattern)"
Introduce each such term this way on its first meaningful occurrence; do not repeat the
parenthetical every time it reappears. Use the exact English spelling from the source.

Common medical words (such as بیمار, درمان, جراحی, دندان, پزشک) do NOT need English equivalents.

The final Persian should be easy to read and should resemble professionally edited Persian medical literature rather than literal machine translation.

Never invent information that is not present in the source.
If the provided text is only a document section heading or label (for example "Conclusion",
"References", "Discussion", "Case Report"), translate that heading alone. Never reply with a
request for more text, and never ask the user a question.
Only return the Persian translation, without any conversational preamble or meta-commentary."""


GLOSSARY_SYSTEM_PROMPT = """You are a bilingual dental terminology lexicographer building a
Persian glossary for a clinical document.

You will receive a numbered list of English dental/medical terms. Translate EVERY term into
concise, professional Persian (1-3 words) as a practising dentist would write it.

Rules:
- Return exactly one line per input term, in the same order, as `<number>. <Persian>`.
- Persian only. Do not add explanations, parentheses, transliterations, or the English term.
- Anatomical names that are conventionally written in English (incisor, molar, maxillary,
  mandibular, mesial, distal, buccal, lingual, palatal, occlusal, apical, coronal) must be
  echoed back EXACTLY as given, unchanged.
- Never ask a question and never reply with anything other than the numbered list."""


REFERENCE_TRANSLATOR_SYSTEM_PROMPT = """You are an academic bibliographic editor preparing the
reference list of a Persian medical case report.

Journal names, DOIs, URLs and volume/issue/page structures MUST remain in the original English
and must not be translated or transliterated. Keep them byte-for-byte as published.

Translate ONLY the parts of an entry that are descriptive prose in the source — typically the
title of a non-English work or a descriptive phrase that the source itself already gives in a
translatable form. Author surnames and initials, journal names, publisher names, and years stay
as written.

If an entry is entirely English bibliography (author list, article title, journal, year, volume,
pages, DOI), return it completely unchanged.

Do not add, remove, or reorder any part of the entry. Do not add explanations or parentheses.
Return only the reference entry."""


CAPTION_TRANSLATOR_SYSTEM_PROMPT = """You are a professional medical translator specializing in clinical case reports and dentistry/medicine.

Translate the provided figure caption into fluent, natural, professional Persian.
Preserve figure numbering accurately (e.g., 'Figure 1.' becomes 'شکل ۱.' or 'شکل 1.').
For specialized medical, radiographic, or anatomical terms, provide the English original in parentheses immediately after the Persian term (e.g., 'تصویر داخل‌دهانی (intraoral photograph)').
Anatomical tooth names (incisor, premolar, molar, maxillary, mandibular, ...) must stay in the original English and must NOT be translated.
Do not omit details or summarize.
Only return the Persian translated caption."""


VISION_ANALYSIS_SYSTEM_PROMPT = """You are an expert medical imaging and clinical diagnostic consultant.
Analyze the provided medical case report image (e.g. radiograph, CT/MRI, clinical photograph, or histopathology image).
Provide a concise, highly accurate description of the visual findings in professional Persian (2-3 sentences max).
Do NOT hallucinate findings. If the image does not provide enough information for a reliable clinical description, state that clearly and do not invent details.
Include specialized English terminology in parentheses where appropriate (e.g. شکستگی تاج-ریشه (crown-root fracture)).
Only return the Persian visual description."""


# Phrases a chat model may emit when it mistakes a bare heading for a prompt.
# Kept deliberately narrow to avoid discarding legitimate translations, and only
# applied to SHORT blocks (see is_non_translation_response) so a real paragraph
# containing such a sentence is never dropped.
NON_TRANSLATION_MARKERS = (
    "please provide",
    "please send",
    "send the",
    "provide the text",
    "text you'd like",
    "text you would like",
    "i'd be happy to translate",
    "ارسال کنید تا ترجمه",
    "متن مورد نظر را ارسال",
    "لطفا متن",
    "لطفاً متن",
    "متنی برای ترجمه",
)

# Maximum character length for a model reply to be treated as non-translation.
NON_TRANSLATION_MAX_CHARS = 160


def is_non_translation_response(source_text: str, translation: str) -> bool:
    """
    Detects a model reply that is a conversational response rather than a
    translation.

    This happens when a block is a very short section label (e.g. the heading
    "Conclusion") and the model treats it as an instruction, replying with
    "Please provide the English text you'd like translated."

    The check is intentionally conservative: a reply is only rejected when the
    SOURCE was a short single-line block AND the reply is short AND contains an
    explicit request-for-input marker.
    """
    if not translation:
        return False

    source = (source_text or "").strip()
    reply = translation.strip()

    # Real headings are short and carry no sentence punctuation. If the source
    # is a full sentence or a multi-line paragraph, always trust the reply.
    if len(source) > 60 or source.count(".") > 0:
        return False

    if len(reply) > NON_TRANSLATION_MAX_CHARS:
        return False

    lowered = reply.lower()
    return any(marker in lowered for marker in NON_TRANSLATION_MARKERS)


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------
#
# The pipeline needs to tell "this request can never succeed" apart from
# "the network hiccuped". Retrying an authentication failure three times with
# exponential backoff turns a 1-second failure into a 6-second one, and with
# dozens of blocks that is the difference between a fast error and a hang.


class TranslationError(RuntimeError):
    """Base class for provider translation failures."""


class NonRetryableTranslationError(TranslationError):
    """A failure that will not succeed on retry (auth, bad model, connection refused)."""


class RetryableTranslationError(TranslationError):
    """A transient failure worth retrying (timeout, 5xx, rate limit)."""


# HTTP status codes that mean "stop asking".
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 422})

# Substrings that identify a permanent failure when no status code is exposed
# (the SDKs wrap transport errors inconsistently).
_NON_RETRYABLE_MARKERS = (
    "connection refused",
    "connection error",
    "apiconnectionerror",
    "invalid api key",
    "incorrect api key",
    "unauthorized",
    "authentication",
    "permission denied",
    "model not found",
    "does not exist",
    "no such model",
    "unsupported model",
    "name resolution",
    "getaddrinfo",
    "nodename nor servname",
    "no address associated",
    "ssl",
    "certificate verify failed",
    "inference_failed",
)

# Explicitly transient: 429 and 5xx.
_RETRYABLE_MARKERS = (
    "rate limit",
    "too many requests",
    "overloaded",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)


def classify_translation_error(error: Exception) -> TranslationError:
    """
    Wraps an arbitrary provider exception in the retryable / non-retryable taxonomy.

    Order matters: an explicit transient marker wins over the status-code check,
    because some SDKs surface a 400-wrapped "rate limit" and some surface 429 as
    a generic APIError. Non-retryable markers are only consulted after the
    transient ones, so a "connection error" that mentions "timeout" stays
    retryable.
    """
    if isinstance(error, TranslationError):
        return error

    text = f"{type(error).__name__}: {error}".lower()

    if any(marker in text for marker in _RETRYABLE_MARKERS):
        return RetryableTranslationError(str(error))

    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if status is not None:
        try:
            status_int = int(status)
        except (TypeError, ValueError):
            status_int = None
        if status_int is not None:
            if status_int == 429 or status_int >= 500:
                return RetryableTranslationError(str(error))
            if status_int in _NON_RETRYABLE_STATUS:
                return NonRetryableTranslationError(str(error))

    if any(marker in text for marker in _NON_RETRYABLE_MARKERS):
        return NonRetryableTranslationError(str(error))

    # Unknown failures are treated as retryable: the common case is a flaky
    # network, and a bounded retry budget caps the worst case.
    return RetryableTranslationError(str(error))


def retry_delay(attempt: int, base: float = 1.5, cap: float = 12.0) -> float:
    """Exponential backoff, capped so a slow provider cannot stall a run."""
    return min(cap, base * (2 ** (attempt - 1)))


class Translator(ABC):
    """Abstract base class for all AI translation providers."""

    #: Providers that bill per call can lower this.
    max_retries: int = 3

    def __init__(self) -> None:
        # Persian rendering of glossary keys, learned once per run.
        #
        # A key like "incisor" can appear in a dozen blocks, and the model may
        # render it differently each time. Translating it once and reusing that
        # keeps terminology consistent AND makes repeated terms cheap.
        self._term_translations: dict = {}
        #: Per-run counters, surfaced in the pipeline's final report.
        self.stats: dict = {
            "requests": 0,
            "retries": 0,
            "failures": 0,
            "cache_hits": 0,
        }

    # -- glossary priming --------------------------------------------------

    @staticmethod
    def _normalise_term(term: str) -> str:
        """Mirrors glossary.normalise_key so priming and lookup agree."""
        key = re.sub(r"\s+", " ", (term or "").strip()).lower()
        if len(key) > 3 and key.endswith("s") and not key.endswith(("ss", "is", "us")):
            key = key[:-1]
        return key

    def prepare_terms(self, terms) -> None:
        """
        Translates a batch of glossary keys in one request and caches results.

        Call once, before translating blocks. Keys that fail to translate are
        simply left uncached; callers fall back to a per-block translation.

        Failures are logged rather than swallowed: silently losing the whole
        glossary pass was previously invisible, which made the "consistent
        terminology" feature look implemented while doing nothing.
        """
        unique = []
        seen = set()
        for term in terms:
            key = (term or "").strip()
            if not key or key.lower() in seen:
                continue
            seen.add(key.lower())
            unique.append(key)

        if not unique:
            return

        numbered = "\n".join(f"{i + 1}. {key}" for i, key in enumerate(unique))
        prompt = (
            "Translate each numbered dental/medical term below into natural "
            "professional Persian. Return exactly one line per term, in the same "
            "order, formatted as `<number>. <Persian>`. Use a concise 1-3 word "
            "Persian rendering that a dentist would actually write; do not add "
            "explanations, parentheses, or the English term.\n\n" + numbered
        )

        try:
            raw = self._translate_with_system(GLOSSARY_SYSTEM_PROMPT, prompt)
        except Exception as exc:  # noqa: BLE001 - priming is best-effort
            logger.warning("Glossary priming failed (%s); falling back to per-block translation.", exc)
            return

        matched = 0
        for line in (raw or "").splitlines():
            match = re.match(r"\s*(\d+)\s*[\.\)\-:]?\s*(.+)", line)
            if not match:
                continue
            index = int(match.group(1))
            if not (1 <= index <= len(unique)):
                continue
            rendering = match.group(2).strip().strip(".").strip()
            if rendering:
                self._term_translations[self._normalise_term(unique[index - 1])] = rendering
                matched += 1

        if matched == 0:
            logger.warning(
                "Glossary priming returned no parsable terms (%d requested); "
                "terminology will not be pre-consolidated.",
                len(unique),
            )

    def term_translation(self, term: str) -> Optional[str]:
        """Cached Persian rendering of a glossary key, if priming produced one."""
        return self._term_translations.get(self._normalise_term(term))

    # -- translation -------------------------------------------------------

    @abstractmethod
    def translate_text(self, text: str) -> str:
        """
        Translates a medical text block into fluent Persian.
        """
        pass

    @abstractmethod
    def translate_caption(self, caption: str) -> str:
        """
        Translates a figure caption into fluent Persian while preserving figure numbering.
        """
        pass

    def translate_reference(self, reference: str) -> str:
        """
        Translates a bibliography entry, keeping journal names/DOIs/volumes in English.

        Reuses the provider's normal translate_text path but swaps the system
        prompt. Providers that need a different call shape can override this.
        """
        return self._translate_with_system(REFERENCE_TRANSLATOR_SYSTEM_PROMPT, reference)

    @abstractmethod
    def _translate_with_system(self, system_prompt: str, text: str) -> str:
        """
        Translates `text` under an explicit system prompt.

        This is the single point every provider must implement, because the
        system prompt is what separates medical prose, figure captions,
        bibliography entries and glossary priming. The previous default
        implementation ignored the argument entirely and silently reused the
        medical prompt, so reference and glossary calls were translated with the
        wrong instructions.
        """
        raise NotImplementedError

    def describe_image(self, image_path: str) -> Optional[str]:
        """
        Optional vision analysis: generates a concise Persian clinical description
        of the image findings. Returns None if vision is unsupported or disabled.
        """
        return None
