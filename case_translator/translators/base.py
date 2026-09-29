"""
Base translator abstraction and medical translation guidelines.
"""

import re
from abc import ABC, abstractmethod
from typing import Optional


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


class Translator(ABC):
    """Abstract base class for all AI translation providers."""

    def __init__(self) -> None:
        # Persian rendering of glossary keys, learned once per run.
        #
        # A key like "incisor" can appear in a dozen blocks, and the model may
        # render it differently each time. Translating it once and reusing that
        # keeps terminology consistent AND makes repeated terms cheap.
        self._term_translations: dict = {}

    def prepare_terms(self, terms) -> None:
        """
        Translates a batch of glossary keys in one request and caches results.

        Call once, before translating blocks. Keys that fail to translate are
        simply left uncached; callers fall back to a per-block translation.
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
            raw = self.translate_text(prompt)
        except Exception:
            # Glossary priming is an optimisation; never fail a run over it.
            return

        for line in (raw or "").splitlines():
            match = re.match(r"\s*(\d+)\s*[\.\)\-:]?\s*(.+)", line)
            if not match:
                continue
            index = int(match.group(1))
            if not (1 <= index <= len(unique)):
                continue
            rendering = match.group(2).strip().strip(".").strip()
            if rendering:
                self._term_translations[unique[index - 1].lower()] = rendering

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

    def _translate_with_system(self, system_prompt: str, text: str) -> str:
        """Provider hook for translating with a specific system prompt."""
        return self.translate_text(text)

    def describe_image(self, image_path: str) -> Optional[str]:
        """
        Optional vision analysis: generates a concise Persian clinical description
        of the image findings. Returns None if vision is unsupported or disabled.
        """
        return None
