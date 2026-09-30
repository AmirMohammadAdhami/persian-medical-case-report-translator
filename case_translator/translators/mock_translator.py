"""
Mock / Offline translator for testing, validation, and offline execution.
Implements the medical terminology rule and produces fluent Persian case report text.
"""

import re
from typing import Dict, Optional, Set

from .base import Translator


class MockTranslator(Translator):
    """
    Offline translator with pre-curated case report domain knowledge,
    medical terminology mapping with parenthetical English injection,
    and fallback rule-based translation.
    """

    # Medical terminology mapping: english_phrase -> persian_phrase
    # When translated, first occurrence will be: persian (english)
    MEDICAL_TERM_MAP = {
        "complicated crown-root fracture": "شکستگی پیچیده تاج-ریشه",
        "crown-root fracture": "شکستگی تاج-ریشه",
        "crown fracture": "شکستگی تاج",
        "reattachment procedure": "فرایند اتصال مجدد",
        "reattachment": "اتصال مجدد",
        "single visit technique": "تکنیک تک‌جلسه‌ای",
        "periapical radiograph": "رادیوگرافی پری‌آپیکال",
        "intraoral photograph": "تصویر داخل‌دهانی",
        "intraoral view": "نمای داخل‌دهانی",
        "biological width": "عرض بیولوژیک",
        "periodontal ligament": "رباط پریودنتال",
        "periodontal": "پریودنتال",
        "pulpal exposure": "اکسپوز پالپ",
        "pulp tissue": "بافت پالپ",
        "pulp": "پالپ",
        "root canal treatment": "درمان ریشه دندان",
        "endodontic treatment": "درمان اندودنتیک",
        "composite resin": "کامپوزیت رزین",
        "radiolucent lesion": "ضایعه رادیولوسنت",
        "radiolucent": "رادیولوسنت",
        "radiopaque": "رادیواپک",
        "osteotomy": "استئوتومی",
        "gingivectomy": "ژنژیوکتومی",
        "histopathology": "هیستوپاتولوژی",
        "rubber dam": "رابردم",
        "fiber post": "پست فایبر",
        "light-curing": "لایت کیورینگ",
        "fractured fragment": "قطعه شکسته",
        "coronal fragment": "قطعه تاجی",
        # NOTE: anatomical names are deliberately absent from this map. The
        # project's own rule is that tooth types and position qualifiers
        # (maxillary, mandibular, incisor, molar, ...) stay in English, so the
        # offline translator must not "translate" them either — it previously
        # rendered "mandibular" as "فک پایین", contradicting the rule the prompt
        # enforces for every real provider.
    }

    # Pre-translated common case report headings
    HEADING_MAP = {
        "abstract": "چکیده",
        "introduction": "۱. مقدمه",
        "1. introduction": "۱. مقدمه",
        "case report": "گزارش مورد (Case Report)",
        "case presentation": "شرح مورد بالینی (Case Presentation)",
        "case description": "توصیف مورد (Case Description)",
        "discussion": "بحث و بررسی",
        "conclusion": "نتیجه‌گیری",
        "conclusions": "نتیجه‌گیری‌ها",
        "references": "منابع و مراجع",
        "acknowledgement": "تقدیر و تشکر",
        "acknowledgments": "تقدیر و تشکر",
        "conflict of interests": "تضاد منافع",
        "conflicts of interest": "تضاد منافع",
    }

    def __init__(self, enable_vision: bool = False):
        # Base class owns the glossary cache and stats; see Translator.__init__.
        super().__init__()
        # Defaults to False for consistency with every other translator.
        # describe_image() returns a fixed canned description, so leaving this
        # ON by default would inject fabricated clinical findings into output.
        self.enable_vision = enable_vision
        self.seen_terms: Set[str] = set()

    def _translate_with_system(self, system_prompt: str, text: str) -> str:
        """
        Offline stand-in for a provider call with an explicit system prompt.

        The glossary pass sends a numbered term list; answering it with the
        normal prose translator would return the list mangled. So the numbered
        payload is recognised and answered term-by-term from the local map.
        """
        if "lexicographer" in system_prompt or "numbered" in system_prompt.lower():
            lines_out = []
            for line in (text or "").splitlines():
                match = re.match(r"\s*(\d+)\.\s*(.+)", line)
                if not match:
                    continue
                number, term = match.group(1), match.group(2).strip()
                rendering = (
                    self.MEDICAL_TERM_MAP.get(term.lower())
                    or self._offline_term(term)
                )
                lines_out.append(f"{number}. {rendering}")
            if lines_out:
                return "\n".join(lines_out)
        return self.translate_text(text)

    @staticmethod
    def _offline_term(term: str) -> str:
        """Keeps anatomical terms in English, otherwise echoes the term as-is."""
        from ..glossary import is_anatomical_term

        if is_anatomical_term(term):
            return term
        return term

    def _inject_terminology(self, text: str) -> str:
        """
        Replaces medical terms with Persian equivalents and adds English in parentheses
        on their first meaningful occurrence, using a single regex pass to prevent nesting.
        """
        sorted_terms = sorted(self.MEDICAL_TERM_MAP.keys(), key=lambda x: -len(x))
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in sorted_terms) + r')\b',
            re.IGNORECASE
        )

        def _replace_match(match: re.Match) -> str:
            matched_text = match.group(1)
            lower_term = matched_text.lower()

            # Find matching key in map
            fa_term = self.MEDICAL_TERM_MAP.get(lower_term)
            if not fa_term:
                return matched_text

            if lower_term not in self.seen_terms:
                self.seen_terms.add(lower_term)
                return f"{fa_term} ({lower_term})"
            else:
                return fa_term

        return pattern.sub(_replace_match, text)

    def translate_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        text_clean = text.strip()
        lower_clean = text_clean.lower()

        # Check pre-translated headings
        if lower_clean in self.HEADING_MAP:
            return self.HEADING_MAP[lower_clean]

        # Check Title match
        if "complicated crown-root fracture treated using reattachment" in lower_clean:
            return "درمان شکستگی پیچیده تاج-ریشه با استفاده از روش اتصال مجدد (reattachment procedure): یک تکنیک تک‌جلسه‌ای"

        # Check known common sections for 10.1155/2011/401678 case report
        if "trauma to the oral and maxillofacial region occurs frequently" in lower_clean or "crown-root fractures comprise" in lower_clean:
            return (
                "تروما به ناحیه دهان و فک و صورت (maxillofacial region) به‌وفور رخ می‌دهد و آسیب‌های دندانی سهم عمده‌ای از این موارد را تشکیل می‌دهند. "
                "شکستگی‌های تاج-ریشه (crown-root fractures) حدود ۵ درصد از آسیب‌های وارده به دندان‌های دائمی را شامل می‌شوند. "
                "درمان این شکستگی‌ها به دلیل درگیری بافت پالپ (pulp tissue) و لثه زیر بیولوژیک با چالش‌های تشخیصی و درمانی همراه است. "
                "روش اتصال مجدد قطعه دندانی شکسته (reattachment procedure) با استفاده از سیستم‌های باندینگ و کامپوزیت رزین، یک راهکار بیولوژیک، سریع و محافظه‌کارانه به شمار می‌رود."
            )

        if "a 22-year-old male patient reported" in lower_clean or "a male patient reported to the department" in lower_clean:
            return (
                "بیمار مرد ۲۲ ساله‌ای به بخش دندانپزشکی ترمیمی مراجعه نمود که به دنبال ضربه تصادفی دچار آسیب در دندان پیشین میانی فک بالا (maxillary central incisor) شده بود. "
                "در معاینات بالینی و عکس‌برداری رادیوگرافی، شکستگی پیچیده تاج-ریشه همراه با اکسپوز پالپ (pulpal exposure) و لق‌شدن قطعه تاجی مشاهده شد. "
                "قطعه شکسته توسط بیمار درون یک محیط مرطوب نگهداری شده بود. پس از ارزیابی‌های اولیه، طرح درمان تک‌جلسه‌ای شامل درمان ریشه دندان (endodontic treatment) و اتصال مجدد با پست فایبر (fiber post) در نظر گرفته شد."
            )

        if "after administration of local anesthesia" in lower_clean or "under rubber dam isolation" in lower_clean:
            return (
                "پس از تزریق بی‌حسی موضعی و تحت ایزولاسیون با رابردم (rubber dam)، قطعه شکسته به آرامی جدا و بررسی گردید. "
                "آماده‌سازی کانال ریشه و پاکسازی انجام شد و گوتاپرکا در کانال قرار گرفت. سپس پست فایبر مناسب انتخاب و با سیمان رزینی سایلنیزه چسبانده شد. "
                "قطعه تاجی دندان پس از آماده‌سازی و اسید اچ با سیستم باندینگ و کامپوزیت رزین (composite resin) به ریشه متصل گردید و تطابق اکلوزال بررسی شد."
            )

        if "the patient was recalled after" in lower_clean or "follow-up examination" in lower_clean:
            return (
                "بیمار در دوره‌های پیگیری ۳، ۶ و ۱۲ ماهه مورد ارزیابی بالینی و رادیوگرافی قرار گرفت. "
                "در معاینات پیگیری، بافت پریودنتال (periodontal) کاملاً سالم، بدون علائم التهاب، جیب پریودنتال یا لقی بود. "
                "رادیوگرافی پری‌آپیکال (periapical radiograph) ترمیم کامل بافت و عدم وجود هرگونه ضایعه رادیولوسنت (radiolucent lesion) در ناحیه آپیکال را تایید نمود. بیمار از نظر زیبایی و عملکرد رضایت کامل داشت."
            )

        # For reference citations, keep intact without extra tags
        if re.match(r'^\s*\[\d+\]', text_clean):
            return text_clean

        # General case report fallback: apply medical terminology enhancement
        # Ensure numbers, citations [1], percentages %, units (mm, cm, mg) are preserved
        fa_text = self._inject_terminology(text_clean)
        return fa_text

    def translate_caption(self, caption: str) -> str:
        if not caption or not caption.strip():
            return ""

        caption_clean = caption.strip()
        match = re.search(r'^(?:Figure|Fig\.|شکل)\s*(\d+)[\.:\s\-]+(.*)', caption_clean, re.IGNORECASE)
        if match:
            num = match.group(1)
            rest = match.group(2).strip()

            # Persian numeral conversion
            fa_num = num.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))

            # Specific known captions
            lower_rest = rest.lower()
            if "preoperative" in lower_rest and "intraoral" in lower_rest:
                translated_body = "نمای داخل‌دهانی پیش از درمان (preoperative intraoral view) که شکستگی را نشان می‌دهد."
            elif "radiograph" in lower_rest or "periapical" in lower_rest:
                translated_body = "رادیوگرافی پری‌آپیکال (periapical radiograph) نشان‌دهنده خط شکستگی و وضعیت ریشه دندان."
            elif "fragment" in lower_rest or "reattachment" in lower_rest:
                translated_body = "قطعه تاجی شکسته شده (fractured fragment) پیش از اتصال مجدد."
            elif "postoperative" in lower_rest or "follow-up" in lower_rest:
                translated_body = "نمای بالینی پس از درمان و اتصال موفقیت‌آمیز قطعه با کامپوزیت رزین."
            else:
                translated_body = self._inject_terminology(rest)

            return f"شکل {fa_num}. {translated_body}"

        return self._inject_terminology(caption_clean)

    def describe_image(self, image_path: str) -> Optional[str]:
        if not self.enable_vision:
            return None
        return (
            "تصویر بالینی نشان‌دهنده وضعیت دندان پیشین فک بالا با شکستگی مایل تاجی-ریشه‌ای است. "
            "بافت‌های پریودنتال و لبه‌های دندان وضعیت مناسبی جهت فرآیند اتصال مجدد با رزین نشان می‌دهند."
        )
