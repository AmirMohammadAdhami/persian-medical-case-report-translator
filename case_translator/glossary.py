"""
Curated glossary of dental/clinical terms used to build the interactive UI.

Two jobs:

1. **Clickable terms.** Every (Persian (English)) pair in the rendered output is
   auto-detected and turned into a clickable element. The curated CLINICAL_TERMS
   below seed the dictionary with known-good Persian renderings so the same term
   is worded identically everywhere in the document.

2. **Anatomical protection.** ANATOMICAL_TERMS must never be translated. They are
   handled in the prompt (see base.MEDICAL_TRANSLATOR_SYSTEM_PROMPT); this list is
   the reference copy used for validation and documentation.

Matching is deliberately case-insensitive and word-boundary based so that
"incisor" does not fire inside "incisors" incorrectly, while
"maxillary right lateral incisor" matches as a whole phrase.
"""

import re
from typing import Dict, Iterable, List, Optional


# Tooth types and positional qualifiers: NEVER translated.
# A mistranslation here is a clinical error (lateral vs central, maxillary vs
# mandibular), so these stay in English in the output.
ANATOMICAL_TERMS = [
    # tooth types
    "central incisor",
    "lateral incisor",
    "incisor",
    "canine",
    "premolar",
    "molar",
    "wisdom tooth",
    "deciduous tooth",
    "primary tooth",
    "permanent tooth",
    # position / surface qualifiers
    "maxillary",
    "mandibular",
    "mesial",
    "distal",
    "buccal",
    "lingual",
    "palatal",
    "occlusal",
    "apical",
    "coronal",
    "interproximal",
    "proximal",
    "cervical",
    "radicular",
]


# Curated clinical terms -> preferred Persian rendering.
# These seed the interactive glossary so terminology is consistent document-wide.
# Order matters only for documentation; lookups are by exact key.
CLINICAL_TERMS: Dict[str, str] = {
    "Complicated Crown-Root Fracture": "شکستگی پیچیده تاج-ریشه",
    "Crown-Root Fracture": "شکستگی تاج-ریشه",
    "Crown Fracture": "شکستگی تاج",
    "Root Fracture": "شکستگی ریشه",
    "Reattachment Procedure": "روش اتصال مجدد",
    "Reattachment": "اتصال مجدد",
    "Fiber Post": "پست فایبر",
    "Biologic Width": "عرض بیولوژیک",
    "Biological Width": "عرض بیولوژیک",
    "Orthodontic Extrusion": "اکستروژن ارتودنتیک",
    "Isolation": "ایزولاسیون",
    "Protrusive Eruptive Pattern": "الگوی رویش پروتروزیو",
    "Dental Trauma": "تروما دندانی",
    "Traumatic Dental Injuries": "آسیب‌های دندانی تروماتیک",
    "Intraoral": "داخل‌دهانی",
    "Radiograph": "رادیوگراف",
    "Periapical Radiograph": "رادیوگراف پری‌اپیکال",
    "Pulp": "پالپ",
    "Pulpotomy": "پالپوتومی",
    "Pulpectomy": "پالپکتومی",
    "Root Canal Treatment": "درمان کانال ریشه",
    "Composite Resin": "رزین کامپوزیت",
    "Enamel": "مینا",
    "Dentin": "عاج",
    "Cementum": "سِمان",
    "Periodontal Ligament": "لیگامان پریودنتال",
    "Alveolar Bone": "استخوان آلوئولار",
    "Gingiva": "لثه",
    "Vitality Test": "تست وایتالیتی",
    "Percussion Test": "تست پرکاشن",
    "Mobility": "موبیلیتی",
    "Splinting": "اسپلینت‌گذاری",
    "Follow-Up": "پیگیری",
    "Prognosis": "پروگنوز",
    "Anterior Teeth": "دندان‌های قدامی",
    "Posterior Teeth": "دندان‌های خلفی",
    "Apexification": "اپکسیفیکاسیون",
    "MTA": "مینرال تری‌اکساید اگریگیت",
}


def normalise_key(term: str) -> str:
    """
    Canonical lookup key for a glossary term.

    Lower-cases, collapses whitespace, and strips a trailing plural 's'. Without
    the plural fold, "crown-root fractures" (what the model writes) would miss
    the authored "crown-root fracture" note and fall back to a generic blurb.
    """
    key = re.sub(r"\s+", " ", (term or "").strip()).lower()
    if len(key) > 3 and key.endswith("s") and not key.endswith(("ss", "is", "us")):
        key = key[:-1]
    return key


def notes_index() -> Dict[str, str]:
    """GLOSSARY_NOTES keyed by normalise_key, for tolerant lookups."""
    return {normalise_key(k): v for k, v in GLOSSARY_NOTES.items()}


def note_for(term: str, index: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Authored explanation for a term, tolerating case and plural differences."""
    idx = index if index is not None else notes_index()
    return idx.get(normalise_key(term))


def all_glossary_keys() -> List[str]:
    """Every term worth priming with a single batch translation pass."""
    keys = list(CLINICAL_TERMS.keys())
    seen = {k.lower() for k in keys}
    for term in ANATOMICAL_TERMS:
        if term.lower() not in seen:
            seen.add(term.lower())
            keys.append(term)
    return keys


# Authored explanations surfaced in the interactive UI when a reader clicks a
# term. Keys are lowercase English terms. Definitions are deliberately short
# and clinically literal — they are reference notes, not a textbook.
GLOSSARY_NOTES: Dict[str, str] = {
    "complicated crown-root fracture": (
        "شکستگی‌ای که هم تاج و هم ریشهٔ دندان را درگیر می‌کند و خط شکست به زیر "
        "سطح لثه یا به سمت پالپ امتداد یافته است. درمان آن معمولاً پیچیده‌تر از "
        "شکستگی تاج تنهاست و ممکن است نیاز به جراحی یا اکستروژن داشته باشد."
    ),
    "crown-root fracture": (
        "شکستگی دندان به‌صورت همزمان در ناحیهٔ تاج و ریشه، بدون درگیری پالپ یا با "
        "درگیری آن، بسته به عمق شکست."
    ),
    "crown fracture": "شکستگی محدود به تاج دندان، شامل مینا و احتمالاً عاج.",
    "root fracture": (
        "شکستگی در ناحیهٔ ریشهٔ دندان؛ بر اساس محل (گردنی، میانی، اپیکال) و "
        "درجهٔ جابه‌جایی قطعات طبقه‌بندی می‌شود."
    ),
    "reattachment procedure": (
        "روشی که در آن قطعهٔ شکستهٔ دندان با استفاده از رزین کامپوزیت و "
        "احتمالاً پست فایبر به دندان اصلی بازچسبانده می‌شود؛ مزیت آن حفظ شکل و "
        "رنگ طبیعی دندان است."
    ),
    "reattachment": "بازچسباندن قطعهٔ شکستهٔ دندان به باقی‌ماندهٔ آن.",
    "fiber post": (
        "پست تقویت‌شده با الیاف شیشه‌ای که برای تثبیت بازسازی در کانال ریشه "
        "به کار می‌رود؛ مدول الاستیسیتهٔ نزدیک به عاج دارد و خطر شکست ریشه را "
        "کم می‌کند."
    ),
    "biologic width": (
        "فاصلهٔ ثابت میان قاعدهٔ شیار لثه و تاج استخوان آلوئولار (حدود ۲ میلی‌متر)؛ "
        "نقض آن منجر به تحلیل استخوان و التهاب مزمن لثه می‌شود."
    ),
    "orthodontic extrusion": (
        "اکستروژن ارتودنتیک: حرکت دادن تدریجی دندان به سمت تاج با نیروی ارتودنتیک "
        "برای قابل‌دسترس قرار دادن حاشیهٔ شکست یا فراهم کردن عرض بیولوژیک مناسب."
    ),
    "isolation": (
        "ایزولاسیون: جداسازی دندان از بزاق و رطوبت دهان، معمولاً با رابردام، "
        "برای ایجاد محیط خشک در ترمیم‌های چسبنده."
    ),
    "protrusive eruptive pattern": (
        "الگوی رویش پروتروزیو: الگویی از رویش که در آن دندان در حین حرکت فک "
        "به سمت جلو بیشتر در معرض سایش و ضربه قرار می‌گیرد."
    ),
    "dental trauma": "آسیب تروماتیک به ساختارهای دندان و بافت‌های پشتیبان آن.",
    "traumatic dental injuries": (
        "آسیب‌های دندانی تروماتیک: ضربات وارده به دندان و بافت‌های اطراف آن که "
        "می‌تواند تاج، ریشه، پالپ، لثه و استخوان آلوئولار را درگیر کند."
    ),
    "intraoral": "داخل‌دهانی: مربوط به داخل حفرهٔ دهان.",
    "radiograph": "رادیوگراف: تصویربرداری با پرتو ایکس برای بررسی ساختارهای دندانی و استخوانی.",
    "periapical radiograph": (
        "رادیوگراف پری‌اپیکال: تصویربرداری با زاویهٔ موازی یا نیمساز که اپکس ریشه "
        "و استخوان پری‌اپیکال را نشان می‌دهد."
    ),
    "pulp": "پالپ: بافت نرم داخل دندان شامل اعصاب، عروق و بافت پیوندی.",
    "pulpotomy": "پالپوتومی: برداشتن بخش تاجی پالپ با حفظ پالپ ریشه.",
    "pulpectomy": "پالپکتومی: برداشتن کامل پالپ از کانال ریشه.",
    "root canal treatment": (
        "درمان کانال ریشه: پاک‌سازی، شکل‌دهی و پر کردن کانال ریشهٔ دندان "
        "برای حفظ دندان در صورت آسیب پالپ."
    ),
    "composite resin": (
        "رزین کامپوزیت: مادهٔ ترمیمی همرنگ دندان بر پایهٔ رزین که با نور "
        "پلیمریزه می‌شود."
    ),
    "enamel": "مینا: سخت‌ترین بافت بدن که تاج دندان را می‌پوشاند.",
    "dentin": "عاج: بافت زیر مینا و سمان که بلورهای اصلی تاج و ریشه را می‌سازد.",
    "cementum": "سِمان: بافت معدنی‌شدهٔ پوشانندهٔ سطح ریشهٔ دندان.",
    "periodontal ligament": (
        "لیگامان پریودنتال: بافت همبندی میان سمان ریشه و استخوان آلوئولار که "
        "دندان را در جای خود نگه می‌دارد."
    ),
    "alveolar bone": "استخوان آلوئولار: بخشی از استخوان فک که حفرهٔ دندان را در بر می‌گیرد.",
    "gingiva": "لثه: بافت نرم پوشانندهٔ استخوان آلوئولار و گردن دندان.",
    "vitality test": (
        "تست وایتالیتی: ارزیابی پاسخ پالپ به محرک‌هایی مانند سرما، گرما یا "
        "تحریک الکتریکی."
    ),
    "percussion test": (
        "تست پرکاشن: ضربهٔ ملایم به دندان برای بررسی التهاب لیگامان پریودنتال."
    ),
    "mobility": "موبیلیتی: میزان تحرک دندان که نشانهٔ آسیب بافت نگهدارنده است.",
    "splinting": (
        "اسپلینت‌گذاری: بستن دندان آسیب‌دیده به دندان‌های مجاور برای تثبیت در "
        "دورهٔ ترمیم."
    ),
    "follow-up": "پیگیری: معاینات دوره‌ای برای ارزیابی ترمیم و سلامت پالپ.",
    "prognosis": "پروگنوز: پیش‌آگهی نتیجهٔ درمان و احتمال حفظ دندان.",
    "anterior teeth": "دندان‌های قدامی: دندان‌های جلوی دهان شامل ثنایا و نیش.",
    "posterior teeth": "دندان‌های خلفی: پرمولرها و مولرها.",
    "apexification": (
        "اپکسیفیکاسیون: ایجاد سدّ سخت در انتهای کانال ریشه‌های نابالغ نکروزه "
        "برای امکان پر کردن کانال."
    ),
    "mta": (
        "مینرال تری‌اکساید اگریگیت: مادهٔ سیمانی زیست‌سازگار که در درمان‌های "
        "اندودنتیک برای ترمیم سوراخ‌شدگی‌ها و اپکسیفیکاسیون به کار می‌رود."
    ),
    # --- Terms that appear in real case reports and would otherwise fall
    # --- back to the generic note in the UI.
    "biological width": (
        "عرض بیولوژیک: فاصلهٔ ثابت میان قاعدهٔ شیار لثه و تاج استخوان آلوئولار "
        "(حدود ۲ میلی‌متر). نقض آن باعث التهاب مزمن لثه و تحلیل استخوان می‌شود."
    ),
    "crown-root fractures": (
        "شکستگی‌های تاج-ریشه: شکستگی‌هایی که هم تاج و هم ریشهٔ دندان را درگیر "
        "می‌کنند و خط شکست ممکن است به زیر لثه یا به پالپ امتداد یابد."
    ),
    "endodontic treatment": (
        "درمان اندودنتیک: درمان داخل‌دندانی شامل پاک‌سازی، شکل‌دهی و پر کردن "
        "سیستم کانال ریشه برای حفظ دندان با پالپ آسیب‌دیده."
    ),
    "fractured fragment": (
        "قطعهٔ شکسته: بخش جدا‌شدهٔ دندان که در صورت سالم بودن می‌تواند با روش "
        "اتصال مجدد به دندان بازگردانده شود."
    ),
    "gingivectomy": (
        "جینجیوکتومی: برداشتن جراحی بخشی از بافت لثه برای دسترسی به حاشیهٔ "
        "ترمیم یا اصلاح شکل لثه."
    ),
    "histopathology": (
        "هیستوپاتولوژی: بررسی میکروسکوپی بافت برای تشخیص ماهیت ضایعه یا "
        "تأیید پاسخ التهابی."
    ),
    "maxillofacial region": (
        "ناحیهٔ فک و صورت (ماگزیلوفاشیال): محدودهٔ آناتومیک شامل فک بالا، فک "
        "پایین و ساختارهای صورت."
    ),
    "osteotomy": (
        "استئوتومی: برش یا برداشتن برنامه‌ریزی‌شدهٔ استخوان، در اندودنتیک "
        "معمولاً برای دسترسی به اپکس یا جراحی پری‌اپیکال."
    ),
    "periodontal": (
        "پریودنتال: مربوط به بافت‌های نگهدارندهٔ دندان شامل لثه، لیگامان "
        "پریودنتال، سمان و استخوان آلوئولار."
    ),
    "preoperative intraoral view": (
        "نمای داخل‌دهانی پیش از عمل: تصویر بالینی که وضعیت دندان و بافت‌های "
        "نرم را قبل از شروع درمان نشان می‌دهد."
    ),
    "pulp tissue": (
        "بافت پالپ: بافت نرم داخل دندان شامل اعصاب، عروق خونی و بافت پیوندی "
        "که مسئول حس و تغذیهٔ دندان است."
    ),
    "pulpal exposure": (
        "مواجههٔ پالپ: باز شدن پالپ به دلیل شکستگی یا پوسیدگی؛ نیازمند "
        "درمان پالپ مانند پالپوتومی، پالپکتومی یا درمان کانال ریشه."
    ),
    "radiolucent lesion": (
        "ضایعهٔ رادیولوسنت: ناحیهٔ تیره در رادیوگراف که نشان‌دهندهٔ کاهش "
        "تراکم استخوان یا بافت معدنی‌شده است."
    ),
    "rubber dam": (
        "رابردام: ورقهٔ لاتکس یا غیرلاتکس برای ایزولاسیون دندان از بزاق و "
        "ایجاد میدان خشک در درمان‌های چسبنده و اندودنتیک."
    ),
    "surgical crown lengthening": (
        "افزایش طول تاج جراحی: برداشتن جراحی بافت لثه و گاه استخوان برای "
        "قرار گرفتن حاشیهٔ ترمیم روی مینا و فراهم کردن عرض بیولوژیک مناسب."
    ),
    "dual-cure composite resin": (
        "رزین کامپوزیت دوال‌کور: کامپوزیتی که هم با نور و هم با واکنش شیمیایی "
        "پلیمریزه می‌شود؛ مناسب نواحی با دسترسی محدود به نور."
    ),
    "glass fiber post": (
        "پست فایبر شیشه‌ای: پست کامپوزیتی تقویت‌شده با الیاف شیشه برای تثبیت "
        "بازسازی در کانال ریشه؛ مدول الاستیسیتهٔ نزدیک به عاج دارد."
    ),
    "conservative reattachment": (
        "اتصال مجدد محافظه‌کارانه: بازچسباندن قطعهٔ شکستهٔ دندان بدون آماده‌سازی "
        "گستردهٔ دندان، با کمترین آسیب به ساختار باقی‌مانده."
    ),
    "intraoral photograph": (
        "تصویر داخل‌دهانی: عکس بالینی گرفته‌شده از داخل دهان برای ثبت وضعیت "
        "دندان و بافت‌های نرم."
    ),
    "pulp exposure": (
        "نمایان شدن پالپ: باز شدن پالپ به دلیل شکستگی یا پوسیدگی؛ نیازمند "
        "درمان پالپ مانند پولپوتومی، پولپکتومی یا درمان کانال ریشه."
    ),
    "incisal function": (
        "عملکرد برشی: نقش دندان‌های قدامی در بریدن غذا؛ حفظ آن از اهداف مهم "
        "بازسازی دندان‌های جلویی است."
    ),
    "endodontic treatment": (
        "درمان اندودنتیک: درمان داخل‌دندانی شامل پاک‌سازی، شکل‌دهی و پر کردن "
        "سیستم کانال ریشه برای حفظ دندان با پالپ آسیب‌دیده."
    ),
    "subgingival fractures": (
        "شکستگی‌های زیرلثه‌ای: شکستگی‌هایی که خط آن‌ها به زیر سطح لثه امتداد "
        "یافته است؛ دسترسی و ترمیم آن‌ها دشوار است و اغلب به افزایش طول تاج یا "
        "اکستروژن نیاز دارند."
    ),
}


def anatomical_pattern() -> re.Pattern:
    """
    Compiled matcher for anatomical terms, longest phrase first.

    Longest-first matters: "maxillary right lateral incisor" must be able to
    match before the bare "incisor" does.
    """
    ordered = sorted(ANATOMICAL_TERMS, key=len, reverse=True)
    alternation = "|".join(re.escape(term) for term in ordered)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def is_anatomical_term(term: str) -> bool:
    """
    True when the given term must stay in English and should not be clickable.

    Matches on a word-boundary search, not a full match: anatomical names appear
    as phrases ("maxillary central incisor") whose head noun is one of the
    protected words, and requiring the whole phrase to equal a single list entry
    would let those through.
    """
    return bool(anatomical_pattern().search((term or "").strip()))


def iter_known_terms(extra: Iterable[str] = ()) -> List[str]:
    """Glossary keys plus any extra terms discovered at runtime, de-duplicated."""
    keys = all_glossary_keys()
    seen = {k.lower() for k in keys}
    for term in extra:
        if term and term.lower() not in seen:
            seen.add(term.lower())
            keys.append(term)
    return keys
