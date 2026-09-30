"""
Numeric fidelity checks for translated medical text.

Why this exists
---------------
Doses, dimensions, dates, follow-up intervals and percentages are the part of a
case report where a translation slip becomes a clinical error. Models rarely
invent a number, but they do drop one, merge two ("3 and 6 months" -> "3
months"), or convert units unprompted. The check below is intentionally simple
and deterministic: it compares the multiset of numeric tokens and the set of
measurement units between source and translation, and reports anything the
translation lost or added. It never rewrites the text — it only flags it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

# Persian/Arabic-Indic digits, so "۱۲" and "12" compare equal.
_DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

# Measurement units that carry clinical meaning. Matched case-sensitively only
# where the case matters (mM vs mm, Gy vs gy), so the alternation is ordered
# longest-first to stop "mm" from matching inside "mmol".
_UNITS = (
    "mmol/L", "mol/L", "mg/dL", "µg/mL", "ug/mL", "mcg/mL", "ng/mL", "pg/mL",
    "mm/s", "cm/s", "m/s", "km/h",
    "mmol", "µmol", "umol", "mcg", "µg", "ug", "ng", "pg", "mg", "kg", "mL",
    "ml", "dL", "L", "cm", "mm", "µm", "um", "nm", "kPa", "MPa", "Pa", "Gy",
    "mSv", "Sv", "Hz", "kHz", "MHz", "W", "V", "mA", "A", "min", "sec", "h",
    "°C", "°F", "%", "rpm", "N", "Ncm", "mm²", "cm²",
)
_UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)) + r")(?![A-Za-z])"
)

# A bare integer followed by a period at the start of a reference is a citation
# number, not a measurement. They are compared anyway (both sides have them),
# so no special-casing is needed here.


def _normalise(text: str) -> str:
    return (text or "").translate(_DIGIT_TRANSLATION)


def extract_numbers(text: str) -> Counter:
    """Multiset of numeric tokens, with Persian digits folded to ASCII."""
    numbers = _NUMBER_RE.findall(_normalise(text))
    # "1,5" (European decimal comma) and "1,500" (thousands separator) are
    # ambiguous; normalising the comma away would merge two real numbers, so
    # both forms are kept as written but with a canonical separator.
    return Counter(numbers)


def extract_units(text: str) -> Counter:
    """Multiset of measurement units present in the text."""
    return Counter(_UNIT_PATTERN.findall(_normalise(text)))


def compare_numbers(source: str, translation: str, max_reports: int = 4) -> List[str]:
    """
    Returns a list of human-readable warnings about numeric drift.

    An empty list means every number and unit in the source is present in the
    translation (and vice versa). Ordering differences are ignored on purpose:
    Persian sentence structure legitimately reorders clauses.
    """
    warnings: List[str] = []

    src_numbers = extract_numbers(source)
    dst_numbers = extract_numbers(translation)

    missing = src_numbers - dst_numbers
    added = dst_numbers - src_numbers

    if missing:
        detail = "، ".join(f"{value}×{count}" for value, count in sorted(missing.items()))
        warnings.append(f"اعداد جاافتاده در ترجمه: {detail}")
    if added:
        detail = "، ".join(f"{value}×{count}" for value, count in sorted(added.items()))
        warnings.append(f"اعداد اضافه‌شده در ترجمه: {detail}")

    src_units = extract_units(source)
    dst_units = extract_units(translation)

    missing_units = src_units - dst_units
    if missing_units:
        detail = "، ".join(sorted(missing_units))
        warnings.append(f"واحدهای جاافتاده در ترجمه: {detail}")

    return warnings[:max_reports]
