"""
Tests for the clinical glossary and the interactive-term contract.

Covers two rules that protect clinical accuracy:
  - Anatomical tooth names must stay English and must not be clickable.
  - Every clickable term must have an authored explanation, so clicking never
    dead-ends in a generic placeholder.
"""

import pytest

from case_translator.glossary import (
    GLOSSARY_NOTES,
    all_glossary_keys,
    is_anatomical_term,
    normalise_key,
    note_for,
    notes_index,
)


class TestAnatomicalProtection:
    """Anatomical names are never translated, so they are never glossary terms."""

    @pytest.mark.parametrize("term", [
        "incisor",
        "premolar",
        "molar",
        "maxillary",
        "mandibular",
        "mesial",
        "buccal",
        "occlusal",
    ])
    def test_single_anatomical_words_are_protected(self, term):
        assert is_anatomical_term(term)

    @pytest.mark.parametrize("term", [
        "maxillary right lateral incisor",
        "mandibular left first molar",
        "maxillary central incisor",
    ])
    def test_anatomical_phrases_are_protected(self, term):
        """A phrase is protected when it contains an anatomical head noun."""
        assert is_anatomical_term(term)

    @pytest.mark.parametrize("term", [
        "Fiber Post",
        "Reattachment Procedure",
        "Biologic Width",
        "Orthodontic Extrusion",
    ])
    def test_clinical_terms_are_not_anatomical(self, term):
        assert not is_anatomical_term(term)

    def test_case_insensitive(self):
        assert is_anatomical_term("MAXILLARY RIGHT LATERAL INCISOR")


class TestNoteLookup:
    """Note lookup tolerates the case and plural forms a model actually writes."""

    def test_exact_match(self):
        assert note_for("Fiber Post")

    def test_case_insensitive(self):
        assert note_for("fiber post")

    def test_plural_folds_to_singular(self):
        """Regression: 'crown-root fractures' missed the authored singular note."""
        assert note_for("crown-root fractures") is not None
        assert note_for("crown-root fractures") == note_for("crown-root fracture")

    def test_words_ending_in_ss_are_not_depluralised(self):
        assert normalise_key("glass") == "glass"

    def test_unknown_term_returns_none(self):
        assert note_for("not a real term xyz") is None

    def test_index_is_keyed_by_normalised_form(self):
        index = notes_index()
        assert normalise_key("Fiber Post") in index


class TestNoteCoverage:
    """
    Every curated clinical term must have an authored note.

    A clickable term with no note falls back to a generic "not documented"
    message, which looks broken to a reader. This test keeps the two lists in
    sync as terms are added.
    """

    def test_every_curated_anatomical_term_has_no_note_requirement(self):
        """Anatomical terms are never clickable, so they need no note."""
        from case_translator.glossary import ANATOMICAL_TERMS
        for term in ANATOMICAL_TERMS:
            assert is_anatomical_term(term)

    def test_curated_clinical_terms_are_covered(self):
        from case_translator.glossary import CLINICAL_TERMS
        index = notes_index()
        missing = [
            term for term in CLINICAL_TERMS
            if not is_anatomical_term(term) and normalise_key(term) not in index
        ]
        assert missing == [], f"curated terms with no authored note: {missing}"

    def test_keys_are_unique_after_normalisation(self):
        """Two curated keys must not collapse to the same normalised key."""
        from case_translator.glossary import CLINICAL_TERMS
        seen = {}
        for term in CLINICAL_TERMS:
            key = normalise_key(term)
            assert key not in seen or seen[key] == term, (
                f"{term!r} and {seen[key]!r} both normalise to {key!r}")
            seen[key] = term


class TestNonTermLabels:
    """
    Structural labels and institutional affiliations look like glossary pairs
    but are not clinical terms.

    Regression: "گروه دندان‌پزشکی ترمیمی و اندودنتیکس (Conservative Dentistry and
    Endodontics)" was being rendered as a clickable glossary term.
    """

    @pytest.mark.parametrize("label", [
        "Conservative Dentistry and Endodontics",
        "Department of Restorative Dentistry",
        "School of Dentistry",
        "Tehran University of Medical Sciences",
        "Department of Endodontics",
    ])
    def test_affiliations_are_not_terms(self, label):
        from case_translator.html_renderer import _is_non_term_label
        assert _is_non_term_label(label)

    @pytest.mark.parametrize("label", [
        "Case Report",
        "Introduction",
        "Discussion",
        "References",
    ])
    def test_section_labels_are_not_terms(self, label):
        from case_translator.html_renderer import _is_non_term_label
        assert _is_non_term_label(label)

    @pytest.mark.parametrize("term", [
        "Fiber Post",
        "Complicated Crown-Root Fracture",
        "Biologic Width",
        "Rubber Dam",
    ])
    def test_clinical_terms_still_qualify(self, term):
        from case_translator.html_renderer import _is_non_term_label
        assert not _is_non_term_label(term)


class TestTermExtraction:
    """
    Extracting the Persian half of a pair is boundary-driven, not length-driven.

    The character cap must never win over a sentence boundary: with the cap
    checked first, the walk stopped mid-word and captured ~40 characters of
    unrelated prose as the "term".
    """

    @staticmethod
    def _prefix(text):
        from case_translator.html_renderer import HTMLRenderer
        return HTMLRenderer._persian_prefix(text)

    def test_sentence_boundary_wins_over_length_cap(self):
        """Prose after a full stop is not part of the term."""
        seg = "انجام شد. پست فایبر شیشه‌ای در کانال ریشه سیمان گردید و تکه دند"
        term, consumed = self._prefix(seg)
        # Either no term (correct refusal) or a short plausible one — never a
        # 40-character slice of prose.
        assert term == "" or len(term.split()) <= 5, f"over-captured: {term!r}"

    def test_consumed_length_does_not_clip_preceding_word(self):
        seg = "مقدار "
        term, consumed = self._prefix(seg)
        assert term == "مقدار"
        assert consumed == len(seg)

    def test_multi_word_term_is_kept_whole(self):
        seg = "شکستگی پیچیده تاج-ریشه"
        term, consumed = self._prefix(seg)
        assert term == "شکستگی پیچیده تاج-ریشه"
        assert consumed == len(seg)

    @pytest.mark.parametrize("seg,expected", [
        ("شد و پست فایبر", "پست فایبر"),
        ("و پست فایبر", "پست فایبر"),
        ("با پست فایبر شیشه‌ای", "پست فایبر شیشه‌ای"),
    ])
    def test_leading_function_words_are_dropped(self, seg, expected):
        """
        Function words immediately before the term belong to the prose.

        Trimming stops at the first content word, so a noun like "استفاده" that
        could legitimately start a term is never stripped.
        """
        term, _ = self._prefix(seg)
        assert term == expected

    def test_leading_content_noun_is_kept(self):
        """A noun that could begin a real term must not be trimmed away."""
        term, _ = self._prefix("استفاده از پست فایبر")
        assert term == "استفاده از پست فایبر"

    def test_over_long_run_is_refused(self):
        seg = "یک دو سه چهار شش هفت هشت نه ده یازده دوازده سیزده"
        term, consumed = self._prefix(seg)
        assert term == "" and consumed == 0

    def test_parenthesis_aborts_the_scan(self):
        seg = "متن ) داخل"
        term, _ = self._prefix(seg)
        assert term == ""
