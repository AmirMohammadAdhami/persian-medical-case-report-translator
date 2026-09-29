"""
Unit tests for translators and medical terminology preservation.
"""

import pytest
from case_translator.translators import MockTranslator, get_translator
from case_translator.translators.base import is_non_translation_response


def test_mock_translator_headings():
    t = MockTranslator()
    assert t.translate_text("Introduction") == "۱. مقدمه"
    assert t.translate_text("Abstract") == "چکیده"
    assert t.translate_text("Discussion") == "بحث و بررسی"
    assert t.translate_text("Conclusion") == "نتیجه‌گیری"


def test_mock_translator_medical_terms_parentheses():
    t = MockTranslator()
    # First occurrence introduces English term in parentheses
    text1 = "Examination showed a radiolucent lesion in the mandibular area."
    translated1 = t.translate_text(text1)
    assert "(radiolucent)" in translated1 or "(radiolucent lesion)" in translated1

    # Second occurrence uses Persian term without re-introducing English in parentheses
    text2 = "No recurrent radiolucent lesion was detected at follow-up."
    translated2 = t.translate_text(text2)
    # The term was already seen, so it shouldn't introduce "(radiolucent)" again in the second text
    assert "(radiolucent" not in translated2


def test_caption_translation():
    t = MockTranslator()
    cap = "Figure 1. Preoperative intraoral photograph showing crown fracture."
    translated = t.translate_caption(cap)
    assert "شکل ۱" in translated
    assert "(preoperative intraoral view)" in translated or "داخل‌دهانی" in translated


def test_vision_description():
    t = MockTranslator(enable_vision=True)
    desc = t.describe_image("some_path.png")
    assert desc is not None
    assert "شکستگی" in desc or "بالینی" in desc


def test_get_translator_factory():
    t_mock = get_translator("mock")
    assert isinstance(t_mock, MockTranslator)

    with pytest.raises(ValueError):
        get_translator("nonexistent_provider")


class TestNonTranslationResponseGuard:
    """
    Chat models sometimes answer a bare section heading with a conversational
    request for input instead of a translation. That reply must be rejected so
    the pipeline can fall back to the source text.
    """

    def test_detects_english_meta_reply_for_bare_heading(self):
        assert is_non_translation_response(
            "Conclusion",
            "Please provide the English text you'd like translated.",
        )

    def test_detects_persian_meta_reply_for_bare_heading(self):
        assert is_non_translation_response(
            "References",
            "متن مرجع (References) مورد نظر را ارسال کنید تا ترجمه کنم.",
        )

    def test_accepts_real_heading_translation(self):
        assert not is_non_translation_response("Discussion", "بحث و بررسی")
        assert not is_non_translation_response("Conclusion", "نتیجه‌گیری")

    def test_accepts_long_paragraph_containing_marker_phrase(self):
        # A genuine paragraph that happens to contain a marker phrase must be
        # kept, because the source is long and sentence-like.
        source = (
            "The authors state that readers should provide the necessary "
            "clinical details before drawing conclusions from this report."
        )
        assert not is_non_translation_response(
            source, "نویسندگان بیان می‌کنند که خوانندگان باید این را ارسال کنند."
        )

    def test_empty_translation_is_not_flagged(self):
        assert not is_non_translation_response("Conclusion", "")



class TestBaseUrlWiring:
    """
    --base-url must reach every provider, not just opencode.

    Regression: the flag was accepted by the CLI but silently dropped for the
    gemini and openai providers, so pointing them at a proxy did nothing.
    """

    def test_openai_uses_custom_base_url(self):
        t = get_translator("openai", api_key="sk-test", base_url="https://proxy.example/v1")
        assert t.base_url == "https://proxy.example/v1"
        assert str(t.client.base_url).startswith("https://proxy.example/v1")

    def test_gemini_uses_custom_base_url(self):
        t = get_translator("gemini", api_key="AIza-test", base_url="https://gemini-proxy.example")
        assert t.base_url == "https://gemini-proxy.example"

    def test_defaults_have_no_base_url(self):
        assert get_translator("openai", api_key="sk-test").base_url is None
        assert get_translator("gemini", api_key="AIza-test").base_url is None

    def test_trailing_slash_is_normalised(self):
        t = get_translator("openai", api_key="sk-t", base_url="https://proxy.example/v1/")
        assert t.base_url == "https://proxy.example/v1"
