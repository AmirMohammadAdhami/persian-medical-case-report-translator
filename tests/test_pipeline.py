"""
End-to-end integration test for CaseReportPipeline.
"""

from pathlib import Path
from case_translator.pipeline import CaseReportPipeline
from case_translator.translators import MockTranslator


def test_pipeline_end_to_end(tmp_path):
    pdf_path = Path("sample_case_report.pdf")
    assert pdf_path.exists(), "sample_case_report.pdf must exist for integration test"

    progress_log = []

    def log_progress(step, total, msg):
        progress_log.append((step, total, msg))

    translator = MockTranslator(enable_vision=True)
    pipeline = CaseReportPipeline(
        translator=translator,
        output_dir=tmp_path / "output",
        progress_callback=log_progress
    )

    result_html = pipeline.run(str(pdf_path))
    assert result_html.exists()
    assert result_html.name == "article.html"

    # Verify that all 6 steps executed
    assert len(progress_log) == 6
    assert [p[0] for p in progress_log] == [1, 2, 3, 4, 5, 6]

    # Verify assets were copied
    assets_dir = tmp_path / "output" / "assets"
    assert assets_dir.exists()
    figures = list(assets_dir.glob("figure-*.*"))
    assert len(figures) == 2

    # Verify HTML contents
    html_text = result_html.read_text(encoding="utf-8")
    assert '<html lang="fa" dir="rtl">' in html_text
    assert 'شکل ۱' in html_text or 'شکل 1' in html_text
    assert 'شکل ۲' in html_text or 'شکل 2' in html_text
    assert 'assets/figure-1' in html_text
    assert 'assets/figure-2' in html_text
    assert 'AI Vision' in html_text
