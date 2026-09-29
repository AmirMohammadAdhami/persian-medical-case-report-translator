"""
Unit tests for the Downloader module.
"""

from pathlib import Path
import pytest
from case_translator.downloader import (
    ArticleDownloader,
    DownloaderError,
    InaccessibleArticleError,
    InvalidPDFError,
)


def test_is_url():
    downloader = ArticleDownloader()
    assert downloader.is_url("https://onlinelibrary.wiley.com/doi/epdf/10.1155/2011/401678") is True
    assert downloader.is_url("http://example.com/paper.pdf") is True
    assert downloader.is_url("sample_case_report.pdf") is False
    assert downloader.is_url("C:\\path\\to\\file.pdf") is False


def test_local_file_not_found():
    downloader = ArticleDownloader()
    with pytest.raises(FileNotFoundError):
        downloader.get_pdf("non_existent_file_12345.pdf")


def test_invalid_pdf_content(tmp_path):
    invalid_file = tmp_path / "not_a_pdf.pdf"
    invalid_file.write_text("Hello, this is not a valid PDF file content.", encoding="utf-8")

    downloader = ArticleDownloader()
    with pytest.raises(InvalidPDFError):
        downloader.get_pdf(str(invalid_file))


def test_html_error_page_detected_as_inaccessible(tmp_path):
    html_file = tmp_path / "paywall.pdf"
    html_file.write_text("<!DOCTYPE html><html><body>Access Denied Cloudflare</body></html>" * 5, encoding="utf-8")

    downloader = ArticleDownloader()
    with pytest.raises(InaccessibleArticleError):
        downloader.get_pdf(str(html_file))


def test_valid_pdf_file():
    pdf_path = Path("sample_case_report.pdf")
    if pdf_path.exists():
        downloader = ArticleDownloader()
        resolved = downloader.get_pdf(str(pdf_path))
        assert resolved.exists()
        assert resolved.name == "sample_case_report.pdf"
