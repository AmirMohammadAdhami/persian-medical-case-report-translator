"""
Downloader module for obtaining medical case report PDFs from URLs or local paths.
Handles HTTP errors, access restrictions, paywalls, and local file validation gracefully.
"""

import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests


class DownloaderError(Exception):
    """Base exception for downloader failures."""
    pass


class InaccessibleArticleError(DownloaderError):
    """Raised when an article cannot be downloaded due to access restrictions or paywalls."""
    pass


class InvalidPDFError(DownloaderError):
    """Raised when the fetched or provided file is not a valid PDF."""
    pass


class BaseDownloader(ABC):
    """Abstract interface for retrieving a PDF file."""

    @abstractmethod
    def get_pdf(self, source: str, destination_dir: Optional[Path] = None) -> Path:
        """
        Retrieves the PDF and returns the local Path to it.
        """
        pass


class ArticleDownloader(BaseDownloader):
    """
    Downloader that resolves both local PDF files and remote URLs.
    Supports legitimate direct downloads while respecting anti-bot / paywall protections.
    """

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def is_url(self, source: str) -> bool:
        """Checks if the source is a valid web URL."""
        try:
            parsed = urlparse(source.strip())
            return parsed.scheme in ("http", "https")
        except Exception:
            return False

    def validate_pdf_content(self, file_path: Path) -> None:
        """Validates that the file begins with the PDF magic header."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {file_path}")

        if file_path.stat().st_size < 100:
            raise InvalidPDFError(f"File at {file_path} is too small to be a valid PDF.")

        with open(file_path, "rb") as f:
            header = f.read(1024)
            if b"%PDF-" not in header:
                # Check if it looks like an HTML error page
                if b"<html" in header.lower() or b"<!doctype html" in header.lower():
                    raise InaccessibleArticleError(
                        f"The file '{file_path.name}' is an HTML web page rather than a PDF document. "
                        "This typically indicates a login portal, Cloudflare challenge, or paywall page."
                    )
                raise InvalidPDFError(f"File at {file_path} lacks a valid %PDF header.")

    def get_pdf(self, source: str, destination_dir: Optional[Path] = None) -> Path:
        """
        Accepts a local file path or a URL.
        If local, validates and returns the path.
        If URL, downloads the PDF safely or raises InaccessibleArticleError.
        """
        source = source.strip().strip("'\"")

        if not self.is_url(source):
            # Local file path
            local_path = Path(source).expanduser().resolve()
            if not local_path.exists():
                raise FileNotFoundError(
                    f"File not found: '{source}'. Please verify the path or provide a valid URL."
                )
            self.validate_pdf_content(local_path)
            return local_path

        # Handle Remote URL
        return self._download_url(source, destination_dir)

    def _download_url(self, url: str, destination_dir: Optional[Path] = None) -> Path:
        """Downloads a PDF from a remote URL."""
        dest_dir = destination_dir or Path(tempfile.gettempdir()) / "case_translator"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Derive a filename from URL or DOI
        filename = self._generate_filename(url)
        target_path = dest_dir / filename

        try:
            response = self.session.get(url, stream=True, timeout=self.timeout_seconds)
        except requests.exceptions.SSLError as e:
            raise InaccessibleArticleError(
                f"SSL certificate verification failed for '{url}': {e}"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise DownloaderError(
                f"Failed to connect to '{url}'. Please check your network connection: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise DownloaderError(
                f"Connection to '{url}' timed out after {self.timeout_seconds} seconds."
            ) from e
        except requests.exceptions.RequestException as e:
            raise DownloaderError(f"HTTP request error: {e}") from e

        # Handle HTTP status codes
        if response.status_code in (401, 403):
            raise InaccessibleArticleError(
                f"HTTP {response.status_code} Forbidden/Unauthorized accessing '{url}'.\n"
                "The publisher or host requires institutional authentication, anti-bot verification, or a subscription paywall.\n"
                "Please download the PDF manually in your browser, then pass the file path:\n"
                "  python main.py <path_to_downloaded_pdf>"
            )
        elif response.status_code == 404:
            raise DownloaderError(f"Article not found (HTTP 404) at '{url}'.")
        elif response.status_code >= 400:
            raise DownloaderError(f"HTTP error {response.status_code} occurred while downloading from '{url}'.")

        # Check content type and first chunk
        content_type = response.headers.get("Content-Type", "").lower()
        first_chunk = next(response.iter_content(chunk_size=1024), b"")

        if b"<html" in first_chunk.lower() or "text/html" in content_type:
            raise InaccessibleArticleError(
                f"The server at '{url}' returned an HTML web page instead of a PDF.\n"
                "This usually indicates a Cloudflare verification challenge, paywall, or landing page.\n"
                "Please download the PDF directly in your browser, then pass the local file:\n"
                "  python main.py <path_to_downloaded_pdf>"
            )

        if b"%PDF-" not in first_chunk:
            raise InvalidPDFError(
                f"The downloaded resource from '{url}' does not appear to be a valid PDF file."
            )

        # Stream save to target path
        with open(target_path, "wb") as f:
            f.write(first_chunk)
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        self.validate_pdf_content(target_path)
        return target_path

    def _generate_filename(self, url: str) -> str:
        """Derives a clean filename from the URL."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        name = path.split("/")[-1] if path else "article.pdf"

        # Sanitize filename
        name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"
        return name
