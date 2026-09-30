#!/usr/bin/env python3
"""
Medical Case Report Translator CLI
Translates English medical case reports into fluent Persian RTL HTML,
preserving reading order and relative figure positions.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from case_translator.downloader import DownloaderError, InaccessibleArticleError, InvalidPDFError
from case_translator.pipeline import CaseReportPipeline
from case_translator.translators import (
    ZEN_FREE_MODELS,
    get_translator,
    list_opencode_models,
)


def print_progress(step: int, total: int, message: str) -> None:
    """Prints clear, human-friendly step progress."""
    print(f"[{step}/{total}] {message}")


def _is_local_endpoint(base_url: str) -> bool:
    return any(host in (base_url or "") for host in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]"))


def _probe_local_endpoint(base_url: str, api_key: str, timeout: float = 2.0):
    """
    Cheap reachability check for a local (bridge) endpoint.

    Returns (reachable, status_code). A 401/403 means the bridge is up but the
    token does not match, which is a different problem from "not running".
    """
    import requests

    url = base_url.rstrip("/") + "/models"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        return True, response.status_code
    except Exception:
        return False, None


def _fa_digits(value) -> str:
    return str(value).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def _print_summary(report: dict) -> None:
    """Prints the end-of-run translation report."""
    if not report:
        return

    total = report.get("total", 0)
    translated = report.get("translated", 0)
    failed = report.get("failed", 0)
    skipped = report.get("skipped", 0)
    kept = report.get("kept_original", 0)

    print(f"\nTranslated {translated}/{total} text block(s).")
    if kept:
        print(f"{kept} bibliography entry(ies) were intentionally kept in English.")
    if failed or skipped:
        print(
            f"\nWARNING: {failed + skipped} of {total} block(s) were NOT translated "
            "and remain in the source language.",
            file=sys.stderr,
        )
        if report.get("circuit_breaker_tripped"):
            print(
                "  Translation stopped early after repeated failures "
                "(the provider was most likely unreachable).",
                file=sys.stderr,
            )
        for failure in (report.get("failures") or [])[:3]:
            print(f"  - block {failure['id']} (page {failure['page']}): {failure['error'][:120]}", file=sys.stderr)
        if len(report.get("failures") or []) > 3:
            print(f"  ... and {len(report['failures']) - 3} more.", file=sys.stderr)
        print("  These blocks are flagged in the HTML output.", file=sys.stderr)

    numeric = report.get("numeric_warnings") or []
    if numeric:
        print(f"\nNote: {len(numeric)} block(s) show numeric/unit drift vs. the source;", file=sys.stderr)
        print("  they are flagged in the HTML for manual review.", file=sys.stderr)

    cache = report.get("cache") or {}
    if cache.get("enabled"):
        print(f"Cache: {cache.get('hits', 0)} hit(s), {cache.get('misses', 0)} miss(es).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate medical case reports from PDF/URL into fluent Persian HTML."
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Local PDF file path or article URL (e.g. 'article.pdf' or 'https://...')"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List the model IDs available on the OpenCode gateway for your API key, then exit"
    )
    parser.add_argument(
        "--provider",
        choices=["opencode", "gemini", "openai", "mock"],
        default=None,
        help="AI translation provider: 'opencode' (OpenCode Zen / Go gateway: DeepSeek, GLM, Kimi, MiMo), 'gemini', 'openai', or 'mock'"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the selected AI provider (can also be set via OPENCODE_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use (e.g. 'deepseek-v4-flash-free', 'glm-5.3', 'gemini-3.8-flash', 'gpt-4o-mini'). Run --list-models to see OpenCode options"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom API base URL. OpenCode default: 'https://opencode.ai/zen/v1' (Zen); use 'https://opencode.ai/zen/go/v1' for the Go plan (or via OPENCODE_BASE_URL)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory to store the generated HTML and assets (default: 'output')"
    )
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Enable AI visual analysis of radiographs and clinical figures"
    )

    # --- review / output options ---
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Render each block with the original English text beside the Persian translation (for proofreading)"
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Inline figures as base64 data URIs so the HTML works as a single portable file"
    )
    parser.add_argument(
        "--keep-assets",
        action="store_true",
        help="Do not delete figures left over from a previous run in the assets folder"
    )

    # --- performance options ---
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="How many translation requests to run in parallel (default: 4)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the on-disk translation cache"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for the translation cache (default: .translation-cache next to the output folder)"
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=CaseReportPipeline.DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help="Stop translating after this many consecutive failures (default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress and warnings"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # --list-models: inspect the OpenCode gateway catalogue and exit.
    if args.list_models:
        base_url = args.base_url or os.getenv("OPENCODE_BASE_URL")
        try:
            models = list_opencode_models(api_key=args.api_key, base_url=base_url)
        except ValueError as e:
            print(f"\nConfiguration Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"\nCould not list models: {e}", file=sys.stderr)
            return 1

        endpoint = base_url or "https://opencode.ai/zen/v1 (default)"
        print(f"\nModel IDs available on: {endpoint}\n")
        free = [m for m in models if m.endswith("-free")]
        paid = [m for m in models if not m.endswith("-free")]

        print(f"Free models ({len(free)}):")
        for m in free:
            print(f"  {m}")
        print(f"\nAll models ({len(paid)} billable):")
        for m in paid:
            print(f"  {m}")
        print(
            "\nNote: Zen (zen/v1) and Go (zen/go/v1) have separate catalogues.\n"
            "Free '*--free' IDs are available on Zen only."
        )
        return 0

    if not args.source:
        parser.error("the following arguments are required: source (or use --list-models)")

    # Determine provider automatically if not specified
    provider = args.provider
    if not provider:
        if os.getenv("OPENCODE_API_KEY"):
            provider = "opencode"
        elif args.api_key or os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "mock"
            print("Notice: No API key found in environment; running with high-accuracy offline medical translator.")

    # A local bridge endpoint is free to probe, and doing so turns "the whole
    # run silently returns English" into an immediate, actionable message.
    if provider == "opencode":
        resolved_base = args.base_url or os.getenv("OPENCODE_BASE_URL") or ""
        resolved_key = args.api_key or os.getenv("OPENCODE_API_KEY") or ""
        if _is_local_endpoint(resolved_base):
            reachable, status = _probe_local_endpoint(resolved_base, resolved_key)
            if not reachable:
                print(
                    f"\nConfiguration Error: the configured OpenCode endpoint is not responding.\n"
                    f"  OPENCODE_BASE_URL={resolved_base}\n\n"
                    "This endpoint is the local bridge. Start it first in another terminal:\n"
                    "  python run_opencode_bridge.py\n\n"
                    "Or switch to a direct (paid) gateway in .env:\n"
                    '  OPENCODE_BASE_URL="https://opencode.ai/zen/v1"\n'
                    '  OPENCODE_API_KEY="sk-your-real-key"\n'
                    '  OPENCODE_MODEL="deepseek-v4.1-flash"\n',
                    file=sys.stderr,
                )
                return 1
            if status in (401, 403):
                print(
                    f"\nConfiguration Error: the bridge at {resolved_base} rejected the token.\n\n"
                    "The bridge requires the token it printed at startup. Copy it into .env:\n"
                    '  OPENCODE_API_KEY="<token printed by run_opencode_bridge.py>"\n\n'
                    "To use a fixed token instead, set OPENCODE_BRIDGE_TOKEN before starting the bridge.\n",
                    file=sys.stderr,
                )
                return 1

    output_dir = Path(args.output_dir)

    try:
        translator = get_translator(
            provider=provider,
            api_key=args.api_key,
            model_name=args.model,
            base_url=args.base_url,
            enable_vision=args.enable_vision
        )
    except Exception as e:
        print(f"\nConfiguration Error: {e}", file=sys.stderr)
        return 1

    pipeline = CaseReportPipeline(
        translator=translator,
        output_dir=output_dir,
        progress_callback=print_progress,
        workers=args.workers,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        use_cache=not args.no_cache,
        bilingual=args.bilingual,
        embed_images=args.embed_images,
        clear_assets=not args.keep_assets,
        max_consecutive_failures=args.max_consecutive_failures,
    )

    try:
        output_html = pipeline.run(args.source)
        _print_summary(getattr(pipeline, "last_report", {}) or {})
        print("\nDone.")
        print("\nOutput:")
        print(output_html)
        return 0

    except InaccessibleArticleError as e:
        print(f"\nAccess Error:\n{e}", file=sys.stderr)
        return 2

    except InvalidPDFError as e:
        print(f"\nPDF Error:\n{e}", file=sys.stderr)
        return 3

    except FileNotFoundError as e:
        print(f"\nFile Not Found:\n{e}", file=sys.stderr)
        return 4

    except DownloaderError as e:
        print(f"\nDownload Error:\n{e}", file=sys.stderr)
        return 5

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"\nExecution Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
