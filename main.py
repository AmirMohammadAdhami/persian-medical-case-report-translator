#!/usr/bin/env python3
"""
Medical Case Report Translator CLI
Translates English medical case reports into fluent Persian RTL HTML,
preserving reading order and relative figure positions.
"""

import argparse
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

    args = parser.parse_args()

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
        progress_callback=print_progress
    )

    try:
        output_html = pipeline.run(args.source)
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

    except Exception as e:
        print(f"\nExecution Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
