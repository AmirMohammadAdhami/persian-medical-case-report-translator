# Medical Case Report Translator (مترجم گزارش‌های موردی پزشکی)

A specialized Python 3.13 tool designed to translate English medical case report articles into fluent, professional Persian. It extracts PDF structures, preserves multi-column reading order, retains original figures at their exact logical positions, enforces medical terminology rules (with English original in parentheses), and generates an elegant RTL HTML document.

---

## Key Features

1. **Local PDF & URL Support**:
   - Primary support for local PDF files: `python main.py article.pdf`.
   - Handles remote URLs with graceful error handling (reports clear instructions when a URL requires authentication, Cloudflare verification, or a paywall).
2. **Column-Aware Reading Order Preservation**:
   - Reconstructs multi-column medical literature reading flows.
   - Detects full-width title/abstract bands, two-column sections, and dividers.
   - Merges broken lines and de-hyphenates line breaks.
3. **Figure & Image Preservation**:
   - Extracts embedded original images without altering, redrawing, or replacing them.
   - Saves figures into `output/assets/`.
   - Links figures to their corresponding captions (`Figure X` / `شکل X`).
   - Keeps images in their exact position in the reading flow (e.g. `Paragraph -> Figure 1 -> Caption 1 -> Paragraph`).
4. **Professional Persian Medical Translation**:
   - Pluggable AI architecture supporting **OpenCode** (DeepSeek, Qwen, etc.), Google Gemini, OpenAI, or an offline medical translator.
   - **Anatomical names stay English**: tooth types and position qualifiers (`incisor`, `lateral incisor`, `premolar`, `molar`, `maxillary right lateral incisor`, `mandibular`, `mesial`, …) are never translated, because a wrong rendering risks a real anatomical error (lateral vs central, maxillary vs mandibular).
   - **Clinical terms are explicitly paired**: advanced clinical/procedural terms are written as `[Persian] ([English])` — e.g. `شکستگی پیچیده تاج-ریشه (Complicated Crown-Root Fracture)`, `پست فایبر (Fiber Post)`. Every such pair becomes **clickable** in the output, opening a short Persian explanation.
   - **Journal names, DOIs and volume/issue formatting stay English** in the reference list, preserving academic integrity rather than producing literal translations of journal titles.
   - Common words (`بیمار`, `درمان`, `جراحی`) are not cluttered with English equivalents.
5. **Interactive Glossary**:
   - Detected terms carry `data-term` and open a panel with an authored definition on click or keyboard (Enter/Space); Escape or an outside click closes it.
   - Terminology is primed once per run, so a recurring term keeps the same Persian rendering throughout the document.
   - Anatomical names, section labels and institutional affiliations are deliberately excluded from the clickable set.
6. **Optional AI Vision Analysis**:
   - `--enable-vision` produces concise Persian descriptions of radiographs, CT/MRI, and clinical photographs without hallucination.
7. **RTL Persian HTML Output**:
   - Produces responsive, clean HTML (`output/article.html`) with `<html lang="fa" dir="rtl">`.
   - Bi-directional typography handling ensures English terms inside parentheses render correctly without inverted punctuation.
   - Print-ready stylesheet support.

---

## Architecture

The project enforces separation of concerns across dedicated modules:

```text
Downloader (case_translator/downloader.py)
    ↓
PDF Parser (case_translator/pdf_parser.py)
    ↓
Document Model (case_translator/models.py)
    ↓
Figure Processing (case_translator/figure_processor.py)
    ↓
Glossary (case_translator/glossary.py)   ← term list, anatomical guards, definitions
    ↓
Translation Pipeline (case_translator/translators/)
    ↓
HTML Renderer (case_translator/html_renderer.py)   ← clickable terms + explanation panel
```

---

## Installation (Python 3.13)

1. Clone or navigate to the repository directory:
   ```bash
   cd "case report translator"
   ```

2. Create and activate a Python 3.13 virtual environment:
   ```bash
   py -3.13 -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Copy the example `.env` file and fill in your API key:

```bash
copy .env.example .env
```

### OpenCode Zen (Recommended)

OpenCode exposes an OpenAI-compatible gateway to models like DeepSeek, GLM, Kimi, and MiMo.

> **Important — Zen and Go are separate gateways with separate model catalogues.**
>
> | Gateway | Base URL | Notes |
> |---|---|---|
> | **Zen** (pay-as-you-go) | `https://opencode.ai/zen/v1` | Free `*-free` models live here |
> | **Go** (subscription) | `https://opencode.ai/zen/go/v1` | Different catalogue, no `-free` variants |
>
> A model ID valid on one gateway returns `HTTP 400 / inference_failed` on the other.

```env
# Zen gateway (default)
OPENCODE_API_KEY="sk-your-opencode-api-key"
OPENCODE_BASE_URL="https://opencode.ai/zen/v1"
OPENCODE_MODEL="deepseek-v4-flash-free"
```

| Variable | Description | Default |
|---|---|---|
| `OPENCODE_API_KEY` | Your OpenCode API key (starts with `sk-`) | *required* |
| `OPENCODE_BASE_URL` | Gateway endpoint: `https://opencode.ai/zen/v1` (Zen) or `https://opencode.ai/zen/go/v1` (Go) | `https://opencode.ai/zen/v1` |
| `OPENCODE_MODEL` | Model ID — must exist on the gateway above | `deepseek-v4.1-flash` |

#### Finding a valid model ID

Model IDs rotate frequently. Rather than guessing, ask your key what it can use:

```bash
python main.py --list-models
```

This prints the free and billable models available on the configured gateway.

**Free Zen models** (no billing required):

`deepseek-v4-flash-free`, `mimo-v2.6-flash-free`, `mimo-v2.5-free`,
`longcat-2.5-preview-free`, `ling-3.0-flash-fin-free`, `nemotron-3-ultra-free`,
`nemotron-3.5-lightning-free`, `space-bunny-free`, `jev-1.13-free`

**Billable Zen models** (good for translation):

`deepseek-v4.1-flash`, `deepseek-v4-flash`, `deepseek-v4-pro`, `glm-5.3`, `kimi-k3`, `qwen3.8-max`

---

### Using free Zen models (bridge required)

> **Direct API calls to free Zen models return `HTTP 403`.**
>
> Since 2026-09-16 the Zen free tier validates an `x-opencode-session` header
> that only the official OpenCode client sets. Third-party scripts calling
> `opencode.ai/zen/v1` directly are rejected, regardless of how valid the API
> key is.
>
> This project does **not** forge that header. Instead it routes through the
> official client using a small local bridge.

**How it works:**

```text
case_translator (OpenAI SDK)
    -> http://127.0.0.1:4097/v1          <- local bridge (opencode_bridge.py)
        -> opencode CLI server           <- official client, authenticates properly
            -> Zen free model
```

**Setup — terminal 1:**

```bash
python run_opencode_bridge.py
```

This starts `opencode serve` and the bridge together, and prints the endpoint to use.
(You need the official CLI authenticated once with `opencode auth login`.)

**Setup — terminal 2:**

```bash
python main.py "sample_case_report.pdf" --provider opencode
```

Your `.env` for this mode:

```env
OPENCODE_API_KEY="local-bridge"                  # placeholder; the CLI sends its own
OPENCODE_BASE_URL="http://127.0.0.1:4097/v1"
OPENCODE_MODEL="mimo-v2.6-flash-free"
```

**Running the two halves manually** (if you prefer separate processes):

```bash
# terminal 1
opencode serve --port 4096

# terminal 2
python -m case_translator.opencode_bridge --port 4097 --model mimo-v2.6-flash-free

# terminal 3
python main.py "sample_case_report.pdf" --provider opencode
```

**Prefer no bridge?** Use a billable model and call Zen directly — those are
plain key-authenticated requests with no session requirement:

```bash
python main.py "sample_case_report.pdf" --provider opencode --model deepseek-v4.1-flash
```

**Cleanup.** Stopping the launcher (Ctrl-C, or closing the terminal) stops both
halves. On Windows both child processes are also bound to a job object, so they
are killed even if the launcher is force-terminated — no orphaned processes left
holding ports 4096/4097.

### Gemini Model Note

`gemini-2.5-flash` has been retired for new API users and returns `HTTP 404 NOT_FOUND`. The built-in default is therefore **`gemini-3.8-flash`**. You can override it with `--model` or the `GEMINI_MODEL` environment variable.

### Other Providers (Optional)

```env
# Google Gemini
GEMINI_API_KEY="your-gemini-api-key"
# Optional: custom endpoint / proxy (defaults to the official Google endpoint)
GEMINI_BASE_URL=""

# OpenAI
OPENAI_API_KEY="your-openai-api-key"
# Optional: any OpenAI-compatible endpoint (proxy, Azure gateway, Ollama, ...)
OPENAI_BASE_URL=""
```

Both `*_BASE_URL` variables are optional and can also be set per-run with `--base-url`, which takes precedence over the environment. All three providers honour it.

If no API key is configured, the system automatically uses the built-in offline medical translator (`--provider mock`), allowing full local testing and validation.

---

## Usage

### 1. Using OpenCode Zen (Recommended)

```bash
# Auto-detected if OPENCODE_API_KEY is set in .env
python main.py "sample_case_report.pdf"

# Explicit provider + model selection
python main.py "sample_case_report.pdf" --provider opencode --model deepseek-v4-flash-free

# See which models your key can use
python main.py --list-models

# Using the Go subscription gateway instead of Zen
python main.py "sample_case_report.pdf" --provider opencode --base-url "https://opencode.ai/zen/go/v1" --model glm-5.3

# With a custom base URL
python main.py "sample_case_report.pdf" --provider opencode --api-key "sk-..." --model "kimi-k3" --base-url "https://opencode.ai/zen/v1"

# With AI Vision enabled
python main.py "sample_case_report.pdf" --provider opencode --model deepseek-v4.1-flash --enable-vision
```

### 2. Using Other Providers

```bash
# Google Gemini
python main.py "sample_case_report.pdf" --provider gemini --model gemini-3.8-flash

# OpenAI
python main.py "sample_case_report.pdf" --provider openai --model gpt-4o-mini
```

### 3. Translating from an Article URL

```bash
python main.py "https://example.com/open-access-case-report.pdf"
```

*Note: If the article is hosted behind a paywall, institutional login, or Cloudflare bot challenge (such as Wiley or Nature direct links), the tool will gracefully detect the restriction and prompt you to download the PDF manually and pass the local file path.*

### 4. CLI UX Progress

```text
[1/6] Downloading / verifying PDF...
[2/6] Extracting document structure & reading order...
[3/6] Extracted 2 figure(s) to assets...
[4/6] Translating medical text into fluent Persian...
[5/6] Processing figure captions & visual analysis...
[6/6] Building RTL Persian HTML document...

Done.

Output:
output\article.html
```

---

## Output Structure

The output directory contains:

```text
output/
    article.html            # Self-contained RTL Persian HTML document
    assets/
        figure-1.png        # Original extracted figure 1
        figure-2.png        # Original extracted figure 2
        ...
```

Open `output/article.html` in any web browser to view the translated article.

---

## Running Tests

Execute the comprehensive test suite with `pytest`:

```bash
.\.venv\Scripts\python.exe -m pytest -v
```
