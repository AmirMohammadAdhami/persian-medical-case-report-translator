# Medical Case Report Translator (مترجم کیس ریپورت ها و مقالات تخصصی پزشکی و دندانپزشکی)

A specialized Python 3.13 tool designed to translate English medical case report articles into fluent, professional Persian. It extracts PDF structures, preserves multi-column reading order, retains original figures at their exact logical positions, enforces medical terminology rules (with English original in parentheses), and generates an elegant RTL HTML document.

---

## GitHub repository metadata

**Description** (the one-line "About" field):

```text
Translate English dental & medical case reports into fluent Persian — anatomical names protected, clinical terms clickable with explanations, references kept in English.
```

**Topics** (tags for the GitHub *Topics* field):

```text
persian  farsi  medical-translation  dental  case-report  pdf-parser  pymupdf  rtl  nlp  python  opencode  gemini  openai  medical-nlp  glossary  healthcare
```

**Suggested homepage:** leave empty — this is a CLI tool, not a hosted service.

---

## Key Features

1. **Local PDF & URL Support**:
   - Primary support for local PDF files: `python main.py article.pdf`.
   - Handles remote URLs with graceful error handling (reports clear instructions when a URL requires authentication, Cloudflare verification, or a paywall).
2. **Column-Aware Reading Order Preservation**:
   - Reconstructs multi-column medical literature reading flows.
   - Detects full-width title/abstract bands, two-column sections, and dividers.
   - Merges broken lines, de-hyphenates line breaks, and rejoins paragraphs split by a page break.
   - **Reflows line-per-block PDFs**: exports produced by phone/reader "print to PDF" put every visual line in its own text block. Those are stitched back into real paragraphs before translation, instead of being sent to the model as sentence fragments.
3. **Figure & Image Preservation**:
   - Extracts embedded original images without altering, redrawing, or replacing them.
   - Re-encodes formats a browser cannot render (JPEG2000/JBIG2, CMYK) to PNG automatically.
   - Groups the panels of a multi-part figure into a single figure entry, so it gets one caption and one number.
   - Links figures to their corresponding captions (`Figure X` / `شکل X`) and numbers them in reading order.
   - Saves figures into `output/assets/` (stale figures from a previous run are cleared first; use `--keep-assets` to disable).
4. **Professional Persian Medical Translation**:
   - Pluggable AI architecture supporting **OpenCode** (DeepSeek, Qwen, etc.), Google Gemini, OpenAI, or an offline medical translator.
   - **Anatomical names stay English**: tooth types and position qualifiers (`incisor`, `lateral incisor`, `premolar`, `molar`, `maxillary right lateral incisor`, `mandibular`, `mesial`, …) are never translated, because a wrong rendering risks a real anatomical error (lateral vs central, maxillary vs mandibular).
   - **Clinical terms are explicitly paired**: advanced clinical/procedural terms are written as `[Persian] ([English])` — e.g. `شکستگی پیچیده تاج-ریشه (Complicated Crown-Root Fracture)`, `پست فایبر (Fiber Post)`. Every such pair becomes **clickable** in the output, opening a short Persian explanation.
   - **Journal names, DOIs and volume/issue formatting stay English** in the reference list, preserving academic integrity rather than producing literal translations of journal titles.
   - Common words (`بیمار`, `درمان`, `جراحی`) are not cluttered with English equivalents.
5. **Interactive Glossary**:
   - Detected terms carry `data-term` and open a panel with an authored definition on click or keyboard (Enter/Space); Escape or an outside click closes it.
   - Terminology is primed once per run, so a recurring term keeps the same Persian rendering throughout the document.
   - Every clickable term is guaranteed an explanation: authored notes win, and any term without one gets a definition generated from the Persian rendering actually used in the document.
   - Anatomical names, section labels and institutional affiliations are deliberately excluded from the clickable set.
6. **Bilingual Review Mode** (`--bilingual`):
   - Renders each block with the original English text beside the Persian translation, so a reader can verify accuracy without opening the source PDF.
   - A toggle button in the header switches the original column off and on; the printed stylesheet hides it.
7. **Numeric Fidelity Check**:
   - After each paragraph is translated, the multiset of numbers and the set of measurement units are compared between source and target. Doses, sizes, dates and percentages that were dropped or altered are flagged inline and summarised at the end of the run.
8. **Honest Failure Reporting**:
   - Failed blocks are counted, and after N consecutive failures the run stops instead of hammering a dead endpoint (circuit breaker).
   - A warning banner at the top of the HTML states exactly how many blocks were not translated, and each affected block is badged in place.
   - Non-retryable errors (bad key, connection refused, unknown model) fail immediately instead of retrying three times.
9. **Optional AI Vision Analysis**:
   - `--enable-vision` produces concise Persian descriptions of radiographs, CT/MRI, and clinical photographs without hallucination. Failures are reported, not swallowed.
10. **RTL Persian HTML Output**:
   - Produces responsive, clean HTML (`output/article.html`) with `<html lang="fa" dir="rtl">`.
   - Bi-directional typography handling ensures English terms inside parentheses render correctly without inverted punctuation.
   - No external assets are required to display it: no web-font CDN, no third-party requests (use `--embed-images` to also inline the figures).
   - Print-ready stylesheet support.
11. **Speed & Cost Controls**:
   - **Parallel translation** (`--workers`, default 4) instead of strictly sequential requests.
   - **On-disk translation cache** keyed by model + system prompt + source text, so re-running after changing only the output directory costs nothing.
   - Identical blocks are translated once.

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
Translation Pipeline (case_translator/pipeline.py)
    ├── Cache (case_translator/cache.py)
    └── Numeric verification (case_translator/verification.py)
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

## Which PDFs work, and what structure they need

The parser reads the PDF's own text layer — it does not OCR. That single fact
explains most quality differences between documents.

### Supported (works well)

| Input | Notes |
|---|---|
| Single-column article PDFs | Titles, headings, paragraphs, captions, references all detected. |
| Two-column journal PDFs (Wiley, Elsevier, Springer, MDPI, Hindawi, …) | Column bands are reconstructed, full-width title/abstract handled. |
| Line-per-block exports ("print to PDF" from a phone, reader app, or notes app) | Reflowed into paragraphs and rejoined across page breaks. |
| Articles with ruled tables | Detected and translated cell by cell, then rendered as a real `<table>`. |
| Image-only or scanned PDFs | **Not supported** — there is no text layer to translate (see below). |

### What the parser relies on

1. **A real text layer.** Verify with `python -c "import pymupdf; print(pymupdf.open('f.pdf')[0].get_text()[:200])"`. If that prints nothing, the PDF is a scan and needs OCR first (e.g. `ocrmypdf --language eng input.pdf output.pdf`).
2. **Body text at a consistent font size.** The most common size on the first pages is treated as the body size; titles are ≥1.35× and headings ≥1.15× that. A document with wildly varying sizes may misclassify a heading as a paragraph (it is still translated — just styled as body text).
3. **A "References" (or "Bibliography" / "منابع") heading.** Everything after it, up to the next section heading, is kept in English. Without such a heading the bibliography is treated as prose and translated.
4. **Section headings as separate blocks.** Bold/short/numbered labels are recognised, including Wiley's `3 | Differential Diagnosis` form.
5. **Loopback-safe metadata layout.** Journal mastheads, "OPEN ACCESS" ribbons, correspondence blocks, received/accepted dates, keywords and licence text are detected and excluded — they are never translated.
6. **Explicitly numbered figure captions** (`Figure 1.`, `Fig. 2:`, `شکل ۳.`). Figures without a caption are still extracted and placed in reading order; they just carry no caption text.

### Check a PDF before spending API calls

```bash
python inspect_pdf.py "article.pdf"
```

It prints the text-layer status, body font size, column layout per page, and how
the parser classified every block — including whether the bibliography was
recognised and whether the file needed line reflow. Example output for a
line-per-block export:

```text
  pages: 5   pages with text: 3
  body font size (most common): 25.0
  font sizes seen: [25.0]
  page 1: 1-column   images: 0
  ...
  title: ''
  block types: {'paragraph': 3}
  note: no reference list detected — the bibliography, if any, will be translated.
  reflowed paragraphs (line-per-block input): 3
```

### Known limitations

- **Scanned/photographed articles** have no extractable text. Run OCR first.
- **Figures published as vector art** (not embedded raster images) are not extracted as images.
- **Equations** are extracted as text and translated like prose; check them manually.
- **Tables without ruling lines** are not detected as tables (PyMuPDF's table finder needs borders); their content is translated as ordinary paragraphs.
- **Pages with no text at all** (e.g. a truncated export) produce no blocks, as expected.

---

## Configuration

Copy the example `.env` file and fill in your API key:

```bash
copy .env.example .env
```

> **Only one setup block may be active.** `.env.example` contains two blocks that
> define the same variable names (`OPENCODE_API_KEY`, `OPENCODE_BASE_URL`,
> `OPENCODE_MODEL`). The bridge block is commented out on purpose: if both are
> active the *last* definition wins, so a copied `.env` silently points at a
> local bridge that is not running — and every run then fails to connect.

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
OPENCODE_MODEL="deepseek-v4.1-flash"
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

This starts `opencode serve` and the bridge together and prints the endpoint **and
a generated access token**:

```text
[run] starting bridge on port 4097 (model: mimo-v2.6-flash-free)

  Point the translator at this endpoint:
    OPENCODE_BASE_URL="http://127.0.0.1:4097/v1"
    OPENCODE_MODEL="mimo-v2.6-flash-free"
    OPENCODE_API_KEY="<generated token>"
```

**Setup — terminal 2:** copy those three values into `.env`, then:

```bash
python main.py "sample_case_report.pdf" --provider opencode
```

`OPENCODE_API_KEY` must hold the bridge token. The bridge rejects requests
without it (see *Security* below).

**Running the two halves manually:**

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

### Bridge security

The bridge binds to `127.0.0.1` only, but a loopback port is reachable by **any
web page you have open**. A plain HTML form can POST `text/plain` to it without
a CORS preflight, so without protection a page on any site could spend your
OpenCode quota (a local CSRF; the attacker never reads the response).

The bridge therefore:

- requires `Authorization: Bearer <token>` on every request,
- refuses any request that carries an `Origin` header (i.e. anything a browser
  issues),
- validates the `Host` header against loopback names (blocks DNS rebinding),
- requires `Content-Type: application/json` on POST.

The token is generated at startup and printed, or pinned by setting
`OPENCODE_BRIDGE_TOKEN` before launching. `--insecure-no-auth` disables the
check and prints a warning; it is only reasonable on a single-user machine.

The `opencode serve` password is passed through the environment
(`OPENCODE_SERVER_PASSWORD`), never on a command line, because `argv` is visible
in the process list.

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
python main.py "sample_case_report.pdf" --provider opencode --model deepseek-v4.1-flash

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

### 4. Proofreading with the bilingual view

```bash
python main.py "sample_case_report.pdf" --bilingual
```

Each paragraph, heading and caption is rendered as two columns: the Persian
translation and the original English. A button in the header hides the original
column when you are done checking.

### 5. Speed and cost

```bash
# Translate 8 blocks at a time
python main.py "sample_case_report.pdf" --workers 8

# Re-run without the cache (forces fresh model calls)
python main.py "sample_case_report.pdf" --no-cache

# Keep the cache somewhere specific
python main.py "sample_case_report.pdf" --cache-dir .cache
```

The cache lives in `.translation-cache/` next to the output folder and is keyed
by model + system prompt + source text. Delete the folder to start clean.

### 6. Producing a single portable HTML file

```bash
python main.py "sample_case_report.pdf" --embed-images
```

Figures are inlined as base64 data URIs, so `article.html` can be emailed or
copied anywhere and still show the images.

### 7. CLI UX Progress

```text
[1/6] Downloading / verifying PDF...
[2/6] Extracting document structure & reading order...
[3/6] Extracted 2 figure(s) to assets...
[4/6] Translating medical text into fluent Persian...
[5/6] Processing figure captions & visual analysis...
[6/6] Building RTL Persian HTML document...

Translated 27/27 text block(s).
Cache: 0 hit(s), 30 miss(es).

Done.

Output:
output\article.html
```

When something goes wrong, the summary is explicit rather than silent:

```text
Translated 12/27 text block(s).

WARNING: 15 of 27 block(s) were NOT translated and remain in the source language.
  Translation stopped early after repeated failures (the provider was most likely unreachable).
  - block p1_b4 (page 1): RuntimeError: OpenCode API request failed on model ...
  These blocks are flagged in the HTML output.
```

---

## Output Structure

The output directory contains:

```text
output/
    article.html            # RTL Persian HTML document
    assets/
        figure-1.png        # Original extracted figure 1
        figure-2.jpeg       # Original extracted figure 2
        ...
```

Open `output/article.html` in any web browser to view the translated article.

The page loads **no external resources**. It uses a system Persian font stack
(Vazirmatn / Vazir / IRANSans / Tahoma), so it renders offline and does not
report the reader's IP to a font CDN. Figures are referenced from `assets/`
unless you pass `--embed-images`.

---

## Running Tests

Execute the comprehensive test suite with `pytest`:

```bash
.\.venv\Scripts\python.exe -m pytest -v
```

`tests/test_regressions.py` covers the previously reported defects (dead
glossary cache, silent translation failures, dead glossary notes, duplicate
`.env` keys, reference-section detection, line reflow, author/byline handling,
downloader resource leaks, bridge authentication).

`sample_case_report.pdf` is a generated fixture; recreate it with
`python generate_sample_pdf.py` if it is missing.
