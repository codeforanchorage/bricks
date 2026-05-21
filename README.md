# Brick OCR Test Pipeline

Test harness for cataloguing the ~13,000 commemorative paver bricks being
removed from Town Square Park (Municipality of Anchorage). It runs nadir
(downward-facing) brick photos through four text-extraction methods and
writes a comparison so we can judge which approach to scale up.

**Why:** the bricks are being lifted and moved by section to a warehouse,
where people will come to reclaim their own. OCR-reading every brick and
matching it against the Municipality's official brick list lets us tell
someone up front whether their brick survived the move — so they don't have
to search a whole pallet.

| Method | Engine | Runs on |
|--------|--------|---------|
| 1 | **PaddleOCR** | local OCR, GPU if available (CPU fallback) |
| 2 | **Claude Sonnet 4.6** | `claude-sonnet-4-6` via the Anthropic API |
| 3 | **Gemini 2.5 Pro** | `gemini-2.5-pro` via the Google Gemini API |
| 4 | **Gemini 2.5 Flash** | `gemini-2.5-flash` via the Google Gemini API |

PaddleOCR detects text **line by line**; the pipeline then groups those
detections back into bricks by spatial layout (see `group_bricks.py`), so both
methods produce **one row per brick** for an apples-to-apples comparison. Pass
`--no-group` to inspect PaddleOCR's raw per-line detections instead.

## Requirements

- **Python 3.11, 3.12, or 3.13.**
  > ⚠️ **Not Python 3.14.** `paddlepaddle` (PaddleOCR's runtime) publishes no
  > wheels for 3.14, so the PaddleOCR method cannot run there. This machine
  > currently has only Python 3.14 installed — see Setup step 1.
- An Anthropic API key (Claude methods) and a Google Gemini API key.
- Optional: an NVIDIA GPU for PaddleOCR acceleration.

## Setup

### 1. Python environment

Install a supported Python and create a virtual environment:

```powershell
winget install Python.Python.3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Dependencies

```powershell
pip install -r requirements.txt
```

That installs the LLM methods (`anthropic`, `google-genai`, `Pillow`) and
`paddleocr`. PaddleOCR also needs the `paddlepaddle` runtime, installed
**separately** to match your hardware — pick one:

```powershell
# CPU build — always works, fine for small test batches:
pip install paddlepaddle

# GPU build (CUDA) — get the exact command for your CUDA version from
# https://www.paddlepaddle.org.cn/en/install/quick , e.g.:
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

> **GPU note — RTX 5060 (Blackwell).** This machine's GPU is an NVIDIA RTX 5060
> Laptop (Blackwell, compute capability `sm_120`). PaddlePaddle's published GPU
> wheels target CUDA 12.6, which predates Blackwell support, so GPU init may
> fail. The pipeline **automatically falls back to CPU** in that case — fine for
> a 2-image test. Pass `--no-gpu` to skip the GPU attempt entirely.

### 3. API keys

The two LLM methods need API keys:

- **Anthropic** (Claude Sonnet) — from https://console.anthropic.com/
- **Google Gemini** (Gemini Pro & Flash) — from https://aistudio.google.com/

**Recommended — `.env` file.** Put the keys in `brick-ocr/.env`, one per line:

```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

`pipeline.py` loads `.env` automatically on startup. The file is git-ignored,
stays out of shell history, and is separate from Claude Code's own
`ANTHROPIC_API_KEY` auth.

**Alternative — environment variables.** A real environment variable wins: if
a key is already set in the environment, `.env` does not override it.

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."        # current terminal only
$env:GEMINI_API_KEY = "..."
```

If a method's key is missing the pipeline still runs the others and records an
error row for the method that could not run.

### 4. Test images

Drop the brick JPEGs into `test_images/`. For the first test, that's the 2
nadir crops exported from the Insta360 X3 app.

## Usage

Run from inside the `brick-ocr/` directory:

```powershell
python pipeline.py --input test_images/ --output output/results.csv
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--input` | (required) | Folder of `.jpg` / `.jpeg` / `.png` images |
| `--output` | (required) | CSV file to write |
| `--methods` | `sonnet,gemini-pro,gemini-flash` | Methods to run; add `paddle` for the OCR cross-check |
| `--tiles` | `1x1` | Split each image into an `RxC` overlapping tile grid (see below) |
| `--tile-overlap` | `0.2` | Tile overlap as a fraction of tile size |
| `--no-gpu` | off | Force PaddleOCR onto the CPU |
| `--cpu-threads` | `8` | PaddleOCR CPU inference threads |
| `--no-group` | off | Report raw PaddleOCR lines instead of grouped bricks |
| `--brick-gap` | `1.5` | Line-into-brick grouping sensitivity (see below) |

To run a subset of methods:

```powershell
python pipeline.py --input test_images/ --output output/results.csv --methods paddle,gemini-pro
```

## Output

**Terminal** — a per-image comparison with one stacked block per method, each
result tagged with its confidence (PaddleOCR also shows its raw 0-1 score).

**CSV** — written to `--output` incrementally (one image at a time, with a
flush, so an interrupted run keeps every finished image), with columns:

| Column | Notes |
|--------|-------|
| `filename` | source image |
| `method` | `paddleocr`, `claude-sonnet`, `gemini-pro`, `gemini-flash` |
| `brick_inscription` | one row per brick; a brick's lines joined with ` / ` |
| `confidence` | `high` / `medium` / `low`, plus `none` (nothing found) or `error` |

The CSV is the seed of the eventual full Excel catalog (section, brick number,
inscription lines).

## Tiling for higher-resolution OCR

A single wide photo spreads its pixels across dozens of bricks, and vision
APIs downscale each image (Anthropic caps it at ~1.15 MP) — so on a wide shot
every brick reaches the model at only ~100 px. `--tiles RxC` (e.g. `--tiles
3x3`) crops each photo into an `R x C` grid and OCRs every tile separately, so
each brick is read at far higher effective resolution; for the LLM methods
each tile is also its own full-detail API call.

```powershell
python pipeline.py --input test_images/ --output output/results.csv --tiles 3x3
```

Tiles overlap (`--tile-overlap`, default 0.2) so a brick on a cut line still
appears whole in one tile; the pipeline then removes the duplicate detections
the overlap produces (by box overlap for PaddleOCR, by text similarity for the
LLM methods) and maps everything back to full-image coordinates before grouping.

Notes:
- Choose the grid so each brick is comfortably smaller than the overlap band
  (`2 x tile-overlap x tile-size`), or boundary bricks may be split.
- `--tiles 3x3` means **9 API calls per image, per LLM method** — more
  thorough, ~9x the cost. Size the grid to your images.
- If the re-shot photos are tight (just a few bricks per frame), `--tiles 1x1`
  (the default) is fine and tiling is unnecessary.

## Grouping PaddleOCR lines into bricks

`group_bricks.py` reassembles PaddleOCR's individual text detections into
bricks in two spatial passes: same-row detections merge into lines, then
vertically-adjacent, horizontally-aligned lines stack into a brick (normally
1-3 lines). Thresholds are multiples of the median detected text height, so
they scale with image resolution.

The defaults are tuned on a synthetic test grid and **will likely need tuning
against real brick photos**. The most layout-sensitive knob is `--brick-gap`
(how large a vertical gap still counts as "same brick"):

- Separate bricks getting **merged** into one → **lower** `--brick-gap`.
- One brick's lines being **split** apart → **raise** `--brick-gap`.

The pipeline warns when a grouped brick exceeds 3 lines, a likely sign of
over-merging. Use `--no-group` to see the raw detections while tuning.

## File structure

```
brick-ocr/
  pipeline.py        whole-image comparison pipeline + CSV writer
  brick_pipeline.py  per-brick pipeline: detect -> crop -> OCR each brick
  detect_bricks.py   classical-CV paver detector (mortar-joint grid)
  consensus.py       collapses the comparison CSV into a triaged catalogue
  ocr_paddle.py      PaddleOCR wrapper (Method 1)
  ocr_anthropic.py   Anthropic provider — Claude Sonnet (Method 2)
  ocr_google.py      Google provider — Gemini 2.5 Pro & Flash (Methods 3-4)
  vision_ocr.py      shared prompt + image/JSON handling for the LLM methods
  tiling.py          splits an image into overlapping tiles
  group_bricks.py    groups PaddleOCR line detections into bricks
  compare.py         stacked terminal comparison output
  requirements.txt   dependencies
  test_images/       input JPEGs go here
  output/            results, crops, and annotated images land here
```

## Not yet implemented (planned)

Batch processing with a progress bar (parallel worker processes — better than
more threads per image on this 24-core machine) · equirectangular→nadir
projection from raw `.insp` files · CV brick-edge detection for layouts the
text-proximity grouping can't resolve · section tagging (A–K) from folder
names · Excel export.
