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
  pipeline.py          whole-image comparison pipeline + CSV writer
  brick_pipeline.py    per-brick pipeline: detect -> crop -> OCR each brick
  single_pipeline.py   one-photo-per-brick production runner (parallel, resumable)
  detect_bricks.py     classical-CV paver detector (mortar-joint grid)
  parse_xls_list.py    parse the source Excel workbook of new brick numbers
  parse_brick_list.py  parse the by-area brick-list PDF (superseded by the .xls)
  parse_tsp_list.py    parse the original by-name (all bricks) PDF into a CSV
  reocr_tsp_pdf.py     re-transcribe the scanned by-name PDF with a vision LLM
  resolve_tsp_rows.py  settle disputed rows via isolated strip reads -> v2 list
  merge_lists.py       merge both lists into the master lookup table
  match.py             match the OCR catalogue against an official list
  make_review_page.py  pack the review queue into one offline review.html
                       (--receiver-url adds autosave-to-server;
                        --photo-base-url uses hosted derivatives instead of
                        embedding -- required for big queues)
  make_search_page.py  build the pickup-counter search page (search.html)
  make_derivatives.py  build thumbs/ + zoom/ JPEG trees for Dreamhost
  make_strips.py       render each scanned-list row as an image; review
                       pages (--master) show it so humans read the PRINT,
                       not the OCR of it
  apply_decisions.py   fold a reviewer's decisions.csv back into the catalogue
  make_report.py       build the Parks & Rec Excel workbook (per-section,
                       reviewer notes, Unofficial-bricks sheet)
  classify_photos.py   label unmatched photos: single brick / stack / other
  run_pipeline.py      the whole loop in one command (see workflow below)
  web/                 Dreamhost hosting: receiver.php + DEPLOY_DREAMHOST.md
  consensus.py         collapses a comparison CSV into a triaged catalogue
  ocr_paddle.py        PaddleOCR wrapper
  ocr_anthropic.py     Anthropic provider (Claude)
  ocr_google.py        Google provider (Gemini)
  vision_ocr.py        shared prompt + image/JSON handling for the LLM methods
  tiling.py            splits an image into overlapping tiles
  group_bricks.py      groups PaddleOCR line detections into bricks
  compare.py           stacked terminal comparison output
  requirements.txt     dependencies
  reference/           official Municipality brick lists (see below)
  tests/               pytest suite incl. the 26-photo regression gate
  test_images/         input JPEGs go here
  output/              results, crops, and annotated images land here
```

## The production batch run

`single_pipeline.py` is the runner for the full warehouse batch (one photo =
one brick). It is built to survive a ~13,000-photo run:

```powershell
python single_pipeline.py --input photos/ --output output/singles.csv --workers 8
```

- The default method is **`gemini-lite-31`** (`gemini-3.1-flash-lite`),
  validated 2026-07-29 on the labeled set: 26/26 matched, 0 wrong IDs —
  including three worn bricks the previous default left in review. It
  replaces `gemini-flash` (`gemini-2.5-flash`), which Google deprecates on
  2026-10-16. `gemini-flash-3` (Gemini 3 Flash Preview) scored identically
  and serves as the second-opinion/escalation method.
- `--workers N` (default 8) OCRs images concurrently; the LLM calls are
  network-bound, so threads scale nearly linearly (a serial run would take
  ~8 hours; 8 workers cut it to ~1).
- Every API call retries transient failures (429/5xx, malformed replies)
  3 times with backoff inside the provider (`ocr_google.py` /
  `ocr_anthropic.py`) before an `ERROR:` row is recorded.
- Every row is flushed as it is written -- a crash or Ctrl-C loses nothing.
- `--resume` keeps a previous run's good rows and redoes only missing images
  and `ERROR:` rows. An *empty* read is kept (the model really saw no text);
  an `ERROR:` read is retried.
- Subdirectories are walked, and two folder conventions are read back into
  the catalogue -- needed both because the warehouse pallet labels are
  arbitrary (K, H1..H6, ...) and do **not** encode the park section:

  | layout | when | tags recorded |
  |---|---|---|
  | `photos/pallets/<pallet>/IMG.jpg` | section unknown (the usual case) | pallet only |
  | `photos/<SECTION>/<pallet>/IMG.jpg` | section known at photo time | section + pallet |

  A pallet-only photo simply matches against the full master list, and the
  match itself reveals the section (every master row carries one) -- the
  pallet tag is what the pickup counter needs to find the physical brick.
  The `image` column is the path relative to `--input` (so repeated camera
  filenames on different pallets stay distinct rows, including across
  `--resume`). A flat folder of photos works exactly as before (both columns
  empty); a nested folder matching neither convention is reported at startup
  and its photos get no tag -- rename it before the run, not after.
  Downstream tools treat `image` as an opaque key, so matched CSVs, the
  review page, and decisions files join up unchanged; give
  `make_review_page.py --photos` the same root the run used as `--input`.

So the crash-recovery loop is simply: re-run the same command with `--resume`
until the end-of-run summary reports no remaining ERROR reads.

## Tests

```powershell
python -m pytest tests/
```

No API calls, pure CSV -- safe to run anywhere. Two layers:

- **Unit tests** pin every measured matching behaviour: the scan-confusable
  folds, the phonetic (voice-transcription) folds, token-containment scoring
  and its uniqueness margin, identical-copy handling, the E/F/G
  original-number ranges, NO-BRICK detection, and the resume bookkeeping.
- **The regression gate** (`tests/test_regression_labeled.py`) replays the
  26 labeled warehouse photos' frozen OCR reads (`tests/fixtures/`, from the
  production method `gemini-3.1-flash-lite`) through `match.py` against the
  committed `reference/master_list.csv` and asserts the validated baseline:
  **26/26 matched to the verified brick, zero false positives**. Any change
  to the matching layers -- or a master-list rebuild that breaks
  identification -- fails here first.

## Review and handoff workflow

**One command runs the whole loop** (configuration in `.env` — API keys,
receiver URL/token, staff login, photo base URL):

```powershell
python run_pipeline.py            # ocr(resume) -> merge -> GATE -> match ->
                                  # classify -> pull decisions -> apply ->
                                  # pages -> Excel report
python run_pipeline.py --rescan   # also re-read troublesome scan rows (API $)
python run_pipeline.py --only pages,report   # regenerate outputs only
```

The regression gate runs before anything ships: if the rebuilt data breaks
validated matching, the pipeline aborts and the previous master list is
kept at `reference/master_list.csv.prev`. Reviewer decisions are pulled
straight from the hosted receiver (its `action=list`/`action=fetch` GET
API), so a hosted review round needs no manual downloads at all.

The individual steps, for running by hand:

```powershell
# 1. OCR the photos (parallel, resumable -- see above)
python single_pipeline.py --input photos/ --output output/singles.csv --workers 8

# 2. Identify each photo against the master list
python match.py --catalog output/singles.csv --reference reference/master_list.csv \
    --output output/matched.csv --scan-ocr
#    -> matched.csv, review_matched.csv (unmatched queue), duplicates_matched.csv (QA),
#       missing_<S>.csv per photographed section (lost/broken candidates)

# 3. Pack the review queue into ONE self-contained page and send it out
python make_review_page.py --review output/review_matched.csv \
    --photos photos/ --catalog output/singles.csv --output output/review.html
```

`review.html` opens in any browser, fully offline: each undecided photo is
embedded next to its top candidates as clickable choices (plus "None of
these" / "Can't read the photo"). Choices autosave in the browser, and one
button downloads `decisions.csv` — the reviewer emails that single small file
back. Multiple reviewers / sittings produce multiple decisions files; all are
accepted below.

```powershell
# 4. Fold the human decisions back in
python apply_decisions.py --matched output/matched.csv \
    --decisions decisions.csv --output output/matched_final.csv

# 5. Build the deliverable: the Parks & Rec Excel workbook
python make_report.py --master reference/master_list.csv \
    --matched output/matched_final.csv --output output/brick_report.xlsx
```

The workbook has a Summary sheet (per-section totals), an All-bricks sheet,
and one sheet per section, each row a brick with buyer, inscription, both
numbers, review flags, and its photo status — `Present` rows highlighted. A
blank photo status means *not photographed yet*, which is **not** evidence a
brick is missing until its section is photographed in full; the Summary sheet
says so in words, because that distinction is the whole point of the count.

## Manual browser checks (the JS the test suite can't run)

The page generators are unit-tested, but the in-browser behaviour is not.
After regenerating pages, a two-minute pass over this list catches what
pytest can't:

**search.html** (log in, hard-refresh)
1. Type a surname -> results appear as you type; `Esc` clears.
2. Type a misspelling ("JOLY COY") -> the right brick still ranks first.
3. Type a brick number -> both numbering eras listed, era labelled.
4. A photographed brick shows "at pickup site -- pallet X" and a
   `verify` button; clicking it shows photo + OCR read + scan row.
   Clicking the photo or the scan row magnifies it in place (the full
   zoom image loads only on that click); click again or `Esc` closes.
5. If any reviewer confirmed "None of these": search their brick's words
   -> an "unofficial" entry appears with pallet and note.

**review.html**
6. Photos and candidate strip images load; hovering a strip magnifies it.
7. Picking a candidate turns the card green; "Clear choice" un-decides
   it; both survive a page reload (localStorage).
8. With the receiver live: a decision shows "saved to server HH:MM"
   within a few seconds.
9. Stack photos sit at the bottom under the banner, one-click "Stack /
   pallet overview photo" first, candidates still available beneath.
10. "Download decisions.csv" produces a file with your reviewer name.

**all three pages** (search.html, review.html, fp_review.html)
11. The header nav bar clicks through to the other two pages without a
    second login prompt (one basic-auth realm covers the directory);
    review.html and fp_review.html show today's date as "built ..." in
    the nav.

## The two official lists

There are two Municipality lists, and they complement each other. Neither
alone is enough.

| | `brick_list_xls.csv` (by area) | `tsp_brick_list_v2.csv` (by name) |
|---|---|---|
| source | `TSP Bricks All.xls`, the 2008 source workbook | `TSP Bricks ALL - OG List by Name - OCR.pdf`, a scan |
| rows | 8,281 | 13,389 |
| coverage | **areas A–D and H–K only** | **all bricks, including E, F, G** |
| gives you | the section + grid position (Column/Row) | the brick # and the **buyer's name** |
| text quality | clean (digital source) | two noisy transcriptions, cross-checked (see below) |

The by-name list's canonical form is **v2** (`resolve_tsp_rows.py`): each row
carries the coordinate parse of the scan's embedded text (`full_name`) *and*
a vision-model re-read (`alt_name`) — whole-page (`reocr_tsp_pdf.py`) where
the two agree, an isolated single-row strip where they disputed. The two
transcriptions err in complementary ways (the parse has right rows / noisy
glyphs; the model has clean glyphs / occasional wrong-row slips), so
`merge_lists.py` and `match.py` score against both and keep the better —
neither ever replaces the other. Strip renumberings are accepted only when
they fill an unclaimed, in-range brick number; everything else stays flagged
(`verified` / `flag` columns) for the review pile. `tsp_brick_list.csv` (the
parse alone) is kept as the v2 build input.

`brick_list_xls.csv` (from `parse_xls_list.py`) supersedes `brick_list.csv`,
the older parse of the printed PDF (`ABCDHIJK.pdf`, a.k.a. "ABCDHIJK New
Brick Numbers.pdf"): the workbook is what that PDF was printed from, and the
PDF text-extraction had lost 407 rows (mostly area C), truncated 68
inscriptions, and mis-sectioned 111 boundary bricks.

A separate data caveat that applies to *both* lists: some entries were
transcribed **by voice**, so the list can spell a name phonetically while the
brick spells it properly (BRIAN vs BRYAN, CATHY vs KATHY). Matching folds the
classic phonetic equivalences (`consensus.phonetic_fold`) so those pairs
compare as equal.

Why two lists exist (per the 2009 *How to Find Your Brick* brochure and
muni.org): the 2008 renovation relocated ~8,000 bricks and **renumbered** them
— those are areas A–D/H–K and the by-area list ("new brick numbers"). Areas
E, F and G were *not* moved in 2008, so they kept their **original**
(certificate) numbers and never appeared in the by-area list. The by-name
list's Brick# is the original number for every brick. The original numbers of
the unmoved areas are contiguous ranges, so for them the number alone gives
the section:

| original # | area | 2008 fate |
|---|---|---|
| 1–3,377 | F | unmoved — original # still valid |
| 8,279–9,126 | G | unmoved — original # still valid |
| 9,127–10,070 | E | unmoved — original # still valid |
| everything else | A–D, H–K | relocated + renumbered — look up by text |

For a *moved* brick the two numbers are unrelated ("Travis E Williams" is
`#3770` originally, `#7305` in area I now) — join on inscription text, never
on the id.

### The master list

`merge_lists.py` combines both lists into `reference/master_list.csv` — one
row per original brick with `orig_id`, `new_id`, `section`, `moved`, `status`,
`buyer`, both inscriptions, and two review columns: `og_verified` (the v2
list's per-row trust tier: `agreed` / `strip` / `parse`) and `flag`, a
`;`-joined list of v2's row flags (`number?:…`, `page0`, …) plus two checks
added at merge time — `dup_orig` (the same original number appears on more
than one OG row: a residual scan id collision, so inside E/F/G two rows claim
one physical brick) and `orig_range` (an impossible certificate number,
outside 1–13,344). Flagged rows keep their best-effort assignment; the flag
routes them to human review, it does not withhold the data. Current counts:
1,374 rows carry a flag (308 dup-id, 3 out-of-range, 1,203 carried from v2). Unmoved bricks get their section from the
number ranges; moved bricks are text-joined to the by-area list (word-blocked
fuzzy match at ≥0.80, plus a stricter rescue pass — lower score but a clear
margin over the best *differently-inscribed* runner-up *and* buyer-surname
corroboration; identical-copy batches are then assigned one-to-one, strongest
join first, so a weak rescue can never displace a rightful 1.00 owner —
contested bricks go to the best-scoring claimant and the loser drops to
review). Current yield with the v2 OG list and
the .xls source: 12,902 of 13,389 rows fully resolved, 468 flagged `unjoined`
for review, 19 sales recorded as "NO BRICK NO INSCRIPTION". By-area rows no OG
row claimed land in `master_unclaimed.csv`.

```bash
python merge_lists.py --og reference/tsp_brick_list_v2.csv \
    --new reference/brick_list_xls.csv --output reference/master_list.csv
```

`match.py` scores each candidate two ways and reports which won in the
`match_basis` column: `text` (whole-string similarity, threshold `--min-score`)
and `tokens` (containment — each read word scored against its best counterpart
in the inscription, for worn bricks whose read is a noisy *subset* like
"GRAND BEATY BUDDY" for "HAROLD G BEATY 1938-1991 MY 'BUDDY'"). Because a
generic word set fits many bricks, a `tokens` match must also beat the best
*differently-inscribed* candidate by a clear margin; in exchange it gets a
slightly lower score bar. Hallucinated reads tie across several bricks and are
rejected by the margin; distinctive subsets stand alone and pass. A `tokens`
winner that fails its own gate does not drag the photo to review when a
whole-string candidate clears the normal bar on its own — a noisy containment
tie can outscore the true text match by a hair (measured on the pallet-K test:
"JOLY COY" hit a 0.94 tokens tie while the real *Joey Coy* stood at text
0.93), so the matcher falls back to the best text-basis candidate.

When the catalogue carries a `section` column (single_pipeline.py fills it
from the pallet folder convention), `match.py` scopes each photo to its own
section's bricks — plus the master rows with no section assigned, which could
be anywhere — since the pallet says where the brick was lifted from. That
both shrinks the candidate pool and disambiguates identical inscriptions
sold in different sections. The whole list stays the fallback: an accepted
cross-section match is kept but flagged `off-section` in the `section_check`
column (mis-sorted brick or mis-tagged folder — either way a human should
glance at it), and the end-of-run summary counts them. Review-queue
candidates for a tagged photo come from its section too. The global
`--section` flag overrides the per-row tags.

The section tags also drive the missing-brick reports: every section with at
least one tagged photo gets a `missing_<S>.csv` — its official bricks that no
photo (from any section) has matched, keyed on (section, id) since bare ids
collide across the two numbering eras. A section with no photos yet gets no
report, because it would just list itself in full. As always, a missing list
is only a real lost/broken list once its section is photographed in full;
with `--section` the report covers exactly that one section instead.

`match.py` accepts the master list directly, so a warehouse photo resolves to
section + current number + buyer in one step:

```bash
python match.py --catalog output/singles.csv \
    --reference reference/master_list.csv \
    --output output/matched.csv --scan-ocr
```

Bricks that stay unmatched are the human-review queue: alongside the matched
CSV, `match.py` writes `review_<output name>` listing each unmatched brick's
`--top` N best candidates (default 5), ranked best-first, one row per
candidate — identical-inscription copies collapsed to one slot, scored
against the *whole* reference (not the word-blocked pool, since a badly
misread brick may share no words with its own inscription). A reviewer sees
the photo plus its five most plausible bricks instead of re-searching the
list; `--top 0` disables it.

As continuous QA, `match.py` also writes `duplicates_<output name>` whenever
more than one photo claims the same official brick. That is either a
duplicate photo (harmless) or a false positive — and the file's `copies`
column (how many identical copies of that inscription exist in the reference)
tells them apart: `n_claims > copies` means at least one claim **is** wrong.
This audits the matcher's zero-false-positive record for free on every batch.

This is the backbone of the pickup workflow: a visitor (or staff) searches the
master list by surname or inscription → *does the brick exist?* (`status`) →
*which section / pallet group?* (`section`); photo-matching against pallets
then confirms *present / broken* per brick; the review file catches the rest.

Because the by-name list is a scan, match against it (or the master list)
with `--scan-ocr`, which folds the confusable letter groups (I/L/T/1/J, E/F,
O/D/Q/0) on both sides. Rebuilding the by-name list from the PDF, in order:

```bash
python parse_tsp_list.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf" \
    --output reference/tsp_brick_list.csv                      # coordinate parse
python reocr_tsp_pdf.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf" \
    --pages all --output reference/tsp_reocr_full.csv \
    --compare reference/tsp_brick_list.csv                     # whole-page re-OCR (~$2)
python resolve_tsp_rows.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf" \
    --reocr reference/tsp_reocr_full.csv \
    --output reference/tsp_brick_list_v2.csv                   # strip-resolve disputes (~$1)
```

`--section` does not apply to the by-name list (it has no section column).

## Not yet implemented (planned)

Equirectangular→nadir projection from raw `.insp` files · CV brick-edge
detection for layouts the text-proximity grouping can't resolve.
