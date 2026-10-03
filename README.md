# Visual Findings Library

This repository builds a VisualDx-style medical image library: a browsable,
findings-tagged collection of disease imagery curated from PMC open-access
articles (reviews, case series, original research and case reports) with
commercial-use licenses.

Current disease scope: **SLE**, **dermatomyositis**, **ankylosing
spondylitis**, **rheumatoid arthritis**, **systemic sclerosis**,
**psoriasis**, **psoriatic arthritis**, **sarcoidosis**, **gout**, and
**atopic dermatitis** (see `src/visual_pilot/data/diseases.json`).

## Pipeline at a glance

```text
                        Visual Findings Library pipeline
                        ================================

  DISCOVERY                           LLM STAGES (OpenRouter)
  +---------------------------+       P2 caption triage      (text, batched)
  | Europe PMC REST search    |       P3 figure judgment     (vision)
  |  FIG: caption field,      |       P5 display captions    (text)
  |  license + OA filters     |       P4 finding extraction  (text)
  +---------------------------+
              |
              v
 +=====================================================================+
 | 1. DISCOVER    discover.py, europepmc.py     -> articles + figures  |
 |    passes: overview reviews/series per disease, then reviews/series |
 |    per under-target finding, then backfill with any article type;  |
 |    JATS XML from s3://pmc-oa-opendata; one row per <fig>;           |
 |    caption must name an approved finding -> pending, else rejected  |
 +=====================================================================+
              |  status: pending        (rejects -> caption_rejected)
              v
 +=====================================================================+
 | 2. TRIAGE      triage.py  (P2, batches of 40)                       |
 |    caption-only keep / uncertain / drop; drops treatment figures    |
 +=====================================================================+
              |  status: caption_kept | caption_uncertain
              v
 +=====================================================================+
 | 3. JUDGE       judge.py   (P3, vision model)                        |
 |    image bytes -> base64 data URL; panel bboxes, modality,          |
 |    findings, subtype, demographics                                  |
 +=====================================================================+
              |  status: vision_accepted      (else vision_rejected)
              v
 +=====================================================================+
 | 4. STORE       store.py                                -> panels    |
 |    crops or whole-figure plates, WebP q90 @2048px + 400px thumbs,   |
 |    sha256 dedup, attribution, proposed findings (unapproved)        |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 5. DESCRIBE    describe.py  (P5)              -> display captions   |
 |    standalone title + description, viewer section/subsection;       |
 |    treatment images excluded and their files deleted               |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 6. EXTRACT     extract_findings.py  (P4)             -> findings    |
 |    clinical sections -> disease->finding assertions with            |
 |    verbatim-verified quotes; rebuilds image-source finding rows     |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 7. REPORT      report.py (no LLM)                    -> reports/    |
 |    per-disease funnel, panel distributions, pair coverage,          |
 |    cost ledger, HTML/CSV spot-check sheets                          |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 8. SERVE       viewer/  (FastAPI)                    -> localhost   |
 |    per-disease tabbed browser over panels/thumbs; JSON API          |
 +=====================================================================+
```

`run-all` orchestrates stages 1–7 in rounds. It first finishes any figures an
interrupted run left mid-pipeline, then runs one `overview` round, one
`manifestation` round and up to `--max-rounds` (default 3) `backfill` rounds.
Each round re-plans the pairs still under `--finding-image-target`
(default 10), discovers new articles, and runs `triage -> judge -> store ->
describe` on them in batches of `--batch-size` (default 50). It stops when a
backfill round finds nothing new, at `--max-runtime-seconds` (default 4 h), or
when `--budget-usd` is spent, and finishes with `extract` + `report`.

## What each stage does

### 1. `discover`: figure-first discovery (`discover.py`, `europepmc.py`)

Europe PMC indexes figure captions under the `FIG:` field, so each search
lands on articles that contain an image of the finding rather than articles
that merely discuss the disease:

- pairs are approved (disease, finding) rows from `findings_vocab`, ordered
  by current published image count (fewest first); pairs at the target are
  skipped;
- caption phrasing comes from `pair_terms.caption_terms`: vocabulary labels
  and synonyms with disease/modality words stripped and singular forms added
  ("Systemic-sclerosis digital ulcers" -> "digital ulcer"), plus hand-written
  `CAPTION_TERM_OVERRIDES` where derivation is too generic;
- query: `FIG:"term"` (exact phrase tier), then, if the pair still needs
  articles, `FIG:(word* AND word*)` (all words in the same caption), AND the
  disease in `TITLE`/`ABSTRACT`, `OPEN_ACCESS:y`, `IN_PMC:y`, and a CC
  license clause; `resultType=core` supplies license, publication types and
  retraction data in the same call (stored in `article_source_metadata`);
- passes run broad sources first: `overview` (per disease: narrative reviews
  and case series whose title surveys the presentation), `manifestation` (per
  pair: reviews and case series about the finding), then `backfill` (any
  article type); systematic reviews and meta-analyses are excluded from the
  first two, and hits are taken reviews first (`source_quality.article_tier`);
- only notices (errata, retractions, corrections) are dropped, and
  `pmc.license_allows` re-checks every license locally;
- up to `--per-pair` (default 25) new articles per pair per round are
  fetched **in memory** from the public `s3://pmc-oa-opendata` bucket and
  parsed by `jats.parse_article` into one `figures` row per `<fig>`;
  figure-level `<permissions>` licenses override the article license;
- a figure becomes `pending` only when its caption names an approved finding
  of the disease; other figures, third-party wording, disallowed licenses and
  missing graphics land as `caption_rejected` with no LLM call;
- for case reports, the abstract and case-presentation sections are kept in
  `figures.case_age_text` as patient-age evidence when the caption has none;
- every query is logged in `pair_search_attempts` (`policy_version`
  `epmc-fig.v1`) with hit counts and returned/new PMCIDs.

### 2. `triage`: caption triage (`triage.py`, prompt P2)

`pending` figures go to a text model (`VP_TRIAGE_MODEL`) in batches of
`VP_TRIAGE_BATCH` (40). Per figure the model returns a route:

```text
  P2 route        ->  figure status          ->  next
  keep            ->  caption_kept           ->  judge
  uncertain       ->  caption_uncertain      ->  judge
  drop            ->  caption_rejected       ->  (end)
  third_party     ->  caption_rejected       ->  (end, hard exclusion)
```

P2 drops diagrams, charts, non-human images, mixed or multi-disease montages
and **treatment figures** (before/after, drug response, follow-up healing,
intraoperative/postoperative views, injections, devices), so those are never
downloaded. Contradictory drops (model says "drop" but also flags a real
patient image of a target disease) are downgraded to `uncertain`. Figures the
model omits stay `pending` with `attempts` bumped.

### 3. `judge`: vision judgment (`judge.py`, prompt P3)

`caption_kept`/`caption_uncertain` figures are ordered by a deterministic
metadata priority score (disease/modality/finding hits, coverage gaps,
license, diagram penalties; order only, never rejection). Image bytes are
fetched into memory, normalized (`prepare_for_llm`: TIFF→PNG, downscale to
`VP_IMAGE_MAX_EDGE`) and sent as **base64 data URLs**, because the S3 bucket
serves figures as `binary/octet-stream`.

P3 returns per-panel structured output: `include`, `disease_key`,
`subtype`, `modality`, `body_site`, `findings[]`, `bbox`, demographics
(age group, skin tone, stated ethnicity), typicality, confidence.
Post-validation clamps/swaps bboxes, demotes unknown finding keys to
`proposed_findings`, excludes non-pilot diseases and normalizes subtypes.
≥1 included panel → `vision_accepted`, else `vision_rejected`; fetch/LLM
failures → `vision_error` (retried up to 3 attempts).

### 4. `store`: panel materialization (`store.py`)

`vision_accepted` figures are refetched; a display copy capped at
`VP_ORIGINAL_MAX_EDGE` (default 2048px) is written to
`figures/{pmcid}/{stem}.webp`. The verbatim bytes stay refetchable from
the PMC S3 bundle (`figures.sha256` verifies pixels on refetch). Each
included panel is cropped from its bbox with 2% padding, capped at
`VP_PANEL_MAX_EDGE` (default 2048px) and encoded per `VP_PANEL_FORMAT`
(default WebP at `VP_PANEL_QUALITY`=90):

```text
data/visual_pilot/
|-- visual_pilot.sqlite
|-- figures/{pmcid}/{stem}.webp                  # capped display originals
|-- panels/{disease}/{modality}/{panel_id}.webp  # crops
|-- thumbs/{panel_id}.webp                       # 400px thumbnails
`-- reports/                                     # report output
```

- **crop mode** `whole_figure` applies to ND licenses, missing/tiny
  (<3%) bboxes, >30% panel overlaps, and single-disease multi-panel plates;
- **dedup:** identical encoded-image sha256 reuses the existing file (own
  row, attribution and license kept);
- `proposed_findings` upsert `findings_vocab` as unapproved entries
  (`approved=0`, `proposed_by_llm=1`), exactly once per figure;
- each panel gets a citation `attribution_text` (authors, title, journal,
  DOI, license, figure label).

### 5. `describe`: display captions and sections (`describe.py`, prompt P5)

Article captions carry figure and panel letters, citation marks
("tendon.19 A"), cross-references and permission notes. For each new panel,
P5 (`VP_DESCRIBE_MODEL`, default `VP_EXTRACT_MODEL`) writes a standalone
`display_title` and a 1–2 sentence `display_description` limited to that
image, and picks the viewer `display_section` and `display_subsection` (e.g.
an SLE skin group, an AS stage, or the finding a DM skin image best shows)
from the listed options. The viewer shows the description as Context, keeps
the article caption under Source, and falls back to rule routing when no
valid section was chosen. `describe --force` rewrites existing captions after
a P5 change.

P5 also flags `treatment_related` images that slipped past triage. A flagged
panel gets a reversible `panel_curation` exclusion (reason
`treatment_related`), and its crop, thumbnail and figure original are deleted
unless a published panel still uses them.

### 6. `extract`: text findings (`extract_findings.py`, prompt P4)

For each `parsed` article, clinical sections (heading keywords:
"clinic", "manifest", "imaging", "histo", …; whole body as fallback,
~120k chars) go to P4 (`VP_EXTRACT_MODEL`). Assertions are post-validated:
non-pilot diseases dropped, unknown keys → `proposed_finding`, and quotes
must appear **verbatim** in the source (normalized substring or ≥85%
4-gram overlap, ≤40 words) or are dropped as `unverified_quote`.

Rows land in `disease_findings` (`source='text'`); panels' stored
findings are rebuilt as `source='image'` rows. Delete+reinsert per article
keeps reruns idempotent (`--force` to redo, `--image-only` to rebuild
image rows only).

### 7. `report`: pilot report (`report.py`, no LLM)

Writes `reports/pilot_report.md` + `.json`: per-disease funnel
(select → parse → triage → vision → stored), panel distributions,
zero-image vocabulary findings, skin-tone distribution, cost ledger,
access failures, and human spot-check sheets (`spot_accepted.html`,
`spot_caption_rejected.html` + CSVs).

Two pair-coverage sections report every approved (disease, finding) pair,
even pairs with zero articles or images, from the same frozen gallery
snapshot the scheduler and viewer use:

- `pair_coverage`: histogram buckets plus `floor`/`target`/`cap` (defaults
  3/10/20) and per-disease `pairs_at_floor`, `pairs_at_target`, `pairs_full`,
  `zero_image_pairs`, `tier_counts`, `under_target`, and plate counts;
- `pair_funnel`: per pair, retrieved and licensed articles, caption-matched
  figures, eligible/published/reserve distinct images, deficits, `tier`,
  `milestone`, `blocked_reason`, search attempts, `next_action`, structured
  `rejection_categories`, and `selection_reserves`. Rejection categories and
  reserves are separate concepts and are never merged.

### 8. `serve`: library viewer (`viewer/`, FastAPI)

`python3 -m src.visual_pilot.cli serve --port 8765` starts a local
browser: per-disease pages with section tabs (subsections such as SLE skin
groups or AS stages), sorted classic → variant → atypical then confidence.
JSON API under `/api/...`, images under `/media/...` (panels/thumbs/figures
only). Includes an SLE↔DM skin comparison page and a Pediatric view.
Unapproved proposed findings are filtered out of every response.

To audit an existing library against the publication policy (no collages,
unusable snippets, non-patient graphics, normal/control images or veterinary
material), run `python3 -m src.visual_pilot.curation_audit`; add `--apply` to
back up SQLite and save reversible exclusions. See
[curation review](docs/visual_curation_review.md).

## Curation policies

- **Review-first sources.** Galleries rank sources review > case
  series/original study > case report > atypical (drug-induced, rare or
  unusual presentations; `source_quality.article_tier`), so each pair leads
  with broad material and case reports fill what is left. This is ranking,
  not exclusion.
- **No treatment images.** Before/after, drug-response, follow-up healing,
  surgical, laser, injection and device images are excluded (P2 drops them
  before download; P5 is a safety net), so the library shows untreated
  disease and does not promote treatments. Disease a drug caused (e.g.
  drug-induced lupus) is kept.
- **Optional age.** Patient age never gates publication: a stated age sorts
  an image into adult or pediatric, otherwise it shows as "Not stated".
- **Licensing.** Commercial-use gate at article and figure level; ND figures
  are stored whole, never cropped; third-party images are excluded.

## Balanced pair coverage (3/10/20)

Coverage is measured per approved `(disease_key, finding_key)` pair by the
selected gallery, not by retrieved articles or stored rows.

- **Milestones.** `VP_FINDING_IMAGE_FLOOR` (default 3) is the initial
  coverage milestone, `VP_FINDING_IMAGE_TARGET` (default 10) the expansion
  goal, and `VP_FINDING_GALLERY_CAP` (default 20) the maximum published
  gallery size per pair, across all modalities, tabs, and age groups.
  Inconsistent overrides (`floor > target` or `target > cap`) fail loudly at
  startup.
- **Gallery selection and reserves.** `gallery.select_gallery` collapses
  identical hashes, documented same-patient/reuse groups, and undocumented
  same-figure source families into distinct groups, then fills the cap with a
  soft two-per-article preference before a score-ordered second pass.
  Eligible surplus beyond the cap is stored as **reserves**: retained and
  inspectable, never deleted or counted as a rejection.
- **Lane tiers.** Each pair's lane (`manifestation_lanes`) is tiered by
  published distinct count: `empty` (0), `below_floor` (1–2),
  `below_target` (3–9), `expanding` (10–19), `full` (20).
- **Whole-figure plates.** A multi-panel figure whose panels all show the
  same configured disease is stored once as a plate. A plate showing one
  approved finding credits that pair once; a plate showing several
  (`plate_kind='combined'`, under "Combined views") credits no pair.
- **One source of truth.** The viewer, scheduler, and report read the same
  `gallery.coverage_snapshot`, so they agree on pair counts and IDs.

## Guarantees

```text
  Idempotent     every stage resumes from `status`; INSERT OR IGNORE /
                 delete+reinsert make reruns safe
  Budget-capped  --budget-usd enforced across all LLM calls; exhausted
                 budget leaves rows pending for clean resume
  Auditable      every LLM call recorded in `llm_calls` (prompt version,
                 schema, tokens, cost); response cache keyed on input
                 hash makes reruns free
  Licensed       commercial-use gate at article AND figure level;
                 ND figures stored whole, never cropped
  Memory-only    JATS XML and image bytes are never written to disk
                 until the store stage saves approved panels
```

## Repository layout

```text
turborag/
|-- README.md
|-- env.example                  # provider key + VP_* settings
|-- requirements.txt
|-- docs/                        # coverage plans, curation review
|-- src/
|   `-- visual_pilot/            # the pipeline package
|       |-- cli.py               # stage runner + run-all orchestrator
|       |-- config.py            # env loading + VP_* settings
|       |-- db.py                # sqlite schema: articles/figures/panels/...
|       |-- diseases.py          # disease + findings vocab seeding
|       |-- europepmc.py         # Europe PMC FIG: caption search client
|       |-- discover.py          # stage 1: pair planning, fetch, figure rows
|       |-- pair_terms.py        # caption phrasings per (disease, finding)
|       |-- pmc.py               # S3 bundle fetch, licenses, image prep
|       |-- jats.py              # JATS XML parser
|       |-- parse.py             # figure-row helpers + body-section cache
|       |-- triage.py            # stage 2 (P2)
|       |-- judge.py             # stage 3 (P3, vision)
|       |-- store.py             # stage 4
|       |-- describe.py          # stage 5 (P5): captions, sections, treatment filter
|       |-- extract_findings.py  # stage 6 (P4)
|       |-- report.py            # stage 7
|       |-- gallery.py           # per-pair gallery selection + reserves
|       |-- coverage.py          # lane tiers + coverage snapshot
|       |-- publication.py       # publication eligibility rules
|       |-- source_quality.py    # article tiers (review-first ranking)
|       |-- curation_audit.py    # reversible publication audit
|       |-- llm.py               # OpenRouter client: budget, cache, retries
|       |-- prompts.py           # P2-P5 systems + JSON schemas
|       |-- data/                # diseases.json, findings_vocab.json
|       |-- site_export.py       # static GitHub Pages export of the viewer
|       `-- viewer/              # FastAPI library browser (static HTML/JS)
|-- site/                        # exported static site (deployed to Pages)
|-- .github/workflows/pages.yml  # deploys site/ on push to main
|-- tests/visual_pilot/
`-- data/visual_pilot/           # runtime artifacts (gitignored)
```

## Setup

Python 3.10+ is required (the macOS system `python3` is 3.9):

```bash
uv venv --python 3.12 venv && source venv/bin/activate
pip install -r requirements.txt
cp env.example .env   # fill in OPENROUTER_API_KEY (discovery needs no key)
```

The default models (`stealth/space-bunny-alpha` on OpenRouter) are free.
Store `VP_DATA_DIR` on an APFS or other small-block filesystem: on an exFAT
drive with 128 KB clusters, every thumbnail and its macOS `._` companion file
take a full cluster, inflating the library about 8×.

## Usage

```bash
# one-shot: full pipeline, all diseases (add --budget-usd X to cap spend)
python3 -m src.visual_pilot.cli run-all --disease all

# or stage by stage (all flags: --disease --limit --dry-run --budget-usd
#                        --pmcids --per-pair --batch-size ...)
python3 -m src.visual_pilot.cli init                         # create DB + seed
python3 -m src.visual_pilot.cli discover --disease sle --dry-run  # pairs under target
python3 -m src.visual_pilot.cli discover --disease sle --per-pair 25
python3 -m src.visual_pilot.cli triage  --disease sle
python3 -m src.visual_pilot.cli judge   --disease sle
python3 -m src.visual_pilot.cli store   --disease sle
python3 -m src.visual_pilot.cli describe --disease sle   # standalone captions + sections
python3 -m src.visual_pilot.cli extract --disease sle
python3 -m src.visual_pilot.cli report
python3 -m src.visual_pilot.cli serve --port 8765    # browse the library
```

`discover --pmcids ...` skips search and fetches exactly those articles,
attributing each to the diseases its title or abstract names (or to the one
`--disease` given when it names none); `run-all --pmcids ...` then takes
them through triage, judge, store and describe. The 20-article pilot set
lives in `src/visual_pilot/data/parity_pmcids.json`.

`discover --limit N` caps the number of pairs searched. Re-running discovery
skips articles already parsed for the disease, so each round reaches deeper
into the Europe PMC results for pairs that are still under target.

### Static site (GitHub Pages)

```bash
python3 -m src.visual_pilot.cli export-site     # writes site/
git add site && git commit -m "Update site" && git push
```

`export-site` calls the viewer API in-process and writes each response as
JSON beside copies of the viewer pages; the pages read those files when
`window.VP_STATIC` is set. No images are copied: every panel links to its
figure in the public PMC open-data bucket, and cropped panels carry a
normalized `crop` box the page draws in a canvas. `.github/workflows/pages.yml`
deploys `site/` on every push to `main` that changes it (repository
Settings → Pages → Source: GitHub Actions).

See `src/visual_pilot/README.md` for stage details, env vars and PMC access
notes.

## Testing

```bash
pytest tests/visual_pilot
```
