# Visual Findings Library

This repository builds a VisualDx-style medical image library: a browsable,
findings-tagged collection of disease imagery curated from PMC open-access
articles (reviews, case series, original research and case reports) with
commercial-use licenses.

Disease scope: the single disease list, `data/main_database_topics.json`
(2,136 topics across 20 specialties). This includes 493 reviewed additions
awaiting image curation; master-catalog membership does not imply published
images. See [the new-topic run instructions](docs/new-topics-2026-10-28.md)
for the additions-only manifest and launcher. Ten topics (**SLE**, **dermatomyositis**, **ankylosing
spondylitis**, **rheumatoid arthritis**, **systemic sclerosis**,
**psoriasis**, **psoriatic arthritis**, **sarcoidosis**, **gout**, **atopic
dermatitis**) carry short keys (`sle`, `dm`, …), subtypes and hand-curated
findings (`findings_vocab.json`); every other topic's findings are generated
by `build-vocab` (see [Disease catalog](#disease-catalog)).

## Pipeline at a glance

```text
                        Visual Findings Library pipeline
                        ================================

  DISCOVERY                           LLM STAGES (OpenCode + OpenRouter)
  +---------------------------+       P2 caption triage      (text, batched)
  | Europe PMC REST search    |       P3 figure judgment     (vision)
  |  FIG: caption field,      |       P5 display captions    (text)
  |  license + OA filters     |       P8 mimic audit         (text)
  +---------------------------+       P4 finding extraction  (text)
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
 |    crop boxes or whole-figure plates (no image files), pixel        |
 |    sha256 dedup, attribution, proposed findings (unapproved)        |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 5. DESCRIBE    describe.py  (P5)              -> display captions   |
 |    standalone title + description, viewer section/subsection;       |
 |    treatment images excluded from the library                       |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 5b. AUDIT      mimic_audit.py  (P8)           -> mimic exclusions   |
 |    title + abstract + caption: does the image show the topic or a   |
 |    look-alike? other_disease / uncertain excluded, reversibly       |
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
 |    per-disease tabbed browser, images from PMC S3; JSON API         |
 +=====================================================================+
```

`run-all` orchestrates stages 1–7 in rounds. It first finishes any figures an
interrupted run left mid-pipeline, then runs one `overview` round, one
`manifestation` round and up to `--max-rounds` (default 3) `backfill` rounds.
Each round re-plans the pairs still under `--finding-image-target`
(default 10), discovers new articles, and runs `triage -> judge -> store ->
describe` on them in batches of `--batch-size` (default 150). It stops when a
backfill round finds nothing new, at `--max-runtime-seconds` (default 4 h), or
when `--budget-usd` is spent, and finishes with `extract` + `report`.

Within a round the stages overlap:

```text
  discover thread      triage thread             main thread
  ---------------      -------------             -----------
  search ahead  --->   batch arrivals    --->    judge -> store -> describe
  (8 workers),         (<= --batch-size,         (16 vision calls in flight)
  fetch + parse,       or what arrived in
  stream PMCIDs        20 s), P2 triage
```

Discovery hands each pair's newly parsed articles to the triage thread as
soon as they are stored; triage stays up to two batches ahead of judging.
The next round plans only after the current round is fully drained, so it
sees the coverage the round earned. `--max-runtime-seconds` is checked
before every search and batch; figures left unfinished resume at the start
of the next run. Every log line is timestamped, and
`reports/timings_<utc>.json` is rewritten after each batch.

## Disease catalog

`diseases.load_diseases()` builds the catalog from
`data/main_database_topics.json`, whose only grouping is `specialties`
(specialty -> topics). A topic's key is its `topic_id`; its name, synonyms,
MONDO/MeSH ids, specialty, subtypes and summary come from the list.

Discovery searches approved `(disease, finding)` pairs, so every topic needs a
findings vocabulary. The ten curated topics use `findings_vocab.json`;
for the other topics, `build-vocab` (prompt P6, `topic_vocab.py`) turns each
topic's clinical, radiological and pathology manifestations into 3–12 visible
findings with a label, category, synonyms and the `caption_terms` a figure
caption would contain, and writes `src/visual_pilot/data/topic_findings_vocab.json`:

```bash
python3 -m src.visual_pilot.cli build-vocab              # topics without findings
python3 -m src.visual_pilot.cli build-vocab --topics vitiligo --force
```

Post-validation drops findings whose caption terms are generic ("rash",
"mass", a lone short word) or just the disease name. Generated findings are
seeded as approved, like the curated file, and carry `source`, `model` and
`prompt_version` for review; review or edit the JSON before a long run.
New finding categories `endoscopy`, `gross`, `nuclear` and `clinical_general`
route to matching generic viewer tabs.

LLM prompts P2–P4 list only the diseases of the articles in each call
(`prompts.scoped`), so prompt size does not grow with the catalog; triage
batches are grouped by article disease. Upper-case synonyms such as "ITP" or
"UC" are matched case-sensitively, so they never match ordinary words.

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
- topics under `VP_TOPIC_IMAGE_FLOOR` images also get the `disease` pass,
  which searches by disease name rather than caption: case reports naming it
  in the title, then in the abstract, then any article titled with it (never
  the full text, where the disease is mostly a differential). Its figures pass the
  local gate when the caption names a patient image type, a lesion or body
  site ("rash on the trunk") or the disease itself;
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
`proposed_findings`, excludes diseases outside the catalog and normalizes subtypes.
≥1 included panel → `vision_accepted`, else `vision_rejected`; fetch/LLM
failures → `vision_error` (retried up to 3 attempts).

### 4. `store`: panel materialization (`store.py`)

No image files are written. Pages load every panel straight from its
figure's public URL in the PMC open-data S3 bucket (`figures.image_url`)
and draw the crop in the browser. `vision_accepted` figures are decoded
in memory (judge handoff or refetch; `figures.sha256` verifies the pixels)
only to measure each included panel: its bbox crop with 2% padding gives
the stored `width`/`height` and a sha256 of the cropped pixels.
`store.crop_box` turns `bbox_json` + `crop_mode` into the normalized crop
box the pages draw. The data directory holds only the database and reports:

```text
data/visual_pilot/
|-- visual_pilot.sqlite
`-- reports/                                     # report output
```

- **crop mode** `whole_figure` applies to ND licenses, missing/tiny
  (<3%) bboxes, >30% panel overlaps, and single-disease multi-panel plates;
- **dedup:** identical cropped-pixel sha256 is counted as a duplicate (own
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
`treatment_related`).

### 5b. `audit-mimics`: look-alike diseases (`mimic_audit.py`, prompt P8)

The vision judge attributes an image from what it shows, so a mimic can pass
as the topic ("Langerhans cell histiocytosis presented as persistent diaper
dermatitis", "Becker nevus presenting as pseudofolliculitis barbae"). For
each new panel, P8 (text only, `VP_DESCRIBE_MODEL`) reads the topic with its
synonyms and subtypes, the article title and abstract, the caption and the
citing sentences, and answers `topic`, `other_disease` or `uncertain`. The
last two get a reversible `panel_curation` exclusion whose reason starts
with `mimic_audit` and names the diagnosis; a later `topic` verdict lifts it,
and exclusions from other stages or manual review are never touched.
`run-all` runs it after `describe` in every batch; on its own:

```bash
python3 -m src.visual_pilot.cli audit-mimics                     # every panel (cached)
python3 -m src.visual_pilot.cli audit-mimics --disease rubella
```

### 6. `extract`: text findings (`extract_findings.py`, prompt P4)

For each `parsed` article, clinical sections (heading keywords:
"clinic", "manifest", "imaging", "histo", …; whole body as fallback,
~120k chars) go to P4 (`VP_EXTRACT_MODEL`). Assertions are post-validated:
diseases outside the catalog dropped, unknown keys → `proposed_finding`, and quotes
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
JSON API under `/api/...`; images load from PMC S3 with the crop box drawn
in the page, exactly as on the static site. Includes an SLE↔DM skin comparison page and a Pediatric view.
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
- **Unlabeled fill.** A panel that passes every publication gate except
  naming its finding (the source text never uses the vocabulary's wording)
  is `unlabeled_publishable`. `gallery.library_panel_ids` tops each topic's
  default view up with these, ranked as a pair gallery is, until the topic
  shows `VP_UNLABELED_FILL_TARGET` (default 20; 0 disables) distinct images.
  They show without a finding label, credit no pair, never appear under a
  finding filter, and skip bytes already eligible as finding evidence.
- **Topic cap.** `VP_TOPIC_IMAGE_CAP` (default 0, off) limits a topic's
  default view to that many distinct images: pair galleries round-robin
  (each pair's best first), then combined plates, then the unlabeled fill.
  Pair credit is unaffected. `run-all --topic-image-target N` drops a topic
  from later discovery rounds once its default view shows N images.
  `--skip-resume` leaves unfinished figures to another run working on them
  and goes straight to discovery, so two runs can share the database.
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
  Memory-only    JATS XML and image bytes are never written to disk;
                 pages stream images from PMC S3
```

## Repository layout

```text
turborag/
|-- README.md
|-- env.example                  # provider key + VP_* settings
|-- requirements.txt
|-- docs/                        # curation review
|-- data/main_database_topics.json  # the single disease list (the catalog)
|-- src/
|   `-- visual_pilot/            # the pipeline package
|       |-- cli.py               # stage runner + run-all orchestrator
|       |-- config.py            # env loading + VP_* settings
|       |-- db.py                # sqlite schema: articles/figures/panels/...
|       |-- diseases.py          # disease catalog + vocab seeding
|       |-- topic_vocab.py       # build-vocab (P6): findings for index topics
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
|       |-- llm.py               # LLM clients: budget, cache, retries
|       |-- prompts.py           # P2-P5 systems + JSON schemas
|       |-- data/                # findings_vocab.json,
|       |                        # topic_findings_vocab.json (generated)
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
cp env.example .env   # fill in OpenCode + OpenRouter keys
```

Text stages use `opencode-go/iceman-contributor-free` through OpenCode Go:
caption triage (P2), text extraction (P4), display descriptions (P5), findings
vocabulary (P6), and mimic audit (P8). The image judge (P3) uses
`z-ai/glm-5.3-flash` through OpenRouter, restricted to **OpenInference only**.
Provider fallback is disabled; an unavailable OpenInference route returns an
error without trying Relace or another server.

Set `OPENCODE_GO_API_KEY` (the existing `OPENCODE_API_KEY` is also accepted)
and `OPENROUTER_API_KEY` in `.env`. Iceman uses schema instructions in the
prompt because its API rejects `response_format`; replies are validated
locally with one repair retry. GLM uses JSON mode. Both clients stream and
share the existing budget, cache and retry controls. OpenRouter's actual
provider, response ID and reported cost are recorded in the ledger. Its
allowlist is part of the cache identity, preventing reuse of old unrestricted
OpenRouter results.

Re-run the live smoke checks without writing to the production database:

```bash
python -m src.visual_pilot.smoke_providers --clinical
# vision checks alone:
python -m src.visual_pilot.smoke_providers --clinical --vision-only
```

These check every text prompt schema, image recognition, rejection of a
diagram and acceptance of a published CC-BY clinical photograph. See
[`docs/provider-evaluation-2026-10-27.md`](docs/provider-evaluation-2026-10-27.md)
for results and limits. The current model overlay is in
[`env.iceman.example`](env.iceman.example).

Before caption triage or vision analysis, the pipeline checks the existing
`site/data/<topic>/panels.json` files. A figure already shown under every
topic attributed to its article is marked `already_published`, with the
matched identities retained in `triage_json`. Figure IDs and image URLs are
checked before fetching; exact RGB pixel hashes are checked before the
vision call. A match under only some of an article's topics does not skip
the figure. `VP_PUBLISHED_SITE_DIR` selects another site directory when
needed. This prevents the working database from re-analyzing images already
live in the separately exported site. Export-time deduplication remains in
place as a final check.

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
them through triage, judge, store and describe.

### Throughput settings

The vision judge dominated the pilot (about 11 of 18 hours, at ~40 s per call
with 4 in flight). The settings that matter for a full run, all in `.env`:

| Setting | Default | Effect |
|---|---|---|
| `VP_JUDGE_CONCURRENCY` | 64 | vision calls in flight |
| `VP_CONCURRENCY` | 64 | triage/describe/extract calls in flight |
| `VP_TRIAGE_BATCH` | 20 | figures per triage call |
| `VP_*_REASONING_EFFORT` | empty for Iceman, `low` for GLM in `env.example` | per stage: `TRIAGE`, `JUDGE`, `DESCRIBE`, `EXTRACT`, `VOCAB` |
| `VP_SEARCH_CONCURRENCY` / `VP_EPMC_RPS` | 8 / 8 | Europe PMC searches in flight / per second |
| `VP_FETCH_CONCURRENCY` / `VP_S3_RPS` | 64 / 150 | article and image fetches in flight / per second |
| `--batch-size` | 150 | articles per judge batch |
| `VP_LLM_MAX_IN_FLIGHT` | 128 | all LLM calls in flight across stages |
| `VP_LLM_RPM` | 0 (unpaced) | LLM requests per minute across stages, spaced evenly; set to a provider's per-minute limit |
| `VP_LLM_MAX_OUTPUT_TOKENS` | 16384 | output-token cap per request, reasoning included (0 = provider default) |

Iceman uses its default reasoning; GLM accepts `low`, `medium`, `high` or
`off`. A set value is part of the `llm_calls` cache key, so changing it
re-asks the model. HTTP 429s back off and retry
(`VP_RATE_LIMIT_RETRIES`); an HTTP 402 (provider credits/quota spent) stops the
run like an exhausted `--budget-usd`. Both are counted in the timings report
(`llm_429`, `llm_402`).

Run long jobs under `caffeinate -is` on macOS. An idle Mac sleeps mid-run,
and because the process clock stops during sleep, request timeouts never
fire: calls look hung for as long as the machine sleeps.

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

`export-site --merge` folds a database that holds only some topics (or only
part of a topic's images) into the existing site: each topic keeps every image
it already shows and gains the database's images it does not show yet (no
shared image hash, panel or figure), up to `VP_TOPIC_IMAGE_CAP`; topics the
site lacks come whole from the database, and no topic ever shrinks.

See `src/visual_pilot/README.md` for stage details, env vars and PMC access
notes.

## Testing

```bash
pytest tests/visual_pilot
```
