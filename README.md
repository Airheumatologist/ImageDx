# Visual Findings Library

This repository builds a VisualDx-style medical image library: a browsable,
findings-tagged collection of disease imagery curated from PMC open-access
review articles with commercial-use licenses.

Current disease scope: **SLE**, **dermatomyositis**, **ankylosing
spondylitis**, **rheumatoid arthritis**, **systemic sclerosis**,
**psoriasis**, **psoriatic arthritis**, **sarcoidosis**, **gout**, and
**atopic dermatitis** (see `src/visual_pilot/data/diseases.json`). The
authoritative spec is `docs/visual_pilot_plan.md`.

## Pipeline at a glance

```text
                        Visual Findings Library pipeline
                        ================================

  PREBUILT INDEXES                    LLM STAGES (OpenRouter)
  +---------------------------+       P1 article relevance   (text)
  | turbopuffer PMC namespace |       P2 caption triage      (text, batched)
  |  chunk-level: title BM25, |       P3 figure judgment     (vision)
  |  content BM25, dense ANN  |       P4 finding extraction  (text)
  +---------------------------+
              |
              v
 +=====================================================================+
 | 1. SELECT      select_articles.py                    -> articles    |
 |    synonym + finding/modality queries, RRF fusion,                  |
 |    pub-type filter, license gate, P1 relevance triage               |
 +=====================================================================+
              |  status: relevant
              v
 +=====================================================================+
 | 2. PARSE       parse.py (ranked batches)             -> figures     |
 |    JATS XML from s3://pmc-oa-opendata; one row per <fig>;           |
 |    pre-rejects: third-party, bad license, missing graphic           |
 +=====================================================================+
              |  status: pending        (rejects -> caption_rejected)
              v
 +=====================================================================+
 | 3. TRIAGE      triage.py  (P2, batches of 40)                       |
 |    caption-only keep / uncertain / drop                             |
 +=====================================================================+
              |  status: caption_kept | caption_uncertain
              v
 +=====================================================================+
 | 4. JUDGE       judge.py   (P3, vision model)                        |
 |    image bytes -> base64 data URL; panel bboxes, modality,          |
 |    findings, subtype, demographics; metadata priority ordering      |
 +=====================================================================+
              |  status: vision_accepted      (else vision_rejected)
              v
 +=====================================================================+
 | 5. STORE       store.py                                -> panels    |
 |    bbox crops (2% pad), whole-figure mode for ND/overlap/tiny,      |
 |    WebP q90 @2048px + 400px thumbs, sha256 dedup, attribution,      |
 |    proposed findings upserted to findings_vocab (unapproved)        |
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
 |    per-disease funnel, panel distributions, coverage gaps, cost     |
 |    ledger, access failures, HTML/CSV spot-check sheets              |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 8. SERVE       viewer/  (FastAPI, stage 8)           -> localhost   |
 |    per-disease tabbed browser over panels/thumbs; JSON API          |
 +=====================================================================+
```

`run-all` orchestrates 1-7: it selects once, then expands in yield-driven
batches (default 50 articles/disease) — `parse -> triage -> judge -> store`
per batch — stopping after 2 consecutive zero-yield batches, a safety limit
(`--max-articles`, `--max-runtime-seconds`), or budget exhaustion
(`--budget-usd`). It finishes with `extract` + `report`.

## What each stage does

### 1. `select` — article selection (`select_articles.py`)

Retrieves candidate review articles per disease from the **prebuilt
turbopuffer PMC chunk index** (`TURBOPUFFER_NAMESPACE_PMC`, default
`medical_database_pmc`):

- per disease synonym: title BM25 (top 200), page_content BM25 (top 300),
  dense ANN via DeepInfra embeddings (top 300);
- plus up to `VP_VISUAL_QUERY_CAP` (default 12) finding/modality content
  queries ("<disease> <subtype> <finding> <modality>"), round-robined across
  image-bearing categories and prioritized toward findings with fewer
  stored panels;
- all ranked lists fused per-PMCID with **RRF (k=60)**, visual hits weighted
  12x; qualifying passages kept as `retrieval_evidence_json`;
- filters: `has_full_text` AND review-type (`publication_type` Contains
  "Review" OR `article_type` = "review-article"); Python-side exclusions
  drop case reports, trials, letters, meta-analyses, etc.;
- license gate via the per-article S3 metadata JSON (`license_allows` →
  `crop` / `whole_figure` / excluded);
- relevance: the deterministic title rule first, else **P1** LLM triage on
  title + abstract.

### 2. `parse` — figure extraction (`parse.py`, `jats.py`, `pmc.py`)

For each `relevant` article, fetches the JATS bundle **in memory** from the
public `s3://pmc-oa-opendata` bucket (`{pmcid}.{version}/` dirs; the old
`oa.fcgi`/`oa_package` endpoints are dead — see stage-0 findings below).
`jats.parse_article` yields one `figures` row per `<fig>`:

- graphic hrefs resolve to S3 HTTPS URLs via `media_urls`/ListObjectsV2;
- figure-level `<permissions>` licenses override the article license;
- pre-rejects (`caption_rejected`): third-party wording, disallowed
  license, missing graphic — everything else becomes `pending`.

Articles are processed in **ranked batches**: `article_rank` scores each
candidate on retrieval evidence, finding/modality mentions and coverage
gaps; a bounded JATS caption peek reranks the shortlist and can *rescue*
licensed `license_ok`/`irrelevant` articles whose captions demonstrably
depict the target disease.

### 3. `triage` — caption triage (`triage.py`, prompt P2)

`pending` figures go to a text model (`VP_TRIAGE_MODEL`, default
`stealth/space-bunny-alpha` on OpenRouter) in batches of `VP_TRIAGE_BATCH` (40). Per
figure the model
returns a route:

```text
  P2 route        ->  figure status          ->  next
  keep            ->  caption_kept           ->  judge
  uncertain       ->  caption_uncertain      ->  judge
  drop            ->  caption_rejected       ->  (end)
  third_party     ->  caption_rejected       ->  (end, hard exclusion)
```

Contradictory drops (model says "drop" but also flags a real patient image
of a target disease) are auto-downgraded to `uncertain`; a one-time
`revisit_conflicting_rejections` pass re-queues historical contradictions.
Figures the model omits stay `pending` with `attempts` bumped.

### 4. `judge` — vision judgment (`judge.py`, prompt P3)

`caption_kept`/`caption_uncertain` figures are ordered by a deterministic
metadata priority score (disease/modality/finding hits, coverage gaps,
license, diagram penalties — order only, never rejection). Image bytes are
fetched into memory, normalized (`prepare_for_llm`: TIFF→PNG, downscale to
`VP_IMAGE_MAX_EDGE`) and sent as **base64 data URLs** — the LLM provider
cannot fetch the S3 URLs (`binary/octet-stream`, stage-0 finding).

P3 returns per-panel structured output: `include`, `disease_key`,
`subtype`, `modality`, `body_site`, `findings[]`, `bbox`, demographics
(age group, skin tone, stated ethnicity), typicality, confidence.
Post-validation clamps/swaps bboxes, demotes unknown finding keys to
`proposed_findings`, excludes non-pilot diseases, normalizes subtypes.
≥1 included panel → `vision_accepted`, else `vision_rejected`; fetch/LLM
failures → `vision_error` (retried up to 3 attempts).

### 5. `store` — panel materialization (`store.py`)

`vision_accepted` figures are refetched; a display copy capped at
`VP_ORIGINAL_MAX_EDGE` (default 2048px) is written to
`figures/{pmcid}/{stem}.webp` — the verbatim bytes stay refetchable from
the PMC S3 bundle (`figures.sha256` verifies pixels on refetch). Each
included panel is cropped from its bbox with 2% padding, capped at
`VP_PANEL_MAX_EDGE` (default 2048px) and encoded per `VP_PANEL_FORMAT`
(default WebP at `VP_PANEL_QUALITY`=90; `png` keeps lossless archival
crops):

```text
data/visual_pilot/
|-- visual_pilot.sqlite
|-- figures/{pmcid}/{stem}.webp                  # capped display originals
|-- panels/{disease}/{modality}/{panel_id}.webp  # crops
|-- thumbs/{panel_id}.webp                       # 400px thumbnails
`-- reports/                                     # stage-7 output
```

- **crop mode** `whole_figure` applies to ND licenses, missing/tiny
  (<3%) bboxes and >30% panel overlaps;
- **dedup:** identical encoded-image sha256 reuses the existing file (own
  row, attribution and license kept);
- `proposed_findings` upsert `findings_vocab` as unapproved entries
  (`approved=0`, `proposed_by_llm=1`) — exactly once per figure;
- each panel gets a citation `attribution_text` (authors, title, journal,
  DOI, license, figure label).

### 6. `extract` — text findings (`extract_findings.py`, prompt P4)

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

### 7. `report` — pilot report (`report.py`, no LLM)

Writes `reports/pilot_report.md` + `.json`: per-disease funnel
(select → parse → triage → vision → stored), panel distributions,
zero-image vocabulary findings, skin-tone distribution, cost ledger,
access failures, and human spot-check sheets (`spot_accepted.html`,
`spot_caption_rejected.html` + CSVs).

### 8. `serve` — library viewer (`viewer/`, FastAPI)

`python3 -m src.visual_pilot.cli serve --port 8765` starts a local
browser: per-disease pages with modality/body-site/finding tabs, grouped
by subtype, sorted classic → variant → atypical then confidence. JSON API
under `/api/...`, images under `/media/...` (panels/thumbs/figures only).
Includes an SLE↔DM skin comparison page. Unapproved proposed findings are
filtered out of every response.

Image cards show a readable depiction label; opening one reveals the full
image, concise article context, and source/license links. SLE skin groups use
the depicted findings, with vascular findings separate from cutaneous lupus
subtypes. The Pediatric view gathers known child/adolescent images.

The clinical publication policy excludes collages, unusable snippets,
non-patient graphics, normal/control images, and veterinary material. To audit
an existing library, run `python3 -m src.visual_pilot.curation_audit`; add
`--apply` to back up SQLite and save reversible exclusions. The images and
original judgments are retained. See [curation review](docs/visual_curation_review.md)
for the diagnosis and audit details.

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
|-- env.example                  # provider keys + VP_* settings
|-- requirements.txt
|-- docs/                        # specifications, plan, curation review
|-- scripts/                     # synthetic fixture generator for viewer tests
|-- src/
|   `-- visual_pilot/            # the pipeline package
|       |-- cli.py               # stage runner + run-all orchestrator
|       |-- config.py            # env loading + VP_* settings
|       |-- db.py                # sqlite schema: articles/figures/panels/...
|       |-- diseases.py          # disease + findings vocab seeding
|       |-- retrieval.py         # turbopuffer namespace + DeepInfra embeds
|       |-- pmc.py               # S3 bundle fetch, licenses, image prep
|       |-- jats.py              # JATS XML parser
|       |-- select_articles.py   # stage 1
|       |-- article_rank.py      # candidate reranking + caption signals
|       |-- parse.py             # stage 2
|       |-- triage.py            # stage 3 (P2)
|       |-- judge.py             # stage 4 (P3, vision)
|       |-- store.py             # stage 5
|       |-- extract_findings.py  # stage 6 (P4)
|       |-- report.py            # stage 7
|       |-- llm.py               # provider client: budget, cache, retries
|       |-- prompts.py           # P1-P4 systems + JSON schemas
|       |-- data/                # diseases.json, findings_vocab.json
|       `-- viewer/              # FastAPI library browser (static HTML/JS)
|-- tests/visual_pilot/
`-- data/visual_pilot/           # runtime artifacts (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
cp env.example .env   # fill in DEEPINFRA_API_KEY + TURBOPUFFER_API_KEY
```

## Usage

```bash
# one-shot: full pipeline, all diseases, capped spend
python3 -m src.visual_pilot.cli run-all --disease all --budget-usd X

# or stage by stage (all flags: --disease --limit --dry-run --budget-usd
#                        --pmcids --batch-size --max-articles ...)
python3 -m src.visual_pilot.cli init                 # create DB + seed
python3 -m src.visual_pilot.cli select  --disease sle
python3 -m src.visual_pilot.cli parse   --disease sle
python3 -m src.visual_pilot.cli triage  --disease sle
python3 -m src.visual_pilot.cli judge   --disease sle
python3 -m src.visual_pilot.cli store   --disease sle
python3 -m src.visual_pilot.cli extract --disease sle
python3 -m src.visual_pilot.cli report
python3 -m src.visual_pilot.cli serve --port 8765    # browse the library
```

See `src/visual_pilot/README.md` for stage details, env vars, and stage-0
findings.

## Fresh data run

The previous pilot datasets, SQLite backups, image files, model-response
caches, generated reports, and one-off evaluation artifacts were cleared on
2026-09-27. The replacement ten-disease pilot completed on 2026-09-28 UTC:
420 parsed review articles, 167 saved images, **123 published images across
all ten diseases**, and 6,486 text finding assertions. The publication audit
saved 44 reversible exclusions; excluded images remain available on disk.

The run used the new pipeline and the configured OpenRouter models:

```bash
python3 -m src.visual_pilot.cli run-all --disease all --limit 80 \
  --batch-size 20 --max-articles 60 --max-runtime-seconds 7200 --budget-usd 5
python3 -m src.visual_pilot.curation_audit --disease all --apply
python3 -m src.visual_pilot.cli report
python3 -m src.visual_pilot.cli serve --port 8765
```

Runtime data is under `data/visual_pilot/` (or `VP_DATA_DIR`); reports include
`pilot_report.md`, `curation_audit_all.json`, and `viewer_qa.json`.
The disease scope now drives CLI choices, prompt schemas, subtype validation,
and publication matching. The viewer supports catalog diseases through
generic tabs where a custom layout is not defined. Verification: 349 tests
passed, all ten live disease pages returned HTTP 200, and all 334 saved image
and thumbnail paths loaded and decoded successfully.

### Threefold article expansion

On 2026-09-28, the candidate limit was raised from 80 to 240 per disease and
the per-run article cap from 60 to 180. Because screening reduced the actual
article count, retrieval was widened to 480 candidates per disease. Caption
rescue inspected up to 100 candidates per disease, with a targeted pass for
eight candidates beyond that shortlist. The final library has **1,408 parsed
review articles** (3.35× the initial run), **287 saved images**, and **184
published images** (61 more, a 49.6% increase). The 6,486 original text
finding assertions and every original image were preserved.

The expansion used `run-all --image-only` with the same license, patient-image,
whole-figure, and named-manifestation gates. The latest comparison and commands
are in `data/visual_pilot/reports/expansion_3x/expansion_report.md`; the preview
remains at `http://127.0.0.1:8765/`. Final QA checked all ten disease pages,
184 visible cards, and all 574 saved image and thumbnail routes.

## Testing

```bash
pytest tests/visual_pilot
```
