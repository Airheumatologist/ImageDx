# Visual Findings Library

This repository builds a VisualDx-style medical image library: a browsable,
findings-tagged collection of disease imagery curated from PMC open-access
articles (case reports, original research and reviews) with commercial-use
licenses.

Current disease scope: **SLE**, **dermatomyositis**, **ankylosing
spondylitis**, **rheumatoid arthritis**, **systemic sclerosis**,
**psoriasis**, **psoriatic arthritis**, **sarcoidosis**, **gout**, and
**atopic dermatitis** (see `src/visual_pilot/data/diseases.json`). The
authoritative spec is `docs/visual_pilot_plan.md`.

## Pipeline at a glance

```text
                        Visual Findings Library pipeline
                        ================================

  DISCOVERY                           LLM STAGES (OpenRouter)
  +---------------------------+       P2 caption triage      (text, batched)
  | Europe PMC REST search    |       P3 figure judgment     (vision)
  |  FIG: caption field,      |       P4 finding extraction  (text)
  |  license + OA filters     |
  +---------------------------+
              |
              v
 +=====================================================================+
 | 1. DISCOVER    discover.py, europepmc.py     -> articles + figures  |
 |    passes: overview reviews/series per disease, then reviews/series |
 |    per under-target finding, then backfill with any article type;  |
 |    caption names the finding, title/abstract names the disease;    |
 |    JATS XML from s3://pmc-oa-opendata; one row per <fig>;           |
 |    caption must name an approved finding -> pending, else rejected  |
 |    pre-rejects: third-party, bad license, missing graphic           |
 +=====================================================================+
              |  status: pending        (rejects -> caption_rejected)
              v
 +=====================================================================+
 | 2. TRIAGE      triage.py  (P2, batches of 40)                       |
 |    caption-only keep / uncertain / drop                             |
 +=====================================================================+
              |  status: caption_kept | caption_uncertain
              v
 +=====================================================================+
 | 3. JUDGE       judge.py   (P3, vision model)                        |
 |    image bytes -> base64 data URL; panel bboxes, modality,          |
 |    findings, subtype, demographics; metadata priority ordering      |
 +=====================================================================+
              |  status: vision_accepted      (else vision_rejected)
              v
 +=====================================================================+
 | 4. STORE       store.py                                -> panels    |
 |    bbox crops (2% pad), whole-figure mode for ND/overlap/tiny,      |
 |    WebP q90 @2048px + 400px thumbs, sha256 dedup, attribution,      |
 |    proposed findings upserted to findings_vocab (unapproved)        |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 5. EXTRACT     extract_findings.py  (P4)             -> findings    |
 |    clinical sections -> disease->finding assertions with            |
 |    verbatim-verified quotes; rebuilds image-source finding rows     |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 6. REPORT      report.py (no LLM)                    -> reports/    |
 |    per-disease funnel, panel distributions, pair coverage           |
 |    milestones + per-pair funnel, cost ledger, access failures,      |
 |    HTML/CSV spot-check sheets                                       |
 +=====================================================================+
              |
              v
 +=====================================================================+
 | 7. SERVE       viewer/  (FastAPI, stage 7)           -> localhost   |
 |    per-disease tabbed browser over panels/thumbs; JSON API          |
 +=====================================================================+
```

`run-all` orchestrates 1-6 in rounds: each round re-plans from current
coverage, runs `discover` for every approved pair still under
`--finding-image-target` (default 10), then `triage -> judge -> store` on the
new articles in batches of `--batch-size` (default 50). It stops when a round
finds no new articles, after `--max-rounds` (default 3), at
`--max-runtime-seconds`, or when `--budget-usd` is spent, and finishes with
`extract` + `report`.

## What each stage does

### 1. `discover` — figure-first discovery (`discover.py`, `europepmc.py`)

Europe PMC indexes figure captions under the `FIG:` field, so each search
lands on articles that contain an image of the finding rather than articles
that merely discuss the disease:

- pairs are approved (disease, finding) rows from `findings_vocab`, ordered
  by current stored image count (fewest first); pairs at the target are
  skipped;
- caption phrasing comes from `pair_terms.caption_terms`: vocabulary labels
  and synonyms with disease/modality words stripped and singular forms added
  ("Systemic-sclerosis digital ulcers" -> "digital ulcer"), plus hand-written
  `CAPTION_TERM_OVERRIDES` where derivation is too generic;
- query: `FIG:"term"` (exact phrase tier), then, if the pair still needs
  articles, `FIG:(word* AND word*)` (all words in the same caption) — AND the
  disease in `TITLE`/`ABSTRACT`, `OPEN_ACCESS:y`, `IN_PMC:y`, and a CC
  license clause; `resultType=core` supplies license, publication types and
  retraction data in the same call (stored in `article_source_metadata`);
- passes run broad sources first: `overview` (per disease: narrative reviews
  and case series whose title surveys the presentation, with a caption naming
  any finding), `manifestation` (per pair: reviews and case series about the
  finding), then `backfill` (any article type) for pairs still under target;
  systematic reviews and meta-analyses are excluded from the first two, and
  atypical sources are left to backfill;
- any article type is eligible in backfill; only notices (errata,
  retractions, corrections) are dropped, and `pmc.license_allows` re-checks
  every license locally;
- up to `--per-pair` (default 25) new articles per pair per round are
  fetched **in memory** from the public `s3://pmc-oa-opendata` bucket and
  parsed by `jats.parse_article` into one `figures` row per `<fig>`;
  figure-level `<permissions>` licenses override the article license;
- a figure becomes `pending` only when its caption names an approved finding
  of the disease; other figures, third-party wording, disallowed licenses and
  missing graphics land as `caption_rejected` with no LLM call;
- every query is logged in `pair_search_attempts` (`policy_version`
  `epmc-fig.v1`) with hit counts and returned/new PMCIDs.

### 2. `triage` — caption triage (`triage.py`, prompt P2)

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

### 3. `judge` — vision judgment (`judge.py`, prompt P3)

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

### 4. `store` — panel materialization (`store.py`)

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

### 5. `extract` — text findings (`extract_findings.py`, prompt P4)

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

### 6. `report` — pilot report (`report.py`, no LLM)

Writes `reports/pilot_report.md` + `.json`: per-disease funnel
(select → parse → triage → vision → stored), panel distributions,
zero-image vocabulary findings, skin-tone distribution, cost ledger,
access failures, and human spot-check sheets (`spot_accepted.html`,
`spot_caption_rejected.html` + CSVs).

Two pair-coverage sections report every approved (disease, finding) pair —
even pairs with zero articles or images — from the same frozen gallery
snapshot the scheduler and viewer use:

- `pair_coverage` — the milestone summary: legacy histogram buckets plus
  `floor`/`target`/`cap` (defaults 3/10/20) and per-disease `pairs_at_floor`,
  `pairs_at_target`, `pairs_full`, `zero_image_pairs`, `tier_counts`,
  `under_target`, and plate counts;
- `pair_funnel` — per pair: `retrieved_unique_articles`,
  `licensed_articles`, `pair_supported_caption_figures`,
  `eligible_distinct`, `published_distinct`, `reserve_distinct`,
  `floor_deficit`/`target_deficit`/`cap_remaining`, `tier`, `milestone`,
  `blocked_reason`, `attempted_strategies`, `last_deficit_reduction`,
  `next_action`, `pending_work`, `unresolved_legacy_candidates`, structured
  `rejection_categories` (license/third-party, review type, no patient
  image, attribution unclear/other disease, unsupported
  finding, mixed plate, quality, retrieval error), and `selection_reserves`
  (`distinct_groups`, `duplicate_or_diversity_rows`). Rejection categories
  and reserves are separate concepts and are never merged.

### 7. `serve` — library viewer (`viewer/`, FastAPI)

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

## Balanced pair coverage (3/10/20)

Coverage is measured per approved `(disease_key, finding_key)` pair by the
selected gallery, not by retrieved articles or stored rows.

- **Milestones.** `VP_FINDING_IMAGE_FLOOR` (default 3) is the initial
  coverage milestone, `VP_FINDING_IMAGE_TARGET` (default 10) the expansion
  goal, and `VP_FINDING_GALLERY_CAP` (default 20) the maximum published
  gallery size per pair — across all modalities, tabs, and age groups, not
  per tab. Ten is a milestone, not a ceiling; galleries may grow to 20 once
  lower-coverage lanes are served. Inconsistent overrides
  (`floor > target` or `target > cap`) fail loudly at startup.
- **Gallery selection and reserves.** `gallery.select_gallery` collapses
  identical hashes, documented same-patient/reuse groups, and
  undocumented same-figure source families into distinct groups, then fills
  the cap with a soft two-per-article preference before a score-ordered
  second pass. Eligible surplus beyond the cap is stored as **reserves** —
  retained and inspectable, never truncated, deleted, or counted as a
  rejection. Modality/pediatric filters draw subsets of the same capped
  gallery.
- **Lane tiers and blocked reasons.** Each pair's lane is tiered by
  published distinct count: `empty` (0), `below_floor` (1–2),
  `below_target` (3–9), `expanding` (10–19), `full` (20). Lane rows
  (`manifestation_lanes`) keep `status` (`open`/`covered`), `tier`,
  `last_served_sequence`, `blocked_reason` (e.g. `search_plan_exhausted` or
  a safety-limit pause), `search_policy_version`, and
  `last_deficit_reduction_at`. A blocked lane reports its deficit honestly
  instead of spinning or claiming coverage.
- **`pair_search_attempts` ledger.** Bounded replenishment records each
  disease-scoped query round: policy version, round, query/filter hash,
  BM25 depth (300, 600, then 1200 — at most three automatic rounds per
  policy version, ≤6 unattempted query variants per round), returned and
  newly discovered PMCIDs, pending outcomes, errors, and completion time.
  Completed attempts are never repeated; interrupted ones resume.
- **Reporting.** `pilot_report.json` mirrors all of this under
  `pair_coverage` and `pair_funnel` (field list in stage 7 above), and
  `pilot_report.md` renders per-disease pair tables with honest blocked
  reasons and next actions, plus a rejection-categories summary kept
  separate from reserve counts. The viewer, scheduler, and report read the
  same `gallery.coverage_snapshot`, so they agree on pair counts and IDs.

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
|       |-- europepmc.py         # Europe PMC FIG: caption search client
|       |-- discover.py          # stage 1: pair planning, fetch, figure rows
|       |-- pair_terms.py        # caption phrasings per (disease, finding)
|       |-- pmc.py               # S3 bundle fetch, licenses, image prep
|       |-- jats.py              # JATS XML parser
|       |-- parse.py             # figure-row helpers + body-section cache
|       |-- triage.py            # stage 2 (P2)
|       |-- judge.py             # stage 3 (P3, vision)
|       |-- store.py             # stage 4
|       |-- extract_findings.py  # stage 5 (P4)
|       |-- report.py            # stage 6
|       |-- llm.py               # provider client: budget, cache, retries
|       |-- prompts.py           # P2-P4 systems + JSON schemas
|       |-- data/                # diseases.json, findings_vocab.json
|       `-- viewer/              # FastAPI library browser (static HTML/JS)
|-- tests/visual_pilot/
`-- data/visual_pilot/           # runtime artifacts (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
cp env.example .env   # fill in OPENROUTER_API_KEY (discovery needs no key)
```

## Usage

```bash
# one-shot: full pipeline, all diseases, capped spend
python3 -m src.visual_pilot.cli run-all --disease all --budget-usd X

# or stage by stage (all flags: --disease --limit --dry-run --budget-usd
#                        --pmcids --per-pair --batch-size ...)
python3 -m src.visual_pilot.cli init                         # create DB + seed
python3 -m src.visual_pilot.cli discover --disease sle --dry-run  # pairs under target
python3 -m src.visual_pilot.cli discover --disease sle --per-pair 25
python3 -m src.visual_pilot.cli triage  --disease sle
python3 -m src.visual_pilot.cli judge   --disease sle
python3 -m src.visual_pilot.cli store   --disease sle
python3 -m src.visual_pilot.cli extract --disease sle
python3 -m src.visual_pilot.cli report
python3 -m src.visual_pilot.cli serve --port 8765    # browse the library
```

`discover --limit N` caps the number of pairs searched. Re-running discovery
skips articles already parsed for the disease, so each round reaches deeper
into the Europe PMC results for pairs that are still under target.

See `src/visual_pilot/README.md` for stage details, env vars, and stage-0
findings.

## Europe PMC discovery switch (2026-10-01)

Discovery moved from turbopuffer article retrieval (review articles only, P1
relevance triage) to Europe PMC figure-caption search across all article
types. On the same database, the last turbopuffer run had parsed 891 review
articles into 3,626 figures and stored 116 images (about 3% of figures). A
first pilot of six dermatomyositis pairs (10 articles each) found 49 articles
in 12 seconds, queued 90 caption-matched figures, and stored 37 images (46%
of judged figures), mostly from case reports. Case-report age evidence now
comes from the abstract or case-presentation section when the caption states
none (`figures.case_age_text`).
The pre-switch database is saved as
`data/visual_pilot/visual_pilot.pre_epmc_20261001.sqlite`.

## Review-first sources and optional age (2026-10-01)

Galleries rank sources review > case series/original study > case report >
atypical (drug-induced, treatment-story or rare presentations; see
`source_quality.article_tier`), so a pair leads with broad material and
case reports fill what is left. Patient age no longer gates publication: a
stated age sorts an image into adult or pediatric, otherwise it shows as
"Not stated". The age gate had vetoed 1,764 review panels the judge accepted;
`requeue-age-vetoes` lifted them without LLM calls, raising published images
from 790 to 1,152 (reviews 41 → 235, case series/studies 71 → 256) and
pairs with images from 118 to 137 of 169. The pre-change database is saved as
`data/visual_pilot/visual_pilot.pre_age_optional_20261001.sqlite`.

The sections below describe earlier runs on the previous pipeline.

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
