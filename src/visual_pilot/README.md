# Visual Findings Library pilot

Self-contained pilot for a VisualDx-style image library covering **SLE**,
**dermatomyositis**, **ankylosing spondylitis**, **rheumatoid arthritis**,
**systemic sclerosis**, **psoriasis**, **psoriatic arthritis**, **sarcoidosis**,
**gout**, and **atopic dermatitis**, built only from PMC open-access articles
(case reports, original research and reviews) with commercial-use licenses. The disease index and approved visual
findings are in `data/diseases.json` and `data/findings_vocab.json`. Spec:
`docs/visual_pilot_plan.md`.

## Usage

```bash
# From the repo root, using the system python3.
python3 -m src.visual_pilot.cli init                 # create DB + seed diseases/vocab
python3 -m src.visual_pilot.cli <stage> --disease all --limit N --dry-run --budget-usd X
python3 -m src.visual_pilot.cli run-all --disease all --budget-usd X
```

Stages: `init | discover | triage | judge | store | describe | extract |
report | serve | run-all`. Every stage is idempotent and resumes from the `status`
column. Data lives under `data/visual_pilot/` (override with `VP_DATA_DIR`):
`visual_pilot.sqlite`, `panels/`, `thumbs/`, `figures/`, `reports/`.

## Fresh data run

The previous pilot data and generated artifacts were cleared on 2026-09-27.
The replacement ten-disease run completed on 2026-09-28 UTC with 420 parsed
articles, 167 saved images, 123 published images, and 6,486 text finding
assertions. All ten diseases have published images. The audit saved 44
reversible exclusions and a pre-curation database backup.

The run used `run-all --disease all --limit 80 --batch-size 20
--max-articles 60 --max-runtime-seconds 7200 --budget-usd 5`, followed by
`curation_audit --disease all --apply` and `report`. Use
`python3 -m src.visual_pilot.cli serve --port 8765` to preview the library.
Reports and verification results live in `data/visual_pilot/reports/`.

The 2026-09-28 image expansion then increased the candidate limit to 240 and
finally 480 per disease, with a per-run article cap of 180. Caption rescue used
100-candidate inspection passes and a targeted eight-article tail. It finished
with 1,408 parsed review articles (3.35× the original), 287 saved images, and
184 published images (61 more than the initial run). All 6,486 original text
finding assertions and existing images were preserved. The same clinical-image
publication criteria applied. See
`data/visual_pilot/reports/expansion_3x/expansion_report.md` for the full
commands and per-disease counts; `viewer_qa.json` in that directory verifies
all ten pages and all 574 saved media routes.

The subsequent full regeneration used dynamic approved finding queries,
800 retrieval candidates per disease, 100-article processing batches, and a
360-article per-disease run limit. It parsed 1,767 fresh articles and extracted
22,054 text finding records. A merge preserved all 184 previously approved
panels, including 20 whose new judgments lost the prior supported finding or
did not produce a panel. The final curated library has 1,768 parsed articles,
307 saved panels, 204 published panels across all ten diseases, and 27,987
text finding records. The image audit and page/media QA are recorded in
`data/visual_pilot/reports/regeneration_summary.md`. Psoriasis has 29
published panels; its tabs display image sections with content plus an Eye
article-evidence section for documented uveitis. No psoriasis ocular photo
passed the current review-article, license, and publication criteria.

## Discovery (Europe PMC figure-caption search)

`discover` plans every approved (disease, finding) pair whose published
gallery count (`gallery.published_coverage`) is under
`VP_FINDING_IMAGE_TARGET`, fewest images first, and searches in one of three
passes (`--pass`): `overview` runs one query per disease for narrative
reviews and case series whose title surveys its presentation and whose
captions name any of its findings; `manifestation` restricts each pair's
query to reviews and case series whose title or abstract names the finding;
`backfill` (the default) accepts any article type. The first two exclude
systematic reviews and meta-analyses and skip atypical sources, and every
pass takes hits reviews first (`source_quality.article_tier`). For each pair the backfill pass queries the
[Europe PMC REST API](https://europepmc.org/RestfulWebService) with the
finding in a figure caption (`FIG:`), the disease in the title or abstract,
`OPEN_ACCESS:y`, `IN_PMC:y` and a CC license clause. Caption phrasings come
from `pair_terms.caption_terms` (vocabulary labels and synonyms without
disease/modality words, singular forms, and `CAPTION_TERM_OVERRIDES`). The
exact-phrase tier runs first; when it returns fewer than `--per-pair` new
articles, a words tier requires every content word of a term in the same
caption (`FIG:(sacroiliac* AND erosion*)`).

Core records carry license, publication types and retraction data; they are
cached in `article_source_metadata` (`source_quality.normalize_record`) and
the license is re-checked locally with `pmc.license_allows`. Every article
type is eligible except notices (errata, corrections, retractions). Articles
are fetched in memory from the PMC open-data S3 bucket and parsed into
`figures`; only figures whose caption names an approved finding of the
disease are `pending` for triage. For case reports (typed as a case, or an
abstract that reports a case), the abstract and case-presentation sections
are stored in `figures.case_age_text` as patient-age evidence when the
caption states none. Age only sorts images into adult or pediatric; an image
without a stated age is still published ("Not stated").

Each query is logged in `pair_search_attempts` (`policy_version`
`epmc-fig.v1`; overview queries under finding key `_overview`). `run-all`
runs one overview round, one manifestation round, then backfill rounds
(`--max-rounds`, default 3; `--skip-review-passes` goes straight to
backfill), each followed by triage, judge and store on the new articles in batches
of `--batch-size` (default 50), and stops early when a round finds nothing
new or `--max-runtime-seconds` / `--budget-usd` is reached.

Report and viewer rebuild one deterministic primary representative for each
covered approved disease/finding pair while retaining all eligible alternatives.
Existing manual or locked representative selections are preserved.

Config env vars (see `env.example`): `VP_TRIAGE_MODEL`, `VP_EXTRACT_MODEL`,
`VP_JUDGE_MODEL` (all default to `stealth/space-bunny-alpha` on OpenRouter),
`VP_TRIAGE_BATCH` (P2 batch size, default 40), `VP_LLM_PROVIDER` (default
`openrouter`), `VP_IMAGE_MAX_EDGE`, `VP_CONCURRENCY`, `VP_FETCH_CONCURRENCY`,
`VP_NCBI_API_KEY`, `VP_DATA_DIR`. Provider keys come from `.env`
(`OPENROUTER_API_KEY`, or `OPENCODE_API_KEY` / `DEEPINFRA_API_KEY` for the
alternative chat providers) via `config.py`. Discovery needs no key.

## Whole-figure plates and coverage targets

Since `clinical-panels.v5`, a multi-panel figure whose panels are all
human-patient images of the same configured disease is stored once as a
whole-figure plate (`panel_label='whole'`, `crop_mode='whole_figure'`,
`bbox=[0,0,1,1]`). Per-panel judge labels stay in `vision_json` as
classification metadata; tiles are never cropped. A plate depicting one
approved finding (`plate_kind='same_finding'`) credits that pair once;
a plate depicting several (`plate_kind='combined'`, listed only under
"Combined views" in the viewer) credits no pair — its findings live in
`plate_findings_json`. Unlabeled single-disease plates, plates mixing a
patient image with a chart/diagram, and multi-disease plates stay
unpublished.

`requeue-plates` (`--disease`, `--pmcids`, `--dry-run`) recomputes plates
deterministically for license-allowed `vision_rejected` compound figures —
no LLM calls — and flips publishable ones to `vision_accepted` for `store`.
`triage --retriage-montages` returns caption-rejected montage/collage drops
whose reason also names a patient-image modality to `pending` once per P2
version, so the judge can evaluate them under the whole-plate rule.

Coverage is measured as distinct published images per approved
(disease, finding) pair (sha256-distinct), targeting
`VP_FINDING_IMAGE_TARGET` (default 10, CLI `--finding-image-target`). A
pair is `covered` only at target; `discover` searches only pairs under
target. Recommended growth run:
`run-all --disease all --per-pair 25 --max-rounds 3 --budget-usd X`. The
report's "Per-pair image coverage" section shows the images/pair histogram
and per-disease under-target pairs.

## Display captions and sections (`describe`, prompt P5)

Article captions are written for the article, with figure and panel letters,
citation marks ("tendon.19 A"), cross-references and permission notes. After
`store`, `describe` sends each new panel's caption, mentions, panel label and
judge metadata, plus the disease's viewer sections, to P5
(`VP_DESCRIBE_MODEL`, default `VP_EXTRACT_MODEL`). The model writes a
standalone `display_title` and a 1–2 sentence `display_description` limited
to that image, and picks `display_section` (a viewer tab key) and
`display_subsection` (a listed group, such as an SLE skin group or an AS
stage, or a finding key for finding-grouped tabs). Choices outside the listed
options are stored as null. The viewer shows the description as Context, keeps
the article caption under Source, and falls back to rule routing when no
valid section was chosen. `run-all` runs it after every `store` batch; use
`describe --force` to rewrite existing captions after a P5 change.

Treatment images are excluded: caption triage (P2) drops before/after,
drug-response, follow-up healing, intraoperative/postoperative, injection and
device figures before download. P5 also returns `treatment_related`; a flagged
panel gets a `panel_curation` exclusion (reason `treatment_related`) and its
crop, thumbnail and figure original are deleted unless a published panel
still uses them. Disease a drug caused (e.g. drug-induced lupus) is kept.

## Stage 0 findings

_Checked 2026-09-25 on 10 real PMC OA review articles (SLE/DM/AS), with
separate LLM access checks. The one-off probe scripts and pilot outputs
were retired during the 2026-09-27 data cleanup._

- **PMC infrastructure (2025 reorg).** The old endpoints are dead:
  `oa.fcgi` 404s, `oa_file_list.csv` 404s, `oa_package` tarballs gone,
  `oa_comm/xml/all/` empty. Everything now lives in the public S3 bucket
  `s3://pmc-oa-opendata` under per-article version dirs
  `{pmcid}.{version}/` containing `{pmcid}.{version}.json` (metadata incl.
  license + `media_urls`), `.xml`, `.txt`, `.pdf` and the figure files under
  their real names.
- **License source (chosen):** the Europe PMC core record returned by the
  discovery search (`license` field) — treated as final for the article;
  figure-level JATS `<permissions>` override it per figure. Normalized to
  `cc0|cc-by|cc-by-sa|cc-by-nd|cc-by-nc*|other|none` by
  `pmc.normalize_license` (`license_allows` → `crop` / `whole_figure` /
  excluded per §2).
- **Figure access (chosen):** direct public HTTPS URLs into
  `pmc-oa-opendata.s3.amazonaws.com/{pmcid}.{version}/{file}` (spec
  preference 1) — resolver maps `<graphic xlink:href>` to dir contents via
  `media_urls` or ListObjectsV2. Unresolved graphics get `needs_bytes` (the
  Europe PMC `/bin/` figure endpoint 403s; Europe PMC `fullTextXML` is the
  XML fallback only).
- **LLM URL fetch: FAILS on DeepInfra.** The S3 bucket serves figures as
  `binary/octet-stream`; DeepInfra rejects them ("must serve a supported
  image MIME type"), 0/5 fetched. **Base64 mode is required** — confirmed
  working (`prepare_for_llm` + `to_data_url`, strict `json_schema` accepted).
- **Probe results:** 10 articles → licenses: 7 `cc-by`, 3 `cc-by-nc`
  (excluded); 38/38 graphic hrefs resolved to S3 URLs, 0 needs_bytes;
  image formats seen: jpg, webp; **TIFF share 0/38 (0%)** — TIFF→PNG support
  is implemented anyway.
- **API:** `get_license`, `license_allows`, `get_article_bundle`
  (`xml_text` + href→`ImageRef` resolver + `metadata`),
  `fetch_image_bytes`, `prepare_for_llm` (TIFF/other→PNG + downscale to
  `VP_IMAGE_MAX_EDGE`, in memory), `to_data_url`; per-host rate limiting
  (NCBI ≤3 req/s, ≤10 with `VP_NCBI_API_KEY`; others ~5) with retries on
  429/5xx. Nothing is written to disk.
