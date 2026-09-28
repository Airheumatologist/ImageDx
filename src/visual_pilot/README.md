# Visual Findings Library pilot

Self-contained pilot for a VisualDx-style image library covering **SLE**,
**dermatomyositis**, **ankylosing spondylitis**, **rheumatoid arthritis**,
**systemic sclerosis**, **psoriasis**, **psoriatic arthritis**, **sarcoidosis**,
**gout**, and **atopic dermatitis**, built only from PMC open-access review
articles with commercial-use licenses. The disease index and approved visual
findings are in `data/diseases.json` and `data/findings_vocab.json`. Spec:
`docs/visual_pilot_plan.md`.

## Usage

```bash
# From the repo root, using the system python3.
python3 -m src.visual_pilot.cli init                 # create DB + seed diseases/vocab
python3 -m src.visual_pilot.cli <stage> --disease all --limit N --dry-run --budget-usd X
python3 -m src.visual_pilot.cli run-all --disease all --budget-usd X
```

Stages: `init | select | parse | triage | judge | store | extract | report |
serve | run-all`. Every stage is idempotent and resumes from the `status`
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
passed the current review-article, license, and single-panel criteria.

Article selection queries every approved, disease-specific finding by default.
`VP_VISUAL_QUERY_CAP=0` means all findings; a positive cap bounds the set and
prioritizes findings with fewer stored panels. Per synonym, title BM25 retrieves
up to 500 results, page-content BM25 up to 750, and dense ANN up to 750. Each
visual finding query retrieves up to 300 results. Turbopuffer returns compact
job-specific projections with a server-side maximum of one row per PMCID, and
retrieval jobs are batched with `Namespace.multi_query` when available (ordered
sequential fallback otherwise). Only shortlisted unique PMCIDs are hydrated
with citation metadata and abstracts in bounded batches; page-content evidence
queries request passage and section fields without abstracts.

With finite `--limit`, publication type filtering happens before the cap. The
shortlist first reserves distinct candidates round-robin across finding lanes,
up to `VP_MANIFESTATION_QUOTA` (default 20 unique articles per finding), then
fills remaining slots by global RRF. Selected finding ranks are persisted in
`manifestation_candidates`; the downstream queue tracks per-finding lane
status in `manifestation_lanes`. Logical Turbopuffer bytes queried and returned,
request count, and query count are printed and recorded under
`_turbopuffer_billing` in `reports/stage2_counts.json`.

Report and viewer rebuild one deterministic primary representative for each
covered approved disease/finding pair while retaining all eligible alternatives.
Existing manual or locked representative selections are preserved.

Matching passages and section labels are retained in
`articles.retrieval_evidence_json`. Article ranking uses that evidence and a
bounded JATS caption check; figure ranking orders the vision queue by image
relevance and coverage gaps. Each `run-all` batch is triaged, judged, and stored
before the next batch is selected. It continues while a
batch adds distinct stored images or covers new approved findings, stopping
after two empty-yield batches. `--batch-size`, `--max-articles`,
`--max-runtime-seconds`, and `--zero-yield-batches` set safety limits. A
standalone `parse` invocation processes one ranked batch and can be rerun.

Config env vars (see `env.example`): `VP_TRIAGE_MODEL`, `VP_EXTRACT_MODEL`,
`VP_JUDGE_MODEL` (all default to `stealth/space-bunny-alpha` on OpenRouter),
`VP_TRIAGE_BATCH` (P2 batch size, default 40), `VP_LLM_PROVIDER` (default `openrouter`),
`VP_IMAGE_MAX_EDGE`, `VP_CONCURRENCY`,
`VP_NCBI_API_KEY`, `VP_VISUAL_QUERY_CAP` (default 0, all approved findings),
`VP_MANIFESTATION_QUOTA` (default 20), `VP_DATA_DIR`. Provider keys come from `.env`
(`OPENROUTER_API_KEY`, `DEEPINFRA_API_KEY`, `TURBOPUFFER_API_KEY`) via
`config.py`; the primary LLM provider is OpenRouter (`VP_LLM_PROVIDER=openrouter`),
while DeepInfra is used only for query embeddings.

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
- **License source (chosen):** the per-article metadata JSON
  (`license_code`, `is_pmc_openaccess`) — spec option (c). Fallback: the
  `<permissions>` license ref in the JATS XML. Normalized to
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
