# Visual Findings Library pilot

Self-contained pilot for a VisualDx-style image library covering **SLE**,
**dermatomyositis** and **ankylosing spondylitis**, built only from PMC
open-access review articles with commercial-use licenses. Spec:
`docs/visual_pilot_plan.md`.

## Usage

```bash
# From the repo root, using the system python3.
python3 -m src.visual_pilot.cli init                 # create DB + seed diseases/vocab
python3 -m src.visual_pilot.cli <stage> --disease all --limit N --dry-run --budget-usd X
```

Stages: `init | select | parse | triage | judge | store | extract | report |
serve | run-all`. Every stage is idempotent and resumes from the `status`
column. Data lives under `data/visual_pilot/` (override with `VP_DATA_DIR`):
`visual_pilot.sqlite`, `panels/`, `thumbs/`, `figures/`, `reports/`.

Config env vars (see `env.example`): `VP_TRIAGE_MODEL` (default
`meta-llama/Llama-4-Scout-17B-16E-Instruct` — Qwen3-235B timed out / 429'd on
every batched P2 call), `VP_EXTRACT_MODEL`, `VP_JUDGE_MODEL`, `VP_TRIAGE_BATCH`
(P2 batch size, default 40), `VP_LLM_PROVIDER`, `VP_IMAGE_MAX_EDGE`, `VP_CONCURRENCY`,
`VP_NCBI_API_KEY`, `VP_DATA_DIR`. Provider keys come from the existing `.env`
(`DEEPINFRA_API_KEY`, `OPENCODE_API_KEY`, `TURBOPUFFER_API_KEY`) via
`src.config`; the default provider is DeepInfra (`VP_LLM_PROVIDER`).

## Stage 0 findings

_Checked 2026-09-25 via `scripts/vp_stage0_probe.py` on 10 real PMC OA review
articles (SLE/DM/AS) and `scripts/vp_smoke_llm.py` for the LLM checks._

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
  (`xml_text` + href→`ImageRef` resolver + `LicenseInfo`),
  `fetch_image_bytes`, `prepare_for_llm` (TIFF/other→PNG + downscale to
  `VP_IMAGE_MAX_EDGE`, in memory), `to_data_url`; per-host rate limiting
  (NCBI ≤3 req/s, ≤10 with `VP_NCBI_API_KEY`; others ~5) with retries on
  429/5xx. Nothing is written to disk.

