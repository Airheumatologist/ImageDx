# Visual Findings Library: 3-disease pilot — spec

> **Repo adjustments (authoritative, override the text below where they conflict):**
> - Repo root is `/Volumes/Vibing/Turborag` (not `/Volumes/Vibing/Review Article`). Python 3.14 system interpreter (`python3`); pillow, lxml, jsonschema, httpx, openai, turbopuffer, fastapi, pytest, ruff are installed. Add any new runtime deps (pillow, lxml, jsonschema) to `requirements.txt`.
> - There is no `tests/` dir yet; create `tests/visual_pilot/`. `.gitignore` ignores `test_*.py` except under `tests/`. Add `data/visual_pilot/` to `.gitignore`.
> - LLM provider: **OpenCode Zen** (OpenAI-compatible, `OPENCODE_API_KEY`, `OPENCODE_BASE_URL`). DeepInfra (`DEEPINFRA_API_KEY`, `DEEPINFRA_BASE_URL`) is used only for query embeddings/reranking. Defaults: `VP_TRIAGE_MODEL`/`VP_EXTRACT_MODEL`/`VP_JUDGE_MODEL` all default to `space-bunny-free` (multimodal, so it covers the vision judge). Triage batch size: `VP_TRIAGE_BATCH` (default 40).
> - Existing `frontend/` and existing `src/*.py` modules must not be modified (import/reuse only).

The orchestrator owns sections 2, 4 and 6 (decisions, schema, prompts). Implementers must not change them.

---

## 1. Goal and context

**Goal:** Build a local pilot of a VisualDx-style image library for **systemic lupus erythematosus (SLE), dermatomyositis (DM) and ankylosing spondylitis (AS)**. Each image is:
- taken only from **PMC open-access review articles** with **licenses that allow commercial use**
- judged by an LLM, which decides whether to include it and assigns disease, subtype, modality and findings
- tagged with skin tone judged from the image, plus ethnicity only when the article states it
- shown on a local disease-page viewer with attribution under every image

**Existing repo:**
- `src/retriever_turbopuffer.py`: `TurbopufferRetriever`, which queries turbopuffer PMC/PubMed namespaces. Rows are **chunk-level** (one row per text chunk, so one article appears many times) with attributes: `id, doc_id, pmcid, pmid, doi, title, page_content, abstract, section_title, section_type, chunk_id, chunk_index, journal, nlm_unique_id, year, article_type, publication_type, evidence_*, source_family, country, has_full_text`.
- `src/config.py`: loads `TURBOPUFFER_API_KEY`, `TURBOPUFFER_REGION`, `TURBOPUFFER_NAMESPACE_PMC`, embedding settings and more from `.env`.
- turbopuffer has **no license field and no figure data**. Licenses and figures come from PMC.

**Core rule:** No image is written to disk until the vision LLM accepts it. XML and images are fetched into memory and passed to the LLM, preferably as URLs. Only accepted panels and their metadata are saved. Rejected figures leave only a metadata row: caption, URL and reason.

---

## 2. Decisions

| Topic | Default |
|---|---|
| Cutaneous lupus (ACLE/SCLE/DLE) | Part of SLE with a `subtype` tag |
| Non-radiographic axSpA | Part of AS with a `subtype = nr-axSpA` tag |
| Other SpA (PsA, reactive, IBD-associated) | Excluded |
| DM subtypes | Classic, amyopathic (CADM), juvenile (JDM), anti-MDA5 and cancer-associated, all included and tagged |
| Polymyositis, IBM, necrotizing myopathy panels | Rejected (`other_disease`) |
| Drug-induced and neonatal lupus | Excluded |
| Allowed licenses | CC0, CC BY, CC BY-SA → panel crops allowed; CC BY-ND → **whole figure only, no crop**; NC, custom/other and author manuscripts → excluded |
| Third-party images ("reproduced/adapted with permission", "©", "courtesy of") | Rejected |
| Publication types | Include `Review`. Exclude any article also tagged `Case Reports`, `Meta-Analysis`, `Systematic Review`, `Clinical Trial*`, `Randomized Controlled Trial`, `Letter`, `Editorial`, `Comment`. This also removes "case report and review of the literature" papers. |
| Race | **Not inferred.** Use `skin_tone` (light/medium/dark/unknown, judged from the image, skin/nail/mucosa panels only) + `stated_ethnicity` (only with a verbatim quote) + `study_region` (metadata, never converted to race) |
| PHI checks, takedown, formal gold set, near-duplicate detection | Out of scope for the pilot. Only exact-hash dedup. |
| Storage | Local only: SQLite plus a folder of accepted images |
| Viewer | A standalone minimal FastAPI + HTML page for the pilot; existing `frontend/` untouched |
| Models | Configurable through env vars: a cheap text model for the relevance check and caption triage, a strong vision model for the judge. |

---

## 3. Architecture and layout

- **New package:** `src/visual_pilot/`
  - `config.py`: reads `.env` via the existing `src/config.py` where possible
  - `db.py`: schema and data access
  - `diseases.py` plus data files: disease seed and vocabulary
  - `select_articles.py`: stage 2
  - `pmc.py`: stages 0 and 3 (license, XML and image access, all in memory)
  - `llm.py`: provider abstraction, JSON-schema output, retries, response cache, cost tracking, batch support
  - `prompts.py`: P1–P4 constants and JSON schemas exactly as in §6
  - `triage.py`: stage 4
  - `judge.py`: stage 5
  - `store.py`: stage 6 (crop, thumbnail, write)
  - `extract_findings.py`: stage 7
  - `report.py`: stage 9
  - `viewer/`: stage 8
  - `cli.py`: entry point
- **CLI:** `python -m src.visual_pilot.cli {init|select|parse|triage|judge|store|extract|report|serve|run-all}`
  - Every stage takes `--disease {sle,dm,as,all}`, `--limit N`, `--dry-run` and `--budget-usd X`, and **resumes** from the status column.
- **Data folder:** `data/visual_pilot/` (overridable with `VP_DATA_DIR`, used by tests)
  - `visual_pilot.sqlite`
  - `panels/{sle|dm|as}/{modality}/{panel_id}.png`
  - `thumbs/{panel_id}.webp`
  - `figures/{pmcid}/{figure_file}` (original of each accepted figure)
  - `reports/`
- **New env vars:** `VP_TRIAGE_MODEL`, `VP_JUDGE_MODEL`, `VP_EXTRACT_MODEL`, `VP_LLM_PROVIDER` (opencode, default opencode), `VP_IMAGE_MAX_EDGE=1568`, `VP_NCBI_API_KEY` (optional), `VP_CONCURRENCY`, `VP_DATA_DIR`. Never commit `.env`. Document them in `env.example`.

---

## 4. Data model (SQLite)

**`diseases`**
- `disease_key` (sle/dm/as) PK, `name`, `mondo_id`, `mesh_id`, `synonyms_json`, `subtypes_json`

**`findings_vocab`**
- `finding_key` PK, `disease_keys_json`, `label`, `synonyms_json`, `category` (skin, mucosa, nail, capillaroscopy, histology, radiology_xray, ct, mri, us, echo, eye, clinical_msk), `approved` (bool)
- `proposed_by_llm` (bool), `proposal_count`

**`articles`**
- `pmcid` PK, `pmid`, `doi`, `title`, `journal`, `year`, `country`, `publication_types_json`
- `license_code`, `license_url`, `oa_subset`, `retrieval_score`
- `primary_disease_keys_json`, `relevance_decision`, `relevance_reason`
- `status`: `candidate` → `license_ok` / `license_rejected` → `relevant` / `irrelevant` → `parsed` / `parse_error`

**`figures`**
- `figure_id` PK (`{pmcid}:{fig_xml_id}`), `pmcid`, `label`, `caption`, `in_text_mentions_json`
- `fig_permissions_text`, `effective_license`, `image_url`, `image_format`, `sha256` (set when fetched)
- `status`: `pending` → `caption_kept` / `caption_uncertain` / `caption_rejected` → `vision_accepted` / `vision_rejected` / `vision_error` → `stored`
- `triage_json`, `vision_json`, `error`, `attempts`

**`panels`**
- `panel_id` PK, `figure_id`, `pmcid`, `panel_label`, `disease_key`, `subtype`, `modality`, `body_site`
- `findings_json`, `typicality`, `stage`, `age_group`
- `skin_tone`, `stated_ethnicity`, `stated_ethnicity_quote`, `study_region`
- `annotations_present`, `bbox_json`, `crop_mode` (panel / whole_figure), `confidence`, `rationale`
- `image_path`, `thumb_path`, `width`, `height`, `sha256`
- `attribution_text`, `license_code`, `license_url`, `source_url`

**`disease_findings`**
- `disease_key`, `finding_key`, `subtype`, `frequency_text`, `frequency_pct_low`, `frequency_pct_high`
- `source` (text / image), `pmcid`, `quote`

**`llm_calls`**
- `call_id`, `stage`, `model`, `input_hash`, `request_meta_json`, `response_json`
- `input_tokens`, `output_tokens`, `cost_usd`, `created_at`
- Serves as both the **cache** (skip when the `input_hash` already exists) and the cost ledger.

---

## 5. Pipeline stages

### Stage 0: Source checks (run before building stages 3–6)
Confirm and write down in the pilot README section (`src/visual_pilot/README.md`):
1. **License source.** Pick one and use it consistently:
   - (a) the PMC OA file list CSV (`ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv`, which has a License column)
   - (b) the PMC OA Web Service (`https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMCxxxx`, which returns the license and package link)
   - (c) metadata in the PMC Article Datasets on AWS (`s3://pmc-oa-opendata`, public)
2. **Figure access, in order of preference:**
   - (1) public HTTPS URLs to figure files in the AWS PMC dataset, passed straight to the LLM
   - (2) Europe PMC full-text XML (`https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML`) plus figure URLs
   - (3) the OA package `.tar.gz` fetched into memory (`tarfile` on `BytesIO`), with images sent as base64

   Record which one works.
3. **Can the LLM fetch the URLs?** Test the chosen vision provider on about 5 image URLs. If it fails, use base64 mode.
4. **TIFF share:** Measure what fraction of figures are TIFF. TIFF must be converted to PNG in memory and sent as base64.

**Done when:** a short written note of the chosen license source and figure access mode, and a working function `get_article_bundle(pmcid)` that returns XML text plus a figure-href → URL/bytes resolver without writing to disk.

### Stage 1: Diseases and vocabulary
- **Seed:** the 3 diseases with MONDO and MeSH IDs looked up from the official sources (don't guess them), synonyms and subtypes.
  - SLE: "systemic lupus erythematosus", "SLE", "lupus"; subtypes ACLE, SCLE, DLE, lupus nephritis, NPSLE.
  - DM: "dermatomyositis", "juvenile dermatomyositis", "JDM", "amyopathic dermatomyositis", "CADM", "anti-MDA5"; subtypes classic, CADM, JDM, anti-MDA5, cancer-associated.
  - AS: "ankylosing spondylitis", "axial spondyloarthritis", "axSpA", "radiographic axial spondyloarthritis", "Bechterew"; subtypes r-axSpA, nr-axSpA.
- **Vocabulary:** Seed `findings_vocab` (all `approved=1`) with:
  - **SLE:** malar rash (nasolabial sparing), discoid plaque, scarring alopecia (DLE), non-scarring alopecia, SCLE annular, SCLE papulosquamous, oral/palatal ulcer, livedo reticularis, cutaneous vasculitis, Raynaud phenomenon, periungual erythema, Jaccoud arthropathy, interface dermatitis, dermal mucin, lupus band (DIF), lupus nephritis class I–VI, wire-loop lesion, full-house immunofluorescence, pleural effusion, pericardial effusion, NPSLE white-matter lesions, Libman–Sacks vegetation, tortuous capillaries (capillaroscopy)
  - **DM:** heliotrope rash, Gottron papules, Gottron sign, V-sign, shawl sign, holster sign, mechanic's hands, periungual erythema, ragged cuticles (Samitz sign), poikiloderma, scalp dermatomyositis, calcinosis cutis, flagellate erythema, MDA5 palmar papules, MDA5 cutaneous ulcers, dilated/giant capillaries, capillary dropout, capillary hemorrhages, perifascicular atrophy, perivascular inflammation, MxA expression, MAC deposition, muscle edema (STIR MRI), fascial edema, calcinosis (radiograph), ILD NSIP pattern, ILD organizing-pneumonia pattern, rapidly progressive ILD
  - **AS:** sacroiliitis (radiograph grade), SI erosions, SI sclerosis, SI ankylosis, syndesmophytes, bamboo spine, Romanus lesion, vertebral squaring, dagger sign, enthesophyte, SI bone marrow edema (STIR), fat metaplasia, backfill, corner inflammatory lesion, corner fat lesion, hyperkyphosis, loss of lumbar lordosis, anterior uveitis, hypopyon, Achilles enthesitis, dactylitis

**Done when:** `cli init` creates the DB and seeds are loaded idempotently.

### Stage 2: Article selection
1. **Query turbopuffer** (PMC namespace) per disease with each synonym. Use title BM25 and hybrid dense+BM25 over abstract and text, reusing `TurbopufferRetriever` helpers or the tpuf client directly. Filters: `has_full_text = true`, `publication_type` contains `Review`.
   - Check first how `publication_type` is stored (list vs string) before writing filters, and apply exclusions in Python if server-side filtering can't express them.
2. **Collapse chunks to articles** by `pmcid`, keeping the best score.
3. **Exclude** any article whose publication types include an excluded type (section 2).
4. **Join licenses** from stage 0 and keep only allowed codes.
5. **Relevance check:** All licensed candidate reviews go through prompt P1 on title plus abstract. A bounded caption check may rescue an otherwise excluded broad review when an eligible figure specifically covers a pilot disease.
6. **Checkpoint:** Print and save `reports/stage2_counts.json` with, per disease: candidates, after type filter, after license filter, relevant. This file is a funnel report; article count no longer blocks stage 3.

7. **Visual retrieval ranking:** Rank with retained disease/finding/modality query passages, source section, title and abstract. On a bounded shortlist, inspect JATS captions and favor eligible clinical, imaging and histology figures that match uncovered findings. Keep broad reviews when a specific figure supplies relevant evidence. The JATS check follows the existing third-party, image availability and license rules; uncertain captions remain available to stage 4.

8. **Yield expansion:** `run-all` processes per-disease batches of 50 and compares newly stored distinct image hashes and newly covered approved findings after triage, vision review and storage. It continues while batches add either kind of coverage, and stops after two consecutive zero-yield batches. Defaults cap a run at 600 articles per disease and 900 seconds; configure with `--batch-size`, `--max-articles`, `--max-runtime-seconds`, and `--zero-yield-batches`. Runs resume from database status. Standalone `parse` processes one ranked batch; rerun it to continue.

**Done when:** the `articles` table is filled, the counts file exists, and a unit test covers publication-type exclusion and chunk collapse.

### Stage 3: In-memory figure parsing
For each relevant article:
1. Fetch the JATS XML into memory and parse it with `lxml`.
2. For each `<fig>`, extract:
   - `@id`, `<label>`, the full `<caption>` text (title plus paragraphs)
   - the `<graphic xlink:href>`
   - figure-level `<permissions>` (license, copyright statement)
   - every body paragraph containing an `<xref ref-type="fig" rid=...>` pointing to it, trimmed to about 600 characters each, at most 3
3. **`effective_license`:** the figure-level license if present, otherwise the article license. If figure permissions contain a copyright holder other than the authors or journal, or any "reproduced/adapted with permission" wording, set `status = caption_rejected` with reason `third_party`.
4. Resolve `image_url` (or mark `needs_bytes`). Don't download.
5. **`study_region`:** taken from the article `country` attribute or the corresponding-author affiliation country in the XML. Label which source was used.
6. Also keep the body section texts in memory for stage 7, or re-fetch them there. **Don't store full text on disk.**

**Done when:** `figures` rows exist, and a unit test runs a fixture JATS file (the only fixture on disk, in `tests/fixtures/`) covering captions, xrefs, permissions and multi-paragraph captions.

### Stage 4: Caption triage (text LLM, batched)
- Send 30–50 figures per call using prompt P2, and put the results in `figures.triage_json`.
- **Routing:**
  - `drop` or `third_party = true` → `caption_rejected`
  - `keep` → `caption_kept`
  - `uncertain` → `caption_uncertain`

  Contradictory model output that says `drop` while explicitly describing an
  eligible target-disease patient image is routed to `caption_uncertain` for
  vision review. The next triage run also re-queues matching historical P2
  rejections. Third-party and disallowed-license content remains rejected.

  `caption_kept` and `caption_uncertain` both go on to stage 5.

**Done when:** every `pending` figure has a triage result; the caption-rejection counts by reason are in the report.

### Stage 5: Vision judge
- **One call per figure**, prompt P3. Input:
  - the image (URL, or in-memory base64 after TIFF→PNG conversion and downscaling to `VP_IMAGE_MAX_EDGE`)
  - label, caption, in-text mentions
  - article title and the `primary_disease_keys` list
  - the vocabulary filtered to the article's diseases
- Validate the output against the JSON schema; on failure, retry once with the validation error attached. Store it in `figures.vision_json`.
- **Status:** `vision_accepted` if at least one panel has `include=true`, otherwise `vision_rejected`.
- **Batching:** Use the provider's batch API when available (requests built in memory).

**Done when:** schema validation is enforced, the cache prevents paying twice for the same input, and costs are logged.

### Stage 6: Save accepted panels
For each `vision_accepted` figure:
1. Fetch the original full-resolution bytes into memory, or reuse them.
2. Write the original to `figures/{pmcid}/`.
3. **Crop each included panel** from the normalized bbox, scaled to the original's pixel size, with 2% padding. Save as PNG under `panels/{disease}/{modality}/` plus a WebP thumbnail with a 400 px long edge.
4. **Whole-figure fallback** (`crop_mode = whole_figure`) when any of these holds:
   - the license is ND
   - the bbox is missing or covers under 3% of the figure area
   - panel bboxes overlap by more than 30%
5. **Exact dedup:** If a panel's image `sha256` already exists, link the new row to the existing file (same image in another review). Keep one file and store every source attribution.
6. **Attribution text:** `"{Authors et al.} {Title}. {Journal} {Year}. doi:{doi}. {License} ({license_url}). Figure {label}{panel}."`
7. **Proposed findings:** Increment `proposal_count` in `findings_vocab` with `approved=0` for each proposed finding. They're not shown in the viewer until a human approves them.

**Done when:** files exist and the DB paths resolve; a unit test covers bbox scaling, the fallback rules and dedup.

### Stage 7: Findings from review text
- For each relevant article, run prompt P4 on the body sections covering clinical features, diagnosis, imaging, histopathology and classification. Pick sections by title keywords, with the whole body capped at about 30k tokens as fallback.
- Save the extracted disease → finding assertions in `disease_findings` (`source=text`) with quotes.
- Also derive `source=image` rows from `panels`.

### Stage 8: Viewer
- A FastAPI app (`cli serve`) serving `/api/diseases`, `/api/diseases/{key}/panels?modality=&subtype=&skin_tone=&finding=&typicality=` and the static images, plus one HTML page per disease.
- **Tabs:**
  - **SLE:** Skin (grouped by ACLE / SCLE / DLE), Mucosa, Musculoskeletal, Renal histology, Skin histology/DIF, Imaging, Capillaroscopy
  - **DM:** Skin (grouped by sign), Nailfold/Capillaroscopy, Muscle histology, MRI, Lung CT, Calcinosis. Subtype filter: JDM / CADM / anti-MDA5.
  - **AS:** SI radiograph, SI MRI, Spine radiograph/CT, Clinical, Eye. Grouped by stage: nr-axSpA → early r-axSpA → advanced.
- **Sorting:** classic before variant before atypical, then by confidence.
- **Filters:**
  - skin-tone filter and group headers on the SLE and DM skin tabs
  - a finding filter using approved vocabulary only
- **Each card shows:** thumbnail (click for full size), disease, subtype, findings, skin tone, stated ethnicity (if any), study region, and the attribution line with a DOI link.
- **"Key findings" panel:** the top `disease_findings` rows with quotes.
- **SLE vs DM skin comparison page:** side by side, e.g. Gottron papules vs SLE knuckle-sparing rash, heliotrope rash vs malar rash.

### Stage 9: Pilot report
Written to `reports/pilot_report.md` and `.json`:
- **Funnel per disease:** candidates → type filter → license filter → relevant → figures → caption kept/uncertain/rejected (by reason) → vision accepted/rejected (by reason) → panels stored
- **Counts per disease:**
  - panels per modality, subtype and finding
  - findings from the vocabulary with **zero** images (coverage gaps)
  - skin-tone distribution for skin panels
- **Costs:** total and per accepted panel, per stage
- **Access failures:** URL-mode fetch failures and TIFF conversions
- **Spot-check sheets:**
  - every accepted panel (thumbnail + tags), for full review, since the pilot is small
  - every caption-rejected figure (caption + reason + URL), to judge whether triage is safe to keep at scale

---

## 6. Prompts (use verbatim; tune only after the pilot report)

All calls: temperature 0, strict JSON-schema output, and the system prompt ends with "Return only JSON."

**P1: Article relevance (text model)**
> System: You screen open-access medical review articles for an image library covering three diseases: systemic lupus erythematosus (sle; includes cutaneous lupus subtypes ACLE/SCLE/DLE, lupus nephritis, neuropsychiatric lupus), dermatomyositis (dm; includes juvenile, amyopathic, anti-MDA5, cancer-associated), and ankylosing spondylitis (as; includes radiographic and non-radiographic axial spondyloarthritis). Given a title and abstract, decide which of these diseases the article substantially covers (a main topic, or a major section devoted to it). Passing mentions do not count. Mark `is_narrative_review` false if the article is actually a case report, systematic review, meta-analysis, trial, guideline methodology paper, or basic-science-only paper with no clinical presentation content.
>
> Schema: `{"primary_disease_keys": ["sle"|"dm"|"as"], "is_narrative_review": bool, "decision": "relevant"|"irrelevant", "reason": str}`

**P2: Caption triage (text model, batched)**
> System: You triage figure captions from open-access medical review articles about SLE, dermatomyositis, or ankylosing spondylitis. For each figure, decide from the label, caption and in-text mentions alone whether it likely contains at least one real-patient image: clinical photograph, dermoscopy, nailfold capillaroscopy, histopathology, immunohistochemistry, immunofluorescence, cytology, radiograph, CT, MRI, ultrasound, echocardiogram, PET, endoscopy, ophthalmic image, or gross specimen. Non-patient content includes diagrams, schematics, pathways, mechanism figures, flowcharts, algorithms, charts/graphs, tables, drawings/illustrations, and photos of equipment. A multi-panel figure counts as `keep` if any panel likely qualifies. Set `third_party` true if the caption says the image is reproduced, adapted, reprinted or used with permission from another source, carries a copyright notice (©), or is "courtesy of" someone, and quote the phrase. Use `uncertain` when the caption does not say what the image shows (e.g. "Representative case"). Never guess `drop` for an image-like caption.
>
> Schema: `{"results": [{"figure_id": str, "category": "clinical_photo"|"dermoscopy"|"capillaroscopy"|"histology"|"immunofluorescence"|"radiology"|"ultrasound"|"echo"|"endoscopy"|"ophthalmic"|"gross"|"mixed"|"diagram"|"chart"|"flowchart"|"table"|"illustration"|"other", "is_real_patient_image": true|false|null, "third_party": bool, "third_party_quote": str|null, "diseases_mentioned": ["sle"|"dm"|"as"|"other"], "route": "keep"|"drop"|"uncertain", "reason": str}]}`

**P3: Vision judge (vision model, one figure per call)**
> System: You curate images for a clinical visual-diagnosis library covering only: sle (subtypes: ACLE, SCLE, DLE, lupus_nephritis, NPSLE, other_systemic), dm (classic, CADM, JDM, anti_MDA5, cancer_associated), as (r_axSpA, nr_axSpA). You receive one figure from an open-access review article, its caption, the in-text mentions, the article's topic diseases, and an allowed findings vocabulary.
>
> Rules:
> 1. Identify every panel (use the panel letters in the image or caption; a single-image figure is panel "A"). Give each panel a tight normalized bbox [x0, y0, x1, y1] in 0–1 image coordinates.
> 2. `include` = true only if the panel is a real-patient image that visibly shows a finding of sle, dm or as. Exclude: diagrams or illustrations, charts, normal or control images, other diseases (including comparison panels of other diseases, polymyositis, inclusion body myositis, psoriatic arthritis), unreadable quality, or images where the caption indicates third-party copyright.
> 3. Tag the disease **the panel shows**, using the caption as evidence, not simply the article's topic. Review figures often contrast diseases.
> 4. `findings`: use only `finding_key` values from the vocabulary provided, each with the caption or in-text phrase that supports it (or "visual" if you identified it only from the image). Put anything clearly present but missing from the vocabulary in `proposed_findings` as short clinical terms.
> 5. `typicality`: classic (textbook presentation), variant (recognized less common form), atypical (unusual; the caption usually says so).
> 6. `skin_tone`: only for panels showing skin, nails, lips or oral mucosa. Judge only from visible, adequately lit skin: light (≈ Fitzpatrick I–II), medium (III–IV), dark (V–VI), unknown if not assessable. Never infer it from the country, the caption or the journal. Use null for non-skin panels.
> 7. `stated_ethnicity`: only if the caption or in-text mentions explicitly state it; give the verbatim quote. Otherwise null. Never infer ethnicity or race.
> 8. `age_group`: child, adolescent, adult, older_adult, unknown. Use the caption, or clear visual cues for children.
> 9. `stage`: for AS use nr_axSpA, early, advanced (ankylosis or bamboo spine) or unknown; for others, a short text or null.
> 10. `confidence` 0–1 reflects the disease attribution and findings together.
>
> Schema: `{"figure_id": str, "figure_is_compound": bool, "panels": [{"panel_label": str, "bbox": [n,n,n,n], "include": bool, "exclusion_reason": null|"diagram"|"chart"|"normal_control"|"other_disease"|"poor_quality"|"third_party"|"not_patient_image", "disease_key": "sle"|"dm"|"as"|null, "subtype": str|null, "modality": "clinical_photo"|"dermoscopy"|"capillaroscopy"|"histology_he"|"histology_ihc"|"immunofluorescence"|"radiograph"|"ct"|"mri"|"ultrasound"|"echo"|"pet"|"endoscopy"|"ophthalmic"|"gross"|"other", "body_site": str|null, "findings": [{"finding_key": str, "evidence": str}], "proposed_findings": [str], "typicality": "classic"|"variant"|"atypical"|null, "stage": str|null, "age_group": str, "skin_tone": "light"|"medium"|"dark"|"unknown"|null, "stated_ethnicity": str|null, "stated_ethnicity_quote": str|null, "annotations_present": bool, "confidence": number, "rationale": str}]}`

**P4: Findings from text (text model)**
> System: From the review text sections provided, extract statements that link one of sle, dm or as to a clinical, imaging, histologic or capillaroscopic finding. Map each to a `finding_key` from the provided vocabulary when possible, otherwise put it in `proposed_finding`. Capture the stated frequency exactly as written (e.g. "30–60%", "most patients", "pathognomonic") and parse it into numeric low/high percentages only when numbers are explicit. Include a verbatim quote of 40 words or fewer. Do not add facts that are not in the text.
>
> Schema: `{"assertions": [{"disease_key": str, "subtype": str|null, "finding_key": str|null, "proposed_finding": str|null, "frequency_text": str|null, "pct_low": number|null, "pct_high": number|null, "specificity_text": str|null, "quote": str}]}`

---

## 7. Workstreams

| # | Workstream | Depends on | Delivers | Done when |
|---|---|---|---|---|
| **W1** | Foundations | none | package skeleton, `config.py`, `db.py` (schema §4, idempotent `init`), `cli.py` skeleton with shared flags, status helpers, seed files and loader (stage 1) | `cli init` works twice without error; unit tests for schema and seeding pass; ruff clean |
| **W2** | Source checks + PMC access | W1 | stage 0 findings note; `pmc.py` with `get_license(pmcid)`, `get_article_bundle(pmcid)`, `resolve_image(href) -> url or bytes`, TIFF→PNG and downscale in memory; polite rate limiting (NCBI ≤3 req/s without key, ≤10 with key) and retries | stage 0 checks pass on 10 real PMCIDs, each in URL and bytes mode; no files written (verified by test with a temp-dir check) |
| **W3** | Article selection | W1, W2 (license) | stage 2 incl. P1 call via W4's `llm.py`, counts checkpoint | counts file for 3 diseases; unit tests for type exclusion and chunk collapse |
| **W4** | LLM layer | W1 | `llm.py`: provider abstraction (text + vision, URL and base64 images), JSON-schema enforcement with one repair retry, `llm_calls` cache and cost ledger, batch-API support, budget guard that stops when `--budget-usd` is reached; prompt constants P1–P4 exactly as in §6 | mocked tests for cache hit, schema repair, budget stop; one live call per prompt in a smoke script |
| **W5** | Parsing + triage | W2, W4 | stage 3 (`parse`) and stage 4 (`triage`) | fixture JATS test passes; triage run on 5 real articles fills `triage_json` |
| **W6** | Judge + store | W2, W4, W5 | stage 5 (`judge`) and stage 6 (`store`) | 5 real articles end to end; crops look right; ND whole-figure rule and dedup tested |
| **W7** | Text findings | W4, W5 | stage 7 (`extract`) | `disease_findings` rows with quotes for the 5 test articles |
| **W8** | Viewer | W1 (schema); synthetic rows, then real data | stage 8 (`serve`) | 3 disease pages + comparison page render with filters; attribution visible on every card |
| **W9** | Report | W3, W5–W7 | stage 9 (`report`) + spot-check sheets | report generated from the DB with no LLM calls |

## 8. Run order for the pilot
1. `cli init`
2. `cli run-all --disease all --budget-usd <X>` to select, parse and review successive yield batches.
3. `cli report`, then `cli serve` if running the stages separately.
4. Human review: accepted panels, caption-rejected figures, proposed findings, and coverage gaps.

## 9. Testing and quality rules
- Unit tests make **no network calls**; the only fixtures on disk are the small JATS sample and synthetic images.
- A test enforces that no image is written to disk before stage 6 (check the data folder before and after stages 3–5).
- Every stage is idempotent and resumable. A rerun makes zero new LLM calls when the inputs haven't changed.
- `ruff check src/visual_pilot tests/visual_pilot` and `pytest tests/visual_pilot` pass before each handoff. Secrets come only from `.env`.

## 10. Known risks
- Figure-level third-party material not clearly worded in the caption. Review flagged edge cases by hand; legal review before any commercial use.
- VLM bboxes can be imprecise → whole-figure fallback + manual review.
- Disease yield can differ → review per-disease counts, distinct stored images and remaining finding coverage after each run.
- Likely coverage gaps: darker-skin examples of malar rash, heliotrope rash, Gottron papules; AS clinical photos.
- Unconfirmed: PMC S3 layout and whether the LLM can fetch URLs (stage 0).
