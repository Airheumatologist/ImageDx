# Visual Findings Library

This repository builds a VisualDx-style medical image library: a browsable,
findings-tagged collection of disease imagery curated from PMC open-access
review articles with commercial-use licenses.

Current disease scope: **SLE**, **dermatomyositis**, **ankylosing
spondylitis** (see `src/visual_pilot/data/diseases.json`). The authoritative
spec is `docs/visual_pilot_plan.md`.

## What the pipeline does

For each disease, the pipeline:

1. selects relevant review articles from the prebuilt turbopuffer PMC chunk
   index (BM25 + dense retrieval, license + publication-type filters, P1
   relevance triage)
2. fetches and parses the JATS full text from the `pmc-oa-opendata` S3 bucket
3. triages figure captions (P2), judges figure images with a vision model
   (P3), and stores approved panels with finding attributions (P4)
4. extracts findings from text and refreshes panel-derived finding links
5. emits per-disease reports and serves a local browsable library

All LLM stages are schema-validated, budget-capped, and recorded in
`llm_calls`; every stage is idempotent and resumes from the `status` column.

## Repository layout

```text
turborag/
|-- README.md
|-- env.example
|-- requirements.txt
|-- docs/                        # plan, baselines, eval results
|-- scripts/                   # vp_* probes, smoke tests, eval harnesses
|-- src/
|   `-- visual_pilot/          # the pipeline package
|       |-- cli.py             # stage runner: init|select|parse|triage|judge|store|extract|report|serve|run-all
|       |-- config.py          # env loading + VP_* settings
|       |-- retrieval.py       # turbopuffer PMC namespace + DeepInfra embeddings
|       |-- select_articles.py # stage 2 article selection
|       |-- data/              # diseases.json, findings_vocab.json
|       `-- viewer/            # FastAPI library browser (static HTML/JS)
|-- tests/visual_pilot/
`-- data/visual_pilot/         # runtime artifacts (gitignored): sqlite, panels, thumbs, figures, reports
```

## Setup

```bash
pip install -r requirements.txt
cp env.example .env   # fill in DEEPINFRA_API_KEY + TURBOPUFFER_API_KEY
```

## Usage

```bash
python3 -m src.visual_pilot.cli init                          # create DB + seed diseases/vocab
python3 -m src.visual_pilot.cli run-all --disease all --budget-usd X
python3 -m src.visual_pilot.cli <stage> --disease all --limit N --dry-run --budget-usd X
python3 -m src.visual_pilot.cli serve --port 8765             # browse the library
```

See `src/visual_pilot/README.md` for stage details, env vars, and stage-0
findings.

## Testing

```bash
pytest tests/visual_pilot
```
