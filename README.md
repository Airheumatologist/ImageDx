# Elixir Medical RAG Runtime

This repository serves the runtime layer for the Elixir medical RAG system.

It includes:

- a FastAPI backend for retrieval, reranking, answer generation, streaming, and backpressure
- a Next.js frontend for the chat experience
- a turbopuffer-based retriever over prebuilt PMC, PubMed, and DailyMed namespaces
- xAI-backed generation and query decomposition, with DeepInfra-backed embeddings and reranking
- a SQLite query cache for repeat requests

This repo does not build the indexes. It assumes the turbopuffer namespaces already exist and are populated.

## What This System Does

For each user query, the runtime:

1. preprocesses the question into retrieval-ready search queries
2. branches into standard literature retrieval or USMLE option-aware retrieval
3. retrieves evidence from PMC and PubMed, and optionally DailyMed for drug-specific questions
4. reranks and aggregates the literature evidence when the standard path is used
5. generates a grounded answer with citations
6. checks citation availability and writes the result to cache

## Architecture

```text
Browser
  |
  v
Next.js frontend
  |
  v
frontend/src/app/api/chat/stream/route.ts
  |
  v
FastAPI backend (src/api_server.py)
  |
  +-- request backpressure + inflight capacity
  |
  +-- request logging + shutdown drain
  |
  v
MedicalRAGPipeline (src/rag_pipeline.py)
  |
  +-- Query preprocessing
  |     |
  |     +-- typo correction and query compression
  |     +-- drug intent detection
  |     +-- USMLE detection
  |     +-- thread-aware conversation summary
  |     +-- retrieval query fanout
  |     +-- USMLE option query generation
  |
  +-- Retrieval
  |     |
  |     +-- standard path
  |     |     +-- PMC dense + PMC BM25
  |     |     +-- PubMed dense + PubMed BM25
  |     |     +-- DailyMed multi-query side path for drug questions
  |     |
  |     +-- USMLE path
  |           +-- option-specific dense retrieval
  |           +-- direct passthrough to synthesis
  |
  +-- Ranking and selection
  |     |
  |     +-- rerank passages
  |     +-- aggregate to paper level
  |     +-- apply evidence and metadata boosts
  |     +-- append DailyMed labels
  |
  +-- Answer generation
  |     |
  |     +-- prompt selection
  |     +-- xAI or DeepInfra chat completion
  |     +-- citation filtering
  |     +-- PDF availability checks
  |
  +-- Cache
        |
        +-- SQLite query cache
```

## Runtime Flow

### Standard query path

```text
user query
  |
  v
thread_context
  |
  +-- conversation_id -> xAI headers
  +-- follow_up_count -> cap enforcement
  |
  v
cache lookup
  |
  +-- hit  -> return cached answer
  |
  +-- miss
        |
        v
   query preprocessing
        |
        v
   build one retrieval query fanout
        |
        +-- standard literature path
        |     |
        |     +-- PMC dense + PMC BM25
        |     +-- PubMed dense + PubMed BM25
        |     +-- optional DailyMed side path
        |
        +-- USMLE option path
              |
              +-- option-specific dense retrieval
        |
        v
   standard path only:
     passage reranking -> paper aggregation -> answer generation
        |
        v
   citation cleanup and PDF checks
        |
        v
   cache write and response
```

### USMLE-style query path

USMLE-style multiple-choice questions use a different retrieval branch.

```text
USMLE vignette
  |
  v
option parsing
  |
  +-- success
  |     |
  |     +-- generate one retrieval query per answer choice
  |     +-- run dense retrieval for each option
  |     +-- dedupe passages
  |     +-- skip reranker
  |     +-- synthesize directly from retrieved evidence
  |
  +-- failure
        |
        +-- fall back to a compact stem query
```

### DailyMed Side Path

Drug-focused queries can launch a parallel DailyMed lookup while the literature retrieval runs.

```text
drug intent detected
  |
  +-- normalize drug names / keywords
  +-- build shared embeddings once
  +-- run DailyMed multi-query search
  |
  +-- literature path continues in parallel
```

### Production Controls

The API now has explicit runtime controls so the model stays responsive under load.

```text
incoming request
  |
  +-- request id + structured logging
  +-- inflight capacity check
  |     |
  |     +-- accept -> execute pipeline
  |     +-- reject -> 429 retry later
  |
  +-- shutdown drain
        |
        +-- wait for inflight work
        +-- join stream threads
```

## Main Components

### Backend API

`src/api_server.py` hosts the FastAPI app and is responsible for:

- shared pipeline startup
- request logging and request IDs
- inflight backpressure controls
- streaming response lifecycle
- graceful shutdown behavior
- follow-up cap enforcement for threaded conversations

Main endpoints:

- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`
- `GET /api/v1/health`
- `POST /api/v1/debug/decompose`

### Query preprocessing

`src/query_preprocessor.py` converts a raw question into a retrieval package.

It handles:

- typo correction when confidence is high
- compact semantic query generation
- retrieval query fanout
- drug intent detection for DailyMed routing
- USMLE-style question detection and option parsing
- conversation summaries for thread-aware answers
- xAI conversation headers when thread context is present

### Retrieval

`src/retriever_turbopuffer.py` is the active retrieval backend.

It queries three turbopuffer namespaces:

- PMC
- PubMed
- DailyMed

The literature path uses hybrid retrieval:

- dense vector search
- BM25-style sparse retrieval
- reciprocal-rank fusion across query variants and sources
- fixed raw source buckets for PMC and PubMed before reranking
- option-specific dense retrieval for USMLE multiple-choice questions
- multi-query DailyMed matching for drug labels

### Ranking and evidence selection

`src/reranker.py` reranks retrieved passages, groups them at the paper level, and applies evidence-aware boosts before the final context is built.

```text
retrieved chunks
  |
  +-- score by reranker
  +-- apply metadata boosts
  +-- group by paper
  +-- filter to final evidence set
```

### Generation

`src/rag_pipeline.py` selects the appropriate prompt, calls the configured LLM, normalizes citations, and prepares the final API response.

```text
retrieved evidence
  |
  +-- prompt selection
  +-- xAI or DeepInfra chat completion
  +-- citation normalization
  +-- PDF availability checks
  +-- cached response write
```

### Cache

`src/query_cache.py` stores responses in SQLite. Cache keys include pipeline and model context so changes to prompts or models do not collide with older results.

## Repository Layout

```text
turborag/
|-- README.md
|-- env.example
|-- requirements.txt
|-- Dockerfile.api
|-- scripts/
|   `-- benchmark_e2e_chat.py
|-- src/
|   |-- api_server.py
|   |-- config.py
|   |-- prompts.py
|   |-- query_cache.py
|   |-- query_preprocessor.py
|   |-- rag_pipeline.py
|   |-- reranker.py
|   |-- retriever_factory.py
|   |-- retriever_turbopuffer.py
|   |-- retry_utils.py
|   `-- specialty_journals.py
|-- frontend/
|   |-- package.json
|   |-- next.config.ts
|   `-- src/app/
|       |-- api/chat/stream/route.ts
|       |-- globals.css
|       |-- layout.tsx
|       |-- page.tsx
|       `-- page.module.css
```

## Requirements

### Backend

- Python 3.11+
- access to xAI and DeepInfra
- access to turbopuffer namespaces

Install dependencies:

```bash
pip install -r requirements.txt
```

### Frontend

- Node.js 20+

Install dependencies:

```bash
cd frontend
npm install
```

## Environment Setup

Start from `env.example` and create a local `.env`.

Minimum required secrets:

```env
OPENCODE_API_KEY=...
DEEPINFRA_API_KEY=...
TURBOPUFFER_API_KEY=...
```

Important runtime settings:

```env
LLM_PROVIDER=opencode
LLM_MODEL=space-bunny-free
QUERY_PREPROCESSOR_LLM_MODEL=space-bunny-free
EMBEDDING_PROVIDER=deepinfra
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B

TURBOPUFFER_NAMESPACE_PMC=medical_database_pmc
TURBOPUFFER_NAMESPACE_PUBMED=medical_database_pubmed
TURBOPUFFER_NAMESPACE_DAILYMED=medical_database_dailymed

CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
QUERY_CACHE_DIR=data/cache
```

Concurrency-related settings for container deploys:

```env
WEB_CONCURRENCY=4
GUNICORN_TIMEOUT=300
GUNICORN_GRACEFUL_TIMEOUT=90
GUNICORN_KEEPALIVE=5
GUNICORN_BACKLOG=2048
GUNICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker
API_EXECUTOR_MAX_WORKERS=64
UPSTREAM_HTTP_MAX_CONNECTIONS=200
UPSTREAM_HTTP_MAX_KEEPALIVE=100
UPSTREAM_HTTP_KEEPALIVE_EXPIRY=30

# Used when running uvicorn directly (not via gunicorn worker)
UVICORN_LOOP=auto
UVICORN_HTTP=auto
UVICORN_LIMIT_CONCURRENCY=128
UVICORN_BACKLOG=2048
```

## Running Locally

### Start the backend

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

Optional uvicorn perf flags for local stress testing:

```bash
uvicorn src.api_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --loop ${UVICORN_LOOP:-auto} \
  --http ${UVICORN_HTTP:-auto} \
  --limit-concurrency ${UVICORN_LIMIT_CONCURRENCY:-128} \
  --backlog ${UVICORN_BACKLOG:-2048}
```

### Start the frontend

```bash
cd frontend
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`.

## Production Backend (Container)

`Dockerfile.api` runs gunicorn with env-driven runtime knobs. Example:

```bash
docker build -f Dockerfile.api -t turborag-api:latest .
docker run --rm -p 8000:8000 --env-file .env turborag-api:latest
```

Tune with env vars (for example, `WEB_CONCURRENCY`, `GUNICORN_TIMEOUT`, `GUNICORN_KEEPALIVE`, `GUNICORN_BACKLOG`) without changing the image.

## Azure App Service (Linux container)

This is the fastest path from a fork to a managed deployment.

1. Build and push your API image to ACR.

```bash
az acr create -g <rg> -n <acr-name> --sku Basic
az acr build -r <acr-name> -t turborag-api:latest -f Dockerfile.api .
```

2. Create an App Service plan and web app using that image.

```bash
az appservice plan create -g <rg> -n <plan-name> --is-linux --sku P1v3
az webapp create -g <rg> -p <plan-name> -n <app-name> \
  --deployment-container-image-name <acr-name>.azurecr.io/turborag-api:latest
```

3. Configure app settings (secrets + concurrency/runtime knobs).

```bash
az webapp config appsettings set -g <rg> -n <app-name> --settings \
  WEBSITES_PORT=8000 \
  WEB_CONCURRENCY=4 \
  API_EXECUTOR_MAX_WORKERS=64 \
  GUNICORN_TIMEOUT=300 \
  GUNICORN_GRACEFUL_TIMEOUT=90 \
  GUNICORN_KEEPALIVE=5 \
  GUNICORN_BACKLOG=2048 \
  UPSTREAM_HTTP_MAX_CONNECTIONS=200 \
  UPSTREAM_HTTP_MAX_KEEPALIVE=100 \
  UPSTREAM_HTTP_KEEPALIVE_EXPIRY=30 \
  OPENCODE_API_KEY=<...> \
  DEEPINFRA_API_KEY=<...> \
  TURBOPUFFER_API_KEY=<...>
```

4. Verify health.

```bash
curl https://<app-name>.azurewebsites.net/api/v1/health
```

## Streaming

The frontend uses a dedicated Next.js route handler at `frontend/src/app/api/chat/stream/route.ts` to proxy server-sent events without buffering. This avoids idle timeout issues that can happen with standard rewrite-based proxying during slower retrieval or reranking phases.

## API

### Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the treatment for myxedema coma?","stream":false}'
```

### Streaming chat

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize contraindications of lisinopril","stream":true}'
```

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

## Testing

Use the end-to-end benchmark harness for current runtime validation:

```bash
python scripts/benchmark_e2e_chat.py --help
python scripts/benchmark_e2e_chat.py \
  --base-url http://localhost:8000 \
  --concurrency-steps 1,2,4,8 \
  --duration-seconds 60 \
  --query "What is the treatment for myxedema coma?"
```

`pytest` is still available if you add local tests, but this branch currently ships the benchmark harness instead of the old unit-test bundle.

## Notes

- retrieval is turbopuffer-only in the current architecture
- the pipeline expects prebuilt namespaces and does not ingest source data
- DailyMed is used as a targeted side path for drug-focused questions
- USMLE multiple-choice questions can bypass the reranker after option-aware retrieval
- the backend enforces follow-up caps for threaded conversations
- the `scripts/benchmark_e2e_chat.py` harness is the current runtime test entrypoint
- SQLite cache files are created at runtime under `data/cache/`
