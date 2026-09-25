# Plan: Move the API and LLM to Azure Central US

## Goal

Reduce the latency Indian users see. Move the API server, and the Azure OpenAI model it calls, from **South India** to **Central US**, next to the services that already run in the US:

| Component | Today | After |
|---|---|---|
| API server (Azure App Service, Linux container) | South India | **Central US** |
| LLM (Azure OpenAI) | South India | **Central US** |
| Embedding + reranker (DeepInfra, Qwen3 0.6B) | US | US (unchanged) |
| Vector/BM25 search (turbopuffer, `gcp-us-central1`, Iowa) | US | US (unchanged) |
| Frontend (Next.js) | unchanged | unchanged (repoint to the new API) |

Azure **Central US** is in Iowa, the same area as turbopuffer's `gcp-us-central1`. Search calls become short cross-cloud hops within one region.

## Why

A chat request makes several calls **one after another**. Today each call crosses the ocean between India and the US:

```
User (India) -> API (South India)
                 |- LLM query rewrite        (Azure OpenAI, South India)
                 |- Embedding                 (DeepInfra, US)           <- ocean round trip
                 |- Hybrid search             (turbopuffer, US)          <- ocean round trip (+1 on title-search fallback)
                 |- DailyMed search           (turbopuffer, US)          <- in parallel with the hybrid search
                 |- Rerank ~100 passages      (DeepInfra, US)           <- ocean round trip, large upload
                 '- LLM answer generation     (Azure OpenAI, South India)
```

Passage text is also downloaded from turbopuffer into India, then uploaded back to DeepInfra in the US for reranking.

After the move:

```
User (India) --- one ocean crossing ---> API (Central US)
                                          |- LLM query rewrite     (Azure OpenAI, Central US)  in-region
                                          |- Embedding             (DeepInfra, US)             in-country
                                          |- Hybrid + DailyMed     (turbopuffer, Iowa)         in-region
                                          |- Rerank                (DeepInfra, US)             in-country
                                          '- LLM answer (streamed) (Azure OpenAI, Central US)  in-region
User (India) <-- streamed response (SSE) -
```

The user pays the India-US round trip once for the request and once to start the streamed response. Tokens then flow over the open connection with no extra delay per token.

The PDF-link check calls Europe PMC (EBI, UK) in both setups. It isn't affected by this change.

## Cost

Azure retail prices (public pricing API), App Service Premium v3 **P1v3 Linux**:

| Region | $/hour | ~$/month (730 h) |
|---|---|---|
| South India | $0.1649 | ~$120 |
| Central US | $0.1700 | ~$124 |

That's about **+$3.70 per instance per month**. Multiply by the production instance count. Egress costs are negligible because responses are text.

Azure OpenAI is billed per token. Prices are usually the same across regions for the same model and deployment type. **Confirm for the model we use.**

## Prerequisites to check before starting

1. **Model availability in Central US.** Confirm the exact model and version we use in South India can be deployed in Central US, with the same deployment type (Standard / Global Standard / Data Zone). If not, pick the nearest US region that has it (e.g. East US 2) and note the extra latency (~20-30 ms round trip), which is still small compared with crossing the ocean.
2. **Quota.** Request tokens-per-minute (TPM) and requests-per-minute (RPM) quota in Central US at least equal to current South India production usage, plus headroom.
3. **Content filter / safety settings.** Copy the same content-filter policy to the new deployment.
4. **Data residency / compliance.** Confirm with whoever owns compliance that processing Indian users' queries in a US region is acceptable. Retrieval (DeepInfra, turbopuffer) is already in the US, but the LLM and app server currently are not.
5. **LLM client code path.** In this repo `LLM_PROVIDER` supports `xai` and `deepinfra` (`src/config.py`, `src/rag_pipeline.py::_create_llm_client`, `src/query_preprocessor.py`). Confirm how the production build points at Azure OpenAI (the branch/config used in prod). The steps below assume it's configured through environment variables (endpoint, API key, API version, deployment name). If it's hard-coded, make the endpoint configurable first.

## Steps

### 1. Create the Azure OpenAI deployment in Central US

- Create (or reuse) an Azure OpenAI resource in **Central US**.
- Deploy the **same model and version** under the same deployment name as South India. Keeping the name identical means only the endpoint and key change.
- Apply the same content-filter policy and quota.
- Smoke-test it with a single chat-completion request.

### 2. Create the App Service in Central US (side-by-side with South India)

Reuse the existing resource group and container registry (ACR). The registry's region only affects image pulls at deploy time.

```bash
az appservice plan create -g <rg> -n <plan>-cus -l centralus --is-linux --sku P1v3
az webapp create -g <rg> -p <plan>-cus -n <app>-cus \
  --deployment-container-image-name <acr>.azurecr.io/turborag-api:<same-tag-as-prod>
```

Copy the production configuration from the South India app:

- All app settings (secrets, gunicorn/concurrency knobs, `UPSTREAM_HTTP_*`, `TURBOPUFFER_*`, `DEEPINFRA_*`, `CORS_ALLOWED_ORIGINS`).
- **Change only the LLM settings** so they point at the Central US Azure OpenAI resource (endpoint + key; deployment name and API version unchanged).
- Leave `TURBOPUFFER_REGION=gcp-us-central1` as is.
- Instance count / autoscale rules, Always On, health check path `/api/v1/health`, registry pull identity/credentials, logging/App Insights.

### 3. Validate the new app (before any user traffic)

- `GET https://<app>-cus.azurewebsites.net/api/v1/health` returns 200 with the pipeline ready.
- Run a fixed set of 20-50 real (de-identified) queries against **both** apps **from a machine in India**. Record for each app:
  - End-to-end latency (non-streaming) and **time to first token** (streaming, `/api/v1/chat/stream`).
  - Per-stage timings returned in `retrieval_stats`: `preprocess_ms`, `embedding_ms`, `literature_retrieval_ms`, `dailymed_retrieval_ms`, `rerank_ms`, `pdf_check_ms`.
- Load test with `scripts/benchmark_e2e_chat.py` (cache-busting on) at concurrency 1, 8, 32. Check throughput, p50/p95 latency and error/429 rates.
- Spot-check answer quality on a handful of queries. Answers should match production since nothing but location changed.

**Expected result:**
- `embedding_ms`, `literature_retrieval_ms`, `dailymed_retrieval_ms` and `rerank_ms` drop sharply.
- The time to first streamed token as measured from India improves.
- The share of latency that is the LLM itself (not network) should stay the same.

### 4. Cut over

- Point the frontend at the new API. The Next.js stream proxy (`frontend/src/app/api/chat/stream/route.ts`) reads `API_REWRITE_TARGET`, falling back to `NEXT_PUBLIC_API_URL`. Update it and redeploy the frontend.
  - If the API is behind a custom domain, move the domain and TLS certificate to the new app instead. The frontend then needs no change.
- If the browser calls the API directly anywhere, make sure the new app's `CORS_ALLOWED_ORIGINS` includes the production frontend origin.
- Watch error rate, latency, and Azure OpenAI 429s/quota for the first hours after cutover.

### 5. Roll back if needed

- Keep the South India app and South India Azure OpenAI deployment **running and unchanged** for 1-2 weeks.
- To roll back, point the frontend (or custom domain) back at the South India app.
- After the stable period, scale the South India plan down, then decommission it.

## Notes for the team

- **Cold cache.** The query cache is on the instance's local disk (`QUERY_CACHE_DIR=data/cache`), so the new app starts cold. Expect slightly slower responses until it warms up. Don't compare cached old vs uncached new numbers; use cache-busting in benchmarks.
- **Streaming.** The streaming route already avoids buffering. Make sure any gateway or Front Door added in front of the new app does not buffer SSE responses.
- **Out of scope.** No changes to the embedding model, reranker, turbopuffer region, indexes, or retrieval/rerank settings. Quality should be identical; only latency should change.

## Done when

- [ ] Central US Azure OpenAI deployment is live with the same model/version, quota and content filter.
- [ ] Central US App Service runs the same image and config, with only the LLM endpoint changed.
- [ ] India-side benchmark shows lower time to first token and end-to-end latency than South India, with no increase in errors.
- [ ] Frontend (or custom domain) points at the Central US app.
- [ ] South India resources kept for rollback, with a date set to decommission them.
