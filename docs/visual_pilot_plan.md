# Visual Findings Library pilot — pipeline throughput plan

This document replaces the original build spec. It is a work plan for a
multi-agent coordinator: a set of workstreams with disjoint file ownership,
fixed interface contracts, a dependency graph, and verification gates.

**Goal:** make `run-all` finish hundreds of articles substantially faster
**without changing any image, any LLM decision, or any stored output.**
Speed must come from scheduling, concurrency, and removing duplicate network
work — never from changing what the models see or what gets written.

> **Source of truth for everything this plan does not change**
> - Original spec (decisions, schema, stages, prompts, testing rules):
>   `git show 2712b07:docs/visual_pilot_plan.md`. Every `§N` reference in
>   code comments, READMEs and tests points to that version.
> - Prompts P1–P4 and schemas: `src/visual_pilot/prompts.py` (do not edit).
> - SQLite schema: `src/visual_pilot/db.py` (only additive changes in W2).

---

## 0. Status log (updated 2026-09-27, main @ `e117af1`)

The historical verification results below describe the completed pilot.
Its runtime data, parity snapshots, live-run copies, generated reports, and
one-off probe/evaluation artifacts were cleared on 2026-09-27 before a new
full disease index is supplied. Record a fresh baseline for future parity
checks; the old paths below are historical references.

**All workstreams complete: W0–W11. Gates G0–G3 passed.**

Merged onto `main`, in order:

| Workstream | Commit(s) | Result |
|---|---|---|
| W0 timing/parity harness | `b077934` | `timing.py`, `parity.py`, C1 keys, `VP_LLM_CACHE_ONLY` |
| W2 schema | `ac201fe` | C2 columns + dedup index, additive migrations |
| W10 triage tidy | `4710afe` | single-pass P2 batching unchanged |
| W4a parallel select | `63c1f43` | pooled retrieval, deterministic ordering |
| W3 llm client | `7b59887` | C4 `iter_many`, 429/`Retry-After`, timeouts |
| W1 pmc limits | `9e5ed6f` | C3: `VP_S3_RPS` limiter, hinted bundles, LRU caches |
| W4b license/hints | `94a56a7` | `s3_prefix`/`media_files_json` persisted, pools wired |
| W5 parallel parse | `d0e2c55` | C6 outputs, `sections_for`, bounded caches |
| W6 streaming judge | `05868a2` | C5 `originals.py`, `iter_many` pipeline, ≤2× buffer |
| W7 parallel store | `b250e5f` | originals handoff + fetch fallback, ordered apply |
| W8 extract reuse | `ce1fb79` | `sections_for` reuse, vocab cache, P4 `iter_many` |
| Coordinator fixes | `6c33972`, `5063008`, `9593d42`, `d69fcd7`, `e117af1` | see "nondeterminism fixes" below |
| W11 probe | `reports/concurrency_probe.md` | **keep `VP_JUDGE_CONCURRENCY=4`** (see below) |
| W9 run-all scheduling | `307ef2b`, `4e50cca`, merge `7dea4b1` | in-run judge retry, round-robin diseases, fetch-only prefetch, shared read conn |

Verification state: `pytest tests/visual_pilot` → **312 passed**,
`ruff` clean, `parity compare reports/parity_baseline
reports/parity_candidate_g3` → **identical** (G3).

`e117af1` (parity.py): `attempts` on `error`-bearing figure rows is
normalized alongside `error` — W9's in-run judge retries legitimately
raise it between the recorded baseline and a cache-only replay.

G3 timing (cache-only candidate; baseline was live): stage total 7.2 s
vs 6.0 s at G2; S3 limiter wait 41.7 s vs 29.3 s — prefetch overlap puts
more concurrent fetches on the S3 limiter, so wait time moves into the
background rather than disappearing. A supervised live `run-all` on a
scratch copy of the main DB exercised resume drain (12 retriable
`vision_error` judged), round-robin (dm/as exhausted instantly — only
sle had `relevant` rows left), in-batch retries, prefetch, and store
with real DeepInfra calls; stopped during `extract` (unchanged W8 path)
after the W9 behavior was proven.

### Provider situation (important for the next wave)

OpenCode `space-bunny-free` rejects the union-type JSON schemas used by
P2/P3/P4 (`"type": ["string","null"]`) with HTTP 400 — this is an upstream
regression, not something to work around by editing prompts (still
forbidden). All other OpenCode free models are gated to the official client
(403 "free tier can only be used from within OpenCode"); paid models require
account funds.

**Parity baseline and all live validation ran on DeepInfra instead:**
`VP_LLM_PROVIDER=deepinfra` (added to `_LLM_PROVIDER_CREDENTIALS`; uses
`DEEPINFRA_API_KEY`/`DEEPINFRA_BASE_URL`) with
`VP_TRIAGE_MODEL=VP_EXTRACT_MODEL=VP_JUDGE_MODEL=zai-org/GLM-5.3-Flash`.
Parity/live runs must export these env vars so `input_hash` (which covers
the model name) matches the seeded ledger.

Note: GLM-5.3-Flash is nondeterministic at temperature 0 — identical P3
inputs produced 65/63/63/64 accepts across the W11 probe runs. Parity
therefore relies on the seeded ledger, never on re-decision.

**Resolved 2026-09-27 (`5219c7a`):** production defaults switched to
OpenRouter `stealth/space-bunny-alpha` (`VP_LLM_PROVIDER=openrouter`;
`opencode`/`deepinfra` remain supported providers). This changes
`input_hash`'s model component, so the DeepInfra-keyed parity baseline
below cannot validate future candidates — record a fresh baseline before
the next parity gate.

### Nondeterminism / correctness fixes made during gating

- `jats.py` (`6c33972`): mention windows used `id()` on transient lxml
  element proxies — GC-dependent, silently moved windows between parses.
  Replaced with deterministic per-paragraph `rid`/cursor tracking.
- `extract_findings.py` (`5063008` + `ce1fb79`): `_vocabulary` ran on the
  shared sqlite connection from the prepare pool → intermittent "tuple
  index out of range" silently dropping one article per run. W8's redesign
  removed all conn access from pooled work (vocab precomputed on the main
  thread) — keep it that way.
- Extract's per-article refetch could transiently fall back to EuropePMC's
  `fullTextXML` (different serialization → different sections → different
  P4 `input_hash` → silent cache miss → dropped rows; observed on
  PMC11816486, 14 rows). W8's `sections_for` reuse + pinned `s3_prefix`
  fallback removes the second-fetch drift inside a run.
- `apply_response` now keys proposal upserts on "article already applied"
  (`_has_text_rows`) instead of `cached`, so a fresh-DB cache-only replay
  restores `findings_vocab` proposals exactly (§6.2 requires identical
  `proposal_count`), while `--force` re-applies still do not re-count
  (`9593d42`).
- `parity.py` compare hardening (`9593d42`): order-insensitive row
  comparison (streamed apply changes insertion order/rowids);
  `disease_findings.id` excluded; `figures.error` normalized to presence
  (live transient text vs replayed "cache miss:" differ legitimately);
  cache-miss markers allowed only on rows whose baseline row errored
  (failed calls are never ledgered and cannot replay).

### W11 probe result

84 uncached P3 calls per run at `VP_JUDGE_CONCURRENCY` = 4/8/12/16 on
DeepInfra: zero 429s and zero timeouts at every level (provider queues
rather than rejects), but p95 latency blew past the 1.5× criterion at
8 (+145%), 12 (+68%), 16 (+60%). **Recommendation applied: default stays
4.** Throughput still scaled (22.9→48.5 calls/min at 16); revisit only if
throughput-over-tail-latency is ever preferred.

### What remains

1. ~~W9~~ — merged (`7dea4b1`). ~~G3~~ — passed.
2. Deferred items (true cross-disease concurrency, A/B input changes) —
   unchanged, post-G3 only.
3. Resolved decision: Production model defaults switched permanently to
   OpenRouter `stealth/space-bunny-alpha` (`VP_LLM_PROVIDER=openrouter`), replacing
   DeepInfra GLM and OpenCode `space-bunny-free`. Free tier ($0.00), 1M context,
   3.45x faster overall (6.4x faster on vision judging), with schema injection.
4. Live-run observation for follow-up: `extract`'s P4 fan-out over a
   large un-extracted backlog runs at provider speed (~7 calls/min on
   DeepInfra GLM) — a full `run-all` on the main DB still needs either
   the runtime cap to be honored mid-extract or a bounded extract.

Known instrumentation gap: the `parse` stage timer shows ~0.1s because
`select_batch`'s caption peeks do the fetching inside `cli.py`'s untimed
selection call — W9's overlap work should account stage time correctly.

---

## 1. Invariants (non-negotiable for every agent)

A change that violates any of these is rejected at the gate, regardless of
speed gains.

1. **Same model inputs.** Prompts, system text, schemas, prompt versions,
   models, temperature 0, P2 batch composition rules (`VP_TRIAGE_BATCH`, 40),
   one figure per P3 call, and the P3 image preparation
   (`pmc.prepare_for_llm`, `VP_IMAGE_MAX_EDGE=1568`) are unchanged. Therefore
   every `llm_calls.input_hash` for the same work is byte-identical.
2. **Same stored images.** Panels are cropped from the *original* fetched
   bytes with the same bbox scaling, 2% padding, whole-figure rules (ND
   license, bbox <3%, overlap >30%) and exact-hash dedup, then capped at
   `VP_PANEL_MAX_EDGE` and encoded per `VP_PANEL_FORMAT`/`VP_PANEL_QUALITY`
   (default WebP q90; `png` restores lossless archival crops). Thumbnails
   stay 400 px WebP. The stored "original" is a WebP display copy capped at
   `VP_ORIGINAL_MAX_EDGE`; the downscaled LLM copy is never stored.
3. **No image or full text on disk before acceptance.** Nothing is written to
   disk for a figure until it is `vision_accepted` (spec core rule; enforced by
   the existing no-disk tests in `test_parse.py` / `test_pmc.py`). Full article
   text is never written to disk or to SQLite. Caches are in-memory only.
4. **Same gates.** License policy, third-party rejection, P1 relevance,
   caption-uncertain → vision, contradictory-drop revisit, post-validation, and
   the attempts-bounded `vision_error` retry all behave as today.
5. **Same yield semantics.** `run-all` still measures per-batch yield (new
   distinct image hashes + newly covered approved findings) only after that
   batch's figures are finished, and still stops after
   `--zero-yield-batches` consecutive empty batches.
6. **Resumable and idempotent.** Every stage still resumes from status
   columns; a rerun with unchanged inputs makes zero live LLM calls.
7. **Polite to NCBI.** NCBI hosts keep ≤3 req/s (≤10 with
   `VP_NCBI_API_KEY`). Only the public S3 bucket gets a higher limit.

### Explicitly out of scope (never do these for speed)

- Multiple figures per vision call; smaller `VP_IMAGE_MAX_EDGE`; lossy
  re-encoding of stored panels; storing the LLM copy.
- Writing figure bytes or article XML/text to a disk cache.
- Skipping triage, license, third-party or relevance checks.
- Shrinking the caption-peek shortlist (it changes which articles are
  selected).

### Requires an A/B accuracy study first (not part of this plan)

- Batching P1 (multiple articles per call).
- Reordering P3 user content to improve provider prefix caching.
- Changing the P2 batch size.

Each of these changes `input_hash` values and invalidates the response cache.
Only take them on as separate, evaluated proposals.

---

## 2. Baseline evidence (2026-09-26 ledger)

| Stage | Calls | Avg gap between completions | Notes |
|---|---:|---:|---|
| P1 relevance | 2,817 | 1.3 s | ~495 in / 90 out tokens |
| P2 caption triage | 126 | 17.6 s | 40 figures/call, ~1.5k out tokens |
| **P3 vision judge** | 1,327 | **13.3 s** | Peak ~240/h ≈ 53 s/call at 4 in flight |

DB at snapshot: 731 parsed articles, ~1,850 figures, 400 stored figures.
~2.5 figures/article; ~70% of figures reach P3. All 4,280 ledger calls
succeeded on the first schema attempt (repair retries are not a cost).

### Diagnosed bottlenecks (ranked)

| # | Bottleneck | Where |
|---|---|---|
| B1 | One `vision_error` (attempts <3) pauses expansion of the whole disease; judge runs with `max_retries=0`, so any timeout triggers it | `cli.py` `_cmd_run_all`, `_unfinished_figures` |
| B2 | Judge hard-capped at `min(VP_CONCURRENCY, 4)` and runs in fetch-4 → judge-4 lockstep chunks; slowest call gates each chunk | `judge.py` `run` |
| B3 | Public S3 bucket shares the 5 req/s default limiter; caps license, peek, parse, judge fetch and store fetch combined | `pmc.py` `_DEFAULT_RPS`, `RATE_LIMITER` |
| B4 | Parse fetches/parses articles one at a time | `parse.py` `run` |
| B5 | Duplicate fetches: license metadata refetched at parse (256-entry LRU); caption peek discards ~50 of 100 bundles per batch (64-entry LRU); judge's image bytes refetched by store; store and extract refetch JATS | `pmc.py`, `parse.py`, `store.py`, `extract_findings.py` |
| B6 | Store is one figure at a time (fetch, decode, panel encode, thumb); no index on `panels.sha256` | `store.py`, `db.py` |
| B7 | Selection queries (per synonym ×3 buckets + 12 visual queries) and embeddings run serially | `select_articles.py` `retrieve_for_disease` |
| B8 | Stages are hard barriers per batch; diseases run strictly SLE → DM → AS on one shared 900 s clock, so later diseases can be starved | `cli.py` `_cmd_run_all` |
| B9 | 300 s LLM timeout with no retry pins a worker for 5 min on a stalled call | `config.py`, `llm.py` |
| B10 | Minor CPU/DB waste: per-row vocab reload + regex compile in triage; per-article vocab query in extract; `_snapshot` rescans panels | `triage.py`, `extract_findings.py`, `cli.py` |

---

## 3. Coordinator protocol

### 3.1 Working model

- One agent per workstream (W0–W11). Each works in its **own git worktree
  and branch** named `perf/wN-<slug>` off the current integration branch.
- Agents only edit files listed under **Owns** for their workstream. Reading
  any file is fine. Needing a change in someone else's file → request it from
  the coordinator; do not edit.
- Interface contracts in §4 are fixed up front so dependent workstreams can
  build against them in parallel (mock the dependency in tests until it
  merges).
- The coordinator merges branches in dependency order, reruns verification on
  the integration branch after each merge, and runs the parity gates (§6).

### 3.2 Per-workstream definition of done

Every workstream must, before handoff:

1. Meet its acceptance criteria.
2. Pass `ruff check src/visual_pilot tests/visual_pilot` and
   `python3 -m pytest tests/visual_pilot` (no network in unit tests).
3. Add/adjust unit tests for new behavior, including an explicit invariant
   test where listed.
4. Not modify `prompts.py`, the existing `frontend/`, or any `src/*.py`
   outside `src/visual_pilot/`.
5. Submit a handoff note (§8 template).

### 3.3 Environment notes for agents

- Repo root `/Volumes/Vibing/Turborag`; interpreter `python3` (3.14).
- Worktrees were kept at `/Volumes/Vibing/Turborag-worktrees/wN` on
  `perf/wN-*` branches (merged branches are preserved; checkouts were
  removed after merge — recreate with `git worktree add`).
- Unit tests use `VP_DATA_DIR` temp dirs and `pmc.set_http_client` /
  mocked LLM clients; see `tests/visual_pilot/conftest.py`.
- Live runs must use a **scratch** `VP_DATA_DIR`, never the main
  `data/visual_pilot/`, unless the coordinator says otherwise.
- The recorded parity baseline predates the provider switch; replaying it
  needs `VP_LLM_PROVIDER=deepinfra
  VP_TRIAGE_MODEL=zai-org/GLM-5.3-Flash
  VP_EXTRACT_MODEL=zai-org/GLM-5.3-Flash
  VP_JUDGE_MODEL=zai-org/GLM-5.3-Flash`
  so `input_hash` matches its seeded ledger (§0 provider note). For new
  baselines/candidates under the OpenRouter defaults, export nothing —
  `OPENROUTER_API_KEY` is in the gitignored `.env`, auto-loaded by config.
- Parity artifacts live under `reports/` (gitignored):
  `parity_baseline/` is the recorded live baseline; candidates go to
  `reports/parity_candidate_*`.
- Never commit `.env` or anything under `data/` or `reports/`.

---

## 4. Interface contracts (fixed before Phase 1)

Owners implement these exactly; consumers code against them.

### C1 — config keys (`config.py`, owner W0)

| Key | Default | Used by |
|---|---|---|
| `VP_S3_RPS` | `20` | W1 limiter for `pmc-oa-opendata.s3.amazonaws.com` |
| `VP_FETCH_CONCURRENCY` | `8` | W5 parse pool, W6 image fetch pool, W7 store pool, license pool |
| `VP_JUDGE_CONCURRENCY` | `4` (raise only after W11 probe) | W6 in-flight P3 calls |
| `VP_P1_CONCURRENCY` | `VP_CONCURRENCY` | W4 P1 `call_many` |
| `VP_JUDGE_TIMEOUT_SECONDS` | `120` | W6 LLM client timeout for P3 |
| `VP_RATE_LIMIT_RETRIES` | `4` | W3 retries on HTTP 429 only |
| `VP_ORIGINALS_CACHE_MB` | `512` | W6/W7 in-memory originals handoff |
| `VP_TIMINGS` | `1` | W0 timing report on/off |
| `VP_LLM_CACHE_ONLY` | `0` | W0 hook in `llm.py`: when `1`, a cache miss raises `LLMError("cache miss: <stage> <input_hash>")` instead of calling the provider (parity runs) |

`VP_CONCURRENCY` keeps its meaning for everything not listed.

### C2 — schema additions (`db.py`, owner W2)

Additive, idempotent migrations (extend `_ARTICLE_MIGRATIONS` pattern):

- `articles.s3_prefix TEXT` — e.g. `PMC123.1`
- `articles.media_files_json TEXT` — JSON list of image basenames in the
  article dir
- `articles.authors_json TEXT` — JSON list used by attribution
- `articles.author_count INTEGER`
- `articles.journal_name TEXT` — JATS journal title fallback
- `CREATE INDEX IF NOT EXISTS idx_panels_sha256 ON panels(sha256)`

These hold metadata only (no full text).

### C3 — PMC access (`pmc.py`, owner W1)

- `LicenseInfo` gains optional fields `prefix: str | None = None` and
  `media_files: tuple[str, ...] = ()`, filled by `get_license` from the same
  metadata/listing it already fetches.
- `get_article_bundle(pmcid, *, use_cache=True, prefix=None,
  media_files=None) -> ArticleBundle`. When `prefix` (and optionally
  `media_files`) are given, skip `_list_keys` and the metadata JSON fetch;
  fetch only the XML. The resolver output must be identical to the no-hint
  path for the same article (test required).
- Limiter: the S3 host uses `config.VP_S3_RPS`; NCBI hosts unchanged; other
  hosts keep `_DEFAULT_RPS`. Existing 429/5xx backoff and `Retry-After`
  handling unchanged.
- In-process caches sized for a full run: `_article_metadata` and
  `_article_bundle_cached` LRU sizes configurable (defaults ≥2,000 metadata,
  ≥256 bundles). Memory only.
- `fetch_image_bytes` unchanged in output.
- Timing hook (from W0): limiter wait time recorded per host.

### C4 — LLM layer (`llm.py`, owner W3)

- `LLMClient(..., timeout_seconds: float | None = None)` — per-client timeout
  (default `config.VP_LLM_TIMEOUT_SECONDS`).
- `LLMClient.iter_many(requests: Iterable[dict], max_in_flight: int | None =
  None) -> Iterator[BatchResult]` — consumes `requests` lazily, keeps at most
  `max_in_flight` (default `self.concurrency`) calls running, yields results in
  **completion order**; `BatchResult.index` is the position in the consumed
  iterable. `call_many` keeps its current signature and ordering semantics
  (may be reimplemented on top of `iter_many`).
- HTTP 429 (`openai.RateLimitError`) is retried up to
  `config.VP_RATE_LIMIT_RETRIES` times honoring `Retry-After` (fallback
  exponential, cap 30 s), **independently of** `max_retries`, which continues
  to govern timeouts/connection/5xx.
- Cache lookup, input hashing, ledger rows and budget semantics unchanged.
  `input_hash` for any given request must be byte-identical to today (test
  required).

### C5 — originals handoff (`src/visual_pilot/originals.py`, new, owner W6)

In-memory, process-local, thread-safe, bounded by
`VP_ORIGINALS_CACHE_MB` (LRU eviction):

- `put(figure_id: str, sha256: str, data: bytes) -> None`
- `take(figure_id: str, expected_sha256: str) -> bytes | None` — returns the
  bytes only when `sha256(data) == expected_sha256`, then removes the entry.
- `discard(figure_id: str) -> None`, `clear() -> None`

Judge calls `put` **only for figures it marks `vision_accepted`**; every
other figure's bytes are dropped immediately. Store calls `take` and falls
back to `pmc.fetch_image_bytes` on miss.

### C6 — parse outputs (`parse.py`, owner W5)

- Parse persists `s3_prefix`, `media_files_json` (if not already set by
  license), `authors_json`, `author_count`, `journal_name` on the article row.
- `parse.sections_for(pmcid) -> list[tuple[str, str]] | None` — in-memory,
  bounded cache of `parsed.body_sections` for articles parsed in this
  process; `None` on miss. Consumed by W8.
- `parse.select_batch` keeps peeked-but-unselected bundles in memory across
  batches within the process (bounded; memory only). Selection results must be
  identical to today for the same DB state (test required).

### C7 — timing report (`src/visual_pilot/timing.py`, new, owner W0)

- `timing.stage(name)` context manager; `timing.record(kind, seconds,
  **labels)`; `timing.write_report(path)`.
- `run-all` writes `reports/timings_<utc>.json`: per-stage wall time, limiter
  wait per host, HTTP fetch time, LLM latency p50/p95 per stage, peak
  in-flight LLM calls, counts of 429s/timeouts.

---

## 5. Workstreams

Dependencies refer to merged branches. "Owns" lists the only files the agent
may edit (tests for those modules included).

### Phase 0 — measurement and parity harness (serial; gate G0)

#### W0 — Timing, config keys, parity harness
- **Owns:** `src/visual_pilot/timing.py` (new), `src/visual_pilot/parity.py`
  (new), `src/visual_pilot/config.py`, minimal timing hooks in `llm.py`,
  `pmc.py`, `cli.py`; `tests/visual_pilot/test_timing.py`,
  `tests/visual_pilot/test_parity.py`; `env.example` (document C1 keys).
- **Depends on:** none. Runs alone; everything else waits for it.
- **Tasks:**
  1. Add all C1 config keys (defaults preserve current behavior except where
     a later workstream switches them on).
  2. Implement C7 and hook it into: `RATE_LIMITER.wait` (wait seconds per
     host), `pmc._request` (fetch seconds), `LLMClient.call_json` (latency per
     stage, cached vs live), each stage in `_cmd_run_all`. Add the
     `VP_LLM_CACHE_ONLY` check in `LLMClient.call_json` right after the cache
     lookup (before the dry-run and budget branches). A budget cannot detect
     cache misses: `space-bunny-free` is priced at $0, and `--budget-usd 0`
     makes `run-all` exit before any stage runs.
  3. Implement the parity harness (`python3 -m src.visual_pilot.parity`), see
     §6.2: `prepare`, `run`, `compare` subcommands.
  4. Choose the **parity set**: 20 PMCIDs already `parsed` in the main DB,
     stratified ~7 per disease, including at least one CC BY-ND article, one
     TIFF figure, one multi-disease article, one article with a
     `caption_rejected` third-party figure. Commit the list as
     `src/visual_pilot/data/parity_pmcids.json`.
- **Acceptance:** timing report produced by a dry `run-all` on a scratch DB;
  harness `prepare/run/compare` works end to end on the parity set with the
  **unchanged** code (baseline compare = identical to itself); hooks add no
  behavioral change (all existing tests pass unchanged).
- **Gate G0:** coordinator runs the baseline (§6.2) and records
  `reports/parity_baseline/` plus the baseline timing report.

### Phase 1 — low-risk, independent fixes (parallel; gate G1)

All Phase 1 workstreams can run concurrently; they have disjoint files.

#### W1 — PMC access limits and caches (B3, B5)
- **Owns:** `src/visual_pilot/pmc.py`, `tests/visual_pilot/test_pmc.py`.
- **Depends on:** W0.
- **Tasks:** implement C3 (S3 limiter via `VP_S3_RPS`, larger in-memory
  caches, `LicenseInfo.prefix/media_files`, hinted `get_article_bundle`).
- **Acceptance / tests:**
  - Resolver parity: hinted and unhinted bundles resolve every href in the
    fixture to identical `ImageRef`s.
  - Hinted path issues zero list/metadata requests (mock transport count).
  - NCBI host rps unchanged; S3 host uses `VP_S3_RPS`.
  - Existing no-disk-write test still passes.

#### W2 — Schema additions (B5, B6)
- **Owns:** `src/visual_pilot/db.py`, `tests/visual_pilot/test_foundations.py`.
- **Depends on:** W0.
- **Tasks:** implement C2 migrations + index.
- **Acceptance:** `init_db` idempotent on a fresh DB and on a copy of the
  pre-change schema; index present; no existing column changed.

#### W3 — LLM client streaming, timeout, 429 handling (B2, B9)
- **Owns:** `src/visual_pilot/llm.py`, `tests/visual_pilot/test_llm.py`.
- **Depends on:** W0.
- **Tasks:** implement C4.
- **Acceptance / tests:**
  - `input_hash` golden test: fixed P1/P2/P3 requests hash to recorded values
    captured from the pre-change code.
  - `iter_many` never exceeds `max_in_flight` (instrumented fake client);
    consumes a lazy generator; yields completion order with correct indices.
  - 429 retried with `Retry-After`; timeouts still obey `max_retries=0`.
  - Cache hit, budget stop, schema repair tests unchanged and passing.

#### W4 — Parallel selection retrieval (B7)
- **Owns:** `src/visual_pilot/select_articles.py`,
  `src/visual_pilot/retrieval.py`, `tests/visual_pilot/test_select.py`,
  `tests/visual_pilot/test_visual_retrieval.py`.
- **Depends on:** W0 (part b also W1, W2).
- **Tasks:**
  - (a) Run the independent turbopuffer queries in `retrieve_for_disease`
    on a thread pool (verify client thread safety; otherwise one namespace
    handle per worker). Collect results, then append ranked lists **in the
    same order as the sequential code** before RRF. Add
    `VisualRetriever.embed_queries(list[str])` using one batched embeddings
    request; fall back per item on failure.
  - (b) After W1/W2 merge: `apply_license` persists `s3_prefix` and
    `media_files_json` from `LicenseInfo`; license pool uses
    `VP_FETCH_CONCURRENCY`; P1 `LLMClient` uses `VP_P1_CONCURRENCY`.
- **Acceptance / tests:** with a mocked namespace returning fixed rows, the
  parallel `retrieve_for_disease` output (scores, attrs, matched passages) is
  identical to the sequential implementation (keep a private sequential
  reference in the test); batched embeddings produce the same vectors as
  per-item calls in a mocked client; P1 requests unchanged.

#### W10 — Triage CPU/DB tidy-up (B10)
- **Owns:** `src/visual_pilot/triage.py`, `tests/visual_pilot/test_triage.py`.
- **Depends on:** W0.
- **Tasks:** cache approved vocab rows, disease terms and compiled regexes
  once per `run` for `_explicit_target_visual`; no behavioral change.
- **Acceptance:** existing triage tests pass; new test asserts identical
  routing on a fixture set before/after.

**Gate G1** (after W1, W2, W3, W4a, W10 merge): full test suite + parity
compare against baseline (§6) + timing report.

### Phase 2 — concurrency in the heavy stages (parallel where noted; gate G2)

#### W5 — Parallel parse, peek cache, metadata persistence (B4, B5)
- **Owns:** `src/visual_pilot/parse.py`, `tests/visual_pilot/test_parse.py`,
  `tests/visual_pilot/test_run_all_yield.py` (only parse-related fixtures).
- **Depends on:** W1, W2.
- **Tasks:**
  1. Fetch + `jats.parse_article` on a pool of `VP_FETCH_CONCURRENCY`
     workers; all DB writes and commits stay on the calling thread, in the
     same article order as today.
  2. Use hinted `get_article_bundle(prefix=..., media_files=...)` when the
     article row has them.
  3. Implement C6 (persist author/journal metadata; `sections_for`; keep
     unselected peeks across batches, bounded).
- **Acceptance / tests:** identical `figures` rows (all columns except
  timestamps) for the fixture articles vs sequential path; `select_batch`
  returns the same PMCIDs in the same order; no-disk test passes; parse
  error isolation unchanged.

#### W6 — Streaming vision judge + originals handoff (B1 partial, B2)
- **Owns:** `src/visual_pilot/judge.py`, `src/visual_pilot/originals.py`
  (new), `tests/visual_pilot/test_judge_store.py` (judge parts),
  `tests/visual_pilot/test_figure_priority.py`,
  `tests/visual_pilot/test_originals.py` (new).
- **Depends on:** W3.
- **Tasks:**
  1. Replace chunked fetch/judge with a pipeline: a fetch pool
     (`VP_FETCH_CONCURRENCY`) feeds prepared figures, in the existing
     `rank_figures` priority order, into `client.iter_many(...,
     max_in_flight=VP_JUDGE_CONCURRENCY)`. Bound buffered prepared images to
     ~2× in-flight to cap memory.
  2. Remove the hard `min(..., 4)` cap; use `VP_JUDGE_CONCURRENCY` and
     `timeout_seconds=VP_JUDGE_TIMEOUT_SECONDS`.
  3. Apply results and commit on the main thread as they complete; same
     status transitions, same `vision_json` post-validation, same attempts
     accounting, same budget stop.
  4. Implement C5; `put` only on `vision_accepted`.
- **Acceptance / tests:** for mocked fetch + mocked LLM, the set of P3
  requests (`input_hash` values) and final figure rows equal the old
  implementation's; in-flight never exceeds the setting; bytes for
  rejected/error figures are not retained (assert `originals` empty for them);
  no-disk test passes.

#### W7 — Parallel store using handoff and persisted metadata (B5, B6)
- **Owns:** `src/visual_pilot/store.py`, `tests/visual_pilot/test_judge_store.py`
  (store parts; coordinate with W6 on the shared file — W6 edits judge tests
  first, W7 rebases).
- **Depends on:** W2, W6 (C5), W5 (C6 metadata).
- **Tasks:**
  1. Get originals via `originals.take(figure_id, sha256)`, falling back to
     `pmc.fetch_image_bytes`. Verify the fetched bytes' sha256 matches
     `figures.sha256` when present; on mismatch, record a store error and
     leave the figure `vision_accepted` (do not silently store different
     pixels).
  2. Build attribution from `articles.authors_json/author_count/
     journal_name`; fall back to the JATS refetch only when missing.
  3. Run fetch + decode + crop + image encoding on a worker pool; perform
     file writes for a figure, panel inserts, proposal upserts and the status
     flip in one transaction on the main thread (dedup check uses the new
     index and must see earlier panels from the same run — process results in
     deterministic figure order).
- **Acceptance / tests:** byte-identical panel images, thumbs, `panels` rows
  (excluding timestamps) vs sequential store for fixture figures, including
  ND whole-figure, dedup within one run and across runs, and proposal counts.

#### W8 — Extract reuses in-memory sections (B5, B10)
- **Owns:** `src/visual_pilot/extract_findings.py` and its tests.
- **Depends on:** W5 (C6).
- **Tasks:** use `parse.sections_for(pmcid)` when available, else refetch
  (hinted bundle); cache `_vocabulary` per disease-key tuple; start P4 calls
  as articles become ready via `iter_many`.
- **Acceptance:** identical P4 `input_hash` values vs refetch path for
  fixture articles; no text written to disk/DB.

#### W11 — Provider concurrency probe (ops task, no source edits)
- **Owns:** nothing in `src/`; writes `reports/concurrency_probe.md` in a
  scratch `VP_DATA_DIR`.
- **Depends on:** W3, W6.
- **Tasks:** on a scratch DB with ~80 uncached P3 figures (never the parity
  set's cached inputs), run `judge` at `VP_JUDGE_CONCURRENCY` = 4, 8, 12, 16.
  Record throughput (calls/min), p50/p95 latency, 429 count, timeout count,
  `vision_error` count. Recommend the highest setting where 429s + timeouts
  stay ≤2% and p95 latency grows ≤50% over the 4-worker run.
- **Acceptance:** coordinator sets the default `VP_JUDGE_CONCURRENCY` (via a
  one-line W0-owned config change) from the recommendation.

**Gate G2** (after W4b, W5, W6, W7, W8 merge and W11 recommendation
applied): full tests + parity compare + timing report.

### Phase 3 — orchestration (serial; gate G3)

#### W9 — `run-all` scheduling (B1, B8)
- **Owns:** `src/visual_pilot/cli.py`,
  `tests/visual_pilot/test_run_all_yield.py`.
- **Depends on:** G2.
- **Tasks, in order (each separately tested):**
  1. **In-run vision retry:** after a batch's `judge` pass, rerun `judge` for
     that batch's `vision_error` figures with attempts < 3 (bounded to
     `MAX_ATTEMPTS - 1` extra passes) before evaluating `unfinished`. Only if
     figures remain unfinished does the existing pause apply. The pause
     semantics (never count an unfinished batch as zero-yield) are preserved.
  2. **Fair disease scheduling:** replace strict SLE → DM → AS expansion with
     round-robin batches across in-scope diseases, each with its own
     zero-yield counter and article cap, all under the shared
     `--max-runtime-seconds` and budget. Articles already parsed for one
     disease are not re-parsed for another (existing status gating).
  3. **Stage overlap:** while batch N runs triage → judge → store, prefetch
     and parse batch N+1 in a background worker (its own SQLite connection;
     WAL + busy timeout already set). Batch N+1's figures are not triaged
     until batch N's yield is recorded, so yield accounting and batch
     composition are unchanged. Selection for N+1 happens after N's `store`
     only if coverage-gap ranking would change; otherwise it may be selected
     early — **the harness must show identical selected PMCID sequences**, or
     this sub-task keeps selection after `store` and only overlaps the
     network fetch of already-selected articles.
  4. Keep the triage accumulator semantics: P2 batches are always built from
     the full pending set of a batch in `figure_id` order, chunked by
     `VP_TRIAGE_BATCH` (never per-article micro-batches).
  5. Use cheaper `_snapshot` queries (aggregate SQL) without changing
     results; reuse one connection for budget checks.
- **Acceptance / tests:** existing yield tests pass; new tests cover
  retry-then-continue, round-robin with independent zero-yield stops, and
  overlap not changing batch membership or P2 batch contents.

**Implementation notes** (from the first W9 attempt's read-through —
verified against merged code; no code written yet):

- Task 1 (in-run retry): re-invoke `COMMANDS["judge"]` scoped to the
  batch's `pmcids` after the judge pass and **before** `store`, bounded by
  `judge.MAX_ATTEMPTS - 1` extra passes. Judge already filters
  `vision_error` rows with `attempts < MAX_ATTEMPTS`, so a recovered figure
  still gets stored in-batch. The hardcoded `3` in `_unfinished_figures`
  should become `judge.MAX_ATTEMPTS`.
- Task 3 (overlap): early selection is **not** provably identical —
  `coverage_gaps` (panels written by batch N's `store`) feeds
  `score_article`'s coverage term and the rescue shortlisting inside
  `ranked_pending_articles`, which commits writes via
  `_rescue_if_caption_matches`. Selecting batch N+1 before batch N's store
  can change both the selected sequence and the `articles` table — a
  parity hazard. **Use the fetch-only variant:** keep `select_batch` on
  the main thread after N's `store`; overlap only a write-free prefetch
  (bundle fetch + `jats.parse_article` → warm `parse._JATS_CACHE` via
  `parse._bundle_and_parsed`/`_jats_put`) for the next batch's top-ranked
  `relevant` candidates on a worker with its own `db.connect()`. Cache
  warmth never changes outputs.
- Task 2 (round-robin): needs per-disease `{processed, zero_yield}` state
  with a deque rotation, keeping shared runtime/budget checks per batch.
- Task 5 (`_snapshot`): the findings loop can become an aggregate
  `SELECT DISTINCT COALESCE(json_extract(je.value,'$.finding_key'), je.value)
  FROM panels, json_each(findings_json)` intersected with the approved set;
  all budget/snapshot/unfinished reads can share one long-lived connection
  (WAL autocommit sees fresh commits per statement).
- Timing caveat: the `parse` stage timer reports ~0 because
  `select_batch`'s caption peeks already warm `pmc._article_bundle_cached`
  inside `cli.py`'s untimed selection call — keep the accounting honest
  when restructuring the loop.

**Gate G3:** full tests + parity compare + timing report + a supervised live
`run-all` on a scratch copy of the main DB.

### Deferred (only after G3, only if measurements justify)

- Running diseases truly concurrently (requires a PMCID claim/lock for
  multi-disease articles, per-disease yield attribution that ignores panels
  stored by other workers, and shared budget/runtime accounting).
- A/B-evaluated input-changing items from §1.

---

## 6. Verification gates

### 6.1 Every gate

1. `ruff check src/visual_pilot tests/visual_pilot`
2. `python3 -m pytest tests/visual_pilot`
3. Parity compare (6.2) — must be **identical**.
4. Timing report (6.3) — recorded; regressions explained.

### 6.2 Parity harness (built by W0)

Principle: if every model input is byte-identical, a run seeded with the
baseline's `llm_calls` makes **zero live LLM calls**, and every downstream
artifact must match exactly.

- `parity prepare --out <dir>`: creates a scratch `VP_DATA_DIR` containing a
  fresh DB with: `diseases`, `findings_vocab` (as of baseline), the parity
  set's `articles` rows reset to `relevant` (license and relevance fields
  kept), and no figures/panels.
- **Baseline (G0, only W0's hooks, which change no behavior):** `prepare`,
  then run `run-all --skip-select --pmcids <set> --disease all` with live LLM
  calls (`run-all` also runs `extract` and `report` at the end). Save the
  scratch dir as `reports/parity_baseline/` (DB + files). The recorded
  baseline was rebuilt on merged Phase-2 code because the extract stage's
  refetch could drift between runs (see §0) — rebuilding on the final code
  keeps the comparison honest; all earlier partial baselines were discarded.
  Prefer a baseline with zero `vision_error` figures, but the compare now
  tolerates them (below), so a rerun is not strictly required.
- **Candidate (each gate):** `prepare`, copy the baseline `llm_calls` table
  into the scratch DB, run the same command with `VP_LLM_CACHE_ONLY=1` so any
  changed model input fails loudly as a cache miss.
- `parity compare <baseline> <candidate>` must report:
  - zero cache-miss errors **except on rows whose baseline row already
    errored** (failed calls are never ledgered, so a cache-only replay
    cannot reproduce them — same figure, same status, different error text
    is expected);
  - a candidate `llm_calls` row count equal to the baseline's (no new rows);
  - identical sets of `llm_calls.input_hash` referenced per stage;
  - identical `figures` rows (status, triage_json, vision_json, sha256,
    image_url, image_format, effective_license, attempts) with `error`
    normalized to presence — transient error text is volatile across any
    two runs;
  - identical `panels` rows except timestamps (panel_id, bbox, crop_mode,
    sha256, image_path, thumb_path, attribution_text, license fields,
    findings_json);
  - byte-identical files under `figures/`, `panels/`, `thumbs/`;
  - identical `disease_findings` rows excluding the autoincrement `id`
    and `created_at` (streamed apply legitimately changes insertion order);
  - identical `findings_vocab.proposal_count` values (extract upserts
    proposals on first apply including cache-only replays; re-applies
    over existing text rows do not re-count);
  - no files written for non-accepted figures.
- Selection parity (for W4): `select --dry-run` plus a harness hook that
  dumps per-disease ranked `(pmcid, score)` lists; must match baseline exactly
  on the same turbopuffer namespace snapshot (run baseline and candidate
  back-to-back).

### 6.3 Timing report

Compare `reports/timings_*.json` for the parity run at each gate. Track:
total wall time; per-stage wall time; S3 limiter wait; P3 calls/min and
p95 latency; parse articles/min; store figures/min. Targets are expectations,
not gates: G1 should cut limiter wait sharply; G2 should raise P3 throughput
roughly in proportion to the probed concurrency; G3 should remove idle gaps
between stages and let all three diseases progress within one runtime window.

Note: the parity candidate run is all cache hits, so it measures network and
orchestration cost only. P3 throughput is measured by W11 and by the G3
supervised live run.

---

## 7. Dependency graph and schedule

```
Phase 0:  W0 ──► G0                                        [done]
Phase 1:  G0 ──► W1, W2, W3, W4a, W10 (parallel) ──► G1    [done]
Phase 2:  W1+W2 ──► W5 ──► W8                              [done]
          W3 ──► W6 ──► W7 (also needs W2, W5)             [done]
          W1+W2 ──► W4b                                    [done]
          W3+W6 ──► W11 (probe) ──► config default         [done: keep 4]
          all ──► G2                                       [passed]
Phase 3:  G2 ──► W9 ──► G3                                 [W9 next]
```

Maximum parallelism: 5 agents in Phase 1; 3–4 in Phase 2 (W5 ∥ W6 ∥ W4b,
then W7 ∥ W8, then W11). **All parallel phases are complete; W9 is a
single-agent serial workstream — no further fan-out needed.**

### File ownership matrix

All workstreams except W9 are merged; the table below still governs W9 —
only `cli.py` and `test_run_all_yield.py` remain open for editing, every
other file is frozen.

| File | Owner | Others |
|---|---|---|
| `config.py`, `timing.py`, `parity.py`, `env.example` | W0 (merged) | read-only |
| `pmc.py` | W1 (merged) | read-only |
| `db.py` | W2 (merged) | read-only |
| `llm.py` | W3 (merged) | read-only |
| `select_articles.py`, `retrieval.py` | W4 (merged) | read-only |
| `parse.py` | W5 (merged) | read-only |
| `judge.py`, `originals.py` | W6 (merged) | read-only |
| `store.py` | W7 (merged) | read-only |
| `extract_findings.py` | W8 (merged) | read-only |
| `cli.py` | **W9 (open)** | read-only |
| `triage.py` | W10 (merged) | read-only |
| `test_judge_store.py` | W6/W7 (merged; store tests live in `test_store.py`) | — |
| `test_run_all_yield.py` | **W9 (open)** | — |
| `prompts.py`, `frontend/`, `src/*.py` | nobody | never edit |

---

## 8. Handoff note template

```
Workstream: Wn — <title>
Branch: perf/wn-<slug>   Base: <commit>
Files changed: <list — must be within Owns>
Contracts implemented/consumed: <C-ids>
Tests added/changed: <list>
Verification: ruff <pass/fail>, pytest <N passed>
Invariant evidence: <e.g. input_hash golden test, byte-identical crops test>
Behavior flags/defaults changed: <env keys and values>
Known limitations / follow-ups: <list>
```

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Provider rate limits / schema compatibility | W11 probe ran on DeepInfra GLM: zero 429s at c≤16, p95 latency fails criterion >4 → default stays 4. `space-bunny-free` upstream currently 400s on union-type schemas — resolved: production defaults moved to OpenRouter `stealth/space-bunny-alpha` (§0) |
| Higher S3 rate triggers throttling (503 SlowDown) | Existing backoff honors `Retry-After`; `VP_S3_RPS` is tunable; timing report tracks retries |
| Thread-safety of shared SQLite connection | All writes stay on the owning stage's main thread; LLM ledger writes stay under the existing lock; background parse (W9) uses its own connection |
| Memory growth from in-memory caches | Every cache bounded (C3 LRU sizes, C5 MB cap, C6 bounded); judge buffers ≤2× in-flight |
| Parallel code changes call order | Call order is not an input; parity harness checks inputs and outputs, not order. Coverage-based figure priority still decides dispatch order |
| Turbopuffer client not thread-safe | W4 verifies; fallback one namespace handle per worker |
| Parity baseline drift (vocab approvals, provider model updates) | Baseline and candidate always seed from the same frozen baseline DB; live comparisons (selection) run back-to-back |
| Pipelined selection changes which articles are picked | W9 task 3 must prove identical PMCID sequences or keep selection after `store` |
