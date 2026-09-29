"""Stage 3: in-memory figure parsing from PMC JATS XML (workstream W5).

For each ``relevant`` article, fetch the JATS bundle into memory
(``pmc.get_article_bundle``), parse it with ``jats.parse_article`` and write
one ``figures`` row per ``<fig>`` (INSERT OR IGNORE — resumable). Pre-rejects
(third-party wording or foreign copyright holder, disallowed effective
license, missing graphic) land directly as ``caption_rejected`` with a
synthetic ``triage_json``; everything else becomes ``pending`` for stage 4.

No full text and no images are ever written to disk. ``articles.study_region``
records which source supplied the region, and parse failures leave the
article at ``parse_error`` with the exception in ``articles.error``.

Throughput-plan changes (docs/visual_pilot_plan.md §5 W5, contracts C3/C6):
bundle fetches and ``jats.parse_article`` run on a pool of
``config.VP_FETCH_CONCURRENCY`` workers while all DB writes and commits stay
on the calling thread in the same article order as the sequential path.
Articles whose rows already carry ``s3_prefix`` (and optionally
``media_files_json``) fetch via the hinted ``get_article_bundle`` path.
Parse persists the C2 metadata columns and keeps ``body_sections`` in a
bounded in-memory cache (``sections_for``) for stage 7. Peeked-but-unselected
bundles stay in the bounded ``_JATS_CACHE`` across batches in this process.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import logging
import threading
import time

from . import article_rank, config, db, diseases, jats, pmc
from . import manifestation_queue

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_ARTICLES = 2000
DEFAULT_MAX_RUNTIME_SECONDS = 3600

# Caption-peek results (bundle + parsed article), bounded LRU. Peeked but
# unselected bundles are kept across select_batch calls (C6) so a later batch
# — or the parse itself — reuses them instead of refetching. Memory only.
_JATS_CACHE_LIMIT = 256
_JATS_CACHE: OrderedDict[str, tuple[object, jats.ParsedArticle]] = OrderedDict()
_CACHE_LOCK = threading.Lock()

# C6: parsed.body_sections for articles parsed in this process (consumed by
# extract via sections_for). Bounded LRU, memory only — full text is never
# written to disk or SQLite.
_SECTIONS_CACHE_LIMIT = 256
_SECTIONS_CACHE: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()


def _jats_get(pmcid: str) -> tuple[object, jats.ParsedArticle] | None:
    with _CACHE_LOCK:
        entry = _JATS_CACHE.get(pmcid)
        if entry is not None:
            _JATS_CACHE.move_to_end(pmcid)
        return entry


def _jats_put(pmcid: str, entry: tuple[object, jats.ParsedArticle]) -> None:
    with _CACHE_LOCK:
        _JATS_CACHE[pmcid] = entry
        _JATS_CACHE.move_to_end(pmcid)
        while len(_JATS_CACHE) > _JATS_CACHE_LIMIT:
            _JATS_CACHE.popitem(last=False)


def _jats_drop(pmcid: str) -> None:
    with _CACHE_LOCK:
        _JATS_CACHE.pop(pmcid, None)


def sections_for(pmcid: str) -> list[tuple[str, str]] | None:
    """C6: ``parsed.body_sections`` for an article parsed in this process.

    Returns ``None`` on a miss (never-parsed or evicted); the consumer then
    refetches via the hinted bundle path. Memory only, bounded LRU.
    """
    with _CACHE_LOCK:
        sections = _SECTIONS_CACHE.get(pmcid)
        if sections is not None:
            _SECTIONS_CACHE.move_to_end(pmcid)
        return sections


def _sections_put(pmcid: str, sections: list[tuple[str, str]]) -> None:
    with _CACHE_LOCK:
        _SECTIONS_CACHE[pmcid] = sections
        _SECTIONS_CACHE.move_to_end(pmcid)
        while len(_SECTIONS_CACHE) > _SECTIONS_CACHE_LIMIT:
            _SECTIONS_CACHE.popitem(last=False)


def _row_get(row, key):
    """``row[key]`` for dicts and sqlite3.Rows; None when absent."""
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _bundle_for_row(article_row):
    """``pmc.get_article_bundle``, C3-hinted when the row has ``s3_prefix``.

    ``media_files_json`` (when present, e.g. persisted by the license stage
    or an earlier parse) is passed through as the ``media_files`` hint. The
    hinted path produces a resolver identical to the unhinted one, so figure
    rows are unchanged either way.
    """
    prefix = _row_get(article_row, "s3_prefix")
    if not prefix:
        return pmc.get_article_bundle(article_row["pmcid"])
    media = db.from_json(_row_get(article_row, "media_files_json"), None)
    if not isinstance(media, list):
        media = None
    return pmc.get_article_bundle(
        article_row["pmcid"], prefix=prefix, media_files=media
    )


def _bundle_and_parsed(article_row) -> tuple[object, jats.ParsedArticle]:
    """Fetch + JATS-parse one article. No DB access — safe on a pool worker."""
    pmcid = article_row["pmcid"]
    cached = _jats_get(pmcid)
    if cached is not None:
        return cached
    bundle = _bundle_for_row(article_row)
    return bundle, jats.parse_article(bundle.xml_text)


def coverage_gaps(conn, disease_key: str) -> dict[str, int]:
    """Under-target approved findings -> current distinct-image count."""
    wanted = {
        r["finding_key"]
        for r in conn.execute(
            "SELECT finding_key FROM findings_vocab WHERE approved=1 "
            "AND EXISTS (SELECT 1 FROM json_each(disease_keys_json) je "
            "WHERE je.value=?)",
            (disease_key,),
        )
    }
    counts = manifestation_queue.published_coverage(conn, disease_key)
    target = config.VP_FINDING_IMAGE_TARGET
    return {key: counts.get(key, 0) for key in wanted if counts.get(key, 0) < target}


def _caption_candidates(article_row: dict, disease_key: str) -> list[dict] | None:
    """Cheap JATS caption peek for shortlist ranking; cached for parse_article."""
    pmcid = article_row["pmcid"]
    try:
        cached = _jats_get(pmcid)
        if cached is None:
            bundle = _bundle_for_row(article_row)
            cached = (bundle, jats.parse_article(bundle.xml_text))
            _jats_put(pmcid, cached)
        bundle, parsed = cached
        return [
            {
                "caption": fig.caption,
                "eligible": _figure_row(
                    pmcid, fig, article_row.get("license_code"), bundle.resolver
                )["status"] == "pending",
            }
            for fig in parsed.figures
        ]
    except Exception as exc:  # noqa: BLE001 - preserve passage/title fallback
        logger.debug("caption peek failed for %s: %s", pmcid, exc)
        return None


def _strong_visual_passage(article_row: dict, disease_key: str) -> bool:
    """Require finding/modality evidence in a matched passage before rescue."""
    for passage in article_row.get("matched_passages") or []:
        if not isinstance(passage, dict):
            continue
        features = article_rank.score_article(
            {"matched_passages": [passage]}, disease_key
        )["parts"]
        if features.get("finding_context", 0) > 0 and (
            features.get("modality", 0) > 0 or features.get("visual_section", 0) > 0
        ):
            return True
    return False


def _rescue_if_caption_matches(
    conn, article_row: dict, disease_key: str, min_captions: int = 1
) -> bool:
    """Allow a licensed review through when enough target-disease figures do."""
    if pmc.license_allows(article_row.get("license_code")) is None:
        return False
    captions = article_row.get("caption_candidates") or []
    confirming = sum(
        bool(signal["useful"] and (signal["disease_hits"] or signal["finding_hits"]))
        for signal in (
            article_rank.caption_signal(
                caption.get("caption", ""),
                disease_key,
                eligible=caption.get("eligible", False),
            )
            for caption in captions
        )
    )
    if confirming < min_captions:
        return False
    keys = set(db.from_json(article_row.get("primary_disease_keys_json"), []) or [])
    keys.add(disease_key)
    db.set_status(
        conn,
        "articles",
        article_row["pmcid"],
        "relevant",
        primary_disease_keys_json=db.to_json(sorted(keys)),
        relevance_decision="relevant",
        relevance_reason="visual_figure_caption_rescue",
    )
    conn.commit()
    article_row["status"] = "relevant"
    article_row["primary_disease_keys_json"] = db.to_json(sorted(keys))
    return True


def _rescue_persisted_candidates(conn, disease_key, gaps, pmcids) -> list[dict]:
    """W6 caption-rescue lane over persisted ``candidate`` articles.

    License-checks the strongest candidate rows (review-filtered at
    retrieval, type-filtered at persistence) that carry strong visual passage
    evidence, then rescues the license-passing ones whose captions confirm at
    least ``VP_CAPTION_RESCUE_MIN_CAPTIONS`` target-disease figures. Only
    called on the persisting selection path — it writes license outcomes and
    relevance flips.
    """
    from . import select_articles

    peek = config.VP_CAPTION_RESCUE_PEEK
    if not peek:
        return []
    rows = [
        dict(row)
        for row in db.rows_with_status(
            conn, "articles", "candidate", disease=disease_key
        )
    ]
    if pmcids is not None:
        rows = [row for row in rows if row["pmcid"] in pmcids]
    for row in rows:
        row["matched_passages"] = db.from_json(
            row.get("retrieval_evidence_json"), []
        ) or []
    rows = [row for row in rows if _strong_visual_passage(row, disease_key)]
    if not rows:
        return []
    ranked = article_rank.rank_articles(rows, disease_key, coverage_gaps=gaps)
    shortlist = [row for row, _score in ranked[:peek]]
    outcomes = select_articles.join_licenses([row["pmcid"] for row in shortlist])
    licensed = []
    for row, (pmcid, outcome) in zip(shortlist, outcomes):
        select_articles.apply_license(conn, pmcid, outcome)
        fresh = conn.execute(
            "SELECT status, license_code FROM articles WHERE pmcid=?", (pmcid,)
        ).fetchone()
        if fresh is not None:
            row["status"] = fresh["status"]
            row["license_code"] = fresh["license_code"]
        if row.get("status") == "license_ok":
            licensed.append(row)
    conn.commit()
    if not licensed:
        return []
    with ThreadPoolExecutor(max_workers=config.VP_FETCH_CONCURRENCY) as pool:
        caption_results = list(
            pool.map(lambda row: _caption_candidates(row, disease_key), licensed)
        )
    rescued = []
    for row, candidates in zip(licensed, caption_results):
        if candidates is not None:
            row["caption_candidates"] = candidates
        if _rescue_if_caption_matches(
            conn, row, disease_key,
            min_captions=config.VP_CAPTION_RESCUE_MIN_CAPTIONS,
        ):
            rescued.append(row)
    return rescued


def ranked_pending_articles(
    conn,
    disease_key: str,
    *,
    gaps=None,
    pmcids: set[str] | None = None,
    peek_limit: int = 100,
    persist: bool = True,
) -> list[dict]:
    """Rerank relevant candidates with bounded JATS caption evidence."""
    rows = [
        dict(row)
        for row in db.rows_with_status(conn, "articles", "relevant", disease=disease_key)
    ]
    if pmcids is not None:
        rows = [row for row in rows if row["pmcid"] in pmcids]
    rescue_rows = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM articles WHERE status IN ('license_ok','irrelevant')"
        )
        if pmcids is None or row["pmcid"] in pmcids
    ]
    for row in rows:
        row["matched_passages"] = db.from_json(
            row.get("retrieval_evidence_json"), []
        ) or []
    rescue_rows = [
        row
        for row in rescue_rows
        if pmc.license_allows(row.get("license_code")) is not None
    ]
    for row in rescue_rows:
        row["matched_passages"] = db.from_json(row.get("retrieval_evidence_json"), []) or []
    rescue_rows = [
        row for row in rescue_rows if _strong_visual_passage(row, disease_key)
    ]
    rows.extend(rescue_rows)
    current_gaps = gaps if gaps is not None else coverage_gaps(conn, disease_key)
    initial = article_rank.rank_articles(rows, disease_key, coverage_gaps=current_gaps)
    shortlisted = initial[: max(0, peek_limit)]
    rescues: set[str] = set()
    with ThreadPoolExecutor(max_workers=config.VP_FETCH_CONCURRENCY) as pool:
        caption_results = list(
            pool.map(
                lambda item: _caption_candidates(item[0], disease_key),
                shortlisted,
            )
        )
    for (row, _score), candidates in zip(shortlisted, caption_results):
        if candidates is not None:
            row["caption_candidates"] = candidates
        if row.get("status") != "relevant" and _rescue_if_caption_matches(conn, row, disease_key, min_captions=1):
            rescues.add(row["pmcid"])
    if persist:
        rows.extend(
            _rescue_persisted_candidates(conn, disease_key, current_gaps, pmcids)
        )
    rows = [row for row in rows if row.get("status") == "relevant"]
    return [
        row
        for row, _score in article_rank.rank_articles(
            rows, disease_key, coverage_gaps=current_gaps
        )
    ]


def select_batch(
    conn,
    disease_key: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    pmcids: set[str] | None = None,
    peek_captions: bool = True,
    persist: bool = True,
) -> list[dict]:
    """Next resumable article batch with finding-lane reservations."""
    gaps = manifestation_queue.sync_candidates(conn, disease_key) if persist else coverage_gaps(conn, disease_key)
    ranked = ranked_pending_articles(
        conn,
        disease_key,
        gaps=gaps,
        pmcids=pmcids,
        peek_limit=(min(max(0, batch_size * 2), 100) if peek_captions else 0),
        persist=persist,
    )
    selected = manifestation_queue.reserve_batch(
        conn, disease_key, ranked, batch_size, gaps, persist=persist
    )
    # C6: peeked-but-unselected bundles stay in the bounded _JATS_CACHE so a
    # later batch in this process reuses them (previously they were dropped).
    return selected


def _figure_row(pmcid: str, fig: jats.FigureInfo, article_license: str | None, resolver) -> dict:
    """Build the figures-row payload (status decided here)."""
    figure_id = f"{pmcid}:{fig.fig_id}"
    effective_license = (
        pmc.normalize_license(fig.fig_license_raw)
        if fig.fig_license_raw
        else article_license
    )
    image_url = image_format = error = None
    if fig.graphic_href:
        ref = resolver(fig.graphic_href)
        image_url = ref.url
        image_format = ref.format
        if ref.needs_bytes:
            image_url = None
            error = "needs_bytes"

    status = "pending"
    triage = None
    if fig.third_party:
        status = "caption_rejected"
        triage = {
            "route": "drop",
            "reason": "third_party",
            "source": "parse",
            "third_party": True,
            "third_party_quote": fig.third_party_reason,
        }
    elif pmc.license_allows(effective_license) is None:
        status = "caption_rejected"
        triage = {"route": "drop", "reason": "license", "source": "parse"}
    elif not fig.graphic_href:
        status = "caption_rejected"
        triage = {"route": "drop", "reason": "no_graphic", "source": "parse"}

    return {
        "figure_id": figure_id,
        "pmcid": pmcid,
        "label": fig.label,
        "caption": fig.caption,
        "in_text_mentions_json": db.to_json(fig.in_text_mentions),
        "fig_permissions_text": fig.permissions_text,
        "effective_license": effective_license,
        "image_url": image_url,
        "image_format": image_format,
        "status": status,
        "triage_json": db.to_json(triage) if triage else None,
        "error": error,
    }


def _insert_figure(conn, row: dict) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO figures "
        "(figure_id, pmcid, label, caption, in_text_mentions_json, "
        " fig_permissions_text, effective_license, image_url, image_format, "
        " status, triage_json, error) VALUES "
        "(:figure_id, :pmcid, :label, :caption, :in_text_mentions_json, "
        " :fig_permissions_text, :effective_license, :image_url, :image_format, "
        " :status, :triage_json, :error)",
        row,
    )


def study_region_for(article_row, parsed: jats.ParsedArticle) -> str | None:
    country = (article_row["country"] or "").strip()
    if country:
        return f"{country} (article metadata)"
    if parsed.corresp_country:
        return f"{parsed.corresp_country} (corresponding-author affiliation)"
    if parsed.first_aff_country:
        return f"{parsed.first_aff_country} (first author affiliation)"
    return None


def _c6_article_fields(article_row, bundle, fig_rows) -> dict:
    """C6: ``s3_prefix``/``media_files_json`` to persist on the article row.

    Only filled when the row does not already carry them (the license stage
    may have persisted hints first). Values come from data the bundle already
    fetched — the S3 metadata dict, falling back to the resolved figure refs
    (``{S3_BASE}/{prefix}/{name}``) — so this costs zero extra requests.
    """
    meta = getattr(bundle, "metadata", None) or {}
    prefix = None
    if meta.get("pmcid") and meta.get("version") is not None:
        prefix = f"{meta['pmcid']}.{meta['version']}"
    media_files: set[str] = set()
    for media in meta.get("media_urls") or []:
        parts = str(media).split("?")[0].split("/", 3)
        if len(parts) == 4 and parts[3]:
            media_files.add(parts[3].rsplit("/", 1)[-1])
    if prefix is None or not media_files:
        for row in fig_rows:
            url = row.get("image_url") or ""
            if not url.startswith(pmc.S3_BASE + "/"):
                continue
            dir_prefix, _, name = url[len(pmc.S3_BASE) + 1:].rpartition("/")
            if prefix is None and dir_prefix:
                prefix = dir_prefix
            if name:
                media_files.add(name)
    fields: dict = {}
    if not _row_get(article_row, "s3_prefix") and prefix:
        fields["s3_prefix"] = prefix
    if not _row_get(article_row, "media_files_json") and media_files:
        fields["media_files_json"] = db.to_json(sorted(media_files))
    return fields


def _apply_parsed(conn, article_row, bundle, parsed, stats) -> None:
    """Main thread only: figure inserts + article fields for one article."""
    pmcid = article_row["pmcid"]
    fig_rows = []
    for fig in parsed.figures:
        row = _figure_row(pmcid, fig, article_row["license_code"], bundle.resolver)
        _insert_figure(conn, row)
        fig_rows.append(row)
        stats[row["status"]] = stats.get(row["status"], 0) + 1
    db.set_status(
        conn,
        "articles",
        pmcid,
        "parsed",
        study_region=study_region_for(article_row, parsed),
        authors_json=db.to_json(parsed.authors),
        author_count=parsed.author_count,
        journal_name=parsed.journal_name,
        **_c6_article_fields(article_row, bundle, fig_rows),
    )
    _sections_put(pmcid, parsed.body_sections)


def _finish_article(conn, article_row, outcome, stats) -> str:
    """Apply one fetched/parsed outcome on the calling thread.

    ``outcome`` is either ``(bundle, parsed)`` or the worker's ``Exception``;
    all DB writes and the commit happen here, so error isolation and write
    order match the sequential path exactly.
    """
    pmcid = article_row["pmcid"]
    try:
        if isinstance(outcome, Exception):
            raise outcome
        bundle, parsed = outcome
        _apply_parsed(conn, article_row, bundle, parsed, stats)
        status = "parsed"
    except Exception as exc:  # noqa: BLE001 - one bad article must not stop the run
        logger.warning("parse failed for %s: %s", pmcid, exc)
        db.set_status(conn, "articles", pmcid, "parse_error", error=str(exc))
        stats["parse_error_articles"] = stats.get("parse_error_articles", 0) + 1
        status = "parse_error"
    finally:
        _jats_drop(pmcid)
    conn.commit()
    manifestation_queue.record_article_outcome(conn, pmcid, status)
    return status


def parse_article(conn, article_row, stats: dict) -> str:
    """Fetch + parse one article and write its figure rows. Returns status."""
    try:
        outcome = _bundle_and_parsed(article_row)
    except Exception as exc:  # noqa: BLE001 - same isolation as the pool path
        outcome = exc
    return _finish_article(conn, article_row, outcome, stats)


def _prefetched(articles: list):
    """Yield ``(article_row, outcome)`` in input order while fetching ahead.

    ``get_article_bundle`` + ``jats.parse_article`` run on a pool of
    ``config.VP_FETCH_CONCURRENCY`` workers; each outcome is either
    ``(bundle, parsed)`` or the worker's ``Exception``. Results are keyed by
    article position and yielded strictly in input order — only the
    fetch/parse step is parallel; the caller applies DB writes on its own
    thread in the same order as the sequential path. Lookahead is bounded
    (2x workers) so a mid-run stop leaves little work in flight.
    """
    workers = max(1, int(getattr(config, "VP_FETCH_CONCURRENCY", 8)))
    ahead = workers * 2
    pending: dict[int, Future] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def submit(index: int) -> None:
            pending[index] = pool.submit(_bundle_and_parsed, articles[index])

        for index in range(min(ahead, len(articles))):
            submit(index)
        for index, row in enumerate(articles):
            future = pending.pop(index)
            try:
                yield row, future.result()
            except Exception as exc:  # noqa: BLE001 - per-article isolation
                yield row, exc
            follow = index + ahead
            if follow < len(articles):
                submit(follow)


def run(args) -> int:
    in_scope = (
        list(diseases.DISEASE_KEYS) if args.disease == "all" else [args.disease]
    )
    conn = db.init_db()
    batch_size = max(1, int(getattr(args, "batch_size", None) or DEFAULT_BATCH_SIZE))
    max_articles = max(1, int(getattr(args, "max_articles", None) or DEFAULT_MAX_ARTICLES))
    max_runtime = max(1, int(getattr(args, "max_runtime_seconds", None) or DEFAULT_MAX_RUNTIME_SECONDS))
    pmcid_filter = getattr(args, "pmcids", None)
    if pmcid_filter:
        allowed = set(pmcid_filter)
        unique = {}
        for key in in_scope:
            batch = select_batch(
                conn,
                key,
                min(max_articles, len(allowed)),
                pmcids=allowed,
                peek_captions=not args.dry_run,
                persist=not args.dry_run,
            )
            for row in batch:
                unique.setdefault(row["pmcid"], row)
        articles = list(unique.values())
        articles.sort(key=lambda r: (r["retrieval_score"] or 0.0), reverse=True)
    else:
        unique = {}
        for key in in_scope:
            room = max_articles - len(unique)
            if room <= 0:
                break
            for row in select_batch(
                conn, key, min(batch_size, room), peek_captions=not args.dry_run,
                persist=not args.dry_run,
            ):
                unique.setdefault(row["pmcid"], row)
        articles = list(unique.values())
    limit = getattr(args, "limit", None)
    if limit is not None:
        articles = articles[:limit]

    if args.dry_run:
        print(f"dry-run: {len(articles)} relevant articles in next ranked batch")
        for row in articles[:20]:
            score = article_rank.score_article(row, in_scope[0])["score"]
            print(f"  {row['pmcid']}  visual-score={score:.2f}  {row['title']}")
        if len(articles) > 20:
            print(f"  ... and {len(articles) - 20} more")
        conn.close()
        return 0

    stats: dict[str, int] = {}
    started = time.monotonic()
    processed = 0
    for row, outcome in _prefetched(articles):
        if processed and time.monotonic() - started >= max_runtime:
            print(f"parse stopped at runtime safety limit ({max_runtime}s); rerun to resume")
            break
        status = _finish_article(conn, row, outcome, stats)
        print(f"{row['pmcid']}: {status}")
        processed += 1

    status_counts = {
        r["status"]: r["n"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM figures GROUP BY status"
        )
    }
    article_counts = {
        r["status"]: r["n"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM articles GROUP BY status"
        )
    }
    print(f"articles processed: {processed}; statuses: {article_counts}")
    print(f"figures: {status_counts}")
    conn.close()
    return 0
