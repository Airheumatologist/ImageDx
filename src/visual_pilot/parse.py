"""Stage 3: in-memory figure parsing from PMC JATS XML (workstream W5a).

For each ``relevant`` article, fetch the JATS bundle into memory
(``pmc.get_article_bundle``), parse it with ``jats.parse_article`` and write
one ``figures`` row per ``<fig>`` (INSERT OR IGNORE — resumable). Pre-rejects
(third-party wording or foreign copyright holder, disallowed effective
license, missing graphic) land directly as ``caption_rejected`` with a
synthetic ``triage_json``; everything else becomes ``pending`` for stage 4.

No full text and no images are ever written to disk. ``articles.study_region``
records which source supplied the region, and parse failures leave the
article at ``parse_error`` with the exception in ``articles.error``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import time

from . import article_rank, config, db, diseases, jats, pmc

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_ARTICLES = 600
DEFAULT_MAX_RUNTIME_SECONDS = 900
_JATS_CACHE: dict[str, tuple[object, jats.ParsedArticle]] = {}


def coverage_gaps(conn, disease_key: str) -> set[str]:
    """Approved findings not yet represented by a stored panel for disease."""
    wanted = {
        r["finding_key"]
        for r in conn.execute(
            "SELECT finding_key FROM findings_vocab WHERE approved=1 "
            "AND EXISTS (SELECT 1 FROM json_each(disease_keys_json) je "
            "WHERE je.value=?)",
            (disease_key,),
        )
    }
    found: set[str] = set()
    for row in conn.execute(
        "SELECT findings_json FROM panels WHERE disease_key=?", (disease_key,)
    ):
        for value in db.from_json(row["findings_json"], []) or []:
            key = value.get("finding_key") if isinstance(value, dict) else value
            if key:
                found.add(str(key))
    return wanted - found


def _caption_candidates(article_row: dict, disease_key: str) -> list[dict] | None:
    """Cheap JATS caption peek for shortlist ranking; cached for parse_article."""
    pmcid = article_row["pmcid"]
    try:
        if pmcid not in _JATS_CACHE:
            bundle = pmc.get_article_bundle(pmcid)
            _JATS_CACHE[pmcid] = (bundle, jats.parse_article(bundle.xml_text))
        bundle, parsed = _JATS_CACHE[pmcid]
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


def _rescue_if_caption_matches(conn, article_row: dict, disease_key: str) -> bool:
    """Allow a licensed review through when a specific target-disease figure does."""
    if pmc.license_allows(article_row.get("license_code")) is None:
        return False
    captions = article_row.get("caption_candidates") or []
    confirms_target = any(
        signal["useful"] and (signal["disease_hits"] or signal["finding_hits"])
        for signal in (
            article_rank.caption_signal(
                caption.get("caption", ""),
                disease_key,
                eligible=caption.get("eligible", False),
            )
            for caption in captions
        )
    )
    if not confirms_target:
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


def ranked_pending_articles(
    conn,
    disease_key: str,
    *,
    gaps=None,
    pmcids: set[str] | None = None,
    peek_limit: int = 100,
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
    with ThreadPoolExecutor(max_workers=config.VP_CONCURRENCY) as pool:
        caption_results = list(
            pool.map(
                lambda item: _caption_candidates(item[0], disease_key),
                shortlisted,
            )
        )
    for (row, _score), candidates in zip(shortlisted, caption_results):
        if candidates is not None:
            row["caption_candidates"] = candidates
        if row.get("status") != "relevant" and _rescue_if_caption_matches(conn, row, disease_key):
            rescues.add(row["pmcid"])
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
) -> list[dict]:
    """Next resumable article batch for one disease."""
    ranked = ranked_pending_articles(
        conn,
        disease_key,
        pmcids=pmcids,
        peek_limit=(min(max(0, batch_size * 2), 100) if peek_captions else 0),
    )
    selected = ranked[: max(0, batch_size)]
    keep = {row["pmcid"] for row in selected}
    for pmcid in set(_JATS_CACHE) - keep:
        _JATS_CACHE.pop(pmcid, None)
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


def parse_article(conn, article_row, stats: dict) -> str:
    """Fetch + parse one article and write its figure rows. Returns status."""
    pmcid = article_row["pmcid"]
    try:
        if pmcid in _JATS_CACHE:
            bundle, parsed = _JATS_CACHE[pmcid]
        else:
            bundle = pmc.get_article_bundle(pmcid)
            parsed = jats.parse_article(bundle.xml_text)
            _JATS_CACHE[pmcid] = (bundle, parsed)
        for fig in parsed.figures:
            row = _figure_row(pmcid, fig, article_row["license_code"], bundle.resolver)
            _insert_figure(conn, row)
            stats[row["status"]] = stats.get(row["status"], 0) + 1
        db.set_status(
            conn,
            "articles",
            pmcid,
            "parsed",
            study_region=study_region_for(article_row, parsed),
        )
        status = "parsed"
    except Exception as exc:  # noqa: BLE001 - one bad article must not stop the run
        logger.warning("parse failed for %s: %s", pmcid, exc)
        db.set_status(conn, "articles", pmcid, "parse_error", error=str(exc))
        stats["parse_error_articles"] = stats.get("parse_error_articles", 0) + 1
        status = "parse_error"
    finally:
        _JATS_CACHE.pop(pmcid, None)
    conn.commit()
    return status


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
                conn, key, min(batch_size, room), peek_captions=not args.dry_run
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
    for row in articles:
        if processed and time.monotonic() - started >= max_runtime:
            print(f"parse stopped at runtime safety limit ({max_runtime}s); rerun to resume")
            break
        status = parse_article(conn, row, stats)
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
