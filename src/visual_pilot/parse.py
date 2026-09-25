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

import json
import logging

from . import config, db, diseases, jats, pmc

logger = logging.getLogger(__name__)

DEFAULT_CAP = 150
_COUNTS_FILE = "stage2_counts.json"


def _over_cap_diseases(in_scope: list[str]) -> list[str]:
    path = config.reports_dir() / _COUNTS_FILE
    if not path.exists():
        return []
    try:
        counts = json.loads(path.read_text())
    except ValueError:
        return []
    return [k for k in in_scope if (counts.get(k) or {}).get("over_cap")]


def _capped_articles(conn, in_scope: list[str], cap: int) -> dict[str, dict]:
    """{pmcid: article row}: union over diseases of the top-`cap` relevant."""
    picked: dict[str, dict] = {}
    for key in in_scope:
        rows = db.rows_with_status(conn, "articles", "relevant", disease=key)
        rows.sort(key=lambda r: (r["retrieval_score"] or 0.0), reverse=True)
        for row in rows[:cap]:
            picked.setdefault(row["pmcid"], row)
    return picked


def _articles_to_parse(conn, in_scope: list[str], cap: int, accept_cap: bool):
    if accept_cap:
        rows = _capped_articles(conn, in_scope, cap)
    else:
        rows = {}
        for key in in_scope:
            for row in db.rows_with_status(conn, "articles", "relevant", disease=key):
                rows.setdefault(row["pmcid"], row)
    return sorted(
        rows.values(), key=lambda r: (r["retrieval_score"] or 0.0), reverse=True
    )


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
        bundle = pmc.get_article_bundle(pmcid)
        parsed = jats.parse_article(bundle.xml_text)
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
    conn.commit()
    return status


def run(args) -> int:
    in_scope = (
        list(diseases.DISEASE_KEYS) if args.disease == "all" else [args.disease]
    )
    cap = getattr(args, "cap", None) or DEFAULT_CAP
    accept_cap = bool(getattr(args, "accept_cap", False))

    over_cap = _over_cap_diseases(in_scope)
    if over_cap and not accept_cap:
        print(
            "relevant articles exceed the stage-2 cap for: "
            + ", ".join(over_cap)
            + ". Human confirmation is required; rerun with --accept-cap to "
            "parse only the top-capped articles per disease."
        )
        return 3

    conn = db.init_db()
    articles = _articles_to_parse(conn, in_scope, cap, accept_cap)
    pmcid_filter = getattr(args, "pmcids", None)
    if pmcid_filter:
        allowed = set(pmcid_filter)
        articles = [r for r in articles if r["pmcid"] in allowed]
    limit = getattr(args, "limit", None)
    if limit is not None:
        articles = articles[:limit]

    if args.dry_run:
        print(f"dry-run: {len(articles)} relevant articles would be parsed")
        for row in articles[:20]:
            print(f"  {row['pmcid']}  score={row['retrieval_score']:.5f}  {row['title']}")
        if len(articles) > 20:
            print(f"  ... and {len(articles) - 20} more")
        conn.close()
        return 0

    stats: dict[str, int] = {}
    for row in articles:
        status = parse_article(conn, row, stats)
        print(f"{row['pmcid']}: {status}")

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
    print(f"articles processed: {len(articles)}; statuses: {article_counts}")
    print(f"figures: {status_counts}")
    conn.close()
    return 0
