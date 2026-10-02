"""Figure-row construction from parsed JATS (used by ``discover``).

``_figure_row`` decides each ``figures`` row's initial status: third-party
wording or a foreign copyright holder, a disallowed effective license, or a
missing graphic land directly as ``caption_rejected`` with a synthetic
``triage_json``; everything else is ``pending``. ``_c6_article_fields``
persists the S3 prefix and media list so later stages refetch cheaply, and
``sections_for`` keeps ``body_sections`` in a bounded in-memory cache for the
extract stage. No full text and no images are ever written to disk.
"""

from __future__ import annotations

from collections import OrderedDict
import threading

from . import db, jats, pmc

# parsed.body_sections for articles parsed in this process (consumed by
# extract via sections_for). Bounded LRU, memory only — full text is never
# written to disk or SQLite.
_SECTIONS_CACHE_LIMIT = 256
_SECTIONS_CACHE: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
_CACHE_LOCK = threading.Lock()


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
