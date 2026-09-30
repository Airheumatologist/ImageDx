"""Europe PMC core metadata and bounded, auditable article-source signals.

Citation counts cover Europe PMC's citation network. Citations per publication
year are an age adjustment, not a field-normalized metric or journal impact
factor. Unknown metadata is kept unknown; a failed fetch cannot clear a known
retraction. This module does not make or change licensing decisions.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone

from . import config, db, pmc

logger = logging.getLogger(__name__)


def normalize_record(record: dict, *, fetched_at: str | None = None) -> dict:
    journal_info = record.get("journalInfo") or {}
    journal = journal_info.get("journal") or {}
    types = (record.get("pubTypeList") or {}).get("pubType") or []
    if isinstance(types, str):
        types = [types]
    corrections = (record.get("commentCorrectionList") or {}).get("commentCorrection") or []
    if isinstance(corrections, dict):
        corrections = [corrections]
    # Retraction *in* links mark the original article. Retraction *of* marks
    # the notice; both are unsuitable as candidate image articles.
    retracted = any("retract" in str(value).casefold() for value in types) or any(
        "retract" in str(item.get("type") or "").casefold()
        for item in corrections if isinstance(item, dict)
    ) or str(record.get("isRetracted", "")).upper() == "Y"
    try:
        citations = max(0, int(record["citedByCount"]))
    except (KeyError, ValueError, TypeError):
        citations = None
    return {
        "source": "europe_pmc_core",
        "source_url": "https://europepmc.org/RestfulWebService",
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        "record_id": record.get("id"), "record_source": record.get("source"),
        "journal_title": journal.get("title") or record.get("journalTitle"),
        "journal_abbreviation": journal.get("medlineAbbreviation"),
        "journal_issn": journal.get("issn") or record.get("journalIssn"),
        "journal_essn": journal.get("essn"),
        "publication_types": types,
        "abstract": record.get("abstractText"),
        "first_publication_date": record.get("firstPublicationDate"),
        "publication_year": record.get("pubYear") or journal_info.get("yearOfPublication"),
        "cited_by_count": citations,
        "retraction_status": "retracted" if retracted else "not_flagged",
        "retraction_evidence": [item for item in corrections if isinstance(item, dict)
                                and "retract" in str(item.get("type") or "").casefold()],
        "status": "ok",
    }


def fetch_metadata(pmcids: list[str]) -> dict[str, dict]:
    """Fetch core records using the established PMC retry/rate-limit client."""
    result = {}
    size = config.VP_EPMC_LICENSE_BATCH
    ids = list(dict.fromkeys(pmcids))
    for offset in range(0, len(ids), size):
        chunk = ids[offset:offset + size]
        try:
            response = pmc._post(pmc.EPMC_SEARCH_POST, {
                "query": "PMCID:(" + " OR ".join(chunk) + ")",
                "resultType": "core", "format": "json", "pageSize": "1000",
            })
            records = response.json().get("resultList", {}).get("result", [])
            wanted = set(chunk)
            for record in records:
                if record.get("pmcid") in wanted:
                    result[record["pmcid"]] = normalize_record(record)
            for key in chunk:
                result.setdefault(key, {"source": "europe_pmc_core", "status": "missing_record"})
        except (pmc.PmcError, ValueError, AttributeError, TypeError) as exc:
            logger.warning("Europe PMC ranking metadata unavailable: %s", exc)
            for key in chunk:
                result[key] = {"source": "europe_pmc_core", "status": "fetch_failed"}
    return result


def enrich_shortlist(conn, rows: list[dict], *, persist: bool = True) -> None:
    """Attach cached/fresh metadata to rows; refresh positive records weekly.

    Missing/failed records are retried next time. Stale positive metadata is
    retained on transient errors (with the refresh failure explicitly exposed).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.VP_RANK_METADATA_TTL_DAYS)
    pending = []
    cached = {}
    for row in rows:
        stored = conn.execute("SELECT metadata_json,fetched_at FROM article_source_metadata WHERE pmcid=?",
                              (row["pmcid"],)).fetchone()
        metadata = db.from_json(stored["metadata_json"], {}) if stored else {}
        cached[row["pmcid"]] = metadata
        try:
            fresh = stored is not None and datetime.fromisoformat(stored["fetched_at"]) >= cutoff
        except (ValueError, TypeError):
            fresh = False
        if fresh:
            row["source_metadata"] = metadata
        else:
            pending.append(row["pmcid"])
    fetched = fetch_metadata(pending) if pending else {}
    for row in rows:
        key = row["pmcid"]
        if key not in fetched:
            continue
        metadata = fetched[key]
        if metadata.get("status") != "ok" and cached[key]:
            metadata = {**cached[key], "refresh_status": metadata["status"]}
        row["source_metadata"] = metadata
        if persist and metadata.get("status") == "ok" and not metadata.get("refresh_status"):
            conn.execute("INSERT INTO article_source_metadata(pmcid,source,metadata_json,fetched_at) "
                         "VALUES(?,?,?,?) ON CONFLICT(pmcid) DO UPDATE SET "
                         "source=excluded.source,metadata_json=excluded.metadata_json,fetched_at=excluded.fetched_at",
                         (key, "europe_pmc_core", db.to_json(metadata), metadata["fetched_at"]))
    if persist:
        conn.commit()


def attach_cached(conn, rows: list[dict]) -> None:
    """Keep known retractions active even outside this run's fetch shortlist."""
    for row in rows:
        stored = conn.execute("SELECT metadata_json FROM article_source_metadata WHERE pmcid=?",
                              (row["pmcid"],)).fetchone()
        if stored:
            row["source_metadata"] = db.from_json(stored["metadata_json"], {}) or {}


def quality_signal(metadata: dict | None, *, today: date | None = None) -> dict:
    """Secondary bounded signals; no unsupported proxy for journal quality."""
    metadata = metadata or {}
    now = today or date.today()
    age = None
    raw_date = metadata.get("first_publication_date")
    try:
        published = date.fromisoformat(str(raw_date))
        if published <= now:
            age = max(1.0, (now - published).days / 365.25)
    except ValueError:
        try:
            year = int(metadata["publication_year"])
            if 1800 <= year <= now.year:
                age = max(1.0, now.year - year + 0.5)
        except (KeyError, TypeError, ValueError):
            pass
    count = metadata.get("cited_by_count")
    annualized = count / age if isinstance(count, (int, float)) and age else None
    # A million citations cannot overwhelm an actual pair-specific image.
    citation_score = min(2.0, math.log1p(annualized) / 3.0) if annualized is not None else 0.0
    try:
        preferences = json.loads(config.VP_JOURNAL_PREFERENCES)
        preference = max((float(preferences.get(issn, 0)) for issn in
                          (metadata.get("journal_issn"), metadata.get("journal_essn"))), default=0)
        preference = max(0.0, min(1.0, preference))
    except (ValueError, TypeError, AttributeError):
        preference = 0.0
    types = metadata.get("publication_types") or []
    type_score = 0.25 if any(str(value).casefold() == "review" for value in types) else 0.0
    return {"score": citation_score + preference + type_score,
            "parts": {"citation_age_adjusted": citation_score, "journal_preference": preference,
                      "article_type": type_score},
            "citations_per_publication_year": annualized, "publication_age_years": age,
            "retracted": metadata.get("retraction_status") == "retracted",
            "metadata_status": metadata.get("refresh_status") or metadata.get("status", "not_enriched"),
            "source": metadata.get("source"),
            "limitation": "Europe PMC citation coverage; age-adjusted, not field-normalized; no journal impact factor"}
