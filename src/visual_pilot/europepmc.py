"""Europe PMC figure-caption search (figure-first discovery).

Europe PMC indexes figure captions under the ``FIG:`` field, so one query per
(disease, finding) pair returns articles that contain a figure whose caption
names the finding. ``resultType=core`` carries license, publication types and
bibliographic fields, so no second metadata call is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlencode

from . import pair_terms, pmc

SEARCH_URL = f"{pmc.EPMC_REST}/search"
# Europe PMC has brief outages (5xx bursts, read timeouts); 8 retries back off
# for ~47 s in total instead of ~15 s, so one blip does not drop a disease's search.
SEARCH_RETRIES = 8

# Europe PMC license values that can normalize to an allowed code; the exact
# gate is re-applied locally with pmc.license_allows.
_LICENSE_CLAUSE = '(LICENSE:"cc by" OR LICENSE:"cc0" OR LICENSE:"cc by-sa" OR LICENSE:"cc by-nd")'
_OPEN = f"OPEN_ACCESS:y AND IN_PMC:y AND {_LICENSE_CLAUSE}"

# Narrative reviews and case series. Systematic reviews and meta-analyses
# illustrate with forest plots and flow diagrams, rarely patients.
_OVERVIEW_TYPES = (
    '((PUB_TYPE:"review" OR PUB_TYPE:"review-article" OR TITLE:"case series" '
    'OR ABSTRACT:"case series") NOT PUB_TYPE:"systematic review" '
    'NOT PUB_TYPE:"meta-analysis" NOT TITLE:"systematic review")'
)
# Titles of articles that survey how a disease presents.
_OVERVIEW_TITLE = "(" + " OR ".join(
    f"TITLE:{t}" for t in (
        "manifestation*", '"clinical features"', '"clinical presentation"',
        '"clinical spectrum"', '"clinical aspects"', "phenotyp*", "diagnos*",
        "imaging", "overview", "update", "dermoscop*", "atlas", '"case series"',
    )
) + ")"

# Notices, not articles: never sources of patient images.
_EXCLUDED_PUB_TYPES = frozenset({
    "retraction of publication", "retracted publication", "published erratum",
    "correction", "expression of concern",
})


@dataclass
class Hit:
    pmcid: str
    pmid: str | None
    doi: str | None
    title: str
    journal: str | None
    year: int | None
    pub_types: list[str]
    license_raw: str | None
    license_code: str
    abstract: str = ""
    raw: dict = field(default_factory=dict, repr=False)


def _quote(term: str) -> str:
    return '"' + term.replace('"', " ").strip() + '"'


def _words_clause(term: str, field: str = "FIG") -> str | None:
    words = pair_terms.content_words(term)
    if len(words) < 2:
        return None
    parts = [f"{w}*" if len(w) >= 4 else w for w in words]
    return f"{field}:(" + " AND ".join(parts) + ")"


def _term_clauses(terms: list[str], mode: str, fields: tuple[str, ...]) -> list[str]:
    out = []
    for term in terms:
        if not term.strip():
            continue
        for field in fields:
            clause = _words_clause(term, field) if mode == "words" else f"{field}:{_quote(term)}"
            if clause:
                out.append(clause)
    return list(dict.fromkeys(out))


def _disease_clause(disease_terms: list[str], fields=("TITLE", "ABSTRACT")) -> str:
    return " OR ".join(f"{f}:{_quote(t)}" for t in disease_terms if t.strip() for f in fields)


def build_query(finding_terms: list[str], disease_terms: list[str], mode: str = "phrase",
                scope: str = "all") -> str:
    """Caption names the finding; title or abstract names the disease.

    ``phrase`` matches each term as an exact caption phrase; ``words`` only
    requires every content word of a multi-word term in the same caption
    ("erosions of the sacroiliac joint" for "sacroiliac erosion").
    ``scope="reviews"`` further requires a narrative review or case series
    whose title or abstract is about the finding ("dactylitis in psoriatic
    arthritis"), so broad articles that picture it are found first.
    """
    clauses = _term_clauses(finding_terms, mode, ("FIG",))
    if not clauses:
        return ""
    query = f"({' OR '.join(clauses)}) AND ({_disease_clause(disease_terms)})"
    if scope == "reviews":
        topic = _term_clauses(finding_terms, mode, ("TITLE", "ABSTRACT"))
        query += f" AND ({' OR '.join(topic)}) AND {_OVERVIEW_TYPES}"
    return f"{query} AND {_OPEN}"


def build_overview_query(disease_terms: list[str], caption_terms: list[str]) -> str:
    """Narrative reviews and case series surveying the disease's presentation.

    The disease must be in the title ("Psoriatic arthritis: clinical
    manifestations") and some caption must name any of its findings, which
    keeps out molecular and epidemiology reviews without patient images.
    """
    dis = _disease_clause(disease_terms, ("TITLE",))
    figs = _term_clauses(caption_terms, "phrase", ("FIG",))
    if not dis or not figs:
        return ""
    return (f"({dis}) AND {_OVERVIEW_TITLE} AND ({' OR '.join(figs)}) "
            f"AND {_OVERVIEW_TYPES} AND {_OPEN}")


_CASE_TYPES = (
    '(PUB_TYPE:"case reports" OR PUB_TYPE:"case-report" OR TITLE:case OR TITLE:"a patient")'
)


def build_disease_query(disease_terms: list[str], cases_only: bool = False) -> str:
    """Any article whose title names the disease, without a caption clause.

    For topics the caption-phrase passes leave nearly empty: their captions
    rarely repeat the vocabulary's phrasing, so the figures are gated locally
    on caption and citing text instead (``discover._store_article``).
    ``cases_only`` keeps case reports and series, which picture patients far
    more often than the disease's research papers and reviews.
    """
    dis = _disease_clause(disease_terms, ("TITLE",))
    if not dis:
        return ""
    cases = f" AND {_CASE_TYPES}" if cases_only else ""
    return f"({dis}){cases} AND {_OPEN}"


def _hit(rec: dict) -> Hit | None:
    pmcid = rec.get("pmcid")
    if not pmcid:
        return None
    types = [str(t) for t in (rec.get("pubTypeList") or {}).get("pubType", []) if t]
    raw = rec.get("license")
    year = rec.get("pubYear")
    return Hit(
        pmcid=pmcid,
        pmid=rec.get("pmid"),
        doi=rec.get("doi"),
        title=rec.get("title") or "",
        journal=(rec.get("journalInfo") or {}).get("journal", {}).get("title"),
        year=int(year) if str(year or "").isdigit() else None,
        pub_types=types,
        license_raw=raw,
        license_code=pmc.normalize_license(raw),
        abstract=rec.get("abstractText") or "",
        raw=rec,
    )


def eligible(hit: Hit) -> bool:
    if pmc.license_allows(hit.license_code) is None:
        return False
    return not any(t.strip().lower() in _EXCLUDED_PUB_TYPES for t in hit.pub_types)


def search(query: str, *, limit: int = 100, page_size: int = 100) -> tuple[int, list[Hit]]:
    """Return (hitCount, up to ``limit`` hits) in Europe PMC relevance order."""
    hits: list[Hit] = []
    cursor = "*"
    total = 0
    while len(hits) < limit:
        params = {
            "query": query, "format": "json", "resultType": "core",
            "pageSize": str(min(page_size, 1000)), "cursorMark": cursor,
        }
        data = pmc._request(f"{SEARCH_URL}?{urlencode(params)}", max_retries=SEARCH_RETRIES).json()
        total = int(data.get("hitCount") or 0)
        records = (data.get("resultList") or {}).get("result") or []
        for rec in records:
            hit = _hit(rec)
            if hit is not None:
                hits.append(hit)
        nxt = data.get("nextCursorMark")
        if not records or not nxt or nxt == cursor:
            break
        cursor = nxt
    return total, hits[:limit]
