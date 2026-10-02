"""Europe PMC core-record normalization for ``article_source_metadata``.

Unknown metadata is kept unknown. This module does not make or change
licensing decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re

# Article tiers for ranking image sources toward a disease overview; lower is
# better. Ranking only — no tier is excluded, so a case report still fills a
# pair when nothing better shows the finding.
TIER_REVIEW, TIER_SERIES, TIER_CASE, TIER_ATYPICAL = 0, 1, 2, 3
TIER_LABELS = {
    TIER_REVIEW: "review", TIER_SERIES: "series_or_study",
    TIER_CASE: "case_report", TIER_ATYPICAL: "atypical",
}
_REVIEW_TYPE = re.compile(r"review|meta-analysis|guideline|consensus", re.I)
_CASE_TYPE = re.compile(r"case", re.I)
_SERIES_TEXT = re.compile(
    r"case series|series of \d+|\b(?:[3-9]|\d{2,})\s+(?:cases|patients)\b"
    r"|\b(?:three|four|five|six|seven|eight|nine|ten)\s+(?:cases|patients)\b", re.I,
)
# Drug reactions, coincidences and rarities: not how the disease usually looks.
_ATYPICAL_TEXT = re.compile(
    r"\binduced\b|\bparadoxical|\brare\b|\bunusual|\batypical|\buncommon"
    r"|\bmimick?|\bmasquerad|\btriggered\b|drug reaction|\bvaccin"
    r"|\bcoexist|\bconcomitant|\bfollowing (?:treatment|therapy)", re.I,
)
# A single case built around one drug (``-mab``/``-nib``/``-cept``/``-kinra``
# names a biologic or small molecule) is a treatment story; trials and reviews
# of a drug still picture typical baseline disease, so only cases demote.
_TREATMENT_TEXT = re.compile(
    r"\btreated with|\bin the treatment of|\btreatment with|\btherapeutic"
    r"|\b\w+(?:mab|nib|kinra|(?<!con)cept)\b", re.I,
)
_CASE_TITLE = re.compile(r"\bcase\b|\ba patient\b", re.I)


def article_tier(publication_types, title: str | None = "", abstract: str | None = "") -> int:
    """Rank an image source: review < case series/original study < case report < atypical.

    Atypical wording in the title (drug-induced, paradoxical, rare, mimics,
    …) demotes any article type to last, as does a case report or case
    series about a treatment (treated with, a named biologic). A case
    signal (case type, or "case"/"a patient" in the title) outranks a review
    type, since "a case report and review of the literature" is a case
    report. A case whose title or abstract reports a series of three or
    more counts as a series.
    """
    title = title or ""
    if _ATYPICAL_TEXT.search(title):
        return TIER_ATYPICAL
    types = [str(t) for t in publication_types or []]
    if any(_CASE_TYPE.search(t) for t in types) or _CASE_TITLE.search(title):
        if _TREATMENT_TEXT.search(title):
            return TIER_ATYPICAL
        return TIER_SERIES if _SERIES_TEXT.search(f"{title} {abstract or ''}") else TIER_CASE
    if any(_REVIEW_TYPE.search(t) for t in types) or re.search(r"\breview\b", title, re.I):
        return TIER_REVIEW
    return TIER_SERIES


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
