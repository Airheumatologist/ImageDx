"""Stage 2: article selection from the turbopuffer PMC namespace.

Per disease, per synonym: title BM25 (top_k 200), page_content BM25
(top_k 300) and dense ANN (top_k 300), plus a bounded set of visual
finding/modality page_content BM25 queries, under the review filter
``Or(publication_type Contains "Review", article_type Eq "review-article")``
AND ``has_full_text = true``. Rows are chunk-level: each ranked list is
collapsed to per-pmcid best rank and all lists for a disease are fused with
plain RRF (k=60) into ``retrieval_score``.

Publication-type exclusions are applied in Python (``passes_type_filter``).
Batched Europe PMC licenses use a public PMC fallback for absent metadata;
every license-cleared article goes through prompt P1.

The license filter runs before the ``--limit`` cap: every type-passed
article is persisted as a candidate, then candidates are license-checked
in finding-lane priority order (``license_priority_order``) until each
disease accumulates ``limit`` license-passing articles. Undercovered finding
lanes can continue beyond that quota until usable candidate targets are met.
Verdicts and queue stop reasons are checkpointed incrementally.
"""

from __future__ import annotations

import json
import logging
import re
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from . import config, db, diseases, llm, pmc, timing, queue_state
from .prompts import P1

logger = logging.getLogger(__name__)

PILOT_KEYS = set(diseases.DISEASE_KEYS)

TITLE_TOP_K = 500
CONTENT_TOP_K = 750
DENSE_TOP_K = 750
VISUAL_QUERY_TOP_K = 300
MAX_EVIDENCE_PER_ARTICLE = 8
MAX_EVIDENCE_TEXT_CHARS = 1200
VISUAL_QUERY_RRF_WEIGHT = 12.0
RRF_K = 60
METADATA_BATCH_SIZE = 100
# turbopuffer rejects a multi_query needing more than 16 concurrent
# per-namespace permits ("requires N permits, max is 16"), so every
# multi_query batch is capped at this many subqueries.
MULTI_QUERY_BATCH = 16
# License outcomes are applied in chunks of VP_EPMC_LICENSE_BATCH *
# VP_EPMC_CONCURRENCY (one Europe PMC batch per in-flight worker) so
# per-disease license-passing quotas can stop the pass between chunks
# (overshoot is at most one chunk per disease). Computed at use time.
def _license_chunk() -> int:
    return config.VP_EPMC_LICENSE_BATCH * config.VP_EPMC_CONCURRENCY
DEFAULT_CAP = 150
ABSTRACT_MAX_CHARS = 4000

# Discovery returns only the attributes needed to type-filter and rank hits.
# Passage text is requested only for page-content jobs; title, abstract and
# citation metadata are fetched later for the shortlisted unique PMCIDs.
DISCOVERY_ATTRIBUTES = ["pmcid", "publication_type", "article_type"]
EVIDENCE_ATTRIBUTES = [
    *DISCOVERY_ATTRIBUTES, "page_content", "section_title", "section_type",
]
METADATA_ATTRIBUTES = [
    "pmcid", "pmid", "doi", "title", "abstract", "journal", "year", "country",
    "publication_type", "article_type",
]
_PER_PMCID_LIMIT = {"per": {"attributes": ["pmcid"], "limit": 1}}

_CATEGORY_MODALITY = {
    "skin": "clinical photograph",
    "mucosa": "clinical photograph",
    "nail": "clinical photograph",
    "clinical_msk": "clinical photograph",
    "capillaroscopy": "capillaroscopy image",
    "histology": "histology micrograph",
    "radiology_xray": "radiograph",
    "ct": "CT scan",
    "mri": "MRI",
    "us": "ultrasound image",
    "echo": "echocardiogram",
    "eye": "ophthalmic image",
}
_VISUAL_SECTION_CUE = re.compile(
    r"\b(clinical|physical examination|cutaneous|dermatolog|imaging|radiolog|"
    r"mri|computed tomography|histolog|patholog|capillaroscop|photograph|"
    r"ophthalm|ocular|eye)\w*\b",
    re.I,
)
_VISUAL_PASSAGE_CUE = re.compile(
    r"\b(photo(?:graph)?s?|images?|figure\s+\d|radiograph|x[ -]?ray|"
    r"mri|magnetic resonance|ct scan|computed tomography|ultrasound|"
    r"histolog|biopsy|histopatholog|micrograph|capillaroscop|rash|papules?|"
    r"plaques?|erythema|ulcer|erosion|lesion|sacroiliitis|bone marrow edema|"
    r"ophthalm|ocular|uveitis|iritis|dry eye|conjunctivitis|"
    r"keratoconjunctivitis|meibomian|slit[ -]?lamp|fundoscop\w*|fundus|"
    r"optical coherence tomography|fluorescein angiograph\w*)\b",
    re.I,
)

# Review articles show up under either field (verified live: publication_type
# is a list of journal labels, article_type a JATS string).
_REVIEW_OR = [
    "Or",
    [["publication_type", "Contains", "Review"], ["article_type", "Eq", "review-article"]],
]
_REVIEW_PUBTYPE = ["publication_type", "Contains", "Review"]
_REVIEW_ARTICLE_TYPE = ["article_type", "Eq", "review-article"]

# Whether the server accepted the nested Or filter (auto-detected once).
_or_filter_supported: bool | None = None
_BILLING_LOCK = Lock()

# §2/history: publication labels whose *substring* disqualifies an article.
_EXCLUDED_LABEL_PARTS = (
    "case",
    "meta-analys",
    "systematic review",
    "clinical trial",
    "randomized",
    "randomised",
    "letter",
    "editorial",
    "comment",
    "protocol",
    "erratum",
    "correction",
    "retraction",
)
_EXCLUDED_ARTICLE_TYPES = frozenset(
    {
        "case-report",
        "letter",
        "editorial",
        "correction",
        "retraction",
        "article-commentary",
    }
)

_LICENSE_PASSED = frozenset(
    {"license_ok", "relevant", "irrelevant", "parsed", "parse_error"}
)
_RELEVANT_STATUSES = frozenset({"relevant", "parsed", "parse_error"})


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def passes_type_filter(
    publication_types: list[str] | None, article_type: str | None = None
) -> bool:
    """Keep narrative-review-like articles; drop case/trial/letter/etc. tags."""
    labels = [
        str(t).strip().lower()
        for t in (publication_types or [])
        if str(t).strip()
    ]
    atype = str(article_type or "").strip().lower()
    if atype in _EXCLUDED_ARTICLE_TYPES:
        return False
    if atype != "review-article" and not any("review" in t for t in labels):
        return False
    for label in labels:
        if any(part in label for part in _EXCLUDED_LABEL_PARTS):
            return False
    return True


def shortlist_articles(
    articles: dict[str, dict],
    limit: int | None,
    per_finding_quota: int = config.VP_MANIFESTATION_QUOTA,
) -> tuple[list[tuple[str, dict]], int, list[dict]]:
    """Type-filter candidates, reserve manifestation lanes, then fill by RRF.

    Each finding lane gets one distinct PMC article per round before any lane
    gets its next, up to ``per_finding_quota`` articles. Shared PMCIDs count
    once toward the global cap; an already-selected shared candidate does not
    consume another lane's quota, allowing that lane to reserve its next hit.
    ``limit=None`` returns every type-passed article in global RRF order.
    """
    ranked = sorted(articles.items(), key=lambda item: item[1]["score"], reverse=True)
    type_passed: list[tuple[str, dict]] = []
    for pmcid, info in ranked:
        attrs = info.get("attrs") or {}
        types = attrs.get("publication_type") or []
        if isinstance(types, str):
            types = [types]
        if passes_type_filter(types, attrs.get("article_type")):
            type_passed.append((pmcid, info))

    if limit is None:
        return type_passed, len(type_passed), [
            {**candidate, "pmcid": pmcid}
            for pmcid, info in type_passed
            for candidate in info.get("manifestation_candidates", [])
        ]
    cap = max(0, int(limit))
    if cap == 0:
        return [], len(type_passed), []

    eligible_ids = {pmcid for pmcid, _ in type_passed}
    lane_candidates: dict[str, dict[str, dict]] = {}
    for pmcid, info in type_passed:
        for candidate in info.get("manifestation_candidates", []):
            finding_key = str(candidate.get("finding_key") or "")
            if not finding_key or pmcid not in eligible_ids:
                continue
            lane = lane_candidates.setdefault(finding_key, {})
            prior = lane.get(pmcid)
            if prior is None or int(candidate.get("best_rank") or 0) < int(
                prior.get("best_rank") or 0
            ):
                lane[pmcid] = {**candidate, "pmcid": pmcid}

    lane_order = list(lane_candidates)
    lane_items = {
        finding_key: sorted(
            candidates.values(),
            key=lambda item: (
                int(item.get("best_rank") or 0),
                -float(item.get("retrieval_score") or 0),
                str(item["pmcid"]),
            ),
        )
        for finding_key, candidates in lane_candidates.items()
    }
    reserved: list[str] = []
    reserved_set: set[str] = set()
    lane_counts = {finding_key: 0 for finding_key in lane_order}
    lane_positions = {finding_key: 0 for finding_key in lane_order}
    quota = max(0, int(per_finding_quota))
    while len(reserved) < cap and quota:
        made_progress = False
        for finding_key in lane_order:
            if len(reserved) >= cap:
                break
            if lane_counts[finding_key] >= quota:
                continue
            candidates = lane_items[finding_key]
            position = lane_positions[finding_key]
            while position < len(candidates) and candidates[position]["pmcid"] in reserved_set:
                position += 1
            lane_positions[finding_key] = position
            if position >= len(candidates):
                continue
            pmcid = candidates[position]["pmcid"]
            lane_positions[finding_key] += 1
            reserved.append(pmcid)
            reserved_set.add(pmcid)
            lane_counts[finding_key] += 1
            made_progress = True
        if not made_progress:
            break

    selected_ids = list(reserved)
    for pmcid, _info in type_passed:
        if len(selected_ids) >= cap:
            break
        if pmcid not in reserved_set:
            selected_ids.append(pmcid)
            reserved_set.add(pmcid)
    selected_set = set(selected_ids)
    selected = [(pmcid, info) for pmcid, info in type_passed if pmcid in selected_set]
    selected_candidates = [
        {**candidate, "pmcid": pmcid}
        for pmcid, info in selected
        for candidate in info.get("manifestation_candidates", [])
    ]
    return selected, len(type_passed), selected_candidates


def license_priority_order(type_passed: list[tuple[str, dict]]) -> list[str]:
    """License-check order for a disease's type-passed candidates.

    Finding lanes interleave round-robin (best rank first within a lane) so
    every undercovered manifestation lands inside the licensed window before
    the remaining articles follow in global RRF order.
    """
    lane_candidates: dict[str, dict[str, dict]] = {}
    for pmcid, info in type_passed:
        for candidate in info.get("manifestation_candidates", []):
            finding_key = str(candidate.get("finding_key") or "")
            if not finding_key:
                continue
            lane = lane_candidates.setdefault(finding_key, {})
            prior = lane.get(pmcid)
            if prior is None or int(candidate.get("best_rank") or 0) < int(
                prior.get("best_rank") or 0
            ):
                lane[pmcid] = {**candidate, "pmcid": pmcid}
    lane_items = {
        finding_key: sorted(
            candidates.values(),
            key=lambda item: (
                int(item.get("best_rank") or 0),
                -float(item.get("retrieval_score") or 0),
                str(item["pmcid"]),
            ),
        )
        for finding_key, candidates in lane_candidates.items()
    }
    order: list[str] = []
    seen: set[str] = set()
    lane_positions = {finding_key: 0 for finding_key in lane_items}
    while True:
        made_progress = False
        for finding_key, candidates in lane_items.items():
            position = lane_positions[finding_key]
            while position < len(candidates) and candidates[position]["pmcid"] in seen:
                position += 1
            lane_positions[finding_key] = position
            if position >= len(candidates):
                continue
            pmcid = candidates[position]["pmcid"]
            lane_positions[finding_key] += 1
            order.append(pmcid)
            seen.add(pmcid)
            made_progress = True
        if not made_progress:
            break
    order.extend(pmcid for pmcid, _ in type_passed if pmcid not in seen)
    return order


def rrf_scores(
    ranked_lists: list[list[str]], k: int = RRF_K,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Weighted RRF per pmcid: each list contributes weight/(k + best_rank)."""
    scores: dict[str, float] = {}
    for list_index, pmcids in enumerate(ranked_lists):
        weight = weights[list_index] if weights and list_index < len(weights) else 1.0
        seen: set[str] = set()
        for rank, pmcid in enumerate(pmcids, start=1):
            if not pmcid or pmcid in seen:
                continue
            seen.add(pmcid)
            scores[pmcid] = scores.get(pmcid, 0.0) + weight / (k + rank)
    return scores


def _is_visual_passage(
    row: dict,
) -> bool:
    """Identify image-bearing passage evidence while leaving candidate recall broad."""
    passage = str(row.get("page_content") or "").strip()
    if not passage:
        return False
    section = " ".join(
        str(row.get(key) or "") for key in ("section_title", "section_type")
    )
    return bool(
        _VISUAL_SECTION_CUE.search(section) or _VISUAL_PASSAGE_CUE.search(passage)
    )


def _is_acronym(term: str) -> bool:
    letters = [c for c in term if c.isalpha()]
    return bool(letters) and len(term) <= 6 and all(c.isupper() for c in letters)


def synonym_matches(title: str, synonym: str) -> bool:
    """Word-boundary match; all-caps acronyms (SLE, JDM) match case-sensitively."""
    pattern = r"\b" + re.escape(synonym) + r"\b"
    flags = 0 if _is_acronym(synonym) else re.IGNORECASE
    return re.search(pattern, title, flags) is not None


def title_rule_diseases(title: str, diseases_data: dict[str, dict]) -> set[str]:
    """Diseases the title clearly covers per the §5 stage-2 title rule.

    Acronyms like "SLE" match only as exact-case tokens. A title that matches
    "lupus" while also saying "drug-induced" or "neonatal" does NOT count for
    sle — those go to the P1 relevance check instead.
    """
    matched: set[str] = set()
    for key, info in diseases_data.items():
        for synonym in info.get("synonyms", []):
            if not synonym_matches(title, synonym):
                continue
            if (
                key == "sle"
                and synonym.lower() == "lupus"
                and re.search(r"drug[\s-]*induced|neonatal", title, re.IGNORECASE)
            ):
                continue
            matched.add(key)
            break
    return matched


def visual_queries_for_disease(
    disease_key: str,
    diseases_data: dict[str, dict] | None = None,
    findings: list[dict] | None = None,
    max_queries: int | None = None,
    coverage_counts: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """Build deterministic, disease-specific finding/modality retrieval queries.

    Findings are selected round-robin across image-bearing categories so a
    large skin vocabulary cannot crowd out imaging or pathology. By default,
    every approved disease-specific finding is queried; ``max_queries`` or
    ``VP_VISUAL_QUERY_CAP`` can impose an explicit operational cap. Each result
    retains its query terms for downstream evidence and ranking.
    """
    if max_queries is None:
        configured_cap = config.VP_VISUAL_QUERY_CAP
        max_queries = configured_cap or None
    elif max_queries <= 0:
        # An explicit zero remains a way for callers to disable visual queries.
        return []
    if disease_key not in PILOT_KEYS:
        return []
    from . import pair_terms

    diseases_data = diseases_data or diseases.load_diseases()
    findings = findings if findings is not None else diseases.load_findings_vocab()
    disease = diseases_data[disease_key]
    eligible = [
        item for item in findings
        if disease_key in item.get("disease_keys", [])
        and item.get("category") in _CATEGORY_MODALITY
        and item.get("approved", True)
    ]
    by_category: dict[str, list[dict]] = {}
    for item in eligible:
        by_category.setdefault(item["category"], []).append(item)
    coverage_counts = coverage_counts or {}
    for bucket in by_category.values():
        bucket.sort(
            key=lambda item: (
                int(coverage_counts.get(str(item.get("finding_key") or ""), 0)),
                str(item.get("finding_key") or ""),
            )
        )
    # Stable category order follows the image modalities in the seed schema.
    categories = list(_CATEGORY_MODALITY)
    selected: list[dict] = []
    while max_queries is None or len(selected) < max_queries:
        added = False
        for category in categories:
            bucket = by_category.get(category, [])
            if len(bucket) > sum(x["category"] == category for x in selected):
                selected.append(bucket[sum(x["category"] == category for x in selected)])
                added = True
                if max_queries is not None and len(selected) >= max_queries:
                    break
        if not added:
            break

    subtypes = disease.get("subtypes", [])
    out = []
    seen_queries: set[str] = set()
    for item in selected:
        finding = str(item.get("label") or item.get("finding_key") or "")
        modality = _CATEGORY_MODALITY[item["category"]]
        pair_synonyms = pair_terms.finding_synonyms(disease_key, item)
        finding_text = " ".join(
            [finding, str(item.get("finding_key") or ""), *pair_synonyms]
        ).lower()
        subtype = ""
        for candidate in subtypes:
            terms = [candidate.get("key", ""), candidate.get("label", "")]
            if any(
                term and re.search(r"\b" + re.escape(term.lower()) + r"\b", finding_text)
                for term in terms
            ):
                subtype = candidate["label"]
                break
        # Axial disease imaging findings split naturally by modality: MRI is
        # the non-radiographic/radiographic assessment query, while plain
        # radiographs specifically retrieve radiographic axSpA.
        if not subtype and disease_key == "as":
            subtype_key = (
                "r_axspa" if item["category"] == "radiology_xray" else
                "nr_axspa" if item["category"] == "mri" else ""
            )
            subtype = next(
                (s["label"] for s in subtypes if s.get("key") == subtype_key), ""
            )
        # Keep BM25 queries focused: the label and disease establish the
        # target; add at most two synonyms with concepts not already present
        # in those terms. Long lists of near-duplicates dilute useful tokens.
        base_tokens = set(re.findall(
            r"[a-z0-9]+", f"{disease['name']} {subtype} {finding}".casefold()
        ))
        distinct_synonyms = []
        for term in pair_synonyms:
            term = str(term).strip()
            tokens = set(re.findall(r"[a-z0-9]+", term.casefold()))
            if not term or not tokens or len(tokens & base_tokens) / len(tokens) > 0.5:
                continue
            if len(term) > 80:
                term = term[:80].rsplit(" ", 1)[0]
            distinct_synonyms.append(term)
            if len(distinct_synonyms) == 2:
                break
        query = " ".join(
            part for part in (
                disease["name"], subtype, finding,
                " ".join(distinct_synonyms), modality,
            ) if part
        )
        if query not in seen_queries:
            seen_queries.add(query)
            out.append({
                "query": query,
                "disease_key": disease_key,
                "finding_key": str(item.get("finding_key") or ""),
                "finding": finding,
                "modality": modality,
                "category": item["category"],
            })
        # W7: under-target findings get extra queries, one per synonym not
        # already used in this finding's base query.
        if (
            coverage_counts.get(str(item.get("finding_key") or ""), 0)
            >= config.VP_FINDING_IMAGE_TARGET
        ):
            continue
        used = {str(term).strip() for term in distinct_synonyms}
        used.add(finding)
        added = 0
        for term in pair_synonyms:
            if added >= config.VP_TARGETED_SYNONYM_QUERIES:
                break
            term = str(term).strip()
            if not term or term in used or term.casefold() == finding.casefold():
                continue
            if len(term) > 80:
                term = term[:80].rsplit(" ", 1)[0]
            query = " ".join(
                part for part in (disease["name"], subtype, term, modality) if part
            )
            used.add(term)
            if query in seen_queries:
                continue
            seen_queries.add(query)
            out.append({
                "query": query,
                "disease_key": disease_key,
                "finding_key": str(item.get("finding_key") or ""),
                "finding": finding,
                "modality": modality,
                "category": item["category"],
                "targeted": True,
            })
            added += 1
    return out


def _visual_findings_from_db(conn) -> list[dict]:
    """Load approved vocabulary rows, including administrator-approved additions."""
    rows = conn.execute(
        "SELECT finding_key, disease_keys_json, label, synonyms_json, category, approved "
        "FROM findings_vocab WHERE approved = 1"
    )
    return [
        {
            "finding_key": row["finding_key"],
            "disease_keys": db.from_json(row["disease_keys_json"], []),
            "label": row["label"],
            "synonyms": db.from_json(row["synonyms_json"], []),
            "category": row["category"],
            "approved": bool(row["approved"]),
        }
        for row in rows
    ]


def _stored_panel_counts(conn, disease_key: str) -> dict[str, int]:
    """Distinct published images per approved finding for coverage-aware queries."""
    from . import manifestation_queue

    return manifestation_queue.published_coverage(conn, disease_key)


# ---------------------------------------------------------------------------
# turbopuffer access
# ---------------------------------------------------------------------------
def _make_retriever():
    """Construct the pilot's PMC namespace handle + embedding client."""
    from .retrieval import VisualRetriever

    return VisualRetriever()


def _review_filters():
    return ["And", [["has_full_text", "Eq", True], _REVIEW_OR]]


def _split_review_filters():
    base = ["has_full_text", "Eq", True]
    return [
        ["And", [base, _REVIEW_PUBTYPE]],
        ["And", [base, _REVIEW_ARTICLE_TYPE]],
    ]


def _record_billing(counters: dict | None, result, query_count: int = 1) -> None:
    """Accumulate Turbopuffer's logical-byte counters when the response has them."""
    if counters is None:
        return
    billing = getattr(result, "billing", None)
    if billing is None and isinstance(result, dict):
        billing = result.get("billing")
    if billing is None:
        return
    def value(name: str) -> int:
        raw = billing.get(name, 0) if isinstance(billing, dict) else getattr(billing, name, 0)
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0
    with _BILLING_LOCK:
        counters["requests"] = counters.get("requests", 0) + 1
        counters["queries"] = counters.get("queries", 0) + query_count
        counters["billable_logical_bytes_queried"] = counters.get(
            "billable_logical_bytes_queried", 0
        ) + value("billable_logical_bytes_queried")
        counters["billable_logical_bytes_returned"] = counters.get(
            "billable_logical_bytes_returned", 0
        ) + value("billable_logical_bytes_returned")


def _ns_query(ns, rank_by, filters, top_k, counters=None, attributes=None) -> list[dict]:
    result = ns.query(
        rank_by=rank_by,
        filters=filters,
        limit={"total": top_k, **_PER_PMCID_LIMIT},
        include_attributes=attributes or DISCOVERY_ATTRIBUTES,
    )
    _record_billing(counters, result)
    return [dict(r) for r in getattr(result, "rows", [])]


def _merge_ranked(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    """Two lists for the same bucket -> one ranked list (dedup by best rank)."""
    best: dict[str, tuple[int, dict]] = {}
    for rows in (rows_a, rows_b):
        for rank, row in enumerate(rows):
            key = str(row.get("id") or row.get("chunk_id") or row.get("pmcid") or "")
            if key not in best or rank < best[key][0]:
                best[key] = (rank, row)
    return [row for _, row in sorted(best.values(), key=lambda t: t[0])]


def _rank_query(ns, rank_by, top_k, counters=None) -> list[dict]:
    """One ranked list for one bucket, under the review + full-text filter."""
    global _or_filter_supported
    if _or_filter_supported is not False:
        try:
            attrs = EVIDENCE_ATTRIBUTES if rank_by[0] == "page_content" else DISCOVERY_ATTRIBUTES
            rows = _ns_query(ns, rank_by, _review_filters(), top_k, counters, attrs)
            _or_filter_supported = True
            return rows
        except Exception as exc:  # noqa: BLE001 - detect filter support once
            logger.warning(
                "Or review filter rejected (%s); falling back to two queries", exc
            )
            _or_filter_supported = False
    filters_a, filters_b = _split_review_filters()
    attrs = EVIDENCE_ATTRIBUTES if rank_by[0] == "page_content" else DISCOVERY_ATTRIBUTES
    rows_a = _ns_query(ns, rank_by, filters_a, top_k, counters, attrs)
    rows_b = _ns_query(ns, rank_by, filters_b, top_k, counters, attrs)
    return _merge_ranked(rows_a, rows_b)


def _rank_jobs(ns, jobs, contexts, counters=None) -> list[list[dict]]:
    """Run ordered retrieval jobs via multi_query, chunked under the permit cap.

    turbopuffer allows at most MULTI_QUERY_BATCH subqueries per multi_query
    (per-namespace concurrency budget). Jobs run in sequential 16-wide
    chunks — a chunk already saturates the budget, so wider parallelism
    would only contend for permits. A failed chunk falls back to sequential
    ``query`` calls for just that chunk (old SDKs hit this path per chunk).
    """
    multi_query = getattr(ns, "multi_query", None)

    def _run(job):
        rank_by, top_k = job
        return _rank_query(ns, rank_by, top_k, counters)

    if not callable(multi_query) or not jobs:
        return [_run(job) for job in jobs]

    all_rows: list[list[dict] | None] = [None] * len(jobs)
    for start in range(0, len(jobs), MULTI_QUERY_BATCH):
        chunk = jobs[start : start + MULTI_QUERY_BATCH]
        queries = []
        for rank_by, top_k in chunk:
            attrs = EVIDENCE_ATTRIBUTES if rank_by[0] == "page_content" else DISCOVERY_ATTRIBUTES
            queries.append({
                "rank_by": rank_by,
                "filters": _review_filters(),
                "limit": {"total": top_k, **_PER_PMCID_LIMIT},
                "include_attributes": attrs,
            })
        try:
            result = multi_query(queries=queries)
            _record_billing(counters, result, query_count=len(queries))
            results = list(getattr(result, "results", []) or [])
            if len(results) != len(chunk):
                raise ValueError(
                    f"multi_query returned {len(results)} results for {len(chunk)} jobs"
                )
            for offset, item in enumerate(results):
                all_rows[start + offset] = [
                    dict(row) for row in (getattr(item, "rows", None) or [])
                ]
        except Exception as exc:  # noqa: BLE001 - per-chunk compatibility fallback
            logger.info(
                "Turbopuffer multi_query unavailable (%s); using query fallback", exc
            )
            for offset, job in enumerate(chunk):
                all_rows[start + offset] = _run(job)
    return [rows if rows is not None else [] for rows in all_rows]


def _embed_synonyms(synonyms: list[str], embed_fn, embed_many_fn) -> list:
    """Embeddings aligned with ``synonyms``; ``None`` skips dense ANN.

    ``embed_many_fn`` issues one batched request for all synonyms. When it is
    absent or raises, fall back to the per-item ``embed_fn`` path — vectors
    are identical either way and dense embedding stays best-effort.
    """
    if embed_many_fn is not None:
        try:
            vectors = list(embed_many_fn(list(synonyms)))
            return (vectors + [None] * len(synonyms))[: len(synonyms)]
        except Exception as exc:  # noqa: BLE001 - dense is best-effort
            logger.warning(
                "batched embedding failed (%s); falling back to per-item", exc
            )
    if embed_fn is None:
        return [None] * len(synonyms)
    embeddings = []
    for synonym in synonyms:
        embedding = None
        try:
            embedding = embed_fn(synonym)
        except Exception as exc:  # noqa: BLE001 - dense is best-effort
            logger.warning("embedding failed for %r: %s", synonym, exc)
        embeddings.append(embedding)
    return embeddings


def retrieve_for_disease(
    ns,
    embed_fn,
    synonyms: list[str],
    visual_queries: list[dict[str, str]] | None = None,
    disease_key: str | None = None,
    embed_many_fn=None,
    billing_counters: dict | None = None,
) -> dict[str, dict]:
    """Fuse ranked lists and preserve the best passage evidence per PMC article.

    The independent turbopuffer queries (per-synonym buckets + per-visual
    lookups) run through chunked multi-query requests (MULTI_QUERY_BATCH
    subqueries each), with a sequential ``query`` fallback per failed chunk.
    Results are replayed in job order so RRF fusion, first-seen attributes,
    and evidence selection stay deterministic.
    """
    synonyms = list(synonyms or [])
    embeddings = _embed_synonyms(synonyms, embed_fn, embed_many_fn)
    for synonym, embedding in zip(synonyms, embeddings):
        if embedding is None:
            logger.info("no embedding for %r; dense ANN skipped", synonym)
    jobs, contexts = _job_specs(
        synonyms, embeddings, visual_queries, disease_key=disease_key
    )
    all_rows = _rank_jobs(ns, jobs, contexts, billing_counters)
    return _fuse_rows(jobs, contexts, all_rows)


def _job_specs(
    synonyms: list[str],
    embeddings: list,
    visual_queries: list[dict[str, str]] | None,
    *,
    disease_key: str | None = None,
) -> tuple[list[tuple[list, int]], list[dict]]:
    """Build the query jobs in issue order, keeping the context each job's
    rows need for the downstream merge (query kind, evidence metadata).
    ``disease_key`` is recorded on every context so evidence and
    manifestation candidates carry explicit disease provenance."""
    jobs: list[tuple[list, int]] = []
    contexts: list[dict] = []
    for synonym, embedding in zip(synonyms, embeddings):
        buckets = [
            (["title", "BM25", synonym], TITLE_TOP_K),
            (["page_content", "BM25", synonym], CONTENT_TOP_K),
        ]
        if embedding is not None:
            buckets.append((["vector", "ANN", embedding], DENSE_TOP_K))
        for rank_by, top_k in buckets:
            jobs.append((rank_by, top_k))
            contexts.append(
                {
                    "query": synonym,
                    "query_kind": "synonym",
                    "disease_key": disease_key,
                    "modality": "",
                    "finding": "",
                }
            )
    for spec in visual_queries or []:
        query = str(spec.get("query") or "").strip()
        if not query:
            continue
        jobs.append((["page_content", "BM25", query], VISUAL_QUERY_TOP_K))
        contexts.append(
            {
                "query": query,
                "query_kind": "visual",
                "disease_key": spec.get("disease_key") or disease_key,
                "finding_key": str(spec.get("finding_key") or ""),
                "modality": str(spec.get("modality") or ""),
                "finding": str(spec.get("finding") or ""),
            }
        )
    return jobs, contexts


def _fuse_rows(
    jobs: list[tuple[list, int]],
    contexts: list[dict],
    all_rows: list[list[dict]],
) -> dict[str, dict]:
    """Replays rows in job order so RRF fusion, first-seen attributes, and
    evidence selection stay deterministic."""
    ranked_lists: list[list[str]] = []
    ranked_weights: list[float] = []
    attrs: dict[str, dict] = {}
    evidence: dict[str, list[dict]] = {}
    manifestation_candidates: list[dict] = []
    for (rank_by, _top_k), context, rows in zip(jobs, contexts, all_rows):
        ranked_lists.append([str(r.get("pmcid") or "") for r in rows])
        ranked_weights.append(1.0)  # keep a broad recall path for every hit
        if context["query_kind"] == "visual":
            visual_rows = [row for row in rows if _is_visual_passage(row)]
            if visual_rows:
                # A qualified visual hit contributes this bonus plus its
                # broad path contribution, making its total weight 12x
                # without suppressing articles that a visual query finds
                # less directly.
                ranked_lists.append(
                    [str(row.get("pmcid") or "") for row in visual_rows]
                )
                ranked_weights.append(VISUAL_QUERY_RRF_WEIGHT - 1.0)
        for row in rows:
            pmcid = str(row.get("pmcid") or "")
            if pmcid and pmcid not in attrs:
                attrs[pmcid] = row
        if rank_by[0] != "page_content":
            continue
        for rank, row in enumerate(rows, start=1):
            pmcid = str(row.get("pmcid") or "")
            if (
                pmcid
                and context["query_kind"] == "visual"
                and context.get("finding_key")
            ):
                manifestation_candidates.append({
                    "disease_key": context.get("disease_key"),
                    "finding_key": context["finding_key"],
                    "pmcid": pmcid,
                    "query": context["query"],
                    "best_rank": rank,
                    "retrieval_score": 1.0 / (RRF_K + rank),
                })
            passage = str(row.get("page_content") or "").strip()
            if not pmcid or not passage or not _is_visual_passage(row):
                continue
            evidence.setdefault(pmcid, []).append({
                "query": context["query"],
                "query_kind": context["query_kind"],
                "disease_key": context.get("disease_key"),
                "finding_key": context.get("finding_key", ""),
                "text": passage[:MAX_EVIDENCE_TEXT_CHARS],
                "section": str(row.get("section_title") or ""),
                "section_type": str(row.get("section_type") or ""),
                "modality": context["modality"],
                "finding": context["finding"],
                "rank": rank,
                "score": 1.0 / (RRF_K + rank),
            })
    scores = rrf_scores(ranked_lists, weights=ranked_weights)
    manifestation_by_pmcid: dict[str, list[dict]] = {}
    for item in manifestation_candidates:
        manifestation_by_pmcid.setdefault(item["pmcid"], []).append(item)
    out = {}
    for pmcid, score in scores.items():
        out[pmcid] = {
            "score": score,
            "attrs": attrs.get(pmcid, {}),
            "matched_passages": _select_evidence(evidence.get(pmcid, [])),
            "manifestation_candidates": manifestation_by_pmcid.get(pmcid, []),
        }
    return out


def _evidence_identity(item: dict) -> tuple[str, str, str, str]:
    """Passage identity: pair provenance + section + text (legacy items
    without disease provenance keep an empty leading key)."""
    return (
        str(item.get("disease_key") or ""),
        str(item.get("finding_key") or ""),
        str(item.get("section") or ""),
        str(item.get("text") or ""),
    )


def _select_evidence(evidence: list[dict]) -> list[dict]:
    """Keep a bounded mix, reserving at least one slot for visual query hits."""
    best_by_passage: dict[tuple[str, str, str, str], dict] = {}
    for item in evidence:
        identity = _evidence_identity(item)
        if not identity[3]:
            continue
        prior = best_by_passage.get(identity)
        item_is_visual = item.get("query_kind") == "visual"
        prior_is_visual = prior and prior.get("query_kind") == "visual"
        if (
            prior is None
            or (item_is_visual and not prior_is_visual)
            or (item_is_visual == prior_is_visual and item.get("score", 0) > prior.get("score", 0))
        ):
            best_by_passage[identity] = dict(item)
    ranked = sorted(
        best_by_passage.values(),
        key=lambda item: (float(item.get("score") or 0), item.get("query_kind") == "visual"),
        reverse=True,
    )
    visual = [item for item in ranked if item.get("query_kind") == "visual"]
    general = [item for item in ranked if item.get("query_kind") != "visual"]
    visual_slots = min(len(visual), max(1, MAX_EVIDENCE_PER_ARTICLE // 2))
    selected = visual[:visual_slots] + general[: MAX_EVIDENCE_PER_ARTICLE - visual_slots]
    if len(selected) < MAX_EVIDENCE_PER_ARTICLE:
        selected_ids = {_evidence_identity(item) for item in selected}
        selected.extend(
            item for item in ranked if _evidence_identity(item) not in selected_ids
        )
    return selected[:MAX_EVIDENCE_PER_ARTICLE]


def abstract_for(ns, pmcid: str) -> str:
    """Re-fetch an abstract by pmcid (resume path; not held in memory)."""
    try:
        result = ns.query(
            filters=["pmcid", "Eq", pmcid],
            limit=1,
            include_attributes=["abstract"],
        )
        rows = getattr(result, "rows", [])
        return str(dict(rows[0]).get("abstract") or "") if rows else ""
    except Exception:  # noqa: BLE001 - best-effort metadata
        return ""


def _hydrate_batch(ns, batch: list[str], billing_counters=None) -> dict[str, dict]:
    """One METADATA_BATCH_SIZE slice: multi_query, else one OR-filter query."""
    hydrated: dict[str, dict] = {}
    multi_query = getattr(ns, "multi_query", None)
    # multi_query needs one permit per subquery (max MULTI_QUERY_BATCH);
    # larger batches go straight to the single OR-filter query instead.
    if callable(multi_query) and len(batch) <= MULTI_QUERY_BATCH:
        try:
            queries = [
                {
                    "filters": ["pmcid", "Eq", pmcid],
                    "limit": 1,
                    "include_attributes": METADATA_ATTRIBUTES,
                }
                for pmcid in batch
            ]
            result = multi_query(queries=queries)
            _record_billing(billing_counters, result, query_count=len(queries))
            results = list(getattr(result, "results", []) or [])
            if len(results) != len(batch):
                raise ValueError("incomplete metadata multi_query response")
            for pmcid, item in zip(batch, results):
                rows = getattr(item, "rows", None) or []
                if rows:
                    hydrated[pmcid] = dict(rows[0])
            return hydrated
        except Exception as exc:  # noqa: BLE001 - compatibility fallback
            logger.info("metadata multi_query unavailable (%s); using batched query", exc)
    # A single OR-filter query hydrates the entire batch when multi_query
    # is absent, so compatibility does not regress to one request per ID.
    filters = ["Or", [["pmcid", "Eq", pmcid] for pmcid in batch]]
    result = ns.query(
        filters=filters,
        limit={"total": len(batch), **_PER_PMCID_LIMIT},
        include_attributes=METADATA_ATTRIBUTES,
    )
    _record_billing(billing_counters, result)
    for raw in getattr(result, "rows", []) or []:
        row = dict(raw)
        pmcid = str(row.get("pmcid") or "")
        if pmcid and pmcid not in hydrated:
            hydrated[pmcid] = row
    return hydrated


def hydrate_metadata(ns, pmcids: list[str], billing_counters=None) -> dict[str, dict]:
    """Fetch citation metadata and abstracts in bounded batches for shortlisted IDs."""
    unique_ids = list(dict.fromkeys(str(p) for p in pmcids if p))
    batches = [
        unique_ids[offset : offset + METADATA_BATCH_SIZE]
        for offset in range(0, len(unique_ids), METADATA_BATCH_SIZE)
    ]
    # Batches are disjoint pmcid slices on a stateless HTTP client, so they
    # run concurrently; the merge is order-independent (no key overlap).
    workers = min(len(batches), config.VP_RETRIEVAL_CONCURRENCY)
    hydrated: dict[str, dict] = {}
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for partial in pool.map(
                lambda b: _hydrate_batch(ns, b, billing_counters), batches
            ):
                hydrated.update(partial)
    else:
        for batch in batches:
            hydrated.update(_hydrate_batch(ns, batch, billing_counters))
    return hydrated


def fetch_abstracts(ns, pmcids: list[str]) -> dict[str, str]:
    """Fetch abstracts for IDs in bounded multi-query batches."""
    return {
        pmcid: str(attrs.get("abstract") or "")
        for pmcid, attrs in hydrate_metadata(ns, pmcids).items()
    }


def _merge_evidence(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Merge evidence by (disease, finding, section, text) while keeping the
    strongest hit — passages belonging to a different pair never collapse."""
    merged: dict[tuple[str, str, str, str], dict] = {}
    for item in [*(existing or []), *(incoming or [])]:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        identity = _evidence_identity({**item, "text": text})
        prior = merged.get(identity)
        if (
            prior is None
            or (item.get("query_kind") == "visual" and prior.get("query_kind") != "visual")
            or (
                item.get("query_kind") == prior.get("query_kind")
                and float(item.get("score") or 0) > float(prior.get("score") or 0)
            )
        ):
            merged[identity] = dict(item)
    return _select_evidence(list(merged.values()))


# ---------------------------------------------------------------------------
# DB + relevance
# ---------------------------------------------------------------------------
def upsert_candidate(
    conn,
    pmcid: str,
    attrs: dict,
    score: float,
    keys: set[str],
    matched_passages: list[dict] | None = None,
) -> str:
    """INSERT OR IGNORE a candidate; refresh early-status rows. Returns status."""
    meta = dict(
        pmid=attrs.get("pmid"),
        doi=attrs.get("doi"),
        title=attrs.get("title"),
        journal=attrs.get("journal"),
        year=attrs.get("year"),
        country=attrs.get("country") or None,
        publication_types_json=db.to_json(attrs.get("publication_type") or []),
        retrieval_score=score,
        retrieval_evidence_json=db.to_json(
            matched_passages[:MAX_EVIDENCE_PER_ARTICLE] if matched_passages else []
        ),
    )
    row = conn.execute(
        "SELECT status, retrieval_score, retrieval_evidence_json, primary_disease_keys_json FROM articles "
        "WHERE pmcid = ?",
        (pmcid,),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT OR IGNORE INTO articles "
            "(pmcid, pmid, doi, title, journal, year, country, "
            " publication_types_json, retrieval_score, retrieval_evidence_json, primary_disease_keys_json, "
            " status) VALUES (:pmcid, :pmid, :doi, :title, :journal, :year, "
            " :country, :publication_types_json, :retrieval_score, :retrieval_evidence_json, :keys, "
            " 'candidate')",
            {**meta, "pmcid": pmcid, "keys": db.to_json(sorted(keys))},
        )
        return "candidate"
    status = row["status"]
    incoming_evidence = matched_passages or []
    previous_evidence = db.from_json(row["retrieval_evidence_json"], []) or []
    merged_evidence = _merge_evidence(previous_evidence, incoming_evidence)
    evidence_json = db.to_json(merged_evidence)
    if status in {"candidate", "license_ok", "license_rejected"}:
        merged_keys = sorted(set(db.from_json(row["primary_disease_keys_json"], [])) | keys)
        merged_score = max(score, row["retrieval_score"] or 0.0)
        conn.execute(
            "UPDATE articles SET pmid=:pmid, doi=:doi, title=:title, "
            "journal=:journal, year=:year, country=:country, "
            "publication_types_json=:publication_types_json, "
            "retrieval_score=:retrieval_score, "
            "retrieval_evidence_json=:evidence_json, "
            "primary_disease_keys_json=:keys, "
            "updated_at=datetime('now') WHERE pmcid=:pmcid",
            {
                **meta,
                "retrieval_score": merged_score,
                "keys": db.to_json(merged_keys),
                "evidence_json": evidence_json,
                "pmcid": pmcid,
            },
        )
    else:
        updates = []
        values: list = []
        if score > (row["retrieval_score"] or 0.0):
            updates.append("retrieval_score=?")
            values.append(score)
        if incoming_evidence and evidence_json != row["retrieval_evidence_json"]:
            updates.append("retrieval_evidence_json=?")
            values.append(evidence_json)
        if updates:
            updates.append("updated_at=datetime('now')")
            values.append(pmcid)
            conn.execute(
                f"UPDATE articles SET {', '.join(updates)} WHERE pmcid=?", values
            )
    return status


def upsert_manifestation_candidates(conn, candidates: list[dict]) -> int:
    """Persist per-finding ranks, retaining the best rank seen for each pair."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='manifestation_candidates'"
    ).fetchone()
    if not exists:
        logger.info("manifestation_candidates table not present; skipping rank persistence")
        return 0
    # The production table may declare a foreign key to articles.pmcid. We
    # insert new article shortlist rows first; rank-only hits outside that
    # shortlist can be recorded on a later run when their article row exists.
    article_ids = {
        row[0] for row in conn.execute("SELECT pmcid FROM articles")
    }
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(manifestation_candidates)")
    }
    has_provenance = {"provenance_status", "provenance_disease_key"} <= columns
    written = 0
    for item in candidates:
        if (
            not item.get("disease_key")
            or not item.get("finding_key")
            or not item.get("pmcid")
            or item["pmcid"] not in article_ids
        ):
            continue
        if has_provenance:
            conn.execute(
                "INSERT INTO manifestation_candidates "
                "(disease_key, finding_key, pmcid, query, best_rank, retrieval_score, "
                "provenance_status, provenance_disease_key) "
                "VALUES (?, ?, ?, ?, ?, ?, 'explicit', ?) "
                "ON CONFLICT(disease_key, finding_key, pmcid) DO UPDATE SET "
                "query=CASE WHEN excluded.best_rank IS NOT NULL AND "
                "(manifestation_candidates.best_rank IS NULL OR "
                "excluded.best_rank < manifestation_candidates.best_rank) "
                "THEN excluded.query ELSE manifestation_candidates.query END, "
                "retrieval_score=CASE WHEN excluded.best_rank IS NOT NULL AND "
                "(manifestation_candidates.best_rank IS NULL OR "
                "excluded.best_rank < manifestation_candidates.best_rank) "
                "THEN excluded.retrieval_score ELSE manifestation_candidates.retrieval_score END, "
                "best_rank=CASE WHEN excluded.best_rank IS NULL THEN manifestation_candidates.best_rank "
                "WHEN manifestation_candidates.best_rank IS NULL THEN excluded.best_rank "
                "ELSE MIN(manifestation_candidates.best_rank, excluded.best_rank) END, "
                "provenance_status='explicit', "
                "provenance_disease_key=excluded.provenance_disease_key, "
                "updated_at=datetime('now')",
                (
                    item["disease_key"], item["finding_key"], item["pmcid"],
                    item["query"], int(item["best_rank"]),
                    float(item["retrieval_score"]), item["disease_key"],
                ),
            )
        else:
            conn.execute(
                "INSERT INTO manifestation_candidates "
                "(disease_key, finding_key, pmcid, query, best_rank, retrieval_score) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(disease_key, finding_key, pmcid) DO UPDATE SET "
                "query=CASE WHEN excluded.best_rank IS NOT NULL AND "
                "(manifestation_candidates.best_rank IS NULL OR "
                "excluded.best_rank < manifestation_candidates.best_rank) "
                "THEN excluded.query ELSE manifestation_candidates.query END, "
                "retrieval_score=CASE WHEN excluded.best_rank IS NOT NULL AND "
                "(manifestation_candidates.best_rank IS NULL OR "
                "excluded.best_rank < manifestation_candidates.best_rank) "
                "THEN excluded.retrieval_score ELSE manifestation_candidates.retrieval_score END, "
                "best_rank=CASE WHEN excluded.best_rank IS NULL THEN manifestation_candidates.best_rank "
                "WHEN manifestation_candidates.best_rank IS NULL THEN excluded.best_rank "
                "ELSE MIN(manifestation_candidates.best_rank, excluded.best_rank) END, "
                "updated_at=datetime('now')",
                (
                    item["disease_key"], item["finding_key"], item["pmcid"], item["query"],
                    int(item["best_rank"]), float(item["retrieval_score"]),
                ),
            )
        written += 1
    return written


def join_license(pmcid: str) -> tuple[str, dict]:
    """Fetch one license (no DB access — safe on the worker pool)."""
    try:
        lic = pmc.get_license(pmcid)
    except Exception as exc:  # noqa: BLE001 - access errors remain retryable
        return pmcid, {
            "status": "candidate",
            "error": str(exc),
        }
    allows = pmc.license_allows(lic.code)
    return pmcid, {
        "status": "license_ok" if allows else "license_rejected",
        "license_code": lic.code,
        "license_url": lic.url,
        "license_raw": lic.raw,
        "oa_subset": lic.oa_subset,
        # C3 hints (W4b): persisted on the article row so parse can use the
        # hinted get_article_bundle path and skip re-listing the S3 dir.
        "s3_prefix": lic.prefix,
        "media_files": lic.media_files,
        "error": None,
    }


def join_licenses(pmcids: list[str], pool=None) -> list[tuple[str, dict]]:
    """License many articles: Europe PMC batched lookups are the final
    source; the S3 join_license path fills only missing records or absent
    license fields (availability fallback, never a confirmation check).

    No DB access. Input order is preserved. EPMC outcomes carry the same
    fields as join_license minus the s3_prefix/media_files hint keys, so
    apply_license leaves those columns untouched for EPMC-sourced rows.
    """
    epmc = pmc.get_licenses_epmc(pmcids)
    outcomes: dict[str, dict] = {}
    missing: list[str] = []
    for pmcid in pmcids:
        lic = epmc.get(pmcid)
        if lic is None or not lic.raw:
            missing.append(pmcid)
            continue
        allows = pmc.license_allows(lic.code)
        outcomes[pmcid] = {
            "status": "license_ok" if allows else "license_rejected",
            "license_code": lic.code,
            "license_url": lic.url,
            "oa_subset": lic.oa_subset,
            "license_source": "epmc",
            "license_raw": lic.raw,
            "error": None,
        }
    timing.count("license_source", len(outcomes), source="epmc")
    if missing:
        if pool is not None:
            fallback = pool.map(join_license, missing)
        else:
            with ThreadPoolExecutor(
                max_workers=config.VP_FETCH_CONCURRENCY
            ) as local:
                fallback = local.map(join_license, missing)
        for pmcid, outcome in fallback:
            outcome["license_source"] = "s3_fallback"
            outcome['fallback_reason'] = 'missing_epmc_license' if pmcid in epmc else 'missing_epmc_record'
            outcomes[pmcid] = outcome
        timing.count("license_source", len(missing), source="s3_fallback")
    return [(pmcid, outcomes[pmcid]) for pmcid in pmcids]


def apply_license(conn, pmcid: str, outcome: dict) -> str:
    from . import queue_state
    fields = {
        "error": outcome.get("error"),
        "relevance_reason": outcome.get("error")
        or f"license:{outcome.get('license_code')}",
    }
    if "s3_prefix" in outcome or "media_files" in outcome:
        fields.update(
            s3_prefix=outcome.get("s3_prefix"),
            media_files_json=db.to_json(list(outcome.get("media_files") or [])),
        )
    if outcome.get("status") == "license_ok":
        fields.update(
            license_code=outcome.get("license_code"),
            license_url=outcome.get("license_url"),
            oa_subset=outcome.get("oa_subset"),
            relevance_reason=None,
        )
    else:
        fields.update(
            license_code=outcome.get("license_code"),
            license_url=outcome.get("license_url"),
            oa_subset=outcome.get("oa_subset"),
        )
    db.set_status(conn, "articles", pmcid, outcome["status"], **fields)
    status = outcome['status']
    queue_state.record(conn, pmcid, 'license', 'error' if status == 'candidate' else 'complete',
                       'access_error' if status == 'candidate' else
                       'missing_license' if outcome.get('license_code') in (None, 'none') else
                       'allowed' if status == 'license_ok' else 'disallowed_license',
                       source=outcome.get('license_source', 's3_fallback'),
                       license_code=outcome.get('license_code'), license_raw=outcome.get('license_raw'),
                       fallback_reason=outcome.get('fallback_reason'), error=outcome.get('error'))
    return outcome["status"]


def _p1_relevant(parsed: dict) -> bool:
    return bool(
        parsed.get("decision") == "relevant"
        and parsed.get("is_narrative_review")
        and parsed.get("primary_disease_keys")
    )


def _p1_request(title: str, abstract: str) -> dict:
    return {
        "stage": "p1",
        "model": config.VP_TRIAGE_MODEL,
        "system": P1.system,
        "schema": P1.schema,
        "prompt_version": P1.version,
        "user_content": f"Title: {title}\nAbstract: {abstract[:ABSTRACT_MAX_CHARS]}",
    }


def apply_relevance(
    conn, pmcid: str, status: str, reason: str | None, keys: list[str] | None
) -> None:
    fields: dict = {
        "error": None,
        "relevance_decision": "relevant" if status == "relevant" else "irrelevant",
        "relevance_reason": reason,
    }
    if keys:
        fields["primary_disease_keys_json"] = db.to_json(sorted(keys))
    db.set_status(conn, "articles", pmcid, status, **fields)


def process_relevance(conn, client, requests, submitted) -> dict:
    """Checkpoint completion-order results, including failures, one at a time."""
    from . import queue_state
    counts = {"completed": 0, "errors": 0, "budget": 0}
    try:
        for result in client.iter_many(requests):
            pmcid = submitted[result.index]
            with client._lock:
                if result.error is not None:
                    budget = isinstance(result.error, llm.BudgetExceeded)
                    counts['budget' if budget else 'errors'] += 1
                    conn.execute("UPDATE articles SET error=?,updated_at=datetime('now') WHERE pmcid=?",
                                 (str(result.error), pmcid))
                    queue_state.record(conn, pmcid, 'relevance', 'deferred' if budget else 'error',
                                       'budget' if budget else 'provider_error', error=str(result.error))
                else:
                    parsed = result.parsed or {}
                    keys = [k for k in parsed.get('primary_disease_keys', []) if k in PILOT_KEYS]
                    relevant = _p1_relevant(parsed)
                    apply_relevance(conn, pmcid, 'relevant' if relevant else 'irrelevant',
                                    parsed.get('reason') or 'p1_verdict', keys if relevant else None)
                    queue_state.record(conn, pmcid, 'relevance', 'complete',
                                       'relevant' if relevant else 'irrelevant')
                    counts['completed'] += 1
                conn.commit()
    except BaseException as exc:
        with client._lock:
            for pmcid in submitted:
                row = conn.execute('SELECT status FROM articles WHERE pmcid=?', (pmcid,)).fetchone()
                if row and row['status'] == 'license_ok':
                    state = conn.execute("SELECT status FROM article_queue_state WHERE pmcid=? AND stage='relevance'", (pmcid,)).fetchone()
                    if state is None or state['status'] == 'running':
                        queue_state.record(conn, pmcid, 'relevance', 'deferred',
                                           'interruption' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'feed_error',
                                           error=str(exc))
            conn.commit()
        raise
    return counts


def needs_manifestation_licenses(conn, disease_key: str, finding_keys=None) -> bool:
    """Continue past a disease quota while deficient lanes lack usable candidates."""
    from . import manifestation_queue
    _floor, _target, cap = config.validate_coverage_settings()
    coverage = manifestation_queue.published_coverage(conn, disease_key)
    for row in conn.execute("SELECT finding_key,disease_keys_json FROM findings_vocab WHERE approved=1"):
        finding = row['finding_key']
        if finding_keys is not None and finding not in finding_keys:
            continue
        if disease_key not in db.from_json(row['disease_keys_json'], []):
            continue
        if coverage.get(finding, 0) >= cap:
            continue
        available = conn.execute(
            "SELECT COUNT(*) FROM manifestation_candidates mc JOIN articles a USING(pmcid) "
            "WHERE mc.disease_key=? AND mc.finding_key=? "
            "AND mc.provenance_status='explicit' "
            "AND mc.provenance_disease_key=mc.disease_key AND (a.status='license_ok' OR "
            "(a.status='relevant' AND EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) WHERE value=?)))",
            (disease_key, finding, disease_key),
        ).fetchone()[0]
        if available >= max(1, config.VP_MANIFESTATION_QUOTA):
            continue
        if conn.execute(
            "SELECT 1 FROM manifestation_candidates mc JOIN articles a USING(pmcid) "
            "WHERE mc.disease_key=? AND mc.finding_key=? AND a.status='candidate' "
            "AND mc.provenance_status='explicit' "
            "AND mc.provenance_disease_key=mc.disease_key LIMIT 1",
            (disease_key, finding),
        ).fetchone():
            return True
    return False


# ---------------------------------------------------------------------------
# Bounded per-pair replenishment (search_policy is the settled authority)
# ---------------------------------------------------------------------------
def _attempt_query(ns, spec, counters=None) -> list[dict]:
    """Issue exactly the request a settled attempt spec describes — once.

    Depth, rank_by, filters and attributes are used unchanged. Unlike the
    broad initial retrieval, replenishment never falls back to a split or
    altered filter: ANY provider rejection (including an unsupported ``Or``)
    propagates so the attempt is recorded as a retryable retrieval error.
    """
    return _ns_query(
        ns, spec["rank_by"], spec["filters"], spec["depth"],
        counters, spec["include_attributes"],
    )


def _attempt_begin(conn, spec) -> None:
    """Durable 'started' row — committed before the provider call so an
    interrupted attempt stays resumable on the same ledger key."""
    conn.execute(
        "INSERT INTO pair_search_attempts "
        "(disease_key, finding_key, policy_version, round_no, query, "
        "query_filter_hash, filters_json, depth, status, started_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'started', datetime('now')) "
        "ON CONFLICT(disease_key, finding_key, policy_version, round_no, "
        "query_filter_hash, depth) DO UPDATE SET "
        "status='started', error=NULL, "
        "started_at=datetime('now'), completed_at=NULL",
        (
            spec["disease_key"], spec["finding_key"], spec["policy_version"],
            spec["round_no"], spec["query"], spec["query_filter_hash"],
            spec["filters_json"], spec["depth"],
        ),
    )
    conn.commit()


def _attempt_record(
    conn, spec, status, *, returned=None, outcomes=None, error=None, new_ids=None
) -> None:
    sets = ["status=?", "error=?"]
    values: list = [status, error]
    # completed_at marks full hydration only; 'retrieved'/'error' rows stay
    # open so a later invocation resumes them on the same ledger key.
    if status == "completed":
        sets.append("completed_at=datetime('now')")
    if returned is not None:
        sets.append("returned_pmcids_json=?")
        values.append(db.to_json(returned))
    if new_ids is not None:
        sets.append("new_pmcids_json=?")
        values.append(db.to_json(new_ids))
    if outcomes is not None:
        sets.append("pending_outcomes_json=?")
        values.append(json.dumps(outcomes, default=str))
    conn.execute(
        "UPDATE pair_search_attempts SET " + ", ".join(sets) + " "
        "WHERE disease_key=? AND finding_key=? AND policy_version=? "
        "AND round_no=? AND query_filter_hash=? AND depth=?",
        (
            *values, spec["disease_key"], spec["finding_key"],
            spec["policy_version"], spec["round_no"], spec["query_filter_hash"],
            spec["depth"],
        ),
    )
    conn.commit()


def _refresh_attempt_outcomes(conn, spec) -> None:
    """Update a completed attempt's outcome map to current article statuses.

    ``retrieval_rows`` and prior statuses are preserved; type-rejected IDs
    keep their marker, and never-persisted IDs read ``missing``.
    """
    row = conn.execute(
        "SELECT returned_pmcids_json, pending_outcomes_json FROM pair_search_attempts "
        "WHERE disease_key=? AND finding_key=? AND policy_version=? AND round_no=? "
        "AND query_filter_hash=? AND depth=?",
        (
            spec["disease_key"], spec["finding_key"], spec["policy_version"],
            spec["round_no"], spec["query_filter_hash"], spec["depth"],
        ),
    ).fetchone()
    if row is None:
        return
    stored = db.from_json(row["pending_outcomes_json"], {}) or {}
    outcomes = stored.get("outcomes") or {}
    for pmcid in db.from_json(row["returned_pmcids_json"], []) or []:
        current = conn.execute(
            "SELECT status FROM articles WHERE pmcid=?", (pmcid,)
        ).fetchone()
        if current is not None:
            outcomes[pmcid] = current["status"]
        elif pmcid not in outcomes:
            outcomes[pmcid] = "missing"
    stored["outcomes"] = outcomes
    _attempt_record(conn, spec, "completed", outcomes=stored)


def _hydrate_attempt(conn, ns, disease_key, finding_key, spec, rows, counters):
    """Persist one attempt's rows as explicit pair candidates.

    Rows are type-filtered by the existing review gate, hydrated for
    metadata, upserted as article shortlist rows with pair-scoped evidence,
    then recorded on the pair with explicit provenance. Returns
    ``(new_ids, outcomes)`` — the PMCIDs newly recorded for this pair
    (candidate rows that already existed never count as new) and the
    current per-PMCID outcome map for the attempt ledger.
    """
    per_pmcid: dict[str, list[tuple[int, dict]]] = {}
    order: list[str] = []
    for rank, row in enumerate(rows, start=1):
        pmcid = str(row.get("pmcid") or "")
        if not pmcid:
            continue
        if pmcid not in per_pmcid:
            per_pmcid[pmcid] = []
            order.append(pmcid)
        per_pmcid[pmcid].append((rank, row))

    typed = [
        pmcid
        for pmcid in order
        if passes_type_filter(
            per_pmcid[pmcid][0][1].get("publication_type") or [],
            per_pmcid[pmcid][0][1].get("article_type"),
        )
    ]
    metadata = hydrate_metadata(ns, typed, counters) if typed else {}
    existing = {
        row["pmcid"]
        for row in conn.execute(
            "SELECT pmcid FROM manifestation_candidates "
            "WHERE disease_key=? AND finding_key=?",
            (disease_key, finding_key),
        )
    }
    records = []
    for pmcid in typed:
        hits = per_pmcid[pmcid]
        attrs = {**hits[0][1], **(metadata.get(pmcid) or {})}
        passages = [
            {
                "disease_key": disease_key,
                "finding_key": finding_key,
                "query": spec["query"],
                "query_kind": "pair_search",
                "text": str(row.get("page_content") or "").strip()[:MAX_EVIDENCE_TEXT_CHARS],
                "section": str(row.get("section_title") or ""),
                "section_type": str(row.get("section_type") or ""),
                "modality": "",
                "finding": finding_key,
                "rank": rank,
                "score": 1.0 / (RRF_K + rank),
            }
            for rank, row in hits
            if str(row.get("page_content") or "").strip()
        ]
        upsert_candidate(
            conn, pmcid, attrs,
            1.0 / (RRF_K + hits[0][0]), {disease_key}, passages,
        )
        records.append({
            "disease_key": disease_key,
            "finding_key": finding_key,
            "pmcid": pmcid,
            "query": spec["query"],
            "best_rank": hits[0][0],
            "retrieval_score": 1.0 / (RRF_K + hits[0][0]),
        })
    upsert_manifestation_candidates(conn, records)
    conn.commit()

    # Every returned PMCID maps to its actual current article status; rows
    # rejected by the type filter or never persisted read explicitly.
    outcomes: dict[str, str] = {}
    statuses = {
        row["pmcid"]: row["status"]
        for row in conn.execute(
            "SELECT pmcid, status FROM articles WHERE pmcid IN "
            f"({','.join('?' for _ in typed)})",
            typed,
        )
    } if typed else {}
    typed_set = set(typed)
    for pmcid in order:
        if pmcid not in typed_set:
            outcomes[pmcid] = "type_rejected"
        else:
            outcomes[pmcid] = statuses.get(pmcid, "missing")
    new_ids = [record["pmcid"] for record in records if record["pmcid"] not in existing]
    return new_ids, outcomes


def replenish_pair(
    conn,
    disease_key: str,
    finding_key: str,
    *,
    round_no: int,
    dry_run: bool = False,
    budget_usd=None,
    max_runtime_seconds=None,
    max_articles=None,
) -> dict:
    """Bounded per-pair replenishment via the settled search policy.

    Each invocation first drains the pair's explicit pending candidates
    (license, then P1 relevance); any article reaching ``relevant`` returns
    ``pending_work`` for the caller to parse. Only then does the round's
    unattempted search specs run exactly as ``search_policy`` defines them —
    query, filters, depth and attributes unchanged. Ledgered attempts resume
    after interruption: ``retrieved`` attempts re-hydrate their stored rows
    without re-querying, and ``completed`` attempts never fire again.
    Provider errors mark the attempt ``error`` — resumable, never proof the
    corpus is exhausted. Optional limits of 0 pause immediately; they are
    checked before every provider call, never clamped.
    """
    from . import search_policy

    started = time.monotonic()
    counters = {
        "requests": 0,
        "queries": 0,
        "billable_logical_bytes_queried": 0,
        "billable_logical_bytes_returned": 0,
    }
    specs = search_policy.attempt_specs(conn, disease_key, finding_key, round_no)
    pending = search_policy.pending_candidates(conn, disease_key, finding_key)
    # ``processed`` tracks every unique PMCID that consumed provider work in
    # this invocation — drained pending candidates AND hydrated search rows —
    # so ``max_articles`` bounds the whole set once per article. ``drained``
    # is the narrower set that completed the license+P1 path here; only it
    # suppresses re-draining (a hydrated candidate still gets drained).
    state = {"queries": 0, "processed": set(), "drained": set(),
             "new_candidates": 0}
    holders = {"retriever": None, "client": None}

    if dry_run:
        return {
            "new_candidates": 0,
            "pending_candidates": len(pending),
            "queries_attempted": 0,
            "status": "dry_run",
            "reason": None,
            "processed_pmcids": [],
            "spent_usd": 0.0,
            "plan": [
                {
                    "query": spec["query"],
                    "depth": spec["depth"],
                    "round_no": spec["round_no"],
                    "attempted": (
                        spec["prior_attempt"]["status"]
                        if spec["prior_attempt"]
                        else None
                    ),
                }
                for spec in specs
            ],
        }

    def _result(status, reason=None):
        return {
            "new_candidates": state["new_candidates"],
            "pending_candidates": len(
                search_policy.pending_candidates(conn, disease_key, finding_key)
            ),
            "queries_attempted": state["queries"],
            "status": status,
            "reason": reason,
            "processed_pmcids": sorted(state["processed"]),
            "spent_usd": (
                holders["client"].spent_usd if holders["client"] else 0.0
            ),
        }

    def _limit_reason():
        if (
            max_runtime_seconds is not None
            and time.monotonic() - started >= max_runtime_seconds
        ):
            return "runtime_limit"
        if (
            budget_usd is not None
            and holders["client"] is not None
            and holders["client"].spent_usd >= budget_usd
        ):
            return "budget"
        if budget_usd is not None and budget_usd <= 0:
            return "budget"
        if max_articles is not None and len(state["processed"]) >= max_articles:
            return "article_limit"
        return None

    def _ns():
        if holders["retriever"] is None:
            holders["retriever"] = _make_retriever()
        return holders["retriever"].ns_pmc

    def _client():
        if holders["client"] is None:
            holders["client"] = llm.LLMClient(
                db_conn=conn,
                budget_usd=budget_usd,
                concurrency=min(8, config.VP_P1_CONCURRENCY),
                timeout_seconds=60,
            )
        return holders["client"]

    def _refresh_pending():
        nonlocal pending
        pending = search_policy.pending_candidates(conn, disease_key, finding_key)
        return pending

    def _drain_pending():
        """License + P1 the pair's pending candidates under hard caps.

        One unique article is counted once even when it consumes both a
        license and a P1 call; limits are checked before every provider
        step. Provider/access failures leave the row's status untouched so
        a later invocation resumes it — the caller must never advance the
        search while candidates remain unfinished.
        """
        _refresh_pending()
        for row in pending:
            pmcid = row["pmcid"]
            status = row["status"]
            if status == "relevant" or pmcid in state["drained"]:
                continue
            reason = _limit_reason()
            if reason:
                return reason
            # From here this unique article consumes provider work (license
            # and/or abstract+P1 calls); count it once toward the limit.
            state["processed"].add(pmcid)
            if status == "candidate":
                try:
                    checked, outcome = join_license(pmcid)
                except Exception as exc:  # noqa: BLE001 - resumable access error
                    return f"access_error: {exc}"
                try:
                    status = apply_license(conn, checked, outcome)
                except Exception as exc:  # noqa: BLE001 - resumable access error
                    return f"access_error: {exc}"
                conn.commit()
                if status == "candidate":
                    # join_license reports unresolved access as a row, not an
                    # exception — still a retryable access error.
                    detail = outcome.get("error") or "license access unresolved"
                    return f"access_error: {detail}"
            if status == "license_ok":
                reason = _limit_reason()
                if reason:
                    return reason
                try:
                    abstracts = fetch_abstracts(_ns(), [pmcid])
                except Exception as exc:  # noqa: BLE001 - resumable access error
                    return f"access_error: {exc}"
                reason = _limit_reason()
                if reason:
                    return reason
                try:
                    counts = process_relevance(
                        conn,
                        _client(),
                        iter([
                            _p1_request(
                                row["title"] or "", abstracts.get(pmcid, "")
                            )
                        ]),
                        [pmcid],
                    )
                except llm.BudgetExceeded:
                    return "budget"
                except Exception as exc:  # noqa: BLE001 - resumable provider error
                    return f"provider_error: {exc}"
                if counts.get("budget"):
                    return "budget"
                if counts.get("errors"):
                    return f"provider_error: {counts['errors']} relevance call(s) failed"
                current = conn.execute(
                    "SELECT status FROM articles WHERE pmcid=?", (pmcid,)
                ).fetchone()
                status = current["status"] if current else status
            state["drained"].add(pmcid)
            # Newly relevant work is handed back immediately rather than
            # spending more of the bounded budget on other pending rows.
            if status == "relevant":
                _refresh_pending()
                return None
        _refresh_pending()
        return None

    def _pending_gate(pause):
        """Relevant work wins, then the drain's pause reason, then any
        unfinished candidates — a search never completes while rows remain."""
        _refresh_pending()
        if any(row["status"] == "relevant" for row in pending):
            return _result("pending_work", reason="relevant_pending")
        if pause:
            return _result("paused", reason=pause)
        if pending:
            return _result("paused", reason="pending_candidates")
        return None

    # Relevant pending work is handed back for parsing immediately — before
    # drains or any new retrieval; a paused limit never swallows it.
    if any(row["status"] == "relevant" for row in pending):
        return _result("pending_work", reason="relevant_pending")
    gate = _pending_gate(_drain_pending())
    if gate:
        return gate

    for spec in specs:
        prior = spec["prior_attempt"] or {}
        if prior.get("status") == "completed":
            continue
        if prior.get("status") == "retrieved":
            # Resume from the stored raw rows; no provider call is issued.
            stored = db.from_json(prior.get("pending_outcomes_json"), {}) or {}
            rows = stored.get("retrieval_rows") or []
        else:
            reason = _limit_reason()
            if reason:
                return _result("paused", reason=reason)
            _attempt_begin(conn, spec)
            state["queries"] += 1  # issued calls count, even failed ones
            try:
                rows = _attempt_query(_ns(), spec, counters)
            except Exception as exc:  # noqa: BLE001 - retryable provider error
                _attempt_record(conn, spec, "error", error=str(exc))
                return _result("retrieval_error", reason=str(exc))
            returned = list(
                dict.fromkeys(
                    str(r.get("pmcid") or "") for r in rows if r.get("pmcid")
                )
            )
            _attempt_record(
                conn, spec, "retrieved", returned=returned,
                outcomes={"retrieval_rows": rows, "outcomes": {}},
            )
            conn.execute(
                "UPDATE manifestation_lanes SET last_search_at=datetime('now') "
                "WHERE disease_key=? AND finding_key=?",
                (disease_key, finding_key),
            )
            conn.commit()
        reason = _limit_reason()
        if reason:
            return _result("paused", reason=reason)
        try:
            new_ids, outcomes = _hydrate_attempt(
                conn, _ns(), disease_key, finding_key, spec, rows, counters
            )
        except Exception as exc:  # noqa: BLE001 - hydration failure stays resumable
            # Keep status 'retrieved' with the raw rows intact; record the
            # error so the next invocation resumes hydration, not the query.
            _attempt_record(conn, spec, "retrieved", error=f"hydration_error: {exc}")
            return _result("retrieval_error", reason=f"hydration_error: {exc}")
        state["new_candidates"] += len(new_ids)
        # Metadata hydration touched every type-passed row — count each
        # unique PMCID once toward the invocation's article limit. Hydrated
        # rows are NOT drained yet: the _drain_pending pass below licenses
        # and P1-checks them before any further search spec fires.
        state["processed"].update(
            pmcid
            for pmcid, outcome in outcomes.items()
            if outcome != "type_rejected"
        )
        _attempt_record(
            conn, spec, "completed", new_ids=new_ids,
            outcomes={"retrieval_rows": rows, "outcomes": outcomes},
        )
        gate = _pending_gate(_drain_pending())
        _refresh_attempt_outcomes(conn, spec)
        if gate:
            return gate

    gate = _pending_gate(None)
    if gate:
        return gate
    next_round = search_policy.next_round(conn, disease_key, finding_key)
    if next_round is None:
        return _result("search_plan_exhausted", reason="all_rounds_complete")
    return _result("round_complete", reason=f"next_round={next_round}")


def _queue_rows(conn, args, statuses):
    where = ['a.status IN (%s)' % ','.join('?' for _ in statuses)]
    params = list(statuses)
    finding = getattr(args, 'finding', None)
    if finding:
        clause = 'mc.pmcid=a.pmcid AND mc.finding_key=?'
        params.append(finding)
        if args.disease != 'all':
            clause += ' AND mc.disease_key=?'
            params.append(args.disease)
        where.append('EXISTS (SELECT 1 FROM manifestation_candidates mc WHERE ' + clause + ')')
    elif args.disease != 'all':
        where.append("(EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) WHERE value=?) "
                     "OR EXISTS (SELECT 1 FROM manifestation_candidates mc WHERE mc.pmcid=a.pmcid AND mc.disease_key=?))")
        params.extend([args.disease, args.disease])
    if args.pmcids:
        where.append('a.pmcid IN (%s)' % ','.join('?' for _ in args.pmcids))
        params.extend(args.pmcids)
    return [dict(r) for r in conn.execute(
        'SELECT a.* FROM articles a WHERE ' + ' AND '.join(where) +
        " ORDER BY (a.error IS NOT NULL) DESC, a.retrieval_score DESC,a.pmcid", params)]


def _queue_backup(conn, name):
    import sqlite3
    from datetime import datetime, timezone
    path = config.reports_dir() / f"pre_{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as backup:
        conn.backup(backup)
    return str(path)


def _save_queue_report(conn, args, name, **details):
    rows = _queue_rows(conn, args, ['candidate','license_ok','license_rejected','relevant','irrelevant','parsed','parse_error'])
    ids = {r['pmcid'] for r in rows}
    states = [dict(r) for r in conn.execute('SELECT * FROM article_queue_state') if r['pmcid'] in ids]
    from collections import Counter
    report = {'disease': args.disease, 'finding': getattr(args,'finding',None),
              'article_statuses': dict(Counter(r['status'] for r in rows)),
              'relevant_for_disease': sum(r['status']=='relevant' and (args.disease=='all' or args.disease in db.from_json(r['primary_disease_keys_json'], [])) for r in rows),
              'queue_outcomes': dict(Counter(f"{r['stage']}:{r['status']}:{r['reason']}" for r in states)),
              'states': states, **details}
    path = config.reports_dir() / f'{name}_{args.disease}_{getattr(args,"finding",None) or "all"}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('states','decisions')}, indent=2))
    print(f'Queue report: {path}')


def run_license_audit(args):
    """Recheck unrecognized licenses; explicit NC licenses are not overridden."""
    if args.dry_run:
        # Dry-run exits before opening the DB or any client: describe the
        # planned scope only.
        print(
            f'Would audit unrecognized licenses '
            f'(disease={args.disease}, limit={args.limit}); '
            f'no DB reads/writes, no provider calls'
        )
        return 0
    conn = db.init_db()
    try:
        rows = [r for r in _queue_rows(conn,args,['license_rejected']) if r['license_code'] in (None,'none','other')]
        if args.limit is not None:
            rows = rows[:args.limit]
        backup = _queue_backup(conn,'license_audit')
        decisions = []
        with ThreadPoolExecutor(max_workers=config.VP_FETCH_CONCURRENCY) as pool:
            for start in range(0,len(rows),config.VP_EPMC_LICENSE_BATCH):
                chunk = rows[start:start + config.VP_EPMC_LICENSE_BATCH]
                for pmcid,outcome in join_licenses([r['pmcid'] for r in chunk],pool):
                    prior = next(r for r in chunk if r['pmcid']==pmcid)
                    apply_license(conn,pmcid,outcome)
                    decisions.append({'pmcid':pmcid,'previous_license':prior['license_code'],**outcome})
                    conn.commit()
        _save_queue_report(conn,args,'license_audit',backup_path=backup,decisions=decisions,
                           audited=len(decisions),recovered=sum(r['status']=='license_ok' for r in decisions))
        return 0
    finally:
        conn.close()


def run_resume(args):
    """Resume persisted queues without repeating retrieval or completed verdicts."""
    if args.dry_run:
        # Dry-run exits before opening the DB or instantiating clients:
        # describe the planned scope only.
        max_requests = args.limit if args.limit is not None else args.max_articles
        print(
            f'Would resume licensed articles and unchecked candidates '
            f'(disease={args.disease}, max {max_requests} relevance calls); '
            f'no DB reads/writes, no provider calls'
        )
        return 0
    conn = db.init_db()
    submitted = []
    client = llm.LLMClient(db_conn=conn,budget_usd=args.budget_usd,
                           concurrency=min(8,config.VP_P1_CONCURRENCY),timeout_seconds=60)
    max_requests = args.limit if args.limit is not None else args.max_articles
    started = time.monotonic()
    try:
        licensed = _queue_rows(conn,args,['license_ok'])
        candidates = _queue_rows(conn,args,['candidate'])
        backup = _queue_backup(conn,'queue_resume')
        # Previous active submissions have no durable verdict and are retryable.
        for row in licensed:
            queue_state.record(conn,row['pmcid'],'relevance','queued','resume_pending')
        conn.commit()
        retriever = _make_retriever()
        abstracts = fetch_abstracts(retriever.ns_pmc,[r['pmcid'] for r in licensed[:max_requests]]) if licensed else {}

        def request(row):
            abstract = abstracts.get(row['pmcid'])
            if abstract is None:
                abstract = abstract_for(retriever.ns_pmc,row['pmcid'])
            with client._lock:
                queue_state.record(conn,row['pmcid'],'relevance','running','submitted')
                conn.commit()
            submitted.append(row['pmcid'])
            return _p1_request(row['title'] or '',abstract)

        def stop_reason():
            if time.monotonic() - started >= args.max_runtime_seconds:
                return 'runtime_limit'
            if args.budget_usd is not None and client.spent_usd >= args.budget_usd:
                return 'budget'
            if len(submitted) >= max_requests:
                return 'request_limit'
            return None

        def feed():
            for row in licensed:
                reason = stop_reason()
                if reason:
                    with client._lock:
                        for remaining in licensed:
                            if remaining['pmcid'] not in submitted:
                                queue_state.record(conn,remaining['pmcid'],'relevance','deferred',reason)
                        for remaining in candidates:
                            queue_state.record(conn,remaining['pmcid'],'license','deferred',reason)
                        conn.commit()
                    return
                yield request(row)
            with ThreadPoolExecutor(max_workers=config.VP_FETCH_CONCURRENCY) as pool:
                for start in range(0,len(candidates),20):
                    reason = stop_reason()
                    findings = {args.finding} if args.finding else None
                    if not reason and args.disease != 'all' and not needs_manifestation_licenses(conn,args.disease,findings):
                        reason = 'manifestation_candidate_target'
                    if reason:
                        with client._lock:
                            for remaining in candidates[start:]:
                                queue_state.record(conn,remaining['pmcid'],'license','deferred',reason)
                            conn.commit()
                        return
                    chunk = candidates[start:start+20]
                    outcomes = join_licenses([r['pmcid'] for r in chunk],pool)
                    accepted = []
                    with client._lock:
                        for pmcid,outcome in outcomes:
                            if apply_license(conn,pmcid,outcome)=='license_ok':
                                accepted.append(next(r for r in chunk if r['pmcid']==pmcid))
                        conn.commit()
                    for row in accepted:
                        reason = stop_reason()
                        if reason:
                            with client._lock:
                                for remaining in accepted:
                                    if remaining['pmcid'] not in submitted:
                                        queue_state.record(conn,remaining['pmcid'],'relevance','deferred',reason)
                                for remaining in candidates[start+20:]:
                                    queue_state.record(conn,remaining['pmcid'],'license','deferred',reason)
                                conn.commit()
                            return
                        yield request(row)

        completed = process_relevance(conn,client,feed(),submitted)
        _save_queue_report(conn,args,'queue_resume',backup_path=backup,
                           checkpoints=completed,submitted=len(submitted),spent_usd=client.spent_usd)
        return 1 if completed['errors'] else 0
    except BaseException as exc:
        for row in _queue_rows(conn,args,['license_ok','candidate']):
            state = conn.execute('SELECT status FROM article_queue_state WHERE pmcid=? AND stage=?',
                                 (row['pmcid'],'relevance' if row['status']=='license_ok' else 'license')).fetchone()
            if state is None or state['status'] in ('queued','running'):
                queue_state.record(conn,row['pmcid'],'relevance' if row['status']=='license_ok' else 'license',
                                   'deferred','interruption' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'feed_error',
                                   error=str(exc))
        conn.commit()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Counts (reports/stage2_counts.json)
# ---------------------------------------------------------------------------
def db_funnel_counts(conn) -> dict[str, dict]:
    """Per-disease license/relevant counts derived from the DB (resumable)."""
    out = {
        k: {"after_license_filter": 0, "relevant": 0} for k in diseases.DISEASE_KEYS
    }
    for row in conn.execute(
        "SELECT status, primary_disease_keys_json FROM articles"
    ):
        keys = db.from_json(row["primary_disease_keys_json"], []) or []
        for key in keys:
            if key not in out:
                continue
            out[key]["after_license_filter"] += row["status"] in _LICENSE_PASSED
            out[key]["relevant"] += row["status"] in _RELEVANT_STATUSES
    return out


def write_counts(
    conn,
    retrieval_counts: dict[str, dict],
    cap: int,
    billing_counters: dict | None = None,
) -> dict[str, dict]:
    """Merge this run's retrieval numbers with DB-derived funnel counts."""
    reports = config.reports_dir()
    path = reports / "stage2_counts.json"
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except ValueError:
            existing = {}
    funnel = db_funnel_counts(conn)
    out: dict[str, dict] = {}
    for key in diseases.DISEASE_KEYS:
        prev = existing.get(key) or {}
        retrieval = retrieval_counts.get(key) or {}
        entry = {
            "candidates": retrieval.get("candidates", prev.get("candidates", 0)),
            "after_type_filter": retrieval.get(
                "after_type_filter", prev.get("after_type_filter", 0)
            ),
            "selected": retrieval.get("selected", prev.get("selected", 0)),
            "after_license_filter": funnel[key]["after_license_filter"],
            "relevant": funnel[key]["relevant"],
            "cap": cap,
        }
        entry["over_cap"] = entry["relevant"] > cap
        out[key] = entry
    if billing_counters:
        out["_turbopuffer_billing"] = dict(billing_counters)
    reports.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    return out


# ---------------------------------------------------------------------------
# --recheck-title-rule: re-run P1 on articles the title rule passed
# ---------------------------------------------------------------------------
def run_recheck_title_rule(args, conn) -> int:
    """P1 re-verdicts for relevance_reason='title_rule' articles.

    Skips retrieval and licensing entirely. `relevant` rows get the P1
    verdict applied (relevant/irrelevant, keys, reason); `parsed` rows (the
    E2E set) are only reported — their status and keys are left alone.
    """
    in_scope = set(
        diseases.DISEASE_KEYS if args.disease == "all" else [args.disease]
    )
    cap = getattr(args, "cap", None) or DEFAULT_CAP
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT pmcid, title, status, primary_disease_keys_json "
            "FROM articles WHERE relevance_reason = 'title_rule' "
            "AND status IN ('relevant', 'parsed')"
        )
    ]
    rows = [
        r
        for r in rows
        if set(db.from_json(r["primary_disease_keys_json"], []) or []) & in_scope
    ]
    if args.pmcids:
        wanted = set(args.pmcids)
        rows = [r for r in rows if r["pmcid"] in wanted]
    rows.sort(key=lambda r: r["pmcid"])
    if args.limit:
        rows = rows[: args.limit]
    n_parsed = sum(r["status"] == "parsed" for r in rows)
    print(
        f"recheck-title-rule: {len(rows)} articles "
        f"({n_parsed} parsed → report only)"
    )
    if args.dry_run or not rows:
        return 0

    ns = _make_retriever().ns_pmc
    abstracts = fetch_abstracts(ns, [r["pmcid"] for r in rows])
    client = llm.LLMClient(
        db_conn=conn,
        budget_usd=args.budget_usd,
        concurrency=config.VP_P1_CONCURRENCY,
    )
    items = [
        (r, _p1_request(r["title"] or "", abstracts.get(r["pmcid"], "")))
        for r in rows
    ]
    flipped = kept = deferred = errors = 0
    results = client.call_many(req for _, req in items)
    for row, result in zip(rows, results):
        pmcid = row["pmcid"]
        if result.error is not None:
            if isinstance(result.error, llm.BudgetExceeded):
                deferred += 1  # unchanged; rerun to resume
            else:
                errors += 1
                conn.execute(
                    "UPDATE articles SET error = ?, "
                    "updated_at = datetime('now') WHERE pmcid = ?",
                    (str(result.error), pmcid),
                )
            continue
        parsed = result.parsed or {}
        keys = [
            k
            for k in (parsed.get("primary_disease_keys") or [])
            if k in PILOT_KEYS
        ]
        if row["status"] == "parsed":
            # E2E articles: print the verdict, never rewrite the row.
            print(
                f"  {pmcid} [parsed, unchanged]: decision={parsed.get('decision')} "
                f"narrative={parsed.get('is_narrative_review')} keys={keys} "
                f"reason={parsed.get('reason')}"
            )
            continue
        if _p1_relevant(parsed):
            apply_relevance(
                conn, pmcid, "relevant", parsed.get("reason") or "p1_relevant", keys
            )
            kept += 1
        else:
            apply_relevance(
                conn,
                pmcid,
                "irrelevant",
                parsed.get("reason") or f"p1_{parsed.get('decision')}",
                None,
            )
            flipped += 1
    conn.commit()
    if deferred:
        print(
            f"LLM budget exhausted: {deferred} articles left at title_rule; "
            "rerun to resume."
        )
    totals = write_counts(conn, {}, cap)
    print(json.dumps(totals, indent=1))
    print(
        f"recheck: {kept} stayed relevant, {flipped} flipped to irrelevant, "
        f"{errors} errors, {deferred} budget-deferred, "
        f"spend=${client.spent_usd:.4f}"
    )
    return 0


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(args) -> int:
    if getattr(args, "dry_run", False):
        # Dry-run exits before opening the DB, building a retriever, or any
        # provider call: describe the planned scope only.
        keys = (
            list(diseases.DISEASE_KEYS)
            if args.disease == "all"
            else [args.disease]
        )
        print(
            f"dry-run: would run select for diseases={keys}, "
            f"limit={getattr(args, 'limit', None)}, "
            f"max_articles={getattr(args, 'max_articles', None)}, "
            f"budget_usd={getattr(args, 'budget_usd', None)}"
        )
        print("dry-run: no DB reads/writes, no retrieval, no LLM calls")
        return 0
    conn = db.init_db()
    if getattr(args, "recheck_title_rule", False):
        try:
            return run_recheck_title_rule(args, conn)
        finally:
            conn.close()

    disease_keys = (
        list(diseases.DISEASE_KEYS) if args.disease == "all" else [args.disease]
    )
    cap = getattr(args, "cap", None) or DEFAULT_CAP
    limit = getattr(args, "limit", None)
    diseases_data = diseases.load_diseases()
    visual_findings = _visual_findings_from_db(conn) or diseases.load_findings_vocab()

    retriever = _make_retriever()
    ns = retriever.ns_pmc
    embed_fn = getattr(retriever, "_embed_query", None)
    embed_many_fn = getattr(retriever, "embed_queries", None)

    # ------------------------------------------------------------------
    # 1-3. Retrieval -> RRF -> type filter, per disease.
    # ------------------------------------------------------------------
    retrieved: dict[str, dict] = {}  # pmcid -> {score, attrs, keys}
    retrieval_counts: dict[str, dict] = {}
    manifestation_records: list[dict] = []
    billing_counters = {
        "requests": 0,
        "queries": 0,
        "billable_logical_bytes_queried": 0,
        "billable_logical_bytes_returned": 0,
    }
    # Query specs are built first (DB reads). All diseases' jobs then run in
    # ONE flat _rank_jobs call: multi_query requests carry at most
    # MULTI_QUERY_BATCH subqueries (turbopuffer's 16-permit namespace budget),
    # so ~340 retrieval queries collapse into ~22 round trips. Disease-level
    # thread parallelism would only contend for the same permit budget.
    specs: dict[str, dict] = {}
    for key in disease_keys:
        synonyms = diseases_data[key]["synonyms"]
        visual_queries = visual_queries_for_disease(
            key,
            diseases_data,
            findings=visual_findings,
            coverage_counts=_stored_panel_counts(conn, key),
        )
        specs[key] = {"synonyms": synonyms, "visual_queries": visual_queries}
        print(
            f"[{key}] {len(synonyms)} synonyms x3 buckets + "
            f"{len(visual_queries)} visual passage queries ..."
        )

    # Embeddings stay batched per disease: one provider failure then costs
    # only that disease's dense-ANN path (per-item fallback still applies).
    flat_jobs: list[tuple[list, int]] = []
    flat_contexts: list[dict] = []
    disease_slices: dict[str, tuple[int, list, list]] = {}
    for key in disease_keys:
        synonyms = specs[key]["synonyms"]
        embeddings = _embed_synonyms(synonyms, embed_fn, embed_many_fn)
        for synonym, embedding in zip(synonyms, embeddings):
            if embedding is None:
                logger.info("no embedding for %r; dense ANN skipped", synonym)
        jobs, contexts = _job_specs(
            synonyms, embeddings, specs[key]["visual_queries"], disease_key=key
        )
        disease_slices[key] = (len(flat_jobs), jobs, contexts)
        flat_jobs.extend(jobs)
        flat_contexts.extend(contexts)

    all_rows = _rank_jobs(ns, flat_jobs, flat_contexts, billing_counters)

    license_order: dict[str, list[str]] = {}
    for key in disease_keys:
        start, jobs, contexts = disease_slices[key]
        articles = _fuse_rows(jobs, contexts, all_rows[start : start + len(jobs)])
        # Every type-passed article is shortlisted; the license pass below
        # admits candidates in finding-lane priority order until the disease
        # reaches ``limit`` license-passing articles, so the cap applies
        # after the license filter rather than before it.
        ranked, n_type_passed, selected_manifestations = shortlist_articles(
            articles, None, config.VP_MANIFESTATION_QUOTA
        )
        license_order[key] = license_priority_order(ranked)
        manifestation_records.extend(
            {**candidate, "disease_key": key}
            for candidate in selected_manifestations
        )
        for pmcid, info in ranked:
            attrs = info["attrs"]
            entry = retrieved.get(pmcid)
            if entry is None:
                retrieved[pmcid] = {
                    "score": info["score"],
                    "attrs": attrs,
                    "keys": {key},
                    "matched_passages": list(info.get("matched_passages") or []),
                }
            else:
                entry["keys"].add(key)
                entry["matched_passages"] = _merge_evidence(
                    entry.get("matched_passages", []),
                    info.get("matched_passages", []),
                )
                if info["score"] > entry["score"]:
                    entry["score"] = info["score"]
                    entry["attrs"] = attrs
        retrieval_counts[key] = {
            "candidates": len(articles),
            "after_type_filter": n_type_passed,
            # Refined to the actual license-passing count after the license
            # pass; the dry-run path reports the license target instead.
            "selected": min(limit, n_type_passed) if limit else n_type_passed,
        }
        print(
            f"[{key}] {len(articles)} candidates, {n_type_passed} pass type filter; "
            f"license target: {limit or n_type_passed} license-passing"
        )

    # ------------------------------------------------------------------
    # 4. Insert candidates (INSERT OR IGNORE; resumable).
    # ------------------------------------------------------------------
    # Hydrate rich metadata only after the disease-level cap and type filter
    # have produced the actual article shortlist. Reuse unique PMCIDs across
    # diseases and carry abstracts forward to P1.
    shortlisted_ids = list(dict.fromkeys(retrieved))
    metadata = hydrate_metadata(ns, shortlisted_ids, billing_counters)
    for pmcid, info in retrieved.items():
        info["attrs"] = {**info.get("attrs", {}), **metadata.get(pmcid, {})}
    for pmcid, info in retrieved.items():
        upsert_candidate(
            conn, pmcid, info["attrs"], info["score"], info["keys"],
            info.get("matched_passages"),
        )
    persisted = upsert_manifestation_candidates(conn, manifestation_records)
    if persisted:
        print(f"manifestation candidates: persisted {persisted} per-finding ranks")
    conn.commit()

    client = llm.LLMClient(
        db_conn=conn,
        budget_usd=args.budget_usd,
        dry_run=False,
        concurrency=config.VP_P1_CONCURRENCY,
    )
    in_scope = set(disease_keys)

    def _in_scope(row) -> bool:
        keys = set(db.from_json(row["primary_disease_keys_json"], []) or [])
        return bool(keys & in_scope)

    license_pool = None
    try:
        # ------------------------------------------------------------------
        # 5-6. License and relevance run overlapped: the license pool applies
        # outcomes on this thread while each ``license_ok`` result streams a
        # P1 request into ``client.iter_many`` — LLM calls for cleared
        # articles start while later licenses are still being fetched.
        #
        # The license filter precedes the ``--limit`` cap: each disease's
        # candidates are checked in license_priority_order until ``limit``
        # of its shortlisted articles are license-passing, so rejections
        # release their slot to the next candidate instead of shrinking the
        # pool that reaches P1.
        # ------------------------------------------------------------------
        status_map = {
            row["pmcid"]: row["status"]
            for row in conn.execute("SELECT pmcid, status FROM articles")
        }
        pending_license = {
            row["pmcid"]
            for row in conn.execute(
                "SELECT pmcid, primary_disease_keys_json FROM articles "
                "WHERE status = 'candidate'"
            )
            if _in_scope(row)
        }
        with client._lock:
            for pmcid in pending_license:
                queue_state.record(conn, pmcid, 'license', 'queued', 'awaiting_license')
            for row in conn.execute("SELECT pmcid FROM article_queue_state WHERE stage='relevance' AND status='running'"):
                queue_state.record(conn, row['pmcid'], 'relevance', 'deferred', 'previous_run_incomplete')
            conn.commit()
        if pending_license:
            target = (
                f"; target: {limit} license-passing per disease" if limit else ""
            )
            print(f"license: {len(pending_license)} candidate articles{target}")

        # License-passing shortlist members per disease — the quota bounding
        # each disease's license expansion under a finite --limit.
        license_ok_counts = {
            key: sum(
                status_map.get(pmcid) in _LICENSE_PASSED
                for pmcid in license_order[key]
            )
            for key in disease_keys
        }
        license_sources: dict[str, int] = {"epmc": 0, "s3_fallback": 0}

        # Rows already license_ok from earlier runs are eligible for P1
        # immediately, before this pass's licensing starts.
        rows = [
            row
            for row in conn.execute(
                "SELECT pmcid, title, primary_disease_keys_json FROM articles "
                "WHERE status = 'license_ok'"
            )
            if _in_scope(row)
        ]
        # Abstracts: held in memory from this run's retrieval when possible,
        # else re-queried from tpuf in parallel batches.
        abstracts: dict[str, str] = {}
        missing = []
        for row in rows:
            attrs = (retrieved.get(row["pmcid"]) or {}).get("attrs") or {}
            if attrs.get("abstract") is not None:
                abstracts[row["pmcid"]] = str(attrs["abstract"])
            else:
                missing.append(row["pmcid"])
        if missing:
            print(f"abstracts: re-querying {len(missing)} via tpuf")
            abstracts.update(fetch_abstracts(ns, missing))

        def _title_for(pmcid: str) -> str:
            attrs = (retrieved.get(pmcid) or {}).get("attrs") or {}
            if attrs.get("title"):
                return str(attrs["title"])
            row = conn.execute(
                "SELECT title FROM articles WHERE pmcid = ?", (pmcid,)
            ).fetchone()
            return str(row["title"] or "") if row else ""

        def _abstract_for(pmcid: str) -> str:
            cached = abstracts.get(pmcid)
            if cached is not None:
                return cached
            attrs = (retrieved.get(pmcid) or {}).get("attrs") or {}
            if attrs.get("abstract") is not None:
                abstract = str(attrs["abstract"])
            else:
                abstract = abstract_for(ns, pmcid)
            abstracts[pmcid] = abstract
            return abstract

        # License fetches are submitted in bounded chunks so a disease stops
        # expanding once its license-passing quota is met; ``_p1_feed``
        # applies each chunk's outcomes before pulling the next, keeping
        # license work and P1 calls overlapped.
        if pending_license:
            license_pool = ThreadPoolExecutor(
                max_workers=config.VP_FETCH_CONCURRENCY
            )

        def _license_iter():
            """Yield chunks of (pmcid, outcome) in license-priority order.

            Each disease's queue is built when its turn starts, so articles
            already resolved while licensing an earlier disease are skipped;
            in-scope candidates no current shortlist ranked run last.
            """
            if license_pool is None:
                return
            license_chunk = _license_chunk()
            queued: set[str] = set()
            for key in disease_keys:
                queue = [
                    pmcid
                    for pmcid in license_order[key]
                    if status_map.get(pmcid) == "candidate" and pmcid not in queued
                ]
                queued.update(queue)
                checked = 0
                for offset in range(0, len(queue), license_chunk):
                    stop_reason = None
                    with client._lock:
                        if args.budget_usd is not None and client.spent_usd >= args.budget_usd:
                            stop_reason = 'budget'
                        elif checked >= getattr(args, 'max_articles', 6000):
                            stop_reason = 'safety_limit'
                        elif limit is not None and license_ok_counts[key] >= limit and not needs_manifestation_licenses(conn, key):
                            stop_reason = 'quota'
                        if stop_reason:
                            for pmcid in queue[offset:]:
                                queue_state.record(conn, pmcid, 'license', 'deferred', stop_reason,
                                                   disease_key=key, license_target=limit)
                            conn.commit()
                    if stop_reason:
                        break
                    checked += len(queue[offset : offset + license_chunk])
                    yield join_licenses(
                        queue[offset : offset + license_chunk], license_pool
                    )
            tail = sorted(pending_license - queued)
            for offset in range(0, len(tail), license_chunk):
                with client._lock:
                    stopped = args.budget_usd is not None and client.spent_usd >= args.budget_usd
                    if stopped:
                        for pmcid in tail[offset:]:
                            queue_state.record(conn, pmcid, 'license', 'deferred', 'budget')
                        conn.commit()
                if stopped:
                    break
                yield join_licenses(
                    tail[offset : offset + license_chunk], license_pool
                )

        submitted: list[str] = []  # BatchResult.index -> pmcid

        def _p1_feed():
            """Yield P1 requests: pre-cleared rows first, then each article
            as its license chunk lands. License DB writes happen here, on
            the calling thread."""
            for row in rows:
                with client._lock:
                    if args.budget_usd is not None and client.spent_usd >= args.budget_usd:
                        for remaining in rows:
                            if remaining['pmcid'] not in submitted:
                                queue_state.record(conn, remaining['pmcid'], 'relevance', 'deferred', 'budget')
                        for remaining in pending_license:
                            queue_state.record(conn, remaining, 'license', 'deferred', 'budget')
                        conn.commit()
                        return
                    queue_state.record(conn, row['pmcid'], 'relevance', 'running', 'submitted')
                    conn.commit()
                submitted.append(row["pmcid"])
                yield _p1_request(row["title"] or "", abstracts.get(row["pmcid"], ""))
            for chunk in _license_iter():
                for pmcid, outcome in chunk:
                    license_sources[outcome.get("license_source") or "epmc"] += 1
                    with client._lock:
                        status = apply_license(conn, pmcid, outcome)
                        if status == 'license_ok':
                            queue_state.record(conn, pmcid, 'relevance', 'running', 'submitted')
                        conn.commit()
                    status_map[pmcid] = status
                    if status != "license_ok":
                        continue
                    for key in (retrieved.get(pmcid) or {}).get("keys") or ():
                        license_ok_counts[key] += 1
                    submitted.append(pmcid)
                    yield _p1_request(_title_for(pmcid), _abstract_for(pmcid))
                with client._lock:
                    conn.commit()

        budget_hits = 0
        print(f"relevance: {len(rows)} pre-cleared + licensed articles through P1")
        completed = process_relevance(conn, client, _p1_feed(), submitted)
        budget_hits = completed['budget']
        print(f"relevance checkpoints: {completed}")
        conn.commit()
        if any(license_sources.values()):
            print(
                "license source: "
                f"epmc={license_sources['epmc']} s3_fallback={license_sources['s3_fallback']}"
            )
        for key in disease_keys:
            retrieval_counts[key]["selected"] = license_ok_counts[key]
            print(
                f"[{key}] {license_ok_counts[key]} articles pass license filter"
                + (f" (target {limit})" if limit else "")
            )
        if budget_hits:
            print(
                f"LLM budget exhausted: {budget_hits} articles left at "
                "license_ok; rerun to resume."
            )
    except llm.BudgetExceeded:
        print(
            f"LLM budget exhausted (${client.spent_usd:.4f} spent); "
            "stopping cleanly. Rerun to resume."
        )
        conn.commit()
    finally:
        if license_pool is not None:
            license_pool.shutdown(wait=True)
        # A hard interruption leaves a durable reason rather than silently
        # appearing as an active queue. Completed outcomes are never reset.
        for pmcid in locals().get('pending_license', set()) | set(locals().get('submitted', [])):
            conn.execute("UPDATE article_queue_state SET status='deferred',reason='interruption',updated_at=datetime('now') WHERE pmcid=? AND status IN ('running','queued')", (pmcid,))
        conn.commit()

    # ------------------------------------------------------------------
    # 7. Counts checkpoint.
    # ------------------------------------------------------------------
    totals = write_counts(conn, retrieval_counts, cap, billing_counters)
    print(json.dumps(totals, indent=1))
    print(f"live LLM spend this run: ${client.spent_usd:.4f}")
    print(f"Turbopuffer billing: {json.dumps(billing_counters, sort_keys=True)}")
    for key in disease_keys:
        entry = totals.get(key) or {}
        if entry.get("over_cap"):
            print(
                f"[{key}] {entry.get('relevant')} relevant articles exceed the "
                f"legacy report threshold of {cap}; parsing uses yield batches."
            )
    conn.close()
    return 0
