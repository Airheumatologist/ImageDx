"""Stage 2: article selection from the turbopuffer PMC namespace.

Per disease, per synonym: title BM25 (top_k 200), page_content BM25
(top_k 300) and dense ANN (top_k 300), plus a bounded set of visual
finding/modality page_content BM25 queries, under the review filter
``Or(publication_type Contains "Review", article_type Eq "review-article")``
AND ``has_full_text = true``. Rows are chunk-level: each ranked list is
collapsed to per-pmcid best rank and all lists for a disease are fused with
plain RRF (k=60) into ``retrieval_score``.

Publication-type exclusions are applied in Python (``passes_type_filter``);
licenses join via ``pmc.get_license`` on a thread pool; relevance is the
title rule, else prompt P1 on ``Title: ... / Abstract: ...``.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from . import config, db, diseases, llm, pmc
from .prompts import P1

logger = logging.getLogger(__name__)

PILOT_KEYS = set(diseases.DISEASE_KEYS)

TITLE_TOP_K = 200
CONTENT_TOP_K = 300
DENSE_TOP_K = 300
VISUAL_QUERY_TOP_K = 150
MAX_EVIDENCE_PER_ARTICLE = 8
MAX_EVIDENCE_TEXT_CHARS = 1200
VISUAL_QUERY_RRF_WEIGHT = 12.0
RRF_K = 60
DEFAULT_CAP = 150
ABSTRACT_MAX_CHARS = 4000

ATTRIBUTES = [
    "pmcid",
    "pmid",
    "doi",
    "title",
    "abstract",
    "journal",
    "year",
    "country",
    "publication_type",
    "article_type",
    "page_content",
    "section_title",
    "section_type",
]

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
    "eye": "clinical photograph",
}
_VISUAL_SECTION_CUE = re.compile(
    r"\b(clinical|physical examination|cutaneous|dermatolog|imaging|radiolog|"
    r"mri|computed tomography|histolog|patholog|capillaroscop|photograph)\w*\b",
    re.I,
)
_VISUAL_PASSAGE_CUE = re.compile(
    r"\b(photo(?:graph)?s?|images?|figure\s+\d|radiograph|x[ -]?ray|"
    r"mri|magnetic resonance|ct scan|computed tomography|ultrasound|"
    r"histolog|biopsy|histopatholog|micrograph|capillaroscop|rash|papules?|"
    r"plaques?|erythema|ulcer|erosion|lesion|sacroiliitis|bone marrow edema)\b",
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
    """Build a deterministic, small set of finding/modality retrieval queries.

    Findings are selected round-robin across image-bearing categories so a
    large skin vocabulary cannot crowd out imaging or pathology. Each result
    retains its query terms for downstream evidence and ranking.
    """
    if max_queries is None:
        max_queries = config.VP_VISUAL_QUERY_CAP
    if disease_key not in PILOT_KEYS or max_queries <= 0:
        return []
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
    while len(selected) < max_queries:
        added = False
        for category in categories:
            bucket = by_category.get(category, [])
            if len(bucket) > sum(x["category"] == category for x in selected):
                selected.append(bucket[sum(x["category"] == category for x in selected)])
                added = True
                if len(selected) >= max_queries:
                    break
        if not added:
            break

    subtypes = disease.get("subtypes", [])
    out = []
    for item in selected:
        finding = str(item.get("label") or item.get("finding_key") or "")
        modality = _CATEGORY_MODALITY[item["category"]]
        finding_text = " ".join(
            [finding, str(item.get("finding_key") or ""), *item.get("synonyms", [])]
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
        query = " ".join(
            part for part in (disease["name"], subtype, finding, modality) if part
        )
        out.append({
            "query": query,
            "finding": finding,
            "modality": modality,
            "category": item["category"],
        })
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
    """Count stored panels per approved finding for coverage-aware queries."""
    counts: dict[str, int] = {}
    for row in conn.execute(
        "SELECT findings_json FROM panels WHERE disease_key = ?", (disease_key,)
    ):
        for finding in db.from_json(row["findings_json"], []) or []:
            key = finding.get("finding_key") if isinstance(finding, dict) else finding
            if key:
                counts[str(key)] = counts.get(str(key), 0) + 1
    return counts


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


def _ns_query(ns, rank_by, filters, top_k) -> list[dict]:
    result = ns.query(
        rank_by=rank_by,
        filters=filters,
        top_k=top_k,
        include_attributes=ATTRIBUTES,
    )
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


def _rank_query(ns, rank_by, top_k) -> list[dict]:
    """One ranked list for one bucket, under the review + full-text filter."""
    global _or_filter_supported
    if _or_filter_supported is not False:
        try:
            rows = _ns_query(ns, rank_by, _review_filters(), top_k)
            _or_filter_supported = True
            return rows
        except Exception as exc:  # noqa: BLE001 - detect filter support once
            logger.warning(
                "Or review filter rejected (%s); falling back to two queries", exc
            )
            _or_filter_supported = False
    filters_a, filters_b = _split_review_filters()
    rows_a = _ns_query(ns, rank_by, filters_a, top_k)
    rows_b = _ns_query(ns, rank_by, filters_b, top_k)
    return _merge_ranked(rows_a, rows_b)


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
) -> dict[str, dict]:
    """Fuse ranked lists and preserve the best passage evidence per PMC article.

    The independent turbopuffer queries (per-synonym buckets + per-visual
    lookups) run on a thread pool; ``ns`` is shared because the Stainless
    client wraps a thread-safe ``httpx.Client`` and ``Namespace.query`` is a
    stateless POST. Ranked lists are replayed below in exactly the order the
    sequential code appended them, so RRF fusion, ``attrs`` first-seen wins,
    and evidence selection are identical to the serial implementation.
    """
    synonyms = list(synonyms or [])
    embeddings = _embed_synonyms(synonyms, embed_fn, embed_many_fn)
    for synonym, embedding in zip(synonyms, embeddings):
        if embedding is None:
            logger.info("no embedding for %r; dense ANN skipped", synonym)

    # Build the query jobs in issue order, keeping the context each job's
    # rows need for the downstream merge (query kind, evidence metadata).
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
                "modality": str(spec.get("modality") or ""),
                "finding": str(spec.get("finding") or ""),
            }
        )

    def _run(job):
        rank_by, top_k = job
        return _rank_query(ns, rank_by, top_k)

    workers = min(len(jobs), max(1, int(config.VP_CONCURRENCY)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            all_rows = list(pool.map(_run, jobs))
    else:
        all_rows = [_run(job) for job in jobs]

    ranked_lists: list[list[str]] = []
    ranked_weights: list[float] = []
    attrs: dict[str, dict] = {}
    evidence: dict[str, list[dict]] = {}
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
            passage = str(row.get("page_content") or "").strip()
            if not pmcid or not passage or not _is_visual_passage(row):
                continue
            evidence.setdefault(pmcid, []).append({
                "query": context["query"],
                "query_kind": context["query_kind"],
                "text": passage[:MAX_EVIDENCE_TEXT_CHARS],
                "section": str(row.get("section_title") or ""),
                "section_type": str(row.get("section_type") or ""),
                "modality": context["modality"],
                "finding": context["finding"],
                "rank": rank,
                "score": 1.0 / (RRF_K + rank),
            })
    scores = rrf_scores(ranked_lists, weights=ranked_weights)
    out = {}
    for pmcid, score in scores.items():
        out[pmcid] = {
            "score": score,
            "attrs": attrs.get(pmcid, {}),
            "matched_passages": _select_evidence(evidence.get(pmcid, [])),
        }
    return out


def _select_evidence(evidence: list[dict]) -> list[dict]:
    """Keep a bounded mix, reserving at least one slot for visual query hits."""
    best_by_passage: dict[tuple[str, str], dict] = {}
    for item in evidence:
        identity = (str(item.get("section") or ""), str(item.get("text") or ""))
        if not identity[1]:
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
        selected_ids = {
            (str(item.get("section") or ""), str(item.get("text") or ""))
            for item in selected
        }
        selected.extend(
            item for item in ranked
            if (str(item.get("section") or ""), str(item.get("text") or "")) not in selected_ids
        )
    return selected[:MAX_EVIDENCE_PER_ARTICLE]


def abstract_for(ns, pmcid: str) -> str:
    """Re-fetch an abstract by pmcid (resume path; not held in memory)."""
    try:
        result = ns.query(
            filters=["pmcid", "Eq", pmcid],
            top_k=1,
            include_attributes=["abstract"],
        )
        rows = getattr(result, "rows", [])
        return str(dict(rows[0]).get("abstract") or "") if rows else ""
    except Exception:  # noqa: BLE001 - best-effort metadata
        return ""


def fetch_abstracts(ns, pmcids: list[str]) -> dict[str, str]:
    """abstract_for on a thread pool of VP_CONCURRENCY workers."""
    if not pmcids:
        return {}
    with ThreadPoolExecutor(max_workers=config.VP_CONCURRENCY) as pool:
        values = pool.map(lambda p: abstract_for(ns, p), pmcids)
    return dict(zip(pmcids, values))


def _merge_evidence(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Merge cross-disease evidence by passage while keeping the strongest hit."""
    merged: dict[tuple[str, str], dict] = {}
    for item in [*(existing or []), *(incoming or [])]:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        identity = (str(item.get("section") or ""), text)
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


def join_license(pmcid: str) -> tuple[str, dict]:
    """Fetch one license (no DB access — safe on the worker pool)."""
    try:
        lic = pmc.get_license(pmcid)
    except Exception as exc:  # noqa: BLE001 - any PMC error rejects the article
        return pmcid, {
            "status": "license_rejected",
            "error": str(exc),
        }
    allows = pmc.license_allows(lic.code)
    return pmcid, {
        "status": "license_ok" if allows else "license_rejected",
        "license_code": lic.code,
        "license_url": lic.url,
        "oa_subset": lic.oa_subset,
        # C3 hints (W4b): persisted on the article row so parse can use the
        # hinted get_article_bundle path and skip re-listing the S3 dir.
        "s3_prefix": lic.prefix,
        "media_files": lic.media_files,
        "error": None,
    }


def apply_license(conn, pmcid: str, outcome: dict) -> str:
    fields = {
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
        "relevance_decision": "relevant" if status == "relevant" else "irrelevant",
        "relevance_reason": reason,
    }
    if keys:
        fields["primary_disease_keys_json"] = db.to_json(sorted(keys))
    db.set_status(conn, "articles", pmcid, status, **fields)


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
            "after_license_filter": funnel[key]["after_license_filter"],
            "relevant": funnel[key]["relevant"],
            "cap": cap,
        }
        entry["over_cap"] = entry["relevant"] > cap
        out[key] = entry
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
    for key in disease_keys:
        synonyms = diseases_data[key]["synonyms"]
        visual_queries = visual_queries_for_disease(
            key,
            diseases_data,
            findings=visual_findings,
            coverage_counts=_stored_panel_counts(conn, key),
        )
        print(
            f"[{key}] {len(synonyms)} synonyms x3 buckets + "
            f"{len(visual_queries)} visual passage queries ..."
        )
        articles = retrieve_for_disease(
            ns,
            embed_fn,
            synonyms,
            visual_queries,
            disease_key=key,
            embed_many_fn=embed_many_fn,
        )
        # --limit caps the candidate set per disease (top N by score).
        ranked = sorted(
            articles.items(), key=lambda kv: kv[1]["score"], reverse=True
        )
        if limit is not None:
            ranked = ranked[:limit]
        type_passed = 0
        for pmcid, info in ranked:
            attrs = info["attrs"]
            types = attrs.get("publication_type") or []
            if isinstance(types, str):
                types = [types]
            if not passes_type_filter(types, attrs.get("article_type")):
                continue
            type_passed += 1
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
            "candidates": len(ranked),
            "after_type_filter": type_passed,
        }
        print(
            f"[{key}] {len(ranked)} candidates, {type_passed} after type filter"
        )

    if args.dry_run:
        print(json.dumps(retrieval_counts, indent=1))
        print("dry-run: no DB writes, no LLM calls")
        conn.close()
        return 0

    # ------------------------------------------------------------------
    # 4. Insert candidates (INSERT OR IGNORE; resumable).
    # ------------------------------------------------------------------
    for pmcid, info in retrieved.items():
        upsert_candidate(
            conn, pmcid, info["attrs"], info["score"], info["keys"],
            info.get("matched_passages"),
        )
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

    try:
        # ------------------------------------------------------------------
        # 5. License for every in-scope `candidate` row.
        # ------------------------------------------------------------------
        pending_license = [
            row["pmcid"]
            for row in conn.execute(
                "SELECT pmcid, primary_disease_keys_json FROM articles "
                "WHERE status = 'candidate'"
            )
            if _in_scope(row)
        ]
        if pending_license:
            print(f"license: {len(pending_license)} candidate articles")
            applied = 0
            with ThreadPoolExecutor(
                max_workers=config.VP_FETCH_CONCURRENCY
            ) as pool:
                for pmcid, outcome in pool.map(join_license, pending_license):
                    apply_license(conn, pmcid, outcome)
                    applied += 1
                    if applied % 200 == 0:
                        conn.commit()
            conn.commit()

        # ------------------------------------------------------------------
        # 6. Relevance for in-scope `license_ok` rows: every article goes
        #    through P1 (the title rule no longer auto-passes).
        # ------------------------------------------------------------------
        rows = [
            row
            for row in conn.execute(
                "SELECT pmcid, title, primary_disease_keys_json FROM articles "
                "WHERE status = 'license_ok'"
            )
            if _in_scope(row)
        ]
        # Abstracts: held in memory from this run's retrieval when possible,
        # else re-queried from tpuf on a thread pool.
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
        p1_items = [
            (row["pmcid"], _p1_request(row["title"] or "", abstracts.get(row["pmcid"], "")))
            for row in rows
        ]

        budget_hits = 0
        if p1_items:
            print(f"relevance: {len(p1_items)} articles through P1")
            results = client.call_many(req for _, req in p1_items)
            for (pmcid, _req), result in zip(p1_items, results):
                if result.error is not None:
                    if isinstance(result.error, llm.BudgetExceeded):
                        budget_hits += 1  # stays license_ok; rerun to resume
                    else:
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
                if _p1_relevant(parsed):
                    apply_relevance(
                        conn,
                        pmcid,
                        "relevant",
                        parsed.get("reason") or "p1_relevant",
                        keys,
                    )
                else:
                    apply_relevance(
                        conn,
                        pmcid,
                        "irrelevant",
                        parsed.get("reason") or f"p1_{parsed.get('decision')}",
                        None,
                    )
        conn.commit()
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

    # ------------------------------------------------------------------
    # 7. Counts checkpoint.
    # ------------------------------------------------------------------
    totals = write_counts(conn, retrieval_counts, cap)
    print(json.dumps(totals, indent=1))
    print(f"live LLM spend this run: ${client.spent_usd:.4f}")
    for key in disease_keys:
        entry = totals.get(key) or {}
        if entry.get("over_cap"):
            print(
                f"[{key}] {entry.get('relevant')} relevant articles exceed the "
                f"legacy report threshold of {cap}; parsing uses yield batches."
            )
    conn.close()
    return 0
