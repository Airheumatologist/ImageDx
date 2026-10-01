"""Exact bounded BM25 requests and ledger identities for pair replenishment."""

from __future__ import annotations

import hashlib
import json

from . import config, db, pair_terms

REVIEW_FILTERS = [
    "And",
    [
        [
            "Or",
            [
                ["publication_type", "Contains", "Review"],
                ["article_type", "Eq", "review-article"],
            ],
        ],
        ["has_full_text", "Eq", True],
    ],
]
PAIR_PENDING_SQL = (
    "SELECT a.* FROM articles a JOIN manifestation_candidates mc ON mc.pmcid=a.pmcid "
    "WHERE mc.disease_key=? AND mc.finding_key=? "
    "AND mc.provenance_status='explicit' AND mc.provenance_disease_key=mc.disease_key "
    "AND a.status IN ('candidate','license_ok','relevant') "
    "AND mc.status IN ('pending','selected','covered') "
    "AND (a.status!='relevant' "
    "OR NOT EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json)) "
    "OR EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) WHERE value=?)) "
    "ORDER BY "
    "CASE a.status WHEN 'relevant' THEN 0 WHEN 'license_ok' THEN 1 ELSE 2 END, "
    "COALESCE(mc.best_rank,2147483647),a.pmcid"
)


def pending_candidates(conn, disease_key: str, finding_key: str) -> list[dict]:
    return [dict(row) for row in conn.execute(PAIR_PENDING_SQL, (disease_key, finding_key, disease_key))]


def attempt_specs(conn, disease_key: str, finding_key: str, round_no: int) -> list[dict]:
    """Completed requests count toward the six-variant round limit on resume."""
    if round_no < 1 or round_no > len(config.PAIR_SEARCH_DEPTHS):
        return []
    row = conn.execute(
        "SELECT * FROM findings_vocab WHERE finding_key=? AND approved=1", (finding_key,),
    ).fetchone()
    if row is None or disease_key not in db.from_json(row["disease_keys_json"], []):
        raise ValueError("Replenishment requires an approved disease/finding pair")
    finding = dict(row)
    finding["synonyms"] = db.from_json(row["synonyms_json"], []) or []
    disease = conn.execute("SELECT * FROM diseases WHERE disease_key=?", (disease_key,)).fetchone()
    if disease is None:
        raise ValueError("Replenishment requires a known disease")
    disease = dict(disease)
    disease["synonyms"] = db.from_json(disease["synonyms_json"], []) or []
    queries = pair_terms.search_queries(disease_key, finding, disease)
    specs = []
    depth = config.PAIR_SEARCH_DEPTHS[round_no - 1]
    filters_json = json.dumps(REVIEW_FILTERS, separators=(",", ":"), sort_keys=True)
    # A new round deliberately retries canonical queries at a deeper top-k.
    # It is a different strategy, not cursor pagination or a repeated attempt.
    for query in queries[:config.PAIR_SEARCH_VARIANTS_PER_ROUND]:
        identity = json.dumps(
            {"query": pair_terms.normalize_query(query), "filters": REVIEW_FILTERS},
            separators=(",", ":"), sort_keys=True,
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        prior = conn.execute(
            "SELECT * FROM pair_search_attempts WHERE disease_key=? AND finding_key=? "
            "AND policy_version=? AND round_no=? AND query_filter_hash=? AND depth=?",
            (disease_key, finding_key, config.PAIR_SEARCH_POLICY_VERSION, round_no, digest, depth),
        ).fetchone()
        specs.append({
            "disease_key": disease_key, "finding_key": finding_key,
            "policy_version": config.PAIR_SEARCH_POLICY_VERSION,
            "round_no": round_no, "query": query, "query_filter_hash": digest,
            "filters_json": filters_json, "depth": depth,
            "rank_by": ["page_content", "BM25", query],
            "filters": json.loads(filters_json),
            "include_attributes": [
                "pmcid", "publication_type", "article_type", "page_content",
                "section_title", "section_type",
            ],
            "prior_attempt": dict(prior) if prior is not None else None,
        })
    return specs


def next_round(conn, disease_key: str, finding_key: str) -> int | None:
    for round_no in range(1, len(config.PAIR_SEARCH_DEPTHS) + 1):
        specs = attempt_specs(conn, disease_key, finding_key, round_no)
        if any(
            not spec["prior_attempt"]
            or spec["prior_attempt"]["status"] != "completed"
            for spec in specs
        ):
            return round_no
    return None
