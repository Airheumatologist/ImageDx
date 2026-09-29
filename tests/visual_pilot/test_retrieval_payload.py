from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from src.visual_pilot import select_articles


class MultiNamespace:
    def __init__(self, result_factory=None):
        self.calls = []
        self.result_factory = result_factory or (lambda query, i: [])

    def multi_query(self, *, queries):
        self.calls.append(queries)
        results = [
            SimpleNamespace(rows=self.result_factory(query, i))
            for i, query in enumerate(queries)
        ]
        return SimpleNamespace(
            results=results,
            billing=SimpleNamespace(
                billable_logical_bytes_queried=123,
                billable_logical_bytes_returned=45,
            ),
        )


def _review_row(pmcid="PMC1", passage="Clinical photograph shows erythema"):
    return {
        "pmcid": pmcid,
        "publication_type": ["Review"],
        "article_type": "review-article",
        "page_content": passage,
        "section_title": "Clinical examination",
        "section_type": "body",
    }


def test_multi_query_projection_limits_order_and_finding_provenance(monkeypatch):
    monkeypatch.setattr(select_articles, "_or_filter_supported", None)
    ns = MultiNamespace(lambda query, i: [_review_row(f"PMC{i}")])
    counters = {}
    result = select_articles.retrieve_for_disease(
        ns,
        None,
        ["disease synonym"],
        [{
            "query": "disease rash clinical photograph",
            "finding_key": "erythema",
            "finding": "erythema",
            "modality": "clinical photograph",
        }],
        billing_counters=counters,
    )

    queries = ns.calls[0]
    assert [query["rank_by"][0] for query in queries] == [
        "title", "page_content", "page_content",
    ]
    assert [query["limit"]["total"] for query in queries] == [500, 750, 300]
    assert all(query["limit"]["per"] == {"attributes": ["pmcid"], "limit": 1}
               for query in queries)
    assert "abstract" not in queries[0]["include_attributes"]
    assert "page_content" not in queries[0]["include_attributes"]
    assert "page_content" in queries[1]["include_attributes"]
    assert "abstract" not in queries[1]["include_attributes"]
    assert list(result) == ["PMC0", "PMC1", "PMC2"]
    assert result["PMC2"]["manifestation_candidates"] == [{
        "finding_key": "erythema",
        "pmcid": "PMC2",
        "query": "disease rash clinical photograph",
        "best_rank": 1,
        "retrieval_score": 1 / (select_articles.RRF_K + 1),
    }]
    assert result["PMC2"]["matched_passages"][0]["finding_key"] == "erythema"
    assert counters == {
        "requests": 1,
        "queries": 3,
        "billable_logical_bytes_queried": 123,
        "billable_logical_bytes_returned": 45,
    }


class QueryOnlyNamespace:
    def __init__(self):
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        rows = []
        if kwargs["rank_by"][0] == "title":
            rows = [_review_row("PMC-title")]
        return SimpleNamespace(rows=rows)


def test_sequential_fallback_preserves_job_order_and_compact_projection(monkeypatch):
    monkeypatch.setattr(select_articles, "_or_filter_supported", None)
    ns = QueryOnlyNamespace()
    jobs = [(["title", "BM25", "first"], 2), (["page_content", "BM25", "second"], 3)]
    contexts = [{}, {}]
    rows = select_articles._rank_jobs(ns, jobs, contexts)
    assert [row[0].get("pmcid") if row else None for row in rows] == ["PMC-title", None]
    assert [call["rank_by"][2] for call in ns.calls] == ["first", "second"]
    assert all(call["limit"]["per"] == {"attributes": ["pmcid"], "limit": 1} for call in ns.calls)
    assert "abstract" not in ns.calls[0]["include_attributes"]
    assert "page_content" not in ns.calls[0]["include_attributes"]


def test_metadata_hydration_batches_unique_pmcs_and_candidate_upsert_keeps_best_rank():
    ns = MultiNamespace(lambda query, i: [{
        "pmcid": query["filters"][2], "title": "Review", "abstract": "A" * 5000,
    }])
    hydrated = select_articles.hydrate_metadata(ns, ["PMC1", "PMC1", "PMC2"])
    assert set(hydrated) == {"PMC1", "PMC2"}
    assert len(ns.calls) == 1
    assert len(ns.calls[0]) == 2
    assert all(query["include_attributes"] == select_articles.METADATA_ATTRIBUTES for query in ns.calls[0])

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE articles (pmcid TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO articles (pmcid) VALUES ('PMC1')")
    conn.execute("""CREATE TABLE manifestation_candidates (
        disease_key TEXT, finding_key TEXT, pmcid TEXT, query TEXT,
        best_rank INTEGER, retrieval_score REAL, status TEXT DEFAULT 'pending',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(disease_key, finding_key, pmcid))""")
    records = [{
        "disease_key": "ra", "finding_key": "erythema", "pmcid": "PMC1",
        "query": "better", "best_rank": 2, "retrieval_score": 1 / 62,
    }, {
        "disease_key": "ra", "finding_key": "erythema", "pmcid": "PMC1",
        "query": "worse", "best_rank": 4, "retrieval_score": 1 / 64,
    }]
    assert select_articles.upsert_manifestation_candidates(conn, records) == 2
    row = conn.execute(
        "SELECT finding_key, query, best_rank, status FROM manifestation_candidates"
    ).fetchone()
    assert row == ("erythema", "better", 2, "pending")


def test_weighted_rrf_is_deterministic_and_deduplicates_per_list():
    first = select_articles.rrf_scores([["PMC1", "PMC1", "PMC2"], ["PMC2"]], weights=[1, 12])
    second = select_articles.rrf_scores([["PMC1", "PMC1", "PMC2"], ["PMC2"]], weights=[1, 12])
    assert first == second
    assert first["PMC1"] == 1 / 61
    assert first["PMC2"] == 1 / 63 + 12 / 61


def _candidate(score, *, finding=None, rank=1, pubtype="Review"):
    lanes = []
    for finding_key in ([finding] if isinstance(finding, str) else finding or []):
        lanes.append({
            "finding_key": finding_key,
            "query": f"query {finding_key}",
            "best_rank": rank,
            "retrieval_score": 1 / (select_articles.RRF_K + rank),
        })
    return {
        "score": score,
        "attrs": {
            "publication_type": [pubtype],
            "article_type": "review-article" if pubtype == "Review" else "",
        },
        "manifestation_candidates": lanes,
    }


def test_finding_reservation_selects_low_global_rrf_article():
    articles = {
        "PMC-global": _candidate(0.9),
        "PMC-finding": _candidate(0.01, finding="rare_finding"),
    }
    selected, type_passed, candidates = select_articles.shortlist_articles(
        articles, limit=1, per_finding_quota=20
    )
    assert [pmcid for pmcid, _ in selected] == ["PMC-finding"]
    assert type_passed == 2
    assert [item["finding_key"] for item in candidates] == ["rare_finding"]


def test_shared_pmcid_across_lanes_is_selected_once_and_keeps_both_provenances():
    articles = {
        "PMC-shared": _candidate(0.8, finding=["lane_a", "lane_b"]),
        "PMC-a": _candidate(0.2, finding="lane_a", rank=2),
        "PMC-b": _candidate(0.1, finding="lane_b", rank=2),
    }
    selected, _, candidates = select_articles.shortlist_articles(
        articles, limit=2, per_finding_quota=1
    )
    ids = [pmcid for pmcid, _ in selected]
    assert len(ids) == len(set(ids)) == 2
    assert "PMC-shared" in ids
    shared_findings = {
        item["finding_key"] for item in candidates if item["pmcid"] == "PMC-shared"
    }
    assert shared_findings == {"lane_a", "lane_b"}


def test_tight_cap_allocates_one_candidate_per_lane_before_second_round():
    articles = {
        "PMC-a1": _candidate(0.9, finding="lane_a", rank=1),
        "PMC-a2": _candidate(0.8, finding="lane_a", rank=2),
        "PMC-b1": _candidate(0.7, finding="lane_b", rank=1),
        "PMC-b2": _candidate(0.6, finding="lane_b", rank=2),
        "PMC-c1": _candidate(0.5, finding="lane_c", rank=1),
    }
    selected, _, _ = select_articles.shortlist_articles(
        articles, limit=3, per_finding_quota=20
    )
    assert {pmcid for pmcid, _ in selected} == {"PMC-a1", "PMC-b1", "PMC-c1"}


def test_excluded_publication_types_do_not_consume_limit_and_none_keeps_all():
    articles = {
        "PMC-case": _candidate(1.0, pubtype="Case Reports"),
        "PMC-review-1": _candidate(0.5),
        "PMC-review-2": _candidate(0.4),
    }
    selected, type_passed, _ = select_articles.shortlist_articles(articles, limit=1)
    assert [pmcid for pmcid, _ in selected] == ["PMC-review-1"]
    assert type_passed == 2
    all_selected, all_passed, _ = select_articles.shortlist_articles(articles, limit=None)
    assert [pmcid for pmcid, _ in all_selected] == ["PMC-review-1", "PMC-review-2"]
    assert all_passed == 2


def test_license_priority_order_interleaves_lanes_then_rrf_tail():
    articles = {
        "PMC-a1": _candidate(0.9, finding="lane_a", rank=1),
        "PMC-b1": _candidate(0.8, finding="lane_b", rank=1),
        "PMC-a2": _candidate(0.7, finding="lane_a", rank=2),
        "PMC-plain": _candidate(0.6),
    }
    type_passed, _, _ = select_articles.shortlist_articles(articles, limit=None)
    order = select_articles.license_priority_order(type_passed)
    assert order == ["PMC-a1", "PMC-b1", "PMC-a2", "PMC-plain"]


def test_license_priority_order_dedupes_shared_lane_candidates():
    articles = {
        "PMC-shared": _candidate(0.9, finding=["lane_a", "lane_b"], rank=1),
        "PMC-b": _candidate(0.5, finding="lane_b", rank=2),
    }
    type_passed, _, _ = select_articles.shortlist_articles(articles, limit=None)
    order = select_articles.license_priority_order(type_passed)
    assert order == ["PMC-shared", "PMC-b"]
