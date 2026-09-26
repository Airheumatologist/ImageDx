"""W3 tests for src/visual_pilot/select_articles.py — stubbed namespace, no network."""

import argparse
import json

import pytest

from src.visual_pilot import db, diseases, llm, pmc, select_articles
from src.visual_pilot import config as vp_config
from src.visual_pilot.prompts import P1


# ---------------------------------------------------------------------------
# passes_type_filter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("pub_types", "article_type", "expected"),
    [
        (["Case Based Review"], "review-article", False),
        (["Study Protocol Systematic Review"], "review-article", False),
        (["3800", "Research Article", "Study Protocol Systematic Review"], "review-article", False),
        (["Immunology", "Mini Review"], "review-article", True),
        (["Review"], "research-article", True),
        (["Review Article"], "review-article", True),
        (["Review"], "letter", False),
        ([], "letter", False),
        (["Notes & Comments"], "review-article", False),
        (["Immunology"], "research-article", False),
        (["Systematic Review"], "review-article", False),
        (["Meta-Analysis", "Review"], "review-article", False),
        (["Randomized Controlled Trial", "Review"], "review-article", False),
        (["Clinical Trial", "Review"], "review-article", False),
        (["Review"], "case-report", False),
        (["Review"], "article-commentary", False),
        (["Review"], "editorial", False),
        (["Review"], "correction", False),
        (["Review"], "retraction", False),
        ([], "review-article", True),
        (["Review", "Erratum"], "review-article", False),
    ],
)
def test_passes_type_filter(pub_types, article_type, expected):
    assert select_articles.passes_type_filter(pub_types, article_type) is expected


# ---------------------------------------------------------------------------
# RRF chunk collapse
# ---------------------------------------------------------------------------
def test_rrf_uses_best_rank_per_pmcid():
    lists = [["A", "A", "B"], ["B", "A"], ["C"]]
    scores = select_articles.rrf_scores(lists, k=60)
    # A: rank 1 (list0, dup ignored) + rank 2 (list1); B: rank 3 + rank 1; C: rank 1
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["B"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["C"] == pytest.approx(1 / 61)


def test_chunk_collapse_one_article_per_pmcid():
    """Duplicate chunks of the same pmcid keep only the best rank."""
    lists = [["PMC1", "PMC1", "PMC1", "PMC2"]]
    scores = select_articles.rrf_scores(lists, k=60)
    assert scores["PMC1"] == pytest.approx(1 / 61)
    assert scores["PMC2"] == pytest.approx(1 / 64)


# ---------------------------------------------------------------------------
# Title rule
# ---------------------------------------------------------------------------
DISEASES = diseases.load_diseases()


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Systemic lupus erythematosus: a review", {"sle"}),
        ("SLE and cardiovascular risk", {"sle"}),
        ("sle mimics in practice", set()),  # lowercase does not match SLE
        ("Neonatal lupus: outcomes and management", set()),
        ("Drug-induced lupus: a review", set()),
        ("Cutaneous lupus erythematosus update", {"sle"}),
        ("Dermatomyositis: an update", {"dm"}),
        ("Juvenile dermatomyositis and JDM variants", {"dm"}),
        ("Ankylosing spondylitis in 2024", {"as"}),
        ("Axial spondyloarthritis review", {"as"}),
        ("Rheumatoid arthritis review", set()),
        ("SLE and dermatomyositis overlap", {"sle", "dm"}),
    ],
)
def test_title_rule(title, expected):
    assert select_articles.title_rule_diseases(title, DISEASES) == expected


# ---------------------------------------------------------------------------
# Fake turbopuffer namespace + fake LLM
# ---------------------------------------------------------------------------
class _NsResult:
    def __init__(self, rows):
        self.rows = rows


class FakeNs:
    """Duck-typed turbopuffer namespace; responses keyed per rank bucket."""

    def __init__(self, responder=None):
        self.calls = []
        self.responder = responder or (lambda kw: [])

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return _NsResult([dict(r) for r in self.responder(kwargs)])


class FakeRetriever:
    def __init__(self, ns, embedding=None):
        self.ns_pmc = ns
        self._embedding = embedding

    def _embed_query(self, query):
        return self._embedding


class FakeLLM:
    """Records call_many requests; serves canned parsed responses."""

    def __init__(self, parsed=None, **kwargs):
        self.kwargs = kwargs
        self.requests = []
        self.spent_usd = 0.0
        self._parsed = parsed or (lambda req: {})

    def call_many(self, requests):
        reqs = list(requests)
        self.requests.extend(reqs)
        return [
            llm.BatchResult(index=i, parsed=self._parsed(req))
            for i, req in enumerate(reqs)
        ]


def _args(**over):
    base = dict(
        disease="all",
        limit=None,
        dry_run=False,
        budget_usd=None,
        cap=None,
        pmcids=None,
        recheck_title_rule=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _row(pmcid, title, pub_types, article_type, abstract="abs", **kw):
    row = dict(
        id=f"{pmcid}:0",
        pmcid=pmcid,
        pmid="1",
        doi="10.1/x",
        title=title,
        abstract=abstract,
        journal="J Test",
        year=2024,
        country="Canada",
        publication_type=pub_types,
        article_type=article_type,
    )
    row.update(kw)
    return row


P1_RELEVANT = {
    "primary_disease_keys": ["sle"],
    "is_narrative_review": True,
    "decision": "relevant",
    "reason": "narrative review of SLE",
}
P1_IRRELEVANT = {
    "primary_disease_keys": [],
    "is_narrative_review": False,
    "decision": "irrelevant",
    "reason": "case report",
}


def _responder(kw):
    """Same rows for every query; the Or review filter is exercised."""
    rank_by = kw.get("rank_by")
    if rank_by is None:  # abstract_for re-query by pmcid
        return []
    return [
        _row("PMC100", "Systemic lupus erythematosus: a review", ["Review"], "review-article"),
        _row("PMC101", "Immune pathways in muscle disease", ["Review"], "review-article"),
        _row("PMC102", "A tricky case", ["Case Based Review"], "review-article"),
        _row("PMC103", "Correspondence notes", ["Review"], "letter"),
    ]


def test_select_run_end_to_end(conn, monkeypatch, capsys):
    ns = FakeNs(_responder)
    fake_llm_holder = {}

    monkeypatch.setattr(
        select_articles, "_make_retriever", lambda: FakeRetriever(ns, embedding=None)
    )
    monkeypatch.setattr(
        pmc,
        "get_license",
        lambda pmcid: pmc.LicenseInfo(
            code="cc-by" if pmcid != "PMC101" else "cc-by-nc",
            url="https://creativecommons.org/licenses/by/4.0/",
            oa_subset="oa",
        ),
    )

    def _fake_llm_cls(**kwargs):
        client = FakeLLM(parsed=lambda req: P1_RELEVANT, **kwargs)
        fake_llm_holder["client"] = client
        return client

    monkeypatch.setattr(llm, "LLMClient", _fake_llm_cls)
    select_articles._or_filter_supported = None

    assert select_articles.run(_args(disease="sle")) == 0

    # Review Or filter + has_full_text used on every bucket query.
    assert ns.calls
    for call in ns.calls:
        if call.get("rank_by") is None:
            continue
        assert call["filters"] == [
            "And",
            [
                ["has_full_text", "Eq", True],
                [
                    "Or",
                    [
                        ["publication_type", "Contains", "Review"],
                        ["article_type", "Eq", "review-article"],
                    ],
                ],
            ],
        ]
    # title BM25 top_k=200, content BM25 top_k=300; dense skipped (no embed).
    buckets = {(c["rank_by"][0], c["top_k"]) for c in ns.calls if c.get("rank_by")}
    assert ("title", 200) in buckets
    assert ("page_content", 300) in buckets
    assert not any(b[0] == "vector" for b in buckets)

    rows = {
        r["pmcid"]: r for r in conn.execute("SELECT * FROM articles")
    }
    # PMC100 goes through P1 like every license_ok article (the title rule no
    # longer auto-passes); PMC101 license rejected (cc-by-nc);
    # PMC102/103 excluded by the type filter and never inserted.
    assert set(rows) == {"PMC100", "PMC101"}
    assert rows["PMC100"]["status"] == "relevant"
    assert rows["PMC100"]["relevance_reason"] == "narrative review of SLE"
    assert rows["PMC100"]["primary_disease_keys_json"] == '["sle"]'
    assert rows["PMC101"]["status"] == "license_rejected"
    # One P1 call for the single license_ok article.
    reqs = fake_llm_holder["client"].requests
    assert len(reqs) == 1
    assert reqs[0]["stage"] == "p1"
    assert "Systemic lupus erythematosus: a review" in reqs[0]["user_content"]

    counts = json.loads(
        (vp_config.reports_dir() / "stage2_counts.json").read_text()
    )
    assert counts["sle"]["candidates"] == 4
    assert counts["sle"]["after_type_filter"] == 2
    assert counts["sle"]["after_license_filter"] == 1
    assert counts["sle"]["relevant"] == 1
    assert counts["sle"]["cap"] == 150
    assert counts["sle"]["over_cap"] is False

    # Resumable: a second run makes no license or LLM calls.
    license_calls = []
    monkeypatch.setattr(pmc, "get_license", lambda p: license_calls.append(p))
    fake_llm_holder["client"].requests.clear()
    assert select_articles.run(_args(disease="sle")) == 0
    assert license_calls == []
    assert fake_llm_holder["client"].requests == []


def test_select_p1_relevance_routing(conn, monkeypatch):
    """Title-rule failures go through P1; keys/decision decide relevance."""

    def responder(kw):
        return [
            _row("PMC200", "Unrelated title one", ["Review"], "review-article"),
            _row("PMC201", "Unrelated title two", ["Review"], "review-article"),
            _row("PMC202", "Unrelated title three", ["Review"], "review-article"),
        ]

    ns = FakeNs(responder)
    monkeypatch.setattr(
        select_articles, "_make_retriever", lambda: FakeRetriever(ns, embedding=None)
    )
    monkeypatch.setattr(
        pmc,
        "get_license",
        lambda pmcid: pmc.LicenseInfo(code="cc-by", url=None, oa_subset="oa"),
    )

    def parsed_for(req):
        if "PMC200" in req["user_content"] or "title one" in req["user_content"]:
            return P1_RELEVANT
        return P1_IRRELEVANT

    holder = {}

    def _fake_llm_cls(**kwargs):
        client = FakeLLM(parsed=parsed_for, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr(llm, "LLMClient", _fake_llm_cls)
    select_articles._or_filter_supported = None
    assert select_articles.run(_args(disease="sle")) == 0

    # All three needed P1 (no title matches); user_content format verified.
    reqs = holder["client"].requests
    assert len(reqs) == 3
    assert all(r["stage"] == "p1" for r in reqs)
    assert all(r["user_content"].startswith("Title: ") for r in reqs)
    assert all("\nAbstract: " in r["user_content"] for r in reqs)

    rows = {r["pmcid"]: r["status"] for r in conn.execute("SELECT pmcid, status FROM articles")}
    assert rows["PMC200"] == "relevant"
    assert rows["PMC201"] == "irrelevant"
    assert rows["PMC202"] == "irrelevant"
    keys = conn.execute(
        "SELECT primary_disease_keys_json FROM articles WHERE pmcid='PMC200'"
    ).fetchone()[0]
    assert db.from_json(keys) == ["sle"]


def test_select_dry_run_no_writes(conn, monkeypatch, capsys):
    ns = FakeNs(_responder)
    monkeypatch.setattr(
        select_articles, "_make_retriever", lambda: FakeRetriever(ns)
    )
    select_articles._or_filter_supported = None
    assert select_articles.run(_args(disease="sle", dry_run=True)) == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()["n"] == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert ns.calls  # queries still ran


def test_retrieve_for_disease_dense_and_rrf():
    """Dense ANN runs when an embedding exists; all lists feed one RRF."""
    emb = [0.1, 0.2]
    seen = []

    def responder(kw):
        rank_by = kw["rank_by"]
        seen.append(rank_by[0])
        if rank_by[0] == "title":
            return [_row("PMC1", "t", ["Review"], "review-article")]
        if rank_by[0] == "page_content":
            return [_row("PMC1", "t", ["Review"], "review-article"),
                    _row("PMC2", "t2", ["Review"], "review-article")]
        if rank_by[0] == "vector":
            return [_row("PMC2", "t2", ["Review"], "review-article")]
        return []

    ns = FakeNs(responder)
    out = select_articles.retrieve_for_disease(ns, lambda q: emb, ["lupus"])
    assert "vector" in seen
    # PMC1: rank1 + rank1; PMC2: rank2 + rank1
    assert out["PMC1"]["score"] == pytest.approx(1 / 61 + 1 / 61)
    assert out["PMC2"]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["PMC1"]["attrs"]["pmcid"] == "PMC1"


# ---------------------------------------------------------------------------
# --recheck-title-rule
# ---------------------------------------------------------------------------
def _seed_title_rule(conn, pmcid, status, keys='["sle"]'):
    conn.execute(
        "INSERT INTO articles (pmcid, title, status, license_code, "
        "relevance_reason, primary_disease_keys_json, retrieval_score) "
        "VALUES (?, ?, ?, 'cc-by', 'title_rule', ?, 0.5)",
        (pmcid, f"Title {pmcid}", status, keys),
    )
    conn.commit()


def test_recheck_title_rule(conn, monkeypatch):
    diseases.seed(conn)
    _seed_title_rule(conn, "PMCA", "relevant")
    _seed_title_rule(conn, "PMCB", "relevant")
    _seed_title_rule(conn, "PMCC", "parsed")  # E2E article: report only
    # an irrelevant article with a different reason is out of scope
    conn.execute(
        "INSERT INTO articles (pmcid, title, status, relevance_reason) "
        "VALUES ('PMCD', 'T', 'irrelevant', 'p1_irrelevant')"
    )
    conn.commit()

    # abstract_for queries ns by pmcid (rank_by absent) -> serve an abstract.
    ns = FakeNs(lambda kw: [dict(abstract="abstract text")])
    monkeypatch.setattr(
        select_articles, "_make_retriever", lambda: FakeRetriever(ns)
    )

    def parsed_for(req):
        if "Title PMCA" in req["user_content"]:
            return {
                **P1_RELEVANT,
                "primary_disease_keys": ["dm"],
                "reason": "dm narrative review",
            }
        return P1_IRRELEVANT

    holder = {}

    def _fake_llm_cls(**kwargs):
        client = FakeLLM(parsed=parsed_for, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr(llm, "LLMClient", _fake_llm_cls)
    assert select_articles.run(_args(recheck_title_rule=True)) == 0

    reqs = holder["client"].requests
    assert len(reqs) == 3  # PMCA, PMCB, PMCC — PMCD out of scope
    assert all(r["stage"] == "p1" for r in reqs)

    rows = {r["pmcid"]: r for r in conn.execute("SELECT * FROM articles")}
    # PMCA stays relevant but now carries P1's keys + reason
    assert rows["PMCA"]["status"] == "relevant"
    assert rows["PMCA"]["primary_disease_keys_json"] == '["dm"]'
    assert rows["PMCA"]["relevance_reason"] == "dm narrative review"
    # PMCB flips to irrelevant
    assert rows["PMCB"]["status"] == "irrelevant"
    # PMCC (parsed) is untouched
    assert rows["PMCC"]["status"] == "parsed"
    assert rows["PMCC"]["relevance_reason"] == "title_rule"
    assert rows["PMCC"]["primary_disease_keys_json"] == '["sle"]'

    counts = json.loads(
        (vp_config.reports_dir() / "stage2_counts.json").read_text()
    )
    assert counts["sle"]["relevant"] == 1  # PMCC only (PMCA moved to dm)
    assert counts["dm"]["relevant"] == 1  # PMCA


# ---------------------------------------------------------------------------
# W4a: parallel retrieval — identical output to the sequential implementation
# ---------------------------------------------------------------------------
def _retrieve_for_disease_sequential(
    ns, embed_fn, synonyms, visual_queries=None
):
    """Private copy of the pre-parallelization logic, kept for comparison."""
    sa = select_articles
    ranked_lists = []
    ranked_weights = []
    attrs = {}
    evidence = {}
    for synonym in synonyms:
        embedding = None
        if embed_fn is not None:
            try:
                embedding = embed_fn(synonym)
            except Exception:
                pass
        buckets = [
            (["title", "BM25", synonym], sa.TITLE_TOP_K),
            (["page_content", "BM25", synonym], sa.CONTENT_TOP_K),
        ]
        if embedding is not None:
            buckets.append((["vector", "ANN", embedding], sa.DENSE_TOP_K))
        for rank_by, top_k in buckets:
            rows = sa._rank_query(ns, rank_by, top_k)
            ranked_lists.append([str(r.get("pmcid") or "") for r in rows])
            ranked_weights.append(1.0)
            for row in rows:
                pmcid = str(row.get("pmcid") or "")
                if pmcid and pmcid not in attrs:
                    attrs[pmcid] = row
            if rank_by[0] == "page_content":
                for rank, row in enumerate(rows, start=1):
                    pmcid = str(row.get("pmcid") or "")
                    passage = str(row.get("page_content") or "").strip()
                    if not pmcid or not passage or not sa._is_visual_passage(row):
                        continue
                    evidence.setdefault(pmcid, []).append({
                        "query": synonym,
                        "query_kind": "synonym",
                        "text": passage[: sa.MAX_EVIDENCE_TEXT_CHARS],
                        "section": str(row.get("section_title") or ""),
                        "section_type": str(row.get("section_type") or ""),
                        "modality": "",
                        "finding": "",
                        "rank": rank,
                        "score": 1.0 / (sa.RRF_K + rank),
                    })
    for spec in visual_queries or []:
        query = str(spec.get("query") or "").strip()
        if not query:
            continue
        rows = sa._rank_query(
            ns, ["page_content", "BM25", query], sa.VISUAL_QUERY_TOP_K
        )
        ranked_lists.append([str(r.get("pmcid") or "") for r in rows])
        ranked_weights.append(1.0)
        visual_rows = [row for row in rows if sa._is_visual_passage(row)]
        if visual_rows:
            ranked_lists.append(
                [str(row.get("pmcid") or "") for row in visual_rows]
            )
            ranked_weights.append(sa.VISUAL_QUERY_RRF_WEIGHT - 1.0)
        for row in rows:
            pmcid = str(row.get("pmcid") or "")
            if pmcid and pmcid not in attrs:
                attrs[pmcid] = row
        for rank, row in enumerate(rows, start=1):
            pmcid = str(row.get("pmcid") or "")
            passage = str(row.get("page_content") or "").strip()
            if not pmcid or not passage or not sa._is_visual_passage(row):
                continue
            evidence.setdefault(pmcid, []).append({
                "query": query,
                "query_kind": "visual",
                "text": passage[: sa.MAX_EVIDENCE_TEXT_CHARS],
                "section": str(row.get("section_title") or ""),
                "section_type": str(row.get("section_type") or ""),
                "modality": str(spec.get("modality") or ""),
                "finding": str(spec.get("finding") or ""),
                "rank": rank,
                "score": 1.0 / (sa.RRF_K + rank),
            })
    scores = sa.rrf_scores(ranked_lists, weights=ranked_weights)
    out = {}
    for pmcid, score in scores.items():
        out[pmcid] = {
            "score": score,
            "attrs": attrs.get(pmcid, {}),
            "matched_passages": sa._select_evidence(evidence.get(pmcid, [])),
        }
    return out


def _parity_responder(kw):
    """Fixed rows per bucket; thread-safe (read-only) for the query pool."""
    rank_by = kw.get("rank_by")
    if rank_by is None:
        return []
    field = rank_by[0]
    term = rank_by[2] if len(rank_by) > 2 else ""
    if field == "title":
        return [
            _row("PMC_A1", f"Review about {term}", ["Review"], "review-article"),
            _row("PMC_A2", "Shared hit", ["Review"], "review-article"),
        ]
    if field == "vector":
        return [
            _row("PMC_A2", "Shared hit", ["Review"], "review-article"),
            _row("PMC_A3", "Dense only", ["Review"], "review-article"),
        ]
    if term == "visual alpha query":
        return [
            _row(
                "PMC_V1", "Visual hit", ["Review"], "review-article",
                page_content="Photograph shows a malar rash.",
                section_title="Clinical findings", section_type="clinical",
            ),
            _row(
                "PMC_V2", "Plain hit", ["Review"], "review-article",
                page_content="Molecular signaling pathways.",
                section_title="Mechanism", section_type="mechanism",
            ),
        ]
    if term == "visual beta query":
        return [
            _row(
                "PMC_V2", "Plain hit", ["Review"], "review-article",
                page_content="Molecular signaling pathways.",
                section_title="Mechanism", section_type="mechanism",
            ),
        ]
    return [
        _row(
            "PMC_C1", "Content hit", ["Review"], "review-article",
            page_content="Biopsy reveals a histologic lesion.",
            section_title="Pathology", section_type="clinical",
        ),
        _row(
            "PMC_A2", "Shared hit", ["Review"], "review-article",
            page_content="General text without cues.",
            section_title="Introduction", section_type="intro",
        ),
    ]


def test_retrieve_for_disease_parallel_matches_sequential(monkeypatch):
    """Same scores, attrs and matched_passages as the serial code path."""
    monkeypatch.setattr(vp_config, "VP_CONCURRENCY", 4)
    ns = FakeNs(_parity_responder)
    synonyms = ["alpha syn", "beta syn"]
    visual_queries = [
        {
            "query": "visual alpha query",
            "finding": "malar rash",
            "modality": "clinical photograph",
        },
        {
            "query": "visual beta query",
            "finding": "other finding",
            "modality": "MRI",
        },
        {"query": "   "},  # empty specs are skipped in both paths
    ]

    batched_calls = []

    def embed_many(queries):
        queries = list(queries)
        batched_calls.append(queries)
        return [[0.1, 0.2] if q == "alpha syn" else None for q in queries]

    embed_one = lambda q: [0.1, 0.2] if q == "alpha syn" else None  # noqa: E731

    select_articles._or_filter_supported = True
    calls_before = len(ns.calls)
    parallel = select_articles.retrieve_for_disease(
        ns,
        embed_one,
        synonyms,
        visual_queries,
        disease_key="sle",
        embed_many_fn=embed_many,
    )
    parallel_query_count = len(ns.calls) - calls_before

    select_articles._or_filter_supported = True
    reference = _retrieve_for_disease_sequential(
        ns, embed_one, synonyms, visual_queries
    )
    reference_query_count = len(ns.calls) - calls_before - parallel_query_count

    assert parallel == reference
    # One batched embeddings request covered every synonym.
    assert batched_calls == [["alpha syn", "beta syn"]]
    # Same number of namespace queries as the sequential implementation.
    assert parallel_query_count == reference_query_count
    # Dense ANN ran only for the synonym whose embedding exists.
    rank_bys = [c["rank_by"][0] for c in ns.calls[: parallel_query_count]]
    assert rank_bys.count("vector") == 1
    assert parallel  # non-empty fused output


def test_retrieve_for_disease_falls_back_to_per_item_embedding(monkeypatch):
    """A failing embed_many_fn falls back to embed_fn per synonym."""
    monkeypatch.setattr(vp_config, "VP_CONCURRENCY", 4)
    ns = FakeNs(_parity_responder)
    per_item_calls = []

    def embed_many(queries):
        raise RuntimeError("batch unsupported")

    def embed_one(q):
        per_item_calls.append(q)
        return [0.5]

    select_articles._or_filter_supported = True
    out = select_articles.retrieve_for_disease(
        ns, embed_one, ["syn one", "syn two"], embed_many_fn=embed_many
    )
    assert per_item_calls == ["syn one", "syn two"]
    rank_bys = [c["rank_by"][0] for c in ns.calls]
    assert rank_bys.count("vector") == 2  # dense ran for both synonyms
    assert out


def test_p1_request_construction_unchanged():
    """P1 request fields and the Title/Abstract truncation are fixed."""
    req = select_articles._p1_request("My title", "A" * 5000)
    assert req["stage"] == "p1"
    assert req["model"] == vp_config.VP_TRIAGE_MODEL
    assert req["system"] == P1.system
    assert req["schema"] == P1.schema
    assert req["prompt_version"] == P1.version
    assert req["user_content"].startswith("Title: My title\nAbstract: ")
    assert req["user_content"] == "Title: My title\nAbstract: " + "A" * 4000
