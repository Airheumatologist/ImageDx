"""Bounded per-pair replenishment: durable attempts, resume, limits (offline)."""

from types import SimpleNamespace

from src.visual_pilot import db, search_policy, select_articles
from balanced_fixtures import add_disease, add_finding, make_db

DISEASE = "d1"
FINDING = "f1"


def _row(pmcid, passage=None):
    return {
        "pmcid": pmcid,
        "publication_type": ["Review"],
        "article_type": "review-article",
        "page_content": passage or f"Figure. Clinical photograph for {pmcid}.",
        "section_title": "Clinical examination",
        "section_type": "body",
    }


class FakeNS:
    """In-memory PMC namespace answering attempt + metadata queries."""

    def __init__(self, hits_by_depth=None, fail=None, reject_depth=None):
        self.calls = []
        self.hits_by_depth = hits_by_depth or {}
        self.fail = fail          # exception raised on every attempt query
        self.reject_depth = reject_depth
        self._failed = False

    def query(self, **kwargs):
        self.calls.append(kwargs)
        filters = kwargs.get("filters")
        depth = (kwargs.get("limit") or {}).get("total")
        if isinstance(filters, list) and filters and filters[0] == "Or":
            # metadata hydration: Or over pmcid Eq clauses
            pmcids = [c[2] for c in filters[1] if isinstance(c, list)]
            return SimpleNamespace(rows=[
                {"pmcid": p, "title": f"Review {p}", "abstract": "A" * 100}
                for p in pmcids
            ])
        # attempt query — an Or arm nested inside And is the unsupported shape
        if self.fail is not None:
            raise self.fail
        if self.reject_depth is not None and depth == self.reject_depth:
            raise RuntimeError(f"provider rejected depth {depth}")
        hits = self.hits_by_depth.get(depth, [])
        return SimpleNamespace(rows=[_row(p) for p in hits])


def _db(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, DISEASE, "Disease One")
    add_finding(conn, FINDING, (DISEASE,), label="Finding One")
    conn.execute(
        "INSERT INTO manifestation_lanes(disease_key,finding_key,status) "
        "VALUES(?,?,'open')", (DISEASE, FINDING),
    )
    conn.commit()
    return conn


def _retriever(monkeypatch, ns):
    monkeypatch.setattr(
        select_articles, "_make_retriever",
        lambda: SimpleNamespace(ns_pmc=ns),
    )


def _mark_relevant(conn, disease_key=DISEASE):
    def _fake(conn, client, requests, submitted):
        for pmcid in submitted:
            conn.execute(
                "UPDATE articles SET status='relevant', "
                "primary_disease_keys_json=? WHERE pmcid=?",
                (db.to_json([disease_key]), pmcid),
            )
            conn.commit()
        return {"completed": len(submitted), "errors": 0, "budget": 0}
    return _fake


def _mark_irrelevant(conn):
    def _fake(conn, client, requests, submitted):
        for pmcid in submitted:
            conn.execute(
                "UPDATE articles SET status='irrelevant' WHERE pmcid=?", (pmcid,)
            )
            conn.commit()
        return {"completed": len(submitted), "errors": 0, "budget": 0}
    return _fake


def _license_ok(pmcid):
    return pmcid, {
        "status": "license_ok", "license_code": "cc-by",
        "license_url": "https://x", "oa_subset": "oa", "error": None,
    }


def _attempts(conn):
    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM pair_search_attempts ORDER BY round_no, query"
        )
    ]


def test_dry_run_returns_plan_without_writes_or_calls(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    ns = FakeNS({"300": ["PMC1"]})
    _retriever(monkeypatch, ns)
    result = select_articles.replenish_pair(
        conn, DISEASE, FINDING, round_no=1, dry_run=True
    )
    assert result["status"] == "dry_run"
    assert result["queries_attempted"] == 0
    assert result["plan"] and all("query" in item for item in result["plan"])
    assert ns.calls == []
    assert conn.execute("SELECT COUNT(*) n FROM pair_search_attempts").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) n FROM articles").fetchone()["n"] == 0
    conn.close()


def test_zero_limits_pause_before_any_provider_call(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    ns = FakeNS({"300": ["PMC1"]})
    _retriever(monkeypatch, ns)
    for kwargs, reason in (
        ({"max_articles": 0}, "article_limit"),
        ({"budget_usd": 0}, "budget"),
        ({"max_runtime_seconds": 0}, "runtime_limit"),
    ):
        result = select_articles.replenish_pair(
            conn, DISEASE, FINDING, round_no=1, **kwargs
        )
        assert result["status"] == "paused"
        assert result["reason"] == reason
        assert result["queries_attempted"] == 0
    assert ns.calls == []
    assert conn.execute("SELECT COUNT(*) n FROM pair_search_attempts").fetchone()["n"] == 0
    conn.close()


def test_pending_candidates_are_drained_before_new_retrieval(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    conn.execute("INSERT INTO articles(pmcid,status,title) VALUES('PEND','candidate','T')")
    conn.execute(
        "INSERT INTO manifestation_candidates"
        "(disease_key,finding_key,pmcid,provenance_status,provenance_disease_key) "
        "VALUES('d1','f1','PEND','explicit','d1')"
    )
    conn.commit()
    ns = FakeNS({"300": ["PMC9"]})
    _retriever(monkeypatch, ns)
    monkeypatch.setattr(select_articles, "join_license", _license_ok)
    monkeypatch.setattr(
        select_articles, "process_relevance", _mark_relevant(conn)
    )
    result = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    # The licensed+relevant pending article returns pending_work; the new
    # retrieval spec never fired.
    assert result["status"] == "pending_work"
    assert result["reason"] == "relevant_pending"
    assert result["queries_attempted"] == 0
    attempt_queries = [c for c in ns.calls if c.get("rank_by")]
    assert attempt_queries == []
    conn.close()


def test_terminal_unsuitable_round_advances_to_next_depth(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    # Two query variants in round 1, each returning unsuitable articles.
    round1 = [f"PMC{i:02d}" for i in range(22)]
    ns = FakeNS({300: round1, 600: [], 1200: []})
    _retriever(monkeypatch, ns)
    monkeypatch.setattr(select_articles, "join_license", _license_ok)
    # Every licensed article is judged irrelevant — terminal unsuitable.
    monkeypatch.setattr(
        select_articles, "process_relevance", _mark_irrelevant(conn)
    )

    first = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    assert first["status"] == "round_complete"
    assert first["reason"] == "next_round=2"
    assert {a["round_no"] for a in _attempts(conn)} == {1}
    assert all(a["status"] == "completed" for a in _attempts(conn))

    next_round = search_policy.next_round(conn, DISEASE, FINDING)
    assert next_round == 2
    second = select_articles.replenish_pair(
        conn, DISEASE, FINDING, round_no=next_round
    )
    assert second["status"] == "round_complete"
    assert second["reason"] == "next_round=3"
    depths = {(a["round_no"], a["depth"]) for a in _attempts(conn)}
    assert (1, 300) in depths and (2, 600) in depths
    conn.close()


def test_all_rounds_complete_returns_search_plan_exhausted(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    ns = FakeNS({300: [], 600: [], 1200: []})
    _retriever(monkeypatch, ns)
    for round_no in (1, 2, 3):
        result = select_articles.replenish_pair(
            conn, DISEASE, FINDING, round_no=round_no
        )
    assert result["status"] == "search_plan_exhausted"
    assert result["reason"] == "all_rounds_complete"
    assert search_policy.next_round(conn, DISEASE, FINDING) is None
    conn.close()


def test_duplicate_only_results_count_zero_new_candidates(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    conn.execute("INSERT INTO articles(pmcid,status) VALUES('DUP','parsed')")
    conn.execute(
        "INSERT INTO manifestation_candidates"
        "(disease_key,finding_key,pmcid,status,provenance_status,"
        "provenance_disease_key) VALUES('d1','f1','DUP','parsed','explicit','d1')"
    )
    conn.commit()
    ns = FakeNS({300: ["DUP", "DUP", "DUP"]})
    _retriever(monkeypatch, ns)
    result = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    # Duplicate-only work produced no new pair candidates; the round still
    # completes so the caller advances to the next unattempted strategy.
    assert result["new_candidates"] == 0
    assert result["status"] == "round_complete"
    # Terminal article status was preserved.
    assert conn.execute(
        "SELECT status FROM articles WHERE pmcid='DUP'"
    ).fetchone()["status"] == "parsed"
    conn.close()


def test_completed_attempts_never_requery_and_started_retry(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    ns = FakeNS({300: ["PMC1"]})
    _retriever(monkeypatch, ns)
    monkeypatch.setattr(select_articles, "join_license", _license_ok)
    monkeypatch.setattr(
        select_articles, "process_relevance", _mark_irrelevant(conn)
    )
    first = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    assert first["status"] == "round_complete"
    issued = len([c for c in ns.calls if c.get("rank_by")])

    # Second invocation over the same round: every spec is completed.
    second = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    assert second["queries_attempted"] == 0
    assert len([c for c in ns.calls if c.get("rank_by")]) == issued

    # A 'started' row simulates a crash between ledger write and provider
    # call: it retries on the same ledger key.
    spec = search_policy.attempt_specs(conn, DISEASE, FINDING, 1)[0]
    conn.execute(
        "UPDATE pair_search_attempts SET status='started', completed_at=NULL "
        "WHERE query_filter_hash=?", (spec["query_filter_hash"],)
    )
    conn.commit()
    third = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    assert third["queries_attempted"] == 1
    assert len([c for c in ns.calls if c.get("rank_by")]) == issued + 1
    conn.close()


def test_retrieved_attempt_resumes_from_stored_rows(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    ns = FakeNS({300: ["PMC1"]})
    _retriever(monkeypatch, ns)
    monkeypatch.setattr(select_articles, "join_license", _license_ok)
    monkeypatch.setattr(
        select_articles, "process_relevance", _mark_irrelevant(conn)
    )
    real_hydrate = select_articles.hydrate_metadata
    state = {"failed": False}

    def flaky(ns_, pmcids, counters=None):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("hydrate blew up")
        return real_hydrate(ns_, pmcids, counters)

    monkeypatch.setattr(select_articles, "hydrate_metadata", flaky)
    first = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    assert first["status"] == "retrieval_error"
    assert "hydration_error" in first["reason"]
    # The successful search is durable: the attempt is 'retrieved'.
    statuses = {a["status"] for a in _attempts(conn)}
    assert "retrieved" in statuses
    specs = search_policy.attempt_specs(conn, DISEASE, FINDING, 1)

    monkeypatch.setattr(select_articles, "hydrate_metadata", real_hydrate)
    queries_before = [c["rank_by"][2] for c in ns.calls if c.get("rank_by")]
    assert queries_before == [specs[0]["query"]]
    second = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    # The retrieved spec re-hydrated its stored rows; only the remaining
    # unattempted spec issued a new query.
    queries_after = [c["rank_by"][2] for c in ns.calls if c.get("rank_by")]
    assert queries_after == [specs[0]["query"], specs[1]["query"]]
    assert second["new_candidates"] == 1
    assert conn.execute(
        "SELECT provenance_status FROM manifestation_candidates "
        "WHERE pmcid='PMC1'"
    ).fetchone()["provenance_status"] == "explicit"
    conn.close()


def test_transient_error_is_retryable_and_never_exhausts(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    ns = FakeNS({300: ["PMC1"]}, fail=RuntimeError("upstream 503"))
    _retriever(monkeypatch, ns)
    monkeypatch.setattr(select_articles, "join_license", _license_ok)
    monkeypatch.setattr(
        select_articles, "process_relevance", _mark_irrelevant(conn)
    )
    first = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    assert first["status"] == "retrieval_error"
    assert "503" in first["reason"]
    attempt = _attempts(conn)[0]
    assert attempt["status"] == "error"
    assert "503" in attempt["error"]
    assert search_policy.next_round(conn, DISEASE, FINDING) == 1

    ns.fail = None
    second = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    assert second["status"] == "round_complete"
    conn.close()


def test_depth_rejection_is_retryable_and_filters_unchanged(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    ns = FakeNS({300: ["PMC1"]}, reject_depth=300)
    _retriever(monkeypatch, ns)
    result = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    assert result["status"] == "retrieval_error"
    attempt = _attempts(conn)[0]
    assert attempt["status"] == "error"
    # The recorded filters/depth match the settled spec exactly — the depth
    # failure never dropped filters or silently lowered top-k.
    spec = search_policy.attempt_specs(conn, DISEASE, FINDING, 1)[0]
    assert attempt["filters_json"] == spec["filters_json"]
    assert attempt["depth"] == spec["depth"]
    # Every issued call carried the full And/Or review filter.
    assert all(
        isinstance(c["filters"], list) and c["filters"][0] == "And"
        and any(isinstance(clause, list) and clause[:1] == ["Or"]
                for clause in c["filters"][1])
        for c in ns.calls
    )
    conn.close()


def test_processed_pmcids_unique_across_drain_and_hydration(tmp_path, monkeypatch):
    """max_articles bounds the unique set spanning drain + hydration.

    One pre-existing pending candidate is drained (license + P1 calls) and
    the hydration pass then discovers three new candidates — every unique
    PMCID that consumed provider work lands in ``processed_pmcids`` exactly
    once, and the article limit pauses before more work starts.
    """
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO articles(pmcid,status,title) VALUES('PEND','candidate','T')"
    )
    conn.execute(
        "INSERT INTO manifestation_candidates"
        "(disease_key,finding_key,pmcid,provenance_status,provenance_disease_key) "
        "VALUES('d1','f1','PEND','explicit','d1')"
    )
    conn.commit()
    ns = FakeNS({300: ["H1", "H2", "H3"]})
    _retriever(monkeypatch, ns)
    monkeypatch.setattr(select_articles, "join_license", _license_ok)
    monkeypatch.setattr(
        select_articles, "process_relevance", _mark_irrelevant(conn)
    )
    result = select_articles.replenish_pair(
        conn, DISEASE, FINDING, round_no=1, max_articles=3
    )
    assert result["status"] == "paused"
    assert result["reason"] == "article_limit"
    ids = result["processed_pmcids"]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 4
    assert set(ids) == {"PEND", "H1", "H2", "H3"}
    conn.close()


def test_new_candidates_carry_explicit_pair_provenance(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    ns = FakeNS({300: ["PMC1", "PMC2"]})
    _retriever(monkeypatch, ns)
    monkeypatch.setattr(select_articles, "join_license", _license_ok)
    monkeypatch.setattr(
        select_articles, "process_relevance", _mark_relevant(conn)
    )
    result = select_articles.replenish_pair(conn, DISEASE, FINDING, round_no=1)
    assert result["new_candidates"] == 2
    assert result["status"] == "pending_work"  # licensed articles went relevant
    rows = conn.execute(
        "SELECT provenance_status,provenance_disease_key,finding_key "
        "FROM manifestation_candidates WHERE disease_key='d1'"
    ).fetchall()
    assert {tuple(r) for r in rows} == {("explicit", "d1", "f1")}
    conn.close()
