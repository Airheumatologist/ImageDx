from src.visual_pilot import db, manifestation_queue, parse
from balanced_fixtures import (
    MALAR_CAPTION,
    add_article,
    add_disease,
    add_figure,
    add_finding,
    add_panel,
    make_db,
)


def _db(tmp_path):
    conn = db.connect(tmp_path / "pilot.sqlite")
    db.init_db(conn)
    return conn


def _seed_lane(conn, finding="f1", disease="d1"):
    conn.execute("INSERT OR IGNORE INTO diseases(disease_key,name) VALUES(?,?)", (disease, disease))
    conn.execute(
        "INSERT OR IGNORE INTO findings_vocab(finding_key,disease_keys_json,label,category,approved) "
        "VALUES(?,?,?,?,1)", (finding, db.to_json([disease]), finding, "skin")
    )


def _candidate(conn, disease, finding, pmcid, provenance="explicit"):
    conn.execute(
        "INSERT INTO manifestation_candidates"
        "(disease_key,finding_key,pmcid,provenance_status,provenance_disease_key) "
        "VALUES(?,?,?,?,?)",
        (disease, finding, pmcid, provenance,
         disease if provenance == "explicit" else None),
    )


def _article(conn, pmcid, evidence, status="relevant", score=1.0):
    conn.execute(
        "INSERT INTO articles(pmcid,status,retrieval_score,retrieval_evidence_json) VALUES(?,?,?,?)",
        (pmcid, status, score, db.to_json(evidence)),
    )


def test_new_and_existing_database_migrations(tmp_path):
    conn = _db(tmp_path)
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"manifestation_candidates", "manifestation_lanes", "manifestation_representatives"} <= tables
    candidate_cols = {r["name"] for r in conn.execute("PRAGMA table_info(manifestation_candidates)")}
    assert {"disease_key", "finding_key", "pmcid", "query", "best_rank", "retrieval_score",
            "status", "last_outcome", "created_at", "updated_at",
            "provenance_status", "provenance_disease_key"} <= candidate_cols
    rep_cols = {r["name"] for r in conn.execute("PRAGMA table_info(manifestation_representatives)")}
    assert {"disease_key", "finding_key", "panel_id", "score", "scoring_json",
            "selection_source", "locked", "updated_at"} <= rep_cols
    conn.close()

    old = db.connect(tmp_path / "old.sqlite")
    old.execute("CREATE TABLE articles(pmcid TEXT PRIMARY KEY,status TEXT NOT NULL DEFAULT 'candidate')")
    old.commit()
    db.init_db(old)
    assert "retrieval_evidence_json" in db.table_columns(old, "articles")
    assert old.execute("SELECT 1 FROM sqlite_master WHERE name='manifestation_representatives'").fetchone()
    old.close()


def test_coverage_uses_published_panels(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "d1", "Disease One")
    add_finding(conn, "malar_rash", ("d1",), label="Malar rash")
    add_article(conn, "PMC1")
    add_figure(conn, "PMC1:fig1", "PMC1", caption=MALAR_CAPTION)
    add_panel(
        conn, tmp_path, "p1", "PMC1:fig1", "PMC1", "d1",
        findings=("malar_rash",), sha256="abc",
    )
    conn.execute(
        "INSERT INTO panel_curation(panel_id,image_sha256,decision,reason,policy_version) "
        "VALUES('p1','abc','exclude','reviewed','v1')"
    )
    conn.commit()
    assert parse.coverage_gaps(conn, "d1") == {"malar_rash": 0}
    conn.execute("DELETE FROM panel_curation WHERE panel_id='p1'")
    conn.commit()
    assert parse.coverage_gaps(conn, "d1") == {"malar_rash": 1}
    conn.close()


def test_round_robin_is_deterministic_and_reserves_each_lane(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn, "f1")
    _seed_lane(conn, "f2")
    for pmcid in ("A", "B", "C", "D"):
        _article(conn, pmcid, [])
    for finding, pmcids in (("f1", ("A", "B", "C")), ("f2", ("A", "D"))):
        for pmcid in pmcids:
            _candidate(conn, "d1", finding, pmcid)
    ranked = [{"pmcid": key} for key in ("A", "B", "C", "D")]
    # Each lane gets one article per round; shared A serves both lanes and
    # consumes a single slot.
    got = manifestation_queue.reserve_batch(conn, "d1", ranked, 3, {"f1", "f2"}, persist=False)
    assert [r["pmcid"] for r in got] == ["A", "B", "D"]
    got_again = manifestation_queue.reserve_batch(conn, "d1", ranked, 3, {"f1", "f2"}, persist=False)
    assert [r["pmcid"] for r in got_again] == [r["pmcid"] for r in got]
    conn.close()


def test_shared_article_consumes_one_slot_and_serves_both_lanes(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn, "f1")
    _seed_lane(conn, "f2")
    for pmcid in ("A", "B", "C"):
        _article(conn, pmcid, [])
    for finding in ("f1", "f2"):
        _candidate(conn, "d1", finding, "A")
    ranked = [{"pmcid": key} for key in ("A", "B", "C")]
    got = manifestation_queue.reserve_batch(conn, "d1", ranked, 3, {"f1", "f2"}, persist=True)
    # No global fallback fills spare capacity with articles unrelated to a lane.
    assert [r["pmcid"] for r in got] == ["A"]
    assert conn.execute(
        "SELECT COUNT(*) n FROM manifestation_candidates WHERE status='selected'"
    ).fetchone()["n"] == 2
    conn.close()


def test_candidate_statuses_resume_without_reselecting_parsed_articles(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn)
    _article(conn, "PMC1", [{
        "disease_key": "d1", "finding_key": "f1",
        "query": "disease finding", "rank": 2,
    }])
    assert manifestation_queue.sync_candidates(conn, "d1") == {"f1": 0}
    row = conn.execute(
        "SELECT provenance_status,provenance_disease_key FROM manifestation_candidates"
    ).fetchone()
    assert tuple(row) == ("explicit", "d1")
    manifestation_queue.record_article_outcome(conn, "PMC1", "parsed")
    manifestation_queue.sync_candidates(conn, "d1")
    row = conn.execute("SELECT status,last_outcome FROM manifestation_candidates").fetchone()
    assert tuple(row) == ("parsed", "article_parsed")
    lane = conn.execute("SELECT status FROM manifestation_lanes").fetchone()
    # A lane whose candidates are all parsed stays open for replenishment —
    # it is never prematurely exhausted.
    assert lane["status"] == "open"
    conn.close()


def test_representative_does_not_block_panel_replace(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn)
    _article(conn, "PMC1", [])
    conn.execute("INSERT INTO figures(figure_id,pmcid,status) VALUES('PMC1:f1','PMC1','stored')")
    conn.execute(
        "INSERT INTO panels(panel_id,figure_id,pmcid,disease_key,findings_json,sha256) "
        "VALUES('p1','PMC1:f1','PMC1','d1',?, 'old')", (db.to_json([{"finding_key":"f1"}]),)
    )
    conn.execute(
        "INSERT INTO manifestation_representatives "
        "(disease_key,finding_key,panel_id,score,locked) VALUES('d1','f1','p1',1,1)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO panels(panel_id,figure_id,pmcid,disease_key,findings_json,sha256) "
        "VALUES('p1','PMC1:f1','PMC1','d1',?, 'new')",
        (db.to_json([{"finding_key":"f1"}]),),
    )
    representative = conn.execute(
        "SELECT panel_id,locked FROM manifestation_representatives WHERE disease_key='d1' AND finding_key='f1'"
    ).fetchone()
    assert tuple(representative) == ("p1", 1)
    conn.close()


def test_existing_panel_fk_is_migrated_without_losing_locked_mapping(tmp_path):
    conn = _db(tmp_path)
    conn.execute("DROP TABLE manifestation_representatives")
    conn.execute(
        "CREATE TABLE manifestation_representatives (disease_key TEXT NOT NULL, "
        "finding_key TEXT NOT NULL, panel_id TEXT NOT NULL REFERENCES panels(panel_id), "
        "score REAL NOT NULL, scoring_json TEXT NOT NULL DEFAULT '{}', "
        "selection_source TEXT NOT NULL DEFAULT 'auto', locked INTEGER NOT NULL DEFAULT 0, "
        "updated_at TEXT DEFAULT (datetime('now')), PRIMARY KEY(disease_key,finding_key))"
    )
    _seed_lane(conn)
    _article(conn, "PMC1", [])
    conn.execute("INSERT INTO figures(figure_id,pmcid,status) VALUES('PMC1:f1','PMC1','stored')")
    conn.execute(
        "INSERT INTO panels(panel_id,figure_id,pmcid,disease_key,findings_json) "
        "VALUES('p1','PMC1:f1','PMC1','d1',?)", (db.to_json([{"finding_key":"f1"}]),)
    )
    conn.execute(
        "INSERT INTO manifestation_representatives "
        "(disease_key,finding_key,panel_id,score,locked) VALUES('d1','f1','p1',1,1)"
    )
    conn.execute("ALTER TABLE articles DROP COLUMN retrieval_evidence_json")
    conn.commit()
    assert "retrieval_evidence_json" not in db.table_columns(conn, "articles")
    assert any(r["table"] == "panels" for r in conn.execute(
        "PRAGMA foreign_key_list(manifestation_representatives)"
    ))

    db.init_db(conn)
    assert "retrieval_evidence_json" in db.table_columns(conn, "articles")
    assert not any(r["table"] == "panels" for r in conn.execute(
        "PRAGMA foreign_key_list(manifestation_representatives)"
    ))
    conn.execute(
        "INSERT OR REPLACE INTO panels(panel_id,figure_id,pmcid,disease_key,findings_json) "
        "VALUES('p1','PMC1:f1','PMC1','d1',?)",
        (db.to_json([{"finding_key":"f1"}]),),
    )
    representative = conn.execute(
        "SELECT panel_id,locked FROM manifestation_representatives WHERE disease_key='d1' AND finding_key='f1'"
    ).fetchone()
    assert tuple(representative) == ("p1", 1)
    conn.close()


def test_terminal_candidate_outcomes_are_auditable(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn)
    for pmcid, status in (("PMC1", "license_rejected"), ("PMC2", "irrelevant")):
        _article(conn, pmcid, [{
            "disease_key": "d1", "finding_key": "f1", "query": "finding",
        }], status=status)
    manifestation_queue.sync_candidates(conn, "d1")
    candidates = {
        row["pmcid"]: (row["status"], row["last_outcome"])
        for row in conn.execute("SELECT pmcid,status,last_outcome FROM manifestation_candidates")
    }
    assert candidates == {
        "PMC1": ("exhausted", "article_license_rejected"),
        "PMC2": ("exhausted", "article_irrelevant"),
    }
    lane = conn.execute("SELECT status FROM manifestation_lanes").fetchone()
    # Terminal candidates no longer retire the lane — bounded replenishment
    # decides exhaustion via the search-policy ledger instead.
    assert lane["status"] == "open"
    conn.close()


def test_relevant_article_for_other_disease_terminates_pair(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn)
    _article(conn, "PMC1", [{
        "disease_key": "d1", "finding_key": "f1", "query": "finding",
    }], status="relevant")
    conn.execute(
        "UPDATE articles SET primary_disease_keys_json='[\"d2\"]' WHERE pmcid='PMC1'"
    )
    manifestation_queue.sync_candidates(conn, "d1")
    row = conn.execute(
        "SELECT status,last_outcome FROM manifestation_candidates"
    ).fetchone()
    assert tuple(row) == ("exhausted", "article_other_disease")
    conn.close()


def test_legacy_unresolved_candidates_stay_retained_and_inactive(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn)
    _article(conn, "PMC1", [{
        "disease_key": "d1", "finding_key": "f1", "query": "finding",
    }])
    # Relevant with this disease in its primary keys stays actionable.
    conn.execute(
        "UPDATE articles SET primary_disease_keys_json='[\"d1\"]' WHERE pmcid='PMC1'"
    )
    _article(conn, "PMC2", [{
        "finding_key": "f1", "query": "legacy without disease",
    }])
    # A pre-existing ambiguous row is retained but never activated.
    _article(conn, "PMC9", [])
    _candidate(conn, "d1", "f1", "PMC9", provenance="unresolved")
    uncovered = manifestation_queue.sync_candidates(conn, "d1")
    assert uncovered == {"f1": 0}
    rows = {
        row["pmcid"]: (row["status"], row["provenance_status"])
        for row in conn.execute(
            "SELECT pmcid,status,provenance_status FROM manifestation_candidates"
        )
    }
    assert rows["PMC1"] == ("pending", "explicit")
    assert rows["PMC9"] == ("pending", "unresolved")
    # Legacy evidence without disease provenance activates nothing.
    assert "PMC2" not in rows
    ranked = [{"pmcid": "PMC9"}, {"pmcid": "PMC1"}]
    got = manifestation_queue.reserve_batch(
        conn, "d1", ranked, 2, {"f1"}, persist=False
    )
    assert [r["pmcid"] for r in got] == ["PMC1"]
    conn.close()
