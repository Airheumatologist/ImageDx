from src.visual_pilot import db, manifestation_queue, parse


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
            "status", "last_outcome", "created_at", "updated_at"} <= candidate_cols
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
    conn = _db(tmp_path)
    _seed_lane(conn)
    _article(conn, "PMC1", [])
    conn.execute("INSERT INTO figures(figure_id,pmcid,status) VALUES('PMC1:f1','PMC1','stored')")
    conn.execute(
        "INSERT INTO panels(panel_id,figure_id,pmcid,disease_key,findings_json,sha256) "
        "VALUES('p1','PMC1:f1','PMC1','d1',?, 'abc')", (db.to_json([{"finding_key":"f1"}]),)
    )
    conn.execute(
        "INSERT INTO panel_curation(panel_id,image_sha256,decision,reason,policy_version) "
        "VALUES('p1','abc','exclude','reviewed','v1')"
    )
    assert parse.coverage_gaps(conn, "d1") == {"f1": 0}
    conn.execute("DELETE FROM panel_curation WHERE panel_id='p1'")
    assert parse.coverage_gaps(conn, "d1") == {"f1": 1}
    conn.close()


def test_round_robin_is_deterministic_and_reserves_each_lane(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn, "f1")
    _seed_lane(conn, "f2")
    for pmcid in ("A", "B", "C", "D"):
        _article(conn, pmcid, [])
    for finding, pmcids in (("f1", ("A", "B", "C")), ("f2", ("A", "D"))):
        for pmcid in pmcids:
            conn.execute(
                "INSERT INTO manifestation_candidates(disease_key,finding_key,pmcid) VALUES('d1',?,?)",
                (finding, pmcid),
            )
    ranked = [{"pmcid": key} for key in ("A", "B", "C", "D")]
    got = manifestation_queue.reserve_batch(conn, "d1", ranked, 3, {"f1", "f2"}, persist=False)
    assert [r["pmcid"] for r in got] == ["A", "B", "C"]
    got_again = manifestation_queue.reserve_batch(conn, "d1", ranked, 3, {"f1", "f2"}, persist=False)
    assert [r["pmcid"] for r in got_again] == [r["pmcid"] for r in got]
    conn.close()


def test_shared_article_is_deduplicated_and_fallback_fills_capacity(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn, "f1")
    _seed_lane(conn, "f2")
    for pmcid in ("A", "B", "C"):
        _article(conn, pmcid, [])
    for finding, pmcids in (("f1", ("A",)), ("f2", ("A",))):
        for pmcid in pmcids:
            conn.execute(
                "INSERT INTO manifestation_candidates(disease_key,finding_key,pmcid) VALUES('d1',?,?)",
                (finding, pmcid),
            )
    ranked = [{"pmcid": key} for key in ("A", "B", "C")]
    got = manifestation_queue.reserve_batch(conn, "d1", ranked, 3, {"f1", "f2"}, persist=True)
    assert [r["pmcid"] for r in got] == ["A", "B", "C"]
    assert conn.execute("SELECT COUNT(*) n FROM manifestation_candidates WHERE status='selected'").fetchone()["n"] == 2
    conn.close()


def test_candidate_statuses_resume_without_reselecting_parsed_articles(tmp_path):
    conn = _db(tmp_path)
    _seed_lane(conn)
    _article(conn, "PMC1", [{"finding_key": "f1", "query": "disease finding", "rank": 2}])
    assert manifestation_queue.sync_candidates(conn, "d1") == {"f1": 0}
    manifestation_queue.record_article_outcome(conn, "PMC1", "parsed")
    manifestation_queue.sync_candidates(conn, "d1")
    row = conn.execute("SELECT status,last_outcome FROM manifestation_candidates").fetchone()
    assert tuple(row) == ("parsed", "article_parsed")
    lane = conn.execute("SELECT status,last_outcome FROM manifestation_lanes").fetchone()
    assert tuple(lane) == ("exhausted", "all_candidates_terminal")
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
        _article(conn, pmcid, [{"finding_key": "f1", "query": "finding"}], status=status)
    manifestation_queue.sync_candidates(conn, "d1")
    candidates = {
        row["pmcid"]: (row["status"], row["last_outcome"])
        for row in conn.execute("SELECT pmcid,status,last_outcome FROM manifestation_candidates")
    }
    assert candidates == {
        "PMC1": ("exhausted", "article_license_rejected"),
        "PMC2": ("exhausted", "article_irrelevant"),
    }
    lane = conn.execute("SELECT status,last_outcome FROM manifestation_lanes").fetchone()
    assert tuple(lane) == ("exhausted", "all_candidates_terminal")
    conn.close()
