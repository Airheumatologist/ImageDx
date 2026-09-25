"""W1 foundations tests: §4 schema, idempotent init/seed, status helpers."""

import sqlite3

import pytest

from src.visual_pilot import cli, db, diseases

EXPECTED_COLUMNS = {
    "diseases": {
        "disease_key",
        "name",
        "mondo_id",
        "mesh_id",
        "synonyms_json",
        "subtypes_json",
    },
    "findings_vocab": {
        "finding_key",
        "disease_keys_json",
        "label",
        "synonyms_json",
        "category",
        "approved",
        "proposed_by_llm",
        "proposal_count",
    },
    "articles": {
        "pmcid",
        "pmid",
        "doi",
        "title",
        "journal",
        "year",
        "country",
        "publication_types_json",
        "license_code",
        "license_url",
        "oa_subset",
        "retrieval_score",
        "primary_disease_keys_json",
        "relevance_decision",
        "relevance_reason",
        "study_region",
        "error",
        "status",
    },
    "figures": {
        "figure_id",
        "pmcid",
        "label",
        "caption",
        "in_text_mentions_json",
        "fig_permissions_text",
        "effective_license",
        "image_url",
        "image_format",
        "sha256",
        "status",
        "triage_json",
        "vision_json",
        "error",
        "attempts",
    },
    "panels": {
        "panel_id",
        "figure_id",
        "pmcid",
        "panel_label",
        "disease_key",
        "subtype",
        "modality",
        "body_site",
        "findings_json",
        "typicality",
        "stage",
        "age_group",
        "skin_tone",
        "stated_ethnicity",
        "stated_ethnicity_quote",
        "study_region",
        "annotations_present",
        "bbox_json",
        "crop_mode",
        "confidence",
        "rationale",
        "image_path",
        "thumb_path",
        "width",
        "height",
        "sha256",
        "attribution_text",
        "license_code",
        "license_url",
        "source_url",
    },
    "disease_findings": {
        "id",
        "disease_key",
        "finding_key",
        "subtype",
        "frequency_text",
        "frequency_pct_low",
        "frequency_pct_high",
        "source",
        "pmcid",
        "quote",
    },
    "llm_calls": {
        "call_id",
        "stage",
        "model",
        "input_hash",
        "request_meta_json",
        "response_json",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "created_at",
    },
}


def test_schema_has_every_spec_table_and_column(conn):
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, expected in EXPECTED_COLUMNS.items():
        assert table in tables, f"missing table {table}"
        actual = set(db.table_columns(conn, table))
        assert expected <= actual, f"{table} missing columns {expected - actual}"


def test_llm_calls_input_hash_is_unique(conn):
    conn.execute(
        "INSERT INTO llm_calls (stage, model, input_hash) VALUES ('triage', 'm', 'h1')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO llm_calls (stage, model, input_hash) VALUES ('triage', 'm', 'h1')"
        )


def test_llm_input_hash_covers_stage_and_model():
    base = db.llm_input_hash("triage", "m1", {"x": 1})
    assert base == db.llm_input_hash("triage", "m1", {"x": 1})
    assert base != db.llm_input_hash("judge", "m1", {"x": 1})
    assert base != db.llm_input_hash("triage", "m2", {"x": 1})
    assert base != db.llm_input_hash("triage", "m1", {"x": 2})


def test_init_db_is_idempotent(conn):
    db.init_db(conn)
    db.init_db(conn)
    assert conn.execute("SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table'").fetchone()["n"] >= len(
        EXPECTED_COLUMNS
    )


def test_seed_twice_gives_same_counts(conn):
    diseases.seed(conn)
    first = {
        table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        for table in ("diseases", "findings_vocab")
    }
    diseases.seed(conn)
    second = {
        table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        for table in ("diseases", "findings_vocab")
    }
    assert first == second
    assert first["diseases"] == 3
    assert first["findings_vocab"] > 0


def test_seed_preserves_llm_proposed_vocab_row(conn):
    diseases.seed(conn)
    conn.execute(
        "INSERT INTO findings_vocab "
        "(finding_key, disease_keys_json, label, synonyms_json, category, "
        " approved, proposed_by_llm, proposal_count) "
        "VALUES ('proposed_thing', '[\"sle\"]', 'Proposed thing', '[]', 'skin', 0, 1, 3)"
    )
    # Also simulate a proposal that reuses an existing (seed) key.
    conn.execute(
        "UPDATE findings_vocab SET approved = 0, proposed_by_llm = 1, proposal_count = 5 "
        "WHERE finding_key = 'malar_rash'"
    )
    diseases.seed(conn)
    row = conn.execute(
        "SELECT approved, proposed_by_llm, proposal_count FROM findings_vocab "
        "WHERE finding_key = 'proposed_thing'"
    ).fetchone()
    assert dict(row) == {"approved": 0, "proposed_by_llm": 1, "proposal_count": 3}
    row = conn.execute(
        "SELECT approved, proposed_by_llm, proposal_count FROM findings_vocab "
        "WHERE finding_key = 'malar_rash'"
    ).fetchone()
    assert dict(row) == {"approved": 0, "proposed_by_llm": 1, "proposal_count": 5}


def test_every_vocab_disease_key_is_valid(conn):
    diseases.seed(conn)
    valid = diseases.disease_keys(conn)
    for row in conn.execute("SELECT finding_key, disease_keys_json FROM findings_vocab"):
        keys = db.from_json(row["disease_keys_json"])
        assert keys, f"{row['finding_key']} has no disease keys"
        assert set(keys) <= valid, f"{row['finding_key']}: unknown keys {set(keys) - valid}"


def _insert_article(conn, pmcid, disease_keys, status="candidate"):
    conn.execute(
        "INSERT INTO articles (pmcid, title, primary_disease_keys_json, status) "
        "VALUES (?, ?, ?, ?)",
        (pmcid, f"title {pmcid}", db.to_json(disease_keys), status),
    )


def _insert_figure(conn, figure_id, pmcid, status="pending"):
    conn.execute(
        "INSERT INTO figures (figure_id, pmcid, label, status) VALUES (?, ?, ?, ?)",
        (figure_id, pmcid, "Fig 1", status),
    )


def _insert_panel(conn, panel_id, figure_id, pmcid):
    conn.execute(
        "INSERT INTO panels (panel_id, figure_id, pmcid, panel_label) VALUES (?, ?, ?, ?)",
        (panel_id, figure_id, pmcid, "A"),
    )


def test_set_status_and_extra_fields(conn):
    _insert_article(conn, "PMC1", ["sle"])
    db.set_status(conn, "articles", "PMC1", "license_ok", license_code="CC BY")
    row = conn.execute("SELECT status, license_code FROM articles WHERE pmcid='PMC1'").fetchone()
    assert (row["status"], row["license_code"]) == ("license_ok", "CC BY")


def test_set_status_unknown_row_and_column(conn):
    _insert_article(conn, "PMC1", ["sle"])
    with pytest.raises(KeyError):
        db.set_status(conn, "articles", "PMC_NOPE", "relevant")
    with pytest.raises(ValueError):
        db.set_status(conn, "articles", "PMC1", "relevant", nope=1)


def test_rows_with_status_and_disease_filter(conn):
    _insert_article(conn, "PMC_SLE", ["sle"], status="relevant")
    _insert_article(conn, "PMC_DM", ["dm"], status="relevant")
    _insert_article(conn, "PMC_BOTH", ["sle", "dm"], status="relevant")
    _insert_article(conn, "PMC_CAND", ["sle"], status="candidate")
    _insert_figure(conn, "PMC_SLE:f1", "PMC_SLE", status="caption_kept")
    _insert_figure(conn, "PMC_DM:f1", "PMC_DM", status="caption_kept")
    _insert_figure(conn, "PMC_BOTH:f1", "PMC_BOTH", status="pending")
    _insert_panel(conn, "P1", "PMC_SLE:f1", "PMC_SLE")
    _insert_panel(conn, "P2", "PMC_DM:f1", "PMC_DM")

    sle_articles = db.rows_with_status(conn, "articles", "relevant", disease="sle")
    assert {r["pmcid"] for r in sle_articles} == {"PMC_SLE", "PMC_BOTH"}

    all_relevant = db.rows_with_status(conn, "articles", ["relevant", "candidate"])
    assert len(all_relevant) == 4

    sle_figs = db.rows_with_status(conn, "figures", ["caption_kept", "pending"], disease="sle")
    assert {r["figure_id"] for r in sle_figs} == {"PMC_SLE:f1", "PMC_BOTH:f1"}

    dm_figs = db.rows_with_status(conn, "figures", "caption_kept", disease="dm")
    assert {r["figure_id"] for r in dm_figs} == {"PMC_DM:f1"}

    # Panels inherit pipeline status from their parent figure and disease
    # scope from their article via the joins.
    sle_panels = db.rows_with_status(conn, "panels", "caption_kept", disease="sle")
    assert {r["panel_id"] for r in sle_panels} == {"P1"}

    all_panels = db.rows_with_status(conn, "panels", "caption_kept")
    assert {r["panel_id"] for r in all_panels} == {"P1", "P2"}

    limited = db.rows_with_status(conn, "articles", "relevant", limit=2)
    assert len(limited) == 2


def test_cli_init_and_stub(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path / "vp"))
    assert cli.main(["init"]) == 0
    out = capsys.readouterr().out
    assert "diseases: 3" in out
    assert "findings_vocab:" in out
    assert (tmp_path / "vp" / "visual_pilot.sqlite").exists()

    # every subcommand is now implemented
    assert all(fn is not None for fn in cli.COMMANDS.values())
