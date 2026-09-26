"""Deterministic figure-priority tests; no image or network calls."""

from src.visual_pilot import db, diseases, judge


def _record(conn, pmcid, figure_id, disease, caption, status="caption_uncertain", license_code="cc-by", mentions=None):
    conn.execute(
        "INSERT OR REPLACE INTO articles (pmcid,title,status,primary_disease_keys_json) "
        "VALUES (?,?, 'parsed', ?)",
        (pmcid, f"Review of {disease}", db.to_json([disease])),
    )
    conn.execute(
        "INSERT OR REPLACE INTO figures (figure_id,pmcid,label,caption,status,effective_license,image_url,in_text_mentions_json) "
        "VALUES (?,?, 'Figure 1',?,?,?,'https://example.test/fig.png',?)",
        (figure_id, pmcid, caption, status, license_code, db.to_json(mentions or [])),
    )
    return dict(conn.execute("SELECT * FROM figures WHERE figure_id=?", (figure_id,)).fetchone())


def test_figure_rank_uses_finding_modality_and_uncovered_findings(conn):
    diseases.seed(conn)
    weak = _record(conn, "PMC_A", "PMC_A:f1", "dm", "Schematic of immune pathways in dermatomyositis")
    strong = _record(
        conn, "PMC_B", "PMC_B:f1", "dm",
        "Clinical photographs show Gottron papules and heliotrope rash in dermatomyositis.",
        mentions=["Figure 1 shows patient photographs of Gottron papules."],
    )
    articles = {r["pmcid"]: dict(r) for r in conn.execute("SELECT pmcid,title,primary_disease_keys_json FROM articles")}
    ranked = judge.rank_figures(conn, [weak, strong], articles)
    assert [r["figure_id"] for r in ranked] == ["PMC_B:f1", "PMC_A:f1"]
    assert ranked[0]["_priority_components"]["matched_findings"]


def test_uncertain_figures_remain_eligible_and_allowed_license_scores_higher(conn):
    diseases.seed(conn)
    uncertain = _record(conn, "PMC_C", "PMC_C:f1", "as", "MRI shows sacroiliac bone marrow edema in axial spondyloarthritis")
    nd = _record(conn, "PMC_D", "PMC_D:f1", "as", "MRI shows sacroiliac bone marrow edema in axial spondyloarthritis", license_code="cc-by-nc")
    articles = {r["pmcid"]: dict(r) for r in conn.execute("SELECT pmcid,title,primary_disease_keys_json FROM articles")}
    ranked = judge.rank_figures(conn, [nd, uncertain], articles)
    assert len(ranked) == 2
    assert ranked[0]["figure_id"] == "PMC_C:f1"
    assert any(f["status"] == "caption_uncertain" for f in ranked)


def test_article_level_license_is_inherited_when_figure_license_is_missing(conn):
    diseases.seed(conn)
    figure = _record(
        conn, "PMC_INHERIT", "PMC_INHERIT:f1", "as",
        "MRI shows sacroiliac erosions in axial spondyloarthritis.", license_code=None,
    )
    # _record creates an article without a license column value. Set the
    # article license to the parser's inherited-license source.
    conn.execute("UPDATE articles SET license_code='cc-by' WHERE pmcid='PMC_INHERIT'")
    article = dict(conn.execute(
        "SELECT pmcid,title,primary_disease_keys_json,license_code FROM articles WHERE pmcid='PMC_INHERIT'"
    ).fetchone())
    _, parts = judge.figure_priority(conn, figure, article)
    assert parts["license"] == 1.0


def test_coverage_bonus_favors_findings_not_yet_in_library(conn):
    diseases.seed(conn)
    edema = _record(conn, "PMC_E", "PMC_E:f1", "as", "MRI showing sacroiliac bone marrow edema in axial spondyloarthritis")
    erosion = _record(conn, "PMC_F", "PMC_F:f1", "as", "MRI showing sacroiliac joint erosions in axial spondyloarthritis")
    conn.execute(
        "INSERT OR REPLACE INTO panels (panel_id,figure_id,pmcid,disease_key,findings_json) "
        "VALUES ('existing','PMC_E:f1','PMC_E','as',?)", (db.to_json([{"finding_key":"si_bone_marrow_edema"}]),)
    )
    articles = {r["pmcid"]: dict(r) for r in conn.execute("SELECT pmcid,title,primary_disease_keys_json FROM articles")}
    ranked = judge.rank_figures(conn, [edema, erosion], articles)
    assert ranked[0]["figure_id"] == "PMC_F:f1"
