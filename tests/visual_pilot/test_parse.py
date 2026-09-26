"""W5a tests for src/visual_pilot/{jats,parse}.py — no network."""

import argparse
import json
from pathlib import Path

import pytest

from src.visual_pilot import db, jats, parse, pmc
from src.visual_pilot import config as vp_config

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "visual_pilot_sample.jats.xml"
)


@pytest.fixture(scope="module")
def parsed() -> jats.ParsedArticle:
    return jats.parse_article(FIXTURE.read_text(encoding="utf-8"))


def _fig(parsed: jats.ParsedArticle, fig_id: str) -> jats.FigureInfo:
    return next(f for f in parsed.figures if f.fig_id == fig_id)


# ---------------------------------------------------------------------------
# Parser against the fixture
# ---------------------------------------------------------------------------
def test_figure_ids_and_fallback(parsed):
    ids = [f.fig_id for f in parsed.figures]
    assert ids == ["f1", "f2", "f3", "f4", "f5", "fig6", "f7"]


def test_caption_includes_title_and_paragraphs(parsed):
    caption = _fig(parsed, "f1").caption
    assert caption.startswith("Malar rash and other cutaneous findings")
    assert "First paragraph of the caption" in caption
    assert "Second paragraph adding histologic detail" in caption


def test_in_text_mentions_multi_rid_and_window(parsed):
    f1 = _fig(parsed, "f1")
    assert len(f1.in_text_mentions) == 2
    long_mention = f1.in_text_mentions[0]
    assert len(long_mention) <= 600
    assert "Figures 1 and 2" in long_mention
    # f2 shares the same multi-rid paragraph.
    f2 = _fig(parsed, "f2")
    assert f2.in_text_mentions and "Figures 1 and 2" in f2.in_text_mentions[0]


def test_third_party_permission_wording(parsed):
    f3 = _fig(parsed, "f3")
    assert f3.third_party is True
    assert "permission" in (f3.third_party_reason or "").lower()


def test_third_party_foreign_holder(parsed):
    f4 = _fig(parsed, "f4")
    assert f4.third_party is True
    assert "International Society of Radiographers" in f4.third_party_reason


def test_fig_level_license_and_non_third_party(parsed):
    f5 = _fig(parsed, "f5")
    assert "by-nd" in (f5.fig_license_raw or "")
    assert f5.third_party is False
    f1 = _fig(parsed, "f1")
    assert f1.fig_license_raw is None
    assert f1.third_party is False


def test_no_graphic_figure(parsed):
    assert _fig(parsed, "f7").graphic_href is None
    assert _fig(parsed, "f1").graphic_href == "fig1.jpg"


def test_article_metadata(parsed):
    assert parsed.article_copyright_holder == "Test Publisher Inc"
    assert parsed.journal_name == "Journal of Test Rheumatology"
    assert parsed.publisher_name == "Test Publisher Inc"
    assert parsed.authors == ["Smith", "Doe", "Roe"]
    assert parsed.author_count == 4
    assert parsed.corresp_country == "Canada"
    assert parsed.first_aff_country == "Germany"
    titles = [t for t, _ in parsed.body_sections]
    assert "Cutaneous findings" in titles
    assert "Diagnosis" in titles


def test_parser_handles_missing_namespaces():
    xml = FIXTURE.read_text(encoding="utf-8").replace(
        ' xmlns:xlink="http://www.w3.org/1999/xlink"', ""
    ).replace("xlink:href", "href")
    parsed = jats.parse_article(xml)
    assert _fig(parsed, "f1").graphic_href == "fig1.jpg"
    assert _fig(parsed, "f5").fig_license_raw


# ---------------------------------------------------------------------------
# Runner (parse.run) with a mocked bundle
# ---------------------------------------------------------------------------
class _Bundle:
    def __init__(self, pmcid, xml, files):
        self.pmcid = pmcid
        self.xml_text = xml
        self.license = pmc.LicenseInfo(code="cc-by", url=None, oa_subset="oa")
        self.metadata = {}
        self._files = set(files)

    def resolver(self, href):
        name = href.rsplit("/", 1)[-1]
        if name in self._files:
            return pmc.ImageRef(
                url=f"https://pmc-oa-opendata.s3.amazonaws.com/{self.pmcid}.1/{name}",
                needs_bytes=False,
                format="jpg",
            )
        return pmc.ImageRef(url=None, needs_bytes=True, format=name.rpartition(".")[2] or None)


def _args(**over):
    base = dict(
        disease="all",
        limit=None,
        dry_run=False,
        budget_usd=None,
        cap=None,
        accept_cap=False,
        pmcids=None,
        batch_size=50,
        max_articles=600,
        max_runtime_seconds=900,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _insert_article(conn, pmcid, keys=("sle",), score=1.0, status="relevant", country="Canada", license_code="cc-by"):
    conn.execute(
        "INSERT INTO articles (pmcid, title, country, license_code, "
        "retrieval_score, primary_disease_keys_json, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (pmcid, f"title {pmcid}", country, license_code, score,
         db.to_json(list(keys)), status),
    )
    conn.commit()


@pytest.fixture()
def bundle_mock(monkeypatch):
    """Route pmc.get_article_bundle to an in-memory fixture bundle."""
    xml = FIXTURE.read_text(encoding="utf-8")
    files = {"fig1.jpg", "fig3.jpg", "fig4.jpg", "fig5.jpg"}
    monkeypatch.setattr(
        pmc, "get_article_bundle", lambda pmcid: _Bundle(pmcid, xml, files)
    )


def test_parse_runner_writes_figure_rows(conn, bundle_mock):
    _insert_article(conn, "PMC1", country="Canada")
    assert parse.run(_args()) == 0

    figs = {
        r["figure_id"]: dict(r)
        for r in conn.execute("SELECT * FROM figures WHERE pmcid='PMC1'")
    }
    assert set(figs) == {
        "PMC1:f1", "PMC1:f2", "PMC1:f3", "PMC1:f4", "PMC1:f5", "PMC1:fig6", "PMC1:f7"
    }
    # f1: pending with a resolved image URL
    assert figs["PMC1:f1"]["status"] == "pending"
    assert figs["PMC1:f1"]["image_url"].endswith("/PMC1.1/fig1.jpg")
    assert figs["PMC1:f1"]["effective_license"] == "cc-by"
    mentions = db.from_json(figs["PMC1:f1"]["in_text_mentions_json"])
    assert len(mentions) == 2
    # f2: graphic exists in XML but not on S3 -> needs_bytes, still pending
    assert figs["PMC1:f2"]["status"] == "pending"
    assert figs["PMC1:f2"]["image_url"] is None
    assert figs["PMC1:f2"]["error"] == "needs_bytes"
    # f3/f4: third-party pre-rejects
    for fid in ("PMC1:f3", "PMC1:f4"):
        assert figs[fid]["status"] == "caption_rejected"
        triage = db.from_json(figs[fid]["triage_json"])
        assert triage["route"] == "drop"
        assert triage["reason"] == "third_party"
        assert triage["source"] == "parse"
    assert "International Society of Radiographers" in (
        db.from_json(figs["PMC1:f4"]["triage_json"])["third_party_quote"]
    )
    # f5: fig-level CC BY-ND -> effective_license nd, allowed (whole figure)
    assert figs["PMC1:f5"]["status"] == "pending"
    assert figs["PMC1:f5"]["effective_license"] == "cc-by-nd"
    # f7: no graphic -> caption_rejected no_graphic
    triage = db.from_json(figs["PMC1:f7"]["triage_json"])
    assert figs["PMC1:f7"]["status"] == "caption_rejected"
    assert triage["reason"] == "no_graphic"

    article = conn.execute("SELECT * FROM articles WHERE pmcid='PMC1'").fetchone()
    assert article["status"] == "parsed"
    assert article["study_region"] == "Canada (article metadata)"

    # Resumable: rerun parses nothing new.
    assert parse.run(_args()) == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM figures").fetchone()["n"] == 7


def test_parse_runner_license_reject_and_region_fallbacks(conn, monkeypatch):
    xml = FIXTURE.read_text(encoding="utf-8")
    bundle = _Bundle("PMCN", xml, {"fig1.jpg"})
    monkeypatch.setattr(pmc, "get_article_bundle", lambda pmcid: bundle)
    _insert_article(
        conn, "PMCN", keys=("dm",), country=None, license_code="cc-by-nc"
    )
    assert parse.run(_args(disease="dm")) == 0
    fig = conn.execute(
        "SELECT * FROM figures WHERE figure_id='PMCN:f1'"
    ).fetchone()
    assert fig["status"] == "caption_rejected"
    assert db.from_json(fig["triage_json"])["reason"] == "license"
    article = conn.execute("SELECT * FROM articles WHERE pmcid='PMCN'").fetchone()
    assert article["study_region"] == "Canada (corresponding-author affiliation)"


def test_parse_runner_parse_error(conn, monkeypatch):
    def boom(pmcid):
        raise pmc.PmcNotFoundError("404 for x")

    monkeypatch.setattr(pmc, "get_article_bundle", boom)
    _insert_article(conn, "PMCERR")
    assert parse.run(_args()) == 0
    article = conn.execute("SELECT * FROM articles WHERE pmcid='PMCERR'").fetchone()
    assert article["status"] == "parse_error"
    assert "404" in article["error"]


def test_specific_eligible_figure_rescues_broad_review(conn):
    _insert_article(conn, "PMCRESCUE", keys=(), score=0.1, status="irrelevant")
    article = dict(conn.execute(
        "SELECT * FROM articles WHERE pmcid='PMCRESCUE'"
    ).fetchone())
    article["caption_candidates"] = [{
        "caption": "Clinical photographs showing Gottron papules in dermatomyositis",
        "eligible": True,
    }]
    assert parse._rescue_if_caption_matches(conn, article, "dm") is True
    saved = conn.execute(
        "SELECT status, primary_disease_keys_json, relevance_reason "
        "FROM articles WHERE pmcid='PMCRESCUE'"
    ).fetchone()
    assert saved["status"] == "relevant"
    assert "dm" in db.from_json(saved["primary_disease_keys_json"])
    assert saved["relevance_reason"] == "visual_figure_caption_rescue"


def test_visual_rescue_respects_article_license(conn):
    _insert_article(
        conn, "PMCBADLIC", keys=(), score=0.1, status="irrelevant",
        license_code="cc-by-nc",
    )
    article = dict(conn.execute(
        "SELECT * FROM articles WHERE pmcid='PMCBADLIC'"
    ).fetchone())
    article["caption_candidates"] = [{
        "caption": "Clinical photographs showing Gottron papules in dermatomyositis",
        "eligible": True,
    }]
    assert parse._rescue_if_caption_matches(conn, article, "dm") is False
    assert conn.execute(
        "SELECT status FROM articles WHERE pmcid='PMCBADLIC'"
    ).fetchone()["status"] == "irrelevant"


def test_parse_uses_resumable_batches_without_fixed_cap(conn, vp_data_dir, bundle_mock):
    reports = vp_config.reports_dir()
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage2_counts.json").write_text(
        json.dumps({"sle": {"over_cap": True, "relevant": 200, "cap": 150}})
    )
    for i in range(3):
        _insert_article(conn, f"PMC{i}", keys=("sle",), score=float(3 - i))
    # A batch processes the top two, and the next invocation resumes at PMC2.
    assert parse.run(_args(batch_size=2)) == 0
    parsed = {
        r["pmcid"]
        for r in conn.execute("SELECT pmcid FROM articles WHERE status='parsed'")
    }
    assert parsed == {"PMC0", "PMC1"}
    assert parse.run(_args(batch_size=2)) == 0
    parsed = {
        r["pmcid"]
        for r in conn.execute("SELECT pmcid FROM articles WHERE status='parsed'")
    }
    assert parsed == {"PMC0", "PMC1", "PMC2"}


def test_articles_migration_idempotent(vp_data_dir):
    conn = db.connect()
    try:
        conn.execute("CREATE TABLE articles (pmcid TEXT PRIMARY KEY, status TEXT)")
        db.init_db(conn)
        cols = set(db.table_columns(conn, "articles"))
        assert {"study_region", "error"} <= cols
        db.init_db(conn)  # second init must not fail on existing columns
        db.init_db(conn)
    finally:
        conn.close()


def test_parse_dry_run_no_writes(conn, bundle_mock, capsys):
    _insert_article(conn, "PMC1")
    assert parse.run(_args(dry_run=True)) == 0
    assert "dry-run" in capsys.readouterr().out
    assert conn.execute("SELECT COUNT(*) AS n FROM figures").fetchone()["n"] == 0
    assert conn.execute(
        "SELECT status FROM articles WHERE pmcid='PMC1'"
    ).fetchone()["status"] == "relevant"


def test_stages_3_4_write_no_files_beyond_db_and_reports(
    conn, vp_data_dir, bundle_mock, monkeypatch
):
    """Spec §9: stages 3-4 must not write full text or images to disk."""
    from src.visual_pilot import llm, triage

    class _OneShotLLM:
        def __init__(self, **kwargs):
            self.spent_usd = 0.0

        def call_many(self, requests):
            reqs = list(requests)
            return [
                llm.BatchResult(index=i, parsed={"results": []}, meta={})
                for i, _ in enumerate(reqs)
            ]

    monkeypatch.setattr(llm, "LLMClient", lambda **kw: _OneShotLLM())
    _insert_article(conn, "PMC1")
    parse.run(_args())
    triage.run(_args())

    allowed = {"visual_pilot.sqlite", "visual_pilot.sqlite-wal", "visual_pilot.sqlite-shm"}
    unexpected = []
    for path in vp_data_dir.rglob("*"):
        rel = path.relative_to(vp_data_dir)
        if path.is_dir() or rel.parts[0] == "reports" or str(rel) in allowed:
            continue
        unexpected.append(str(rel))
    assert unexpected == []
