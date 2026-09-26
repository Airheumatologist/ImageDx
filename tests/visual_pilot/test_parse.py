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


def test_in_text_mentions_deterministic_across_parses():
    """Regression: mention marks used to key on id() of transient lxml
    proxies, which get recycled after GC — identical XML could yield
    different mention windows across runs."""
    import gc

    xml = FIXTURE.read_text(encoding="utf-8")
    baseline = {
        f.fig_id: list(f.in_text_mentions) for f in jats.parse_article(xml).figures
    }
    for _ in range(3):
        gc.collect()
        again = {
            f.fig_id: list(f.in_text_mentions)
            for f in jats.parse_article(xml).figures
        }
        assert again == baseline


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


@pytest.fixture(autouse=True)
def _clear_parse_caches():
    """parse keeps process-local caches (_JATS_CACHE/_SECTIONS_CACHE); reset
    them per test so mocked bundles never leak across tests."""
    parse._JATS_CACHE.clear()
    parse._SECTIONS_CACHE.clear()
    yield


@pytest.fixture()
def bundle_mock(monkeypatch):
    """Route pmc.get_article_bundle to an in-memory fixture bundle."""
    xml = FIXTURE.read_text(encoding="utf-8")
    files = {"fig1.jpg", "fig3.jpg", "fig4.jpg", "fig5.jpg"}
    monkeypatch.setattr(
        pmc, "get_article_bundle",
        lambda pmcid, **_kw: _Bundle(pmcid, xml, files),
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


# ---------------------------------------------------------------------------
# W5: parallel parse, hinted bundles, C6 outputs
# ---------------------------------------------------------------------------
_ARTICLE_CMP_COLS = (
    "status",
    "study_region",
    "error",
    "s3_prefix",
    "media_files_json",
    "authors_json",
    "author_count",
    "journal_name",
)


def _figures_snapshot(conn) -> dict:
    """figures rows keyed by figure_id, excluding timestamp columns."""
    cols = [
        c for c in db.table_columns(conn, "figures")
        if c not in {"created_at", "updated_at"}
    ]
    return {
        r["figure_id"]: {c: r[c] for c in cols}
        for r in conn.execute("SELECT * FROM figures")
    }


def _articles_snapshot(conn) -> dict:
    return {
        r["pmcid"]: {c: r[c] for c in _ARTICLE_CMP_COLS}
        for r in conn.execute("SELECT * FROM articles")
    }


def test_parallel_parse_matches_sequential_reference(
    conn, vp_data_dir, bundle_mock, tmp_path
):
    """B4/C6: pooled fetch+parse must produce rows identical to sequential."""
    pmcids = [f"PMC{i}" for i in range(5)]
    for i, pmcid in enumerate(pmcids):
        _insert_article(conn, pmcid, score=float(5 - i))
    assert parse.run(_args()) == 0
    parallel_figures = _figures_snapshot(conn)
    parallel_articles = _articles_snapshot(conn)

    conn2 = db.init_db(db.connect(tmp_path / "seq.sqlite"))
    try:
        for i, pmcid in enumerate(pmcids):
            _insert_article(conn2, pmcid, score=float(5 - i))
        stats: dict = {}
        for row in parse.select_batch(conn2, "sle", 50):
            parse.parse_article(conn2, row, stats)
        sequential_figures = _figures_snapshot(conn2)
        sequential_articles = _articles_snapshot(conn2)
    finally:
        conn2.close()

    assert parallel_figures == sequential_figures
    assert parallel_articles == sequential_articles


def test_parse_error_isolation_parallel(conn, monkeypatch):
    """One raising article must not fail the rest of the batch."""
    xml = FIXTURE.read_text(encoding="utf-8")
    files = {"fig1.jpg", "fig3.jpg", "fig4.jpg", "fig5.jpg"}

    def fake_bundle(pmcid, **_kw):
        if pmcid == "PMCBAD":
            raise pmc.PmcError("boom")
        return _Bundle(pmcid, xml, files)

    monkeypatch.setattr(pmc, "get_article_bundle", fake_bundle)
    _insert_article(conn, "PMCOK", score=2.0)
    _insert_article(conn, "PMCBAD", score=1.0)
    assert parse.run(_args()) == 0
    statuses = {
        r["pmcid"]: r["status"]
        for r in conn.execute("SELECT pmcid, status FROM articles")
    }
    assert statuses == {"PMCOK": "parsed", "PMCBAD": "parse_error"}
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM figures WHERE pmcid='PMCOK'"
    ).fetchone()["n"] == 7


def test_parse_uses_hinted_bundle_when_s3_prefix_present(conn, monkeypatch):
    """C3: rows carrying s3_prefix/media_files_json fetch via the hint path."""
    xml = FIXTURE.read_text(encoding="utf-8")
    files = {"fig1.jpg", "fig3.jpg", "fig4.jpg", "fig5.jpg"}
    calls = []

    def spy(pmcid, **kwargs):
        calls.append((pmcid, kwargs))
        return _Bundle(pmcid, xml, files)

    monkeypatch.setattr(pmc, "get_article_bundle", spy)
    _insert_article(conn, "PMCHINT")
    conn.execute(
        "UPDATE articles SET s3_prefix='PMCHINT.2', media_files_json=? "
        "WHERE pmcid='PMCHINT'",
        (db.to_json(sorted(files)),),
    )
    conn.commit()
    assert parse.run(_args()) == 0
    assert calls
    assert all(
        call == (
            "PMCHINT",
            {"prefix": "PMCHINT.2", "media_files": sorted(files)},
        )
        for call in calls
    )


def test_parse_persists_c6_article_metadata(conn, bundle_mock):
    """C6: parse fills s3_prefix/media_files/authors/journal on the row."""
    _insert_article(conn, "PMC1")
    assert parse.run(_args()) == 0
    row = conn.execute(
        "SELECT * FROM articles WHERE pmcid='PMC1'"
    ).fetchone()
    assert row["status"] == "parsed"
    assert db.from_json(row["authors_json"]) == ["Smith", "Doe", "Roe"]
    assert row["author_count"] == 4
    assert row["journal_name"] == "Journal of Test Rheumatology"
    # Derived from the bundle's resolved refs (mock metadata is empty):
    # image URLs look like {S3_BASE}/PMC1.1/{file}.
    assert row["s3_prefix"] == "PMC1.1"
    assert db.from_json(row["media_files_json"]) == [
        "fig1.jpg",
        "fig3.jpg",
        "fig4.jpg",
        "fig5.jpg",
    ]


def test_parse_preserves_existing_hints(conn, bundle_mock):
    """C6: s3_prefix/media_files_json already set (e.g. by license) are kept."""
    _insert_article(conn, "PMC1")
    conn.execute(
        "UPDATE articles SET s3_prefix='PMC1.9', media_files_json='[\"x.jpg\"]' "
        "WHERE pmcid='PMC1'"
    )
    conn.commit()
    assert parse.run(_args()) == 0
    row = conn.execute(
        "SELECT s3_prefix, media_files_json FROM articles WHERE pmcid='PMC1'"
    ).fetchone()
    assert row["s3_prefix"] == "PMC1.9"
    assert db.from_json(row["media_files_json"]) == ["x.jpg"]


def test_sections_for_returns_sections_after_parse(conn, bundle_mock):
    """C6: sections_for serves body_sections of in-process parsed articles."""
    assert parse.sections_for("PMC1") is None
    _insert_article(conn, "PMC1")
    assert parse.run(_args()) == 0
    sections = parse.sections_for("PMC1")
    assert sections is not None
    titles = [title for title, _ in sections]
    assert "Cutaneous findings" in titles
    assert "Diagnosis" in titles
    assert parse.sections_for("PMC_NEVER_PARSED") is None


def test_select_batch_keeps_peeked_unselected_bundles(conn, bundle_mock):
    """C6: peeked-but-unselected bundles stay cached across batches; the
    selected PMCIDs and their order are unchanged for the same DB state."""
    for i in range(4):
        _insert_article(conn, f"PMC{i}", score=float(4 - i))
    first = [r["pmcid"] for r in parse.select_batch(conn, "sle", 2)]
    assert first == ["PMC0", "PMC1"]
    # peek_limit = min(2*2, 100) = 4: all four were peeked, the unselected
    # pair must remain cached for the next batch instead of being dropped.
    assert {"PMC2", "PMC3"} <= set(parse._JATS_CACHE)
    second = [r["pmcid"] for r in parse.select_batch(conn, "sle", 2)]
    assert second == ["PMC0", "PMC1"]
