"""W8 tests: extract reuses in-memory sections, vocab cache, P4 iter_many."""

from argparse import Namespace
from pathlib import Path

import pytest

from src.visual_pilot import config, diseases, extract_findings, jats, llm, parse, pmc
from src.visual_pilot.llm import BatchResult

FIXTURE_XML = (
    Path(__file__).resolve().parents[1] / "fixtures" / "visual_pilot_sample.jats.xml"
).read_text()


@pytest.fixture(autouse=True)
def _clean_sections_cache():
    """parse._SECTIONS_CACHE is process-global: isolate it per test."""
    parse._SECTIONS_CACHE.clear()
    yield
    parse._SECTIONS_CACHE.clear()


def _args(**kw):
    base = dict(
        disease="all", limit=None, dry_run=False, budget_usd=None,
        pmcids=None, cap=None, accept_cap=False, force=False,
    )
    base.update(kw)
    return Namespace(**base)


def _article(conn, pmcid="PMC1", disease_keys='["dm"]', s3_prefix=None, media_files=None):
    diseases.seed(conn)  # findings_vocab / disease_findings FKs
    conn.execute(
        "INSERT OR REPLACE INTO articles (pmcid, title, journal, year, doi, status, "
        "license_code, license_url, study_region, primary_disease_keys_json, "
        "s3_prefix, media_files_json) "
        "VALUES (?, 'T', 'J', 2024, '10.1/x', 'parsed', 'cc-by', "
        "'https://creativecommons.org/licenses/by/4.0/', 'Spain (article metadata)', "
        "?, ?, ?)",
        (pmcid, disease_keys, s3_prefix, media_files),
    )
    conn.commit()
    return dict(conn.execute("SELECT * FROM articles WHERE pmcid=?", (pmcid,)).fetchone())


def _bundle(xml=FIXTURE_XML):
    def _make(pmcid, **kw):
        return pmc.ArticleBundle(
            pmcid=pmcid,
            xml_text=xml,
            resolver=lambda h: pmc.ImageRef(url=None, needs_bytes=True),
        )
    return _make


def _boom(*a, **kw):
    raise AssertionError("get_article_bundle must not be called")


def _input_hash(client, request):
    return client._input_hash(
        request["stage"],
        request["model"],
        request["prompt_version"],
        request["system"],
        request["user_content"],
        [],
    )


# ---------------------------------------------------------------------------
# sections_for reuse vs hinted refetch — identical P4 request (parity-critical)
# ---------------------------------------------------------------------------
def test_prepare_request_identical_memory_vs_refetch(conn, monkeypatch):
    article = _article(conn, s3_prefix="PMC1.9", media_files='["f1.png", "f2.png"]')
    vocab = extract_findings._vocabulary(conn, ["dm"])

    # Path A: sections_for miss -> hinted refetch + jats parse.
    bundle_calls = []

    def _recording_bundle(pmcid, **kw):
        bundle_calls.append((pmcid, kw))
        return _bundle()(pmcid, **kw)

    monkeypatch.setattr(parse, "sections_for", lambda pmcid: None)
    monkeypatch.setattr(pmc, "get_article_bundle", _recording_bundle)
    ctx_refetch = extract_findings.prepare_article(article, vocab)
    assert ctx_refetch is not None
    # Refetch uses the persisted hints (no listing round-trip upstream).
    assert bundle_calls == [
        ("PMC1", {"prefix": "PMC1.9", "media_files": ["f1.png", "f2.png"]})
    ]

    # Path B: sections_for hit -> zero fetches, same sections.
    real_sections = jats.parse_article(FIXTURE_XML).body_sections
    monkeypatch.setattr(parse, "sections_for", lambda pmcid: real_sections)
    monkeypatch.setattr(pmc, "get_article_bundle", _boom)
    ctx_cached = extract_findings.prepare_article(article, vocab)
    assert ctx_cached is not None

    # Picked sections, source text and the full P4 request are identical;
    # therefore the input_hash is byte-identical either way.
    assert ctx_cached["sections"] == ctx_refetch["sections"]
    assert ctx_cached["source_text"] == ctx_refetch["source_text"]
    assert ctx_cached["request"] == ctx_refetch["request"]
    client = llm.LLMClient(db_conn=None)
    assert _input_hash(client, ctx_cached["request"]) == _input_hash(
        client, ctx_refetch["request"]
    )


def test_prepare_miss_without_hints(conn, monkeypatch):
    """No persisted hints -> get_article_bundle falls back to full lookup."""
    article = _article(conn)  # s3_prefix / media_files_json are NULL
    vocab = extract_findings._vocabulary(conn, ["dm"])
    bundle_calls = []

    def _recording_bundle(pmcid, **kw):
        bundle_calls.append((pmcid, kw))
        return _bundle()(pmcid, **kw)

    monkeypatch.setattr(parse, "sections_for", lambda pmcid: None)
    monkeypatch.setattr(pmc, "get_article_bundle", _recording_bundle)
    ctx = extract_findings.prepare_article(article, vocab)
    assert ctx is not None
    assert bundle_calls == [("PMC1", {"prefix": None, "media_files": None})]


def test_prepare_no_sections_returns_none(conn, monkeypatch):
    xml = "<article><body></body></article>"
    article = _article(conn)
    vocab = extract_findings._vocabulary(conn, ["dm"])
    monkeypatch.setattr(parse, "sections_for", lambda pmcid: None)
    monkeypatch.setattr(pmc, "get_article_bundle", _bundle(xml))
    assert extract_findings.prepare_article(article, vocab) is None


# ---------------------------------------------------------------------------
# run(): vocabulary cache + streaming P4
# ---------------------------------------------------------------------------
def _p4_response(quote="illustrated in Figures 1 and 2"):
    return {
        "assertions": [
            {
                "disease_key": "dm",
                "subtype": None,
                "finding_key": "gottron_papules",
                "proposed_finding": None,
                "frequency_text": "common",
                "pct_low": None,
                "pct_high": None,
                "specificity_text": None,
                "quote": quote,
            }
        ]
    }


class _IterClient:
    """iter_many stub: records requests, replays canned per-index results."""

    def __init__(self, results=None, default=None, reverse=False):
        self.requests = []
        self.spent_usd = 0.0
        self.concurrency = 4
        self.results = results or {}
        self.default = default if default is not None else _p4_response()
        self.reverse = reverse
        self.saw_iterable_type = None

    def iter_many(self, requests, max_in_flight=None):
        self.saw_iterable_type = type(requests).__name__
        out = []
        for i, req in enumerate(requests):
            self.requests.append(req)
            out.append(
                self.results.get(
                    i,
                    BatchResult(index=i, parsed=self.default, meta={"cached": False}),
                )
            )
        if self.reverse:
            out.reverse()
        yield from out


def test_run_vocab_cache_once_per_disease_tuple(conn, monkeypatch):
    _article(conn, "PMC1")
    _article(conn, "PMC2")
    _article(conn, "PMC3", disease_keys='["sle"]')
    monkeypatch.setattr(parse, "sections_for", lambda pmcid: None)
    monkeypatch.setattr(pmc, "get_article_bundle", _bundle())

    calls = {"n": 0}
    real_vocab = extract_findings._vocabulary

    def _counting(conn_, keys):
        calls["n"] += 1
        return real_vocab(conn_, keys)

    monkeypatch.setattr(extract_findings, "_vocabulary", _counting)
    client = _IterClient()
    monkeypatch.setattr(extract_findings.llm, "LLMClient", lambda **kw: client)

    assert extract_findings.run(_args()) == 0
    # {"dm"} shared by PMC1+PMC2 computes once; {"sle"} once more.
    assert calls["n"] == 2
    assert len(client.requests) == 3
    n = conn.execute(
        "SELECT COUNT(*) n FROM disease_findings WHERE source='text'"
    ).fetchone()["n"]
    assert n == 3


def test_run_streams_requests_and_maps_completion_order(conn, monkeypatch):
    _article(conn, "PMC1")
    _article(conn, "PMC2")
    _article(conn, "PMC3")

    def _flaky_bundle(pmcid, **kw):
        if pmcid == "PMC2":
            raise pmc.PmcError("503 boom")
        return _bundle()(pmcid, **kw)

    monkeypatch.setattr(parse, "sections_for", lambda pmcid: None)
    monkeypatch.setattr(pmc, "get_article_bundle", _flaky_bundle)
    # Second consumed request (PMC3 — PMC2 never yields one) errors, and
    # results stream back in reversed completion order.
    client = _IterClient(
        results={1: BatchResult(index=1, error=llm.LLMError("provider boom"))},
        reverse=True,
    )
    monkeypatch.setattr(extract_findings.llm, "LLMClient", lambda **kw: client)

    assert extract_findings.run(_args()) == 0
    # Requests arrive lazily as a generator, not a prebuilt list.
    assert client.saw_iterable_type == "generator"
    # PMC2's prepare error consumed a stats error; only PMC1+PMC3 sent P4.
    assert len(client.requests) == 2
    pmcids = {
        r["pmcid"]
        for r in conn.execute(
            "SELECT DISTINCT pmcid FROM disease_findings WHERE source='text'"
        )
    }
    # PMC1 applied (index 0); PMC3's llm error left it untouched.
    assert pmcids == {"PMC1"}


def test_run_cache_only_miss_counts_errors(conn, monkeypatch):
    """VP_LLM_CACHE_ONLY: a cache miss surfaces as a per-article error, like
    any other call failure — nothing is written."""
    _article(conn)
    monkeypatch.setattr(parse, "sections_for", lambda pmcid: None)
    monkeypatch.setattr(pmc, "get_article_bundle", _bundle())
    monkeypatch.setattr(config, "VP_LLM_CACHE_ONLY", 1)
    client = llm.LLMClient(db_conn=conn)  # real client, empty llm_calls
    monkeypatch.setattr(extract_findings.llm, "LLMClient", lambda **kw: client)

    assert extract_findings.run(_args()) == 0
    assert (
        conn.execute("SELECT COUNT(*) n FROM disease_findings").fetchone()["n"] == 0
    )
    assert conn.execute("SELECT COUNT(*) n FROM llm_calls").fetchone()["n"] == 0


def test_run_writes_no_article_text_or_images(conn, vp_data_dir, monkeypatch):
    """Sections stay in memory: nothing but the sqlite DB hits the data dir."""
    _article(conn)
    real_sections = jats.parse_article(FIXTURE_XML).body_sections
    monkeypatch.setattr(parse, "sections_for", lambda pmcid: real_sections)
    monkeypatch.setattr(pmc, "get_article_bundle", _boom)
    client = _IterClient()
    monkeypatch.setattr(extract_findings.llm, "LLMClient", lambda **kw: client)

    assert extract_findings.run(_args()) == 0
    files = {p.name for p in vp_data_dir.rglob("*") if p.is_file()}
    assert files <= {
        "visual_pilot.sqlite",
        "visual_pilot.sqlite-wal",
        "visual_pilot.sqlite-shm",
    }
