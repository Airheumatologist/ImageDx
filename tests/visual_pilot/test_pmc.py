"""W2 tests for src/visual_pilot/pmc.py — no network (httpx.MockTransport)."""

import io
import time

import httpx
import pytest
from PIL import Image

from src.visual_pilot import pmc


@pytest.fixture()
def fast_limiter(monkeypatch):
    limiter = pmc.RateLimiter(rps=10000.0)
    monkeypatch.setattr(pmc, "RATE_LIMITER", limiter)
    return limiter


@pytest.fixture()
def mock_http(fast_limiter):
    """Inject an httpx.MockTransport client; returns a route registrar."""

    routes: dict[str, httpx.Response] = {}

    def add(substring: str, response: httpx.Response):
        routes[substring] = response

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pat, resp in routes.items():
            if pat in url:
                return resp
        return httpx.Response(404, text="not found")

    pmc.set_http_client(httpx.Client(transport=httpx.MockTransport(handler)))
    pmc.reset_caches()
    yield add
    pmc.set_http_client(None)
    pmc.reset_caches()


@pytest.fixture()
def spy_http(fast_limiter):
    """Like mock_http but also records every requested URL."""

    routes: dict[str, httpx.Response] = {}
    requests: list[str] = []

    def add(substring: str, response: httpx.Response):
        routes[substring] = response

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requests.append(url)
        for pat, resp in routes.items():
            if pat in url:
                return resp
        return httpx.Response(404, text="not found")

    pmc.set_http_client(httpx.Client(transport=httpx.MockTransport(handler)))
    pmc.reset_caches()
    yield add, requests
    pmc.set_http_client(None)
    pmc.reset_caches()


def _json_resp(payload) -> httpx.Response:
    return httpx.Response(200, json=payload)


def _text_resp(text: str) -> httpx.Response:
    return httpx.Response(200, text=text)


LIST_KEYS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>pmc-oa-opendata</Name><Prefix>PMC999.</Prefix>
  <KeyCount>3</KeyCount><MaxKeys>200</MaxKeys><IsTruncated>false</IsTruncated>
  <Contents><Key>PMC999.1/PMC999.1.json</Key></Contents>
  <Contents><Key>PMC999.1/PMC999.1.xml</Key></Contents>
  <Contents><Key>PMC999.1/fig1.jpg</Key></Contents>
</ListBucketResult>"""

META_JSON = {
    "pmcid": "PMC999",
    "version": 1,
    "license_code": "CC BY",
    "is_pmc_openaccess": True,
    "media_urls": ["s3://pmc-oa-opendata/PMC999.1/fig1.jpg?md5=abc"],
    "xml_url": "s3://pmc-oa-opendata/PMC999.1/PMC999.1.xml?md5=x",
}
ARTICLE_XML = (
    '<article xmlns:xlink="http://www.w3.org/1999/xlink">'
    '<front><article-meta><permissions>'
    '<license license-type="open-access" xlink:href="https://creativecommons.org/licenses/by/4.0/"/>'
    "</permissions></article-meta></front>"
    '<body><fig id="f1"><label>Figure 1</label>'
    '<caption><p>A figure.</p></caption>'
    '<graphic xlink:href="fig1.jpg"/></fig></body></article>'
)


def _register_article(add):
    jpeg = _img_bytes("JPEG", size=(32, 24))
    add("list-type=2&prefix=PMC999", _text_resp(LIST_KEYS_XML))
    add("/PMC999.1/PMC999.1.json", _json_resp(META_JSON))
    add("/PMC999.1/PMC999.1.xml", _text_resp(ARTICLE_XML))
    add("/PMC999.1/fig1.jpg", httpx.Response(200, content=jpeg))


# A second article with varied hrefs: resolvable, extensionless, absolute
# URL, and genuinely missing (needs_bytes).
LIST_KEYS_XML_777 = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>pmc-oa-opendata</Name><Prefix>PMC777.</Prefix>
  <KeyCount>5</KeyCount><MaxKeys>200</MaxKeys><IsTruncated>false</IsTruncated>
  <Contents><Key>PMC777.2/PMC777.2.json</Key></Contents>
  <Contents><Key>PMC777.2/PMC777.2.xml</Key></Contents>
  <Contents><Key>PMC777.2/figA.jpg</Key></Contents>
  <Contents><Key>PMC777.2/figB.tiff</Key></Contents>
  <Contents><Key>PMC777.2/unused.png</Key></Contents>
</ListBucketResult>"""

META_JSON_777 = {
    "pmcid": "PMC777",
    "version": 2,
    "license_code": "CC BY",
    "is_pmc_openaccess": True,
    "xml_url": "s3://pmc-oa-opendata/PMC777.2/PMC777.2.xml?md5=x",
}

ARTICLE_XML_777 = (
    '<article xmlns:xlink="http://www.w3.org/1999/xlink">'
    '<front><article-meta><permissions>'
    '<license license-type="open-access" xlink:href="https://creativecommons.org/licenses/by/4.0/"/>'
    "</permissions></article-meta></front>"
    "<body>"
    '<fig id="fa"><graphic xlink:href="figA.jpg"/></fig>'
    '<fig id="fb"><graphic xlink:href="figB"/></fig>'
    '<fig id="fe"><graphic xlink:href="https://cdn.example.com/ext.png"/></fig>'
    '<fig id="fm"><graphic xlink:href="missing.png"/></fig>'
    "</body></article>"
)


def _register_article_777(add):
    add("list-type=2&prefix=PMC777", _text_resp(LIST_KEYS_XML_777))
    add("/PMC777.2/PMC777.2.json", _json_resp(META_JSON_777))
    add("/PMC777.2/PMC777.2.xml", _text_resp(ARTICLE_XML_777))


# ---------------------------------------------------------------------------
# License normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CC BY", "cc-by"),
        ("cc-by", "cc-by"),
        ("https://creativecommons.org/licenses/by/4.0/", "cc-by"),
        ("CC BY 4.0", "cc-by"),
        ("CC BY-SA", "cc-by-sa"),
        ("CC BY-SA 4.0", "cc-by-sa"),
        ("CC BY-ND", "cc-by-nd"),
        ("CC BY-NC", "cc-by-nc"),
        ("CC BY-NC-SA", "cc-by-nc-sa"),
        ("CC BY-NC-ND", "cc-by-nc-nd"),
        ("CC0", "cc0"),
        ("public domain", "cc0"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
        ("some custom publisher license", "other"),
        ("non-commercial use only", "other"),
        (None, "none"),
        ("", "none"),
    ],
)
def test_normalize_license(raw, expected):
    assert pmc.normalize_license(raw) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("cc0", "crop"),
        ("cc-by", "crop"),
        ("cc-by-sa", "crop"),
        ("cc-by-nd", "whole_figure"),
        ("cc-by-nc", None),
        ("cc-by-nc-sa", None),
        ("cc-by-nc-nd", None),
        ("other", None),
        ("none", None),
    ],
)
def test_license_allows(code, expected):
    assert pmc.license_allows(code) == expected


def test_get_license_from_s3_metadata(mock_http):
    _register_article(mock_http)
    lic = pmc.get_license("PMC999")
    assert lic.code == "cc-by"
    assert lic.source == "s3_metadata"
    assert lic.oa_subset == "oa"
    assert "creativecommons.org" in lic.url
    assert pmc.license_allows(lic.code) == "crop"


def test_get_license_nc_excluded(mock_http):
    meta = dict(META_JSON, license_code="CC BY-NC")
    mock_http("list-type=2&prefix=PMC998", _text_resp(LIST_KEYS_XML.replace("PMC999", "PMC998")))
    mock_http("/PMC998.1/PMC998.1.json", _json_resp(meta))
    lic = pmc.get_license("PMC998")
    assert lic.code == "cc-by-nc"
    assert pmc.license_allows(lic.code) is None


# ---------------------------------------------------------------------------
# Bundle + resolver
# ---------------------------------------------------------------------------
def test_resolver_url_vs_needs_bytes(mock_http):
    _register_article(mock_http)
    bundle = pmc.get_article_bundle("PMC999")
    assert bundle.pmcid == "PMC999"
    assert "<fig" in bundle.xml_text
    ref = bundle.resolver("fig1.jpg")
    assert ref.url == "https://pmc-oa-opendata.s3.amazonaws.com/PMC999.1/fig1.jpg"
    assert ref.needs_bytes is False
    assert ref.format == "jpg"
    missing = bundle.resolver("fig9.jpg")
    assert missing.needs_bytes is True and missing.url is None


def test_fetch_image_bytes(mock_http):
    _register_article(mock_http)
    data = pmc.fetch_image_bytes(pmc.ImageRef(url=f"{pmc.S3_BASE}/PMC999.1/fig1.jpg"))
    assert data[:2] == b"\xff\xd8"  # real JPEG magic


# ---------------------------------------------------------------------------
# prepare_for_llm
# ---------------------------------------------------------------------------
def _img_bytes(fmt: str, size=(2000, 1000)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 200, 30)).save(buf, format=fmt)
    return buf.getvalue()


def test_prepare_for_llm_tiff_to_png_and_downscale():
    mime, out = pmc.prepare_for_llm(_img_bytes("TIFF"), max_edge=1568)
    assert mime == "image/png"
    with Image.open(io.BytesIO(out)) as im:
        assert im.format == "PNG"
        assert max(im.size) == 1568
        assert im.size == (1568, 784)


def test_prepare_for_llm_small_jpeg_passthrough():
    data = _img_bytes("JPEG", size=(800, 600))
    mime, out = pmc.prepare_for_llm(data)
    assert mime == "image/jpeg"
    assert out == data


def test_to_data_url_roundtrip():
    url = pmc.to_data_url("image/png", b"\x89PNGdata")
    assert url.startswith("data:image/png;base64,")
    import base64

    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNGdata"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
def test_rate_limiter_spacing():
    limiter = pmc.RateLimiter(rps=20.0)  # 50ms slots
    t0 = time.monotonic()
    for _ in range(3):
        limiter.wait("host.example")
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.09  # at least ~2 intervals between 3 slots


def test_ncbi_rate_depends_on_api_key(monkeypatch):
    monkeypatch.delenv("VP_NCBI_API_KEY", raising=False)
    rates = pmc._default_host_rps()
    assert rates["www.ncbi.nlm.nih.gov"] == 3.0
    monkeypatch.setenv("VP_NCBI_API_KEY", "abc123")
    rates = pmc._default_host_rps()
    assert rates["www.ncbi.nlm.nih.gov"] == 10.0


# ---------------------------------------------------------------------------
# No disk writes (spec §9 testing rule)
# ---------------------------------------------------------------------------
def _tree_snapshot(path):
    if not path.exists():
        return set()
    return {str(p.relative_to(path)) for p in path.rglob("*")}


def test_no_files_written_during_fetch_and_prepare(vp_data_dir, tmp_path, mock_http):
    _register_article(mock_http)
    before = (_tree_snapshot(vp_data_dir), _tree_snapshot(tmp_path))
    bundle = pmc.get_article_bundle("PMC999")
    ref = bundle.resolver("fig1.jpg")
    data = pmc.fetch_image_bytes(ref)
    mime, blob = pmc.prepare_for_llm(data)
    assert mime == "image/jpeg"
    after = (_tree_snapshot(vp_data_dir), _tree_snapshot(tmp_path))
    assert before == after


# ---------------------------------------------------------------------------
# C3: LicenseInfo hints, hinted bundles, S3 limiter, cache sizes
# ---------------------------------------------------------------------------
def test_license_info_carries_s3_hints(mock_http):
    _register_article(mock_http)
    lic = pmc.get_license("PMC999")
    assert lic.prefix == "PMC999.1"
    assert lic.media_files == ("fig1.jpg",)


def test_hinted_bundle_skips_listing_and_metadata(spy_http):
    add, requests = spy_http
    add("/PMC999.1/PMC999.1.xml", _text_resp(ARTICLE_XML))
    bundle = pmc.get_article_bundle(
        "PMC999", prefix="PMC999.1", media_files=("fig1.jpg",)
    )
    ref = bundle.resolver("fig1.jpg")
    assert ref.url == f"{pmc.S3_BASE}/PMC999.1/fig1.jpg"
    assert not any("list-type=2" in url for url in requests)
    assert not any(url.endswith(".json") for url in requests)
    assert sum("PMC999.1/PMC999.1.xml" in url for url in requests) == 1


def test_hinted_bundle_prefix_only_matches_unhinted(spy_http):
    """A prefix-only hint cannot reproduce media_urls-derived file sets, so
    it transparently falls back to the standard lookup — identical refs."""
    add, requests = spy_http
    _register_article(add)
    hinted = pmc.get_article_bundle("PMC999", prefix="PMC999.1", use_cache=False)
    hinted_requests = list(requests)
    requests.clear()
    plain = pmc.get_article_bundle("PMC999", use_cache=False)
    for href in list(pmc._iter_local_hrefs(plain.xml_text)) + ["nope.gif"]:
        assert hinted.resolver(href) == plain.resolver(href), href
    # The hinted prefix's XML was fetched directly (exactly once).
    assert sum("PMC999.1/PMC999.1.xml" in url for url in hinted_requests) == 1


def test_hinted_and_unhinted_resolver_parity(spy_http):
    """Every href in the fixture resolves to identical ImageRefs."""
    add, _ = spy_http
    _register_article_777(add)
    plain = pmc.get_article_bundle("PMC777", use_cache=False)
    hinted = pmc.get_article_bundle(
        "PMC777",
        use_cache=False,
        prefix="PMC777.2",
        media_files=("figA.jpg", "figB.tiff", "unused.png"),
    )
    hrefs = list(pmc._iter_local_hrefs(plain.xml_text))
    assert {"figA.jpg", "figB", "missing.png"} <= set(hrefs)
    hrefs += ["nope.gif", "subdir/figA.jpg"]
    for href in hrefs:
        assert plain.resolver(href) == hinted.resolver(href), href


def test_incomplete_media_hint_falls_back_to_listing(spy_http):
    add, requests = spy_http
    _register_article(add)
    bundle = pmc.get_article_bundle(
        "PMC999", prefix="PMC999.1", media_files=("wrong.png",)
    )
    ref = bundle.resolver("fig1.jpg")
    assert ref.url == f"{pmc.S3_BASE}/PMC999.1/fig1.jpg"
    assert any("list-type=2" in url for url in requests)


def test_stale_prefix_hint_recovers_unhinted(spy_http):
    """A hint pointing at a version dir with no XML falls back cleanly."""
    add, requests = spy_http
    _register_article(add)
    bundle = pmc.get_article_bundle("PMC999", prefix="PMC999.9")
    ref = bundle.resolver("fig1.jpg")
    assert ref.url == f"{pmc.S3_BASE}/PMC999.1/fig1.jpg"


def test_s3_and_ncbi_host_rates(monkeypatch):
    monkeypatch.delenv("VP_NCBI_API_KEY", raising=False)
    rates = pmc._default_host_rps()
    assert rates["pmc-oa-opendata.s3.amazonaws.com"] == pmc.config.VP_S3_RPS == 20.0
    assert rates["www.ncbi.nlm.nih.gov"] == 3.0
    assert rates["eutils.ncbi.nlm.nih.gov"] == 3.0
    monkeypatch.setattr(pmc.config, "VP_S3_RPS", 7.5)
    rates = pmc._default_host_rps()
    assert rates["pmc-oa-opendata.s3.amazonaws.com"] == 7.5
    assert rates["www.ncbi.nlm.nih.gov"] == 3.0


def test_cache_sizes_cover_full_run():
    assert pmc._article_metadata.cache_info().maxsize >= 2000
    assert pmc._article_bundle_cached.cache_info().maxsize >= 256
