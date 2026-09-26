"""PMC access for the visual pilot (workstream W2, stages 0 and 3 helpers).

Verified against the live 2025 PMC Article Datasets layout: the old
``oa.fcgi`` web service, ``oa_file_list.csv`` and ``oa_package`` tarballs are
gone (404). ``s3://pmc-oa-opendata`` now serves public HTTPS objects under
per-article version dirs::

    {pmcid}.{version}/{pmcid}.{version}.json   article metadata (license!)
    {pmcid}.{version}/{pmcid}.{version}.xml    JATS XML
    {pmcid}.{version}/{pmcid}.{version}.txt/.pdf
    {pmcid}.{version}/{figure files}           images under real filenames

License source (chosen): the per-article metadata JSON (``license_code``),
with the JATS ``<permissions>`` block as fallback.
Figure access (chosen): direct public HTTPS S3 URLs (spec preference 1).
Fallbacks: Europe PMC fullTextXML for the XML itself; unresolved graphics get
``needs_bytes`` (the Europe PMC ``/bin/`` endpoint returns 403).

Everything is in memory: nothing here writes to disk. httpx calls go through
a per-host token-bucket rate limiter (NCBI <=3 req/s, <=10 with
``VP_NCBI_API_KEY``; the public ``pmc-oa-opendata`` S3 bucket uses
``config.VP_S3_RPS`` (default 20); other hosts ~5 req/s) plus retries on
429/5xx.

Contract C3 (docs/visual_pilot_plan.md §4): ``LicenseInfo`` carries the
article's S3 ``prefix`` and ``media_files`` when they were already fetched;
``get_article_bundle`` accepts those as hints to skip the bucket listing and
metadata JSON while producing a resolver identical to the no-hint path.
"""

from __future__ import annotations

import base64
import io
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import PurePosixPath
from urllib.parse import urlparse

import httpx
from lxml import etree
from PIL import Image

from . import config, timing

S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
EPMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
S3_XMLNS = "http://s3.amazonaws.com/doc/2006-03-01/"

USER_AGENT = "TurboRAG-visual-pilot/0.1 (local research tool)"

# Hosts subject to NCBI's courtesy rate limits.
_NCBI_HOSTS = {"www.ncbi.nlm.nih.gov", "ftp.ncbi.nlm.nih.gov", "eutils.ncbi.nlm.nih.gov"}
_NCBI_RPS = 3.0
_NCBI_RPS_WITH_KEY = 10.0
_DEFAULT_RPS = 5.0

# The public bucket is not an NCBI courtesy host; it gets its own limit
# (contract C1/C3): config.VP_S3_RPS, default 20 req/s.
_S3_HOST = urlparse(S3_BASE).netloc


def _env_cache_size(name: str, default: int) -> int:
    try:
        return max(16, int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


# In-process LRU sizes (contract C3). Memory only, sized for a full run of
# hundreds-to-thousands of articles; tunable via env for larger corpora.
_METADATA_CACHE_SIZE = _env_cache_size("VP_PMC_METADATA_CACHE", 2048)
_BUNDLE_CACHE_SIZE = _env_cache_size("VP_PMC_BUNDLE_CACHE", 512)
_LIST_KEYS_CACHE_SIZE = _env_cache_size("VP_PMC_LIST_CACHE", 2048)

_WEB_FORMATS = {"JPEG", "PNG", "WEBP"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".webp"}
_LICENSE_CODES = (
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-nd",
    "cc-by-nc",
    "cc-by-nc-sa",
    "cc-by-nc-nd",
)


@dataclass(frozen=True)
class LicenseInfo:
    code: str  # cc0|cc-by|cc-by-sa|cc-by-nd|cc-by-nc*|other|none
    url: str | None
    oa_subset: str | None  # "oa" when in the PMC open-access subset
    raw: str | None = None
    source: str | None = None  # which source produced it (s3_metadata|xml)
    # C3 hints for get_article_bundle: the article's S3 dir and the image
    # basenames in it, reused from the metadata/listing get_license already
    # fetched (empty when the license came from the JATS fallback alone).
    prefix: str | None = None
    media_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageRef:
    url: str | None
    needs_bytes: bool = False
    format: str | None = None  # file extension inferred from the resolved name


@dataclass
class ArticleBundle:
    pmcid: str
    xml_text: str
    resolver: Callable[[str], ImageRef]
    metadata: dict = field(default_factory=dict)


class RateLimiter:
    """Per-host token-bucket-ish limiter: at most ``rps`` requests/host."""

    def __init__(self, rps: float = _DEFAULT_RPS, host_rps: dict[str, float] | None = None):
        self.default_rps = rps
        self.host_rps = host_rps or {}
        self._next_allowed: dict[str, float] = {}
        self._lock = threading.Lock()

    def rps_for(self, host: str) -> float:
        return self.host_rps.get(host, self.default_rps)

    def wait(self, host: str) -> None:
        interval = 1.0 / self.rps_for(host)
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_allowed.get(host, 0.0))
            self._next_allowed[host] = slot + interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)
        # W0/C7: observe limiter wait per host; no behavior change.
        timing.record("limiter_wait", max(delay, 0.0), host=host)


def _default_host_rps() -> dict[str, float]:
    # Read env lazily so VP_NCBI_API_KEY can be set after import.
    rps = _NCBI_RPS_WITH_KEY if os.getenv("VP_NCBI_API_KEY") else _NCBI_RPS
    host_rps = {host: rps for host in _NCBI_HOSTS}
    # Public S3 bucket: not an NCBI host; contract C1 VP_S3_RPS (default 20).
    # Read at call time so tests/deploys can adjust config.VP_S3_RPS.
    host_rps[_S3_HOST] = float(getattr(config, "VP_S3_RPS", 20.0))
    return host_rps


RATE_LIMITER = RateLimiter(host_rps=_default_host_rps())

_client: httpx.Client | None = None


def http_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=httpx.Timeout(30.0, read=60.0),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
    return _client


def set_http_client(client: httpx.Client | None) -> None:
    """Test hook: inject a client (e.g. built on httpx.MockTransport)."""
    global _client
    _client = client


def _request(url: str, *, max_retries: int = 4) -> httpx.Response:
    """GET with per-host rate limiting and backoff on 429/5xx."""
    host = urlparse(url).netloc
    started = time.monotonic()
    try:
        return _request_inner(url, host, max_retries=max_retries)
    finally:
        # W0/C7: total fetch wall time per host (includes retries/backoff).
        timing.record("http_fetch", time.monotonic() - started, host=host)


def _request_inner(url: str, host: str, *, max_retries: int = 4) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        RATE_LIMITER.wait(host)
        try:
            resp = http_client().get(url)
        except httpx.TransportError as exc:
            last_exc = exc
            resp = None
            timing.count("http_error", host=host)
        if resp is not None and resp.status_code == 200:
            return resp
        if resp is not None and resp.status_code in (404, 403, 410):
            raise PmcNotFoundError(f"{resp.status_code} for {url}")
        retryable = resp is None or resp.status_code == 429 or resp.status_code >= 500
        if not retryable:
            raise PmcError(f"HTTP {resp.status_code} for {url}")
        if resp is not None and resp.status_code == 429:
            timing.count("http_429", host=host)
        if attempt >= max_retries:
            break
        retry_after = float(resp.headers.get("Retry-After", 0)) if resp is not None else 0.0
        time.sleep(max(retry_after, min(8.0, 2.0**attempt)))
    raise PmcError(f"giving up on {url} after {max_retries + 1} tries: {last_exc}")


class PmcError(Exception):
    pass


class PmcNotFoundError(PmcError):
    pass


# -----------------------------------------------------------------------------
# License
# -----------------------------------------------------------------------------
def normalize_license(raw: str | None) -> str:
    """Map a raw license string/URL to a canonical code."""
    if not raw:
        return "none"
    text = raw.strip().lower()
    if not text or text in {"none", "unknown", "not open access"}:
        return "none"
    squashed = re.sub(r"[^a-z0-9]+", "", text)
    if "publicdomain" in squashed or squashed.startswith("cc0"):
        return "cc0"
    if "byncnd" in squashed:
        return "cc-by-nc-nd"
    if "byncsa" in squashed:
        return "cc-by-nc-sa"
    if "bync" in squashed:
        return "cc-by-nc"
    if "bynd" in squashed or "noderiv" in squashed:
        return "cc-by-nd"
    if "bysa" in squashed or "sharealike" in squashed:
        return "cc-by-sa"
    if re.search(r"\bby\b", text) or "ccby" in squashed or "creativecommon" in squashed:
        return "cc-by"
    return "other"


def license_allows(code: str) -> str | None:
    """§2 license policy: crop | whole_figure | None (excluded)."""
    if code in {"cc0", "cc-by", "cc-by-sa"}:
        return "crop"
    if code == "cc-by-nd":
        return "whole_figure"
    return None


def license_url_for(code: str, raw: str | None = None) -> str | None:
    """Canonical CC URL; prefer a URL found in the raw text when present."""
    if raw:
        match = re.search(r"https?://[^\s<>\"']*creativecommons\.org[^\s<>\"']*", raw)
        if match:
            return match.group(0)
    if code == "cc0":
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    if code.startswith("cc-"):
        return f"https://creativecommons.org/licenses/{code[3:]}/4.0/"
    return None


def _license_hints(meta: dict | None, pmcid: str) -> dict:
    """C3 hints for get_article_bundle from data get_license already fetched.

    ``_figure_files`` reuses meta's ``media_urls`` or the (cached) bucket
    listing, so this adds no requests to the unhinted path.
    """
    if meta:
        prefix = _article_prefix(meta)
        files = _figure_files(pmcid, meta)
    else:
        # No metadata JSON, but the listing was already fetched (cached).
        prefix = _latest_prefix(pmcid)
        files = _figure_files(pmcid, None)
    return {"prefix": prefix, "media_files": tuple(sorted(files))}


def get_license(pmcid: str) -> LicenseInfo:
    """License for one article, from the S3 metadata JSON (fallback: JATS XML)."""
    meta = _article_metadata(pmcid)
    if meta is not None:
        code = normalize_license(meta.get("license_code"))
        if code != "none" or not meta.get("is_pmc_openaccess", True):
            return LicenseInfo(
                code=code,
                url=license_url_for(code, meta.get("license_code")),
                oa_subset="oa" if meta.get("is_pmc_openaccess") else None,
                raw=meta.get("license_code"),
                source="s3_metadata",
                **_license_hints(meta, pmcid),
            )
    # Fallback: license declared in the article XML itself.
    lic = _license_from_xml(pmcid)
    if lic is not None:
        return replace(lic, **_license_hints(meta, pmcid))
    return LicenseInfo(code="none", url=None, oa_subset=None, raw=None, source="none")


def _license_from_xml(pmcid: str) -> LicenseInfo | None:
    try:
        xml_text = _fetch_xml_text(pmcid)
    except PmcError:
        return None
    for href, ltype in _iter_xml_licenses(xml_text):
        raw = " ".join(part for part in (ltype, href) if part)
        code = normalize_license(raw)
        if code != "none":
            return LicenseInfo(
                code=code,
                url=license_url_for(code, raw),
                oa_subset="oa",
                raw=raw,
                source="xml",
            )
    return None


def _iter_xml_licenses(xml_text: str):
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except etree.XMLSyntaxError:
        return
    for lic in root.iter():
        if etree.QName(lic).localname != "license":
            continue
        href = lic.get("{http://www.w3.org/1999/xlink}href") or lic.get("href")
        ltype = lic.get("license-type")
        yield href, ltype


# -----------------------------------------------------------------------------
# Article bundle
# -----------------------------------------------------------------------------
def get_article_bundle(
    pmcid: str,
    *,
    use_cache: bool = True,
    prefix: str | None = None,
    media_files: Iterable[str] | None = None,
) -> ArticleBundle:
    """XML text + figure-href resolver for one article (all in memory).

    Contract C3: when ``prefix`` (and optionally ``media_files``, e.g. from
    ``LicenseInfo`` or ``articles.s3_prefix``/``media_files_json``) are
    supplied, the metadata JSON fetch is skipped and, if the hint validates
    against the article's hrefs, the bucket listing too — only the XML is
    fetched. The resolver output is identical to the no-hint path; a hint
    that cannot reproduce it falls back to the real lookup.
    """
    hint = None if media_files is None else tuple(media_files)
    if use_cache:
        return _article_bundle_cached(pmcid, prefix, hint)
    return _build_bundle(pmcid, prefix=prefix, media_files=hint)


@lru_cache(maxsize=_BUNDLE_CACHE_SIZE)
def _article_bundle_cached(
    pmcid: str,
    prefix: str | None = None,
    media_files: tuple[str, ...] | None = None,
) -> ArticleBundle:
    return _build_bundle(pmcid, prefix=prefix, media_files=media_files)


def _build_bundle(
    pmcid: str,
    prefix: str | None = None,
    media_files: tuple[str, ...] | None = None,
) -> ArticleBundle:
    meta: dict | None = None
    if prefix:
        # Hinted path: trust the caller's S3 prefix; fetch only the XML.
        xml_text = _fetch_s3_xml(prefix)
        files = (
            _files_from_hint(xml_text, media_files)
            if xml_text is not None
            else None
        )
        if files is None:
            # No media_files hint, a stale prefix (no XML there), or a hint
            # that cannot reproduce the no-hint resolver output: recover via
            # the standard lookup so the refs come out identical.
            meta = _article_metadata(pmcid)
            files = _figure_files(pmcid, meta)
            if xml_text is None:
                prefix = None  # stale prefix: re-resolve prefix + XML below
    if not prefix:
        meta = _article_metadata(pmcid)
        xml_text = _fetch_xml_text(pmcid, meta)
        files = _figure_files(pmcid, meta)
        prefix = _article_prefix(meta) or _latest_prefix(pmcid)

    def resolver(href: str) -> ImageRef:
        name = PurePosixPath(href).name
        if href.startswith("http://") or href.startswith("https://"):
            return ImageRef(url=href, needs_bytes=False, format=_ext_of(href))
        candidates = [name]
        stem, _, ext = name.rpartition(".")
        if not ext:
            candidates = [f"{name}{e}" for e in sorted(_IMAGE_EXTS)]
        match = next((f for f in files if f in candidates), None)
        if match and prefix:
            return ImageRef(
                url=f"{S3_BASE}/{prefix}/{match}",
                needs_bytes=False,
                format=_ext_of(match),
            )
        return ImageRef(url=None, needs_bytes=True, format=ext or None)

    return ArticleBundle(
        pmcid=pmcid,
        xml_text=xml_text,
        resolver=resolver,
        metadata=meta or {},
    )


# -----------------------------------------------------------------------------
# S3 dataset internals
# -----------------------------------------------------------------------------
def _article_prefix(meta: dict | None) -> str | None:
    if not meta:
        return None
    pmcid, version = meta.get("pmcid"), meta.get("version")
    if pmcid and version is not None:
        return f"{pmcid}.{version}"
    return None


@lru_cache(maxsize=_METADATA_CACHE_SIZE)
def _article_metadata(pmcid: str) -> dict | None:
    """Fetch {pmcid}.{version}.json for the latest version, if present."""
    key = _latest_metadata_key(pmcid)
    if key is None:
        return None
    try:
        resp = _request(f"{S3_BASE}/{key}")
    except PmcError:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _latest_metadata_key(pmcid: str) -> str | None:
    keys = _list_keys(f"{pmcid}.")
    best: tuple[int, str] | None = None
    pattern = re.compile(rf"^{re.escape(pmcid)}\.(\d+)/{re.escape(pmcid)}\.\d+\.json$")
    for key in keys:
        match = pattern.match(key)
        if match:
            version = int(match.group(1))
            if best is None or version > best[0]:
                best = (version, key)
    return best[1] if best else None


@lru_cache(maxsize=_LIST_KEYS_CACHE_SIZE)
def _list_keys(prefix: str) -> tuple[str, ...]:
    """ListObjectsV2 (public bucket, unsigned).

    Cached per prefix: the listing for an article dir is stable within a run
    and is otherwise fetched up to three times per article (metadata-key
    lookup, figure files, latest prefix).
    """
    keys: list[str] = []
    token: str | None = None
    for _ in range(10):  # pagination cap per article prefix
        url = f"{S3_BASE}/?list-type=2&prefix={prefix}&max-keys=200"
        if token:
            url += f"&continuation-token={token}"
        resp = _request(url)
        root = etree.fromstring(resp.content)
        ns = {"s": S3_XMLNS}
        keys.extend(k.text or "" for k in root.findall("s:Contents/s:Key", ns))
        token_el = root.find("s:NextContinuationToken", ns)
        if token_el is None or not token_el.text:
            return tuple(keys)
        token = token_el.text
    return tuple(keys)


def _figure_files(pmcid: str, meta: dict | None = None) -> frozenset[str]:
    """Basenames of image files in the article dir (media_urls or listing)."""
    files: set[str] = set()
    if meta:
        for media in meta.get("media_urls") or []:
            # entries look like s3://bucket/key?md5=...
            path = media.split("?")[0].split("/", 3)
            if len(path) == 4:
                files.add(PurePosixPath(path[3]).name)
    if not files:
        for key in _list_keys(f"{pmcid}."):
            name = PurePosixPath(key).name
            if _ext_of(name) in _IMAGE_EXTS:
                files.add(name)
    return frozenset(files)


def _files_from_hint(
    xml_text: str, media_files: tuple[str, ...] | None
) -> frozenset[str] | None:
    """The ``media_files`` hint validated against the article's local hrefs.

    Returns the hint as a frozenset when every local href in the XML resolves
    the same way the no-hint path's file set would resolve it; ``None`` when
    no hint was given or the hint is incomplete/mismatched (caller falls back
    to the real metadata/listing rather than emitting different refs).
    """
    if media_files is None:
        return None
    files = frozenset(media_files)
    for href in _iter_local_hrefs(xml_text):
        if not any(candidate in files for candidate in _href_candidates(href)):
            return None
    return files


def _iter_local_hrefs(xml_text: str):
    """Every non-URL href value referenced anywhere in the article XML."""
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except etree.XMLSyntaxError:
        return
    for el in root.iter():
        href = el.get("{http://www.w3.org/1999/xlink}href") or el.get("href")
        if href and not href.startswith(("http://", "https://")):
            yield href


def _href_candidates(href: str) -> list[str]:
    """Filenames a local href could resolve to (mirrors the bundle resolver)."""
    name = PurePosixPath(href).name
    stem, _, ext = name.rpartition(".")
    if ext:
        return [name]
    return [f"{name}{e}" for e in sorted(_IMAGE_EXTS)]


def _fetch_xml_text(pmcid: str, meta: dict | None = None) -> str:
    return _fetch_xml_for_prefix(pmcid, _article_prefix(meta) or _latest_prefix(pmcid))


def _fetch_xml_for_prefix(pmcid: str, prefix: str | None) -> str:
    xml_text = _fetch_s3_xml(prefix) if prefix else None
    if xml_text is not None:
        return xml_text
    # Fallback: Europe PMC full text XML.
    resp = _request(f"{EPMC_REST}/{pmcid}/fullTextXML")
    return resp.text


def _fetch_s3_xml(prefix: str | None) -> str | None:
    """JATS XML text at ``{prefix}/{prefix}.xml``; None when absent/not XML."""
    if not prefix:
        return None
    try:
        resp = _request(f"{S3_BASE}/{prefix}/{prefix}.xml")
        if resp.text.strip().startswith("<"):
            return resp.text
    except PmcError:
        pass
    return None


def _latest_prefix(pmcid: str) -> str | None:
    keys = _list_keys(f"{pmcid}.")
    versions = []
    for key in keys:
        match = re.match(rf"^{re.escape(pmcid)}\.(\d+)/", key)
        if match:
            versions.append(int(match.group(1)))
    if not versions:
        return None
    return f"{pmcid}.{max(versions)}"


def _ext_of(name: str) -> str | None:
    ext = PurePosixPath(urlparse(name).path).suffix.lower()
    return ext.lstrip(".") or None


# -----------------------------------------------------------------------------
# Image fetch + preparation (memory only)
# -----------------------------------------------------------------------------
def fetch_image_bytes(ref: ImageRef) -> bytes:
    """Fetch image bytes into memory. Raises if the ref has no URL."""
    if not ref.url:
        raise PmcError("ImageRef has no URL (needs_bytes=True)")
    return _request(ref.url).content


def prepare_for_llm(data: bytes, max_edge: int | None = None) -> tuple[str, bytes]:
    """Normalize image bytes for the LLM: TIFF/other -> PNG, downscale.

    Returns (mime, bytes). JPEG/PNG/WEBP within max_edge pass through
    unchanged; anything else is re-encoded as PNG; oversized images are
    resized so the long edge is <= max_edge.
    """
    max_edge = max_edge or config.VP_IMAGE_MAX_EDGE
    with Image.open(io.BytesIO(data)) as im:
        fmt = (im.format or "").upper()
        if fmt in _WEB_FORMATS and max(im.size) <= max_edge:
            mime = "image/jpeg" if fmt == "JPEG" else f"image/{fmt.lower()}"
            return mime, data
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if max(im.size) > max_edge:
            scale = max_edge / max(im.size)
            im = im.resize(
                (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                Image.Resampling.LANCZOS,
            )
        out = io.BytesIO()
        if fmt == "JPEG":
            im.save(out, format="JPEG", quality=90)
            return "image/jpeg", out.getvalue()
        im.save(out, format="PNG")
        return "image/png", out.getvalue()


def to_data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def reset_caches() -> None:
    """Drop the in-process caches (tests)."""
    _article_metadata.cache_clear()
    _article_bundle_cached.cache_clear()
    _list_keys.cache_clear()
