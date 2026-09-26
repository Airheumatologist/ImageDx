"""W0 tests for src/visual_pilot/timing.py and its hooks (contract C7)."""

import json
import threading
import time

import pytest

from src.visual_pilot import llm, pmc, timing

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
VALID = json.dumps({"ok": True})
MODEL = "test-model"


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt=100, completion=50):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _FakeClient:
    def __init__(self, items):
        self.completions = _FakeCompletions(items)
        self.chat = type("Chat", (), {"completions": self.completions})


def make_client(items, conn=None, **kwargs):
    client = llm.LLMClient(provider="opencode", db_conn=conn, **kwargs)
    client._client = _FakeClient(items)
    return client


def _call(client, **over):
    kw = {
        "stage": "triage",
        "model": MODEL,
        "system": "sys",
        "user_content": "user",
        "schema": SCHEMA,
        "prompt_version": "v1",
    }
    kw.update(over)
    return client.call_json(**kw)


@pytest.fixture(autouse=True)
def _reset_timing():
    timing.reset()
    yield
    timing.reset()


# ---------------------------------------------------------------------------
# timing module
# ---------------------------------------------------------------------------
def test_stage_and_record_and_report(tmp_path):
    with timing.stage("parse"):
        time.sleep(0.001)
    with timing.stage("parse"):
        pass
    timing.record("limiter_wait", 0.25, host="s3.example")
    timing.record("limiter_wait", 0.75, host="s3.example")
    timing.record("llm_latency", 1.0, stage="p3", source="live")
    timing.count("llm_429")
    timing.count("llm_429")

    path = timing.write_report(tmp_path / "reports" / "timings_x.json")
    report = json.loads(path.read_text())

    assert report["stages"]["parse"]["calls"] == 2
    assert report["stages"]["parse"]["wall_seconds"] > 0
    wait = report["limiter_wait_seconds_by_host"]["s3.example"]
    assert wait["count"] == 2 and wait["total_seconds"] == pytest.approx(1.0)
    llm_p3 = report["llm_latency_seconds_by_stage"]["p3"]["live"]
    assert llm_p3["count"] == 1 and llm_p3["p50_seconds"] == 1.0
    assert report["counts"]["llm_429"] == 2


def test_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("VP_TIMINGS", "0")
    with timing.stage("parse"):
        pass
    timing.record("limiter_wait", 1.0, host="x")
    timing.count("http_429", host="x")
    with timing.inflight("llm"):
        pass
    path = timing.write_report(tmp_path / "t.json")
    report = json.loads(path.read_text())
    assert report["enabled"] is False
    assert report["stages"] == {}
    assert report["counts"] == {}
    assert report["peak_in_flight"] == {}


def test_inflight_peak_tracks_concurrency():
    def worker():
        with timing.inflight("llm"):
            time.sleep(0.05)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert timing.report()["peak_in_flight"]["llm"] == 4


def test_thread_safe_record():
    def worker():
        for _ in range(50):
            timing.record("http_fetch", 0.001, host="h")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = timing.report()["http_fetch_seconds_by_host"]["h"]
    assert stats["count"] == 400


def test_percentiles():
    vals = [float(i) for i in range(1, 101)]
    assert timing._percentile(vals, 50) == pytest.approx(50.5)
    assert timing._percentile(vals, 95) == pytest.approx(95.05)
    assert timing._percentile([], 50) is None


# ---------------------------------------------------------------------------
# llm.py hooks
# ---------------------------------------------------------------------------
def test_llm_cache_only_raises_on_miss(conn, monkeypatch):
    from src.visual_pilot import config

    monkeypatch.setattr(config, "VP_LLM_CACHE_ONLY", 1)
    client = make_client([VALID], conn)
    with pytest.raises(llm.LLMError, match="cache miss: triage"):
        _call(client)
    # No provider call was made.
    assert len(client._client.completions.calls) == 0


def test_llm_cache_only_passes_on_hit(conn, monkeypatch):
    from src.visual_pilot import config

    # Populate the cache with cache-only OFF, then flip it on.
    client = make_client([VALID], conn)
    parsed, meta = _call(client)
    assert meta["cached"] is False

    monkeypatch.setattr(config, "VP_LLM_CACHE_ONLY", 1)
    parsed2, meta2 = _call(client)
    assert parsed2 == {"ok": True} and meta2["cached"] is True


def test_llm_latency_recorded_cached_and_live(conn):
    client = make_client([VALID], conn)
    _call(client)  # live
    _call(client)  # cache hit
    report = timing.report()
    sources = report["llm_latency_seconds_by_stage"]["triage"]
    assert sources["live"]["count"] == 1
    assert sources["cache"]["count"] == 1
    assert report["peak_in_flight"]["llm"] == 1


def test_llm_429_counted(conn, monkeypatch):
    import httpx
    import openai

    # Post-C4, 429s consume the independent VP_RATE_LIMIT_RETRIES budget;
    # zero it so the single queued error propagates after one attempt.
    monkeypatch.setattr("src.visual_pilot.config.VP_RATE_LIMIT_RETRIES", 0)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    err = openai.RateLimitError(
        "rl", response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
        body=None,
    )
    client = make_client([err], conn, max_retries=0)
    with pytest.raises(openai.RateLimitError):
        _call(client)
    assert timing.report()["counts"]["llm_429"] == 1


# ---------------------------------------------------------------------------
# pmc.py hooks
# ---------------------------------------------------------------------------
def test_pmc_request_records_fetch_and_wait():
    class _Resp:
        status_code = 200
        headers = {}
        content = b"ok"

        def json(self):
            return {}

    class _Client:
        def get(self, url):
            return _Resp()

    pmc.set_http_client(_Client())
    try:
        resp = pmc._request("https://example.org/x")
        assert resp.status_code == 200
    finally:
        pmc.set_http_client(None)
    report = timing.report()
    assert report["http_fetch_seconds_by_host"]["example.org"]["count"] == 1
    assert report["limiter_wait_seconds_by_host"]["example.org"]["count"] == 1


def test_pmc_request_counts_429(monkeypatch):
    class _Resp:
        status_code = 429
        headers = {}
        content = b""

    class _Client:
        def get(self, url):
            return _Resp()

    monkeypatch.setattr(time, "sleep", lambda s: None)
    pmc.set_http_client(_Client())
    try:
        with pytest.raises(pmc.PmcError):
            pmc._request("https://example.org/x", max_retries=1)
    finally:
        pmc.set_http_client(None)
    assert timing.report()["counts"]["http_429|host=example.org"] == 2
