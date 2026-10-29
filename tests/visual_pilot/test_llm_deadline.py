import time

from src.visual_pilot.llm import _openai_httpx as httpx
import pytest

from src.visual_pilot import llm


class _Trickle(httpx.SyncByteStream):
    """Keep-alive body: a byte every 50 ms, never finishing on its own."""

    def __init__(self, chunks: int) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for _ in range(self.chunks):
            time.sleep(0.05)
            yield b" "

    def close(self) -> None:
        self.closed = True


def test_deadline_stream_raises_read_timeout_after_deadline():
    request = httpx.Request("POST", "https://example.invalid")
    inner = _Trickle(chunks=1000)
    stream = llm._DeadlineStream(inner, request, time.monotonic() + 0.2)
    start = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        for _ in stream:
            pass
    assert time.monotonic() - start < 1.0
    stream.close()
    assert inner.closed


def test_deadline_stream_passes_body_through_before_deadline():
    request = httpx.Request("POST", "https://example.invalid")
    stream = llm._DeadlineStream(_Trickle(chunks=3), request, time.monotonic() + 10)
    assert b"".join(stream) == b"   "


def test_client_uses_deadline_transport(monkeypatch):
    monkeypatch.setattr(llm.config, "MIMO_API_KEY", "k")
    monkeypatch.setattr(llm.config, "MIMO_BASE_URL", "https://example.invalid/v1")
    client = llm.LLMClient()._openai()
    transport = client._client._transport
    assert isinstance(transport, llm._DeadlineTransport)
    assert transport.max_seconds == llm.config.VP_LLM_MAX_REQUEST_SECONDS


def test_env_proxy_honors_https_proxy_and_no_proxy(monkeypatch):
    for name in ("HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    assert llm._env_proxy("https://openrouter.ai/api/v1") is None
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("NO_PROXY", "localhost,internal.invalid")
    assert llm._env_proxy("https://openrouter.ai/api/v1") == "http://proxy.invalid:3128"
    assert llm._env_proxy("https://internal.invalid/v1") is None
