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
    monkeypatch.setattr(llm.config, "llm_credentials", lambda provider: ("k", "https://example.invalid/v1"))
    client = llm.LLMClient(provider="opencode")._openai()
    transport = client._client._transport
    assert isinstance(transport, llm._DeadlineTransport)
    assert transport.max_seconds == llm.config.VP_LLM_MAX_REQUEST_SECONDS
