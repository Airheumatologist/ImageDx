"""StepFun request shape and cost accounting for the LLM client."""

import json
from types import SimpleNamespace

from src.visual_pilot import llm
from src.visual_pilot.llm import _openai_httpx

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


class _Stream:
    """Minimal stand-in for openai's chat-completion Stream."""

    def __init__(self, chunks, fail_with=None):
        self.chunks = chunks
        self.fail_with = fail_with
        self.response = SimpleNamespace(request=None)
        self.closed = False

    def __iter__(self):
        yield from self.chunks
        if self.fail_with is not None:
            raise self.fail_with

    def close(self):
        self.closed = True


def _chunks(text, usage):
    half = len(text) // 2
    return [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=part))], usage=None
        )
        for part in (text[:half], text[half:])
    ] + [SimpleNamespace(choices=[], usage=usage)]


class _Completions:
    def __init__(self, usage, failures=()):
        self.calls = []
        self.usage = usage
        self.failures = list(failures)
        self.streams = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        fail_with = self.failures.pop(0) if self.failures else None
        stream = _Stream(_chunks('{"ok": true}', self.usage), fail_with)
        self.streams.append(stream)
        return stream


def _client(monkeypatch, usage, failures=()):
    monkeypatch.setattr(
        llm.config, "llm_credentials", lambda provider: ("k", "https://example.invalid/v1")
    )
    client = llm.LLMClient(provider="stepfun")
    completions = _Completions(usage, failures)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_request_uses_json_mode_and_top_level_reasoning_effort(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    parsed, meta = client.call_json(
        "p2", "step-3.5-flash", "sys", "user", SCHEMA, reasoning_effort="low"
    )
    assert parsed == {"ok": True}
    assert meta["response_format"] == "json_object"
    sent = completions.calls[0]
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["stream"] is True
    assert completions.streams[0].closed
    assert sent["extra_body"] == {"reasoning_effort": "low"}
    assert "not a schema" in sent["messages"][0]["content"]


def test_empty_effort_sends_no_reasoning_field(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    client, completions = _client(monkeypatch, usage)
    client.call_json("p2", "step-3.5-flash", "sys", "user", SCHEMA, reasoning_effort="")
    assert "extra_body" not in completions.calls[0]


def test_cost_prices_cached_prompt_tokens_at_cache_hit_rate(monkeypatch):
    monkeypatch.delenv("VP_MODEL_PRICES_JSON", raising=False)
    usage = SimpleNamespace(
        prompt_tokens=1_000_000, completion_tokens=1_000_000, cached_tokens=400_000
    )
    client, _ = _client(monkeypatch, usage)
    _, meta = client.call_json("p2", "step-3.5-flash", "sys", "user", SCHEMA)
    # 600k uncached * $0.10 + 400k cached * $0.02 + 1M out * $0.30
    assert abs(meta["cost_usd"] - (0.06 + 0.008 + 0.30)) < 1e-9


def test_cached_tokens_reads_openai_style_details():
    usage = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=7))
    assert llm._cached_tokens(usage) == 7
    assert llm._cached_tokens(SimpleNamespace()) == 0


def test_mid_stream_timeout_is_retried(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    client, completions = _client(
        monkeypatch, usage, failures=[_openai_httpx.ReadTimeout("stalled")]
    )
    parsed, meta = client.call_json("p2", "step-3.5-flash", "sys", "user", SCHEMA)
    assert parsed == {"ok": True}
    assert len(completions.calls) == 2


def test_fill_missing_supplies_echoed_keys_before_validation(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"figure_id": {"type": "string"}, "ok": {"type": "boolean"}},
        "required": ["figure_id", "ok"],
    }
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    client, completions = _client(monkeypatch, usage)
    parsed, meta = client.call_json(
        "p3", "step-3.7-flash", "sys", "user", schema, fill_missing={"figure_id": "PMC1:f1"}
    )
    assert parsed == {"ok": True, "figure_id": "PMC1:f1"}
    assert meta["attempts"] == 1


def test_fill_missing_keeps_model_value():
    out = llm._fill_missing('{"figure_id": "model", "ok": true}', {"figure_id": "req"})
    assert json.loads(out)["figure_id"] == "model"
    assert llm._fill_missing("[1]", {"figure_id": "req"}) == "[1]"
