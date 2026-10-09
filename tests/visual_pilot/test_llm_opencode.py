"""OpenCode request shape and cost accounting for the LLM client."""

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
    client = llm.LLMClient(provider="opencode")
    completions = _Completions(usage, failures)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_request_asks_for_json_in_prompt_with_reasoning_effort(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    parsed, meta = client.call_json(
        "p2", "step-5-preview-free", "sys", "user", SCHEMA, reasoning_effort="low"
    )
    assert parsed == {"ok": True}
    assert meta["response_format"] == "prompt_json"
    sent = completions.calls[0]
    assert "response_format" not in sent
    assert sent["stream"] is True
    assert completions.streams[0].closed
    assert sent["extra_body"] == {"reasoning_effort": "low"}
    assert "not a schema" in sent["messages"][0]["content"]


def test_empty_effort_sends_no_reasoning_field(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    client, completions = _client(monkeypatch, usage)
    client.call_json("p2", "step-5-preview-free", "sys", "user", SCHEMA, reasoning_effort="")
    assert "extra_body" not in completions.calls[0]


def test_cost_prices_cached_prompt_tokens_at_cache_hit_rate(monkeypatch):
    monkeypatch.setenv(
        "VP_MODEL_PRICES_JSON",
        '{"priced-model": {"in": 0.10, "cached_in": 0.02, "out": 0.30}}',
    )
    usage = SimpleNamespace(
        prompt_tokens=1_000_000, completion_tokens=1_000_000, cached_tokens=400_000
    )
    client, _ = _client(monkeypatch, usage)
    _, meta = client.call_json("p2", "priced-model", "sys", "user", SCHEMA)
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
    parsed, meta = client.call_json("p2", "step-5-preview-free", "sys", "user", SCHEMA)
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
        "p3", "step-5-preview-free", "sys", "user", schema, fill_missing={"figure_id": "PMC1:f1"}
    )
    assert parsed == {"ok": True, "figure_id": "PMC1:f1"}
    assert meta["attempts"] == 1


def test_normalize_keeps_model_values_and_non_objects():
    out = llm._normalize('{"figure_id": "model", "ok": true}', {}, {"figure_id": "req"})
    assert json.loads(out)["figure_id"] == "model"
    assert llm._normalize("[1]", {}, {"figure_id": "req"}) == "[1]"


def test_normalize_drops_forbidden_keys_and_calls_derived_defaults():
    schema = {"properties": {"panels": {}, "compound": {}}, "additionalProperties": False}
    out = llm._normalize(
        '{"panels": [1, 2], "note": "extra"}',
        schema,
        {"compound": lambda reply: len(reply["panels"]) > 1},
    )
    assert json.loads(out) == {"panels": [1, 2], "compound": True}


def test_judge_derives_compound_from_panel_count():
    from src.visual_pilot import judge

    assert judge._lists_several_panels({"panels": [{}, {}]}) is True
    assert judge._lists_several_panels({"panels": [{}]}) is False
    assert judge._lists_several_panels({}) is False


def test_default_model_is_free():
    from src.visual_pilot import config

    assert config.model_price("step-5-preview-free") == (0.0, 0.0, 0.0)


def test_normalize_unwraps_fenced_and_wrapped_json():
    fenced = '```json\n{"ok": true}\n```'
    assert json.loads(llm._normalize(fenced, SCHEMA, {})) == {"ok": True}
    prose = 'Here is the result: {"ok": true} Hope that helps.'
    assert json.loads(llm._normalize(prose, SCHEMA, {})) == {"ok": True}
    assert llm._normalize("not json", SCHEMA, {}) == "not json"


def test_client_sends_opencode_session_and_user_agent(monkeypatch):
    monkeypatch.setattr(
        llm.config, "llm_credentials", lambda provider: ("k", "https://example.invalid/v1")
    )
    headers = llm.LLMClient(provider="opencode")._openai().default_headers
    assert headers["User-Agent"] == llm.USER_AGENT
    assert headers["x-opencode-session"].startswith("visual-pilot-")
