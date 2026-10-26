"""OpenCode request shape and cost accounting for the LLM client."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

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


def test_default_model_uses_the_responses_api():
    from src.visual_pilot import config

    assert config.DEFAULT_MODEL == "muse-spark-1.3-contributor"
    assert config.uses_responses_api(config.DEFAULT_MODEL)
    assert config.model_price(config.DEFAULT_MODEL) == (0.10, 0.002, 0.20)
    assert config.uses_json_mode("deepseek-v4.1-flash")
    assert not config.uses_responses_api("deepseek-v4.1-flash")
    assert not config.uses_responses_api("step-5-preview-free")
    assert not config.uses_json_mode("step-5-preview-free")
    assert config.model_price("step-5-preview-free") == (0.0, 0.0, 0.0)


class _Responses:
    def __init__(self):
        self.calls = []
        self.stream = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=4,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
        )
        events = [
            SimpleNamespace(type="response.output_text.delta", delta='{"ok"'),
            SimpleNamespace(type="response.output_text.delta", delta=": true}"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=usage)),
        ]
        self.stream = _Stream(events)
        return self.stream


def test_responses_models_send_json_mode_with_images(monkeypatch):
    monkeypatch.setattr(
        llm.config, "llm_credentials", lambda provider: ("k", "https://example.invalid/v1")
    )
    client = llm.LLMClient(provider="opencode")
    responses = _Responses()
    client._client = SimpleNamespace(responses=responses)
    image = llm.ImageInput(data_url="data:image/jpeg;base64,AAAA")
    parsed, meta = client.call_json(
        "p3", "muse-spark-1.3-contributor", "sys", "user", SCHEMA,
        images=[image], reasoning_effort="low",
    )
    assert parsed == {"ok": True}
    assert meta["response_format"] == "responses_json"
    assert (meta["input_tokens"], meta["output_tokens"]) == (10, 4)
    sent = responses.calls[0]
    assert sent["text"] == {"format": {"type": "json_object"}}
    assert sent["reasoning"] == {"effort": "low"}
    assert sent["instructions"].startswith("sys")
    assert sent["input"][0]["content"] == [
        {"type": "input_text", "text": "user"},
        {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"},
    ]
    assert responses.stream.closed


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


def test_requests_carry_the_output_token_cap(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    client.call_json("p2", "step-5-preview-free", "sys", "user", SCHEMA)
    assert completions.calls[0]["max_tokens"] == llm.config.VP_LLM_MAX_OUTPUT_TOKENS


def test_effort_off_disables_thinking(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    client.call_json("p2", "glm-5.3-flash", "sys", "user", SCHEMA, reasoning_effort="off")
    client.call_json("p2", "glm-5.3-flash", "sys", "user2", SCHEMA, reasoning_effort="low")
    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert completions.calls[1]["extra_body"] == {"reasoning_effort": "low"}


def test_pacer_spaces_requests_to_the_rpm_cap(monkeypatch):
    sleeps = []
    clock = [100.0]
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    pacer = llm._RequestPacer(rpm=10)
    for _ in range(3):
        pacer.wait()
    assert sleeps == [6.0, 12.0]
    llm._RequestPacer(rpm=0).wait()  # unpaced: returns at once
    assert sleeps == [6.0, 12.0]


def test_every_attempt_waits_for_the_pacer(monkeypatch):
    waits = []
    monkeypatch.setattr(llm, "_PACER", SimpleNamespace(wait=lambda: waits.append(1)))
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    client.call_json("p2", "step-5-preview-free", "sys", "user", SCHEMA)
    assert len(waits) == len(completions.calls) == 1


def test_json_mode_models_send_response_format(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    _, meta = client.call_json("p2", "deepseek-v4.1-flash", "sys", "user", SCHEMA)
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert meta["response_format"] == "chat_json"
    _, meta = client.call_json("p2", "step-5-preview-free", "sys", "user", SCHEMA)
    assert "response_format" not in completions.calls[1]
    assert meta["response_format"] == "prompt_json"


def _rate_limited():
    response = llm.httpx.Response(
        429, request=llm.httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    )
    return llm.openai.RateLimitError("rate limited", response=response, body=None)


class _RateLimitedCompletions(_Completions):
    """Answers 429 for every request to ``limited_model``."""

    def __init__(self, usage, limited_model):
        super().__init__(usage)
        self.limited_model = limited_model

    def create(self, **kwargs):
        if kwargs["model"] == self.limited_model:
            self.calls.append(kwargs)
            raise _rate_limited()
        return super().create(**kwargs)


def test_free_tier_exhausted_backs_off_like_a_rate_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "")
    monkeypatch.setattr(llm.config, "VP_RATE_LIMIT_RETRIES", 5)
    response = llm.httpx.Response(
        502, request=llm.httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    )
    exhausted = llm.openai.InternalServerError(
        "Error code: 502 - {'error': {'code': 'free_tier_exhausted'}}",
        response=response, body=None,
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 4:  # more than max_retries=3 transient retries allow
            raise exhausted
        return _Completions.create(completions, **kwargs)

    monkeypatch.setattr(completions, "create", create)
    parsed, _ = client.call_json("p3", "minimax/minimax-m3.1-flash:free", "sys", "user", SCHEMA)
    assert parsed == {"ok": True}
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


def test_reply_from_a_substituted_model_is_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    served = ["anyrouter/free-pick", "nvidia/nemotron-3.5-super-vl-120b-a12b"]

    def create(**kwargs):
        stream = _Completions.create(completions, **kwargs)
        model = served.pop(0)
        for chunk in stream.chunks:
            chunk.model = model
        return stream

    monkeypatch.setattr(completions, "create", create)
    parsed, _ = client.call_json(
        "p3", "nvidia/nemotron-3.5-super-vl-120b-a12b:free", "sys", "user", SCHEMA
    )
    assert parsed == {"ok": True}
    assert len(completions.calls) == 2
    assert sleeps == [1.0]


def test_substituted_reply_switches_to_the_fallback_model(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "minimax/minimax-m3.1-flash:free")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)

    def create(**kwargs):
        stream = _Completions.create(completions, **kwargs)
        served = ("reflection/beam-1.0" if kwargs["model"].startswith("nvidia/")
                  else "minimax/minimax-m3.1-flash")
        for chunk in stream.chunks:
            chunk.model = served
        return stream

    monkeypatch.setattr(completions, "create", create)
    parsed, meta = client.call_json(
        "p3", "nvidia/nemotron-3.5-super-vl-120b-a12b:free", "sys", "user", SCHEMA
    )
    assert parsed == {"ok": True}
    assert [c["model"] for c in completions.calls] == [
        "nvidia/nemotron-3.5-super-vl-120b-a12b:free", "minimax/minimax-m3.1-flash:free",
    ]
    assert sleeps == []


def test_rate_limit_switches_to_the_fallback_model(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "deepseek-v4.1-flash")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, _ = _client(monkeypatch, usage)
    completions = _RateLimitedCompletions(usage, "glm-5.3-flash")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    recorded = []
    monkeypatch.setattr(client, "_record", lambda *a: recorded.append(a))
    parsed, meta = client.call_json(
        "p3", "glm-5.3-flash", "sys", "user", SCHEMA, reasoning_effort="off"
    )
    assert parsed == {"ok": True}
    assert [c["model"] for c in completions.calls] == [
        "glm-5.3-flash", "deepseek-v4.1-flash"]
    assert completions.calls[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sleeps == []  # the switch is immediate
    assert meta["model"] == "deepseek-v4.1-flash"
    assert recorded[0][2] == "deepseek-v4.1-flash"


def test_rate_limit_without_fallback_retries_the_same_model(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage, failures=[])
    calls = []
    original = completions.create

    def create(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise _rate_limited()
        return original(**kwargs)

    completions.create = create
    _, meta = client.call_json("p2", "glm-5.3-flash", "sys", "user", SCHEMA)
    assert calls == ["glm-5.3-flash", "glm-5.3-flash"]
    assert meta["model"] == "glm-5.3-flash"


def test_unavailable_endpoint_switches_to_the_fallback_model(monkeypatch):
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "deepseek-v4.1-flash")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    calls = []
    original = completions.create

    def create(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "glm-5.3-flash":
            response = llm.httpx.Response(402, request=llm.httpx.Request("POST", "https://x.invalid"))
            raise llm.openai.APIStatusError(
                "Upstream request failed: Endpoint is unavailable.", response=response, body=None)
        return original(**kwargs)

    completions.create = create
    _, meta = client.call_json("p2", "glm-5.3-flash", "sys", "user", SCHEMA)
    assert calls == ["glm-5.3-flash", "deepseek-v4.1-flash"]
    assert meta["model"] == "deepseek-v4.1-flash"


def test_fallback_key_sends_the_fallback_model_through_its_own_account(monkeypatch):
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "deepseek-v4.1-flash")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "fallback-key")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, primary = _client(monkeypatch, usage)
    primary.create = lambda **kwargs: (_ for _ in ()).throw(_rate_limited())
    keys = []
    fallback = _Completions(usage)

    def new_openai(api_key):
        keys.append(api_key)
        return SimpleNamespace(chat=SimpleNamespace(completions=fallback))

    monkeypatch.setattr(client, "_new_openai", new_openai)
    _, meta = client.call_json("p2", "glm-5.3-flash", "sys", "user", SCHEMA)
    client.call_json("p2", "glm-5.3-flash", "sys", "user2", SCHEMA)
    assert keys == ["fallback-key"]  # built once, reused
    assert [c["model"] for c in fallback.calls] == ["deepseek-v4.1-flash"] * 2
    assert meta["model"] == "deepseek-v4.1-flash"


def test_fallback_account_retries_the_same_model(monkeypatch):
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "fallback-key")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, primary = _client(monkeypatch, usage)
    primary_models = []

    def limited(**kwargs):
        primary_models.append(kwargs["model"])
        raise _rate_limited()

    primary.create = limited
    fallback = _Completions(usage)
    monkeypatch.setattr(
        client, "_new_openai",
        lambda api_key: SimpleNamespace(chat=SimpleNamespace(completions=fallback)),
    )
    _, meta = client.call_json("p2", "deepseek-v4.1-flash", "sys", "user", SCHEMA)
    assert primary_models == ["deepseek-v4.1-flash"]
    assert [c["model"] for c in fallback.calls] == ["deepseek-v4.1-flash"]
    assert meta["model"] == "deepseek-v4.1-flash"


def test_responses_usage_limit_moves_to_the_fallback_account(monkeypatch):
    monkeypatch.setattr(
        llm.config, "llm_credentials", lambda provider: ("k", "https://example.invalid/v1")
    )
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "fallback-key")
    client = llm.LLMClient(provider="opencode")
    response = llm.httpx.Response(429, request=llm.httpx.Request("POST", "https://x.invalid"))
    spent = llm.openai.RateLimitError(
        "Error code: 429 - {'error': {'type': 'GoUsageLimitError'}}", response=response, body=None)
    primary_calls = []

    def limited(**kwargs):
        primary_calls.append(kwargs["model"])
        raise spent

    client._client = SimpleNamespace(responses=SimpleNamespace(create=limited))
    fallback = _Responses()
    keys = []

    def new_openai(api_key):
        keys.append(api_key)
        return SimpleNamespace(responses=fallback)

    monkeypatch.setattr(client, "_new_openai", new_openai)
    parsed, meta = client.call_json("p3", "muse-spark-1.3-contributor", "sys", "user", SCHEMA)
    assert parsed == {"ok": True}
    assert primary_calls == ["muse-spark-1.3-contributor"]
    assert keys == ["fallback-key"]
    assert [c["model"] for c in fallback.calls] == ["muse-spark-1.3-contributor"]
    assert meta["model"] == "muse-spark-1.3-contributor"
    assert meta["response_format"] == "responses_json"


def test_usage_limit_on_every_account_stops_like_a_spent_budget(monkeypatch):
    import pytest

    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm.config, "VP_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm.config, "VP_FALLBACK_API_KEY", "fallback-key")
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    client, primary = _client(monkeypatch, usage)
    response = llm.httpx.Response(429, request=llm.httpx.Request("POST", "https://x.invalid"))
    limit = llm.openai.RateLimitError(
        "Error code: 429 - {'error': {'type': 'GoUsageLimitError'}}", response=response, body=None)
    calls = []

    def limited(**kwargs):
        calls.append(kwargs["model"])
        raise limit

    primary.create = limited
    monkeypatch.setattr(client, "_new_openai", lambda api_key: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=limited))))
    with pytest.raises(llm.BudgetExceeded):
        client.call_json("p3", "deepseek-v4.1-flash", "sys", "user", SCHEMA)
    assert len(calls) == 2 and sleeps == []  # primary, then fallback; no backoff


def test_record_recovers_a_connection_left_on_a_stale_snapshot(tmp_path):
    from src.visual_pilot import db

    path = tmp_path / "vp.sqlite"
    shared = db.init_db(db.connect(path))
    other = db.connect(path)
    client = llm.LLMClient(db_conn=shared, provider="opencode")
    meta = {"attempts": 1, "response_format": "json_object"}

    shared.executemany(
        "INSERT INTO llm_calls (stage, model, input_hash) VALUES ('p2', 'm', ?)",
        [(f"seed{i}",) for i in range(3)],
    )
    shared.commit()
    # A read on the shared connection outlives another connection's commit,
    # so a write on the shared connection fails at once.
    rows = shared.execute("SELECT input_hash FROM llm_calls")
    rows.fetchone()
    other.execute("INSERT INTO llm_calls (stage, model, input_hash) VALUES ('p2', 'm', 'other')")
    other.commit()
    with pytest.raises(sqlite3.OperationalError):
        client._record("h1", "p3", "m", "v1", '{"ok": true}', {}, None, meta)
    rows.fetchall()

    # Once the read ends, the connection writes again instead of failing
    # every later write ("database is locked" in judge._fail).
    client._record("h2", "p3", "m", "v1", '{"ok": true}', {}, None, meta)
    stored = other.execute("SELECT 1 FROM llm_calls WHERE input_hash = 'h2'").fetchone()
    assert stored is not None
