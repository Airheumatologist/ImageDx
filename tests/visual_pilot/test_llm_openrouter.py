"""Xiaomi Token Plan request shape and cost accounting for the LLM client."""

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


def _credentials(monkeypatch):
    monkeypatch.setattr(llm.config, "MIMO_API_KEY", "k")
    monkeypatch.setattr(llm.config, "MIMO_BASE_URL", "https://example.invalid/v1")


def _client(monkeypatch, usage, failures=()):
    _credentials(monkeypatch)
    client = llm.LLMClient()
    completions = _Completions(usage, failures)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_request_asks_for_json_mode_with_reasoning_effort(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    parsed, meta = client.call_json(
        "p2", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA, reasoning_effort="low"
    )
    assert parsed == {"ok": True}
    assert meta["response_format"] == "json_object"
    assert meta["provider"] == "xiaomi"
    sent = completions.calls[0]
    assert sent["model"] == "mimo-v2.6-flash"
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["stream"] is True
    assert completions.streams[0].closed
    assert sent["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "not a schema" in sent["messages"][0]["content"]


def test_empty_effort_sends_no_reasoning_field(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    client, completions = _client(monkeypatch, usage)
    client.call_json("p2", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA, reasoning_effort="")
    assert "extra_body" not in completions.calls[0]


def test_images_go_as_image_url_parts(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    client, completions = _client(monkeypatch, usage)
    image = llm.ImageInput(data_url="data:image/jpeg;base64,AAAA")
    client.call_json("p3", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA, images=[image])
    assert completions.calls[0]["messages"][1]["content"] == [
        {"type": "text", "text": "user"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
    ]


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


def test_cost_uses_the_amount_openrouter_reports(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0, cost=0.0123)
    client, _ = _client(monkeypatch, usage)
    _, meta = client.call_json("p2", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA)
    assert meta["cost_usd"] == 0.0123


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
    parsed, meta = client.call_json("p2", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA)
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
        "p3", "xiaomi/mimo-v2.6-flash", "sys", "user", schema, fill_missing={"figure_id": "PMC1:f1"}
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


def test_normalize_snaps_misspelled_enum_and_drops_null_keyed_items():
    item = {
        "type": "object",
        "properties": {"finding_key": {"type": "string"}, "evidence": {"type": "string"}},
        "required": ["finding_key", "evidence"],
    }
    keys = ["pleomorphic_xanthoastrocytoma", "aps_type_1", "aps_type_2", None]
    schema = {"properties": {"panels": {"type": "array", "items": {"properties": {
        "disease_key": {"type": ["string", "null"], "enum": keys},
        "findings": {"type": "array", "items": item},
    }}}}}
    reply = {"panels": [
        {"disease_key": "pleomorphic_xanthocytoma",
         "findings": [{"finding_key": None, "evidence": "x"}, {"finding_key": "f", "evidence": "y"}]},
        {"disease_key": "aps_type_3", "findings": []},
    ]}
    out = json.loads(llm._normalize(json.dumps(reply), schema, {}))
    assert out["panels"][0] == {
        "disease_key": "pleomorphic_xanthoastrocytoma",
        "findings": [{"finding_key": "f", "evidence": "y"}],
    }
    # A different number is a different disease: left for validation to reject.
    assert out["panels"][1]["disease_key"] == "aps_type_3"


def test_judge_derives_compound_from_panel_count():
    from src.visual_pilot import judge

    assert judge._lists_several_panels({"panels": [{}, {}]}) is True
    assert judge._lists_several_panels({"panels": [{}]}) is False
    assert judge._lists_several_panels({}) is False


def test_default_models_use_opencode_text_and_openrouter_vision():
    from src.visual_pilot import config

    assert config.DEFAULT_MODEL == "opencode-go/iceman-contributor-free"
    assert config.DEFAULT_VISION_MODEL == "z-ai/glm-5.3-flash"
    assert config.OPENROUTER_PROVIDERS == ("open-inference",)
    # Iceman is currently free. Legacy Token Plan prices remain available.
    assert config.model_price(config.DEFAULT_MODEL) == (0.0, 0.0, 0.0)
    assert config.model_price("xiaomi/mimo-v2.6-flash") == (0.0, 0.0, 0.0)
    assert config.MIMO_BASE_URL == "https://token-plan-sgp.xiaomimimo.com/v1"


def test_normalize_unwraps_fenced_and_wrapped_json():
    fenced = '```json\n{"ok": true}\n```'
    assert json.loads(llm._normalize(fenced, SCHEMA, {})) == {"ok": True}
    prose = 'Here is the result: {"ok": true} Hope that helps.'
    assert json.loads(llm._normalize(prose, SCHEMA, {})) == {"ok": True}
    assert llm._normalize("not json", SCHEMA, {}) == "not json"


def test_client_sends_user_agent(monkeypatch):
    _credentials(monkeypatch)
    headers = llm.LLMClient()._openai().default_headers
    assert headers["User-Agent"] == llm.USER_AGENT


def test_token_plan_and_openrouter_ids_share_a_cache_key():
    client = llm.LLMClient()
    old = client._input_hash("p3", "xiaomi/mimo-v2.6-flash", "", "sys", "user", [], "low")
    new = client._input_hash("p3", "mimo-v2.6-flash", "", "sys", "user", [], "low")
    assert old == new


def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.setattr(llm.config, "MIMO_API_KEY", None)
    with pytest.raises(llm.LLMError, match="MIMO_API_KEY"):
        llm.LLMClient()._openai()


def test_requests_carry_the_output_token_cap(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    client.call_json("p2", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA)
    assert completions.calls[0]["max_completion_tokens"] == llm.config.VP_LLM_MAX_OUTPUT_TOKENS


def test_effort_off_disables_thinking(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    client.call_json("p2", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA, reasoning_effort="off")
    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


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
    client.call_json("p2", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA)
    assert len(waits) == len(completions.calls) == 1


def _rate_limited():
    response = llm.httpx.Response(
        429, request=llm.httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    )
    return llm.openai.RateLimitError("rate limited", response=response, body=None)


def test_rate_limit_backs_off_and_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    calls = []
    original = completions.create

    def create(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise _rate_limited()
        return original(**kwargs)

    completions.create = create
    parsed, meta = client.call_json("p2", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA)
    assert parsed == {"ok": True}
    assert calls == ["mimo-v2.6-flash"] * 2
    assert sleeps == [1.0]


def test_spent_credits_stop_like_a_spent_budget(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    client, completions = _client(monkeypatch, usage)
    response = llm.httpx.Response(402, request=llm.httpx.Request("POST", "https://x.invalid"))
    calls = []

    def no_credits(**kwargs):
        calls.append(kwargs["model"])
        raise llm.openai.APIStatusError("Insufficient credits", response=response, body=None)

    completions.create = no_credits
    with pytest.raises(llm.BudgetExceeded):
        client.call_json("p3", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA)
    assert len(calls) == 1 and sleeps == []


def test_mid_stream_error_chunk_is_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cached_tokens=0)
    client, completions = _client(monkeypatch, usage)
    original = completions.create

    def create(**kwargs):
        stream = original(**kwargs)
        if len(completions.calls) == 1:
            stream.chunks.insert(0, SimpleNamespace(
                error={"code": 502, "message": "upstream died"}, choices=[], usage=None))
        return stream

    completions.create = create
    parsed, _ = client.call_json("p3", "xiaomi/mimo-v2.6-flash", "sys", "user", SCHEMA)
    assert parsed == {"ok": True}
    assert len(completions.calls) == 2
    assert sleeps == [1.0]


def test_record_recovers_a_connection_left_on_a_stale_snapshot(tmp_path):
    from src.visual_pilot import db

    path = tmp_path / "vp.sqlite"
    shared = db.init_db(db.connect(path))
    other = db.connect(path)
    client = llm.LLMClient(db_conn=shared)
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
