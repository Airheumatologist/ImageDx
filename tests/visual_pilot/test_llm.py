"""Tests for src/visual_pilot/llm.py — mocked OpenAI client, no network."""

import json
import threading
import time

import httpx
import jsonschema
import openai
import pytest

from src.visual_pilot import config, db, llm
from src.visual_pilot.prompts import P1, P2, P3, P4, PROMPTS

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
    def __init__(self, content, prompt=100, completion=50):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt, completion)


class _FakeCompletions:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, _FakeResponse) else _FakeResponse(item)


class _FakeClient:
    """Duck-typed stand-in for openai.OpenAI (chat.completions.create only)."""

    def __init__(self, items):
        self.completions = _FakeCompletions(items)
        self.chat = type("Chat", (), {"completions": self.completions})


def make_client(items, conn=None, **kwargs):
    client = llm.LLMClient(provider="opencode", db_conn=conn, **kwargs)
    client._client = _FakeClient(items)
    return client


def _bad_request(msg="response_format 'json_schema' is not supported"):
    return openai.BadRequestError(
        msg, response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None
    )


def _rate_limit(retry_after: str | None = None):
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    return openai.RateLimitError(
        "rate limited",
        response=httpx.Response(
            429, headers=headers, request=httpx.Request("POST", "http://x")
        ),
        body=None,
    )


def _timeout():
    return openai.APITimeoutError(request=httpx.Request("POST", "http://x"))


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


# ---------------------------------------------------------------------------
# Cache / ledger
# ---------------------------------------------------------------------------
def test_cache_hit_makes_zero_calls(conn):
    client = make_client([VALID], conn)
    parsed, meta = _call(client)
    assert parsed == {"ok": True}
    assert meta["cached"] is False
    parsed2, meta2 = _call(client)
    assert parsed2 == {"ok": True}
    assert meta2["cached"] is True and meta2["cost_usd"] == 0.0
    assert len(client._client.completions.calls) == 1


def test_ledger_row_written(conn, monkeypatch):
    monkeypatch.setenv(
        "VP_MODEL_PRICES_JSON", '{"test-model": {"in": 0.09, "out": 0.55}}'
    )
    client = make_client([VALID], conn)
    _, meta = _call(client)
    row = conn.execute(
        "SELECT stage, model, input_tokens, output_tokens, cost_usd FROM llm_calls"
    ).fetchone()
    assert row["stage"] == "triage"
    assert row["model"] == MODEL
    assert row["input_tokens"] == 100 and row["output_tokens"] == 50
    # test-model: $0.09/$0.55 per 1M -> 100*0.09e-6 + 50*0.55e-6
    assert row["cost_usd"] == pytest.approx(100 * 0.09e-6 + 50 * 0.55e-6)


def test_schema_repair_retry(conn):
    bad = json.dumps({"bad": 1})
    client = make_client([bad, VALID], conn)
    parsed, meta = _call(client)
    assert parsed == {"ok": True}
    assert meta["attempts"] == 2
    calls = client._client.completions.calls
    assert len(calls) == 2
    assert "failed validation" in calls[1]["messages"][1]["content"]


def test_second_validation_failure_raises(conn):
    client = make_client(["{not json", json.dumps({"bad": 1})], conn)
    with pytest.raises(llm.SchemaValidationError):
        _call(client)


def test_budget_guard(conn, monkeypatch):
    monkeypatch.setenv(
        "VP_MODEL_PRICES_JSON", '{"test-model": {"in": 0.09, "out": 0.55}}'
    )
    # One call costs ~3.65e-5 USD; a budget below that allows exactly one.
    client = make_client([VALID, VALID], conn, budget_usd=1e-5)
    _call(client)
    with pytest.raises(llm.BudgetExceeded):
        _call(client, user_content="different input")  # avoid the cache hit


def test_dry_run_makes_no_call(conn):
    client = make_client([VALID], conn, dry_run=True)
    parsed, meta = _call(client)
    assert parsed is None and meta["dry_run"] is True
    assert len(client._client.completions.calls) == 0


def test_input_hash_changes_with_image_and_version(conn):
    client = make_client([], conn)
    base = client._input_hash("s", "m", "v1", "sys", "user", [])
    assert base == client._input_hash("s", "m", "v1", "sys", "user", [])
    assert base != client._input_hash("s", "m", "v2", "sys", "user", [])
    img = llm.ImageInput(url="https://x/1.png")
    assert base != client._input_hash("s", "m", "v1", "sys", "user", [img])
    img2 = llm.ImageInput(data_url="data:image/png;base64,AAA=", sha256="deadbeef")
    # identical bytes give identical identity; url vs bytes differ
    assert client._input_hash("s", "m", "v1", "sys", "user", [img2]) == client._input_hash(
        "s", "m", "v1", "sys", "user", [img2]
    )


def test_json_object_fallback_on_unsupported_schema(conn):
    client = make_client([_bad_request(), VALID], conn)
    parsed, meta = _call(client)
    assert parsed == {"ok": True}
    calls = client._client.completions.calls
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert client._response_mode[MODEL] == "json_object"


def test_transient_error_retries_then_succeeds(conn):
    client = make_client([_rate_limit(), VALID], conn, max_retries=2)
    parsed, _ = _call(client)
    assert parsed == {"ok": True}
    assert len(client._client.completions.calls) == 2


def test_call_many_results_in_order(conn):
    client = make_client([VALID, VALID, VALID], conn)
    reqs = [
        dict(stage="triage", model=MODEL, system="s", user_content=f"u{i}",
             schema=SCHEMA, prompt_version="v1")
        for i in range(3)
    ]
    results = client.call_many(reqs)
    assert [r.index for r in results] == [0, 1, 2]
    assert all(r.parsed == {"ok": True} and r.error is None for r in results)


# ---------------------------------------------------------------------------
# W3 / contract C4: timeout_seconds, iter_many, independent 429 retries
# ---------------------------------------------------------------------------
def _req(i):
    return dict(
        stage="triage", model=MODEL, system="s", user_content=f"u{i}",
        schema=SCHEMA, prompt_version="v1",
    )


def test_input_hash_golden_values():
    """C4 invariant: input_hash is byte-identical to the pre-change code.

    Golden digests were captured from the b87e626 hash path (unchanged by W0).
    """
    client = llm.LLMClient(provider="opencode")
    image = llm.ImageInput(
        data_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
        sha256="ab" * 32,
    )
    hashes = {
        "p1": client._input_hash(
            "p1", "space-bunny-free", P1.version, P1.system,
            "TITLE: Cutaneous manifestations of lupus\n"
            "ABSTRACT: A narrative review.",
            [],
        ),
        "p2": client._input_hash(
            "p2", "space-bunny-free", P2.version, P2.system,
            "FIGURES:\n[PMC123:f1] Figure 1. Clinical photograph of malar rash.",
            [],
        ),
        "p3": client._input_hash(
            "p3", "space-bunny-free", P3.version, P3.system,
            "FIGURE PMC123:f1\n"
            "CAPTION: Figure 1. Clinical photograph of malar rash.",
            [image],
        ),
    }
    assert hashes == {
        "p1": "12823cc4dfd929a47213daa432b10138afaf92c1fdc33db4c4836d4bc051eec1",
        "p2": "86786291cfe3718a8682da343cd10c0db9d0b840f47f0382da41085a8f734c0d",
        "p3": "3eeeafde065660bb572188f942b208eeb6ef849aae9530b4296e117ec8c881da",
    }


def test_timeout_seconds_configures_openai_client(monkeypatch):
    captured = {}

    class _SpyOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", _SpyOpenAI)

    client = llm.LLMClient(provider="opencode", timeout_seconds=42.0)
    client.api_key = "test-key"
    client._openai()
    assert captured["timeout"] == 42.0

    default_client = llm.LLMClient(provider="opencode")
    default_client.api_key = "test-key"
    default_client._openai()
    assert captured["timeout"] == config.VP_LLM_TIMEOUT_SECONDS


def test_iter_many_completion_order_and_cap(conn, monkeypatch):
    """Later-index calls finish first; results stream in completion order."""
    client = make_client([], conn)
    n = 4
    delays = {0: 0.30, 1: 0.20, 2: 0.10, 3: 0.0}
    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}
    all_started = threading.Event()

    def fake_call_json(**kw):
        i = int(kw["user_content"][1:])
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
            if state["in_flight"] == n:
                all_started.set()
        try:
            # Block until all cap slots are occupied so peak and the finish
            # order are deterministic regardless of thread scheduling.
            assert all_started.wait(timeout=10)
            time.sleep(delays[i])
            return {"i": i}, {"cached": False}
        finally:
            with lock:
                state["in_flight"] -= 1

    monkeypatch.setattr(client, "call_json", fake_call_json)
    results = list(client.iter_many([_req(i) for i in range(n)], max_in_flight=4))
    assert state["peak"] == 4
    assert [r.index for r in results] == [3, 2, 1, 0]
    assert [r.parsed["i"] for r in results] == [3, 2, 1, 0]
    assert all(r.error is None for r in results)


def test_iter_many_respects_max_in_flight_and_captures_errors(conn, monkeypatch):
    client = make_client([], conn)
    delays = {0: 0.20, 1: 0.0, 2: 0.10, 3: 0.05, 4: 0.0, 5: 0.10}
    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}

    def fake_call_json(**kw):
        i = int(kw["user_content"][1:])
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        try:
            time.sleep(delays[i])
            if i == 2:
                raise ValueError("boom")
            return {"i": i}, {"cached": False}
        finally:
            with lock:
                state["in_flight"] -= 1

    monkeypatch.setattr(client, "call_json", fake_call_json)
    results = list(client.iter_many([_req(i) for i in range(6)], max_in_flight=2))
    assert 1 <= state["peak"] <= 2
    assert results[0].index == 1  # zero-delay call completes first
    assert sorted(r.index for r in results) == list(range(6))
    errors = {r.index: r.error for r in results if r.error is not None}
    assert set(errors) == {2}
    assert isinstance(errors[2], ValueError)


def test_iter_many_default_max_in_flight_is_concurrency(conn, monkeypatch):
    client = make_client([], conn, concurrency=2)
    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}

    def fake_call_json(**kw):
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        try:
            time.sleep(0.02)
            return {"ok": True}, {"cached": False}
        finally:
            with lock:
                state["in_flight"] -= 1

    monkeypatch.setattr(client, "call_json", fake_call_json)
    results = list(client.iter_many(_req(i) for i in range(6)))
    assert 1 <= state["peak"] <= 2
    assert sorted(r.index for r in results) == list(range(6))


def test_iter_many_consumes_requests_lazily(conn):
    """iter_many must not drain the request iterable ahead of free slots.

    The generator blocks before yielding item 3 until the consumer has seen
    results; an implementation that materializes ``list(requests)`` up front
    would hit the wait timeout and fail.
    """
    client = make_client([VALID] * 5, conn)
    gate = threading.Event()
    pulled = []

    def gen():
        for i in range(5):
            pulled.append(i)
            if i == 4:
                # The last item can only be pulled after at least one batch
                # of results has been yielded, so the consumer has already
                # opened the gate. Eager draining would hit this wait with
                # the gate closed and fail.
                assert gate.wait(timeout=10), "iterable drained eagerly"
            yield _req(i)

    results = []
    for r in client.iter_many(gen(), max_in_flight=2):
        results.append(r)
        gate.set()
    # Calls complete near-simultaneously, so assert set membership rather
    # than a specific completion order here.
    assert sorted(r.index for r in results) == [0, 1, 2, 3, 4]
    assert pulled == [0, 1, 2, 3, 4]


def test_rate_limit_retries_honor_retry_after(conn, monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    client = make_client(
        [_rate_limit(retry_after="7"), _rate_limit(retry_after="7"), VALID],
        conn,
        max_retries=0,
    )
    parsed, _ = _call(client)
    assert parsed == {"ok": True}
    assert sleeps == [7.0, 7.0]
    assert len(client._client.completions.calls) == 3


def test_rate_limit_fallback_exponential_backoff(conn, monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    client = make_client([_rate_limit(), _rate_limit(), VALID], conn)
    _call(client)
    assert sleeps == [1.0, 2.0]  # 2**attempt, no Retry-After header


def test_rate_limit_retry_after_capped_at_30s(conn, monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    client = make_client([_rate_limit(retry_after="3600"), VALID], conn)
    parsed, _ = _call(client)
    assert parsed == {"ok": True}
    assert sleeps == [30.0]


def test_rate_limit_budget_exhausted(conn, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(config, "VP_RATE_LIMIT_RETRIES", 2)
    client = make_client([_rate_limit()] * 3, conn)
    with pytest.raises(openai.RateLimitError):
        _call(client)
    # initial attempt + VP_RATE_LIMIT_RETRIES retries
    assert len(client._client.completions.calls) == 3


def test_429_budget_independent_of_max_retries(conn, monkeypatch):
    """max_retries=0 still retries 429s on the VP_RATE_LIMIT_RETRIES budget."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = make_client([_rate_limit(), VALID], conn, max_retries=0)
    parsed, _ = _call(client)
    assert parsed == {"ok": True}
    assert len(client._client.completions.calls) == 2


def test_timeout_still_obeys_max_retries_zero(conn, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = make_client([_timeout()], conn, max_retries=0)
    with pytest.raises(openai.APITimeoutError):
        _call(client)
    assert len(client._client.completions.calls) == 1


def test_mixed_transient_and_429_budgets(conn, monkeypatch):
    """A timeout consumes max_retries; a 429 consumes the 429 budget."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    client = make_client([_timeout(), _rate_limit(), VALID], conn, max_retries=1)
    parsed, _ = _call(client)
    assert parsed == {"ok": True}
    # transient backoff 2**0=1, then 429 fallback backoff 2**0=1
    assert sleeps == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Prompt schemas accept/reject examples
# ---------------------------------------------------------------------------
VALID_EXAMPLES = {
    P1.name: {
        "primary_disease_keys": ["sle"],
        "is_narrative_review": True,
        "decision": "relevant",
        "reason": "review of SLE",
    },
    P2.name: {
        "results": [
            {
                "figure_id": "PMC1:f1",
                "category": "clinical_photo",
                "is_real_patient_image": True,
                "third_party": False,
                "third_party_quote": None,
                "diseases_mentioned": ["dm"],
                "route": "keep",
                "reason": "patient photo",
            }
        ]
    },
    P3.name: {
        "figure_id": "PMC1:f1",
        "figure_is_compound": False,
        "panels": [
            {
                "panel_label": "A",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "include": True,
                "exclusion_reason": None,
                "disease_key": "dm",
                "subtype": "classic",
                "modality": "clinical_photo",
                "body_site": "hand",
                "findings": [{"finding_key": "gottron_papules", "evidence": "caption"}],
                "proposed_findings": [],
                "typicality": "classic",
                "stage": None,
                "age_group": "adult",
                "skin_tone": "medium",
                "stated_ethnicity": None,
                "stated_ethnicity_quote": None,
                "annotations_present": False,
                "confidence": 0.9,
                "rationale": "Gottron papules visible",
            }
        ],
    },
    P4.name: {
        "assertions": [
            {
                "disease_key": "dm",
                "subtype": None,
                "finding_key": "gottron_papules",
                "proposed_finding": None,
                "frequency_text": "most patients",
                "pct_low": None,
                "pct_high": None,
                "specificity_text": "pathognomonic",
                "quote": "Gottron papules are seen in most patients",
            }
        ]
    },
}

INVALID_EXAMPLES = {
    P1.name: {"primary_disease_keys": ["sle"], "decision": "relevant"},  # missing fields
    P2.name: {"results": [{"figure_id": "f", "category": "bogus", "route": "keep"}]},
    P3.name: {
        "figure_id": "f",
        "figure_is_compound": False,
        "panels": [
            {**VALID_EXAMPLES[P3.name]["panels"][0], "confidence": 1.5}  # out of range
        ],
    },
    P4.name: {"assertions": [{"disease_key": "dm"}]},  # missing required fields
}


@pytest.mark.parametrize("prompt", list(PROMPTS.values()), ids=lambda p: p.name)
def test_prompt_schema_accepts_valid_example(prompt):
    jsonschema.validate(VALID_EXAMPLES[prompt.name], prompt.schema)


@pytest.mark.parametrize("prompt", list(PROMPTS.values()), ids=lambda p: p.name)
def test_prompt_schema_rejects_invalid_example(prompt):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(INVALID_EXAMPLES[prompt.name], prompt.schema)


def test_prompts_end_with_return_only_json():
    for prompt in PROMPTS.values():
        assert prompt.system.rstrip().endswith("Return only JSON.")
        assert prompt.version


def test_call_prompt_uses_prompt_version(conn):
    client = make_client([json.dumps(VALID_EXAMPLES[P1.name])], conn)
    client.call_prompt("p1", P1, MODEL, "user text")
    row = conn.execute("SELECT request_meta_json FROM llm_calls").fetchone()
    assert db.from_json(row["request_meta_json"])["prompt_version"] == P1.version
