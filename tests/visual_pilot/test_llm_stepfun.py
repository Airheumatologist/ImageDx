"""StepFun request shape and cost accounting for the LLM client."""

from types import SimpleNamespace

from src.visual_pilot import llm

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


class _Completions:
    def __init__(self, usage):
        self.calls = []
        self.usage = usage

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content='{"ok": true}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=self.usage
        )


def _client(monkeypatch, usage):
    monkeypatch.setattr(
        llm.config, "llm_credentials", lambda provider: ("k", "https://example.invalid/v1")
    )
    client = llm.LLMClient(provider="stepfun")
    completions = _Completions(usage)
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
    assert sent["extra_body"] == {"reasoning_effort": "low"}
    assert "Strict Output Schema" in sent["messages"][0]["content"]


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
