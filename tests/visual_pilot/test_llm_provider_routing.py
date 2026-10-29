"""Restricted OpenRouter routing and isolation from text-provider traffic."""

import json

import pytest

from src.visual_pilot import db, llm

MODEL = "z-ai/glm-5.3-flash"
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _client(handler):
    client = llm.LLMClient(max_retries=0)
    client._router_client = llm.openai.OpenAI(
        api_key="router-test-key", base_url="https://router.invalid/v1",
        http_client=llm.openai.DefaultHttpxClient(transport=llm._openai_httpx.MockTransport(handler)),
    )
    return client


def _reply(text):
    chunk = {"id": "gen-test", "object": "chat.completion.chunk", "created": 0,
             "model": MODEL, "provider": "OpenInference",
             "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                       "cost": 0.0001}}
    return llm._openai_httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")


@pytest.mark.parametrize("effort,reasoning", [("low", {"effort": "low"}), ("off", {"enabled": False})])
def test_visual_request_is_strictly_allowlisted_and_records_actual_provider(effort, reasoning):
    sent = []

    def handle(request):
        assert request.headers["Authorization"] == "Bearer router-test-key"
        sent.append(json.loads(request.content))
        return _reply('{"ok":true}')

    client = _client(handle)
    try:
        parsed, meta = client.call_json("p3", MODEL, "sys", "user", SCHEMA,
                                       images=[llm.ImageInput(data_url="data:image/png;base64,AAAA")],
                                       reasoning_effort=effort)
        assert parsed == {"ok": True}
        body = sent[0]
        assert body["model"] == MODEL
        assert body["provider"]["only"] == ["open-inference"]
        assert body["provider"]["order"] == ["open-inference"]
        assert body["provider"]["allow_fallbacks"] is False
        assert body["provider"]["require_parameters"] is True
        assert body["reasoning"] == reasoning
        assert "thinking" not in body
        assert body["max_tokens"] == llm.config.VP_LLM_MAX_OUTPUT_TOKENS
        assert body["messages"][1]["content"][1]["type"] == "image_url"
        assert meta["provider"] == "openrouter"
        assert meta["upstream_provider"] == "OpenInference"
        assert meta["response_id"] == "gen-test"
        assert meta["cost_usd"] == 0.0001
        assert client._client is None and client._go_client is None
    finally:
        client._router_client.close()


def test_schema_repair_preserves_allowlist_and_counts_both_calls():
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return _reply('{"ok":"invalid"}' if len(sent) == 1 else '{"ok":true}')

    client = _client(handle)
    try:
        _, meta = client.call_json("p3", MODEL, "sys", "user", SCHEMA)
        assert meta["attempts"] == 2
        assert meta["cost_usd"] == pytest.approx(0.0002)
        assert all(body["provider"]["only"] == ["open-inference"] for body in sent)
    finally:
        client._router_client.close()


def test_no_allowed_endpoint_fails_without_unrestricted_retry():
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return llm._openai_httpx.Response(404, json={"error": {"message": "No allowed providers"}})

    client = _client(handle)
    try:
        with pytest.raises(llm.openai.NotFoundError):
            client.call_json("p3", MODEL, "sys", "user", SCHEMA)
        assert len(sent) == 1
        assert sent[0]["provider"]["only"] == ["open-inference"]
    finally:
        client._router_client.close()


def test_router_credentials_and_endpoint_are_separate(monkeypatch):
    monkeypatch.setattr(llm.config, "OPENROUTER_API_KEY", "router-key")
    monkeypatch.setattr(llm.config, "OPENROUTER_BASE_URL", "https://router.invalid/v1")
    client = llm.LLMClient()
    router = client._openai(llm.OPENROUTER)
    try:
        assert router.api_key == "router-key"
        assert str(router.base_url) == "https://router.invalid/v1/"
        assert "x-opencode-session" not in router.default_headers
        assert client._client is None and client._go_client is None
    finally:
        router.close()


def test_restricted_routes_cannot_hit_an_unrestricted_old_cache():
    payload = {"prompt_version": "v", "system": "s", "user_content": "u", "images": []}
    old = db.llm_input_hash("p3", MODEL, payload)
    assert llm.LLMClient()._input_hash("p3", MODEL, "v", "s", "u", []) != old


def test_missing_router_key_does_not_use_another_provider(monkeypatch):
    monkeypatch.setattr(llm.config, "OPENROUTER_API_KEY", None)
    with pytest.raises(llm.LLMError, match="OPENROUTER_API_KEY"):
        llm.LLMClient()._openai(llm.OPENROUTER)


def test_mixed_text_and_vision_batch_keeps_routes_isolated():
    sent = []

    def handle(request):
        sent.append((request.headers["Authorization"], json.loads(request.content)))
        return _reply('{"ok":true}')

    client = _client(handle)
    client._go_client = llm.openai.OpenAI(
        api_key="go-test-key", base_url="https://go.invalid/v1",
        http_client=llm.openai.DefaultHttpxClient(transport=llm._openai_httpx.MockTransport(handle)),
    )
    common = dict(system="sys", user_content="user", schema=SCHEMA, reasoning_effort="low")
    try:
        results = client.call_many([
            dict(stage="p2", model="opencode-go/iceman-contributor-free", **common),
            dict(stage="p3", model=MODEL, images=[llm.ImageInput(url="https://image.invalid/a.jpg")], **common),
        ])
        assert all(result.error is None for result in results)
        assert [r.meta["provider"] for r in results] == ["opencode-go", "openrouter"]
        by_key = dict(sent)
        assert "provider" not in by_key["Bearer go-test-key"]
        assert "response_format" not in by_key["Bearer go-test-key"]
        assert by_key["Bearer go-test-key"]["model"] == "iceman-contributor-free"
        assert by_key["Bearer router-test-key"]["provider"]["only"] == ["open-inference"]
    finally:
        client._router_client.close()
        client._go_client.close()
