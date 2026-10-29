"""OpenCode routing, request contract and text-only model protection."""

import json

import pytest

from src.visual_pilot import llm


MODEL = "opencode-go/iceman-contributor-free"
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def test_go_uses_own_credentials_endpoint_and_stable_session(monkeypatch):
    monkeypatch.setattr(llm.config, "OPENCODE_GO_API_KEY", "go-key")
    monkeypatch.setattr(llm.config, "OPENCODE_GO_BASE_URL", "https://go.invalid/v1")
    monkeypatch.setattr(llm.config, "MIMO_API_KEY", "xiaomi-key")
    client = llm.LLMClient()
    go = client._openai(llm.OPENCODE_GO)
    try:
        assert go.api_key == "go-key"
        assert str(go.base_url) == "https://go.invalid/v1/"
        assert go.default_headers["User-Agent"] == llm.USER_AGENT
        assert go.default_headers["x-opencode-session"] == client._go_session_id
        assert client._openai(llm.OPENCODE_GO) is go
        assert client._client is None
    finally:
        go.close()


def test_go_streamed_json_and_cost_ledger(monkeypatch, tmp_path):
    from src.visual_pilot import db

    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        chunks = [
            {"id": "test", "object": "chat.completion.chunk", "created": 0,
             "model": "iceman-contributor-free", "choices": [
                 {"index": 0, "delta": {"content": '{"ok":true}'}, "finish_reason": None}]},
            {"id": "test", "object": "chat.completion.chunk", "created": 0,
             "model": "iceman-contributor-free", "choices": [],
             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return llm._openai_httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    conn = db.init_db(db.connect(tmp_path / "test.sqlite"))
    client = llm.LLMClient(db_conn=conn)
    client._go_client = llm.openai.OpenAI(
        api_key="test-key", base_url="https://go.invalid/v1",
        http_client=llm.openai.DefaultHttpxClient(transport=llm._openai_httpx.MockTransport(handle)),
    )
    try:
        parsed, meta = client.call_json("p2", MODEL, "system", "user", SCHEMA, reasoning_effort="low")
        assert parsed == {"ok": True}
        assert meta["provider"] == "opencode-go"
        assert meta["cost_usd"] == 0
        assert meta["input_tokens"] == 10
        assert meta["output_tokens"] == 5
        assert sent[0]["model"] == "iceman-contributor-free"
        assert "response_format" not in sent[0]
        assert meta["response_format"] == "prompt_json"
        assert sent[0]["stream"] is True
        assert "thinking" not in sent[0]
        assert "reasoning_effort" not in sent[0]
        assert client._client is None
        row = conn.execute("SELECT request_meta_json FROM llm_calls").fetchone()
        assert json.loads(row["request_meta_json"])["provider"] == "opencode-go"
        _, cached = client.call_json("p2", MODEL, "system", "user", SCHEMA, reasoning_effort="low")
        assert cached["cached"] is True and len(sent) == 1
    finally:
        client._go_client.close()
        conn.close()


def test_iceman_images_fail_before_network_or_cache(monkeypatch):
    client = llm.LLMClient()
    monkeypatch.setattr(client, "_cache_lookup", lambda _: pytest.fail("must reject before cache"))
    with pytest.raises(llm.LLMError, match="text only"):
        client.call_json("p3", MODEL, "system", "user", SCHEMA,
                         images=[llm.ImageInput(data_url="data:image/png;base64,AAAA")])


def test_go_missing_key_does_not_fall_back_to_xiaomi(monkeypatch):
    monkeypatch.setattr(llm.config, "OPENCODE_GO_API_KEY", None)
    monkeypatch.setattr(llm.config, "MIMO_API_KEY", "xiaomi-key")
    with pytest.raises(llm.LLMError, match="OPENCODE_GO_API_KEY"):
        llm.LLMClient()._openai(llm.OPENCODE_GO)


def test_iceman_cache_is_separate_from_mimo():
    client = llm.LLMClient()
    assert client._input_hash("p2", MODEL, "v1", "s", "u", []) != client._input_hash(
        "p2", "mimo-v2.6-flash", "v1", "s", "u", [])
