"""W5b tests for src/visual_pilot/triage.py — mocked LLM, no network."""

import argparse
import json

from src.visual_pilot import db, llm, triage


def _args(**over):
    base = dict(
        disease="all",
        limit=None,
        dry_run=False,
        budget_usd=None,
        cap=None,
        pmcids=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _article(conn, pmcid="PMC1", keys=("sle",)):
    conn.execute(
        "INSERT OR IGNORE INTO articles (pmcid, title, primary_disease_keys_json, "
        "status) VALUES (?, ?, ?, 'parsed')",
        (pmcid, f"title {pmcid}", db.to_json(list(keys))),
    )
    conn.commit()


def _figure(conn, figure_id, pmcid="PMC1", status="pending", caption="a caption"):
    conn.execute(
        "INSERT INTO figures (figure_id, pmcid, label, caption, status, "
        "in_text_mentions_json) VALUES (?, ?, ?, ?, ?, '[]')",
        (figure_id, pmcid, "Figure 1", caption, status),
    )
    conn.commit()


def _p2_item(figure_id, route, **kw):
    item = {
        "figure_id": figure_id,
        "category": "clinical_photo",
        "is_real_patient_image": True,
        "third_party": False,
        "third_party_quote": None,
        "diseases_mentioned": ["sle"],
        "route": route,
        "reason": f"routed {route}",
    }
    item.update(kw)
    return item


class FakeLLM:
    """Serves canned call_many results and records requests."""

    def __init__(self, outcomes, **kwargs):
        self.kwargs = kwargs
        self.requests = []
        self.spent_usd = 0.0
        self._outcomes = list(outcomes)

    def call_many(self, requests):
        reqs = list(requests)
        self.requests.extend(reqs)
        results = []
        for i, _req in enumerate(reqs):
            outcome = self._outcomes[i] if i < len(self._outcomes) else {"results": []}
            if isinstance(outcome, Exception):
                results.append(llm.BatchResult(index=i, error=outcome))
            else:
                results.append(llm.BatchResult(index=i, parsed=outcome, meta={}))
        return results


def _install_fake(monkeypatch, outcomes):
    holder = {}

    def factory(**kwargs):
        client = FakeLLM(outcomes, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr(llm, "LLMClient", factory)
    return holder


def _status(conn, figure_id):
    return conn.execute(
        "SELECT status FROM figures WHERE figure_id=?", (figure_id,)
    ).fetchone()["status"]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_triage_routes_all_routes(conn, monkeypatch):
    _article(conn)
    for i in range(4):
        _figure(conn, f"PMC1:f{i}")
    response = {
        "results": [
            _p2_item("PMC1:f0", "keep"),
            _p2_item("PMC1:f1", "drop"),
            _p2_item("PMC1:f2", "uncertain"),
            _p2_item("PMC1:f3", "keep", third_party=True, third_party_quote="© X"),
        ]
    }
    _install_fake(monkeypatch, [response])
    assert triage.run(_args()) == 0

    assert _status(conn, "PMC1:f0") == "caption_kept"
    assert _status(conn, "PMC1:f1") == "caption_rejected"
    assert _status(conn, "PMC1:f2") == "caption_uncertain"
    # third_party=true rejects even when route says keep
    assert _status(conn, "PMC1:f3") == "caption_rejected"
    triage_row = db.from_json(
        conn.execute(
            "SELECT triage_json FROM figures WHERE figure_id='PMC1:f1'"
        ).fetchone()[0]
    )
    assert triage_row["route"] == "drop"
    assert triage_row["reason"] == "routed drop"


def test_missing_result_stays_pending_with_attempt(conn, monkeypatch):
    _article(conn)
    _figure(conn, "PMC1:f0")
    _figure(conn, "PMC1:f1")
    _install_fake(monkeypatch, [{"results": [_p2_item("PMC1:f0", "keep")]}])
    assert triage.run(_args()) == 0
    assert _status(conn, "PMC1:f0") == "caption_kept"
    row = conn.execute(
        "SELECT status, attempts FROM figures WHERE figure_id='PMC1:f1'"
    ).fetchone()
    assert (row["status"], row["attempts"]) == ("pending", 1)


def test_batch_failure_bumps_attempts_and_sets_error(conn, monkeypatch):
    _article(conn)
    _figure(conn, "PMC1:f0")
    err = llm.SchemaValidationError("bad json")
    _install_fake(monkeypatch, [err])
    assert triage.run(_args()) == 0
    row = conn.execute(
        "SELECT status, attempts, error FROM figures WHERE figure_id='PMC1:f0'"
    ).fetchone()
    assert (row["status"], row["attempts"]) == ("pending", 1)
    assert "bad json" in row["error"]


def test_batching_size_40(conn, monkeypatch):
    _article(conn)
    for i in range(41):
        _figure(conn, f"PMC1:f{i:03d}")
    holder = _install_fake(monkeypatch, [{"results": []}, {"results": []}])
    assert triage.run(_args()) == 0
    reqs = holder["client"].requests
    assert len(reqs) == 2
    sizes = [len(json.loads(r["user_content"])["figures"]) for r in reqs]
    assert sizes == [40, 1]
    assert all(r["stage"] == "p2" for r in reqs)


def test_limit_caps_figures(conn, monkeypatch):
    _article(conn)
    for i in range(5):
        _figure(conn, f"PMC1:f{i}")
    holder = _install_fake(monkeypatch, [{"results": []}])
    assert triage.run(_args(limit=2)) == 0
    assert len(json.loads(holder["client"].requests[0]["user_content"])["figures"]) == 2


def test_disease_scope(conn, monkeypatch):
    _article(conn, "PMC_SLE", keys=("sle",))
    _article(conn, "PMC_DM", keys=("dm",))
    _figure(conn, "PMC_SLE:f1", "PMC_SLE")
    _figure(conn, "PMC_DM:f1", "PMC_DM")
    _install_fake(
        monkeypatch, [{"results": [_p2_item("PMC_DM:f1", "keep")]}]
    )
    assert triage.run(_args(disease="dm")) == 0
    assert _status(conn, "PMC_DM:f1") == "caption_kept"
    assert _status(conn, "PMC_SLE:f1") == "pending"


# ---------------------------------------------------------------------------
# Cache: a rerun makes zero new LLM calls
# ---------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "R",
            (),
            {
                "choices": [_FakeMessage(self.content)],
                "usage": type(
                    "U", (), {"prompt_tokens": 10, "completion_tokens": 5}
                ),
            },
        )


class _FakeOpenAI:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)
        self.chat = type("Chat", (), {"completions": self.completions})


def test_rerun_makes_zero_llm_calls(conn, monkeypatch):
    """The llm_calls cache answers the identical rerun for free."""
    _article(conn)
    _figure(conn, "PMC1:f0")
    content = json.dumps({"results": [_p2_item("PMC1:f0", "keep")]})

    real_client = llm.LLMClient(provider="deepinfra", db_conn=conn)
    fake_openai = _FakeOpenAI(content)
    real_client._client = fake_openai
    monkeypatch.setattr(llm, "LLMClient", lambda **kw: real_client)

    assert triage.run(_args()) == 0
    assert _status(conn, "PMC1:f0") == "caption_kept"
    assert len(fake_openai.completions.calls) == 1

    # Second run: figure already routed -> no pending rows -> no requests.
    assert triage.run(_args()) == 0
    assert len(fake_openai.completions.calls) == 1

    # Force a rerun on the same input: reset to pending; the cache must serve
    # the identical P2 request with zero new provider calls.
    conn.execute(
        "UPDATE figures SET status='pending', triage_json=NULL WHERE figure_id='PMC1:f0'"
    )
    conn.commit()
    assert triage.run(_args()) == 0
    assert _status(conn, "PMC1:f0") == "caption_kept"
    assert len(fake_openai.completions.calls) == 1


def test_budget_exceeded_leaves_pending(conn, monkeypatch):
    _article(conn)
    for i in range(41):
        _figure(conn, f"PMC1:f{i:03d}")
    outcomes = [
        llm.BudgetExceeded("spent"),
        llm.BudgetExceeded("spent"),
    ]
    _install_fake(monkeypatch, outcomes)
    assert triage.run(_args()) == 0
    rows = conn.execute(
        "SELECT status, attempts FROM figures WHERE status='pending'"
    ).fetchall()
    assert len(rows) == 41
    assert all(r["attempts"] == 0 for r in rows)
