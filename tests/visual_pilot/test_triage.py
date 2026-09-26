"""W5b tests for src/visual_pilot/triage.py — mocked LLM, no network."""

import argparse
import json

from src.visual_pilot import db, diseases, llm, triage


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
    _article(conn, "PMC_DM", keys=("dm", "sle"))
    _figure(conn, "PMC_SLE:f1", "PMC_SLE")
    _figure(conn, "PMC_DM:f1", "PMC_DM")
    _install_fake(
        monkeypatch, [{"results": [_p2_item("PMC_DM:f1", "keep")]}]
    )
    assert triage.run(_args(disease="dm")) == 0
    assert _status(conn, "PMC_DM:f1") == "caption_kept"
    assert _status(conn, "PMC_SLE:f1") == "pending"


def test_drop_with_explicit_patient_target_caption_routes_uncertain(conn):
    _article(conn, "PMC_DM", keys=("dm",))
    _figure(
        conn,
        "PMC_DM:f1",
        "PMC_DM",
        caption="Clinical photograph of Gottron papules in a patient with dermatomyositis.",
    )
    row = dict(conn.execute("SELECT * FROM figures WHERE figure_id='PMC_DM:f1'").fetchone())
    response = {
        "results": [
            _p2_item(
                "PMC_DM:f1",
                "drop",
                is_real_patient_image=True,
                diseases_mentioned=["sle"],
                reason="not a target disease",
            )
        ]
    }
    triage._apply_batch(conn, [row], llm.BatchResult(index=0, parsed=response, meta={}))
    stored = conn.execute(
        "SELECT status,triage_json FROM figures WHERE figure_id='PMC_DM:f1'"
    ).fetchone()
    assert stored["status"] == "caption_uncertain"
    assert db.from_json(stored["triage_json"], {})["route_adjustment"] == (
        "uncertain_due_to_conflicting_patient_or_target_evidence"
    )


def test_drop_reason_keep_alone_does_not_promote_diagram(conn):
    _article(conn, "PMC_DM_KEEP", keys=("dm",))
    _figure(conn, "PMC_DM_KEEP:f1", "PMC_DM_KEEP", caption="Unclear figure.")
    row = dict(conn.execute("SELECT * FROM figures WHERE figure_id='PMC_DM_KEEP:f1'").fetchone())
    response = {"results": [_p2_item(
        "PMC_DM_KEEP:f1", "drop", is_real_patient_image=False, reason="keep"
    )]}
    triage._apply_batch(conn, [row], llm.BatchResult(index=0, parsed=response, meta={}))
    assert _status(conn, "PMC_DM_KEEP:f1") == "caption_rejected"


def test_historical_conflicting_drop_is_requeued_once_with_license_gate(conn):
    _article(conn, "PMC_OLD", keys=("dm",))
    _figure(conn, "PMC_OLD:good", "PMC_OLD", status="caption_rejected", caption="Clinical photograph of Gottron papules in dermatomyositis")
    _figure(conn, "PMC_OLD:bad_license", "PMC_OLD", status="caption_rejected", caption="Clinical photograph of Gottron papules in dermatomyositis")
    for fid, license_code in (("PMC_OLD:good", "cc-by"), ("PMC_OLD:bad_license", "cc-by-nc")):
        conn.execute(
            "UPDATE figures SET effective_license=?, image_url=?, triage_json=? WHERE figure_id=?",
            (license_code, "https://example.test/figure.jpg", db.to_json(_p2_item(fid, "drop", reason="keep")), fid),
        )
    conn.commit()
    assert triage.revisit_conflicting_rejections(conn, dry_run=True) == 1
    assert _status(conn, "PMC_OLD:good") == "caption_rejected"
    assert triage.revisit_conflicting_rejections(conn) == 1
    assert _status(conn, "PMC_OLD:good") == "caption_uncertain"
    assert _status(conn, "PMC_OLD:bad_license") == "caption_rejected"
    assert triage.revisit_conflicting_rejections(conn) == 0


def test_historical_other_disease_image_is_not_requeued(conn):
    _article(conn, "PMC_OTHER", keys=("dm", "sle"))
    _figure(
        conn, "PMC_OTHER:f1", "PMC_OTHER", status="caption_rejected",
        caption="Renal infarct in an infant with DADA2.",
    )
    conn.execute(
        "UPDATE figures SET effective_license='cc-by', image_url=?, triage_json=?, "
        "in_text_mentions_json=? WHERE figure_id='PMC_OTHER:f1'",
        (
            "https://example.test/figure.jpg",
            db.to_json(_p2_item("PMC_OTHER:f1", "drop", reason="keep")),
            db.to_json(["Lupus and dermatomyositis are discussed elsewhere in the article."]),
        ),
    )
    conn.commit()
    assert triage.revisit_conflicting_rejections(conn) == 0
    assert _status(conn, "PMC_OTHER:f1") == "caption_rejected"


def test_contradictory_patient_signal_does_not_override_third_party_rejection(conn):
    _article(conn, "PMC_DM2", keys=("dm",))
    _figure(
        conn,
        "PMC_DM2:f1",
        "PMC_DM2",
        caption="Clinical photograph of Gottron papules in a patient with dermatomyositis.",
    )
    row = dict(conn.execute("SELECT * FROM figures WHERE figure_id='PMC_DM2:f1'").fetchone())
    response = {
        "results": [
            _p2_item(
                "PMC_DM2:f1",
                "drop",
                is_real_patient_image=True,
                third_party=True,
                third_party_quote="reproduced with permission",
                reason="keep",
            )
        ]
    }
    triage._apply_batch(conn, [row], llm.BatchResult(index=0, parsed=response, meta={}))
    assert _status(conn, "PMC_DM2:f1") == "caption_rejected"


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

    real_client = llm.LLMClient(provider="opencode", db_conn=conn)
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


# ---------------------------------------------------------------------------
# W10: shared per-run _RunContext must not change routing (plan §5 W10, B10)
# ---------------------------------------------------------------------------
# (figure_id suffix, pmcid, caption, image_category, expected) for
# _explicit_target_visual: disease-name hits, vocab hits, vocab disease-key
# misses, other-disease misses and non-image misses.
_EXPLICIT_FIXTURES = [
    (
        "PMC_DM:hit_disease",
        "Clinical photograph of Gottron papules in a patient with dermatomyositis.",
        "clinical_photo",
        True,
    ),
    ("PMC_DM:hit_vocab", "Clinical photograph of heliotrope rash.", None, True),
    (
        "PMC_DM:miss_vocab_disease_mismatch",
        "Photograph showing a butterfly rash.",
        None,
        False,
    ),
    (
        "PMC_DM:miss_other_disease",
        "Photograph of skin changes in systemic sclerosis.",
        "clinical_photo",
        False,
    ),
    (
        "PMC_DM:miss_no_target",
        "Bar chart of patient enrollment over time.",
        "clinical_photo",
        False,
    ),
    ("PMC_SLE:hit_vocab", "Photograph showing a butterfly rash.", None, True),
    (
        "PMC_SLE:hit_disease_synonym",
        "Ultrasound of the hand in a patient with lupus.",
        None,
        True,
    ),
    ("PMC_SLE:miss_not_image", "Table of laboratory values.", "table", False),
]


def _explicit_fixture_row(conn, figure_id):
    return dict(
        conn.execute(
            "SELECT * FROM figures WHERE figure_id=?", (figure_id,)
        ).fetchone()
    )


def test_shared_context_routing_matches_fresh_context(conn):
    """A shared _RunContext returns identical decisions and queries once."""
    diseases.seed(conn)
    _article(conn, "PMC_DM", keys=("dm",))
    _article(conn, "PMC_SLE", keys=("sle",))
    for figure_id, caption, _category, _expected in _EXPLICIT_FIXTURES:
        _figure(conn, figure_id, figure_id.split(":")[0], caption=caption)

    # Uncached path first: each call builds its own context, matching the old
    # per-row reload behavior this workstream replaced.
    fresh = {
        figure_id: triage._explicit_target_visual(
            conn, _explicit_fixture_row(conn, figure_id), category
        )
        for figure_id, _caption, category, _expected in _EXPLICIT_FIXTURES
    }
    for figure_id, _caption, _category, expected in _EXPLICIT_FIXTURES:
        assert fresh[figure_id] == expected, figure_id

    queries = {"vocab": 0, "articles": 0}

    def trace(statement):
        if "FROM findings_vocab" in statement:
            queries["vocab"] += 1
        if "FROM articles" in statement:
            queries["articles"] += 1

    shared = triage._RunContext(conn)
    conn.set_trace_callback(trace)
    try:
        for figure_id, _caption, category, _expected in _EXPLICIT_FIXTURES:
            cached = triage._explicit_target_visual(
                conn, _explicit_fixture_row(conn, figure_id), category, ctx=shared
            )
            assert cached == fresh[figure_id], figure_id
    finally:
        conn.set_trace_callback(None)

    # The cached path loads vocab once per run and article keys once per pmcid
    # (the two fixture figures share each pmcid's row lookup).
    assert queries["vocab"] == 1
    assert queries["articles"] == len(
        {figure_id.split(":")[0] for figure_id, *_ in _EXPLICIT_FIXTURES}
    )


def test_run_routing_outcomes_with_shared_context(conn, monkeypatch):
    """One run covers every routing outcome; cached lookups change nothing."""
    diseases.seed(conn)
    _article(conn, "PMC_DM", keys=("dm",))
    _article(conn, "PMC_SLE", keys=("sle",))
    cases = [
        ("PMC_DM:keep", "PMC_DM", "a caption",
         _p2_item("PMC_DM:keep", "keep"), "caption_kept"),
        ("PMC_DM:uncertain", "PMC_DM", "a caption",
         _p2_item("PMC_DM:uncertain", "uncertain"), "caption_uncertain"),
        ("PMC_DM:drop", "PMC_DM", "a caption",
         _p2_item("PMC_DM:drop", "drop", is_real_patient_image=False),
         "caption_rejected"),
        ("PMC_DM:third_party", "PMC_DM", "a caption",
         _p2_item("PMC_DM:third_party", "keep",
                  third_party=True, third_party_quote="© X"),
         "caption_rejected"),
        # Contradictory drop promoted via disease-name + patient-image hit.
        ("PMC_DM:contra_disease", "PMC_DM",
         "Clinical photograph of Gottron papules in dermatomyositis.",
         _p2_item("PMC_DM:contra_disease", "drop", is_real_patient_image=True),
         "caption_uncertain"),
        # Contradictory drop promoted via a vocab term for the article disease.
        ("PMC_SLE:contra_vocab", "PMC_SLE",
         "Photograph showing a butterfly rash.",
         _p2_item("PMC_SLE:contra_vocab", "drop",
                  is_real_patient_image=True, category=None),
         "caption_uncertain"),
        # Drop stays rejected: patient image but no target-disease evidence.
        ("PMC_DM:contra_miss", "PMC_DM", "Bar chart of enrollment.",
         _p2_item("PMC_DM:contra_miss", "drop",
                  is_real_patient_image=True, category=None),
         "caption_rejected"),
        # Drop stays rejected: caption points at a non-target disease.
        ("PMC_DM:contra_other", "PMC_DM",
         "Photograph of skin changes in systemic sclerosis.",
         _p2_item("PMC_DM:contra_other", "drop", is_real_patient_image=True),
         "caption_rejected"),
    ]
    for figure_id, pmcid, caption, _item, _expected in cases:
        _figure(conn, figure_id, pmcid, caption=caption)

    # A historical contradiction re-queued by the shared-context revisit pass.
    _figure(
        conn, "PMC_DM:revisit", "PMC_DM", status="caption_rejected",
        caption="Clinical photograph of Gottron papules in dermatomyositis",
    )
    conn.execute(
        "UPDATE figures SET effective_license='cc-by', image_url=?, triage_json=? "
        "WHERE figure_id='PMC_DM:revisit'",
        (
            "https://example.test/figure.jpg",
            db.to_json(_p2_item("PMC_DM:revisit", "drop", reason="keep")),
        ),
    )
    conn.commit()

    _install_fake(monkeypatch, [{"results": [case[3] for case in cases]}])
    assert triage.run(_args()) == 0

    for figure_id, _pmcid, _caption, _item, expected in cases:
        assert _status(conn, figure_id) == expected, figure_id
    assert _status(conn, "PMC_DM:revisit") == "caption_uncertain"
    promoted = db.from_json(
        conn.execute(
            "SELECT triage_json FROM figures WHERE figure_id='PMC_DM:contra_disease'"
        ).fetchone()[0]
    )
    assert promoted["route"] == "uncertain"
    assert promoted["route_adjustment"] == (
        "uncertain_due_to_conflicting_patient_or_target_evidence"
    )
