"""Mimic audit (P8): look-alike diseases leave the library, offline with a fake LLM."""

import argparse
import json
import threading

from src.visual_pilot import db, llm, mimic_audit
from balanced_fixtures import add_article, add_disease, add_figure, add_finding, add_panel, make_db

TITLE = "Langerhans Cell Histiocytosis Presented as Persistent Diaper Dermatitis: A Case Report."
ABSTRACT = "A 9-month-old infant with a diaper rash unresponsive to treatment had LCH on biopsy."


class FakeClient:
    verdict = "other_disease"

    def __init__(self, db_conn=None, budget_usd=None):
        self.spent_usd = 0.0
        self.db_lock = threading.Lock()
        self.seen = []

    def iter_many(self, requests):
        for i, req in enumerate(requests):
            self.seen.append(json.loads(req["user_content"]))
            yield llm.BatchResult(
                index=i,
                parsed={"verdict": self.verdict, "actual_diagnosis": "Langerhans cell histiocytosis",
                        "reason": "The title gives LCH as the diagnosis."},
                meta={"cached": False},
            )


def _seed(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "contact_dermatitis", "Allergic and Irritant Contact Dermatitis")
    add_finding(conn, "dd_erythema", ("contact_dermatitis",), label="Diaper erythema")
    add_article(conn, "PMC1", title=TITLE, publication_types=("Case Reports",))
    conn.execute(
        "INSERT INTO article_source_metadata (pmcid, source, metadata_json, fetched_at) "
        "VALUES ('PMC1', 'europepmc', ?, datetime('now'))", (db.to_json({"abstract": ABSTRACT}),),
    )
    add_figure(conn, "PMC1:fig1", "PMC1", caption="Erythematous papules in the diaper area.")
    add_panel(conn, tmp_path, "p1", "PMC1:fig1", "PMC1", "contact_dermatitis",
              findings=("dd_erythema",), sha256="sha-p1")
    conn.commit()
    return conn


def _args(**overrides):
    base = dict(disease="all", pmcids=None, limit=None, dry_run=False, budget_usd=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _curation(tmp_path):
    conn = make_db(tmp_path)
    try:
        return {r["panel_id"]: r["reason"] for r in conn.execute("SELECT * FROM panel_curation")}
    finally:
        conn.close()


def test_mimics_and_uncertain_panels_are_excluded_until_a_topic_verdict(tmp_path, monkeypatch):
    _seed(tmp_path).close()
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    clients = []

    def fake_client(**kwargs):
        clients.append(FakeClient(**kwargs))
        return clients[-1]

    monkeypatch.setattr(llm, "LLMClient", fake_client)

    assert mimic_audit.run(_args()) == 0
    sent = clients[-1].seen[0]
    assert sent["topic"] == "Allergic and Irritant Contact Dermatitis" and sent["article_title"] == TITLE
    assert sent["abstract"] == ABSTRACT and "diaper area" in sent["figure_caption"]
    assert _curation(tmp_path) == {
        "p1": "mimic_audit: other_disease: Langerhans cell histiocytosis"}

    FakeClient.verdict = "uncertain"
    try:
        assert mimic_audit.run(_args()) == 0
        assert _curation(tmp_path)["p1"].startswith("mimic_audit: uncertain")
        FakeClient.verdict = "topic"
        assert mimic_audit.run(_args()) == 0
        assert _curation(tmp_path) == {}
    finally:
        FakeClient.verdict = "other_disease"


def test_other_exclusions_are_never_overwritten_or_lifted(tmp_path, monkeypatch):
    conn = _seed(tmp_path)
    conn.execute(
        "INSERT INTO panel_curation (panel_id, image_sha256, decision, reason, policy_version) "
        "VALUES ('p1', 'sha-p1', 'exclude', 'treatment_related', 'p5.v5')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(llm, "LLMClient", FakeClient)

    assert mimic_audit.run(_args()) == 0
    assert _curation(tmp_path) == {"p1": "treatment_related"}
    FakeClient.verdict = "topic"
    try:
        assert mimic_audit.run(_args()) == 0
        assert _curation(tmp_path) == {"p1": "treatment_related"}
    finally:
        FakeClient.verdict = "other_disease"


def test_exclusion_reason_names_the_verdict_and_diagnosis():
    reason = mimic_audit.exclusion_reason
    assert reason({"verdict": "topic", "actual_diagnosis": "Diaper dermatitis"}) is None
    assert reason({"verdict": "uncertain", "actual_diagnosis": ""}) == "mimic_audit: uncertain"
    assert reason({"verdict": "other_disease", "actual_diagnosis": "Becker  nevus"}) == (
        "mimic_audit: other_disease: Becker nevus")
