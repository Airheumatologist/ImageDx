"""Describe stage (P5): standalone display captions, offline with a fake LLM."""

import argparse
import json
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from src.visual_pilot import describe, llm
from src.visual_pilot.viewer.app import create_app
from balanced_fixtures import (
    add_article,
    add_disease,
    add_figure,
    add_finding,
    add_panel,
    make_db,
)

CAPTION = (
    "Gouty tophi surrounding the flexor tendon.19 A, Clinical appearance of a "
    "middle finger with chronic tophi. B, The tophus distends the A2 pulley."
)


class FakeClient:
    def __init__(self, db_conn=None, budget_usd=None):
        self.spent_usd = 0.0
        self.db_lock = threading.Lock()
        self.seen = []
        self.treatment = False

    def iter_many(self, requests):
        for i, req in enumerate(requests):
            self.seen.append(json.loads(req["user_content"]))
            yield llm.BatchResult(
                index=i,
                parsed={
                    "title": "  Chronic tophi of the middle finger ",
                    "description": "Chronic gouty tophi   impede flexor tendon excursion.",
                    "section": "musculoskeletal",
                    "subsection": "invented",
                    "treatment_related": self.treatment,
                },
                meta={"cached": False},
            )


def _seed(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "gout", "Gout")
    add_finding(conn, "tophus", ("gout",), label="Tophus")
    add_article(conn, "PMC1", title="Orthopaedic management of gout: a review")
    add_figure(conn, "PMC1:fig1", "PMC1", caption=CAPTION)
    add_panel(conn, tmp_path, "p1", "PMC1:fig1", "PMC1", "gout",
              findings=("tophus",), sha256="sha-p1")
    conn.commit()
    return conn


def _args(**overrides):
    base = dict(disease="all", pmcids=None, limit=None, dry_run=False,
                budget_usd=None, force=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_describe_writes_clean_caption_once(tmp_path, monkeypatch):
    conn = _seed(tmp_path)
    conn.close()
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    clients = []

    def fake_client(**kwargs):
        clients.append(FakeClient(**kwargs))
        return clients[-1]

    monkeypatch.setattr(llm, "LLMClient", fake_client)

    assert describe.run(_args()) == 0
    sent = clients[0].seen[0]
    assert sent["figure_caption"] == CAPTION
    assert sent["disease"] == "Gout"
    assert sent["findings"] == [{"finding_key": "tophus", "label": "Tophus"}]
    assert "musculoskeletal" in {section["key"] for section in sent["sections"]}

    conn = make_db(tmp_path)
    row = conn.execute(
        "SELECT display_title, display_description, display_section, display_subsection FROM panels"
    ).fetchone()
    assert row["display_title"] == "Chronic tophi of the middle finger"
    assert row["display_description"] == "Chronic gouty tophi impede flexor tendon excursion."
    assert row["display_section"] == "musculoskeletal"
    assert row["display_subsection"] is None  # not a listed subsection
    conn.close()

    # Already described panels are skipped unless --force.
    assert describe.run(_args()) == 0
    assert len(clients) == 1


def test_viewer_prefers_display_description(tmp_path):
    conn = _seed(tmp_path)
    conn.execute(
        "UPDATE panels SET display_title = 'Chronic tophi', "
        "display_description = 'Chronic gouty tophi impede flexor tendon excursion.'"
    )
    conn.commit()
    conn.close()
    client = TestClient(create_app(str(Path(tmp_path))))
    panel = client.get("/api/diseases/gout/panels").json()["panels"][0]
    assert panel["context"] == "Chronic gouty tophi impede flexor tendon excursion."
    assert panel["figure_caption"] == CAPTION


def test_section_choice_keeps_only_listed_options():
    sections = [
        {"key": "skin", "subsections": ["ACLE", "SCLE"]},
        {"key": "dm_skin", "subsections": ["finding"]},
        {"key": "imaging", "subsections": None},
    ]
    panel = {"findings_json": '[{"finding_key": "gottron_papules"}]'}
    choose = describe._section_choice
    assert choose({"section": "skin", "subsection": "SCLE"}, panel, sections) == ("skin", "SCLE")
    assert choose({"section": "skin", "subsection": "DLE"}, panel, sections) == ("skin", None)
    assert choose({"section": "dm_skin", "subsection": "gottron_papules"}, panel, sections) == ("dm_skin", "gottron_papules")
    assert choose({"section": "dm_skin", "subsection": "heliotrope_rash"}, panel, sections) == ("dm_skin", None)
    assert choose({"section": "imaging", "subsection": "x"}, panel, sections) == ("imaging", None)
    assert choose({"section": "nope", "subsection": None}, panel, sections) == (None, None)


def test_route_panel_prefers_valid_llm_section():
    from src.visual_pilot.viewer.app import route_panel

    base = {"modality": "histology_he", "findings": [], "figure_caption": "Skin biopsy"}
    rule = dict(base)
    route_panel(rule, "gout", {})
    assert rule["tab"] == "histology"
    chosen = dict(base, display_section="imaging")
    route_panel(chosen, "gout", {})
    assert chosen["tab"] == "imaging"
    bogus = dict(base, display_section="not_a_tab")
    route_panel(bogus, "gout", {})
    assert bogus["tab"] == "histology"
    lupus = dict(base, modality="clinical_photo", display_section="skin", display_subsection="DLE")
    route_panel(lupus, "sle", {})
    assert (lupus["tab"], lupus["clinical_group"]) == ("skin", "DLE")
    stage = dict(base, modality="radiograph", stage="nr_axspa",
                 display_section="si_radiograph", display_subsection="advanced")
    route_panel(stage, "as", {})
    assert (stage["tab"], stage["stage"]) == ("si_radiograph", "advanced")


def test_treatment_images_get_a_reversible_exclusion(tmp_path, monkeypatch):
    conn = _seed(tmp_path)
    conn.close()
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    flag = {"treatment": True}

    def fake_client(**kwargs):
        client = FakeClient(**kwargs)
        client.treatment = flag["treatment"]
        return client

    monkeypatch.setattr(llm, "LLMClient", fake_client)

    def published():
        c = make_db(tmp_path)
        try:
            return [r["panel_id"] for r in c.execute("SELECT panel_id FROM published_panels")]
        finally:
            c.close()

    assert describe.run(_args()) == 0
    assert published() == []
    flag["treatment"] = False
    assert describe.run(_args(force=True)) == 0
    assert published() == ["p1"]
