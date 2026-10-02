"""C6 viewer tests: one selected gallery per pair, reserves inspection.

Offline only — scratch SQLite DBs plus real Pillow images under tmp_path,
served through ``create_app`` and the FastAPI TestClient.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from src.visual_pilot import config
from src.visual_pilot.viewer.app import create_app
from balanced_fixtures import (
    MALAR_CAPTION,
    add_article,
    add_disease,
    add_figure,
    add_finding,
    add_panel,
    make_db,
)

CHILD_CAPTION = (
    "Malar rash in a 9-year-old patient with systemic lupus erythematosus"
)
DISCOID_CAPTION = (
    "Discoid plaque in a 40-year-old patient with systemic lupus erythematosus"
)
COMBINED_CAPTION = (
    "Malar rash and discoid plaque in a 30-year-old patient with "
    "systemic lupus erythematosus"
)
CAP = config.VP_FINDING_GALLERY_CAP


def _seed_sle(conn, *, discoid=False):
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
    if discoid:
        add_finding(conn, "discoid_rash", ("sle",), label="Discoid plaque")


def _eligible_panels(conn, root, count, *, pmcid="PMC1", prefix="p"):
    """``count`` eligible malar_rash panels: distinct figures and hashes."""
    add_article(conn, pmcid)
    for i in range(count):
        figure_id = f"{pmcid}:fig{i}"
        caption = CHILD_CAPTION if i == 0 else MALAR_CAPTION
        modality = "dermoscopy" if i % 5 == 0 else "clinical_photo"
        add_figure(conn, figure_id, pmcid, caption=caption)
        add_panel(
            conn, root, f"{prefix}{i}", figure_id, pmcid, "sle",
            findings=("malar_rash",), sha256=f"sha-{prefix}{i}",
            modality=modality,
        )
    conn.commit()


def _app(tmp_path):
    return TestClient(create_app(str(Path(tmp_path))))


def _panel_ids(payload):
    return {p["panel_id"] for p in payload["panels"]}


def test_gallery_cap_and_filters_take_subsets(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    _eligible_panels(conn, tmp_path, 24)
    conn.close()
    client = _app(tmp_path)

    default = client.get("/api/diseases/sle/panels").json()
    assert default["count"] == CAP
    assert default["gallery_cap"] == CAP
    gallery_ids = _panel_ids(default)
    assert len(gallery_ids) == CAP

    # The finding-filtered view is that pair's selected gallery, capped once.
    filtered = client.get(
        "/api/diseases/sle/panels?finding=malar_rash"
    ).json()
    assert filtered["count"] == CAP
    assert _panel_ids(filtered) == gallery_ids

    # Every other filter is a strict subset of the same gallery — never a
    # separate allowance and never extra panel ids.
    dermoscopy = client.get(
        "/api/diseases/sle/panels?finding=malar_rash&modality=dermoscopy"
    ).json()
    assert 0 < dermoscopy["count"] < CAP
    assert _panel_ids(dermoscopy) <= gallery_ids
    assert all(p["modality"] == "dermoscopy" for p in dermoscopy["panels"])

    classic = client.get(
        "/api/diseases/sle/panels?typicality=classic"
    ).json()
    assert classic["count"] == CAP
    assert _panel_ids(classic) <= gallery_ids

    # Tab/pediatric views are subsets of the same selected set as well.
    tabs = client.get("/api/diseases/sle/tabs").json()
    assert "skin" in {t["key"] for t in tabs}
    assert all(p["tab"] == "skin" for p in filtered["panels"])
    pediatric = {p["panel_id"] for p in filtered["panels"] if p["pediatric"]}
    assert pediatric == {"p0"}
    assert pediatric <= gallery_ids


def test_reserves_inspection_and_default_exclusion(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    _eligible_panels(conn, tmp_path, 24)
    conn.close()
    client = _app(tmp_path)

    default = client.get("/api/diseases/sle/panels").json()
    assert default["reserves"]["malar_rash"] == 24 - CAP

    reserves = client.get(
        "/api/diseases/sle/reserves?finding=malar_rash"
    ).json()
    assert reserves["count"] == 24 - CAP
    assert reserves["gallery_cap"] == CAP
    record = reserves["per_finding"]["malar_rash"]
    assert record["eligible_distinct"] == 24
    assert record["published_distinct"] == CAP
    assert record["reserve_distinct"] == 24 - CAP
    reserve_ids = {r["panel_id"] for r in reserves["reserves"]}
    assert len(reserve_ids) == 24 - CAP
    for entry in reserves["reserves"]:
        assert entry["reserve"] is True
        assert entry["finding_key"] == "malar_rash"
        assert entry["selection_reason"]["status"] == "gallery_full"
        assert entry["image"].startswith("/media/panels/")
    # Reserves never leak into the default or filtered gallery.
    assert not reserve_ids & _panel_ids(default)
    filtered = client.get(
        "/api/diseases/sle/panels?finding=malar_rash"
    ).json()
    assert not reserve_ids & _panel_ids(filtered)
    # The union of gallery and reserves covers every eligible group; the
    # store is never truncated.
    assert reserve_ids | _panel_ids(filtered) == {f"p{i}" for i in range(24)}


def test_ineligible_panels_never_reach_gallery_or_reserves(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    _eligible_panels(conn, tmp_path, 3)
    # Hash-matched curation exclusion.
    add_figure(conn, "PMC1:figx", "PMC1")
    add_panel(
        conn, tmp_path, "px", "PMC1:figx", "PMC1", "sle",
        findings=("malar_rash",), sha256="sha-x",
    )
    conn.execute(
        "INSERT INTO panel_curation(panel_id,image_sha256,decision,reason,"
        "policy_version) VALUES('px','sha-x','exclude','audit','v1')"
    )
    # Missing image file.
    add_figure(conn, "PMC1:figm", "PMC1")
    add_panel(
        conn, tmp_path, "pm", "PMC1:figm", "PMC1", "sle",
        findings=("malar_rash",), sha256="sha-m", image=False,
    )
    conn.commit()
    conn.close()
    client = _app(tmp_path)

    default = client.get("/api/diseases/sle/panels").json()
    assert _panel_ids(default) == {"p0", "p1", "p2"}
    reserves = client.get(
        "/api/diseases/sle/reserves?finding=malar_rash"
    ).json()
    assert reserves["count"] == 0
    assert {"px", "pm"}.isdisjoint(
        {r["panel_id"] for r in reserves["reserves"]}
    )


def test_combined_plate_visible_but_earns_no_finding_credit(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn, discoid=True)
    _eligible_panels(conn, tmp_path, 3)
    add_article(conn, "PMC9")
    vision = {
        "figure_is_compound": True,
        "panels": [
            {"panel_label": "a", "disease_key": "sle",
             "modality": "clinical_photo"},
            {"panel_label": "b", "disease_key": "sle",
             "modality": "clinical_photo"},
        ],
    }
    add_figure(conn, "PMC9:fig1", "PMC9", caption=COMBINED_CAPTION,
               vision=vision)
    add_panel(
        conn, tmp_path, "plate1", "PMC9:fig1", "PMC9", "sle",
        findings=(), plate_findings=("malar_rash", "discoid_rash"),
        plate_kind="combined",
    )
    conn.commit()
    conn.close()
    client = _app(tmp_path)

    default = client.get("/api/diseases/sle/panels").json()
    # The combined plate stays reachable in the default (Combined views).
    assert "plate1" in _panel_ids(default)
    plate = next(p for p in default["panels"] if p["panel_id"] == "plate1")
    assert plate["plate_kind"] == "combined"

    # It earns no per-finding credit: the malar_rash gallery still holds
    # exactly its own selected images, and discoid_rash has none.
    filtered = client.get(
        "/api/diseases/sle/panels?finding=malar_rash"
    ).json()
    assert _panel_ids(filtered) == {"p0", "p1", "p2"}
    discoid = client.get(
        "/api/diseases/sle/panels?finding=discoid_rash"
    ).json()
    assert discoid["count"] == 0
    reserves = client.get("/api/diseases/sle/reserves").json()
    assert reserves["per_finding"]["malar_rash"]["published_distinct"] == 3
    assert reserves["per_finding"]["discoid_rash"]["published_distinct"] == 0
    assert "plate1" not in {r["panel_id"] for r in reserves["reserves"]}


def test_unapproved_finding_param_returns_empty(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    add_finding(conn, "proposed_thing", ("sle",), label="Proposed thing",
                approved=0)
    _eligible_panels(conn, tmp_path, 2)
    conn.close()
    client = _app(tmp_path)

    for key in ("proposed_thing", "not_a_finding"):
        payload = client.get(f"/api/diseases/sle/panels?finding={key}").json()
        assert payload["panels"] == []
        assert payload["count"] == 0
        reserves = client.get(
            f"/api/diseases/sle/reserves?finding={key}"
        ).json()
        assert reserves["reserves"] == []
        assert reserves["count"] == 0


def test_primary_representative_comes_from_selected_gallery(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    _eligible_panels(conn, tmp_path, 24)
    conn.close()
    client = _app(tmp_path)

    payload = client.get("/api/diseases/sle/panels").json()
    gallery_ids = _panel_ids(payload)
    primary = payload["representatives"]["malar_rash"]["panel_id"]
    assert primary in gallery_ids
    card = next(p for p in payload["panels"] if primary in p["panel_ids"])
    assert card["is_representative"] is True
    assert card["representative_findings"] == ["malar_rash"]
