"""Gate-2 offline integration: one fixture DB, every consumer agrees.

Eligibility (C1) -> gallery selection (C2/C3) -> lane sync/scheduling (C4)
-> pair funnel/report (C6) -> viewer responses must expose the same frozen
published/reserve sets — plus cap retention (all rows/files preserved) and
honest empty-pair reporting.
"""

from fastapi.testclient import TestClient

from balanced_fixtures import (
    MALAR_CAPTION,
    add_article,
    add_disease,
    add_figure,
    add_finding,
    add_panel,
    make_db,
)
from src.visual_pilot import discover, gallery, pair_reporting
from src.visual_pilot import representatives, report
from src.visual_pilot.viewer.app import create_app


def _fixture_db(tmp_path):
    """sle/malar_rash: 24 eligible images (8 articles x 3 panels) + 1
    license-ineligible panel; sle/discoid_plaque: an approved empty lane."""
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
    add_finding(conn, "discoid_plaque", ("sle",), label="Discoid plaque")
    for i in range(24):
        pmcid = f"PMC{i // 3}"
        add_article(conn, pmcid)
        add_figure(conn, f"{pmcid}:fig{i % 3}", pmcid, caption=MALAR_CAPTION)
        add_panel(
            conn, tmp_path, f"p{i}", f"{pmcid}:fig{i % 3}", pmcid, "sle",
            findings=("malar_rash",), sha256=f"sha-{i}",
        )
    # Ineligible (license) — a rejection category, never a reserve.
    add_article(conn, "PMCnc")
    add_figure(conn, "PMCnc:fig1", "PMCnc", caption=MALAR_CAPTION)
    add_panel(
        conn, tmp_path, "p_nc", "PMCnc:fig1", "PMCnc", "sle",
        findings=("malar_rash",), sha256="sha-nc", license_code="CC-BY-NC",
    )
    conn.commit()
    return conn


def test_all_surfaces_agree_on_published_reserve_and_gap_counts(tmp_path):
    conn = _fixture_db(tmp_path)

    # C2 snapshot is the single source of truth for the other surfaces.
    snapshot = gallery.coverage_snapshot(conn)
    malar = snapshot["sle"]["malar_rash"]
    discoid = snapshot["sle"]["discoid_plaque"]
    published_ids = set(malar["published_panel_ids"])
    reserve_ids = set(malar["reserve_panel_ids"])
    assert (malar["eligible_distinct"], malar["published_distinct"]) == (24, 20)
    assert malar["reserve_distinct"] == 4
    assert published_ids.isdisjoint(reserve_ids)
    assert (malar["tier"], malar["cap_remaining"]) == ("full", 0)
    assert (malar["floor_deficit"], malar["target_deficit"]) == (0, 0)
    assert (discoid["eligible_distinct"], discoid["published_distinct"]) == (0, 0)
    assert discoid["tier"] == "empty" and discoid["floor_deficit"] == 3

    # Discovery plans from the same counts: the covered pair drops out, the
    # empty pair is the only one searched.
    assert gallery.published_coverage(conn, "sle") == {
        "malar_rash": 20, "discoid_plaque": 0,
    }
    plan = discover.plan_pairs(conn, ["sle"], target=10)
    assert [(d, f["finding_key"], have) for d, f, have in plan] == [
        ("sle", "discoid_plaque", 0)
    ]

    # Primary representative comes from the selected gallery, never a reserve.
    representatives.rebuild(conn)
    mapping = representatives.mapping_for_disease(conn, "sle")
    assert mapping["malar_rash"]["panel_id"] in published_ids
    assert "discoid_plaque" not in mapping

    # Report funnel: identical counts, rejection != reserve, honest empty pair.
    funnel = pair_reporting.pair_funnel(conn)
    assert funnel["sle"]["malar_rash"]["published_distinct"] == 20
    assert funnel["sle"]["malar_rash"]["reserve_distinct"] == 4
    assert funnel["sle"]["malar_rash"]["milestone"] == "full"
    assert funnel["sle"]["malar_rash"]["rejection_categories"]["license/third-party"] == 1
    empty = funnel["sle"]["discoid_plaque"]
    assert empty["published_distinct"] == 0 and empty["blocked_reason"] is None
    assert "bounded" in empty["next_action"]

    pairs = report.pair_coverage(conn)
    assert pairs["per_disease"]["sle"]["pairs"] == 2
    assert pairs["per_disease"]["sle"]["pairs_full"] == 1
    assert pairs["per_disease"]["sle"]["zero_image_pairs"] == 1
    assert pairs["histogram"]["0"] == 1 and pairs["histogram"][">=10"] == 1
    dist = report.panel_distribution(conn)
    assert dist["by_finding"]["malar_rash"] == 20  # published, not stored rows

    # Viewer: default + finding views expose exactly the selected gallery.
    conn.close()
    with TestClient(create_app(str(tmp_path))) as client:
        default = client.get("/api/diseases/sle/panels").json()
        default_ids = {p["panel_id"] for p in default["panels"]}
        assert default_ids == published_ids
        assert default["reserves"] == {"discoid_plaque": 0, "malar_rash": 4}
        assert default["gallery_cap"] == 20
        assert "p_nc" not in default_ids

        finding = client.get("/api/diseases/sle/panels?finding=malar_rash").json()
        assert {p["panel_id"] for p in finding["panels"]} == published_ids
        empty_view = client.get("/api/diseases/sle/panels?finding=discoid_plaque").json()
        assert empty_view["panels"] == []

        # Reserves are inspectable separately and never leak into galleries.
        reserves = client.get("/api/diseases/sle/reserves?finding=malar_rash").json()
        assert reserves["count"] == 4
        assert {r["panel_id"] for r in reserves["reserves"]} == reserve_ids
        assert all(r["reserve"] and r["selection_reason"] for r in reserves["reserves"])
        assert reserves["per_finding"]["malar_rash"]["reserve_distinct"] == 4

    # Retention: every stored row survives a full gallery.
    conn = make_db(tmp_path)
    assert conn.execute("SELECT COUNT(*) AS n FROM panels").fetchone()["n"] == 25
    conn.close()
