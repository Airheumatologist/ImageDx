"""C2/C3 gallery-selection tests: grouping, reserves, locks, determinism."""

import pytest

from src.visual_pilot import coverage, gallery
from balanced_fixtures import (
    add_article,
    add_disease,
    add_figure,
    add_finding,
    add_identity_review,
    add_panel,
    gallery_panel,
    identity_review,
    make_db,
)


def _documented(pid, index, **kwargs):
    sha = f"sha-{index}"
    return gallery_panel(
        pid,
        sha256=sha,
        figure_id=f"PMC_{index}:fig1",
        pmcid=f"PMC_{index}",
        identity_review=identity_review(sha, patient_group_key=f"patient-{index}"),
        **kwargs,
    )


def test_cap_20_selects_20_and_reserves_5_distinct_documented():
    panels = [_documented(f"p{i}", i) for i in range(25)]
    out = gallery.select_gallery(panels, "sle", "malar_rash", cap=20)
    assert len(out["published_panel_ids"]) == 20
    assert len(out["reserve_panel_ids"]) == 5
    assert out["eligible_distinct"] == 25
    assert out["identity_unknown_count"] == 0
    assert out["eligible_distinct"] == len(out["published_panel_ids"]) + len(
        out["reserve_panel_ids"]
    )
    # Input rows are never dropped or rewritten by selection.
    assert len(panels) == 25
    for pid in out["reserve_panel_ids"]:
        assert out["selection_reasons"][pid]["status"] == "gallery_full"
    for pid in out["published_panel_ids"]:
        assert out["selection_reasons"][pid]["status"] == "published"


def test_cap_must_be_positive():
    with pytest.raises(ValueError):
        gallery.select_gallery([], "sle", "malar_rash", cap=0)


def test_only_eligible_supported_panels_qualify():
    panels = [
        gallery_panel("ok", sha256="a"),
        gallery_panel("not_eligible", sha256="b", eligible=False),
        gallery_panel("wrong_disease", sha256="c", disease_key="dm"),
        gallery_panel("wrong_finding", sha256="d", supported=("discoid_rash",)),
        gallery_panel("no_support", sha256="e", supported=()),
    ]
    out = gallery.select_gallery(panels, "sle", "malar_rash", cap=20)
    assert out["published_panel_ids"] == ["ok"]
    assert out["eligible_distinct"] == 1
    assert set(out["selection_reasons"]) == {"ok"}


def test_identical_hashes_count_once():
    panels = [
        gallery_panel("p1", sha256="same-bytes", figure_id="PMC1:fig1"),
        gallery_panel("p2", sha256="same-bytes", figure_id="PMC2:fig9"),
    ]
    out = gallery.select_gallery(panels, "sle", "malar_rash", cap=20)
    assert len(out["published_panel_ids"]) == 1
    assert out["eligible_distinct"] == 1
    alias = next(
        pid for pid in ("p1", "p2") if pid not in out["published_panel_ids"]
    )
    reason = out["selection_reasons"][alias]
    assert reason["status"] == "duplicate"
    assert reason["representative_id"] == out["published_panel_ids"][0]


def test_same_confirmed_patient_across_papers_counts_once():
    p1 = gallery_panel(
        "paperA", pmcid="PMC_A", figure_id="PMC_A:fig1", sha256="sha-a",
        identity_review=identity_review("sha-a", patient_group_key="patient-7"),
    )
    p2 = gallery_panel(
        "paperB", pmcid="PMC_B", figure_id="PMC_B:fig3", sha256="sha-b",
        identity_review=identity_review("sha-b", patient_group_key="patient-7"),
    )
    out = gallery.select_gallery([p1, p2], "sle", "malar_rash", cap=20)
    assert len(out["published_panel_ids"]) == 1
    assert out["eligible_distinct"] == 1
    alias = next(
        pid for pid in ("paperA", "paperB") if pid not in out["published_panel_ids"]
    )
    assert out["selection_reasons"][alias]["status"] == "same_patient"


def test_reused_image_family_counts_once():
    p1 = gallery_panel(
        "orig", pmcid="PMC_A", figure_id="PMC_A:fig1", sha256="sha-1",
        identity_review=identity_review("sha-1", reuse_group_key="reuse-1"),
    )
    p2 = gallery_panel(
        "reuse", pmcid="PMC_B", figure_id="PMC_B:fig4", sha256="sha-2",
        identity_review=identity_review("sha-2", reuse_group_key="reuse-1"),
    )
    out = gallery.select_gallery([p1, p2], "sle", "malar_rash", cap=20)
    assert len(out["published_panel_ids"]) == 1
    assert out["eligible_distinct"] == 1
    alias = "orig" if "orig" not in out["published_panel_ids"] else "reuse"
    assert out["selection_reasons"][alias]["status"] == "reused_image"


def test_equal_ages_distinct_documented_patients_stay_separate():
    panels = [
        _documented("pA", 1, source_age={"age_group": "adult"}),
        _documented("pB", 2, source_age={"age_group": "adult"}),
    ]
    out = gallery.select_gallery(panels, "sle", "malar_rash", cap=20)
    assert out["eligible_distinct"] == 2
    assert len(out["published_panel_ids"]) == 2
    assert out["identity_unknown_count"] == 0


def test_same_figure_unknown_identity_is_one_source_family():
    panels = [
        gallery_panel("c1", figure_id="PMC1:fig1", sha256="sha-1"),
        gallery_panel("c2", figure_id="PMC1:fig1", sha256="sha-2"),
    ]
    out = gallery.select_gallery(panels, "sle", "malar_rash", cap=20)
    assert out["eligible_distinct"] == 1
    assert len(out["published_panel_ids"]) == 1
    alias = "c2" if out["published_panel_ids"] == ["c1"] else "c1"
    assert out["selection_reasons"][alias]["status"] == "source_family"
    assert out["identity_unknown_count"] == 1


def test_different_figures_unknown_identities_stay_distinct():
    panels = [
        gallery_panel("u1", figure_id="PMC1:fig1", sha256="sha-1"),
        gallery_panel("u2", figure_id="PMC2:fig2", sha256="sha-2"),
        gallery_panel("u3", figure_id="PMC3:fig1", sha256="sha-3"),
    ]
    out = gallery.select_gallery(panels, "sle", "malar_rash", cap=20)
    assert out["eligible_distinct"] == 3
    assert len(out["published_panel_ids"]) == 3
    # Nonempty hashes alone never verify a patient identity.
    assert out["identity_unknown_count"] == 3


def test_stale_review_hash_is_ignored():
    stale = identity_review("sha-old", patient_group_key="patient-1")
    panels = [
        gallery_panel("p1", figure_id="PMC1:fig1", sha256="sha-new",
                      identity_review=stale),
        gallery_panel("p2", figure_id="PMC1:fig1", sha256="sha-2"),
    ]
    out = gallery.select_gallery(panels, "sle", "malar_rash", cap=20)
    # The stale review cannot document a patient; the panels group only as an
    # unknown-identity source family.
    assert out["eligible_distinct"] == 1
    assert out["identity_unknown_count"] == 1


def test_valid_lock_leads_and_invalid_lock_is_ignored():
    low = gallery_panel("low", sha256="a", confidence=0.1, typicality="variant",
                        width=100, height=100)
    high = gallery_panel("high", sha256="b", confidence=0.95, typicality="classic",
                         width=800, height=800)
    locked = gallery.select_gallery(
        [high, low], "sle", "malar_rash", cap=20, locked_panel_id="low"
    )
    assert locked["published_panel_ids"][0] == "low"
    assert locked["selection_reasons"]["low"]["status"] == "locked"

    invalid = gallery.select_gallery(
        [high, low], "sle", "malar_rash", cap=20, locked_panel_id="gone"
    )
    assert invalid["published_panel_ids"][0] == "high"
    assert invalid["selection_reasons"]["high"]["status"] == "published"


def test_selection_is_deterministic_regardless_of_input_order():
    panels = [_documented(f"p{i}", i, confidence=0.9 - i * 0.01) for i in range(8)]
    forward = gallery.select_gallery(panels, "sle", "malar_rash", cap=20)
    reverse = gallery.select_gallery(panels[::-1], "sle", "malar_rash", cap=20)
    assert forward["published_panel_ids"] == reverse["published_panel_ids"]
    assert forward["reserve_panel_ids"] == reverse["reserve_panel_ids"]
    assert forward["selection_reasons"] == reverse["selection_reasons"]


def test_two_per_article_first_pass_then_soft_fill():
    # Equal scores: group order resolves by ascending representative panel_id.
    panels = [
        gallery_panel("a1", pmcid="PMC_A", figure_id="PMC_A:f1", sha256="a1"),
        gallery_panel("a2", pmcid="PMC_A", figure_id="PMC_A:f2", sha256="a2"),
        gallery_panel("a3", pmcid="PMC_A", figure_id="PMC_A:f3", sha256="a3"),
        gallery_panel("b1", pmcid="PMC_B", figure_id="PMC_B:f1", sha256="b1"),
        gallery_panel("b2", pmcid="PMC_B", figure_id="PMC_B:f2", sha256="b2"),
    ]
    out = gallery.select_gallery(panels, "sle", "malar_rash", cap=4)
    assert out["published_panel_ids"] == ["a1", "b1", "a2", "b2"]
    assert out["reserve_panel_ids"] == ["a3"]
    # The two-per-article preference is soft: a larger cap still fills.
    wider = gallery.select_gallery(panels, "sle", "malar_rash", cap=5)
    assert wider["published_panel_ids"] == ["a1", "b1", "a2", "b2", "a3"]
    assert wider["reserve_panel_ids"] == []


def test_lock_counts_toward_two_per_article():
    panels = [
        gallery_panel("a1", pmcid="PMC_A", figure_id="PMC_A:f1", sha256="a1"),
        gallery_panel("a2", pmcid="PMC_A", figure_id="PMC_A:f2", sha256="a2"),
        gallery_panel("a3", pmcid="PMC_A", figure_id="PMC_A:f3", sha256="a3"),
        gallery_panel("b1", pmcid="PMC_B", figure_id="PMC_B:f1", sha256="b1"),
    ]
    out = gallery.select_gallery(
        panels, "sle", "malar_rash", cap=3, locked_panel_id="a1"
    )
    # a1 locked (counts toward PMC_A), pass 1 gives A one more slot and B one.
    assert out["published_panel_ids"] == ["a1", "a2", "b1"]
    assert out["reserve_panel_ids"] == ["a3"]


def test_tier_boundaries_match_plan():
    expected = {
        0: "empty",
        1: "below_floor",
        2: "below_floor",
        3: "below_target",
        9: "below_target",
        10: "expanding",
        19: "expanding",
        20: "full",
    }
    for count, tier in expected.items():
        assert coverage.coverage_tier(count) == tier


def test_snapshot_reports_distinct_metrics_and_lane_fields(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
    add_article(conn, "PMC1")
    for i in range(3):
        figure_id = f"PMC1:fig{i}"
        add_figure(conn, figure_id, "PMC1")
        add_panel(
            conn, tmp_path, f"p{i}", figure_id, "PMC1", "sle", sha256=f"sha-{i}"
        )
        add_identity_review(
            conn, f"p{i}", sha256=f"sha-{i}", patient_group_key=f"patient-{i}"
        )
    conn.execute(
        "INSERT INTO manifestation_lanes(disease_key,finding_key,status,"
        "last_served_sequence,search_policy_version) "
        "VALUES('sle','malar_rash','open',7,'balanced-pair-search.v1')"
    )
    conn.commit()
    snapshot = gallery.coverage_snapshot(conn, "sle")
    record = snapshot["sle"]["malar_rash"]
    assert record["eligible_distinct"] == 3
    assert record["published_distinct"] == 3
    assert record["reserve_distinct"] == 0
    assert record["floor_deficit"] == 0
    assert record["target_deficit"] == 7
    assert record["cap_remaining"] == 17
    assert record["tier"] == "below_target"
    assert record["last_served_sequence"] == 7
    assert record["search_policy_version"] == "balanced-pair-search.v1"
    assert record["identity_unknown_count"] == 0
    assert len(record["published_panel_ids"]) == 3
    conn.close()


def test_snapshot_preserves_all_rows_and_files_for_25_images(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
    for i in range(25):
        pmcid = f"PMC{i}"
        add_article(conn, pmcid)
        add_figure(conn, f"{pmcid}:fig1", pmcid)
        add_panel(
            conn, tmp_path, f"p{i}", f"{pmcid}:fig1", pmcid, "sle",
            sha256=f"sha-{i}",
        )
        add_identity_review(
            conn, f"p{i}", sha256=f"sha-{i}", patient_group_key=f"patient-{i}"
        )
    conn.commit()
    snapshot = gallery.coverage_snapshot(conn, "sle")
    record = snapshot["sle"]["malar_rash"]
    assert record["published_distinct"] == 20
    assert record["reserve_distinct"] == 5
    assert record["eligible_distinct"] == 25
    assert record["tier"] == "full"
    rows = conn.execute("SELECT COUNT(*) n FROM panels").fetchone()["n"]
    assert rows == 25
    files = list((tmp_path / "panels").glob("*.png"))
    assert len(files) == 25
    conn.close()


def test_reviews_lead_and_atypical_case_reports_fall_to_reserves():
    # The case report has the best image score; tier still outranks score.
    panels = [
        _documented("case", 1, confidence=1.0, article_tier=2),
        _documented("atypical", 2, confidence=1.0, article_tier=3),
        _documented("review", 3, confidence=0.5, article_tier=0),
        _documented("series", 4, confidence=0.5, article_tier=1),
    ]
    out = gallery.select_gallery(panels, "sle", "malar_rash", cap=3)
    assert out["published_panel_ids"] == ["review", "series", "case"]
    assert out["reserve_panel_ids"] == ["atypical"]
