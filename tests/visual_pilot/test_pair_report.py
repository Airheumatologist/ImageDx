"""C6 reporting tests: pair funnel fields, reserves vs rejections, JSON output.

Offline only: scratch SQLite DBs and Pillow images under tmp_path; no
network, providers, or the main database.
"""

import json
from types import SimpleNamespace

from src.visual_pilot import config, db, gallery, llm, pair_reporting, report
from balanced_fixtures import (
    MALAR_CAPTION,
    add_article,
    add_disease,
    add_figure,
    add_finding,
    add_panel,
    make_db,
)


def _seed_pairs(conn):
    """One disease with two approved findings -> two approved pairs."""
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash")
    add_finding(conn, "discoid_rash", ("sle",), label="Discoid rash")


def test_zero_article_pair_reports_honest_gap(tmp_path):
    conn = make_db(tmp_path)
    _seed_pairs(conn)
    conn.commit()
    record = pair_reporting.pair_funnel(conn)["sle"]["discoid_rash"]
    assert record["retrieved_unique_articles"] == 0
    assert record["licensed_articles"] == 0
    assert record["pair_supported_caption_figures"] == 0
    assert record["eligible_distinct"] == 0
    assert record["published_distinct"] == 0
    assert record["reserve_distinct"] == 0
    assert record["tier"] == "empty"
    assert record["milestone"] == "floor_deficit"
    assert record["blocked_reason"] is None
    assert record["pending_work"] == 0
    assert record["unresolved_legacy_candidates"] == 0
    assert record["next_action"] == (
        "Run another bounded discovery round for this pair"
    )
    assert not any(record["rejection_categories"].values())
    assert record["selection_reserves"] == {
        "distinct_groups": 0,
        "duplicate_or_diversity_rows": 0,
    }
    conn.close()


def test_blocked_lane_next_action_names_the_block(tmp_path):
    conn = make_db(tmp_path)
    _seed_pairs(conn)
    conn.execute(
        "INSERT INTO manifestation_lanes(disease_key,finding_key,status,"
        "blocked_reason) VALUES('sle','malar_rash','open','search_plan_exhausted')"
    )
    conn.commit()
    record = pair_reporting.pair_funnel(conn)["sle"]["malar_rash"]
    assert record["blocked_reason"] == "search_plan_exhausted"
    assert "search_plan_exhausted" in record["next_action"]
    # The deficit is still reported; a block is not coverage.
    assert record["published_distinct"] == 0
    assert record["tier"] == "empty"
    conn.close()


def test_ineligible_panel_is_rejection_not_reserve(tmp_path):
    conn = make_db(tmp_path)
    _seed_pairs(conn)
    add_article(conn, "PMC1")
    # Caption names no approved finding -> deterministic eligibility failure.
    add_figure(
        conn, "PMC1:fig1", "PMC1",
        caption="Discoid plaque in a 40-year-old patient with systemic lupus erythematosus",
    )
    add_panel(conn, tmp_path, "p1", "PMC1:fig1", "PMC1", "sle", sha256="sha-1")
    conn.commit()
    record = pair_reporting.pair_funnel(conn)["sle"]["malar_rash"]
    assert record["eligible_distinct"] == 0
    assert record["published_distinct"] == 0
    assert record["reserve_distinct"] == 0
    assert record["rejection_categories"]["unsupported finding"] == 1
    assert sum(record["rejection_categories"].values()) == 1
    # An eligibility failure is never a selection reserve.
    assert record["selection_reserves"]["distinct_groups"] == 0
    assert record["selection_reserves"]["duplicate_or_diversity_rows"] == 0
    conn.close()


def test_eligible_beyond_cap_is_reserve_not_rejection(tmp_path):
    conn = make_db(tmp_path)
    _seed_pairs(conn)
    add_article(conn, "PMC1")
    cap = config.VP_FINDING_GALLERY_CAP
    for i in range(cap + 1):
        figure_id = f"PMC1:fig{i}"
        add_figure(conn, figure_id, "PMC1")
        add_panel(
            conn, tmp_path, f"p{i}", figure_id, "PMC1", "sle", sha256=f"sha-{i}"
        )
    conn.commit()
    record = pair_reporting.pair_funnel(conn)["sle"]["malar_rash"]
    assert record["eligible_distinct"] == cap + 1
    assert record["published_distinct"] == cap
    assert record["reserve_distinct"] == 1
    assert record["selection_reserves"]["distinct_groups"] == 1
    assert record["selection_reserves"]["duplicate_or_diversity_rows"] == 0
    # A held-back eligible image is a reserve, never a rejection.
    assert not any(record["rejection_categories"].values())
    assert record["tier"] == "full"
    assert record["milestone"] == "full"
    assert record["next_action"] == "Gallery full; retain and inspect reserves"
    # The cap truncates nothing: every stored row and file survives.
    assert conn.execute("SELECT COUNT(*) n FROM panels").fetchone()["n"] == cap + 1
    conn.close()


def test_counts_agree_with_gallery_snapshot(tmp_path):
    conn = make_db(tmp_path)
    _seed_pairs(conn)
    add_article(conn, "PMC1")
    for i in range(4):
        figure_id = f"PMC1:fig{i}"
        add_figure(conn, figure_id, "PMC1")
        add_panel(
            conn, tmp_path, f"p{i}", figure_id, "PMC1", "sle", sha256=f"sha-{i}"
        )
    conn.commit()
    snapshot = gallery.coverage_snapshot(conn)
    funnel = pair_reporting.pair_funnel(conn, snapshot=snapshot)
    for finding_key, snap in snapshot["sle"].items():
        rec = funnel["sle"][finding_key]
        for key in (
            "eligible_distinct", "published_distinct", "reserve_distinct",
            "floor_deficit", "target_deficit", "cap_remaining", "tier",
            "blocked_reason", "published_panel_ids", "reserve_panel_ids",
        ):
            assert rec[key] == snap[key], (finding_key, key)
    summary = report.pair_coverage(conn, snapshot=snapshot)
    entry = summary["per_disease"]["sle"]
    assert entry["pairs"] == 2
    assert entry["pairs_at_floor"] == 1       # malar_rash has 4 >= floor 3
    assert entry["pairs_at_target"] == 0
    assert entry["pairs_full"] == 0
    assert entry["zero_image_pairs"] == 1     # discoid_rash
    assert entry["tier_counts"] == {"below_target": 1, "empty": 1}
    assert summary["histogram"]["4"] == 1
    assert summary["histogram"]["0"] == 1
    dist = report.panel_distribution(conn, snapshot=snapshot)
    assert dist["by_finding"]["malar_rash"] == 4
    assert dist["by_finding"]["discoid_rash"] == 0
    conn.close()


def test_pair_funnel_candidate_attempt_and_pending_fields(tmp_path):
    conn = make_db(tmp_path)
    _seed_pairs(conn)
    add_article(conn, "PMC1")
    conn.execute(
        "UPDATE articles SET primary_disease_keys_json='[\"sle\"]' "
        "WHERE pmcid='PMC1'"
    )
    conn.execute(
        "INSERT INTO manifestation_candidates(disease_key,finding_key,pmcid,"
        "status,provenance_status,provenance_disease_key) "
        "VALUES('sle','malar_rash','PMC1','pending','explicit','sle')"
    )
    # A legacy ambiguous candidate stays retained but unresolved; it counts
    # toward the pair's unresolved backlog, not its retrieval totals.
    add_article(conn, "PMC_LEGACY", license_code="CC-BY-NC")
    conn.execute(
        "INSERT INTO manifestation_candidates(disease_key,finding_key,pmcid,"
        "status,provenance_status,provenance_disease_key) "
        "VALUES('sle','malar_rash','PMC_LEGACY','pending','unresolved',NULL)"
    )
    add_figure(conn, "PMC1:fig1", "PMC1")  # default caption supports the pair
    conn.execute(
        "INSERT INTO pair_search_attempts(disease_key,finding_key,"
        "policy_version,round_no,query,query_filter_hash,filters_json,depth,"
        "status,returned_pmcids_json,new_pmcids_json,completed_at) "
        "VALUES('sle','malar_rash','balanced-pair-search.v1',1,"
        "'systemic lupus erythematosus malar rash','h1','{}',300,'completed',"
        "'[\"PMC1\"]','[\"PMC1\"]','2026-09-30T00:00:00')"
    )
    conn.commit()
    record = pair_reporting.pair_funnel(conn)["sle"]["malar_rash"]
    assert record["retrieved_unique_articles"] == 1
    assert record["licensed_articles"] == 1
    # Article membership alone is not support; this caption names both.
    assert record["pair_supported_caption_figures"] == 1
    assert record["pending_work"] == 1
    assert record["unresolved_legacy_candidates"] == 1
    assert record["next_action"] == (
        "Finish pending figures before the next discovery round"
    )
    assert len(record["attempted_strategies"]) == 1
    attempt = record["attempted_strategies"][0]
    assert attempt["depth"] == 300
    assert attempt["round_no"] == 1
    assert attempt["returned_unique_articles"] == 1
    assert attempt["new_unique_articles"] == 1
    conn.close()


def test_report_run_writes_pair_sections_without_llm(tmp_path, monkeypatch):
    def _no_llm(*args, **kwargs):
        raise AssertionError("report must never construct llm.LLMClient")

    monkeypatch.setattr(llm, "LLMClient", _no_llm)
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    conn = make_db(tmp_path)
    _seed_pairs(conn)
    add_article(conn, "PMC1")
    for i in range(2):
        figure_id = f"PMC1:fig{i}"
        add_figure(conn, figure_id, "PMC1")
        add_panel(
            conn, tmp_path, f"p{i}", figure_id, "PMC1", "sle", sha256=f"sha-{i}"
        )
    conn.commit()
    conn.close()

    assert report.run(SimpleNamespace()) == 0
    data = json.loads((tmp_path / "reports" / "pilot_report.json").read_text())

    funnel = data["pair_funnel"]
    assert set(funnel["sle"]) == {"discoid_rash", "malar_rash"}
    rec = funnel["sle"]["malar_rash"]
    for key in (
        "retrieved_unique_articles", "licensed_articles",
        "pair_supported_caption_figures", "eligible_distinct",
        "published_distinct", "reserve_distinct", "floor_deficit",
        "target_deficit", "cap_remaining", "tier", "milestone",
        "blocked_reason", "attempted_strategies", "last_deficit_reduction",
        "next_action", "pending_work", "unresolved_legacy_candidates",
        "rejection_categories", "selection_reserves",
    ):
        assert key in rec, key
    assert rec["published_distinct"] == 2
    assert rec["milestone"] == "floor_deficit"  # 2 < floor 3
    zero = funnel["sle"]["discoid_rash"]
    assert zero["published_distinct"] == 0
    assert zero["tier"] == "empty"
    assert zero["retrieved_unique_articles"] == 0

    coverage = data["pair_coverage"]
    assert (coverage["floor"], coverage["target"], coverage["cap"]) == (
        config.validate_coverage_settings()
    )
    sle = coverage["per_disease"]["sle"]
    assert sle["pairs"] == 2
    assert sle["pairs_at_floor"] == 0
    assert sle["pairs_at_target"] == 0
    assert sle["pairs_full"] == 0
    assert sle["tier_counts"] == {"below_floor": 1, "empty": 1}
    assert set(coverage["histogram"]) == {
        "0", "1", "2", "3", "4", "5-6", "7-9", ">=10"
    }
    assert data["panel_distribution"]["by_finding"]["malar_rash"] == 2

    # Existing report outputs stay intact.
    for key in (
        "funnel", "costs", "access_failures", "representatives",
        "zero_image_findings", "panel_distribution", "pair_coverage",
    ):
        assert key in data
    assert (tmp_path / "reports" / "spot_accepted.html").exists()

    md = (tmp_path / "reports" / "pilot_report.md").read_text()
    assert "## Per-pair funnel" in md
    assert "malar_rash" in md and "discoid_rash" in md
    assert "Rejection categories" in md
    assert "Selection reserves" in md
