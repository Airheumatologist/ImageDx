"""Deficit-first global scheduling and run-all orchestration (offline)."""

from src.visual_pilot import db, diseases, manifestation_queue, pair_terms
from src.visual_pilot import cli as cli_mod
from src.visual_pilot import judge, select_articles
from balanced_fixtures import add_disease, add_finding, make_db


def _record(published, *, tier=None, blocked=None, seq=0):
    from src.visual_pilot import coverage

    return {
        "published_distinct": published,
        "tier": tier or coverage.coverage_tier(published),
        "blocked_reason": blocked,
        "last_served_sequence": seq,
    }


def _snapshot(pairs):
    snap = {}
    for (disease_key, finding_key), record in pairs.items():
        snap.setdefault(disease_key, {})[finding_key] = record
    return snap


def _article(conn, pmcid, status="relevant"):
    conn.execute(
        "INSERT INTO articles(pmcid,status) VALUES(?,?)", (pmcid, status)
    )


def _explicit_candidate(conn, disease, finding, pmcid):
    conn.execute(
        "INSERT INTO manifestation_candidates"
        "(disease_key,finding_key,pmcid,provenance_status,provenance_disease_key) "
        "VALUES(?,?,?,'explicit',?)",
        (disease, finding, pmcid, disease),
    )


def _lane_db(tmp_path, pairs):
    """Scratch DB with synced lanes; ``pairs`` maps (disease,finding)->count."""
    conn = make_db(tmp_path)
    disease_keys = sorted({d for d, _f in pairs})
    for disease_key in disease_keys:
        add_disease(conn, disease_key, disease_key.upper())
    for (disease_key, finding_key) in pairs:
        add_finding(conn, finding_key, (disease_key,), label=finding_key)
    conn.commit()
    for disease_key in disease_keys:
        manifestation_queue.sync_candidates(conn, disease_key)
    return conn


def _actionable_order(pairs, disease_keys=None):
    return [
        (lane["disease_key"], lane["finding_key"])
        for lane in manifestation_queue.actionable_lanes(
            _snapshot(pairs), disease_keys
        )
    ]


def test_highest_priority_empty_lane_first(tmp_path):
    pairs = {
        ("d1", "f_empty"): _record(0),
        ("d1", "f_two"): _record(2),
        ("d1", "f_target"): _record(10),
        ("d1", "f_full"): _record(20),
    }
    # Only the globally-highest tier is actionable; the 20-count lane is at
    # the gallery cap (full) and is never returned.
    assert _actionable_order(pairs, ["d1"]) == [("d1", "f_empty")]
    # Once the empty lane is served/blocked, the next deficit tier leads.
    pairs[("d1", "f_empty")]["blocked_reason"] = "paused_test"
    assert _actionable_order(pairs, ["d1"]) == [("d1", "f_two")]
    pairs[("d1", "f_two")]["blocked_reason"] = "paused_test"
    assert _actionable_order(pairs, ["d1"]) == [("d1", "f_target")]
    pairs[("d1", "f_target")]["blocked_reason"] = "paused_test"
    assert _actionable_order(pairs, ["d1"]) == []


def test_blocked_empty_lane_lets_count_two_proceed(tmp_path):
    pairs = {
        ("d1", "f_empty"): _record(0, blocked="paused_runtime_limit"),
        ("d1", "f_two"): _record(2),
    }
    order = _actionable_order(pairs, ["d1"])
    assert order == [("d1", "f_two")]


def test_equal_lanes_rotate_fairly_with_persisted_sequence(tmp_path):
    conn = _lane_db(tmp_path, {("d1", "f1"): 0, ("d1", "f2"): 0})
    for pmcid in ("A", "B"):
        _article(conn, pmcid)
    _explicit_candidate(conn, "d1", "f1", "A")
    _explicit_candidate(conn, "d1", "f2", "B")
    conn.commit()
    ranked = [{"pmcid": "A"}, {"pmcid": "B"}]
    first = manifestation_queue.reserve_global_batch(
        conn, {"d1": ranked}, 1, disease_keys=["d1"]
    )
    second = manifestation_queue.reserve_global_batch(
        conn, {"d1": ranked}, 1, disease_keys=["d1"]
    )
    assert [row["pmcid"] for row in first] == ["A"]
    assert [row["pmcid"] for row in second] == ["B"]
    seqs = {
        row["finding_key"]: row["last_served_sequence"]
        for row in conn.execute(
            "SELECT finding_key,last_served_sequence FROM manifestation_lanes"
        )
    }
    assert seqs == {"f1": 1, "f2": 2}
    conn.close()


def test_cross_disease_rotation_persists_across_calls(tmp_path):
    conn = _lane_db(tmp_path, {("d1", "f1"): 0, ("d2", "f2"): 0})
    _article(conn, "A")
    _article(conn, "B")
    _explicit_candidate(conn, "d1", "f1", "A")
    _explicit_candidate(conn, "d2", "f2", "B")
    conn.commit()
    conn.close()
    picks = []
    # Two separate connections stand in for separate batch-size-1 runs; the
    # persisted sequence still rotates service across diseases.
    for i in (1, 2):
        conn = db.connect(tmp_path / "visual_pilot.sqlite")
        selected = manifestation_queue.reserve_global_batch(
            conn,
            {"d1": [{"pmcid": "A"}], "d2": [{"pmcid": "B"}]},
            1,
            disease_keys=["d1", "d2"],
        )
        picks.append(selected[0]["pmcid"])
        conn.close()
    assert picks == ["A", "B"]


def test_shared_article_consumes_one_slot_and_serves_both(tmp_path):
    conn = _lane_db(tmp_path, {("d1", "f1"): 0, ("d1", "f2"): 0})
    _article(conn, "A")
    _article(conn, "UNRELATED")  # not a lane candidate: must never be served
    _explicit_candidate(conn, "d1", "f1", "A")
    _explicit_candidate(conn, "d1", "f2", "A")
    conn.commit()
    selected = manifestation_queue.reserve_global_batch(
        conn, {"d1": [{"pmcid": "A"}, {"pmcid": "UNRELATED"}]}, 4,
        disease_keys=["d1"],
    )
    assert len(selected) == 1
    assert selected[0]["pmcid"] == "A"
    assert selected[0]["service_pairs"] == [("d1", "f1"), ("d1", "f2")]
    seqs = [
        row["last_served_sequence"]
        for row in conn.execute("SELECT last_served_sequence FROM manifestation_lanes")
    ]
    assert seqs == [1, 1]
    conn.close()


def test_spare_capacity_excludes_unrelated_covered_lane_articles(tmp_path):
    conn = _lane_db(
        tmp_path, {("d1", "f_low"): 0, ("d1", "f_covered"): 0}
    )
    _article(conn, "LOW")
    _article(conn, "HOT")
    _explicit_candidate(conn, "d1", "f_low", "LOW")
    conn.commit()
    conn.execute(
        "UPDATE manifestation_lanes SET tier='full', status='covered', "
        "blocked_reason=NULL WHERE finding_key='f_covered'"
    )
    conn.commit()
    snapshot = {"d1": {
        "f_low": _record(0),
        "f_covered": _record(20),
    }}
    selected = manifestation_queue.reserve_global_batch(
        conn,
        {"d1": [{"pmcid": "LOW"}, {"pmcid": "HOT"}]},
        10,
        disease_keys=["d1"],
        snapshot=snapshot,
    )
    assert [row["pmcid"] for row in selected] == ["LOW"]
    conn.close()


def test_as_pair_queries_exclude_psoriatic_and_hla_terms():
    disease = {"name": "Ankylosing spondylitis",
               "synonyms": ["ankylosing spondylitis", "axial spondyloarthritis",
                            "axSpA", "HLA-B27"]}
    finding = {"finding_key": "anterior_uveitis", "label": "Anterior uveitis",
               "category": "eye",
               "synonyms": ["acute anterior uveitis", "iritis",
                            "psoriatic uveitis",
                            "uveitis associated with psoriatic arthritis"]}
    queries = pair_terms.search_queries("as", finding, disease)
    assert queries
    joined = " | ".join(queries).casefold()
    assert "psoria" not in joined  # no psoriasis/psoriatic contamination
    assert "hla" not in joined
    assert "axspa" not in joined.replace("axial spondyloarthritis", "")
    assert " axial spondyloarthritis " in f" {joined} "
    # AS eye findings use the slit-lamp modality wording.
    assert "slit lamp" in joined


def test_psa_only_evidence_never_activates_as_lane(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "as", "Ankylosing spondylitis")
    add_disease(conn, "psa", "Psoriatic arthritis")
    add_finding(conn, "uveitis", ("as", "psa"), label="Uveitis")
    _article(conn, "PSA1")
    conn.execute(
        "UPDATE articles SET retrieval_evidence_json=? WHERE pmcid='PSA1'",
        (db.to_json([{
            "disease_key": "psa", "finding_key": "uveitis",
            "query": "psa uveitis", "rank": 1, "score": 0.5,
        }]),),
    )
    _article(conn, "BOTH1")
    conn.execute(
        "UPDATE articles SET retrieval_evidence_json=? WHERE pmcid='BOTH1'",
        (db.to_json([{
            "disease_key": "psa", "finding_key": "uveitis",
            "query": "psa uveitis", "rank": 1, "score": 0.5,
        }, {
            "disease_key": "as", "finding_key": "uveitis",
            "query": "as uveitis", "rank": 2, "score": 0.4,
        }]),),
    )
    conn.commit()
    manifestation_queue.sync_candidates(conn, "as")
    manifestation_queue.sync_candidates(conn, "psa")
    as_rows = {
        row["pmcid"]
        for row in conn.execute(
            "SELECT pmcid FROM manifestation_candidates WHERE disease_key='as' "
            "AND provenance_status='explicit'"
        )
    }
    psa_rows = {
        row["pmcid"]
        for row in conn.execute(
            "SELECT pmcid FROM manifestation_candidates WHERE disease_key='psa' "
            "AND provenance_status='explicit'"
        )
    }
    assert as_rows == {"BOTH1"}  # only the explicit AS evidence activates
    assert psa_rows == {"PSA1", "BOTH1"}
    conn.close()


def _stage_stub(name, calls):
    def _run(args):
        calls.append(name)
        return 0
    return _run


def _stub_stages(monkeypatch, calls):
    for stage in ("parse", "triage", "judge", "store", "extract", "report"):
        monkeypatch.setitem(cli_mod.COMMANDS, stage, _stage_stub(stage, calls))


def _run_all_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    conn = db.connect(tmp_path / "visual_pilot.sqlite")
    db.init_db(conn)
    diseases.seed(conn)
    conn.close()


def test_run_all_never_requeues_plates_and_skip_select_suppresses_search(
    tmp_path, monkeypatch
):
    _run_all_env(tmp_path, monkeypatch)
    calls = []
    _stub_stages(monkeypatch, calls)
    monkeypatch.setitem(
        cli_mod.COMMANDS, "select",
        lambda args: (_ for _ in ()).throw(AssertionError("select stage ran")),
    )
    def _requeue(*args, **kwargs):
        raise AssertionError("run-all must never requeue retained plates")
    monkeypatch.setattr(judge, "requeue_plates", _requeue)
    def _replenish(*args, **kwargs):
        raise AssertionError("--skip-select must not replenish")
    monkeypatch.setattr(select_articles, "replenish_pair", _replenish)

    rc = cli_mod.main(["run-all", "--disease", "sle", "--skip-select"])
    assert rc == 0
    # Downstream drain + reporting ran; select/parse never did.
    assert "select" not in calls
    assert "parse" not in calls
    assert calls[0] == "triage"
    assert "extract" in calls and "report" in calls


def test_run_all_replenishes_once_per_actionable_lane(tmp_path, monkeypatch):
    _run_all_env(tmp_path, monkeypatch)
    conn = db.connect(tmp_path / "visual_pilot.sqlite")
    add_disease(conn, "vpfake", "Fake disease")
    for i in range(10):
        add_finding(conn, f"vp_f{i}", ("vpfake",), label=f"Finding {i}")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        diseases, "disease_keys_from_catalog",
        lambda: (*diseases.DISEASE_KEYS, "vpfake"),
    )
    calls = []
    _stub_stages(monkeypatch, calls)
    replenish_calls = []
    def _replenish(conn, disease_key, finding_key, **kwargs):
        replenish_calls.append((disease_key, finding_key))
        return {"status": "pending_work", "reason": "relevant_pending",
                "new_candidates": 0, "pending_candidates": 1,
                "queries_attempted": 0}
    monkeypatch.setattr(select_articles, "replenish_pair", _replenish)

    rc = cli_mod.main(["run-all", "--disease", "vpfake"])
    assert rc == 0
    assert len(replenish_calls) == 10
    assert {key for key, _ in replenish_calls} == {"vpfake"}


def test_run_all_zero_limits_prevent_expansion_callbacks(tmp_path, monkeypatch):
    _run_all_env(tmp_path, monkeypatch)
    calls = []
    _stub_stages(monkeypatch, calls)
    def _replenish(*args, **kwargs):
        raise AssertionError("zero limits must prevent expansion callbacks")
    monkeypatch.setattr(select_articles, "replenish_pair", _replenish)

    # Zero runtime: no expansion at all, downstream reporting still runs.
    rc = cli_mod.main(
        ["run-all", "--disease", "sle", "--max-runtime-seconds", "0"]
    )
    assert rc == 0
    # Zero article cap behaves the same way (never reset to a default).
    rc = cli_mod.main(["run-all", "--disease", "sle", "--max-articles", "0"])
    assert rc == 0
    # Zero budget stops before the first resume stage — no stage runs at all.
    before = list(calls)
    rc = cli_mod.main(["run-all", "--disease", "sle", "--budget-usd", "0"])
    assert rc == 4
    assert calls == before


def test_run_all_invalid_coverage_flags_fail_validation(tmp_path, monkeypatch):
    _run_all_env(tmp_path, monkeypatch)
    from src.visual_pilot import config
    saved = (
        config.VP_FINDING_IMAGE_FLOOR,
        config.VP_FINDING_IMAGE_TARGET,
        config.VP_FINDING_GALLERY_CAP,
    )
    try:
        rc = cli_mod.main(
            ["run-all", "--disease", "sle", "--finding-image-floor", "0"]
        )
        assert rc == 2
        rc = cli_mod.main(
            ["run-all", "--disease", "sle",
             "--finding-image-target", "30"]  # over the default gallery cap
        )
        assert rc == 2
    finally:
        config.VP_FINDING_IMAGE_FLOOR, config.VP_FINDING_IMAGE_TARGET, \
            config.VP_FINDING_GALLERY_CAP = saved


def test_real_gallery_counts_feed_scheduler(tmp_path):
    """A published panel moves its lane out of the empty tier for scheduling."""
    from balanced_fixtures import MALAR_CAPTION, add_article, add_figure, add_panel

    conn = make_db(tmp_path)
    add_disease(conn, "d1", "Disease One")
    add_finding(conn, "malar_rash", ("d1",), label="Malar rash")
    add_finding(conn, "f_empty", ("d1",), label="Empty finding")
    add_article(conn, "PMC1")
    add_figure(conn, "PMC1:fig1", "PMC1", caption=MALAR_CAPTION)
    add_panel(
        conn, tmp_path, "p1", "PMC1:fig1", "PMC1", "d1",
        findings=("malar_rash",), sha256="sched-sha",
    )
    conn.commit()
    manifestation_queue.sync_candidates(conn, "d1")
    from src.visual_pilot import gallery
    snapshot = gallery.coverage_snapshot(conn)
    order = _actionable_order(
        {(d, f): rec for d, recs in snapshot.items() for f, rec in recs.items()},
        ["d1"],
    )
    # f_empty (0 published) is the highest tier and the only actionable lane.
    assert order == [("d1", "f_empty")]
    # The published panel moved malar_rash out of the empty tier — once the
    # empty lane is blocked it becomes next in line.
    conn.execute(
        "UPDATE manifestation_lanes SET blocked_reason='paused_test' "
        "WHERE finding_key='f_empty'"
    )
    conn.commit()
    snapshot = gallery.coverage_snapshot(conn)
    order = _actionable_order(
        {(d, f): rec for d, recs in snapshot.items() for f, rec in recs.items()},
        ["d1"],
    )
    assert order == [("d1", "malar_rash")]
    lane = conn.execute(
        "SELECT tier,last_published_distinct FROM manifestation_lanes "
        "WHERE finding_key='malar_rash'"
    ).fetchone()
    assert (lane["tier"], lane["last_published_distinct"]) == ("below_floor", 1)
    conn.close()


def test_run_all_clears_transient_pauses_only_in_scope(tmp_path, monkeypatch):
    """A single --disease run unpauses only that disease's lanes."""
    _run_all_env(tmp_path, monkeypatch)
    conn = db.connect(tmp_path / "visual_pilot.sqlite")
    add_disease(conn, "vpin", "In scope")
    add_disease(conn, "vpout", "Out of scope")
    add_finding(conn, "vp_fx", ("vpin", "vpout"), label="Finding X")
    for key in ("vpin", "vpout"):
        conn.execute(
            "INSERT INTO manifestation_lanes"
            "(disease_key,finding_key,status,blocked_reason) "
            "VALUES(?,?,'open','paused_retrieval_error')",
            (key, "vp_fx"),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        diseases, "disease_keys_from_catalog",
        lambda: (*diseases.DISEASE_KEYS, "vpin", "vpout"),
    )
    calls = []
    _stub_stages(monkeypatch, calls)

    def _replenish(*args, **kwargs):
        return {"status": "search_plan_exhausted", "reason": "done",
                "new_candidates": 0, "pending_candidates": 0,
                "queries_attempted": 0}
    monkeypatch.setattr(select_articles, "replenish_pair", _replenish)

    rc = cli_mod.main(["run-all", "--disease", "vpin"])
    assert rc == 0
    conn = db.connect(tmp_path / "visual_pilot.sqlite")
    rows = {
        row["disease_key"]: row["blocked_reason"]
        for row in conn.execute(
            "SELECT disease_key,blocked_reason FROM manifestation_lanes"
        )
    }
    conn.close()
    # The out-of-scope disease keeps its pause; the in-scope transient pause
    # was cleared and the lane exhausted its search plan normally.
    assert rows["vpout"] == "paused_retrieval_error"
    assert rows["vpin"] == "search_plan_exhausted"


def test_run_all_continues_to_deeper_round_after_round_complete(
    tmp_path, monkeypatch
):
    """round_complete without selectable work still tries the next depth."""
    _run_all_env(tmp_path, monkeypatch)
    conn = db.connect(tmp_path / "visual_pilot.sqlite")
    add_disease(conn, "vpdeep", "Deep disease")
    add_finding(conn, "vp_fd", ("vpdeep",), label="Deep finding")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        diseases, "disease_keys_from_catalog",
        lambda: (*diseases.DISEASE_KEYS, "vpdeep"),
    )
    calls = []
    _stub_stages(monkeypatch, calls)

    from src.visual_pilot import search_policy
    state = {"round": 0}
    monkeypatch.setattr(
        search_policy, "next_round",
        lambda conn, d, f: state["round"] + 1 if state["round"] < 2 else None,
    )
    replenish_calls = []

    def _replenish(conn, disease_key, finding_key, *, round_no, **kwargs):
        replenish_calls.append(round_no)
        state["round"] = round_no
        return {"status": "round_complete",
                "reason": f"next_round={round_no + 1}",
                "new_candidates": 0, "pending_candidates": 0,
                "queries_attempted": 1}
    monkeypatch.setattr(select_articles, "replenish_pair", _replenish)

    rc = cli_mod.main(["run-all", "--disease", "vpdeep"])
    assert rc == 0
    # An empty reservation did not stop expansion: rounds 1 and 2 both fired.
    assert replenish_calls == [1, 2]
