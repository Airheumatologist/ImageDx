"""The run-all expansion loop is deterministic under mocked stage handlers."""

import argparse

from src.visual_pilot import cli, db, diseases


def test_run_all_resets_zero_yield_after_positive_batch_then_stops(conn, monkeypatch):
    diseases.seed(conn)
    monkeypatch.setattr(cli, "_cmd_init", lambda args: 0)
    calls = {name: 0 for name in ("select", "parse", "triage", "judge", "store", "extract", "report")}
    batches = [
        [{"pmcid": f"PMC{batch}_{i}"} for i in range(50)]
        for batch in range(3)
    ]
    selected = []

    def fake_select(args):
        calls["select"] += 1
        return 0

    def fake_parse(args):
        calls["parse"] += 1
        return 0

    def fake_stage(name):
        def run(args):
            calls[name] += 1
            if name == "store" and args.pmcids and args.pmcids[0] == "PMC0_0":
                pmcid = "PMC0_0"
                conn.execute(
                    "INSERT INTO articles (pmcid, title, primary_disease_keys_json, status) "
                    "VALUES (?, ?, ?, 'parsed')",
                    (pmcid, "Dermatomyositis review", db.to_json(["dm"])),
                )
                figure_id = f"{pmcid}:f1"
                conn.execute(
                    "INSERT INTO figures (figure_id, pmcid, status) VALUES (?, ?, 'stored')",
                    (figure_id, pmcid),
                )
                conn.execute(
                    "INSERT INTO panels (panel_id, figure_id, pmcid, disease_key, sha256, findings_json) "
                    "VALUES (?, ?, ?, 'dm', 'sha-first-image', ?)",
                    (f"{figure_id}:A", figure_id, pmcid, db.to_json([{"finding_key": "gottron_papules"}])),
                )
                conn.commit()
            return 0
        return run

    def fake_batch(conn_arg, disease_key, batch_size, *, pmcids=None, peek_captions=True):
        if not batches:
            return []
        result = batches.pop(0)
        selected.append(result)
        return result[:batch_size]

    monkeypatch.setattr("src.visual_pilot.parse.select_batch", fake_batch)
    monkeypatch.setitem(cli.COMMANDS, "select", fake_select)
    monkeypatch.setitem(cli.COMMANDS, "parse", fake_parse)
    for name in ("triage", "judge", "store", "extract", "report"):
        monkeypatch.setitem(cli.COMMANDS, name, fake_stage(name))

    args = argparse.Namespace(
        disease="dm",
        dry_run=False,
        budget_usd=None,
        limit=None,
        batch_size=50,
        max_articles=200,
        max_runtime_seconds=900,
        zero_yield_batches=2,
        pmcids=None,
    )
    assert cli._cmd_run_all(args) == 0
    # The first batch adds a distinct image. Two following empty batches run;
    # the zero-yield counter resets after the first positive batch.
    assert len(selected) == 3
    assert calls["parse"] == 3
    assert calls["triage"] == 4  # one resume drain, then one per batch
    assert calls["extract"] == calls["report"] == 1


def _args(**overrides):
    base = dict(
        disease="dm",
        dry_run=False,
        budget_usd=None,
        limit=None,
        batch_size=50,
        max_articles=200,
        max_runtime_seconds=900,
        zero_yield_batches=2,
        pmcids=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _insert_article(conn, pmcid, disease_key):
    conn.execute(
        "INSERT OR IGNORE INTO articles (pmcid, title, primary_disease_keys_json, status) "
        "VALUES (?, ?, ?, 'parsed')",
        (pmcid, f"Review {pmcid}", db.to_json([disease_key])),
    )


def _insert_figure(conn, pmcid, status, attempts=0):
    figure_id = f"{pmcid}:f1"
    conn.execute(
        "INSERT OR IGNORE INTO figures (figure_id, pmcid, status, attempts) "
        "VALUES (?, ?, ?, ?)",
        (figure_id, pmcid, status, attempts),
    )
    return figure_id


def _store_accepted_figures(conn, pmcids, disease_key):
    """Flip vision_accepted figures to stored and add a distinct panel each."""
    for pmcid in pmcids:
        figure_id = f"{pmcid}:f1"
        row = conn.execute(
            "SELECT status FROM figures WHERE figure_id=?", (figure_id,)
        ).fetchone()
        if row is None or row["status"] != "vision_accepted":
            continue
        conn.execute(
            "UPDATE figures SET status='stored' WHERE figure_id=?", (figure_id,)
        )
        conn.execute(
            "INSERT INTO panels (panel_id, figure_id, pmcid, disease_key, sha256) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"{figure_id}:A", figure_id, pmcid, disease_key, f"sha-{pmcid}"),
        )


def test_run_all_retries_vision_error_in_batch_then_stores(conn, monkeypatch, capsys):
    """A transient vision_error recovers on an in-batch judge retry."""
    diseases.seed(conn)
    monkeypatch.setattr(cli, "_cmd_init", lambda args: 0)
    batches = [[{"pmcid": "PMC_retry"}], [{"pmcid": "PMC_second"}]]
    selected = []
    judge_batch_calls = []
    store_batch_calls = []

    def fake_batch(conn_arg, disease_key, batch_size, *, pmcids=None, peek_captions=True):
        if not batches:
            return []
        result = batches.pop(0)
        selected.append(result)
        return result[:batch_size]

    def fake_stage(name):
        def run(args):
            if name == "store" and args.pmcids:
                store_batch_calls.append(list(args.pmcids))
                _store_accepted_figures(conn, args.pmcids, args.disease)
                conn.commit()
            return 0
        return run

    def fake_judge(args):
        if not args.pmcids:
            return 0  # resume drain: nothing queued
        judge_batch_calls.append(list(args.pmcids))
        for pmcid in args.pmcids:
            _insert_article(conn, pmcid, args.disease)
            figure_id = f"{pmcid}:f1"
            row = conn.execute(
                "SELECT status FROM figures WHERE figure_id=?", (figure_id,)
            ).fetchone()
            if row is None:
                status = "vision_error" if pmcid == "PMC_retry" else "vision_accepted"
                attempts = 1 if status == "vision_error" else 0
                _insert_figure(conn, pmcid, status, attempts)
            else:
                # Retry pass: the transient failure recovered.
                conn.execute(
                    "UPDATE figures SET status='vision_accepted' WHERE figure_id=?",
                    (figure_id,),
                )
        conn.commit()
        return 0

    monkeypatch.setattr("src.visual_pilot.parse.select_batch", fake_batch)
    monkeypatch.setitem(cli.COMMANDS, "select", lambda args: 0)
    monkeypatch.setitem(cli.COMMANDS, "judge", fake_judge)
    for name in ("parse", "triage", "store", "extract", "report"):
        monkeypatch.setitem(cli.COMMANDS, name, fake_stage(name))

    assert cli._cmd_run_all(_args()) == 0
    # Judge ran twice for the first batch (initial pass + one retry) and once
    # for the clean second batch; store still ran once per batch, after the
    # recovered figure could be accepted.
    assert judge_batch_calls == [["PMC_retry"], ["PMC_retry"], ["PMC_second"]]
    assert store_batch_calls == [["PMC_retry"], ["PMC_second"]]
    row = conn.execute(
        "SELECT status FROM figures WHERE figure_id='PMC_retry:f1'"
    ).fetchone()
    assert row["status"] == "stored"
    # The recovered figure was stored in-batch, so the disease did not pause
    # and expansion continued to the second batch.
    assert len(selected) == 2
    assert "pausing disease expansion" not in capsys.readouterr().out


def test_run_all_vision_error_at_max_attempts_does_not_pause(conn, monkeypatch, capsys):
    """vision_error with attempts == judge.MAX_ATTEMPTS is not 'unfinished'."""
    from src.visual_pilot import judge as judge_module

    diseases.seed(conn)
    monkeypatch.setattr(cli, "_cmd_init", lambda args: 0)
    batches = [[{"pmcid": "PMC_err"}], [{"pmcid": "PMC_ok"}]]
    selected = []
    judge_batch_calls = []

    def fake_batch(conn_arg, disease_key, batch_size, *, pmcids=None, peek_captions=True):
        if not batches:
            return []
        result = batches.pop(0)
        selected.append(result)
        return result[:batch_size]

    def fake_stage(name):
        def run(args):
            if name == "store" and args.pmcids:
                _store_accepted_figures(conn, args.pmcids, args.disease)
                conn.commit()
            return 0
        return run

    def fake_judge(args):
        if not args.pmcids:
            return 0  # resume drain: nothing queued
        judge_batch_calls.append(list(args.pmcids))
        for pmcid in args.pmcids:
            _insert_article(conn, pmcid, args.disease)
            figure_id = f"{pmcid}:f1"
            row = conn.execute(
                "SELECT status FROM figures WHERE figure_id=?", (figure_id,)
            ).fetchone()
            if pmcid == "PMC_err":
                if row is None:
                    _insert_figure(conn, pmcid, "vision_error", attempts=1)
                else:
                    # Every retry keeps failing; attempts climbs to the cap.
                    conn.execute(
                        "UPDATE figures SET attempts=attempts+1 WHERE figure_id=?",
                        (figure_id,),
                    )
            elif row is None:
                _insert_figure(conn, pmcid, "vision_accepted")
        conn.commit()
        return 0

    monkeypatch.setattr("src.visual_pilot.parse.select_batch", fake_batch)
    monkeypatch.setitem(cli.COMMANDS, "select", lambda args: 0)
    monkeypatch.setitem(cli.COMMANDS, "judge", fake_judge)
    for name in ("parse", "triage", "store", "extract", "report"):
        monkeypatch.setitem(cli.COMMANDS, name, fake_stage(name))

    assert cli._cmd_run_all(_args()) == 0
    # The failing figure got the initial pass plus MAX_ATTEMPTS - 1 retries.
    assert judge_batch_calls == [["PMC_err"]] * judge_module.MAX_ATTEMPTS + [["PMC_ok"]]
    row = conn.execute(
        "SELECT status, attempts FROM figures WHERE figure_id='PMC_err:f1'"
    ).fetchone()
    assert row["status"] == "vision_error"
    assert row["attempts"] == judge_module.MAX_ATTEMPTS
    # Exhausted retries are not unfinished: no pause, yield was still counted
    # (zero here), and expansion continued to the next batch which stored.
    assert len(selected) == 2
    out = capsys.readouterr().out
    assert "pausing disease expansion" not in out
    assert "batch yield" in out
    assert conn.execute(
        "SELECT status FROM figures WHERE figure_id='PMC_ok:f1'"
    ).fetchone()["status"] == "stored"


def test_run_all_round_robins_in_scope_diseases(conn, monkeypatch):
    """Batches interleave across diseases; exhaustion only drops that queue."""
    diseases.seed(conn)
    monkeypatch.setattr(cli, "_cmd_init", lambda args: 0)
    queues = {
        "sle": [[{"pmcid": "PMC_sle_0"}], [{"pmcid": "PMC_sle_1"}]],
        "dm": [[{"pmcid": "PMC_dm_0"}]],
        "as": [[{"pmcid": "PMC_as_0"}], [{"pmcid": "PMC_as_1"}]],
    }
    select_order = []
    stored = []

    def fake_batch(conn_arg, disease_key, batch_size, *, pmcids=None, peek_captions=True):
        select_order.append(disease_key)
        queue = queues.get(disease_key, [])
        return queue.pop(0)[:batch_size] if queue else []

    def fake_stage(name):
        def run(args):
            if name == "store" and args.pmcids:
                for pmcid in args.pmcids:
                    _insert_article(conn, pmcid, args.disease)
                    _insert_figure(conn, pmcid, "stored")
                    conn.execute(
                        "INSERT INTO panels (panel_id, figure_id, pmcid, disease_key, sha256) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (f"{pmcid}:f1:A", f"{pmcid}:f1", pmcid, args.disease, f"sha-{pmcid}"),
                    )
                    stored.append((args.disease, pmcid))
                conn.commit()
            return 0
        return run

    monkeypatch.setattr("src.visual_pilot.parse.select_batch", fake_batch)
    monkeypatch.setitem(cli.COMMANDS, "select", lambda args: 0)
    for name in ("parse", "triage", "judge", "store", "extract", "report"):
        monkeypatch.setitem(cli.COMMANDS, name, fake_stage(name))

    assert cli._cmd_run_all(_args(disease="all")) == 0
    # One batch per disease per rotation; dm's queue empties first without
    # stopping sle/as, and each disease is re-polled until exhausted.
    assert select_order == ["sle", "dm", "as", "sle", "dm", "as", "sle", "as"]
    assert stored == [
        ("sle", "PMC_sle_0"),
        ("dm", "PMC_dm_0"),
        ("as", "PMC_as_0"),
        ("sle", "PMC_sle_1"),
        ("as", "PMC_as_1"),
    ]


def test_run_all_zero_yield_stop_is_per_disease(conn, monkeypatch, capsys):
    """A disease hitting its zero-yield limit does not starve the others."""
    diseases.seed(conn)
    monkeypatch.setattr(cli, "_cmd_init", lambda args: 0)
    queues = {
        "sle": [
            [{"pmcid": "PMC_sle_a"}],
            [{"pmcid": "PMC_sle_b"}],
            [{"pmcid": "PMC_sle_c"}],
        ],
        "dm": [[{"pmcid": "PMC_dm_a"}], [{"pmcid": "PMC_dm_b"}]],
        "as": [],
    }
    select_order = []

    def fake_batch(conn_arg, disease_key, batch_size, *, pmcids=None, peek_captions=True):
        select_order.append(disease_key)
        queue = queues.get(disease_key, [])
        return queue.pop(0)[:batch_size] if queue else []

    def fake_stage(name):
        def run(args):
            # Only sle batches produce panels; dm batches yield nothing.
            if name == "store" and args.pmcids and args.disease == "sle":
                for pmcid in args.pmcids:
                    _insert_article(conn, pmcid, args.disease)
                    _insert_figure(conn, pmcid, "stored")
                    conn.execute(
                        "INSERT INTO panels (panel_id, figure_id, pmcid, disease_key, sha256) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (f"{pmcid}:f1:A", f"{pmcid}:f1", pmcid, args.disease, f"sha-{pmcid}"),
                    )
                conn.commit()
            return 0
        return run

    monkeypatch.setattr("src.visual_pilot.parse.select_batch", fake_batch)
    monkeypatch.setitem(cli.COMMANDS, "select", lambda args: 0)
    for name in ("parse", "triage", "judge", "store", "extract", "report"):
        monkeypatch.setitem(cli.COMMANDS, name, fake_stage(name))

    assert cli._cmd_run_all(_args(disease="all")) == 0
    # dm stops after two consecutive zero-yield batches; sle keeps rotating
    # until its own queue is exhausted.
    assert select_order == ["sle", "dm", "as", "sle", "dm", "sle", "sle"]
    out = capsys.readouterr().out
    assert "dm stopped after 2 consecutive zero-yield batches" in out
    assert "sle stopped" not in out
    assert conn.execute(
        "SELECT status FROM figures WHERE figure_id='PMC_sle_c:f1'"
    ).fetchone()["status"] == "stored"
