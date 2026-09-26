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
