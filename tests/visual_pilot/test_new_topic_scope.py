"""Offline regression checks for the additions-only run's extraction boundary."""
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from balanced_fixtures import add_article, add_disease, add_figure, add_panel, make_db
from src.visual_pilot import db, extract_findings as extract


def _args(**overrides):
    return SimpleNamespace(**{"disease": "all", "diseases": ["gout"], "pmcids": None,
                              "limit": None, "force": False, **overrides})


def _assertion(conn, disease, article, source="text"):
    conn.execute("INSERT INTO disease_findings (disease_key,finding_key,source,pmcid,quote) "
                 "VALUES(?, 'old', ?, ?, 'keep me')", (disease, source, article))


def test_extract_scopes_shared_articles_and_does_not_skip_for_other_topics(tmp_path):
    conn = make_db(tmp_path)
    for key in ("gout", "sle"):
        add_disease(conn, key)
    for pmcid, keys in (("PMC1", ["gout", "sle"]), ("PMC2", ["sle"]), ("PMC3", ["gout"])):
        add_article(conn, pmcid)
        conn.execute("UPDATE articles SET status='parsed',primary_disease_keys_json=? WHERE pmcid=?",
                     (json.dumps(keys), pmcid))
    _assertion(conn, "sle", "PMC1")
    _assertion(conn, "gout", "PMC3")
    todo, skipped = extract._load_articles(conn, _args())
    assert [a["pmcid"] for a in todo] == ["PMC1"]
    assert skipped == 1
    assert json.loads(todo[0]["primary_disease_keys_json"]) == ["gout"]
    assert json.loads(conn.execute("SELECT primary_disease_keys_json FROM articles WHERE pmcid='PMC1'").fetchone()[0]) == ["gout", "sle"]
    # An explicit list takes precedence over the single-topic flag.
    assert extract._load_articles(conn, _args(disease="sle"))[0][0]["pmcid"] == "PMC1"
    assert extract._load_articles(conn, _args(diseases=None, disease="sle"))[0][0]["pmcid"] == "PMC2"
    conn.close()


def test_scoped_apply_retains_unrelated_assertions_and_rejects_out_of_scope(tmp_path, monkeypatch):
    conn = make_db(tmp_path)
    for key in ("gout", "sle"):
        add_disease(conn, key)
    add_article(conn, "PMC1")
    _assertion(conn, "sle", "PMC1")
    _assertion(conn, "gout", "PMC1")
    monkeypatch.setattr(extract, "post_validate", lambda raw, *args: dict(raw))
    stats = defaultdict(int)
    extract.apply_response(conn, {"pmcid": "PMC1"},
                           {"scope_keys": ["gout"], "valid_keys": set(), "source_text": ""},
                           {"assertions": [{"disease_key": "sle", "finding_key": "wrong"},
                                           {"disease_key": "gout", "finding_key": "new", "quote": "new evidence"}]},
                           False, stats)
    rows = list(conn.execute("SELECT disease_key,finding_key,quote FROM disease_findings ORDER BY disease_key"))
    assert [tuple(row) for row in rows] == [("gout", "new", "new evidence"), ("sle", "old", "keep me")]
    assert stats["dropped_disease"] == 1
    assert stats["inserted"] == 1
    conn.close()


def test_scoped_image_rebuild_preserves_other_topics(tmp_path):
    conn = make_db(tmp_path)
    for key in ("gout", "sle"):
        add_disease(conn, key)
    add_article(conn, "PMC1")
    _assertion(conn, "gout", "PMC1", "image")
    _assertion(conn, "sle", "PMC1", "image")
    _assertion(conn, "gout", "PMC1", "text")
    add_figure(conn, "PMC1:fig1", "PMC1")
    add_panel(conn, tmp_path, "p1", "PMC1:fig1", "PMC1", "gout", findings=("tophi",))
    assert extract.rebuild_image_rows(conn, ["gout"]) == 1
    rows = list(conn.execute("SELECT disease_key,source FROM disease_findings ORDER BY disease_key"))
    assert [tuple(row) for row in rows] == [("gout", "text"), ("gout", "image"), ("sle", "image")]
    # Empty scope must not become an unscoped delete.
    assert extract.rebuild_image_rows(conn, []) == 0
    assert conn.execute("SELECT COUNT(*) FROM disease_findings").fetchone()[0] == 3
    conn.close()


def _launcher():
    path = Path(__file__).resolve().parents[2] / "scripts/run_new_topics.py"
    spec = importlib.util.spec_from_file_location("new_topic_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_dry_run_never_starts_subprocess_or_writes_status(tmp_path, monkeypatch):
    launcher = _launcher()
    monkeypatch.setattr(launcher, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("dry-run started subprocess"))
    assert launcher.main(["--dry-run"]) == 0
    assert not list(tmp_path.iterdir())


def test_launcher_stops_before_acquisition_if_vocab_build_is_incomplete(tmp_path, monkeypatch):
    launcher = _launcher()
    monkeypatch.setattr(launcher, "RUN_DIR", tmp_path)
    monkeypatch.setattr(launcher, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(launcher, "LOG", tmp_path / "run.log")
    monkeypatch.setattr(launcher, "missing_vocab", lambda keys: [keys[0]])
    calls = []
    monkeypatch.setattr(launcher, "run_stage", lambda name, *args: calls.append(name))
    with pytest.raises(RuntimeError, match="Acquisition was not started"):
        launcher.main([])
    assert calls == ["build_vocab"]
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "failed"
