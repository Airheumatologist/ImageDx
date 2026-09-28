"""W0 tests for src/visual_pilot/parity.py — fully local, no network."""

import argparse
from pathlib import Path

import pytest

from src.visual_pilot import config, db, parity, diseases


def _make_source_db(path: Path) -> Path:
    """A minimal 'main DB' fixture with the real schema and a few rows."""
    conn = db.init_db(db.connect(path))
    diseases.seed(conn)
    conn.execute(
        "INSERT INTO articles (pmcid, title, license_code, status, "
        " primary_disease_keys_json, relevance_decision, relevance_reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("PMC1", "SLE review", "cc-by", "parsed", '["sle"]', "relevant", "p1"),
    )
    conn.execute(
        "INSERT INTO articles (pmcid, title, license_code, status, "
        " primary_disease_keys_json) VALUES (?, ?, ?, ?, ?)",
        ("PMC2", "DM review", "cc-by-nd", "parsed", '["dm"]'),
    )
    conn.execute(
        "INSERT INTO articles (pmcid, title, license_code, status, "
        " primary_disease_keys_json) VALUES (?, ?, ?, ?, ?)",
        ("PMC_OTHER", "off-parity", "cc-by", "parsed", '["as"]'),
    )
    # A downstream row that must NOT be copied.
    conn.execute(
        "INSERT INTO figures (figure_id, pmcid, status) VALUES ('PMC1:f1', 'PMC1', 'stored')"
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def source_db(tmp_path):
    return _make_source_db(tmp_path / "main" / config.DB_FILENAME)


@pytest.fixture()
def parity_set(monkeypatch):
    """Point parity.parity_pmcids() at a two-article fixture set."""
    monkeypatch.setattr(
        parity, "parity_pmcids", lambda: ["PMC1", "PMC2"]
    )
    return ["PMC1", "PMC2"]


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def test_prepare_copies_and_resets(tmp_path, source_db, parity_set):
    out = tmp_path / "scratch"
    stats = parity.prepare(out, source_db)
    assert stats["articles"] == 2
    assert stats["missing_pmcids"] == []
    assert stats["figures"] == 0 and stats["panels"] == 0
    assert stats["disease_findings"] == 0 and stats["llm_calls"] == 0

    conn = db.connect(out / config.DB_FILENAME)
    rows = {
        r["pmcid"]: dict(r)
        for r in conn.execute("SELECT pmcid, status, license_code, "
                              "relevance_decision FROM articles")
    }
    assert set(rows) == {"PMC1", "PMC2"}  # PMC_OTHER not copied
    assert rows["PMC1"]["status"] == "relevant"  # reset from 'parsed'
    assert rows["PMC1"]["license_code"] == "cc-by"  # license kept
    assert rows["PMC1"]["relevance_decision"] == "relevant"  # relevance kept
    assert rows["PMC2"]["license_code"] == "cc-by-nd"
    assert conn.execute("SELECT COUNT(*) n FROM diseases").fetchone()["n"] == len(diseases.load_diseases())
    assert conn.execute("SELECT COUNT(*) n FROM findings_vocab").fetchone()["n"] > 0
    conn.close()


def test_prepare_missing_pmcid_warns(tmp_path, source_db, monkeypatch, parity_set):
    out = tmp_path / "scratch"
    monkeypatch.setattr(parity, "parity_pmcids", lambda: ["PMC1", "PMC_NOPE"])
    args = argparse.Namespace(out=str(out), source=str(source_db))
    assert parity.cmd_prepare(args) == 1


# ---------------------------------------------------------------------------
# compare helpers
# ---------------------------------------------------------------------------
def _run_dir(path: Path, *, with_files: bool = True) -> Path:
    """A scratch VP_DATA_DIR with a small but complete set of pipeline rows."""
    path.mkdir(parents=True, exist_ok=True)
    conn = db.init_db(db.connect(path / config.DB_FILENAME))
    diseases.seed(conn)
    conn.execute(
        "INSERT INTO articles (pmcid, title, license_code, status, "
        " primary_disease_keys_json) VALUES ('PMC1', 't', 'cc-by', 'parsed', '[\"sle\"]')"
    )
    conn.execute(
        "INSERT INTO figures (figure_id, pmcid, status, sha256, image_url, "
        " image_format, effective_license, attempts, error, triage_json, vision_json) "
        "VALUES ('PMC1:f1', 'PMC1', 'stored', 'abc', "
        " 'https://x/PMC1/f1.jpg', 'jpeg', 'cc-by', 0, NULL, '{}', '{}')"
    )
    conn.execute(
        "INSERT INTO panels (panel_id, figure_id, pmcid, disease_key, sha256, "
        " image_path, thumb_path, findings_json) VALUES ('PMC1_f1_A', 'PMC1:f1', "
        " 'PMC1', 'sle', 'p1', 'panels/sle/skin/PMC1_f1_A.png', "
        " 'thumbs/PMC1_f1_A.webp', '[]')"
    )
    conn.execute(
        "INSERT INTO disease_findings (disease_key, finding_key, source, pmcid) "
        "VALUES ('sle', 'malar_rash', 'text', 'PMC1')"
    )
    conn.execute(
        "INSERT INTO llm_calls (stage, model, input_hash, response_json) "
        "VALUES ('p3', 'm', 'h1', '{}')"
    )
    conn.commit()
    conn.close()
    if with_files:
        fig = path / "figures" / "PMC1"
        fig.mkdir(parents=True)
        (fig / "f1.jpg").write_bytes(b"original-bytes")
        panel = path / "panels" / "sle" / "skin"
        panel.mkdir(parents=True)
        (panel / "PMC1_f1_A.png").write_bytes(b"panel-png")
        thumb = path / "thumbs"
        thumb.mkdir(parents=True)
        (thumb / "PMC1_f1_A.webp").write_bytes(b"thumb")
    return path


def test_compare_identical_dirs_pass(tmp_path):
    base = _run_dir(tmp_path / "base")
    cand = _run_dir(tmp_path / "cand")
    assert parity.compare(base, cand) == []
    args = argparse.Namespace(baseline=str(base), candidate=str(cand))
    assert parity.cmd_compare(args) == 0


def test_compare_figure_row_tamper_fails(tmp_path):
    base = _run_dir(tmp_path / "base")
    cand = _run_dir(tmp_path / "cand")
    conn = db.connect(cand / config.DB_FILENAME)
    conn.execute("UPDATE figures SET status='vision_rejected' WHERE figure_id='PMC1:f1'")
    conn.commit()
    conn.close()
    failures = parity.compare(base, cand)
    assert failures
    assert any("figures:" in f for f in failures)


def test_compare_file_tamper_fails(tmp_path):
    base = _run_dir(tmp_path / "base")
    cand = _run_dir(tmp_path / "cand")
    (cand / "panels" / "sle" / "skin" / "PMC1_f1_A.png").write_bytes(b"DIFFERENT")
    failures = parity.compare(base, cand)
    assert any("panels/" in f for f in failures)


def test_compare_extra_llm_call_fails(tmp_path):
    base = _run_dir(tmp_path / "base")
    cand = _run_dir(tmp_path / "cand")
    conn = db.connect(cand / config.DB_FILENAME)
    conn.execute(
        "INSERT INTO llm_calls (stage, model, input_hash, response_json) "
        "VALUES ('p3', 'm', 'h2', '{}')"
    )
    conn.commit()
    conn.close()
    failures = parity.compare(base, cand)
    assert any("llm_calls" in f for f in failures)


def test_compare_cache_miss_marker_fails(tmp_path):
    base = _run_dir(tmp_path / "base")
    cand = _run_dir(tmp_path / "cand")
    conn = db.connect(cand / config.DB_FILENAME)
    conn.execute(
        "UPDATE figures SET error='llm: cache miss: p3 abc' WHERE figure_id='PMC1:f1'"
    )
    conn.commit()
    conn.close()
    failures = parity.compare(base, cand)
    assert any("cache miss:" in f for f in failures)


def test_compare_file_for_nonaccepted_figure_fails(tmp_path):
    base = _run_dir(tmp_path / "base")
    cand = _run_dir(tmp_path / "cand")
    conn = db.connect(cand / config.DB_FILENAME)
    conn.execute(
        "INSERT INTO figures (figure_id, pmcid, status, image_url) "
        "VALUES ('PMC1:f2', 'PMC1', 'caption_rejected', 'https://x/PMC1/f2.jpg')"
    )
    conn.commit()
    conn.close()
    (cand / "figures" / "PMC1" / "f2.jpg").write_bytes(b"should-not-exist")
    failures = parity.compare(base, cand)
    assert any("never vision_accepted" in f or "candidate-only" in f for f in failures)


# ---------------------------------------------------------------------------
# run subcommand arg assembly (subprocess mocked — no live calls)
# ---------------------------------------------------------------------------
def test_run_builds_command(tmp_path, monkeypatch, parity_set):
    captured = {}

    class _Proc:
        returncode = 0

    def fake_run(cmd, env, cwd):
        captured["cmd"] = cmd
        captured["env"] = env
        return _Proc()

    monkeypatch.setattr(parity.subprocess, "run", fake_run)
    args = argparse.Namespace(
        data_dir=str(tmp_path / "d"), seed_llm_calls=None, extra=["--limit", "2"]
    )
    assert parity.cmd_run(args) == 0
    cmd = captured["cmd"]
    assert "run-all" in cmd and "--skip-select" in cmd and "--disease" in cmd
    assert "PMC1" in cmd and "PMC2" in cmd and "--limit" in cmd
    assert captured["env"]["VP_DATA_DIR"] == str(tmp_path / "d")


def test_run_seed_sets_cache_only(tmp_path, monkeypatch, parity_set, source_db):
    cand_dir = _run_dir(tmp_path / "d", with_files=False)
    captured = {}

    class _Proc:
        returncode = 0

    def fake_run(cmd, env, cwd):
        captured["env"] = env
        return _Proc()

    monkeypatch.setattr(parity.subprocess, "run", fake_run)
    args = argparse.Namespace(
        data_dir=str(cand_dir), seed_llm_calls=str(source_db), extra=[]
    )
    assert parity.cmd_run(args) == 0
    assert captured["env"]["VP_LLM_CACHE_ONLY"] == "1"


def test_run_pmcid_override_skips_default(tmp_path, monkeypatch, parity_set):
    captured = {}

    class _Proc:
        returncode = 0

    def fake_run(cmd, env, cwd):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(parity.subprocess, "run", fake_run)
    args = argparse.Namespace(
        data_dir=str(tmp_path / "d"), seed_llm_calls=None,
        extra=["--pmcids", "PMCX"],
    )
    parity.cmd_run(args)
    cmd = captured["cmd"]
    assert cmd.count("--pmcids") == 1
    assert "PMCX" in cmd and "PMC1" not in cmd
