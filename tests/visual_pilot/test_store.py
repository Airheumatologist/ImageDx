"""W7 tests: parallel store — C5 originals handoff, C6 attribution, ordered apply."""

import hashlib
import io
import threading
import time
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from src.visual_pilot import config, db, diseases, jats, originals, pmc, store


@pytest.fixture(autouse=True)
def _clean_originals():
    originals.clear()
    yield
    originals.clear()


FIXTURE_XML = (
    Path(__file__).resolve().parents[1] / "fixtures" / "visual_pilot_sample.jats.xml"
).read_text()


def _png_bytes(size=(640, 480), color=(120, 60, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _args(**kw):
    base = dict(
        disease="all", limit=None, dry_run=False, budget_usd=None, pmcids=None,
        cap=None, accept_cap=False, force=False, refresh=False,
    )
    base.update(kw)
    return Namespace(**base)


def _article(conn, pmcid="PMC1", persisted=True):
    diseases.seed(conn)
    conn.execute(
        "INSERT OR REPLACE INTO articles (pmcid, title, journal, year, doi, status, "
        "license_code, license_url, study_region, primary_disease_keys_json, "
        "authors_json, author_count, journal_name, s3_prefix, media_files_json) "
        "VALUES (?, 'Dermatomyositis clinical review', 'J', 2024, '10.1/x', 'parsed', 'cc-by', "
        "'https://creativecommons.org/licenses/by/4.0/', 'Spain (article metadata)', "
        "'[\"dm\"]', ?, ?, ?, ?, ?)",
        (
            pmcid,
            '["Doe", "Roe"]' if persisted else None,
            2 if persisted else None,
            "Journal of Test Rheumatology" if persisted else None,
            f"{pmcid}.1" if persisted else None,
            '["fig1.jpg"]' if persisted else None,
        ),
    )
    conn.commit()
    return conn.execute("SELECT * FROM articles WHERE pmcid=?", (pmcid,)).fetchone()


def _figure(conn, fid="PMC1:F1", pmcid="PMC1", url="https://s3/x/f1.png",
            vision=None, license_code="cc-by", sha256=None,
            status="vision_accepted"):
    conn.execute(
        "INSERT OR REPLACE INTO figures (figure_id, pmcid, label, caption, status, "
        "image_url, image_format, effective_license, vision_json, sha256) "
        "VALUES (?, ?, 'Figure 1', 'Clinical image of a patient with dermatomyositis.', ?, ?, 'png', ?, ?, ?)",
        (fid, pmcid, status, url, license_code,
         db.to_json(vision) if vision else None, sha256),
    )
    conn.commit()
    return dict(conn.execute("SELECT * FROM figures WHERE figure_id=?", (fid,)).fetchone())


def _panel(label, bbox, disease="dm", findings=None, proposed=None):
    return {
        "panel_label": label, "bbox": bbox, "include": True,
        "exclusion_reason": None, "disease_key": disease, "subtype": "classic",
        "modality": "clinical_photo", "body_site": "hands",
        "findings": findings or [{"finding_key": "gottron_papules", "evidence": "cap"}],
        "proposed_findings": proposed or [], "typicality": "classic",
        "stage": None, "age_group": "adult", "skin_tone": "light",
        "stated_ethnicity": None, "stated_ethnicity_quote": None,
        "annotations_present": False, "confidence": 0.9, "rationale": "r",
    }


def _vision(panels, fid="PMC1:F1"):
    return {
        "figure_id": fid, "figure_is_compound": len(panels) > 1,
        "panels": panels,
    }


def _mock_bundle(monkeypatch, calls=None):
    def _bundle(pmcid, **kw):
        if calls is not None:
            calls.append((pmcid, kw))
        return pmc.ArticleBundle(
            pmcid=pmcid, xml_text=FIXTURE_XML,
            resolver=lambda h: pmc.ImageRef(url=None, needs_bytes=True),
        )

    monkeypatch.setattr(pmc, "get_article_bundle", _bundle)


def _image_map(blobs):
    """fetch_image_bytes mock keyed on the URL's basename."""
    def _fetch(ref):
        name = ref.url.rsplit("/", 1)[-1]
        if name not in blobs:
            raise pmc.PmcError(f"boom {name}")
        return blobs[name]
    return _fetch


def _db_state(conn):
    """Panels/vocab/figures state for parity comparison (timestamps excluded)."""
    panels = [
        {k: v for k, v in dict(r).items() if k not in ("created_at", "updated_at")}
        for r in conn.execute("SELECT * FROM panels ORDER BY panel_id")
    ]
    vocab = [
        dict(r) for r in conn.execute("SELECT * FROM findings_vocab ORDER BY finding_key")
    ]
    figures = {
        r["figure_id"]: (r["status"], r["error"])
        for r in conn.execute("SELECT figure_id, status, error FROM figures")
    }
    return panels, vocab, figures


def _file_tree(data_dir):
    return {
        str(p.relative_to(data_dir)): p.read_bytes()
        for p in data_dir.rglob("*")
        if p.is_file() and not p.name.startswith("visual_pilot.sqlite")
    }


# ---------------------------------------------------------------------------
# C5: originals handoff + sha256 verification
# ---------------------------------------------------------------------------
def test_store_run_uses_originals_handoff(conn, vp_data_dir, monkeypatch):
    _article(conn)
    blob = _png_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    _figure(conn, vision=_vision([_panel("A", [0, 0, 1, 1])]), sha256=sha)
    originals.put("PMC1:F1", sha, blob)
    monkeypatch.setattr(
        pmc, "fetch_image_bytes",
        lambda ref: (_ for _ in ()).throw(AssertionError("must not refetch")),
    )
    assert store.run(_args()) == 0
    assert conn.execute(
        "SELECT status FROM figures WHERE figure_id='PMC1:F1'"
    ).fetchone()["status"] == "stored"
    assert conn.execute("SELECT COUNT(*) n FROM panels").fetchone()["n"] == 1
    assert (vp_data_dir / "figures" / "PMC1" / "f1.png").exists()
    assert originals._stats() == (0, 0)


def test_store_run_fetches_on_originals_miss(conn, vp_data_dir, monkeypatch):
    _article(conn)
    blob = _png_bytes(color=(1, 2, 3))
    sha = hashlib.sha256(blob).hexdigest()
    _figure(conn, vision=_vision([_panel("A", [0, 0, 1, 1])]), sha256=sha)
    calls = []
    monkeypatch.setattr(
        pmc, "fetch_image_bytes", lambda ref: (calls.append(ref.url), blob)[1]
    )
    assert store.run(_args()) == 0
    assert calls == ["https://s3/x/f1.png"]
    assert conn.execute(
        "SELECT status FROM figures WHERE figure_id='PMC1:F1'"
    ).fetchone()["status"] == "stored"


def test_store_sha_mismatch_leaves_vision_accepted(conn, vp_data_dir, monkeypatch):
    _article(conn)
    _figure(
        conn,
        vision=_vision([_panel("A", [0, 0, 1, 1])]),
        sha256="0" * 64,  # stored sha does not match the fetched bytes
    )
    monkeypatch.setattr(pmc, "fetch_image_bytes", _image_map({"f1.png": _png_bytes()}))
    assert store.run(_args()) == 0
    row = conn.execute(
        "SELECT status, error FROM figures WHERE figure_id='PMC1:F1'"
    ).fetchone()
    assert row["status"] == "vision_accepted"
    assert "sha256 mismatch" in row["error"]
    # Nothing stored: no panel rows, no image files.
    assert conn.execute("SELECT COUNT(*) n FROM panels").fetchone()["n"] == 0
    assert _file_tree(vp_data_dir) == {}
    assert originals._stats() == (0, 0)


def test_store_originals_mismatch_falls_back_to_fetch(conn, vp_data_dir, monkeypatch):
    """A stale originals entry is consumed by take(); the refetched bytes are
    used when they match figures.sha256."""
    _article(conn)
    blob = _png_bytes(color=(9, 9, 9))
    sha = hashlib.sha256(blob).hexdigest()
    _figure(conn, vision=_vision([_panel("A", [0, 0, 1, 1])]), sha256=sha)
    # Entry holds different bytes than the recorded sha -> take() drops it.
    originals.put("PMC1:F1", "stale", _png_bytes(color=(7, 7, 7)))
    monkeypatch.setattr(pmc, "fetch_image_bytes", _image_map({"f1.png": blob}))
    assert store.run(_args()) == 0
    assert conn.execute(
        "SELECT status FROM figures WHERE figure_id='PMC1:F1'"
    ).fetchone()["status"] == "stored"
    assert originals._stats() == (0, 0)


def test_store_fetch_error_keeps_vision_accepted(conn, vp_data_dir, monkeypatch):
    _article(conn)
    _figure(conn, vision=_vision([_panel("A", [0, 0, 1, 1])]))
    monkeypatch.setattr(
        pmc, "fetch_image_bytes",
        lambda ref: (_ for _ in ()).throw(pmc.PmcError("503")),
    )
    assert store.run(_args()) == 0
    row = conn.execute(
        "SELECT status, error FROM figures WHERE figure_id='PMC1:F1'"
    ).fetchone()
    assert row["status"] == "vision_accepted"
    assert row["error"].startswith("store:")
    assert _file_tree(vp_data_dir) == {}


# ---------------------------------------------------------------------------
# C6: attribution from persisted fields, JATS fallback only when missing
# ---------------------------------------------------------------------------
def test_store_attribution_from_persisted_fields(conn, vp_data_dir, monkeypatch):
    _article(conn, persisted=True)
    _figure(conn, vision=_vision([_panel("A", [0, 0, 1, 1])]))
    monkeypatch.setattr(pmc, "fetch_image_bytes", _image_map({"f1.png": _png_bytes()}))
    monkeypatch.setattr(
        pmc, "get_article_bundle",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("persisted C6 fields must not trigger a JATS refetch")
        ),
    )
    assert store.run(_args()) == 0
    attrib = conn.execute("SELECT attribution_text FROM panels").fetchone()[0]
    # authors_json ["Doe","Roe"] + author_count 2 -> "Doe and Roe."
    # journal column 'J' wins over journal_name, matching the old
    # `article["journal"] or parsed.journal_name` expression.
    assert attrib.startswith("Doe and Roe. Dermatomyositis clinical review. J 2024.")


def test_store_attribution_jats_fallback(conn, vp_data_dir, monkeypatch):
    _article(conn, persisted=False)
    _figure(conn, vision=_vision([_panel("A", [0, 0, 1, 1])]))
    calls = []
    _mock_bundle(monkeypatch, calls)
    monkeypatch.setattr(pmc, "fetch_image_bytes", _image_map({"f1.png": _png_bytes()}))
    assert store.run(_args()) == 0
    assert len(calls) == 1
    attrib = conn.execute("SELECT attribution_text FROM panels").fetchone()[0]
    # Fixture XML: first author Smith of 4, journal kept from the row ('J').
    assert attrib.startswith("Smith et al. Dermatomyositis clinical review. J 2024.")


def test_store_attribution_fallback_uses_hints(conn, vp_data_dir, monkeypatch):
    """Rows with s3_prefix/media_files_json but no author metadata refetch via
    the hinted get_article_bundle path."""
    _article(conn, persisted=False)
    conn.execute(
        "UPDATE articles SET s3_prefix='PMC1.7', media_files_json='[\"fig1.jpg\"]' "
        "WHERE pmcid='PMC1'"
    )
    conn.commit()
    _figure(conn, vision=_vision([_panel("A", [0, 0, 1, 1])]))
    calls = []
    _mock_bundle(monkeypatch, calls)
    monkeypatch.setattr(pmc, "fetch_image_bytes", _image_map({"f1.png": _png_bytes()}))
    assert store.run(_args()) == 0
    assert calls == [("PMC1", {"prefix": "PMC1.7", "media_files": ["fig1.jpg"]})]


# ---------------------------------------------------------------------------
# Parallel pool: concurrency cap, deterministic order, byte-parity
# ---------------------------------------------------------------------------
def test_store_concurrency_cap(conn, vp_data_dir, monkeypatch):
    _article(conn)
    blobs = {}
    for i in range(6):
        fid = f"PMC1:F{i + 1}"
        name = f"f{i + 1}.png"
        blobs[name] = _png_bytes(color=(i * 30, i * 20, i * 10))
        _figure(
            conn, fid=fid, url=f"https://s3/x/{name}",
            vision=_vision([_panel("A", None)], fid=fid),
        )
    cap = 3
    monkeypatch.setattr(config, "VP_FETCH_CONCURRENCY", cap)
    in_flight = {"cur": 0, "peak": 0}
    lock = threading.Lock()
    fetch = _image_map(blobs)

    def _fetch(ref):
        with lock:
            in_flight["cur"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["cur"])
        try:
            time.sleep(0.05)
            return fetch(ref)
        finally:
            with lock:
                in_flight["cur"] -= 1

    monkeypatch.setattr(pmc, "fetch_image_bytes", _fetch)
    assert store.run(_args()) == 0
    assert 2 <= in_flight["peak"] <= cap
    assert conn.execute(
        "SELECT COUNT(*) n FROM figures WHERE status='stored'"
    ).fetchone()["n"] == 6


def _seed_parity_case(conn):
    """Single-image figures: one cropped panel, whole-figure dedup, ND, error."""
    diseases.seed(conn)
    blobs = {
        "multi.png": _png_bytes(color=(10, 20, 30)),
        "whole.png": _png_bytes(color=(40, 50, 60)),
        "nd.png": _png_bytes(color=(70, 80, 90)),
    }
    _article(conn, "PMC1", persisted=True)
    _article(conn, "PMC2", persisted=False)
    _figure(
        conn, fid="PMC1:F1", pmcid="PMC1", url="https://s3/x/multi.png",
        sha256=hashlib.sha256(blobs["multi.png"]).hexdigest(),
        vision=_vision(
            [_panel("A", [0.0, 0.0, 0.5, 0.5], proposed=["new_sign_x"])],
            fid="PMC1:F1",
        ),
    )
    # Two figures producing the same whole-figure panel: cross-figure dedup.
    for fid in ("PMC1:F2", "PMC1:F3"):
        _figure(
            conn, fid=fid, pmcid="PMC1", url="https://s3/x/whole.png",
            sha256=hashlib.sha256(blobs["whole.png"]).hexdigest(),
            vision=_vision([_panel("A", None)], fid=fid),
        )
    _figure(
        conn, fid="PMC2:F4", pmcid="PMC2", url="https://s3/x/nd.png",
        license_code="cc-by-nd",
        sha256=hashlib.sha256(blobs["nd.png"]).hexdigest(),
        vision=_vision([_panel("A", [0.0, 0.0, 0.3, 0.3])], fid="PMC2:F4"),
    )
    _figure(  # fetch failure -> stays vision_accepted
        conn, fid="PMC1:F5", pmcid="PMC1", url="https://s3/x/missing.png",
        vision=_vision([_panel("A", [0, 0, 1, 1])], fid="PMC1:F5"),
    )
    return blobs


def test_store_parallel_matches_sequential(conn, vp_data_dir, monkeypatch, tmp_path):
    """store.run (worker pool) produces byte-identical files and identical
    panels/vocab/figures rows vs the sequential store_figure path."""
    _mock_bundle(monkeypatch)
    blobs = _seed_parity_case(conn)

    # Reference: the sequential semantics, one transaction per figure.
    monkeypatch.setattr(pmc, "fetch_image_bytes", _image_map(blobs))
    parsed_by_pmcid = {
        "PMC1": jats.ParsedArticle(
            figures=[], body_sections=[], authors=["Doe", "Roe"], author_count=2,
            article_copyright_holder=None, journal_name="Journal of Test Rheumatology",
            publisher_name=None, corresp_country=None, first_aff_country=None,
        ),
        "PMC2": jats.parse_article(FIXTURE_XML),
    }
    rows = db.rows_with_status(conn, "figures", "vision_accepted")
    for row in rows:
        figure = dict(row)
        article = conn.execute(
            "SELECT * FROM articles WHERE pmcid=?", (figure["pmcid"],)
        ).fetchone()
        try:
            with conn:
                store.store_figure(
                    conn, figure, article, parsed_by_pmcid[figure["pmcid"]], vp_data_dir
                )
                db.set_status(conn, "figures", figure["figure_id"], "stored", error=None)
        except Exception as exc:  # noqa: BLE001 - same isolation as run()
            db.set_status(
                conn, "figures", figure["figure_id"], "vision_accepted",
                error=f"store: {exc}"[:500],
            )
            conn.commit()

    ref_state = _db_state(conn)
    ref_files = _file_tree(vp_data_dir)

    # Parallel path on a fresh, identically-seeded database + data dir.
    dir_b = tmp_path / "vp_b"
    monkeypatch.setenv("VP_DATA_DIR", str(dir_b))
    conn2 = db.init_db()
    try:
        _seed_parity_case(conn2)
        assert store.run(_args()) == 0
        new_state = _db_state(conn2)
        new_files = _file_tree(dir_b)
        dedup_rows = conn2.execute(
            "SELECT COUNT(*) n FROM panels WHERE sha256 IN "
            "(SELECT sha256 FROM panels GROUP BY sha256 HAVING COUNT(*) > 1)"
        ).fetchone()["n"]
    finally:
        conn2.close()

    assert ref_files and new_files
    assert new_files == ref_files
    assert new_state == ref_state
    # Sanity: the fixture actually exercised each path.
    statuses = {fid: s for fid, (s, _e) in new_state[2].items()}
    assert statuses["PMC1:F5"] == "vision_accepted"  # fetch error preserved
    assert statuses["PMC1:F1"] == statuses["PMC2:F4"] == "stored"
    modes = {
        r["crop_mode"]
        for r in conn.execute("SELECT DISTINCT crop_mode FROM panels")
    }
    assert modes == {"panel", "whole_figure"}
    assert dedup_rows >= 2  # F2/F3 whole-figure deduplication


def test_store_rerun_is_idempotent(conn, vp_data_dir, monkeypatch):
    _article(conn)
    blob = _png_bytes()
    _figure(
        conn, vision=_vision([_panel("A", [0, 0, 1, 1], proposed=["term_y"])]),
        sha256=hashlib.sha256(blob).hexdigest(),
    )
    monkeypatch.setattr(pmc, "fetch_image_bytes", _image_map({"f1.png": blob}))
    assert store.run(_args()) == 0
    panels1, vocab1, figures1 = _db_state(conn)
    files1 = _file_tree(vp_data_dir)

    calls = []
    monkeypatch.setattr(
        pmc, "fetch_image_bytes",
        lambda ref: (calls.append(ref.url), blob)[1],
    )
    assert store.run(_args()) == 0
    # Second run: no vision_accepted figures -> no fetches, nothing changes.
    assert calls == []
    assert _db_state(conn) == (panels1, vocab1, figures1)
    assert _file_tree(vp_data_dir) == files1
    row = conn.execute(
        "SELECT proposal_count FROM findings_vocab WHERE finding_key='term_y'"
    ).fetchone()
    assert row["proposal_count"] == 1


def test_store_dedup_across_runs(conn, vp_data_dir, monkeypatch):
    _article(conn)
    blob = _png_bytes(color=(5, 5, 5))
    monkeypatch.setattr(pmc, "fetch_image_bytes", _image_map({"f1.png": blob, "f2.png": blob}))
    _figure(conn, fid="PMC1:F1", url="https://s3/x/f1.png",
            vision=_vision([_panel("A", None)], fid="PMC1:F1"))
    assert store.run(_args()) == 0
    _figure(conn, fid="PMC1:F2", url="https://s3/x/f2.png",
            vision=_vision([_panel("A", None)], fid="PMC1:F2"))
    assert store.run(_args()) == 0
    rows = conn.execute("SELECT image_path FROM panels ORDER BY panel_id").fetchall()
    assert len(rows) == 2
    assert rows[0]["image_path"] == rows[1]["image_path"]
    assert len(list((vp_data_dir / "panels").rglob("*.png"))) == 1


def test_store_marks_compound_source_rejected_with_reason(conn, vp_data_dir, monkeypatch):
    _article(conn)
    vision = _vision([
        _panel("A", [0, 0, 0.5, 0.5]),
        _panel("B", [0.5, 0.5, 1, 1]),
    ])
    _figure(conn, vision=vision)
    _mock_bundle(monkeypatch)
    monkeypatch.setattr(pmc, "fetch_image_bytes", lambda ref: _png_bytes())

    assert store.run(_args()) == 0
    row = conn.execute(
        "SELECT status, vision_json FROM figures WHERE figure_id='PMC1:F1'"
    ).fetchone()
    assert row["status"] == "vision_rejected"
    stored = db.from_json(row["vision_json"], {})
    assert all(panel["include"] is False for panel in stored["panels"])
    assert all("collage" in panel["curation_reason"] for panel in stored["panels"])
    assert conn.execute("SELECT COUNT(*) FROM panels").fetchone()[0] == 0
    assert len(list((vp_data_dir / "panels").rglob("*.png"))) == 0
