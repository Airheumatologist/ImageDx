"""Live-site identities must suppress paid analysis only for covered topics."""

import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from src.visual_pilot import db, judge, published, store, triage
from balanced_fixtures import make_db, add_article, add_figure


@pytest.fixture
def site(tmp_path, monkeypatch):
    root = tmp_path / "site"
    path = root / "data/sle/panels.json"
    path.parent.mkdir(parents=True)
    image = Image.new("RGB", (40, 30), "red")
    sha = store._pixel_sha(image)
    path.write_text(json.dumps({"panels": [{
        "figure_id": "PMC1:F1", "panel_id": "old-panel", "sha256": sha,
        "image": "https://images.invalid/known.jpg",
    }]}))
    monkeypatch.setattr(published.config, "VP_PUBLISHED_SITE_DIR", root)
    return image, published.site_index()


def test_figure_url_and_pixel_identity_are_topic_specific(site):
    image, index = site
    assert index.match({"figure_id": "PMC1:F1"}, ["sle"])
    assert index.match({"image_url": "https://images.invalid/known.jpg"}, ["sle"])
    assert index.match({"figure_id": "PMC1:F1"}, ["dm"]) is None
    assert index.match({"figure_id": "PMC1:F1"}, ["sle", "dm"]) is None
    assert index.match({"figure_id": "PMC1:F1"}, []) is None
    assert index.match({"figure_id": "different"}, ["sle"]) is None


def test_same_pixels_with_different_encoded_metadata_match(site):
    from PIL.PngImagePlugin import PngInfo

    image, index = site
    info = PngInfo()
    info.add_text("source", "different encoding metadata")
    data = io.BytesIO()
    image.save(data, format="PNG", pnginfo=info)
    matches = index.match_bytes({"figure_id": "new-source"}, ["sle"], data.getvalue())
    assert matches["sle"]["kind"] == "pixel_sha256"
    assert index.match_bytes({}, ["dm"], data.getvalue()) is None
    assert index.match_bytes({}, ["sle", "dm"], data.getvalue()) is None


def test_multi_topic_duplicate_requires_all_topics(site):
    _, index = site
    index.figures["dm"] = {"PMC1:F1"}
    assert set(index.match({"figure_id": "PMC1:F1"}, ["sle", "dm"])) == {"sle", "dm"}


def test_different_pixels_are_not_suppressed(site):
    _, index = site
    image = Image.new("RGB", (40, 30), "blue")
    data = io.BytesIO()
    image.save(data, format="PNG")
    assert index.match_bytes({"figure_id": "unseen"}, ["sle"], data.getvalue()) is None


def _seed(tmp_path):
    conn = make_db(tmp_path)
    add_article(conn, "PMC1", license_code="cc-by")
    conn.execute("UPDATE articles SET primary_disease_keys_json='[\"sle\"]'")
    add_figure(conn, "PMC1:F1", "PMC1", license_code="cc-by", triage={"route": "keep"})
    conn.execute("UPDATE figures SET status='caption_kept'")
    conn.commit()
    return conn


def test_judge_skips_live_identity_before_fetch_or_llm(site, tmp_path, monkeypatch):
    conn = _seed(tmp_path)
    monkeypatch.setattr(judge, "rank_figures", lambda conn, figures, articles: figures)
    args = SimpleNamespace(disease="all", pmcids=None, limit=None, dry_run=False)
    queue, _ = judge._load_queue(conn, args)
    assert queue == []
    row = conn.execute("SELECT * FROM figures").fetchone()
    assert row["status"] == "already_published"
    audit = db.from_json(row["triage_json"])
    assert audit["route"] == "keep"
    assert audit["published_duplicate"]["matches"]["sle"]["kind"] == "figure_id"
    assert row["vision_json"] is None
    assert conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 0
    conn.close()


def test_dry_run_reports_duplicate_without_changing_status(site, tmp_path, monkeypatch):
    conn = _seed(tmp_path)
    monkeypatch.setattr(judge, "rank_figures", lambda conn, figures, articles: figures)
    args = SimpleNamespace(disease="all", pmcids=None, limit=None, dry_run=True)
    queue, _ = judge._load_queue(conn, args)
    assert queue == []
    assert conn.execute("SELECT status FROM figures").fetchone()[0] == "caption_kept"
    conn.close()


def test_triage_skips_live_identity_before_text_model(site, tmp_path, monkeypatch):
    conn = _seed(tmp_path)
    conn.execute("UPDATE figures SET status='pending'")
    conn.commit()
    monkeypatch.setattr(triage.db, "init_db", lambda: conn)
    monkeypatch.setattr(triage, "_requeue_rejections", lambda *args: None)
    monkeypatch.setattr(triage.llm, "LLMClient", lambda **kwargs: pytest.fail("no model client should be needed"))
    args = SimpleNamespace(disease="all", pmcids=None, limit=None, dry_run=False)
    assert triage.run(args) == 0


def test_pixel_duplicate_is_not_submitted_to_judge(site, tmp_path, monkeypatch):
    image, index = site
    conn = _seed(tmp_path)
    conn.execute("UPDATE figures SET figure_id='PMC1:new',image_url='https://other.invalid/new.png'")
    conn.commit()
    fig = dict(conn.execute("SELECT * FROM figures").fetchone())
    article = {"primary_disease_keys_json": '["sle"]'}
    data = io.BytesIO()
    image.save(data, format="PNG")
    original = data.getvalue()
    monkeypatch.setattr(judge, "fetch_and_prepare", lambda _: (original, "image/png", original, "png", image.size))
    run = judge._JudgeRun(conn, {"PMC1": article})
    assert list(run.requests([fig])) == []
    assert run.submitted == [] and run.published_skipped == 1
    row = conn.execute("SELECT status,triage_json FROM figures").fetchone()
    assert row["status"] == "already_published"
    assert db.from_json(row["triage_json"])["published_duplicate"]["stage"] == "judge_pixels"
    conn.close()


class _NonReentrantLock:
    """Fail immediately on nested acquisition instead of hanging pytest."""

    held = False

    def __enter__(self):
        assert not self.held, "attempted to reacquire the caller's DB lock"
        self.held = True

    def __exit__(self, *args):
        self.held = False


def test_model_failure_does_not_reacquire_result_loop_lock(site, tmp_path):
    conn = _seed(tmp_path)
    figure = dict(conn.execute("SELECT * FROM figures").fetchone())
    lock = _NonReentrantLock()
    run = judge._JudgeRun(conn, {"PMC1": {}}, db_lock=lock)
    run.submitted = [figure]
    result = SimpleNamespace(index=0, error=ValueError("bad model output"))
    with lock:
        run.apply(result)
    row = conn.execute("SELECT status,attempts FROM figures").fetchone()
    assert row["status"] == "vision_error" and row["attempts"] == 1
    assert run.totals["errors"] == 1
    conn.close()


def test_fetch_failure_acquires_lock_before_recording(site, tmp_path, monkeypatch):
    conn = _seed(tmp_path)
    figure = dict(conn.execute("SELECT * FROM figures").fetchone())
    lock = _NonReentrantLock()
    run = judge._JudgeRun(conn, {"PMC1": {}}, db_lock=lock)
    original = run._fail

    def fail(figure, error):
        assert lock.held
        original(figure, error)

    def fetch(_):
        raise OSError("image unavailable")

    monkeypatch.setattr(run, "_fail", fail)
    monkeypatch.setattr(judge, "fetch_and_prepare", fetch)
    assert list(run.requests([figure])) == []
    assert run.totals["errors"] == 1
    assert conn.execute("SELECT status FROM figures").fetchone()[0] == "vision_error"
    conn.close()


def test_merged_topic_counts_use_site_plus_unique_ready_additions(site, tmp_path, monkeypatch):
    from src.visual_pilot import pair_reporting

    conn = _seed(tmp_path)
    add_figure(conn, "PMC1:new", "PMC1")
    add_figure(conn, "PMC1:unavailable", "PMC1", image_url=None)
    conn.commit()
    monkeypatch.setattr(pair_reporting, "library_rows", lambda _: [
        # Same source figure cannot add a second image to its topic.
        {"disease_key": "sle", "figure_id": "PMC1:F1", "panel_id": "crop", "sha256": "crop-sha"},
        {"disease_key": "sle", "figure_id": "PMC1:new", "panel_id": "new", "sha256": "new-sha"},
        {"disease_key": "sle", "figure_id": "PMC1:new", "panel_id": "alias", "sha256": "new-sha"},
        {"disease_key": "sle", "figure_id": "PMC1:unavailable", "panel_id": "no-url", "sha256": "no-url"},
        # A figure already live for SLE can still add coverage to DM.
        {"disease_key": "dm", "figure_id": "PMC1:F1", "panel_id": "other-topic", "sha256": "other-sha"},
    ])
    assert published.merged_topic_counts(conn, ["sle", "dm", "empty"]) == {"sle": 2, "dm": 1, "empty": 0}
    conn.close()
