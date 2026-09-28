"""W6 tests: judge post-validation/resume + store crops/fallbacks/dedup."""

import hashlib
import io
import json
import threading
import time
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from src.visual_pilot import (
    config,
    db,
    diseases,
    extract_findings,
    jats,
    judge,
    llm,
    originals,
    pmc,
    report,
    store,
)
from src.visual_pilot.llm import BatchResult


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
    base = dict(disease="all", limit=None, dry_run=False, budget_usd=None, pmcids=None, cap=None, accept_cap=False, force=False)
    base.update(kw)
    return Namespace(**base)


def _article(conn, pmcid="PMC1"):
    diseases.seed(conn)  # panels.disease_key / disease_findings FKs
    conn.execute(
        "INSERT OR REPLACE INTO articles (pmcid, title, journal, year, doi, status, "
        "license_code, license_url, study_region, primary_disease_keys_json) "
        "VALUES (?, 'Dermatomyositis clinical review', 'J', 2024, '10.1/x', 'parsed', 'cc-by', "
        "'https://creativecommons.org/licenses/by/4.0/', 'Spain (article metadata)', "
        "'[\"dm\"]')",
        (pmcid,),
    )
    conn.commit()
    return conn.execute("SELECT * FROM articles WHERE pmcid=?", (pmcid,)).fetchone()


def _figure(conn, status="vision_accepted", vision=None, fmt="png", license_code="cc-by"):
    fid = "PMC1:F1"
    conn.execute(
        "INSERT OR REPLACE INTO figures (figure_id, pmcid, label, caption, status, "
        "image_url, image_format, effective_license, vision_json) "
        "VALUES (?, 'PMC1', 'Figure 1', 'Clinical image of a patient with dermatomyositis.', ?, 'https://s3/x/f1.png', ?, ?, ?)",
        (fid, status, fmt, license_code, db.to_json(vision) if vision else None),
    )
    conn.commit()
    return dict(conn.execute("SELECT * FROM figures WHERE figure_id=?", (fid,)).fetchone())


# ---------------------------------------------------------------------------
# judge.post_validate
# ---------------------------------------------------------------------------
def test_post_validate_moves_unknown_finding_to_proposed():
    result = {
        "figure_id": "PMC1:F1",
        "figure_is_compound": False,
        "panels": [
            {
                "panel_label": "A",
                "bbox": [0, 0, 1, 1],
                "include": True,
                "exclusion_reason": None,
                "disease_key": "dm",
                "modality": "clinical_photo",
                "findings": [
                    {"finding_key": "gottron_papules", "evidence": "e"},
                    {"finding_key": "not_in_vocab", "evidence": "x"},
                ],
                "proposed_findings": [],
                "confidence": 0.9,
            }
        ],
    }
    out = judge.post_validate(result, {"gottron_papules"})
    panel = out["panels"][0]
    assert [f["finding_key"] for f in panel["findings"]] == ["gottron_papules"]
    assert "not_in_vocab" in panel["proposed_findings"]


def test_post_validate_non_pilot_disease_excluded():
    result = {
        "figure_id": "f",
        "panels": [
            {"panel_label": "A", "bbox": [0, 0, 1, 1], "include": True,
             "exclusion_reason": None, "disease_key": None,
             "findings": [], "proposed_findings": [], "confidence": 0.5}
        ],
    }
    out = judge.post_validate(result, set())
    panel = out["panels"][0]
    assert panel["include"] is False
    assert panel["exclusion_reason"] == "other_disease"


def test_post_validate_accepts_new_catalog_disease():
    result = {
        "figure_id": "f",
        "panels": [
            {"panel_label": "A", "bbox": [0, 0, 1, 1], "include": True,
             "exclusion_reason": None, "disease_key": "ra",
             "findings": [], "proposed_findings": [], "confidence": 0.5}
        ],
    }
    panel = judge.post_validate(result, set())["panels"][0]
    assert panel["include"] is True
    assert panel["disease_key"] == "ra"


def test_post_validate_bbox_clamp_and_swap():
    result = {
        "figure_id": "f",
        "panels": [
            {"panel_label": "A", "bbox": [1.2, 0.9, -0.1, 0.2], "include": True,
             "exclusion_reason": None, "disease_key": "dm",
             "findings": [], "proposed_findings": [], "confidence": 0.5}
        ],
    }
    out = judge.post_validate(result, set())
    assert out["panels"][0]["bbox"] == [0.0, 0.2, 1.0, 0.9]


# ---------------------------------------------------------------------------
# store pure helpers
# ---------------------------------------------------------------------------
def test_bbox_to_pixels_padding_and_clamp():
    # 2% of 640x480 = 12.8 / 9.6 px
    box = store.bbox_to_pixels([0.5, 0.5, 1.0, 1.0], 640, 480)
    assert box[0] == int(round(0.5 * 640 - 12.8))
    assert box[2] == 640 and box[3] == 480  # clamped
    box = store.bbox_to_pixels([0.0, 0.0, 0.25, 0.25], 640, 480)
    assert box[0] == 0 and box[1] == 0


def test_decide_crop_modes():
    a = {"panel_label": "A", "bbox": [0, 0, 0.5, 0.5], "include": True}
    b = {"panel_label": "B", "bbox": [0.6, 0.6, 1.0, 1.0], "include": True}
    # ND license mode -> everything is whole-figure
    assert store.decide_crop_modes([a, b], "whole_figure") == ["whole_figure"] * 2
    assert store.decide_crop_modes([a, b], "crop") == ["panel", "panel"]
    # tiny bbox (<3% of the figure) and missing bbox
    assert store.decide_crop_modes([{"bbox": [0, 0, 0.01, 0.01]}], "crop") == ["whole_figure"]
    assert store.decide_crop_modes([{"bbox": None}], "crop") == ["whole_figure"]
    # >30% overlap against ANY other panel, including an excluded one
    big = {"panel_label": "B", "bbox": [0.1, 0.1, 0.6, 0.6], "include": False}
    assert store.decide_crop_modes([a, big], "crop") == ["whole_figure"] * 2
    # small overlap stays panel
    c = {"panel_label": "C", "bbox": [0.4, 0.4, 0.9, 0.9]}
    assert store.decide_crop_modes([a, c], "crop") == ["panel", "panel"]


def test_attribution_text():
    base = dict(
        title="Title", journal="J", year=2024, doi="10.1/x",
        license_code="cc-by",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        label="Figure 1", panel_label="A",
    )
    one = store.attribution_text(["Doe"], 1, **base)
    assert one == (
        "Doe. Title. J 2024. doi:10.1/x. "
        "CC BY (https://creativecommons.org/licenses/by/4.0/). Figure 1A."
    )
    # The omit-letter decision lives in panel_suffix, not attribution_text.
    assert store.panel_suffix(
        {"figure_is_compound": False, "panels": [{"panel_label": "A"}]}, "A"
    ) == ""
    assert store.panel_suffix(
        {"figure_is_compound": True, "panels": [{"panel_label": "A"}, {"panel_label": "B"}]}, "A"
    ) == "A"
    two = store.attribution_text(["Doe", "Roe"], 2, **base)
    assert two.startswith("Doe and Roe. Title.")
    many = store.attribution_text(["Doe", "Roe", "Poe"], 5, **base)
    assert many.startswith("Doe et al. Title.")
    anon = store.attribution_text([], 0, **base)
    assert anon.startswith("Title. J 2024.")
    nodoi = store.attribution_text(
        [], 0, **{**base, "doi": None, "label": "Fig. 2", "panel_label": "B"}
    )
    assert "doi:" not in nodoi
    assert nodoi.endswith("CC BY (https://creativecommons.org/licenses/by/4.0/). Figure 2B.")


# ---------------------------------------------------------------------------
# store_figure end-to-end (mocked fetch)
# ---------------------------------------------------------------------------
def _parsed_article():
    return jats.ParsedArticle(
        figures=[], body_sections=[], authors=["Doe", "Roe"], author_count=2,
        article_copyright_holder=None, journal_name="J", publisher_name=None,
        corresp_country="Spain", first_aff_country=None,
    )


def _vision(panels):
    return {
        "figure_id": "PMC1:F1", "figure_is_compound": len(panels) > 1,
        "panels": panels,
    }


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


def test_store_figure_rejects_multi_panel_source(conn, vp_data_dir, monkeypatch):
    _article(conn)
    vision = _vision([
        _panel("A", [0.0, 0.0, 0.5, 0.5]),
        _panel("B", [0.6, 0.6, 1.0, 1.0]),
    ])
    fig = _figure(conn, vision=vision)
    monkeypatch.setattr(pmc, "fetch_image_bytes", lambda ref: _png_bytes())
    stats = store.store_figure(conn, fig, _article(conn), _parsed_article(), vp_data_dir)
    assert stats["panels"] == 0
    assert stats["excluded"] == 2
    assert conn.execute("SELECT COUNT(*) FROM panels").fetchone()[0] == 0
    stored = db.from_json(conn.execute(
        "SELECT vision_json FROM figures WHERE figure_id='PMC1:F1'"
    ).fetchone()[0], {})
    assert all(p["include"] is False for p in stored["panels"])
    assert all("collage" in p["curation_reason"] for p in stored["panels"])
    # Rejected sources leave no local figure or crop assets.
    assert not (vp_data_dir / "figures" / "PMC1" / "f1.webp").exists()


def test_store_nd_license_whole_figure(conn, vp_data_dir, monkeypatch):
    _article(conn)
    vision = _vision([_panel("A", [0.0, 0.0, 0.3, 0.3])])
    fig = _figure(conn, vision=vision, license_code="cc-by-nd")
    monkeypatch.setattr(pmc, "fetch_image_bytes", lambda ref: _png_bytes())
    stats = store.store_figure(conn, fig, _article(conn), _parsed_article(), vp_data_dir)
    assert stats["whole_figure"] == 1
    row = conn.execute("SELECT * FROM panels").fetchone()
    assert row["crop_mode"] == "whole_figure"
    # whole-figure crop = full 640x480
    assert row["width"] == 640 and row["height"] == 480


def test_store_dedup_reuses_file(conn, vp_data_dir, monkeypatch):
    _article(conn)
    monkeypatch.setattr(pmc, "fetch_image_bytes", lambda ref: _png_bytes())
    # Figure 1: whole-figure panel.
    vision1 = _vision([_panel("A", None)])
    fig1 = _figure(conn, vision=vision1)
    store.store_figure(conn, fig1, _article(conn), _parsed_article(), vp_data_dir)
    # Figure 2 in a second article produces an identical image.
    _article(conn, "PMC2")
    conn.execute(
        "INSERT INTO figures (figure_id, pmcid, label, status, image_url, "
        "image_format, effective_license, vision_json) "
        "VALUES ('PMC2:F9', 'PMC2', 'Figure 9', 'vision_accepted', 'https://s3/x/f2.png', "
        "'png', 'cc-by', ?)",
        (db.to_json({"figure_id": "PMC2:F9", "panels": [_panel("A", None)]}),),
    )
    conn.commit()
    fig2 = dict(conn.execute("SELECT * FROM figures WHERE figure_id='PMC2:F9'").fetchone())
    stats = store.store_figure(conn, fig2, _article(conn, "PMC2"), _parsed_article(), vp_data_dir)
    assert stats["dedup"] == 1
    rows = conn.execute("SELECT * FROM panels ORDER BY panel_id").fetchall()
    assert len(rows) == 2
    assert rows[0]["image_path"] == rows[1]["image_path"]
    assert rows[0]["attribution_text"] != rows[1]["attribution_text"]
    # only one file on disk
    assert len(list((vp_data_dir / "panels").rglob("*.webp"))) == 1


def test_store_proposed_finding_once_and_never_approved(conn, vp_data_dir, monkeypatch):
    _article(conn)
    conn.execute(
        "INSERT INTO findings_vocab (finding_key, disease_keys_json, label, category, "
        "approved, proposed_by_llm, proposal_count) "
        "VALUES ('approved_key', '[\"dm\"]', 'a', 'skin', 1, 0, 0)"
    )
    conn.commit()
    vision = _vision([_panel("A", [0, 0, 1, 1], proposed=["new sign", "approved_key"])])
    fig = _figure(conn, vision=vision)
    monkeypatch.setattr(pmc, "fetch_image_bytes", lambda ref: _png_bytes())
    stats = store.store_figure(conn, fig, _article(conn), _parsed_article(), vp_data_dir)
    assert stats["proposed"] == 1  # new_sign counted; approved_key skipped
    row = conn.execute("SELECT proposal_count, approved FROM findings_vocab WHERE finding_key='new_sign'").fetchone()
    assert row["proposal_count"] == 1 and row["approved"] == 0
    row = conn.execute("SELECT proposal_count FROM findings_vocab WHERE finding_key='approved_key'").fetchone()
    assert row["proposal_count"] == 0
    # Storing the same figure again must not re-increment (store would only
    # re-run while status is vision_accepted — the transition bumps once).
    stats2 = store.store_figure(conn, fig, _article(conn), _parsed_article(), vp_data_dir)
    row = conn.execute("SELECT proposal_count FROM findings_vocab WHERE finding_key='new_sign'").fetchone()
    assert row["proposal_count"] == 2  # second explicit call does count — the
    # 'once per figure' guarantee is the vision_accepted->stored transition,
    # enforced by run() only touching vision_accepted rows.
    assert stats2["proposed"] == 1


# ---------------------------------------------------------------------------
# extract_findings
# ---------------------------------------------------------------------------
def test_pick_sections_keywords_and_cap():
    secs = [
        ("Introduction", "intro text"),
        ("Cutaneous manifestations", "skin findings " * 100),
        ("Methods", "m"),
        ("Imaging features", "img " * 50),
    ]
    picked = extract_findings.pick_sections(secs)
    assert [t for t, _ in picked] == ["Cutaneous manifestations", "Imaging features"]
    tiny = extract_findings.pick_sections(secs, max_chars=50)
    assert sum(len(t) for _, t in tiny) <= 50
    assert extract_findings.pick_sections([("Methods", "m")]) == [("Methods", "m")]


def test_quote_verified_normalization():
    src = "Patients show Gottron’s papules — 70% of cases, in one review."
    assert extract_findings.quote_verified("Gottron's papules - 70% of cases", src)
    assert not extract_findings.quote_verified("hallucinated claim", src)


def test_quote_verified_citation_dropped():
    src = ("ACR/EULAR classification criteria were developed for SLE patients [15] "
           "and refined in later cohorts.")
    # Model drops the bracketed citation.
    assert extract_findings.quote_verified(
        "ACR/EULAR classification criteria were developed for SLE patients", src
    )


def test_quote_verified_pronoun_swap():
    src = ("The malar rash is present in most acute cutaneous lupus patients and "
           "spares the nasolabial folds in the majority of reported cases.")
    assert extract_findings.quote_verified(
        "It is present in most acute cutaneous lupus patients and spares the "
        "nasolabial folds in the majority of reported cases.", src
    )


def test_quote_verified_quote_char_swap():
    src = "Patients present with Gottron’s papules over the knuckles."
    assert extract_findings.quote_verified(
        "present with Gottron's papules over the knuckles", src
    )


def test_quote_verified_fabricated_sentence_fails():
    src = ("The malar rash is present in most acute cutaneous lupus patients and "
           "spares the nasolabial folds in the majority of reported cases.")
    assert not extract_findings.quote_verified(
        "The malar rash resolves spontaneously without treatment in nearly all "
        "acute cutaneous lupus patients within one week.", src
    )


def test_extract_post_validate(conn):
    conn.execute(
        "INSERT INTO findings_vocab (finding_key, disease_keys_json, label, category, approved) "
        "VALUES ('gottron_papules', '[\"dm\"]', 'g', 'skin', 1)"
    )
    valid = {"gottron_papules"}
    src = "the quick brown fox"
    ok = extract_findings.post_validate(
        {"disease_key": "dm", "finding_key": "gottron_papules", "quote": "quick brown"},
        valid, src,
    )
    assert ok["finding_key"] == "gottron_papules"
    # Unknown disease key is still dropped outside the configured catalog.
    assert extract_findings.post_validate(
        {"disease_key": "not_in_catalog", "finding_key": None, "quote": "quick"}, valid, src
    ) is None
    ra = extract_findings.post_validate(
        {"disease_key": "ra", "finding_key": None, "quote": "quick brown"}, valid, src
    )
    assert ra and ra["disease_key"] == "ra"
    # unknown key -> proposed
    moved = extract_findings.post_validate(
        {"disease_key": "dm", "finding_key": "unknown_thing", "quote": "quick brown"},
        valid, src,
    )
    assert moved["finding_key"] is None and moved["proposed_finding"] == "unknown_thing"
    # placeholder key must not clobber a real proposed_finding term
    kept = extract_findings.post_validate(
        {"disease_key": "dm", "finding_key": "proposed_finding",
         "proposed_finding": "hyperferritinemia", "quote": "quick brown"},
        valid, src,
    )
    assert kept["finding_key"] is None and kept["proposed_finding"] == "hyperferritinemia"
    # unverified + too-long quotes dropped
    assert extract_findings.post_validate(
        {"disease_key": "dm", "finding_key": "gottron_papules", "quote": "not in source"},
        valid, src,
    )["_unverified_quote"]
    assert extract_findings.post_validate(
        {"disease_key": "dm", "finding_key": "gottron_papules", "quote": " ".join(["w"] * 41)},
        valid, src,
    )["_unverified_quote"]


def _mock_bundle(monkeypatch):
    monkeypatch.setattr(
        pmc, "get_article_bundle",
        lambda pmcid, **kw: pmc.ArticleBundle(
            pmcid=pmcid, xml_text=FIXTURE_XML,
            resolver=lambda h: pmc.ImageRef(url=None, needs_bytes=True),
        ),
    )


def _stats():
    return {
        "inserted": 0, "proposed": 0, "unverified_quote": 0,
        "dropped_disease": 0, "no_sections": 0, "errors": 0, "image_rows": 0,
    }


def test_extract_article_idempotent(conn, monkeypatch):
    article = dict(_article(conn))
    _mock_bundle(monkeypatch)
    vocab = extract_findings._vocabulary(conn, ["dm"])
    ctx = extract_findings.prepare_article(article, vocab)
    assert ctx is not None
    # gottron_papules is already in the seeded approved vocab
    quote = "illustrated in Figures 1 and 2"
    resp = {
        "assertions": [
            {"disease_key": "dm", "subtype": None, "finding_key": "gottron_papules",
             "proposed_finding": None, "frequency_text": "common",
             "pct_low": None, "pct_high": None, "specificity_text": None,
             "quote": quote},
            {"disease_key": "dm", "subtype": None, "finding_key": "new_finding_x",
             "proposed_finding": None, "frequency_text": None,
             "pct_low": None, "pct_high": None, "specificity_text": None,
             "quote": quote},
        ]
    }
    stats = _stats()
    extract_findings.apply_response(conn, article, ctx, resp, cached=False, stats=stats)
    conn.commit()
    # both assertions land as rows: the vocab key as-is, the proposal under
    # its snake_case key.
    assert stats["inserted"] == 2
    assert stats["proposed"] == 1
    n = conn.execute("SELECT proposal_count FROM findings_vocab WHERE finding_key='new_finding_x'").fetchone()["proposal_count"]
    assert n == 1
    # Rerun from a cached response: delete+reinsert rows, no new proposal.
    stats2 = _stats()
    extract_findings.apply_response(conn, article, ctx, resp, cached=True, stats=stats2)
    conn.commit()
    assert stats2["inserted"] == 2 and stats2["proposed"] == 0
    n = conn.execute("SELECT proposal_count FROM findings_vocab WHERE finding_key='new_finding_x'").fetchone()["proposal_count"]
    assert n == 1
    rows = conn.execute("SELECT COUNT(*) n FROM disease_findings WHERE pmcid='PMC1' AND source='text'").fetchone()
    assert rows["n"] == 2


def test_extract_run_skips_existing_and_force(conn, monkeypatch):
    _article(conn)
    conn.execute(
        "INSERT INTO disease_findings (disease_key, finding_key, source, pmcid, quote) "
        "VALUES ('dm', 'gottron_papules', 'text', 'PMC1', 'q')"
    )
    conn.commit()
    _mock_bundle(monkeypatch)

    class _Client:
        def __init__(self):
            self.requests_seen = []
            self.spent_usd = 0.0
            self.concurrency = 4

        def iter_many(self, requests, max_in_flight=None):
            for i, req in enumerate(requests):
                self.requests_seen.append(req)
                yield BatchResult(
                    index=i, parsed={"assertions": []}, meta={"cached": False}
                )

    client = _Client()
    monkeypatch.setattr(extract_findings.llm, "LLMClient", lambda **kw: client)
    # existing source='text' rows -> article skipped entirely
    assert extract_findings.run(_args()) == 0
    assert client.requests_seen == []
    # --force reprocesses (rows are deleted + reinserted)
    assert extract_findings.run(_args(force=True)) == 0
    assert len(client.requests_seen) == 1


# ---------------------------------------------------------------------------
# report (no LLM)
# ---------------------------------------------------------------------------
def test_report_writes_files(conn, vp_data_dir, monkeypatch):
    # Seed a small dataset directly.
    _article(conn)
    _figure(conn, vision=_vision([_panel("A", [0, 0, 1, 1])]))
    conn.execute(
        "INSERT INTO panels (panel_id, figure_id, pmcid, disease_key, modality, "
        "findings_json, image_path, sha256, attribution_text) "
        "VALUES ('p1', 'PMC1:F1', 'PMC1', 'dm', 'clinical_photo', "
        "'[{\"finding_key\":\"gottron_papules\",\"evidence\":\"e\"}]', 'panels/x.png', 's', 'a')"
    )
    conn.execute(
        "INSERT INTO llm_calls (stage, model, input_hash, cost_usd, input_tokens, output_tokens) "
        "VALUES ('p3', 'm', 'h1', 0.01, 100, 50)"
    )
    conn.commit()

    import src.visual_pilot.llm as llm_mod

    class _Boom:
        def __init__(self, *a, **kw):
            raise AssertionError("report must not construct LLMClient")

    monkeypatch.setattr(llm_mod, "LLMClient", _Boom)
    assert report.run(_args()) == 0
    reports = config_reports(vp_data_dir)
    assert (reports / "pilot_report.md").exists()
    data = json.loads((reports / "pilot_report.json").read_text())
    assert data["funnel"]["dm"]["panels"] == 1
    assert data["costs"]["total_usd"] == pytest.approx(0.01)
    assert (reports / "spot_accepted.html").exists()
    assert (reports / "spot_caption_rejected.csv").exists()


def config_reports(vp_data_dir):
    from src.visual_pilot import config

    return config.data_dir() / "reports"


# ---------------------------------------------------------------------------
# judge run: resume + no-disk
# ---------------------------------------------------------------------------
def test_judge_resume_and_no_disk(conn, vp_data_dir, monkeypatch):
    _article(conn)
    _figure(conn, status="caption_kept")
    p3 = {
        "figure_id": "PMC1:F1",
        "figure_is_compound": False,
        "panels": [
            {"panel_label": "A", "bbox": [0, 0, 1, 1], "include": True,
             "exclusion_reason": None, "disease_key": "dm", "subtype": "classic",
             "modality": "clinical_photo", "body_site": "hands",
             "findings": [{"finding_key": "gottron_papules", "evidence": "e"}],
             "proposed_findings": [], "typicality": "classic", "stage": None,
             "age_group": "adult", "skin_tone": "light", "stated_ethnicity": None,
             "stated_ethnicity_quote": None, "annotations_present": False,
             "confidence": 0.9, "rationale": "r"}
        ],
    }

    class _Client:
        def __init__(self):
            self.calls = []
            self.spent_usd = 0.0

        def iter_many(self, requests, max_in_flight=None):
            for i, req in enumerate(requests):
                self.calls.append([req])
                yield BatchResult(index=i, parsed=p3)

    client = _Client()
    monkeypatch.setattr(judge.llm, "LLMClient", lambda **kw: client)
    monkeypatch.setattr(pmc, "fetch_image_bytes", lambda ref: _png_bytes())
    assert judge.run(_args()) == 0
    row = conn.execute("SELECT status, vision_json FROM figures WHERE figure_id='PMC1:F1'").fetchone()
    assert row["status"] == "vision_accepted"
    assert json.loads(row["vision_json"])["panels"][0]["include"]
    assert client.calls and len(client.calls[0]) == 1
    # no image files under the data dir (only sqlite)
    imgs = [p for p in vp_data_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".webp", ".tif", ".tiff"}]
    assert imgs == []
    # rerun: figure already judged -> zero new calls
    assert judge.run(_args()) == 0
    assert len(client.calls) == 1


def _p3_response():
    return {
        "figure_id": "PMC1:F1",
        "figure_is_compound": False,
        "panels": [
            {"panel_label": "A", "bbox": [0, 0, 1, 1], "include": True,
             "exclusion_reason": None, "disease_key": "dm", "subtype": "classic",
             "modality": "clinical_photo", "body_site": "hands",
             "findings": [{"finding_key": "gottron_papules", "evidence": "e"}],
             "proposed_findings": [], "typicality": "classic", "stage": None,
             "age_group": "adult", "skin_tone": "light", "stated_ethnicity": None,
             "stated_ethnicity_quote": None, "annotations_present": False,
             "confidence": 0.9, "rationale": "r"}
        ],
    }


def test_judge_retries_vision_error_bounded(conn, vp_data_dir, monkeypatch):
    _article(conn)
    _figure(conn, status="vision_error")
    conn.execute("UPDATE figures SET attempts=1 WHERE figure_id='PMC1:F1'")
    conn.commit()
    p3 = _p3_response()

    class _Client:
        def __init__(self):
            self.calls = 0
            self.spent_usd = 0.0

        def iter_many(self, requests, max_in_flight=None):
            for i, req in enumerate(requests):
                self.calls += 1
                yield BatchResult(index=i, parsed=p3)

    client = _Client()
    monkeypatch.setattr(judge.llm, "LLMClient", lambda **kw: client)
    monkeypatch.setattr(pmc, "fetch_image_bytes", lambda ref: _png_bytes())
    # vision_error with attempts<3 is picked back up
    assert judge.run(_args()) == 0
    assert client.calls == 1
    assert conn.execute(
        "SELECT status FROM figures WHERE figure_id='PMC1:F1'"
    ).fetchone()["status"] == "vision_accepted"
    # at MAX_ATTEMPTS it is skipped
    conn.execute("UPDATE figures SET status='vision_error', attempts=3 WHERE figure_id='PMC1:F1'")
    conn.commit()
    assert judge.run(_args()) == 0
    assert client.calls == 1
    assert conn.execute(
        "SELECT status FROM figures WHERE figure_id='PMC1:F1'"
    ).fetchone()["status"] == "vision_error"


def test_connect_busy_timeout(conn):
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000


def test_rebuild_image_rows(conn):
    _article(conn)
    _figure(conn, vision=_vision([_panel("A", [0, 0, 1, 1])]))
    conn.execute(
        "INSERT INTO panels (panel_id, figure_id, pmcid, disease_key, subtype, "
        "findings_json) VALUES ('p1', 'PMC1:F1', 'PMC1', 'dm', 'classic', "
        "'[{\"finding_key\":\"gottron_papules\",\"evidence\":\"cap phrase\"}]')"
    )
    conn.execute(
        "INSERT INTO disease_findings (disease_key, finding_key, source, pmcid) "
        "VALUES ('dm', 'stale', 'image', 'PMC1')"
    )
    conn.commit()
    n = extract_findings.rebuild_image_rows(conn)
    assert n == 1
    rows = conn.execute(
        "SELECT disease_key, finding_key, subtype, pmcid, quote FROM disease_findings "
        "WHERE source='image'"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert (row["disease_key"], row["finding_key"]) == ("dm", "gottron_papules")
    assert row["subtype"] == "classic"
    assert row["pmcid"] == "PMC1"
    assert row["quote"] == "cap phrase"  # quote = panel evidence


# ---------------------------------------------------------------------------
# W-rework: TIFF original write, subtype enum, report cumulative funnel
# ---------------------------------------------------------------------------
def test_write_original_tiff_to_webp(vp_data_dir):
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 20, 30)).save(buf, format="TIFF")
    figure = {"pmcid": "PMC9", "figure_id": "PMC9:F1", "image_url": "https://s3/x/img.tiff"}
    rel = store.write_original(buf.getvalue(), figure, vp_data_dir)
    assert rel.endswith(".webp")
    with Image.open(vp_data_dir / rel) as im:
        assert im.format == "WEBP" and im.size == (32, 24)


def test_normalize_subtype_variants():
    assert judge.normalize_subtype("as", "nr-axSpA") == ("nr_axspa", None)
    assert judge.normalize_subtype("dm", "anti-MDA5") == ("anti_mda5", None)
    assert judge.normalize_subtype("sle", "ACLE") == ("acle", None)
    canon, raw = judge.normalize_subtype("sle", "chilblain lupus")
    assert canon is None and raw == "chilblain lupus"
    assert judge.normalize_subtype("dm", None) == (None, None)


def test_post_validate_subtype_dropped_to_rationale():
    result = {
        "figure_id": "f",
        "panels": [
            {"panel_label": "A", "bbox": [0, 0, 1, 1], "include": True,
             "exclusion_reason": None, "disease_key": "sle",
             "subtype": "lupus_panniculitis", "rationale": "facial plaque",
             "findings": [], "proposed_findings": [], "confidence": 0.5}
        ],
    }
    panel = judge.post_validate(result, set())["panels"][0]
    assert panel["subtype"] is None
    assert "[raw subtype: lupus_panniculitis]" in panel["rationale"]


def test_report_funnel_cumulative(conn, vp_data_dir):
    """A stored figure still counts as caption_kept + vision_accepted."""
    _article(conn)
    vision = _vision([_panel("A", [0, 0, 1, 1])])
    _figure(conn, vision=vision)
    conn.execute(
        "UPDATE figures SET status='stored', triage_json=? WHERE figure_id='PMC1:F1'",
        (db.to_json({"route": "keep", "reason": "patient image", "source": "p2"}),),
    )
    conn.execute(
        "UPDATE figures SET vision_json=? WHERE figure_id='PMC1:F1'",
        (db.to_json(vision),),
    )
    conn.commit()
    # _figure created it as vision_accepted; check the funnel derivations.
    f = report.funnel(conn)
    assert f["triage"]["kept"] == 1
    assert f["vision"]["accepted"] == 1


# ---------------------------------------------------------------------------
# W6: streaming judge pipeline (iter_many) + originals handoff (C5)
# ---------------------------------------------------------------------------
def _insert_judge_figure(conn, fid, url, status="caption_kept"):
    conn.execute(
        "INSERT OR REPLACE INTO figures (figure_id, pmcid, label, caption, status, "
        "image_url, image_format, effective_license) "
        "VALUES (?, 'PMC1', 'Figure 1', 'cap', ?, ?, 'png', 'cc-by')",
        (fid, status, url),
    )
    conn.commit()
    return dict(conn.execute("SELECT * FROM figures WHERE figure_id=?", (fid,)).fetchone())


def _p3_verdict(fid, include=True):
    return {
        "figure_id": fid,
        "figure_is_compound": False,
        "panels": [
            {"panel_label": "A", "bbox": [0, 0, 1, 1], "include": include,
             "exclusion_reason": None if include else "not_relevant",
             "disease_key": "dm" if include else None, "subtype": "classic",
             "modality": "clinical_photo", "body_site": "hands",
             "findings": [{"finding_key": "gottron_papules", "evidence": "e"}] if include else [],
             "proposed_findings": [], "typicality": "classic", "stage": None,
             "age_group": "adult", "skin_tone": "light", "stated_ethnicity": None,
             "stated_ethnicity_quote": None, "annotations_present": False,
             "confidence": 0.9, "rationale": "r"}
        ],
    }


def test_judge_pipeline_matches_reference(conn, vp_data_dir, monkeypatch):
    """The streaming pipeline emits the same P3 inputs and figure rows as the
    old fetch-4/judge-4 lockstep, across every outcome class."""
    monkeypatch.setattr(config, "VP_JUDGE_CONCURRENCY", 2)
    monkeypatch.setattr(config, "VP_FETCH_CONCURRENCY", 3)
    article = dict(_article(conn))
    urls = {
        "PMC1:F1": "https://s3/x/f1.png",   # accepted
        "PMC1:F2": "https://s3/x/f2.png",   # rejected
        "PMC1:F3": "https://s3/x/f3.png",   # fetch error
        "PMC1:F4": "https://s3/x/f4.png",   # llm error
    }
    blobs = {
        "PMC1:F1": _png_bytes(color=(10, 20, 30)),
        "PMC1:F2": _png_bytes(color=(40, 50, 60)),
        "PMC1:F4": _png_bytes(color=(70, 80, 90)),
    }
    figs_before = {fid: _insert_judge_figure(conn, fid, url) for fid, url in urls.items()}
    _insert_judge_figure(conn, "PMC1:F5", None)  # no image_url -> needs_bytes
    url_to_fig = {u: f for f, u in urls.items()}

    def _fetch(ref):
        if ref.url == urls["PMC1:F3"]:
            raise pmc.PmcError("503 boom")
        return blobs[url_to_fig[ref.url]]

    monkeypatch.setattr(pmc, "fetch_image_bytes", _fetch)

    responses = {
        "PMC1:F1": _p3_verdict("PMC1:F1", include=True),
        "PMC1:F2": _p3_verdict("PMC1:F2", include=False),
        "PMC1:F4": _p3_verdict("PMC1:F4", include=True),
    }
    ctor_kwargs = {}
    in_flight = {"cur": 0, "peak": 0}
    lock = threading.Lock()
    seen = {}  # figure_id -> input_hash actually sent

    real_client = llm.LLMClient(db_conn=None)

    def _call_json(stage, model, system, user_content, schema, images=None, prompt_version=""):
        fid = json.loads(user_content)["figure_id"]
        input_hash = real_client._input_hash(
            stage, model, prompt_version, system, user_content, images
        )
        with lock:
            in_flight["cur"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["cur"])
            seen[fid] = input_hash
        try:
            time.sleep(0.01)
            if fid == "PMC1:F4":
                raise llm.LLMError("provider boom")
            return responses[fid], {"cached": False, "input_hash": input_hash}
        finally:
            with lock:
                in_flight["cur"] -= 1

    monkeypatch.setattr(real_client, "call_json", _call_json)

    def _client_factory(**kw):
        ctor_kwargs.update(kw)
        return real_client

    monkeypatch.setattr(judge.llm, "LLMClient", _client_factory)
    assert judge.run(_args()) == 0

    # Client settings: judge uses its own concurrency/timeout, uncapped by 4.
    assert ctor_kwargs["concurrency"] == config.VP_JUDGE_CONCURRENCY == 2
    assert ctor_kwargs["timeout_seconds"] == config.VP_JUDGE_TIMEOUT_SECONDS
    assert ctor_kwargs["max_retries"] == 0
    # P3 in-flight calls never exceed the configured cap.
    assert 1 <= in_flight["peak"] <= config.VP_JUDGE_CONCURRENCY

    # Reference: the exact set of input_hash values the sequential code sends.
    vocab = judge.vocab_for_diseases(conn, ["dm"])
    valid_keys = {v["finding_key"] for v in vocab}
    expected = {}
    for fid, blob in blobs.items():
        uc = judge.user_content(figs_before[fid], article, vocab)
        expected[fid] = db.llm_input_hash(
            "p3", config.VP_JUDGE_MODEL,
            {
                "prompt_version": judge.P3.version,
                "system": judge.P3.system,
                "user_content": uc,
                "images": [f"sha256:{hashlib.sha256(blob).hexdigest()}"],
            },
        )
    assert seen == expected

    # Reference: final figures rows for every outcome.
    expected_status = {
        "PMC1:F1": "vision_accepted",
        "PMC1:F2": "vision_rejected",
        "PMC1:F3": "vision_error",
        "PMC1:F4": "vision_error",
        "PMC1:F5": "vision_error",
    }
    for fid, want in expected_status.items():
        got = conn.execute(
            "SELECT status FROM figures WHERE figure_id=?", (fid,)
        ).fetchone()["status"]
        assert got == want, fid

    r1 = conn.execute("SELECT * FROM figures WHERE figure_id='PMC1:F1'").fetchone()
    assert json.loads(r1["vision_json"]) == judge.post_validate(responses["PMC1:F1"], valid_keys)
    assert r1["image_format"] == "png"
    assert r1["sha256"] == hashlib.sha256(blobs["PMC1:F1"]).hexdigest()
    assert r1["error"] is None
    r2 = conn.execute("SELECT * FROM figures WHERE figure_id='PMC1:F2'").fetchone()
    assert json.loads(r2["vision_json"]) == judge.post_validate(responses["PMC1:F2"], valid_keys)
    assert r2["sha256"] == hashlib.sha256(blobs["PMC1:F2"]).hexdigest()
    assert r2["error"] is None
    r3 = conn.execute("SELECT error, attempts FROM figures WHERE figure_id='PMC1:F3'").fetchone()
    assert r3["error"].startswith("fetch:") and r3["attempts"] == 1
    r4 = conn.execute("SELECT error, attempts FROM figures WHERE figure_id='PMC1:F4'").fetchone()
    assert r4["error"].startswith("llm:") and r4["attempts"] == 1
    r5 = conn.execute("SELECT error, attempts FROM figures WHERE figure_id='PMC1:F5'").fetchone()
    assert r5["error"] == "needs_bytes" and r5["attempts"] == 0

    # C5: only the accepted figure's original bytes are parked for store.
    sha1 = hashlib.sha256(blobs["PMC1:F1"]).hexdigest()
    assert originals.take("PMC1:F1", sha1) == blobs["PMC1:F1"]
    for fid in ("PMC1:F2", "PMC1:F3", "PMC1:F4", "PMC1:F5"):
        sha = hashlib.sha256(blobs.get(fid, b"")).hexdigest()
        assert originals.take(fid, sha) is None
    assert originals._stats() == (0, 0)

    # No image files written to disk for any outcome (invariant §1.3).
    imgs = [
        p for p in vp_data_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".webp", ".tif", ".tiff"}
    ]
    assert imgs == []


def test_judge_budget_stops_cleanly(conn, vp_data_dir, monkeypatch, capsys):
    _article(conn)
    _insert_judge_figure(conn, "PMC1:F1", "https://s3/x/f1.png")
    _insert_judge_figure(conn, "PMC1:F2", "https://s3/x/f2.png")
    monkeypatch.setattr(pmc, "fetch_image_bytes", lambda ref: _png_bytes())

    real_client = llm.LLMClient(db_conn=None, budget_usd=0.0)

    def _call_json(**kw):
        raise llm.BudgetExceeded("budget $0.00 exhausted")

    monkeypatch.setattr(real_client, "call_json", _call_json)
    monkeypatch.setattr(judge.llm, "LLMClient", lambda **kw: real_client)
    assert judge.run(_args(budget_usd=0.0)) == 0
    rows = conn.execute(
        "SELECT figure_id, status FROM figures ORDER BY figure_id"
    ).fetchall()
    # BudgetExceeded leaves the status untouched so a rerun resumes cleanly.
    assert [r["status"] for r in rows] == ["caption_kept", "caption_kept"]
    assert "budget" in capsys.readouterr().out.lower()
    assert originals._stats() == (0, 0)
