"""C1 eligibility-surface tests for publication.panel_records/eligible_panels."""

from src.visual_pilot import publication
from balanced_fixtures import (
    add_article,
    add_disease,
    add_figure,
    add_finding,
    add_identity_review,
    add_panel,
    make_db,
)


def _seed_sle(conn):
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash")


def _eligible_panel(conn, root, panel_id="p1", **kwargs):
    add_article(conn, "PMC1")
    add_figure(conn, "PMC1:fig1", "PMC1")
    add_panel(conn, root, panel_id, "PMC1:fig1", "PMC1", "sle", **kwargs)
    conn.commit()


def _record(conn, panel_id):
    rows = publication.panel_records(conn)
    return next(r for r in rows if r["panel_id"] == panel_id)


def test_eligible_panel_carries_supported_findings_and_age(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    _eligible_panel(conn, tmp_path, sha256="sha-1")
    record = _record(conn, "p1")
    assert record["eligible"] is True
    assert record["supported_finding_keys"] == ["malar_rash"]
    assert record["source_age"]["age_group"] == "adult"
    assert publication.eligible_panels(conn)[0]["panel_id"] == "p1"
    conn.close()


def test_panel_records_retains_failures_with_reasons(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    add_article(conn, "PMC1", license_code="CC-BY-NC")
    add_figure(conn, "PMC1:fig1", "PMC1")
    add_panel(conn, tmp_path, "p1", "PMC1:fig1", "PMC1", "sle", sha256="x")
    record = _record(conn, "p1")
    assert record["eligible"] is False
    assert "article license disallows publication" in record["reasons"]
    assert "license/third-party" in record["rejection_categories"]
    assert publication.eligible_panels(conn) == []
    conn.close()


def test_third_party_and_whole_figure_only_licenses(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    add_article(conn, "PMC1")
    add_figure(conn, "PMC1:fig1", "PMC1", triage={"third_party": True})
    add_panel(conn, tmp_path, "p3p", "PMC1:fig1", "PMC1", "sle")
    assert "third-party figure" in _record(conn, "p3p")["reasons"]

    add_article(conn, "PMC2", license_code="CC-BY-ND")
    add_figure(conn, "PMC2:fig1", "PMC2", license_code="CC-BY-ND")
    add_panel(
        conn, tmp_path, "pnd", "PMC2:fig1", "PMC2", "sle",
        license_code="CC-BY-ND", crop_mode="panel",
    )
    reasons = _record(conn, "pnd")["reasons"]
    assert "article license permits whole figures only" in reasons
    add_panel(
        conn, tmp_path, "pnd2", "PMC2:fig1", "PMC2", "sle",
        license_code="CC-BY-ND", crop_mode="whole_figure",
    )
    assert _record(conn, "pnd2")["eligible"] is True
    conn.close()


def test_unknown_age_publishes_but_unsupported_label_rejects(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    add_article(conn, "PMC1")
    add_figure(
        conn, "PMC1:noage", "PMC1",
        caption="Malar rash in a patient with systemic lupus erythematosus",
    )
    add_panel(conn, tmp_path, "pn", "PMC1:noage", "PMC1", "sle")
    record = _record(conn, "pn")
    assert record["eligible"] is True
    assert record["source_age"]["age_group"] == "unknown"

    add_figure(
        conn, "PMC1:nolabel", "PMC1",
        caption="Discoid plaque in a 40-year-old patient with systemic lupus erythematosus",
    )
    add_panel(conn, tmp_path, "pl", "PMC1:nolabel", "PMC1", "sle")
    record = _record(conn, "pl")
    assert record["eligible"] is False
    assert "no approved source-supported finding label" in record["reasons"]
    conn.close()


def test_missing_files_and_dimension_mismatch_are_actionable(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    add_article(conn, "PMC1")
    add_figure(conn, "PMC1:fig1", "PMC1")
    add_panel(conn, tmp_path, "p_img", "PMC1:fig1", "PMC1", "sle", image=False)
    assert any(r.startswith("missing image_path") for r in _record(conn, "p_img")["reasons"])

    add_panel(conn, tmp_path, "p_th", "PMC1:fig1", "PMC1", "sle", thumb=False)
    assert any(r.startswith("missing thumb_path") for r in _record(conn, "p_th")["reasons"])

    add_panel(
        conn, tmp_path, "p_dim", "PMC1:fig1", "PMC1", "sle",
        width=400, height=300, image_size=(200, 150),
    )
    assert "stored image dimensions do not match file" in _record(conn, "p_dim")["reasons"]
    conn.close()


def test_current_hash_exclusions_including_empty_audit_hash(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    _eligible_panel(conn, tmp_path, "p_match", sha256="sha-match")
    _eligible_panel(conn, tmp_path, "p_stale", sha256="sha-new")
    _eligible_panel(conn, tmp_path, "p_empty", sha256=None)
    conn.execute(
        "INSERT INTO panel_curation(panel_id,image_sha256,decision,reason,policy_version) "
        "VALUES('p_match','sha-match','exclude','reviewed','v1')"
    )
    conn.execute(
        "INSERT INTO panel_curation(panel_id,image_sha256,decision,reason,policy_version) "
        "VALUES('p_stale','sha-old','exclude','reviewed','v1')"
    )
    conn.execute(
        "INSERT INTO panel_curation(panel_id,image_sha256,decision,reason,policy_version) "
        "VALUES('p_empty','','exclude','reviewed','v1')"
    )
    conn.commit()
    assert "current-hash audit exclusion" in _record(conn, "p_match")["reasons"]
    assert _record(conn, "p_stale")["eligible"] is True
    assert "current-hash audit exclusion" in _record(conn, "p_empty")["reasons"]
    conn.close()


def test_identity_review_attaches_only_when_hash_current(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    _eligible_panel(conn, tmp_path, "p_ok", sha256="sha-ok")
    _eligible_panel(conn, tmp_path, "p_stale", sha256="sha-new")
    add_identity_review(conn, "p_ok", sha256="sha-ok", patient_group_key="patient-1")
    add_identity_review(conn, "p_stale", sha256="sha-stale", patient_group_key="patient-9")
    conn.commit()
    assert _record(conn, "p_ok")["identity_review"]["patient_group_key"] == "patient-1"
    assert "identity_review" not in _record(conn, "p_stale")
    conn.close()


def test_combined_plate_stays_eligible_but_earns_no_pair_credit(tmp_path):
    conn = make_db(tmp_path)
    _seed_sle(conn)
    add_finding(conn, "discoid_rash", ("sle",), label="Discoid rash")
    add_article(conn, "PMC1")
    vision = {
        "figure_is_compound": True,
        "panels": [
            {"panel_label": "a", "disease_key": "sle", "modality": "clinical_photo"},
            {"panel_label": "b", "disease_key": "sle", "modality": "clinical_photo"},
        ],
    }
    add_figure(
        conn, "PMC1:fig1", "PMC1", vision=vision,
        caption="Malar rash and discoid rash in a 30-year-old patient with "
        "systemic lupus erythematosus",
    )
    add_panel(
        conn, tmp_path, "plate1", "PMC1:fig1", "PMC1", "sle",
        findings=(), plate_findings=("malar_rash", "discoid_rash"),
        plate_kind="combined",
    )
    conn.commit()
    record = _record(conn, "plate1")
    assert record["eligible"] is True
    assert record["supported_finding_keys"] == []
    from src.visual_pilot import gallery

    snapshot = gallery.coverage_snapshot(conn, "sle")
    assert snapshot["sle"]["malar_rash"]["eligible_distinct"] == 0
    assert snapshot["sle"]["malar_rash"]["published_distinct"] == 0
    conn.close()


def test_hla_b27_uveitis_without_as_attribution_earns_no_credit(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "as", "Ankylosing spondylitis")
    add_finding(
        conn, "anterior_uveitis", ("as",), label="Anterior uveitis",
        category="eye", synonyms=("acute anterior uveitis",),
    )
    add_article(conn, "PMC1", title="Uveitis in rheumatology: a review")
    add_figure(
        conn, "PMC1:fig1", "PMC1",
        caption="HLA-B27-associated acute anterior uveitis in a 34-year-old patient",
    )
    add_panel(
        conn, tmp_path, "p_as", "PMC1:fig1", "PMC1", "as",
        findings=("anterior_uveitis",), modality="ophthalmic", body_site="eye",
    )
    conn.commit()
    record = _record(conn, "p_as")
    assert record["eligible"] is False
    assert "disease association not stated in figure caption" in record["reasons"]

    from src.visual_pilot import gallery

    snapshot = gallery.coverage_snapshot(conn, "as")
    assert snapshot["as"]["anterior_uveitis"]["eligible_distinct"] == 0
    conn.close()


def test_mixed_disease_plate_stays_excluded(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "as", "Ankylosing spondylitis")
    add_finding(
        conn, "anterior_uveitis", ("as", "psa"), label="Anterior uveitis",
        category="eye",
    )
    add_article(conn, "PMC1")
    vision = {
        "figure_is_compound": True,
        "panels": [
            {"panel_label": "a", "disease_key": "as", "modality": "ophthalmic"},
            {"panel_label": "b", "disease_key": "psa", "modality": "ophthalmic"},
        ],
    }
    add_figure(
        conn, "PMC1:fig1", "PMC1", vision=vision,
        caption="Anterior uveitis in a 45-year-old patient with ankylosing "
        "spondylitis and psoriatic arthritis",
    )
    add_panel(
        conn, tmp_path, "plate_mixed", "PMC1:fig1", "PMC1", "as",
        findings=(), plate_findings=("anterior_uveitis",),
        plate_kind="combined", modality="ophthalmic", body_site="eye",
    )
    conn.commit()
    record = _record(conn, "plate_mixed")
    assert record["eligible"] is False
    assert "multi-panel plate: multi_disease" in record["reasons"]
    conn.close()


def test_requeue_age_vetoes_restores_only_age_vetoed_accepts(tmp_path):
    from src.visual_pilot import curation, db, judge

    conn = make_db(tmp_path)
    _seed_sle(conn)
    add_article(conn, "PMC1")

    def vetoed(reason, exclusion):
        return {"figure_is_compound": False, "panels": [{
            "panel_label": "A", "bbox": [0, 0, 1, 1], "include": False,
            "exclusion_reason": exclusion, "curation_reason": reason,
            "disease_key": "sle", "modality": "clinical_photo",
            "findings": [{"finding_key": "malar_rash", "evidence": "visual"}],
        }]}

    add_figure(conn, "PMC1:age", "PMC1", vision=vetoed(curation.RETIRED_AGE_REASON, "not_patient_image"))
    add_figure(conn, "PMC1:chart", "PMC1", vision=vetoed("chart or graph", "diagram"))
    conn.execute("UPDATE figures SET status='vision_rejected'")

    assert judge.requeue_age_vetoes(conn, dry_run=True)["requeued"] == 1
    assert conn.execute("SELECT status FROM figures WHERE figure_id='PMC1:age'").fetchone()[0] == "vision_rejected"

    result = judge.requeue_age_vetoes(conn)
    assert result == {"examined": 1, "requeued": 1, "still_rejected": {}}
    rows = {r["figure_id"]: r for r in conn.execute("SELECT * FROM figures")}
    assert rows["PMC1:age"]["status"] == "vision_accepted"
    panel = db.from_json(rows["PMC1:age"]["vision_json"])["panels"][0]
    assert panel["include"] is True and "curation_reason" not in panel
    assert rows["PMC1:chart"]["status"] == "vision_rejected"
    conn.close()
