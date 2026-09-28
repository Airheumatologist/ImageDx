"""Stage 8 viewer tests — FastAPI TestClient on the synthetic seed dataset."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_seed_module():
    spec = importlib.util.spec_from_file_location(
        "vp_seed_synthetic", REPO_ROOT / "scripts" / "vp_seed_synthetic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vp_seed_synthetic"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def client(tmp_path):
    seed_mod = _load_seed_module()
    data_dir = tmp_path / "vp_synth"
    seed_mod.seed(data_dir)
    import sqlite3
    seeded_db = sqlite3.connect(data_dir / "visual_pilot.sqlite")
    # Synthetic fixtures represent preselected full-panel crops; provide the
    # bounds required by the same eligibility policy used for real records.
    seeded_db.execute(
        "UPDATE panels SET bbox_json='[0.05,0.05,0.95,0.95]' WHERE bbox_json IS NULL"
    )
    for figure_id, in seeded_db.execute("SELECT DISTINCT figure_id FROM panels"):
        findings = []
        for (raw,) in seeded_db.execute("SELECT findings_json FROM panels WHERE figure_id=?", (figure_id,)):
            for item in json.loads(raw or "[]"):
                key = item.get("finding_key") if isinstance(item, dict) else item
                label = seeded_db.execute(
                    "SELECT label FROM findings_vocab WHERE finding_key=?", (key,)
                ).fetchone()
                if label:
                    findings.append(label[0].split("(", 1)[0])
        if findings:
            seeded_db.execute(
                "UPDATE figures SET caption=caption || ' ' || ? WHERE figure_id=?",
                ("; ".join(dict.fromkeys(findings)), figure_id),
            )
    seeded_db.execute(
        "UPDATE figures SET caption=caption || CASE "
        "WHEN EXISTS (SELECT 1 FROM panels p WHERE p.figure_id=figures.figure_id AND p.disease_key='dm') THEN ' Dermatomyositis.' "
        "WHEN EXISTS (SELECT 1 FROM panels p WHERE p.figure_id=figures.figure_id AND p.disease_key='as') THEN ' Axial spondyloarthritis.' "
        "ELSE ' Systemic lupus erythematosus.' END"
    )
    seeded_db.commit()
    seeded_db.close()
    from src.visual_pilot.viewer import create_app

    with TestClient(create_app(str(data_dir))) as c:
        yield c


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "/", "/disease/sle", "/disease/dm", "/disease/as", "/disease/ra",
        "/disease/ssc", "/disease/psoriasis", "/disease/psa",
        "/disease/sarcoidosis", "/disease/gout", "/disease/ad",
        "/compare/sle-dm-skin",
    ],
)
def test_pages_200(client, path):
    assert client.get(path).status_code == 200


def test_api_diseases(client):
    data = client.get("/api/diseases").json()
    assert {d["key"] for d in data} == {
        "sle", "dm", "as", "ra", "ssc", "psoriasis", "psa",
        "sarcoidosis", "gout", "ad",
    }
    assert all(d["subtypes"] for d in data)


def test_psoriasis_eye_evidence_populates_tab_without_image(client):
    from src.visual_pilot import db

    conn = db.connect(client.app.state.data_dir / "visual_pilot.sqlite")
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO findings_vocab "
            "(finding_key, disease_keys_json, label, category, approved) "
            "VALUES ('anterior_uveitis', '[\"psoriasis\"]', 'Anterior uveitis', 'eye', 1)"
        )
        conn.execute(
            "INSERT INTO articles (pmcid, title, status) VALUES "
            "('PMC99990001', 'Psoriasis ocular review', 'parsed')"
        )
        conn.execute(
            "INSERT INTO disease_findings "
            "(disease_key, finding_key, source, pmcid, quote) VALUES "
            "('psoriasis', 'anterior_uveitis', 'text', 'PMC99990001', "
            "'Psoriasis is associated with uveitis.')"
        )
        conn.execute(
            "INSERT INTO disease_findings "
            "(disease_key, finding_key, source, pmcid, quote) VALUES "
            "('ssc', 'anterior_uveitis', 'text', 'PMC99990001', "
            "'This article also mentions systemic sclerosis.')"
        )
    conn.close()

    tabs = client.get("/api/diseases/psoriasis/tabs").json()
    eye = next(tab for tab in tabs if tab["key"] == "eye")
    assert eye["evidence_only"] is True
    assert not any(tab["key"] == "capillaroscopy" for tab in tabs)
    evidence = client.get("/api/diseases/psoriasis/eye-evidence").json()
    assert evidence[0]["label"] == "Uveitis"
    assert evidence[0]["article_url"].endswith("/PMC99990001/")
    assert client.get("/api/diseases/ssc/eye-evidence").json() == []


# ---------------------------------------------------------------------------
# Panels: filters, tabs, sort, attribution
# ---------------------------------------------------------------------------
def test_panels_all_have_attribution(client):
    for key in ("sle", "dm", "as"):
        data = client.get(f"/api/diseases/{key}/panels").json()
        assert data["count"] > 0
        for p in data["panels"]:
            assert p["attribution_text"], p["panel_id"]
            assert p["doi_url"] or p["attribution_text"]
            assert p["display_label"]
            assert p["article_title"]
            assert p["article_url"]
            assert p["copyright"] == p["attribution_text"]


def test_panels_expose_figure_label_caption(client):
    for key in ("sle", "dm", "as"):
        panels = client.get(f"/api/diseases/{key}/panels").json()["panels"]
        assert panels
        for p in panels:
            assert p["figure_label"], p["panel_id"]
            assert p["figure_caption"], p["panel_id"]
            assert p["caption_variants"], p["panel_id"]
            assert isinstance(p["in_text_mentions"], list)


def _fkeys(panel):
    return {f["key"] for f in panel["findings"]}


def test_subacute_caption_does_not_match_acute_group():
    from src.visual_pilot.viewer.app import _clinical_group

    assert _clinical_group({
        'figure_caption': 'Annular plaques in subacute cutaneous lupus erythematosus.',
        'findings': [],
    }) == 'SCLE'


def test_caption_label_keeps_depiction_without_trailing_age():
    from src.visual_pilot.viewer.app import _caption_label

    assert _caption_label(
        'Clinical example of a case with erythematous plaque on the right cheek '
        'of a 25-year-old female.'
    ) == 'Erythematous plaque on the right cheek'


def test_caption_label_removes_patient_and_diagnostic_context():
    from src.visual_pilot.viewer.app import _caption_label

    captions = [
        (
            'Subtle annular plaques on the left upper extremity of this female patient '
            'diagnosed with subacute cutaneous lupus erythematosus.'
        ),
        (
            "Coin-shaped erythematous plaques seen on this female patient's left cheek, "
            'biopsy results consistent with discoid lupus.'
        ),
    ]
    assert _caption_label(captions[0]) == "Subtle annular plaques on the left upper extremity"
    assert _caption_label(captions[1]) == "Coin-shaped erythematous plaques on the left cheek"
    assert all(len(_caption_label(caption)) <= 90 for caption in captions)


def test_panel_filters(client):
    data = client.get("/api/diseases/dm/panels", params={"modality": "mri"}).json()
    assert data["count"] >= 1
    assert all(p["modality"] == "mri" for p in data["panels"])

    data = client.get("/api/diseases/sle/panels", params={"finding": "malar_rash"}).json()
    assert all("malar_rash" in _fkeys(p) for p in data["panels"])

    data = client.get("/api/diseases/sle/panels", params={"skin_tone": "dark"}).json()
    assert all(p["skin_tone"] == "dark" for p in data["panels"])

    data = client.get("/api/diseases/dm/panels", params={"subtype": "jdm"}).json()
    assert all(p["subtype"] == "jdm" for p in data["panels"])

    data = client.get("/api/diseases/as/panels", params={"typicality": "classic"}).json()
    assert all(p["typicality"] == "classic" for p in data["panels"])


def test_sort_order_classic_variant_atypical_then_confidence(client):
    order = {"classic": 0, "variant": 1, "atypical": 2}
    for key in ("sle", "dm", "as"):
        panels = client.get(f"/api/diseases/{key}/panels").json()["panels"]
        keys = [(order.get(p["typicality"], 3), -(p["confidence"] or 0)) for p in panels]
        assert keys == sorted(keys)


def test_tab_assignment(client):
    panels = {p["panel_id"]: p for p in client.get("/api/diseases/sle/panels").json()["panels"]}
    by_finding = {}
    for p in panels.values():
        for f in _fkeys(p):
            by_finding.setdefault(f, p["tab"])
    assert by_finding["lupus_nephritis_class"] == "renal_histology"
    assert by_finding["interface_dermatitis"] == "skin_histology"
    assert by_finding["npsle_white_matter_lesions"] == "imaging"
    assert by_finding["tortuous_capillaries"] == "capillaroscopy"
    assert by_finding["oral_ulcer"] == "mucosa"
    assert by_finding["jaccoud_arthropathy"] == "musculoskeletal"
    assert by_finding["malar_rash"] == "skin"

    dm = client.get("/api/diseases/dm/panels").json()["panels"]
    gottron = [p["tab"] for p in dm if "gottron_papules" in _fkeys(p)]
    assert set(gottron) == {"skin"}
    calc = [p["tab"] for p in dm if "calcinosis_cutis" in _fkeys(p)]
    assert set(calc) == {"calcinosis"}

    as_ = client.get("/api/diseases/as/panels").json()["panels"]
    si_mri = [p["tab"] for p in as_ if "si_bone_marrow_edema" in _fkeys(p)]
    assert si_mri == ["si_mri"]
    eye = [p["tab"] for p in as_ if "anterior_uveitis" in _fkeys(p)]
    assert eye == ["eye"]


def test_sle_vascular_findings_ignore_model_subtype_and_histology_mucosa(client):
    import sqlite3

    db_path = client.app.state.data_dir / "visual_pilot.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE panels SET subtype='acle', findings_json='[{\"finding_key\":\"raynaud_phenomenon\",\"evidence\":\"caption says Raynaud phenomenon\"}]' "
        "WHERE panel_id=(SELECT panel_id FROM panels WHERE disease_key='sle' AND modality='clinical_photo' LIMIT 1)"
    )
    conn.execute(
        "UPDATE figures SET caption='Cutaneous vasculitis and Raynaud phenomenon in systemic lupus erythematosus.' "
        "WHERE figure_id=(SELECT figure_id FROM panels WHERE disease_key='sle' AND modality='clinical_photo' LIMIT 1)"
    )
    conn.commit()
    conn.close()

    panels = client.get("/api/diseases/sle/panels").json()["panels"]
    vascular = next(p for p in panels if "raynaud_phenomenon" in _fkeys(p))
    assert vascular["clinical_group"] == "Vascular findings"
    assert vascular["tab"] == "skin"
    assert vascular["clinical_tab"] == "skin"
    from src.visual_pilot.viewer.app import assign_tab, _categories
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    categories = _categories(conn)
    conn.close()
    histology_oral = {"modality": "histology_he", "body_site": "oral mucosa", "findings": [{"key": "oral_ulcer"}]}
    assert assign_tab(histology_oral, __import__("src.visual_pilot.viewer.app", fromlist=["TABS"]).TABS["sle"], categories) == "other"


def test_catalog_disease_without_custom_tabs_is_browsable(client):
    assert client.get("/disease/ra").status_code == 200
    tabs = client.get("/api/diseases/ra/tabs").json()
    panels = client.get("/api/diseases/ra/panels").json()
    assert panels["count"] == len(panels["panels"])
    represented = {p["tab"] for p in panels["panels"]}
    if any(p["pediatric"] for p in panels["panels"]):
        represented.add("pediatric")
    assert {tab["key"] for tab in tabs} == represented
    assert client.get("/api/diseases/not-in-catalog/tabs").status_code == 404

    psoriasis_tabs = client.get("/api/diseases/psoriasis/tabs").json()
    assert "capillaroscopy" not in {tab["key"] for tab in psoriasis_tabs}

    from src.visual_pilot.viewer.app import GENERIC_TABS, assign_tab
    categories = {"joint_swelling": "clinical_msk", "uveitis": "eye", "nailfold_capillaries": "capillaroscopy"}
    assert assign_tab({"modality": "clinical_photo", "findings": [{"key": "joint_swelling"}]}, GENERIC_TABS, categories) == "musculoskeletal"
    assert assign_tab({"modality": "clinical_photo", "findings": [{"key": "uveitis"}]}, GENERIC_TABS, categories) == "eye"
    # Modality wins when an automated finding tag belongs to another image
    # type, so histology and imaging stay in their own sections.
    skin_categories = {"plaque_psoriasis": "skin"}
    assert assign_tab(
        {"modality": "histology_he", "findings": [{"key": "plaque_psoriasis"}]},
        GENERIC_TABS,
        skin_categories,
    ) == "histology"
    assert assign_tab(
        {"modality": "ct", "findings": [{"key": "plaque_psoriasis"}]},
        GENERIC_TABS,
        skin_categories,
    ) == "imaging"
    assert assign_tab(
        {"modality": "ophthalmic", "findings": [{"key": "plaque_psoriasis"}]},
        GENERIC_TABS,
        skin_categories,
    ) == "eye"


def test_tabs_are_based_on_unfiltered_visible_panels(client):
    # A modality filter can empty a section temporarily, but should not remove
    # that section from the disease header.
    all_panels = client.get("/api/diseases/psoriasis/panels").json()["panels"]
    tabs = client.get("/api/diseases/psoriasis/tabs").json()
    assert {tab["key"] for tab in tabs} == {panel["tab"] for panel in all_panels} | (
        {"pediatric"} if any(panel["pediatric"] for panel in all_panels) else set()
    )
    client.get("/api/diseases/psoriasis/panels", params={"modality": "ophthalmic"})
    assert {tab["key"] for tab in client.get("/api/diseases/psoriasis/tabs").json()} == {
        panel["tab"] for panel in all_panels
    } | ({"pediatric"} if any(panel["pediatric"] for panel in all_panels) else set())


def test_caption_label_skips_demographic_lead_in_and_wrong_finding_label():
    from src.visual_pilot.viewer.app import _caption_label, _clinical_group, _finding_supported

    caption = (
        "An 18-year-old female with SLE, diagnosed at age 16, presents acutely. "
        "CT shows bilateral ground-glass opacities and consolidation."
    )
    assert _caption_label(caption).startswith("CT shows bilateral")
    tumid_caption = "Erythematous, edematous, urticarial plaque on the right cheek."
    malar = {"key": "malar_rash", "label": "Malar rash", "evidence": "plaque on cheek"}
    assert not _finding_supported(malar, tumid_caption.casefold(), ["malar rash", "malar_rash"])
    assert _clinical_group({"figure_caption": tumid_caption, "findings": []}) == "Other skin findings"


def test_skin_routing_uses_internal_tags_but_hides_unsupported_label(client):
    import sqlite3

    db_path = client.app.state.data_dir / "visual_pilot.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE figures SET caption='Annular plaques in subacute cutaneous lupus erythematosus.' "
        "WHERE figure_id=(SELECT figure_id FROM panels WHERE disease_key='sle' AND findings_json LIKE '%scle_annular%' LIMIT 1)"
    )
    conn.commit()
    conn.close()
    panels = client.get("/api/diseases/sle/panels").json()["panels"]
    panel = next(p for p in panels if "scle_annular" not in _fkeys(p) and "Annular plaques" in p["display_label"])
    assert panel["tab"] == "skin"
    assert panel["clinical_group"] == "SCLE"


def test_pediatric_tab_is_cross_cutting_and_uses_stored_age_group(client):
    import sqlite3

    db_path = client.app.state.data_dir / "visual_pilot.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE panels SET age_group='adolescent' WHERE panel_id=(SELECT panel_id FROM panels WHERE disease_key='sle' LIMIT 1)")
    conn.execute(
        "UPDATE figures SET caption=caption || ' Adolescent patient.' "
        "WHERE figure_id=(SELECT figure_id FROM panels WHERE disease_key='sle' LIMIT 1)"
    )
    conn.commit()
    conn.close()
    tabs = client.get("/api/diseases/sle/tabs").json()
    pediatric = next(t for t in tabs if t["key"] == "pediatric")
    assert pediatric["cross_cutting"] is True
    panel = client.get("/api/diseases/sle/panels").json()["panels"][0]
    assert panel["pediatric"] is True
    assert panel["age_group_label"] == "Adolescent"
    assert panel["clinical_tab"] == panel["tab"]


# ---------------------------------------------------------------------------
# Unapproved findings hidden
# ---------------------------------------------------------------------------
def test_proposed_findings_hidden(client):
    vocab = client.get("/api/vocab", params={"disease": "sle"}).json()
    assert "proposed_butterfly_flush" not in {v["finding_key"] for v in vocab}
    for p in client.get("/api/diseases/sle/panels").json()["panels"]:
        assert "proposed_butterfly_flush" not in _fkeys(p)
    # filtering by a proposed finding returns nothing
    data = client.get(
        "/api/diseases/sle/panels", params={"finding": "proposed_butterfly_flush"}
    ).json()
    assert data["panels"] == []


# ---------------------------------------------------------------------------
# Media + findings endpoints
# ---------------------------------------------------------------------------
def test_media_serves_and_blocks(client):
    panels = client.get("/api/diseases/sle/panels").json()["panels"]
    img = panels[0]["image"]
    assert client.get(img).status_code == 200
    # outside allowed prefixes
    assert client.get("/media/visual_pilot.sqlite").status_code == 403
    # traversal attempts
    assert client.get("/media/panels/../../visual_pilot.sqlite").status_code in {403, 404}
    assert client.get("/media/../visual_pilot.sqlite").status_code in {403, 404}
    assert client.get("/media/panels/does_not_exist.png").status_code == 404


def test_key_findings(client):
    rows = client.get("/api/diseases/dm/findings").json()
    assert rows
    assert all(r["quote"] for r in rows)
    freqs = [r["pct_high"] or -1 for r in rows]
    assert freqs == sorted(freqs, reverse=True)


def test_key_findings_approved_only_and_deduped(client):
    rows = client.get("/api/diseases/sle/findings").json()
    keys = {r["finding_key"] for r in rows}
    assert "proposed_butterfly_flush" not in keys
    assert all(r["label"] for r in rows)
    # deduped by (finding_key, pmcid)
    pairs = [(r["finding_key"], r["pmcid"]) for r in rows]
    assert len(pairs) == len(set(pairs))


def test_compare_endpoint(client):
    pairs = client.get("/api/compare/sle-dm-skin").json()
    assert len(pairs) == 3
    first = pairs[0]
    assert first["left"]["panels"], "expected DM Gottron panels"
    assert first["right"]["panels"], "expected SLE hand panels"
    assert all(p["disease_key"] == "dm" for p in first["left"]["panels"])
    assert all(p["disease_key"] == "sle" for p in first["right"]["panels"])
    heliotrope = pairs[1]
    assert all("heliotrope_rash" in _fkeys(p) for p in heliotrope["left"]["panels"])
    assert all("malar_rash" in _fkeys(p) for p in heliotrope["right"]["panels"])


def test_duplicate_sha_panels_collapse_to_one_card(client):
    panels = client.get("/api/diseases/dm/panels").json()["panels"]
    dup = [p for p in panels if "shared_gottron" in (p["image"] or "")]
    assert len(dup) == 1, "two panel rows sharing a sha256 render as one card"
    assert len(dup[0]["attribution_variants"]) == 2
    assert len(dup[0]["source_variants"]) == 2
    assert set(dup[0]["panel_ids"]) == {"PMC9000002:F2:pA", "PMC9000002:F2:pB"}


def test_findings_expose_key_label_evidence(client):
    panels = client.get("/api/diseases/sle/panels").json()["panels"]
    finding = next(f for p in panels for f in p["findings"] if f["key"] == "malar_rash")
    assert finding["label"]
    assert "evidence" in finding
