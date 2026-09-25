"""Stage 8 viewer tests — FastAPI TestClient on the synthetic seed dataset."""

import importlib.util
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
    from src.visual_pilot.viewer import create_app

    with TestClient(create_app(str(data_dir))) as c:
        yield c


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    ["/", "/disease/sle", "/disease/dm", "/disease/as", "/compare/sle-dm-skin"],
)
def test_pages_200(client, path):
    assert client.get(path).status_code == 200


def test_api_diseases(client):
    data = client.get("/api/diseases").json()
    assert {d["key"] for d in data} == {"sle", "dm", "as"}
    assert all(d["subtypes"] for d in data)


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
    assert set(dup[0]["panel_ids"]) == {"PMC9000002:F2:pA", "PMC9000002:F2:pB"}


def test_findings_expose_key_label_evidence(client):
    panels = client.get("/api/diseases/sle/panels").json()["panels"]
    finding = next(f for p in panels for f in p["findings"] if f["key"] == "malar_rash")
    assert finding["label"]
    assert "evidence" in finding
