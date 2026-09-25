#!/usr/bin/env python3
"""Seed a synthetic Visual Findings Library dataset for viewer dev/testing.

Writes articles/figures/panels/disease_findings rows plus generated PNG
images into a chosen VP_DATA_DIR. Refuses to write the default data dir
unless --force is given.

Usage:
    python3 scripts/vp_seed_synthetic.py /tmp/vp_synth
    VP_DATA_DIR=/tmp/vp_synth python3 scripts/vp_seed_synthetic.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from src.visual_pilot import config, db, diseases  # noqa: E402

COLOR = {
    "sle": (168, 68, 110),
    "dm": (54, 105, 168),
    "as": (64, 138, 90),
    "proposed": (120, 120, 120),
}

PANELS = [
    # ---- SLE --------------------------------------------------------------
    ("sle", "acle", "clinical_photo", "face", ["malar_rash"], "classic", None, "light", 0.95),
    ("sle", "acle", "clinical_photo", "hands", ["periungual_erythema"], "classic", None, "medium", 0.9),
    ("sle", "scle", "clinical_photo", "back", ["scle_annular"], "variant", None, "medium", 0.8),
    ("sle", "dle", "clinical_photo", "scalp", ["discoid_plaque", "scarring_alopecia"], "atypical", None, "dark", 0.6),
    ("sle", "acle", "clinical_photo", "oral mucosa", ["oral_ulcer"], "classic", None, "light", 0.9),
    ("sle", "npsle", "clinical_photo", "hands", ["jaccoud_arthropathy"], "classic", None, "light", 0.85),
    ("sle", "lupus_nephritis", "histology_he", "kidney", ["lupus_nephritis_class"], "classic", None, None, 0.9),
    ("sle", "lupus_nephritis", "histology_ihc", "kidney", ["wire_loop_lesion"], "variant", None, None, 0.75),
    ("sle", "dle", "immunofluorescence", "skin", ["lupus_band"], "classic", None, None, 0.88),
    ("sle", "dle", "histology_he", "skin", ["interface_dermatitis", "dermal_mucin"], "classic", None, None, 0.8),
    ("sle", "npsle", "mri", "brain", ["npsle_white_matter_lesions"], "variant", None, None, 0.7),
    ("sle", "lupus_nephritis", "radiograph", "chest", ["pleural_effusion"], "classic", None, None, 0.65),
    ("sle", "acle", "capillaroscopy", "nailfold", ["tortuous_capillaries"], "classic", None, None, 0.8),
    # ---- DM ---------------------------------------------------------------
    ("dm", "classic", "clinical_photo", "hands", ["gottron_papules"], "classic", None, "light", 0.96),
    ("dm", "classic", "clinical_photo", "eyelids", ["heliotrope_rash"], "classic", None, "medium", 0.93),
    ("dm", "classic", "clinical_photo", "chest", ["v_sign"], "classic", None, "light", 0.85),
    ("dm", "classic", "clinical_photo", "upper back", ["shawl_sign"], "variant", None, "light", 0.82),
    ("dm", "classic", "clinical_photo", "thigh", ["holster_sign"], "variant", None, "dark", 0.7),
    ("dm", "classic", "clinical_photo", "hands", ["mechanics_hands"], "variant", None, "medium", 0.75),
    ("dm", "cadm", "clinical_photo", "hands", ["gottron_sign"], "classic", None, "light", 0.8),
    ("dm", "jdm", "clinical_photo", "hands", ["gottron_papules"], "classic", None, "light", 0.9),
    ("dm", "anti_mda5", "clinical_photo", "palms", ["mda5_palmar_papules", "mda5_cutaneous_ulcers"], "atypical", None, "medium", 0.72),
    ("dm", "classic", "clinical_photo", "elbow", ["calcinosis_cutis"], "atypical", None, "light", 0.6),
    ("dm", "classic", "radiograph", "elbow", ["calcinosis_radiograph"], "classic", None, None, 0.7),
    ("dm", "classic", "capillaroscopy", "nailfold", ["dilated_giant_capillaries", "capillary_dropout", "capillary_hemorrhages"], "classic", None, None, 0.9),
    ("dm", "classic", "clinical_photo", "nailfold", ["periungual_erythema", "ragged_cuticles"], "classic", None, "light", 0.85),
    ("dm", "classic", "histology_he", "muscle", ["perifascicular_atrophy", "perivascular_inflammation"], "classic", None, None, 0.92),
    ("dm", "classic", "mri", "thigh", ["muscle_edema_stir", "fascial_edema"], "classic", None, None, 0.8),
    ("dm", "anti_mda5", "ct", "lung", ["ild_nsip_pattern", "rapidly_progressive_ild"], "variant", None, None, 0.78),
    # ---- AS ---------------------------------------------------------------
    ("as", "r_axspa", "radiograph", "sacroiliac", ["sacroiliitis", "si_erosions"], "classic", "early", None, 0.9),
    ("as", "r_axspa", "radiograph", "sacroiliac", ["si_sclerosis", "si_ankylosis"], "classic", "advanced", None, 0.88),
    ("as", "nr_axspa", "mri", "sacroiliac", ["si_bone_marrow_edema"], "classic", "nr_axspa", None, 0.92),
    ("as", "r_axspa", "mri", "sacroiliac", ["fat_metaplasia", "backfill"], "variant", "early", None, 0.75),
    ("as", "r_axspa", "radiograph", "spine", ["syndesmophytes", "vertebral_squaring"], "classic", "early", None, 0.85),
    ("as", "r_axspa", "ct", "spine", ["bamboo_spine", "dagger_sign"], "classic", "advanced", None, 0.9),
    ("as", "r_axspa", "clinical_photo", "spine", ["hyperkyphosis", "loss_of_lumbar_lordosis"], "classic", "advanced", None, 0.8),
    ("as", "nr_axspa", "clinical_photo", "heel", ["achilles_enthesitis"], "variant", None, None, 0.65),
    ("as", "r_axspa", "ophthalmic", "eye", ["anterior_uveitis"], "classic", None, None, 0.82),
    # panel tagged only with an unapproved (LLM-proposed) finding — hidden
    ("sle", "acle", "clinical_photo", "face", ["proposed_butterfly_flush"], "classic", None, "light", 0.5),
]

ARTICLES = {
    "sle": ("PMC9000001", "Cutaneous and systemic lupus: a clinical review", "Lupus Rev", 2024),
    "dm": ("PMC9000002", "Dermatomyositis skin signs and myopathy: a review", "Derm Rev", 2023),
    "as": ("PMC9000003", "Axial spondyloarthritis imaging and clinical signs", "Rheum Imaging Rev", 2024),
}

FINDINGS_ROWS = [
    ("sle", "malar_rash", "acle", "30–60%", 30, 60, "Malar rash occurs in 30–60% of SLE patients."),
    ("sle", "lupus_nephritis_class", "lupus_nephritis", "up to 60%", None, 60, "Renal involvement affects up to 60% of patients."),
    ("sle", "oral_ulcer", None, "7–52%", 7, 52, "Oral ulcers are reported in 7–52% of cohorts."),
    ("dm", "gottron_papules", "classic", "70–80%", 70, 80, "Gottron papules are seen in most patients."),
    ("dm", "heliotrope_rash", None, "pathognomonic", None, None, "The heliotrope rash is pathognomonic."),
    ("dm", "rapidly_progressive_ild", "anti_mda5", "frequent in anti-MDA5", None, None, "Anti-MDA5 disease carries rapidly progressive ILD."),
    ("as", "sacroiliitis", "r_axspa", "defining feature", None, None, "Radiographic sacroiliitis defines r-axSpA."),
    ("as", "si_bone_marrow_edema", "nr_axspa", "common", None, None, "MRI bone marrow edema is frequent in nr-axSpA."),
    ("as", "anterior_uveitis", None, "25–30%", 25, 30, "Anterior uveitis complicates 25–30% of cases."),
]


def _png(path: Path, text: str, color: tuple[int, int, int], size=(640, 420)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, size[0] - 8, size[1] - 8], outline=(255, 255, 255), width=3)
    draw.text((20, size[1] // 2 - 10), text, fill=(255, 255, 255))
    img.save(path)


def seed(data_dir: Path) -> dict[str, int]:
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = db.init_db(db.connect(data_dir / config.DB_FILENAME))
    diseases.seed(conn)

    for key, (pmcid, title, journal, year) in ARTICLES.items():
        conn.execute(
            "INSERT OR REPLACE INTO articles (pmcid, pmid, doi, title, journal, year, "
            "license_code, license_url, oa_subset, retrieval_score, "
            "primary_disease_keys_json, status, relevance_decision) "
            "VALUES (?, '99000001', '10.1234/synth.' || ?, ?, ?, ?, 'cc-by', "
            "'https://creativecommons.org/licenses/by/4.0/', 'oa', 1.0, ?, 'parsed', 'relevant')",
            (pmcid, key, title, journal, year, db.to_json([key])),
        )
        conn.execute(
            "INSERT OR REPLACE INTO figures (figure_id, pmcid, label, caption, "
            "in_text_mentions_json, status) "
            "VALUES (?, ?, 'Figure 1', ?, ?, 'vision_accepted')",
            (
                f"{pmcid}:F1",
                pmcid,
                f"Representative clinical and imaging findings in {key}. Synthetic "
                "caption describing what the panels show.",
                db.to_json([f"Synthetic in-text mention of Figure 1 in {key}."]),
            ),
        )

    # One unapproved (LLM-proposed) vocab row — must never be shown.
    conn.execute(
        "INSERT OR REPLACE INTO findings_vocab (finding_key, disease_keys_json, label, "
        "category, approved, proposed_by_llm, proposal_count) "
        "VALUES ('proposed_butterfly_flush', '[\"sle\"]', 'butterfly flush (proposed)', "
        "'skin', 0, 1, 3)"
    )

    import hashlib

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _insert_panel(conn, panel_id, figure_id, pmcid, dk, subtype, modality,
                      body_site, findings, typicality, stage, skin_tone, conf,
                      rel_img, rel_thumb, sha, attrib):
        conn.execute(
            "INSERT OR REPLACE INTO panels (panel_id, figure_id, pmcid, panel_label, "
            "disease_key, subtype, modality, body_site, findings_json, typicality, "
            "stage, age_group, skin_tone, confidence, image_path, thumb_path, "
            "sha256, attribution_text, license_code, license_url, source_url, "
            "study_region, width, height) "
            "VALUES (?, ?, ?, 'A', ?, ?, ?, ?, ?, ?, ?, 'adult', ?, ?, ?, ?, ?, ?, "
            "'cc-by', 'https://creativecommons.org/licenses/by/4.0/', ?, "
            "'Germany (xml_corresp_aff)', 640, 420)",
            (
                panel_id, figure_id, pmcid, dk, subtype, modality, body_site,
                db.to_json(findings), typicality, stage, skin_tone, conf,
                rel_img, rel_thumb, sha, attrib,
                f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
            ),
        )

    for i, (dk, subtype, modality, body_site, findings, typicality, stage, skin_tone, conf) in enumerate(PANELS, start=1):
        pmcid = ARTICLES[dk][0]
        figure_id = f"{pmcid}:F1"
        panel_id = f"{pmcid}:F1:p{i}"
        rel_img = f"panels/{panel_id.replace(':', '_')}.png"
        rel_thumb = f"thumbs/{panel_id.replace(':', '_')}.webp"
        _png(data_dir / rel_img, f"{dk} {findings[0]}", COLOR[dk])
        _png(data_dir / rel_thumb, f"{dk} {findings[0]}", COLOR[dk], size=(320, 210))
        attrib = (
            f"Doe et al. {ARTICLES[dk][1]}. {ARTICLES[dk][2]} {ARTICLES[dk][3]}. "
            "doi:10.1234/synth.x. CC BY 4.0 "
            "(https://creativecommons.org/licenses/by/4.0/). Figure 1A."
        )
        _insert_panel(
            conn, panel_id, figure_id, pmcid, dk, subtype, modality, body_site,
            [{"finding_key": f, "evidence": f"shown on {body_site}"} for f in findings],
            typicality, stage, skin_tone, conf, rel_img, rel_thumb,
            _sha(data_dir / rel_img), attrib,
        )

    # Duplicate-image pair: two panel rows share one file/sha256 but carry
    # their own attributions (mirrors store-time dedup reuse).
    pmcid = ARTICLES["dm"][0]
    conn.execute(
        "INSERT OR REPLACE INTO figures (figure_id, pmcid, label, caption, status) "
        "VALUES (?, ?, 'Figure 2', 'Gottron papules over the knuckles. Synthetic "
        "caption for the shared-image figure.', 'vision_accepted')",
        (f"{pmcid}:F2", pmcid),
    )
    dup_img = "panels/dm/shared_gottron.png"
    dup_thumb = "thumbs/shared_gottron.webp"
    _png(data_dir / dup_img, "dm shared gottron", COLOR["dm"])
    _png(data_dir / dup_thumb, "dm shared gottron", COLOR["dm"], size=(320, 210))
    dup_sha = _sha(data_dir / dup_img)
    for suffix, tail in (("pA", "Figure 2A."), ("pB", "Figure 2B.")):
        _insert_panel(
            conn, f"{pmcid}:F2:{suffix}", f"{pmcid}:F2", pmcid, "dm", "classic",
            "clinical_photo", "hands",
            [{"finding_key": "gottron_papules", "evidence": "duplicate panel"}],
            "classic", None, "light", 0.7, dup_img, dup_thumb, dup_sha,
            f"Doe et al. {ARTICLES['dm'][1]}. {ARTICLES['dm'][2]} {ARTICLES['dm'][3]}. "
            "doi:10.1234/synth.dm. CC BY 4.0 "
            f"(https://creativecommons.org/licenses/by/4.0/). {tail}",
        )

    for dk, finding, subtype, freq, lo, hi, quote in FINDINGS_ROWS:
        conn.execute(
            "INSERT INTO disease_findings (disease_key, finding_key, subtype, "
            "frequency_text, frequency_pct_low, frequency_pct_high, source, pmcid, quote) "
            "VALUES (?, ?, ?, ?, ?, ?, 'synthetic', ?, ?)",
            (dk, finding, subtype, freq, lo, hi, ARTICLES[dk][0], quote),
        )
    # A proposed (unapproved) finding row — must never be shown by the
    # key-findings endpoint until approved.
    conn.execute(
        "INSERT INTO disease_findings (disease_key, finding_key, subtype, "
        "frequency_text, frequency_pct_high, source, pmcid, quote) "
        "VALUES ('sle', 'proposed_butterfly_flush', NULL, '99%', 99.0, 'text', "
        "'PMC9000001', 'proposed quote')"
    )
    conn.commit()
    counts = {
        "articles": conn.execute("SELECT COUNT(*) n FROM articles").fetchone()["n"],
        "figures": conn.execute("SELECT COUNT(*) n FROM figures").fetchone()["n"],
        "panels": conn.execute("SELECT COUNT(*) n FROM panels").fetchone()["n"],
        "disease_findings": conn.execute("SELECT COUNT(*) n FROM disease_findings").fetchone()["n"],
    }
    conn.close()
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default=None, help="target VP_DATA_DIR")
    parser.add_argument("--force", action="store_true", help="allow writing the default data dir")
    args = parser.parse_args(argv)

    target = Path(args.data_dir or config.data_dir()).expanduser().resolve()
    default = config.DEFAULT_DATA_DIR.resolve()
    if target == default and not args.force:
        print(
            f"refusing to write the default data dir {default} without --force",
            file=sys.stderr,
        )
        return 2
    counts = seed(target)
    print(f"seeded synthetic data in {target}")
    print("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
