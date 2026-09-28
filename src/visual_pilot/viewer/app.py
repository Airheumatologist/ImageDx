"""Stage 8: FastAPI viewer for the Visual Findings Library pilot.

Pages are plain HTML/JS (no build step) served from ``viewer/static``; JSON
comes from ``/api/...``; images come from ``/media/...`` which only serves
paths under panels/, thumbs/ and figures/ inside the pilot data dir.

Tab membership lives in ``TABS`` below: an ordered per-disease mapping from
modality / body_site / finding keys / finding categories to a tab, evaluated
first-match-wins. Sorting everywhere is classic < variant < atypical, then
confidence descending. Unapproved (LLM-proposed) findings are filtered out of
every response.
"""

from __future__ import annotations

import sqlite3
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .. import config, db, representatives

STATIC_DIR = Path(__file__).resolve().parent / "static"
MEDIA_PREFIXES = {"panels", "thumbs", "figures"}

TYPICALITY_ORDER = {"classic": 0, "variant": 1, "atypical": 2}

# ---------------------------------------------------------------------------
# Tab membership (§5 stage 8). Each tab's match is a list of alternatives
# ("any_of"): within one alternative every declared key must hit (AND);
# alternatives are OR'd. ``not_*`` exclusions apply to all alternatives.
# Tabs are evaluated in display order, except tabs carrying ``priority`` are
# evaluated before non-priority ones; first match wins, unmatched panels land
# in a trailing "other" tab.
# ---------------------------------------------------------------------------
TABS: dict[str, list[dict]] = {
    "sle": [
        {
            "key": "skin",
            "label": "Skin",
            "skin_tone_filter": True,
            # Clinical grouping is derived from approved finding keys below;
            # panel subtype is model metadata and can mislabel vascular disease.
            "group_by": "clinical_group",
            "group_order": ["ACLE", "SCLE", "DLE", "Vascular findings", "Other skin findings"],
            "match": {
                "any_of": [
                    {
                        "modalities": {"clinical_photo", "dermoscopy"},
                        "categories": {"skin", "nail"},
                    }
                ],
                "not_categories": {"mucosa"},
                "not_body_sites": {"oral mucosa", "mouth", "lips", "tongue"},
            },
        },
        {
            "key": "mucosa",
            "label": "Mucosa",
            "match": {
                "any_of": [
                    {"modalities": {"clinical_photo", "dermoscopy"}, "categories": {"mucosa"}},
                    {
                        "modalities": {"clinical_photo", "dermoscopy"},
                        "body_sites": {"oral mucosa", "mouth", "lips", "tongue", "nasal"},
                    },
                ]
            },
        },
        {
            "key": "musculoskeletal",
            "label": "Musculoskeletal",
            "match": {
                "any_of": [
                    {"modalities": {"clinical_photo"}, "categories": {"clinical_msk"}}
                ]
            },
        },
        {
            "key": "renal_histology",
            "label": "Renal histology",
            "match": {
                "any_of": [
                    {
                        "modalities": {"histology_he", "histology_ihc", "immunofluorescence"},
                        "findings": {
                            "lupus_nephritis_class",
                            "wire_loop_lesion",
                            "full_house_immunofluorescence",
                        },
                    },
                    {
                        "modalities": {"histology_he", "histology_ihc", "immunofluorescence"},
                        "body_sites": {"kidney", "renal"},
                    },
                ]
            },
        },
        {
            "key": "skin_histology",
            "label": "Skin histology / DIF",
            "match": {
                "any_of": [
                    {"modalities": {"histology_he", "histology_ihc", "immunofluorescence"}}
                ],
                "not_body_sites": {"oral mucosa", "mouth", "lips", "tongue", "nasal"},
            },
        },
        {
            "key": "imaging",
            "label": "Imaging",
            "match": {
                "any_of": [
                    {"modalities": {"radiograph", "ct", "mri", "ultrasound", "echo", "pet"}}
                ]
            },
        },
        {
            "key": "capillaroscopy",
            "label": "Capillaroscopy",
            "match": {
                "any_of": [
                    {"modalities": {"capillaroscopy"}},
                    {"categories": {"capillaroscopy"}},
                ]
            },
        },
    ],
    "dm": [
        {
            "key": "skin",
            "label": "Skin",
            "skin_tone_filter": True,
            "group_by": "finding",
            "match": {"any_of": [{"modalities": {"clinical_photo", "dermoscopy"}}]},
        },
        {
            "key": "nailfold",
            "label": "Nailfold / Capillaroscopy",
            "priority": 0,
            "match": {
                "any_of": [
                    {"modalities": {"capillaroscopy"}},
                    {"categories": {"capillaroscopy", "nail"}},
                    {"body_sites": {"nailfold", "nail", "periungual"}},
                ]
            },
        },
        {
            "key": "muscle_histology",
            "label": "Muscle histology",
            "match": {
                "any_of": [
                    {"modalities": {"histology_he", "histology_ihc", "immunofluorescence"}},
                    {"categories": {"histology"}},
                ]
            },
        },
        {
            "key": "mri",
            "label": "MRI",
            "match": {"any_of": [{"modalities": {"mri"}}]},
        },
        {
            "key": "lung_ct",
            "label": "Lung CT",
            "match": {
                "any_of": [
                    {"modalities": {"ct"}},
                    {"body_sites": {"lung", "chest", "thorax"}},
                ]
            },
        },
        {
            "key": "calcinosis",
            "label": "Calcinosis",
            "priority": 0,
            "match": {
                "any_of": [{"findings": {"calcinosis_cutis", "calcinosis_radiograph"}}]
            },
        },
    ],
    "as": [
        {
            "key": "si_radiograph",
            "label": "SI radiograph",
            "group_by": "stage",
            "group_order": ["nr_axspa", "nr-axSpA", "early", "advanced"],
            "match": {
                "any_of": [
                    {
                        "modalities": {"radiograph"},
                        "body_sites": {"sacroiliac", "si joint", "sij", "pelvis"},
                    },
                    {
                        "modalities": {"radiograph"},
                        "findings": {"sacroiliitis", "si_erosions", "si_sclerosis", "si_ankylosis"},
                    },
                ]
            },
        },
        {
            "key": "si_mri",
            "label": "SI MRI",
            "group_by": "stage",
            "group_order": ["nr_axspa", "nr-axSpA", "early", "advanced"],
            "match": {
                "any_of": [
                    {
                        "modalities": {"mri"},
                        "body_sites": {"sacroiliac", "si joint", "sij", "pelvis"},
                    },
                    {
                        "modalities": {"mri"},
                        "findings": {"si_bone_marrow_edema", "fat_metaplasia", "backfill"},
                    },
                ]
            },
        },
        {
            "key": "spine",
            "label": "Spine radiograph / CT",
            "group_by": "stage",
            "group_order": ["nr_axspa", "nr-axSpA", "early", "advanced"],
            "match": {
                "any_of": [
                    {"modalities": {"radiograph", "ct"}},
                    {
                        "modalities": {"mri"},
                        "findings": {"corner_inflammatory_lesion", "corner_fat_lesion", "romanus_lesion"},
                    },
                ]
            },
        },
        {
            "key": "clinical",
            "label": "Clinical",
            "match": {
                "any_of": [
                    {"modalities": {"clinical_photo"}},
                    {"categories": {"clinical_msk"}},
                ]
            },
        },
        {
            "key": "eye",
            "label": "Eye",
            "match": {
                "any_of": [
                    {"modalities": {"ophthalmic"}},
                    {"categories": {"eye"}},
                ]
            },
        },
    ],
}

# New catalog diseases share this broad modality/category routing until they
# receive a disease-specific layout. It keeps their images browsable while
# allowing approved findings and source metadata to remain data-driven.
GENERIC_TABS: list[dict] = [
    {
        "key": "mucosa",
        "label": "Mucosa",
        "priority": 0,
        "match": {"any_of": [{"categories": {"mucosa"}}]},
    },
    {
        "key": "musculoskeletal",
        "label": "Musculoskeletal",
        "priority": 0,
        "match": {"any_of": [{"categories": {"clinical_msk"}}]},
    },
    {
        "key": "eye",
        "label": "Eye",
        "modality_priority": {"ophthalmic"},
        "priority": 0,
        "match": {
            "any_of": [
                {"modalities": {"ophthalmic"}},
                {"categories": {"eye"}},
            ]
        },
    },
    {
        "key": "capillaroscopy",
        "label": "Capillaroscopy",
        "priority": 0,
        "match": {
            "any_of": [
                {"modalities": {"capillaroscopy"}},
                {"categories": {"capillaroscopy"}},
            ]
        },
    },
    {
        "key": "skin",
        "label": "Skin / nail",
        "skin_tone_filter": True,
        "match": {
            "any_of": [
                {"modalities": {"clinical_photo", "dermoscopy"}},
                {"categories": {"skin", "nail"}},
            ]
        },
    },
    {
        "key": "histology",
        "label": "Histology",
        "modality_priority": {"histology_he", "histology_ihc", "immunofluorescence"},
        "match": {
            "any_of": [
                {"modalities": {"histology_he", "histology_ihc", "immunofluorescence"}},
                {"categories": {"histology"}},
            ]
        },
    },
    {
        "key": "imaging",
        "label": "Imaging",
        "modality_priority": {"radiograph", "ct", "mri", "ultrasound", "echo", "pet"},
        "match": {
            "any_of": [
                {"modalities": {"radiograph", "ct", "mri", "ultrasound", "echo", "pet"}},
                {"categories": {"radiology_xray", "ct", "mri", "us", "echo"}},
            ]
        },
    },
]


def _tabs_for(key: str) -> list[dict]:
    return TABS.get(key, GENERIC_TABS)


def _known_disease(conn, key: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM diseases WHERE disease_key = ?", (key,)
    ).fetchone() is not None


def _eye_evidence(conn, disease: str) -> list[dict]:
    """Approved, cited eye findings when the image library has no eye panel."""
    rows = conn.execute(
        "SELECT fv.finding_key, fv.label, fv.disease_keys_json, df.pmcid, df.quote, a.title "
        "FROM disease_findings df "
        "JOIN findings_vocab fv ON fv.finding_key=df.finding_key "
        "JOIN articles a ON a.pmcid=df.pmcid "
        "WHERE df.disease_key=? AND df.source='text' AND fv.approved=1 "
        "AND fv.category='eye' AND a.status='parsed' "
        "AND COALESCE(TRIM(df.quote), '') != '' "
        "ORDER BY CASE WHEN lower(df.quote) LIKE '%psoriasis%' THEN 0 ELSE 1 END, "
        "a.year DESC, df.id", (disease,),
    )
    seen: set[tuple[str, str]] = set()
    result = []
    for row in rows:
        if disease not in db.from_json(row["disease_keys_json"], []):
            continue
        pair = (row["finding_key"], row["pmcid"])
        if pair in seen:
            continue
        seen.add(pair)
        label = row["label"]
        # Psoriasis review passages usually state "uveitis" without a
        # subtype, so the evidence card must not imply anterior disease.
        if disease in {"psoriasis", "psa"} and row["finding_key"] == "anterior_uveitis":
            label = "Uveitis"
        result.append({
            "finding_key": row["finding_key"], "label": label,
            "pmcid": row["pmcid"], "article_title": row["title"],
            "quote": row["quote"],
            "article_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{row['pmcid']}/",
        })
        if len(result) >= 12:
            break
    return result

# Pediatric is a cross-cutting view: the frontend filters explicit age_group
# values while each returned panel also keeps its regular clinical_tab/group.
for _tabs in TABS.values():
    _tabs.append({"key": "pediatric", "label": "Pediatric", "cross_cutting": True})
GENERIC_TABS.append({"key": "pediatric", "label": "Pediatric", "cross_cutting": True})

# Side-by-side comparisons for /compare/sle-dm-skin (§5 stage 8).
COMPARISONS = [
    {
        "title": "Gottron papules vs SLE hand/knuckle-sparing rash",
        "left": {"disease": "dm", "finding": "gottron_papules", "label": "DM: Gottron papules"},
        "right": {
            "disease": "sle",
            "body_sites": {"hand", "hands", "knuckle", "periungual", "fingers", "dorsal hands"},
            "modalities": {"clinical_photo"},
            "label": "SLE: hand/periungual rash",
        },
    },
    {
        "title": "Heliotrope rash vs malar rash",
        "left": {"disease": "dm", "finding": "heliotrope_rash", "label": "DM: heliotrope rash"},
        "right": {"disease": "sle", "finding": "malar_rash", "label": "SLE: malar rash"},
    },
    {
        "title": "V/shawl sign vs SLE photosensitive rash",
        "left": {
            "disease": "dm",
            "findings_any": {"v_sign", "shawl_sign"},
            "label": "DM: V/shawl sign",
        },
        "right": {
            "disease": "sle",
            "subtype_in": {"acle", "scle"},
            "modalities": {"clinical_photo"},
            "label": "SLE: ACLE/SCLE photosensitive rash",
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(value) -> str:
    return str(value or "").strip().lower()


def _approved_keys(conn, disease: str | None = None) -> set[str]:
    rows = conn.execute("SELECT finding_key, disease_keys_json FROM findings_vocab WHERE approved=1")
    return {
        r["finding_key"] for r in rows
        if disease is None or disease in db.from_json(r["disease_keys_json"], [])
    }


def _categories(conn) -> dict[str, str]:
    return {r["finding_key"]: r["category"] for r in conn.execute("SELECT finding_key, category FROM findings_vocab")}


def _sort_panels(panels: list[dict]) -> list[dict]:
    return sorted(
        panels,
        key=lambda p: (
            TYPICALITY_ORDER.get(_norm(p.get("typicality")), 3),
            -(p.get("confidence") or 0.0),
        ),
    )


def _finding_keys(panel: dict) -> set[str]:
    """Finding keys from either findings_json shape (str or dict)."""
    keys: set[str] = set()
    for f in panel.get("findings") or []:
        keys.add(f["key"] if isinstance(f, dict) else str(f))
    return keys


def _routing_finding_keys(panel: dict) -> set[str]:
    """Private modality-compatible approved keys used only for tab routing."""
    keys = panel.get("_routing_findings")
    return {_norm(k) for k in keys} if keys is not None else _finding_keys(panel)


def _panel_json(
    row: sqlite3.Row,
    approved: set[str],
    labels: dict[str, str],
    categories: dict[str, str],
    terms: dict[str, list[str]],
) -> dict:
    raw = db.from_json(row["findings_json"], [])
    findings = []
    routing_findings = []
    modality = _norm(row["modality"])
    body_site = _norm(row["body_site"])
    for f in raw:
        key = f.get("finding_key") if isinstance(f, dict) else f
        if key and key in approved:
            category = _norm(categories.get(key))
            # Reject clear modality/site mismatches from automated visual tags.
            if modality == "ophthalmic" and category != "eye":
                continue
            if modality in {"histology_he", "histology_ihc", "immunofluorescence"}:
                if body_site in {"kidney", "renal"} and category != "histology":
                    continue
            if modality in {"mri", "ct", "radiograph", "ultrasound", "echo", "pet"} and category in {"skin", "nail", "mucosa", "clinical_msk"}:
                continue
            routing_findings.append(key)
            findings.append(
                {
                    "key": key,
                    "label": labels.get(key, key),
                    "evidence": f.get("evidence", "") if isinstance(f, dict) else "",
                }
            )
    thumb = row["thumb_path"] or row["image_path"]
    mentions = db.from_json(row["in_text_mentions_json"], [])
    caption = (row["figure_caption"] or "").strip()
    source_text = " ".join(" ".join([caption, *[str(m) for m in mentions]]).casefold().split())
    for f in findings:
        f["source_supported"] = _finding_supported(f, source_text, terms.get(f["key"], []))
    supported = [f for f in findings if f["source_supported"]]
    evidence = [
        f["evidence"].strip() for f in supported
        if f.get("evidence", "").strip() and f["evidence"].strip().lower() != "visual"
        and " ".join(f["evidence"].casefold().split()) in source_text
    ]
    shown_label = ", ".join(dict.fromkeys(f["label"] for f in supported))
    context = "; ".join(dict.fromkeys(evidence)) or caption
    if not context and mentions:
        context = str(mentions[0]).strip()
    context = " ".join(context.split())
    if len(context) > 320:
        context = context[:317].rsplit(" ", 1)[0] + "…"
    findings = [f for f in findings if f["source_supported"]]
    shown_label = ", ".join(dict.fromkeys(f["label"] for f in findings))
    article_title = row["article_title"] if "article_title" in row.keys() else None
    doi_url = f"https://doi.org/{row['doi']}" if row["doi"] else row["source_url"]
    license_url = row["license_url"] if "license_url" in row.keys() else None
    if not license_url:
        license_url = row["article_license_url"] if "article_license_url" in row.keys() else None
    if not license_url and "figure_license" in row.keys() and str(row["figure_license"] or "").startswith("http"):
        license_url = row["figure_license"]
    article_url = f"https://doi.org/{row['doi']}" if row["doi"] else (
        f"https://pmc.ncbi.nlm.nih.gov/articles/{row['pmcid']}/" if row["pmcid"] else row["source_url"]
    )
    age_group = _norm(row["age_group"])
    age_group_supported = _age_group_supported(row["age_group"], source_text)
    age_group_label = _age_group_label(row["age_group"]) if age_group_supported else "Not stated"
    country = row["study_region"] or (row["article_country"] if "article_country" in row.keys() else None)
    source_variant = {
        "article_title": article_title,
        "article_url": article_url,
        "copyright": row["attribution_text"],
        "license_url": license_url,
        "license_code": row["license_code"],
        "context": context,
        "country": country,
        "age_group_label": age_group_label,
        "age_group_metadata_label": _age_group_label(row["age_group"]),
        "age_group_source_supported": age_group_supported,
        "pmcid": row["pmcid"],
        "panel_ids": [row["panel_id"]],
    }
    return {
        "panel_id": row["panel_id"],
        "figure_id": row["figure_id"],
        "pmcid": row["pmcid"],
        "panel_label": row["panel_label"],
        "disease_key": row["disease_key"],
        "subtype": row["subtype"],
        "modality": row["modality"],
        "body_site": row["body_site"],
        "findings": findings,
        "sha256": row["sha256"],
        "typicality": row["typicality"],
        "stage": row["stage"],
        "age_group": row["age_group"],
        "skin_tone": row["skin_tone"],
        "stated_ethnicity": row["stated_ethnicity"],
        "stated_ethnicity_quote": row["stated_ethnicity_quote"],
        "study_region": row["study_region"],
        "confidence": row["confidence"],
        "rationale": row["rationale"],
        "attribution_text": row["attribution_text"],
        "figure_label": row["figure_label"],
        "figure_caption": row["figure_caption"],
        "in_text_mentions": mentions,
        "license_code": row["license_code"],
        "license_url": license_url,
        "copyright": row["attribution_text"],
        "doi_url": doi_url,
        "article_url": article_url,
        "article_title": article_title,
        "country": country,
        "display_label": shown_label or _caption_label(caption) or row["body_site"] or row["modality"] or "Clinical image",
        "context": context,
        "age_group_label": age_group_label,
        "age_group_variants": [row["age_group"]] if row["age_group"] else [],
        "pediatric": _is_pediatric(age_group),
        "source_variant": source_variant,
        "source_variants": [source_variant],
        "_routing_findings": routing_findings,
        "image": f"/media/{row['image_path']}" if row["image_path"] else None,
        "thumb": f"/media/{thumb}" if thumb else None,
    }


def _labels(conn) -> dict[str, str]:
    return {r["finding_key"]: r["label"] for r in conn.execute("SELECT finding_key, label FROM findings_vocab")}


def _finding_terms(conn) -> dict[str, list[str]]:
    terms = {}
    for r in conn.execute("SELECT finding_key, label, synonyms_json FROM findings_vocab WHERE approved=1"):
        values = [r["finding_key"].replace("_", " "), r["label"].split("(", 1)[0]]
        values.extend(db.from_json(r["synonyms_json"], []))
        terms[r["finding_key"]] = [" ".join(str(v).casefold().split()) for v in values if v]
    return terms


_VASCULAR_FINDINGS = {
    "cutaneous_vasculitis", "raynaud_phenomenon", "livedo_reticularis",
    "digital_ulcer", "perniosis", "chilblain_lupus",
}
_FINDING_SUBTYPE = {
    "acle": {"malar_rash", "acute_cutaneous_lupus"},
    "scle": {"scle_annular", "scle_papulosquamous", "subacute_cutaneous_lupus"},
    "dle": {"discoid_plaque", "discoid_lupus", "scarring_alopecia"},
}


def _age_group_label(age_group) -> str:
    normalized = _norm(age_group)
    return {
        "child": "Child", "children": "Child", "pediatric": "Pediatric",
        "paediatric": "Pediatric", "adolescent": "Adolescent", "infant": "Infant",
    }.get(normalized, age_group or "Unknown")


def _age_group_supported(age_group, source_text: str) -> bool:
    normalized = _norm(age_group)
    terms = {
        "child": ("child", "children", "pediatric", "paediatric"),
        "children": ("child", "children", "pediatric", "paediatric"),
        "pediatric": ("child", "children", "pediatric", "paediatric"),
        "paediatric": ("child", "children", "pediatric", "paediatric"),
        "adolescent": ("adolescent", "teenage", "teenager"),
        "infant": ("infant", "newborn", "neonate"),
        "adult": ("adult",),
        "older_adult": ("older adult", "elderly", "aged"),
    }.get(normalized, ())
    return any(term in source_text for term in terms)


def _finding_supported(finding: dict, source_text: str, terms: list[str]) -> bool:
    # P3's copied evidence can describe an image without naming the finding it
    # assigned. Require the approved key, label, or synonym in source text.
    return any(term and term in source_text for term in terms)


def _caption_label(caption: str) -> str:
    if not caption:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
    image_cue = re.compile(
        r"\b(?:CT|MRI|scan|photograph|photo|image|radiograph|ultrasound|figure|panel|biopsy)\s+"
        r"(?:shows?|demonstrates?|reveals?|depicts?|illustrates?|showing)\b", re.I
    )
    demographic = re.compile(r"\b\d{1,3}[- ]year[- ]old\b|\b(?:male|female) patient\b", re.I)
    sentence = ""
    for candidate in sentences:
        cue = image_cue.search(candidate)
        if cue:
            sentence = candidate[cue.start():].strip()
            break
        if not demographic.search(candidate):
            sentence = candidate.strip()
            break
    if not sentence:
        sentence = sentences[0].strip()
    sentence = re.sub(
        r"^(?:clinical example of (?:a )?case with|representative case of|clinical image of)\s+",
        "", sentence, flags=re.I,
    )
    # Patient demographics belong in the detail view. Preserve the source's
    # visual description while dropping a trailing age/sex clause.
    sentence = re.sub(
        r"\s+(?:of|in)\s+(?:(?:a|an|this)\s+)?\d{1,3}[- ]year[- ]old\b.*$",
        "", sentence, flags=re.I,
    )
    # Keep the visible morphology and location, while dropping patient history
    # and diagnostic explanation that belongs in the image detail view.
    sentence = re.sub(
        r"\bseen\s+(on|in)\s+(?:(?:this|a|an)\s+)?(?:male|female)\s+patient's\s+",
        r"\1 the ", sentence, flags=re.I,
    )
    sentence = re.sub(
        r"\s+(?:of|in)\s+(?:(?:this|a|an)\s+)?(?:male|female)\s+patient\b.*$",
        "", sentence, flags=re.I,
    )
    sentence = re.sub(
        r"\s*(?:,|;)?\s*(?:biopsy(?:\s+results?)?|histopathology|diagnosed\s+with|"
        r"diagnosis\s+of|findings?\s+(?:were\s+)?consistent\s+with)\b.*$",
        "", sentence, flags=re.I,
    )
    sentence = sentence.rstrip(" .")
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    if len(sentence) > 90:
        sentence = sentence[:89].rsplit(" ", 1)[0] + "…"
    return sentence


def _clinical_group(panel: dict) -> str:
    """Use only explicit approved finding keys for cutaneous subtype groups."""
    keys = {_norm(f["key"]) for f in panel.get("findings", []) if f.get("source_supported")}
    caption = _norm(panel.get("figure_caption"))
    if keys & _VASCULAR_FINDINGS or any(term in caption for term in ("cutaneous vasculitis", "raynaud phenomenon", "livedo reticularis")):
        return "Vascular findings"
    if re.search(r"\bsubacute cutaneous lupus\b|\bscle\b", caption):
        return "SCLE"
    if re.search(r"\bacute cutaneous lupus\b|\bacle\b", caption):
        return "ACLE"
    if re.search(r"\bdiscoid lupus\b|\bdle\b", caption):
        return "DLE"
    for subtype, findings in _FINDING_SUBTYPE.items():
        if keys & findings:
            return subtype.upper()
    return "Other skin findings"


def _is_pediatric(age_group) -> bool:
    return _norm(age_group) in {"child", "children", "pediatric", "paediatric", "infant", "adolescent"}


def _exclude_curated(conn, rows: list[sqlite3.Row], panels: list[dict],
                     allowed_findings: set[str] | None = None) -> list[dict]:
    """Apply current-hash audit exclusions and the shared deterministic gate."""
    stored = {}
    try:
        stored = {
            r["panel_id"]: r["image_sha256"]
            for r in conn.execute("SELECT panel_id, image_sha256 FROM panel_curation WHERE decision='exclude'")
        }
    except sqlite3.OperationalError:
        # Older databases are initialized by create_app, but keep API usable
        # for read-only legacy stores where migration cannot be performed.
        pass
    try:
        from ..curation import exclusion_reason
    except ImportError:
        exclusion_reason = None

    visible = []
    for row, panel in zip(rows, panels):
        if stored.get(panel["panel_id"]) == panel.get("sha256") and panel.get("sha256"):
            continue
        # Use the same deterministic eligibility policy as new curation. Missing
        # or invalid bounds are ineligible, so they cannot leak partial figures.
        if exclusion_reason:
            p = dict(row)
            figure = {
                "label": row["figure_label"],
                "caption": row["figure_caption"],
                "triage_json": row["figure_triage_json"] if "figure_triage_json" in row.keys() else None,
                "vision_json": row["figure_vision_json"] if "figure_vision_json" in row.keys() else None,
                "effective_license": row["figure_license"] if "figure_license" in row.keys() else None,
            }
            article = {"title": row["article_title"] if "article_title" in row.keys() else None}
            if exclusion_reason(p, figure, article, allowed_findings=allowed_findings):
                continue
        visible.append(panel)
    return visible


def _collapse_duplicates(panels: list[dict]) -> list[dict]:
    """Panels sharing an image sha256 collapse into one card that lists
    every row's attribution (dedup by sha256 done at store time)."""
    grouped: dict[str, dict] = {}
    for p in panels:
        key = p.get("sha256") or p["panel_id"]
        existing = grouped.get(key)
        if existing is None:
            p["attribution_variants"] = [p["attribution_text"]]
            p["caption_variants"] = [p.get("figure_caption")]
            p["panel_ids"] = [p["panel_id"]]
            grouped[key] = p
            continue
        existing["attribution_variants"].append(p["attribution_text"])
        existing["panel_ids"].append(p["panel_id"])
        if p.get("figure_caption") not in existing["caption_variants"]:
            existing["caption_variants"].append(p.get("figure_caption"))
        known = {f["key"] for f in existing["findings"]}
        for f in p["findings"]:
            if f["key"] not in known:
                existing["findings"].append(f)
                known.add(f["key"])
            elif f.get("source_supported"):
                current = next(item for item in existing["findings"] if item["key"] == f["key"])
                if not current.get("source_supported"):
                    current.update(f)
        existing["_routing_findings"] = sorted(
            _routing_finding_keys(existing) | _routing_finding_keys(p)
        )
        # Merge tag union fields when the duplicate adds information.
        for field in ("disease_key", "subtype", "stage"):
            if existing.get(field) != p.get(field):
                existing[field] = None
        ages = set(existing.get("age_group_variants", []))
        if existing.get("age_group"):
            ages.add(existing["age_group"])
        if p.get("age_group"):
            ages.add(p["age_group"])
        existing["age_group_variants"] = sorted(ages)
        supported_age_labels = {
            variant.get("age_group_label") for variant in existing["source_variants"]
            if variant.get("age_group_label") and variant.get("age_group_label") != "Not stated"
        }
        if p.get("age_group_label") and p["age_group_label"] != "Not stated":
            supported_age_labels.add(p["age_group_label"])
        existing["age_group_label"] = " / ".join(sorted(supported_age_labels)) or "Not stated"
        existing["pediatric"] = any(_is_pediatric(a) for a in ages)
        if p.get("source_variant") not in existing["source_variants"]:
            existing["source_variants"].append(p["source_variant"])
    return list(grouped.values())


def _mark_representatives(panels: list[dict], mapping: dict[str, dict]) -> list[dict]:
    """Annotate cards with representative findings while keeping all cards."""
    for panel in panels:
        member_ids = set(panel.get("panel_ids") or [panel.get("panel_id")])
        primary = sorted(
            finding for finding, selection in mapping.items()
            if selection.get("panel_id") in member_ids
        )
        panel["representative_findings"] = primary
        panel["is_representative"] = bool(primary)
    return panels


def _match_alt(alt: dict, panel: dict, cats: set[str], body: str) -> bool:
    """One alternative: every declared key must hit (AND within it)."""
    findings = _routing_finding_keys(panel)
    if "modalities" in alt and _norm(panel.get("modality")) not in {
        _norm(m) for m in alt["modalities"]
    }:
        return False
    if "body_sites" in alt and body not in {_norm(b) for b in alt["body_sites"]}:
        return False
    if "findings" in alt and not (findings & {_norm(f) for f in alt["findings"]}):
        return False
    if "categories" in alt and not (cats & {_norm(c) for c in alt["categories"]}):
        return False
    return True


def _match(tab_match: dict, panel: dict, categories: dict[str, str]) -> bool:
    findings = _routing_finding_keys(panel)
    cats = {categories.get(k) for k in findings} - {None}
    body = _norm(panel.get("body_site"))
    if cats & {_norm(x) for x in tab_match.get("not_categories", ())}:
        return False
    if body and body in {_norm(x) for x in tab_match.get("not_body_sites", ())}:
        return False
    if findings & {_norm(x) for x in tab_match.get("not_findings", ())}:
        return False
    return any(
        _match_alt(alt, panel, cats, body) for alt in tab_match.get("any_of", [])
    )


def assign_tab(panel: dict, tabs: list[dict], categories: dict[str, str]) -> str:
    modality = _norm(panel.get("modality"))
    # In the generic layout, the actual image modality is authoritative for
    # diagnostic media. A cross-modality finding tag (for example, a skin
    # finding attached to a histology image) must not move the image into a
    # clinical-photo section.
    for tab in tabs:
        if modality and modality in {_norm(m) for m in tab.get("modality_priority", ())}:
            return tab["key"]
    ordered = sorted(tabs, key=lambda t: 0 if t.get("priority") == 0 else 1)
    for tab in ordered:
        if tab.get("cross_cutting") or "match" not in tab:
            continue
        if _match(tab["match"], panel, categories):
            return tab["key"]
    return "other"


def _query_panels(
    conn,
    disease: str,
    *,
    modality=None,
    subtype=None,
    skin_tone=None,
    finding=None,
    typicality=None,
) -> list[sqlite3.Row]:
    where = ["p.disease_key = :disease"]
    params: dict = {"disease": disease}
    if modality:
        where.append("p.modality = :modality")
        params["modality"] = modality
    if subtype:
        where.append("p.subtype = :subtype")
        params["subtype"] = subtype
    if skin_tone:
        where.append("p.skin_tone = :skin_tone")
        params["skin_tone"] = skin_tone
    if finding:
        where.append(
            "EXISTS (SELECT 1 FROM json_each(p.findings_json) je "
            "WHERE COALESCE(json_extract(je.value, '$.finding_key'), je.value) = :finding)"
        )
        params["finding"] = finding
    if typicality:
        where.append("p.typicality = :typicality")
        params["typicality"] = typicality
    sql = (
        "SELECT p.*, a.doi, f.label AS figure_label, f.caption AS figure_caption, "
        "f.in_text_mentions_json, f.effective_license AS figure_license, "
        "f.triage_json AS figure_triage_json, "
        "f.vision_json AS figure_vision_json, "
        "a.title AS article_title, a.license_url AS article_license_url, a.country AS article_country "
        "FROM panels p "
        "LEFT JOIN articles a ON a.pmcid = p.pmcid "
        "LEFT JOIN figures f ON f.figure_id = p.figure_id "
        f"WHERE {' AND '.join(where)}"
    )
    return list(conn.execute(sql, params))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app(data_dir: str | None = None) -> FastAPI:
    data_root = (Path(data_dir) if data_dir is not None else config.data_dir()).resolve()
    app = FastAPI(title="Visual Findings Library")
    app.state.data_dir = data_root

    # Apply idempotent schema upgrades once at startup (including the optional
    # persisted panel-curation and representative tables), not on every API request.
    initialized = db.connect(data_root / config.DB_FILENAME)
    try:
        db.init_db(initialized)
        representatives.rebuild(initialized)
    finally:
        initialized.close()

    def conn() -> sqlite3.Connection:
        return db.connect(data_root / config.DB_FILENAME)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    def _page(name: str) -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / name).read_text(encoding="utf-8"))

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return _page("index.html")

    @app.get("/disease/{key}", response_class=HTMLResponse)
    def disease_page(key: str) -> HTMLResponse:
        c = conn()
        try:
            if not _known_disease(c, key):
                raise HTTPException(404, "unknown disease")
            return _page("disease.html")
        finally:
            c.close()

    @app.get("/compare/sle-dm-skin", response_class=HTMLResponse)
    def compare_page() -> HTMLResponse:
        return _page("compare.html")

    @app.get("/api/diseases")
    def api_diseases() -> list[dict]:
        c = conn()
        try:
            return [
                {
                    "key": r["disease_key"],
                    "name": r["name"],
                    "subtypes": db.from_json(r["subtypes_json"], []),
                }
                for r in c.execute("SELECT * FROM diseases ORDER BY disease_key")
            ]
        finally:
            c.close()

    @app.get("/api/vocab")
    def api_vocab(disease: str | None = Query(default=None)) -> list[dict]:
        c = conn()
        try:
            rows = list(c.execute("SELECT * FROM findings_vocab WHERE approved=1"))
            if disease:
                rows = [r for r in rows if disease in db.from_json(r["disease_keys_json"], [])]
            return [
                {
                    "finding_key": r["finding_key"],
                    "label": r["label"],
                    "category": r["category"],
                    "disease_keys": db.from_json(r["disease_keys_json"], []),
                }
                for r in rows
            ]
        finally:
            c.close()

    @app.get("/api/diseases/{key}/tabs")
    def api_tabs(key: str) -> list[dict]:
        c = conn()
        try:
            if not _known_disease(c, key):
                raise HTTPException(404, "unknown disease")
            tabs = _tabs_for(key)
            approved = _approved_keys(c, key)
            categories = _categories(c)
            labels = _labels(c)
            terms = _finding_terms(c)
            rows = _query_panels(c, key)
            panels = [_panel_json(r, approved, labels, categories, terms) for r in rows]
            panels = _exclude_curated(c, rows, panels, approved)
            for panel in panels:
                panel["tab"] = assign_tab(panel, tabs, categories)

            available = {panel["tab"] for panel in panels}
            if any(panel["pediatric"] for panel in panels):
                available.add("pediatric")
            # The catch-all view is useful only when routing actually leaves
            # visible panels uncategorized.
            if "other" in available:
                tabs = [*tabs, {"key": "other", "label": "Other"}]
            evidence_only_eye = (
                "eye" not in available
                and any(t["key"] == "eye" for t in tabs)
                and bool(_eye_evidence(c, key))
            )
            if evidence_only_eye:
                available.add("eye")
            return [
                {
                    "key": t["key"],
                    "label": t["label"],
                    "group_by": t.get("group_by"),
                    "group_order": t.get("group_order"),
                    "skin_tone_filter": bool(t.get("skin_tone_filter")),
                    "cross_cutting": bool(t.get("cross_cutting")),
                    "evidence_only": evidence_only_eye and t["key"] == "eye",
                }
                for t in tabs
                if t["key"] in available
            ]
        finally:
            c.close()

    @app.get("/api/diseases/{key}/eye-evidence")
    def api_eye_evidence(key: str) -> list[dict]:
        c = conn()
        try:
            if not _known_disease(c, key):
                raise HTTPException(404, "unknown disease")
            return _eye_evidence(c, key)
        finally:
            c.close()

    @app.get("/api/diseases/{key}/panels")
    def api_panels(
        key: str,
        modality: str | None = None,
        subtype: str | None = None,
        skin_tone: str | None = None,
        finding: str | None = None,
        typicality: str | None = None,
    ) -> dict:
        c = conn()
        try:
            if not _known_disease(c, key):
                raise HTTPException(404, "unknown disease")
            approved = _approved_keys(c, key)
            categories = _categories(c)
            labels = _labels(c)
            terms = _finding_terms(c)
            if finding and finding not in approved:
                # unapproved/proposed findings are never shown
                return {"panels": [], "count": 0, "representatives": {}}
            rows = _query_panels(
                c,
                key,
                modality=modality,
                subtype=subtype,
                skin_tone=skin_tone,
                finding=finding,
                typicality=typicality,
            )
            panels = [_panel_json(r, approved, labels, categories, terms) for r in rows]
            panels = _exclude_curated(c, rows, panels, approved)
            if finding:
                panels = [p for p in panels if finding in _finding_keys(p)]
            for p in panels:
                p["tab"] = assign_tab(p, _tabs_for(key), categories)
                p["clinical_tab"] = p["tab"]
                p["clinical_group"] = _clinical_group(p) if key == "sle" and p["tab"] == "skin" else None
                p["source_variant"]["clinical_tab"] = p["clinical_tab"]
                p["source_variant"]["clinical_group"] = p["clinical_group"]
            panels = _collapse_duplicates(panels)
            representative_map = representatives.mapping_for_disease(c, key)
            _mark_representatives(panels, representative_map)
            for p in panels:
                p["clinical_tab"] = p["tab"]
                p["clinical_group"] = _clinical_group(p) if key == "sle" and p["tab"] == "skin" else None
                p["pediatric"] = p.get("pediatric") or any(
                    _is_pediatric(age) for age in p.get("age_group_variants", [])
                )
                p.pop("_routing_findings", None)
            return {
                "panels": _sort_panels(panels), "count": len(panels),
                "representatives": representative_map,
            }
        finally:
            c.close()

    @app.get("/api/diseases/{key}/findings")
    def api_findings(key: str) -> list[dict]:
        c = conn()
        try:
            if not _known_disease(c, key):
                raise HTTPException(404, "unknown disease")
            # Approved vocab keys only — proposed findings are never shown.
            rows = list(
                c.execute(
                    "SELECT df.*, fv.label AS vocab_label, fv.disease_keys_json FROM disease_findings df "
                    "JOIN findings_vocab fv "
                    "  ON fv.finding_key = df.finding_key AND fv.approved = 1 "
                    "WHERE df.disease_key = ? "
                    "ORDER BY CASE df.source WHEN 'text' THEN 0 ELSE 1 END, "
                    "COALESCE(df.frequency_pct_high, -1) DESC, df.id",
                    (key,),
                )
            )
            seen: set[tuple] = set()
            out = []
            for r in rows:
                if key not in db.from_json(r["disease_keys_json"], []):
                    continue
                pair = (r["finding_key"], r["pmcid"])
                if pair in seen:
                    continue
                seen.add(pair)
                out.append(
                    {
                        "finding_key": r["finding_key"],
                        "label": r["vocab_label"],
                        "subtype": r["subtype"],
                        "frequency_text": r["frequency_text"],
                        "pct_low": r["frequency_pct_low"],
                        "pct_high": r["frequency_pct_high"],
                        "quote": r["quote"],
                        "pmcid": r["pmcid"],
                        "source": r["source"],
                    }
                )
                if len(out) >= 12:
                    break
            return out
        finally:
            c.close()

    @app.get("/api/compare/sle-dm-skin")
    def api_compare() -> list[dict]:
        c = conn()
        try:
            labels = _labels(c)
            out = []
            for pair in COMPARISONS:
                entry = {"title": pair["title"], "left": {"label": pair["left"]["label"]}, "right": {"label": pair["right"]["label"]}}
                for side in ("left", "right"):
                    spec = pair[side]
                    approved = _approved_keys(c, spec["disease"])
                    rows = _query_panels(
                        c,
                        spec["disease"],
                        modality=None,
                        finding=spec.get("finding"),
                    )
                    categories = _categories(c)
                    terms = _finding_terms(c)
                    panels = [_panel_json(r, approved, labels, categories, terms) for r in rows]
                    panels = _exclude_curated(c, rows, panels, approved)
                    panels = _collapse_duplicates(panels)
                    if spec.get("finding"):
                        panels = [p for p in panels if spec["finding"] in _finding_keys(p)]
                    if spec.get("findings_any"):
                        wanted = {_norm(f) for f in spec["findings_any"]}
                        panels = [p for p in panels if _finding_keys(p) & wanted]
                    if spec.get("body_sites"):
                        wanted = {_norm(b) for b in spec["body_sites"]}
                        panels = [p for p in panels if _norm(p.get("body_site")) in wanted]
                    if spec.get("subtype_in"):
                        wanted = {_norm(s) for s in spec["subtype_in"]}
                        panels = [p for p in panels if _norm(p.get("subtype")) in wanted]
                    if spec.get("modalities"):
                        wanted = {_norm(m) for m in spec["modalities"]}
                        panels = [p for p in panels if _norm(p.get("modality")) in wanted]
                    entry[side]["panels"] = _sort_panels(panels)[:6]
                    entry[side]["disease"] = spec["disease"]
                out.append(entry)
            return out
        finally:
            c.close()

    @app.get("/media/{path:path}")
    def media(path: str):
        root = data_root
        target = (root / path).resolve()
        # Path traversal guard: must resolve inside data_dir AND under an
        # allowed image subdirectory.
        if not str(target).startswith(str(root) + "/"):
            raise HTTPException(403, "forbidden")
        try:
            rel = target.relative_to(root)
        except ValueError:
            raise HTTPException(403, "forbidden") from None
        if rel.parts[0] not in MEDIA_PREFIXES:
            raise HTTPException(403, "forbidden")
        if not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    return app


def run(args) -> int:
    import uvicorn

    print(f"viewer: http://127.0.0.1:{args.port} (data: {config.data_dir()})")
    uvicorn.run(
        "src.visual_pilot.viewer.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )
    return 0
