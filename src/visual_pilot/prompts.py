"""LLM prompts P1-P4 and their JSON schemas, verbatim from spec section 6.

All calls: temperature 0, strict JSON-schema output, and each system prompt
ends with "Return only JSON." Prompt versions feed the llm_calls cache hash —
bump a PROMPT_VERSION whenever a prompt or its schema changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    schema: dict


P1_VERSION = "p1.v1"
P1_SYSTEM = """You screen open-access medical review articles for an image library covering three diseases: systemic lupus erythematosus (sle; includes cutaneous lupus subtypes ACLE/SCLE/DLE, lupus nephritis, neuropsychiatric lupus), dermatomyositis (dm; includes juvenile, amyopathic, anti-MDA5, cancer-associated), and ankylosing spondylitis (as; includes radiographic and non-radiographic axial spondyloarthritis). Given a title and abstract, decide which of these diseases the article substantially covers (a main topic, or a major section devoted to it). Passing mentions do not count. Mark `is_narrative_review` false if the article is actually a case report, systematic review, meta-analysis, trial, guideline methodology paper, or basic-science-only paper with no clinical presentation content.

Return only JSON."""
P1_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_disease_keys": {
            "type": "array",
            "items": {"type": "string", "enum": ["sle", "dm", "as"]},
        },
        "is_narrative_review": {"type": "boolean"},
        "decision": {"type": "string", "enum": ["relevant", "irrelevant"]},
        "reason": {"type": "string"},
    },
    "required": ["primary_disease_keys", "is_narrative_review", "decision", "reason"],
    "additionalProperties": False,
}
P1 = Prompt(name="p1_relevance", version=P1_VERSION, system=P1_SYSTEM, schema=P1_SCHEMA)


P2_VERSION = "p2.v1"
P2_SYSTEM = """You triage figure captions from open-access medical review articles about SLE, dermatomyositis, or ankylosing spondylitis. For each figure, decide from the label, caption and in-text mentions alone whether it likely contains at least one real-patient image: clinical photograph, dermoscopy, nailfold capillaroscopy, histopathology, immunohistochemistry, immunofluorescence, cytology, radiograph, CT, MRI, ultrasound, echocardiogram, PET, endoscopy, ophthalmic image, or gross specimen. Non-patient content includes diagrams, schematics, pathways, mechanism figures, flowcharts, algorithms, charts/graphs, tables, drawings/illustrations, and photos of equipment. A multi-panel figure counts as `keep` if any panel likely qualifies. Set `third_party` true if the caption says the image is reproduced, adapted, reprinted or used with permission from another source, carries a copyright notice (©), or is "courtesy of" someone, and quote the phrase. Use `uncertain` when the caption does not say what the image shows (e.g. "Representative case"). Never guess `drop` for an image-like caption.

Return only JSON."""
P2_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "figure_id": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "clinical_photo",
                            "dermoscopy",
                            "capillaroscopy",
                            "histology",
                            "immunofluorescence",
                            "radiology",
                            "ultrasound",
                            "echo",
                            "endoscopy",
                            "ophthalmic",
                            "gross",
                            "mixed",
                            "diagram",
                            "chart",
                            "flowchart",
                            "table",
                            "illustration",
                            "other",
                        ],
                    },
                    "is_real_patient_image": {"type": ["boolean", "null"]},
                    "third_party": {"type": "boolean"},
                    "third_party_quote": {"type": ["string", "null"]},
                    "diseases_mentioned": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["sle", "dm", "as", "other"]},
                    },
                    "route": {"type": "string", "enum": ["keep", "drop", "uncertain"]},
                    "reason": {"type": "string"},
                },
                "required": [
                    "figure_id",
                    "category",
                    "is_real_patient_image",
                    "third_party",
                    "third_party_quote",
                    "diseases_mentioned",
                    "route",
                    "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}
P2 = Prompt(name="p2_caption_triage", version=P2_VERSION, system=P2_SYSTEM, schema=P2_SCHEMA)


P3_VERSION = "p3.v1"
P3_SYSTEM = """You curate images for a clinical visual-diagnosis library covering only: sle (subtypes: ACLE, SCLE, DLE, lupus_nephritis, NPSLE, other_systemic), dm (classic, CADM, JDM, anti_MDA5, cancer_associated), as (r_axSpA, nr_axSpA). You receive one figure from an open-access review article, its caption, the in-text mentions, the article's topic diseases, and an allowed findings vocabulary.

Rules:
1. Identify every panel (use the panel letters in the image or caption; a single-image figure is panel "A"). Give each panel a tight normalized bbox [x0, y0, x1, y1] in 0–1 image coordinates.
2. `include` = true only if the panel is a real-patient image that visibly shows a finding of sle, dm or as. Exclude: diagrams or illustrations, charts, normal or control images, other diseases (including comparison panels of other diseases, polymyositis, inclusion body myositis, psoriatic arthritis), unreadable quality, or images where the caption indicates third-party copyright.
3. Tag the disease **the panel shows**, using the caption as evidence, not simply the article's topic. Review figures often contrast diseases.
4. `findings`: use only `finding_key` values from the vocabulary provided, each with the caption or in-text phrase that supports it (or "visual" if you identified it only from the image). Put anything clearly present but missing from the vocabulary in `proposed_findings` as short clinical terms.
5. `typicality`: classic (textbook presentation), variant (recognized less common form), atypical (unusual; the caption usually says so).
6. `skin_tone`: only for panels showing skin, nails, lips or oral mucosa. Judge only from visible, adequately lit skin: light (≈ Fitzpatrick I–II), medium (III–IV), dark (V–VI), unknown if not assessable. Never infer it from the country, the caption or the journal. Use null for non-skin panels.
7. `stated_ethnicity`: only if the caption or in-text mentions explicitly state it; give the verbatim quote. Otherwise null. Never infer ethnicity or race.
8. `age_group`: child, adolescent, adult, older_adult, unknown. Use the caption, or clear visual cues for children.
9. `stage`: for AS use nr_axSpA, early, advanced (ankylosis or bamboo spine) or unknown; for others, a short text or null.
10. `confidence` 0–1 reflects the disease attribution and findings together.

Return only JSON."""
_P3_PANEL = {
    "type": "object",
    "properties": {
        "panel_label": {"type": "string"},
        "bbox": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
        "include": {"type": "boolean"},
        "exclusion_reason": {
            "type": ["string", "null"],
            "enum": [
                "diagram",
                "chart",
                "normal_control",
                "other_disease",
                "poor_quality",
                "third_party",
                "not_patient_image",
                None,
            ],
        },
        "disease_key": {"type": ["string", "null"], "enum": ["sle", "dm", "as", None]},
        "subtype": {"type": ["string", "null"]},
        "modality": {
            "type": "string",
            "enum": [
                "clinical_photo",
                "dermoscopy",
                "capillaroscopy",
                "histology_he",
                "histology_ihc",
                "immunofluorescence",
                "radiograph",
                "ct",
                "mri",
                "ultrasound",
                "echo",
                "pet",
                "endoscopy",
                "ophthalmic",
                "gross",
                "other",
            ],
        },
        "body_site": {"type": ["string", "null"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding_key": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["finding_key", "evidence"],
                "additionalProperties": False,
            },
        },
        "proposed_findings": {"type": "array", "items": {"type": "string"}},
        "typicality": {
            "type": ["string", "null"],
            "enum": ["classic", "variant", "atypical", None],
        },
        "stage": {"type": ["string", "null"]},
        "age_group": {
            "type": "string",
            "enum": ["child", "adolescent", "adult", "older_adult", "unknown"],
        },
        "skin_tone": {
            "type": ["string", "null"],
            "enum": ["light", "medium", "dark", "unknown", None],
        },
        "stated_ethnicity": {"type": ["string", "null"]},
        "stated_ethnicity_quote": {"type": ["string", "null"]},
        "annotations_present": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": [
        "panel_label",
        "bbox",
        "include",
        "exclusion_reason",
        "disease_key",
        "subtype",
        "modality",
        "body_site",
        "findings",
        "proposed_findings",
        "typicality",
        "stage",
        "age_group",
        "skin_tone",
        "stated_ethnicity",
        "stated_ethnicity_quote",
        "annotations_present",
        "confidence",
        "rationale",
    ],
    "additionalProperties": False,
}
P3_SCHEMA = {
    "type": "object",
    "properties": {
        "figure_id": {"type": "string"},
        "figure_is_compound": {"type": "boolean"},
        "panels": {"type": "array", "items": _P3_PANEL},
    },
    "required": ["figure_id", "figure_is_compound", "panels"],
    "additionalProperties": False,
}
P3 = Prompt(name="p3_vision_judge", version=P3_VERSION, system=P3_SYSTEM, schema=P3_SCHEMA)


P4_VERSION = "p4.v1"
P4_SYSTEM = """From the review text sections provided, extract statements that link one of sle, dm or as to a clinical, imaging, histologic or capillaroscopic finding. Map each to a `finding_key` from the provided vocabulary when possible, otherwise put it in `proposed_finding`. Capture the stated frequency exactly as written (e.g. "30–60%", "most patients", "pathognomonic") and parse it into numeric low/high percentages only when numbers are explicit. Include a verbatim quote of 40 words or fewer. Do not add facts that are not in the text.

Return only JSON."""
P4_SCHEMA = {
    "type": "object",
    "properties": {
        "assertions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "disease_key": {"type": "string"},
                    "subtype": {"type": ["string", "null"]},
                    "finding_key": {"type": ["string", "null"]},
                    "proposed_finding": {"type": ["string", "null"]},
                    "frequency_text": {"type": ["string", "null"]},
                    "pct_low": {"type": ["number", "null"]},
                    "pct_high": {"type": ["number", "null"]},
                    "specificity_text": {"type": ["string", "null"]},
                    "quote": {"type": "string"},
                },
                "required": [
                    "disease_key",
                    "subtype",
                    "finding_key",
                    "proposed_finding",
                    "frequency_text",
                    "pct_low",
                    "pct_high",
                    "specificity_text",
                    "quote",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assertions"],
    "additionalProperties": False,
}
P4 = Prompt(name="p4_text_findings", version=P4_VERSION, system=P4_SYSTEM, schema=P4_SCHEMA)

PROMPTS = {p.name: p for p in (P1, P2, P3, P4)}
