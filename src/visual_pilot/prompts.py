"""LLM prompts P2-P5 and their JSON schemas, verbatim from spec section 6.

All calls use JSON mode with the schema appended to the system prompt
(validated locally by llm.py), and each system prompt
ends with "Return only JSON." Prompt versions feed the llm_calls cache hash —
bump a PROMPT_VERSION whenever a prompt or its schema changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .diseases import load_diseases


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    schema: dict


def _catalog_description(keys=None) -> str:
    catalog = load_diseases()
    parts = []
    for key in keys if keys is not None else catalog:
        disease = catalog[key]
        name = disease.get("name", key)
        synonyms = [s for s in disease.get("synonyms", []) if s.casefold() != name.casefold()]
        subtypes = [item.get("label", item.get("key", "")) for item in disease.get("subtypes", [])]
        details = []
        if synonyms:
            details.append("also called " + ", ".join(synonyms))
        if subtypes:
            details.append("includes " + ", ".join(subtypes))
        parts.append(f"{key}: {name}" + (f" ({'; '.join(details)})" if details else ""))
    return "; ".join(parts)


def _keys() -> list[str]:
    return list(load_diseases())


# Placeholders in the P2-P4 system templates, filled with the full catalog for
# the module-level prompts and with a disease subset by ``scoped``.
_CATALOG = "<<CATALOG>>"
_P3_CATALOG = "<<P3_CATALOG>>"
_EXCLUDED = "<<EXCLUDED_EXAMPLES>>"


P2_VERSION = "p2.v6"
P2_TEMPLATE = f"""You triage figure captions from open-access medical articles (case reports, original research and reviews) about {_CATALOG}. Prefer a clear, whole clinical image showing a human patient's disease manifestation. For each figure, decide from the label, caption and in-text mentions alone whether it contains such an image: clinical photograph, dermoscopy, nailfold capillaroscopy, histopathology, immunohistochemistry, immunofluorescence, cytology, radiograph, CT, MRI, ultrasound, echocardiogram, PET, endoscopy, ophthalmic image, or gross specimen. Non-patient content includes diagrams, schematics, pathways, mechanism figures, flowcharts, algorithms, charts/graphs, tables, drawings/illustrations, and photos of equipment. Drop veterinary/animal images. A multi-panel figure, montage or collage is judged as one whole image: route it `keep` (or `uncertain` if the caption is vague) when every panel is a human patient image of the same configured disease, and use the shared category (or `mixed` when the panels use different image types). Drop a multi-panel figure when any panel is a diagram, chart, table, illustration, or normal or healthy control, or when its panels show more than one disease (for example a comparison with another condition). Never route an individual panel as its own image; the library stores the complete figure. Set `third_party` true if the caption says the image is reproduced, adapted, reprinted or used with permission from another source, carries a copyright notice (©), or is "courtesy of" someone, and quote the phrase. Drop treatment figures: before/after or treatment-response comparisons, images during or after a named drug or therapy, follow-up images showing improvement or healing, intraoperative or postoperative views, injection sites, and devices; the library shows untreated disease and does not promote treatments. A disease the caption says a drug caused (e.g. drug-induced lupus) is not a treatment figure, and neither is a presentation whose caption only mentions treatment in passing. Use `uncertain` when the caption does not say what the image shows (e.g. "Representative case"). Never guess `drop` for an image-like caption unless it is clearly a diagram, non-human image, treatment figure, or a mixed or multi-disease montage as described above.

Return only JSON."""
P2_SYSTEM = P2_TEMPLATE.replace(_CATALOG, _catalog_description())
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
                        "items": {"type": "string", "enum": [*_keys(), "other"]},
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


P3_VERSION = "p3.v5"
_OTHER_EXAMPLES = ("polymyositis", "inclusion body myositis", "psoriatic arthritis")


def _excluded_examples(keys=None) -> str:
    catalog = load_diseases()
    folded = " ".join(
        (key + " " + catalog[key].get("name", "") + " " + " ".join(catalog[key].get("synonyms", []))).casefold()
        for key in (keys if keys is not None else catalog)
    )
    examples = [term for term in _OTHER_EXAMPLES if term not in folded]
    return ", " + ", ".join(examples) if examples else ""


def _p3_catalog(keys=None) -> str:
    catalog = load_diseases()
    return "; ".join(
        f"{key} ({catalog[key].get('name', key)}; subtypes: {', '.join(s.get('key', '') for s in catalog[key].get('subtypes', [])) or 'as supported by the provided vocabulary'})"
        for key in (keys if keys is not None else catalog)
    )


P3_TEMPLATE = f"""You curate images for a clinical visual-diagnosis library covering only these configured diseases: {_P3_CATALOG}. You receive one figure from an open-access medical article (case report, original research or review), its caption, the in-text mentions, the full article title and topic diseases, a prior caption-triage result when available, and an allowed findings vocabulary.

Rules:
1. Identify every panel (use the panel letters in the image or caption; a single-image figure is panel "A"). Give each panel a tight normalized bbox [x0, y0, x1, y1] in 0–1 image coordinates.
2. `include` = true only if the panel is a real human-patient image that visibly shows a specific clinical manifestation of one of the configured target diseases. Exclude diagrams, schematics, mechanisms, pathways, classification charts/tables, text-only graphics, normal or control images, veterinary images or animal models, other diseases (including comparison panels of other diseases{_EXCLUDED}), unreadable quality, and images where the caption indicates third-party copyright.
3. Tag the disease **the panel shows**, using panel-specific caption evidence or a disease-focused article title together with a specific captioned manifestation; a broad article topic or general in-text mention alone is insufficient. Figures, especially in reviews, often contrast diseases. A complication that can occur in the disease is not enough to establish attribution. Exclude panels where the caption identifies another disease or where the manifestation's disease association is unclear. Do not propagate a broad term such as vasculitis into every subtype/group: assign only the subtype directly supported for that panel.
4. A montage, collage or multi-panel figure is stored only as the complete figure, never as crops. Set `figure_is_compound` true, still identify every panel with its bbox, and classify each panel exactly as rules 2, 3 and 6–12 describe (disease, modality, findings and the other fields). Set a panel's `include` true if that panel on its own would qualify under rule 2; otherwise set `include` false with its specific `exclusion_reason` (diagram, chart, normal_control, other_disease, not_patient_image, poor_quality, third_party). Do not skip or merge panels: a single diagram, chart, control or other-disease panel keeps the whole figure out of the library.
5. A single-image figure must be clear and substantial. Inside a compound figure, small panels are classified like any other panel and are not excluded for size alone; they are never stored separately.
6. `findings`: use only `finding_key` values from the vocabulary provided, each with the caption or in-text phrase that supports it (or "visual" if you identified it only from the image). Put anything clearly present but missing from the vocabulary in `proposed_findings` as short clinical terms. Do not invent a manifestation from article context. If none can be named, exclude the panel.
7. `typicality`: classic (textbook presentation), variant (recognized less common form), atypical (unusual; the caption usually says so).
8. `skin_tone`: only for panels showing skin, nails, lips or oral mucosa. Judge only from visible, adequately lit skin: light (≈ Fitzpatrick I–II), medium (III–IV), dark (V–VI), unknown if not assessable. Never infer it from the country, the caption or the journal. Use null for non-skin panels.
9. `stated_ethnicity`: only if the caption or in-text mentions explicitly state it; give the verbatim quote. Otherwise null. Never infer ethnicity or race.
10. `age_group`: child, adolescent, adult, older_adult, unknown. Use only the caption or in-text mentions; otherwise use unknown.
11. `stage`: for AS use nr_axSpA, early, advanced (ankylosis or bamboo spine) or unknown; for others, a short text or null.
12. `confidence` 0–1 reflects the disease attribution and findings together.

Return only JSON."""
P3_SYSTEM = P3_TEMPLATE.replace(_P3_CATALOG, _p3_catalog()).replace(_EXCLUDED, _excluded_examples())
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
                "collage",
                "normal_control",
                "other_disease",
                "poor_quality",
                "third_party",
                "not_patient_image",
                None,
            ],
        },
        "disease_key": {"type": ["string", "null"], "enum": [*_keys(), None]},
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


P4_VERSION = "p4.v2"
P4_TEMPLATE = f"""From the article text sections provided, extract statements that link one of {_CATALOG} to a clinical, imaging, histologic or capillaroscopic finding. Map each to a `finding_key` from the provided vocabulary when possible, otherwise put it in `proposed_finding`. Capture the stated frequency exactly as written (e.g. "30–60%", "most patients", "pathognomonic") and parse it into numeric low/high percentages only when numbers are explicit. Include a verbatim quote of 40 words or fewer. Do not add facts that are not in the text.

Return only JSON."""
P4_SYSTEM = P4_TEMPLATE.replace(_CATALOG, _catalog_description())
P4_SCHEMA = {
    "type": "object",
    "properties": {
        "assertions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "disease_key": {"type": "string", "enum": _keys()},
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

P5_VERSION = "p5.v5"
P5_SYSTEM = """You write the display caption for one image in a clinical visual-diagnosis library. The image is shown on its own, outside the article it came from. You receive the original figure caption, the in-text mentions, which part of the figure the image is (one panel, or the whole figure), the image's disease, modality, body site and findings, and the library sections for that disease.

Write:
- `title`: a short name for what the image shows, 3–10 words, sentence case, no final period (e.g. "Chronic tophi of the middle finger").
- `description`: 1–2 plain sentences, at most 300 characters, describing what is visible in this image.
- `section`: the `key` of the one listed section this image belongs in, judged by what the image actually is (its image type and what it shows, e.g. a skin biopsy belongs in a histology section even if it is tagged with a skin finding). When a nailfold or capillaroscopy section is listed, an image mainly of the nailfold, cuticle or nailfold capillaries (by any image type) belongs there; an image mainly of other skin with incidental nail changes stays in the skin section. Use null only when no section fits.
- `treatment_related`: true when the image is about treatment rather than the untreated disease: before/after or treatment-response comparisons, appearance during or after a named drug or therapy, follow-up images showing improvement or healing, intraoperative or postoperative views, injection sites, or devices. Also true when the caption ties what is shown to a named drug or product (except a disease the caption says the drug caused, such as drug-induced lupus, which is false). False for a disease presentation whose caption only mentions treatment in passing, e.g. a baseline image "before treatment" with no after image shown.
- `subsection`: when the chosen section lists `subsections`, the one that fits this image best, copied exactly and chosen by its `subsection_meanings` when given; when the list is ["finding"], the `finding_key` (from `findings`) that this image best illustrates. Use null when the section has no subsections or the caption does not support any of them.

Rules:
- Use only facts stated in the caption or mentions; do not add diagnoses, findings or interpretation they do not state.
- When the image is one panel, describe only that panel; ignore the caption text for other panels.
- When the image is the whole figure, summarize what its parts show together, without letters (say "before and after surgery", not "A–D").
- Remove everything that only makes sense inside the article: figure and panel numbers or letters, citation numbers and reference marks (e.g. "tendon.19"), "see Figure 3", scale bars, staining or magnification boilerplate unless it is the point of the image, permission or copyright notes, and author or institution credits.
- Keep clinically useful context stated in the caption, such as the body site, imaging plane, stain, or treatment timepoint, and keys that explain visible marks (e.g. "arrows mark the double contour sign").
- Do not mention the article, the authors, or "this figure".

Return only JSON."""
P5_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "section": {"type": ["string", "null"]},
        "subsection": {"type": ["string", "null"]},
        "treatment_related": {"type": "boolean"},
    },
    "required": ["title", "description", "section", "subsection", "treatment_related"],
    "additionalProperties": False,
}
P5 = Prompt(name="p5_display_caption", version=P5_VERSION, system=P5_SYSTEM, schema=P5_SCHEMA)


_TEMPLATES = {
    "p2": (P2, lambda keys: P2_TEMPLATE.replace(_CATALOG, _catalog_description(keys))),
    "p3": (P3, lambda keys: P3_TEMPLATE.replace(_P3_CATALOG, _p3_catalog(keys)).replace(
        _EXCLUDED, _excluded_examples(keys))),
    "p4": (P4, lambda keys: P4_TEMPLATE.replace(_CATALOG, _catalog_description(keys))),
}


def _scope_schema(node, full: frozenset, keys: list[str]):
    """Copy ``node`` with every disease-key enum narrowed to ``keys``."""
    if isinstance(node, dict):
        return {
            k: [*keys, *(x for x in v if x not in full)]
            if k == "enum" and isinstance(v, list) and full <= set(v)
            else _scope_schema(v, full, keys)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_scope_schema(item, full, keys) for item in node]
    return node


@lru_cache(maxsize=4096)
def _scoped(name: str, keys: tuple[str, ...]) -> Prompt:
    base, system = _TEMPLATES[name]
    full = frozenset(load_diseases())
    return Prompt(
        name=base.name,
        version=base.version,
        system=system(list(keys)),
        schema=_scope_schema(base.schema, full, list(keys)),
    )


def scoped(name: str, disease_keys) -> Prompt:
    """P2/P3/P4 limited to the diseases of the articles in one call.

    With hundreds of catalog topics the full catalog (and its schema enums)
    would dominate every request; a call only needs the diseases its articles
    are about. Unknown keys are ignored; no known key falls back to the full
    prompt.
    """
    catalog = load_diseases()
    keys = tuple(dict.fromkeys(k for k in catalog if k in set(disease_keys or ())))
    if not keys:
        return _TEMPLATES[name][0]
    return _scoped(name, keys)
