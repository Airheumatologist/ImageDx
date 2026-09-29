"""Conservative, deterministic acceptance rules for usable clinical panels.

This policy is shared by the live judge and the viewer's persisted-row audit.
It deliberately favors a clean library over preserving uncertain crops: source
figures that are collages, diagrams, charts or veterinary material are not
split into guessed subpanels. A single-disease collage is kept whole as one
plate and never split into crops; mixed or multi-disease compound figures
stay unpublished.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache

from . import diseases

POLICY_VERSION = "clinical-panels.v5"

# Stored whole-figure rows carry this panel_label and crop_mode whole_figure.
PLATE_LABEL = "whole"
RADIOLOGY_MODALITIES = {"radiograph", "ct", "mri", "ultrasound", "echo", "pet"}

# Figure-wide language. Captions and titles are stronger evidence than the
# article's general disease context, which must never make an unrelated visual
# eligible by itself.
_FIGURE_DIAGRAM = re.compile(
    r"\b(?:diagram of|schematic (?:of|depicts|shows|illustrates)|flow ?chart|algorithm|pathway figure|mechanism figure|biorender)\b",
    re.I,
)
_FIGURE_CHART = re.compile(
    r"\b(?:classification chart|classification table|histopathology classification|graph of|table of|chart of)\b", re.I
)
_VETERINARY = re.compile(
    r"\b(?:veterinary|canine|feline|in dogs?|in cats?|dogs? with (?:lupus|dermatomyositis)|cats? with (?:lupus|dermatomyositis)|dog model|mouse model|murine model)\b",
    re.I,
)
# These abbreviations have common meanings outside the disease itself. The
# full configured disease name still matches, while the acronym alone cannot
# accidentally establish a figure's disease association.
_AMBIGUOUS_ACRONYMS = {"ad", "as", "dm", "psa", "ra", "ssc"}
_NONPILOT_DISEASES = (
    "scleroderma", "polymyositis", "inclusion body myositis", "covid-19",
    "interferonopathy", "tafro syndrome", "oro-facial granulomatosis",
)
_NEGATIVE = re.compile(
    r"\b(?:normal control|healthy control|unaffected control|negative control|control group|normal image|normal scan|normal mri|normal capillary bed|normal capillary pattern|unremarkable (?:image|scan|mri|finding))\b",
    re.I,
)
_BAD_PANEL = re.compile(
    r"\b(?:diagram of|schematic of|flow ?chart|classification chart|classification table|text[- ]only graphic)\b",
    re.I,
)


def _term_pattern(term: str) -> re.Pattern | None:
    """Compile a configured disease name/synonym with phrase-aware boundaries."""
    value = " ".join(str(term or "").strip().split())
    if len(value) < 3 or value.casefold() in _AMBIGUOUS_ACRONYMS:
        return None
    words = re.split(r"[\s-]+", value)
    body = r"[\s-]+".join(re.escape(word) for word in words if word)
    if not body:
        return None
    return re.compile(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])", re.I)


@lru_cache(maxsize=1)
def _disease_matchers() -> dict[str, tuple[re.Pattern, ...]]:
    """Build exact-phrase matchers from the disease catalog's names/synonyms."""
    matchers = {}
    for key, item in diseases.load_diseases().items():
        patterns = []
        seen = set()
        for term in (item.get("name", ""), *item.get("synonyms", [])):
            pattern = _term_pattern(term)
            if pattern and pattern.pattern not in seen:
                seen.add(pattern.pattern)
                patterns.append(pattern)
        matchers[key] = tuple(patterns)
    return matchers


def _diseases_in_text(text: str) -> set[str]:
    return {
        key for key, patterns in _disease_matchers().items()
        if any(pattern.search(text or "") for pattern in patterns)
    }


_NONPILOT_MATCHERS = tuple(
    pattern for term in _NONPILOT_DISEASES
    if (pattern := _term_pattern(term)) is not None
)


def _has_nonpilot_disease(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _NONPILOT_MATCHERS)


def _json_value(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
    return default if value is None else value


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _get(mapping: dict, key: str, default=None):
    try:
        value = mapping.get(key, default)
    except AttributeError:
        try:
            value = mapping[key]
        except (KeyError, IndexError, TypeError):
            value = default
    return default if value is None else value


def _bbox(panel: dict):
    value = _get(panel, "bbox")
    if value is None:
        value = _json_value(_get(panel, "bbox_json"), None)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(x) for x in value)
    except (TypeError, ValueError):
        return None
    if not all(0 <= x <= 1 for x in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _findings(panel: dict) -> list:
    value = _get(panel, "findings")
    if value is None:
        value = _json_value(_get(panel, "findings_json"), [])
    plate_findings = _get(panel, "plate_findings")
    if plate_findings is None:
        plate_findings = _json_value(_get(panel, "plate_findings_json"), [])
    proposed = _get(panel, "proposed_findings")
    if proposed is None:
        proposed = []
    out = []
    for item in [*(value or []), *(plate_findings or [])]:
        if isinstance(item, dict):
            out.extend((_text(item.get("finding_key")), _text(item.get("evidence"))))
        else:
            out.append(_text(item))
    if isinstance(proposed, list):
        out.extend(_text(item) for item in proposed)
    return [v for v in out if v]


def approved_findings_by_disease(conn) -> dict[str, set[str]]:
    """Approved finding keys per disease from findings_vocab."""
    allowed: dict[str, set[str]] = {}
    for row in conn.execute(
        "SELECT finding_key, disease_keys_json FROM findings_vocab WHERE approved=1"
    ):
        for key in _json_value(row["disease_keys_json"], []) or []:
            allowed.setdefault(str(key), set()).add(row["finding_key"])
    return allowed


def plate_structure(vision) -> dict | None:
    """Classify a compound figure's panels for the whole-figure plate policy.

    Returns None for a non-compound judgment, else
    ``{"plate_class", "disease_key", "patient_panels"}``.
    """
    vision = _json_value(vision, {}) or {}
    panels = vision.get("panels") or []
    if vision.get("figure_is_compound") is not True and len(panels) <= 1:
        return None
    pilot = set(diseases.DISEASE_KEYS)

    def _patient(panel: dict) -> bool:
        return (
            _get(panel, "exclusion_reason") in (None, "collage")
            and _get(panel, "modality") not in (None, "", "other")
            and _get(panel, "disease_key") in pilot
        )

    patient = [p for p in panels if _patient(p)]
    diseases_seen = {_get(p, "disease_key") for p in patient}
    if any(_get(p, "exclusion_reason") == "other_disease" for p in panels) or len(diseases_seen) > 1:
        plate_class = "multi_disease"
    elif any(
        _get(p, "exclusion_reason") not in (None, "collage", "other_disease")
        or _get(p, "modality") in (None, "", "other")
        for p in panels
    ):
        plate_class = "mixed_non_patient"
    elif len(patient) < 2:
        plate_class = "too_few_patient_panels"
    elif len(patient) != len(panels):
        plate_class = "unattributed_panel"
    else:
        plate_class = "single_disease"
    return {
        "plate_class": plate_class,
        "disease_key": next(iter(diseases_seen)) if len(diseases_seen) == 1 else None,
        "patient_panels": patient,
    }


def evaluate_plate(
    vision: dict,
    figure: dict | None,
    article: dict | None,
    allowed_by_disease: dict[str, set[str]],
    image_size: tuple[int, int] | None = None,
) -> dict | None:
    """Build the deterministic whole-figure plate record for a compound
    judgment; None for non-compound figures. ``plate["include"]`` marks a
    publishable single-disease plate; the full approved finding list lives in
    ``plate_findings`` regardless of kind.
    """
    vision = _json_value(vision, {}) or {}
    structure = plate_structure(vision)
    if structure is None:
        return None
    panels = vision.get("panels") or []
    patient = structure["patient_panels"]
    labels = [str(_get(p, "panel_label") or "?") for p in panels]
    modalities = [str(_get(p, "modality")) for p in patient]
    modality_counts = Counter(modalities)
    modality = max(dict.fromkeys(modalities), key=modality_counts.get, default=None)

    def _common(key, default=None):
        values = {_get(p, key) for p in patient}
        return values.pop() if len(values) == 1 else default

    allowed = allowed_by_disease.get(structure["disease_key"]) or set()
    plate_findings = []
    seen: set[str] = set()
    for panel in patient:
        for item in _get(panel, "findings") or []:
            key = item.get("finding_key") if isinstance(item, dict) else item
            if not key or key in seen or key not in allowed:
                continue
            seen.add(key)
            plate_findings.append({
                "finding_key": key,
                "evidence": item.get("evidence", "") if isinstance(item, dict) else "",
            })
    if not plate_findings:
        plate_kind, findings = "unlabeled", []
    elif len(plate_findings) == 1:
        plate_kind, findings = "same_finding", list(plate_findings)
    else:
        plate_kind, findings = "combined", []

    rationale = f"Whole-figure plate of {len(panels)} panels ({', '.join(labels)})."
    parts = " ".join(
        f"{label}: {_text(_get(panel, 'rationale'))}"
        for label, panel in zip(labels, panels)
        if _get(panel, "rationale")
    )
    rationale = (rationale + " " + parts).strip()
    if len(rationale) > 1000:
        rationale = rationale[:997].rsplit(" ", 1)[0] + "..."

    plate_class = structure["plate_class"]
    if plate_class != "single_disease":
        include, reason = False, f"multi-panel plate: {plate_class}"
    elif plate_kind == "unlabeled":
        include, reason = False, "unlabeled single-disease plate"
    else:
        include, reason = True, None
    plate = {
        "panel_label": PLATE_LABEL,
        "bbox": [0, 0, 1, 1],
        "crop_mode": "whole_figure",
        "plate_class": plate_class,
        "plate_kind": plate_kind,
        "source_panels": labels,
        "disease_key": structure["disease_key"],
        "modality": modality,
        "plate_modalities": sorted(set(modalities)),
        "radiology": bool(patient) and all(m in RADIOLOGY_MODALITIES for m in modalities),
        "subtype": _common("subtype"),
        "body_site": _common("body_site"),
        "typicality": _common("typicality"),
        "stage": _common("stage"),
        "stated_ethnicity": _common("stated_ethnicity"),
        "stated_ethnicity_quote": _common("stated_ethnicity_quote"),
        "age_group": _common("age_group", "unknown"),
        "skin_tone": _common("skin_tone"),
        "annotations_present": any(_get(p, "annotations_present") for p in patient),
        "confidence": min((_get(p, "confidence") or 0 for p in patient), default=0),
        "findings": findings,
        "plate_findings": plate_findings,
        "proposed_findings": list(dict.fromkeys(
            str(v) for p in patient for v in (_get(p, "proposed_findings") or [])
        )),
        "rationale": rationale,
        "include": include,
        "exclusion_reason": reason,
    }
    if include:
        gate = exclusion_reason(
            plate,
            {**dict(figure or {}), "vision_json": vision},
            article or {},
            image_size=image_size,
            allowed_findings=allowed_by_disease.get(structure["disease_key"]),
        )
        if gate:
            plate["include"] = False
            plate["exclusion_reason"] = gate
            plate["plate_class"] = "caption_gate"
    return plate


def source_exclusion_reason(figure: dict, article: dict | None = None) -> str | None:
    """Hard exclusions identifiable from article title and figure caption."""
    figure = figure or {}
    article = article or {}
    title = _text(_get(article, "title"))
    caption = _text(_get(figure, "caption"))
    if _FIGURE_DIAGRAM.search(caption):
        return "diagram or schematic"
    if _FIGURE_CHART.search(caption):
        return "chart or classification table"
    if _VETERINARY.search(title + " " + caption):
        return "veterinary image"
    triage = _json_value(_get(figure, "triage_json"), {}) or {}
    if isinstance(triage, dict) and triage.get("is_real_patient_image") is False:
        return "not a patient image"
    return None


def exclusion_reason(
    panel: dict,
    figure: dict,
    article: dict,
    *,
    image_size: tuple[int, int] | None = None,
    allowed_findings: set[str] | None = None,
) -> str | None:
    """Return the first deterministic reason a panel should not be shown.

    Inputs may be persisted SQLite rows or decoded P3 JSON objects. Reasons
    are stable text intended for audit records and viewer filtering.
    """
    panel = panel or {}
    figure = figure or {}
    article = article or {}

    # Explicit model exclusions remain authoritative, with a clearer reason.
    if not _get(panel, "include", True):
        reason = _get(panel, "exclusion_reason")
        return str(reason or "model excluded panel").replace("_", " ")

    vision = _json_value(_get(figure, "vision_json"), {}) or {}
    compound = (
        _get(figure, "figure_is_compound") is True
        or _get(figure, "compound") is True
        or vision.get("figure_is_compound") is True
        or len(vision.get("panels") or []) > 1
        or (_get(figure, "vision_panel_count") or 0) > 1
    )
    if compound:
        # Tiles/snippets of a compound figure are never published; only the
        # whole-figure plate can pass, and only for a single-disease plate
        # that carries an approved finding.
        plate_kind = _get(panel, "plate_kind")
        if plate_kind is None or _get(panel, "crop_mode") != "whole_figure":
            return "multi-panel collage; skip whole source"
        if plate_kind == "unlabeled":
            return "unlabeled single-disease plate"
        structure = plate_structure(vision)
        plate_class = structure["plate_class"] if structure else "not_compound"
        if plate_class != "single_disease":
            return f"multi-panel plate: {plate_class}"
        if structure["disease_key"] != _get(panel, "disease_key"):
            return "multi-panel plate: disease mismatch"

    source_reason = source_exclusion_reason(figure, article)
    if source_reason:
        return source_reason
    caption = _text(_get(figure, "caption"))
    target = _text(_get(panel, "disease_key"))
    matchers = _disease_matchers()
    if target in matchers:
        target_in_caption = target in _diseases_in_text(caption)
        title = _text(_get(article, "title"))
        target_in_title = target in _diseases_in_text(title)
        other_diseases = _diseases_in_text(caption) - {target}
        if (other_diseases or _has_nonpilot_disease(caption)) and not target_in_caption:
            return "caption identifies another disease"
        # Disease-focused titles can establish scope for terse captions such
        # as "Malar rash". General review topics alone cannot attribute a
        # common finding like effusion or vasculitis to the target disease.
        if not target_in_caption and not target_in_title:
            return "disease association not stated in figure caption"

    panel_text = " ".join((
        _text(_get(panel, "rationale")),
        _text(_get(panel, "modality")),
        _text(_get(panel, "body_site")),
    ))
    if _BAD_PANEL.search(panel_text):
        return "diagram, chart, or text-only panel"
    bbox = _bbox(panel)
    if bbox is None:
        vision_panel_count = _get(figure, "vision_panel_count")
        vision = _json_value(_get(figure, "vision_json"), {}) or {}
        panels = vision.get("panels") or []
        single_full_image = (
            _get(figure, "figure_is_compound") is not True
            and vision.get("figure_is_compound") is not True
            and (vision_panel_count == 1 or (not vision_panel_count and len(panels) == 1))
        )
        if not single_full_image and _get(panel, "crop_mode") != "whole_figure":
            return "missing or invalid panel bounds"
    else:
        x0, y0, x1, y1 = bbox
        width, height = x1 - x0, y1 - y0
        license_code = _text(_get(figure, "effective_license")).casefold()
        whole_image = (
            _get(panel, "crop_mode") == "whole_figure"
            or license_code.endswith("-nd")
        )
        # A small article subfigure is rarely useful in this library. In
        # addition to area, require useful extent in each direction so narrow
        # strips do not slip through on a large image.
        if not whole_image and (width * height < 0.12 or width < 0.25 or height < 0.25):
            return "panel crop too small for a useful image"
        if image_size and not whole_image:
            px_w, px_h = int(width * image_size[0]), int(height * image_size[1])
            if min(px_w, px_h) < 160:
                return "panel crop too small for a useful image"

    evidence = _findings(panel)
    evidence_text = " ".join(evidence)
    context = " ".join((caption, panel_text, evidence_text))
    if _NEGATIVE.search(context):
        return "normal or control image"
    if not evidence:
        return "no specific clinical manifestation identified"
    if allowed_findings is not None:
        raw = _get(panel, "findings")
        if raw is None:
            raw = _json_value(_get(panel, "findings_json"), [])
        plate_raw = _get(panel, "plate_findings")
        if plate_raw is None:
            plate_raw = _json_value(_get(panel, "plate_findings_json"), [])
        keys = {
            str(item.get("finding_key") if isinstance(item, dict) else item)
            for item in [*(raw or []), *(plate_raw or [])]
        }
        if not keys & allowed_findings:
            return "no finding approved for this disease"

    # Veterinary terms in panel text can be absent from the parent caption.
    if _VETERINARY.search(panel_text + " " + evidence_text):
        return "veterinary image"
    return None
