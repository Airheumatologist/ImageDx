"""Conservative, deterministic acceptance rules for usable clinical panels.

This policy is shared by the live judge and the viewer's persisted-row audit.
It deliberately favors a clean library over preserving uncertain crops: source
figures that are collages, diagrams, charts or veterinary material are not
split into guessed subpanels.
"""

from __future__ import annotations

import json
import re

POLICY_VERSION = "clinical-panels.v3"

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
_OTHER_DISEASE = re.compile(
    r"\b(?:sarcoidosis|scleroderma|systemic sclerosis|rheumatoid arthritis|psoriatic arthritis|polymyositis|inclusion body myositis|covid-19|interferonopathy|tafro syndrome|oro-facial granulomatosis)\b",
    re.I,
)
_TARGET_CUES = {
    "sle": re.compile(r"\b(?:sle|systemic lupus erythematosus|lupus)\b", re.I),
    "dm": re.compile(r"\b(?:dermatomyositis|juvenile dermatomyositis|anti[- ]?mda5)\b", re.I),
    "as": re.compile(r"\b(?:ankylosing spondylitis|axspa|axial spondyloarthritis|nr[- ]?axspa|r[- ]?axspa)\b", re.I),
}
_NEGATIVE = re.compile(
    r"\b(?:normal control|healthy control|unaffected control|negative control|control group|normal image|normal scan|normal mri|normal capillary bed|normal capillary pattern|unremarkable (?:image|scan|mri|finding))\b",
    re.I,
)
_BAD_PANEL = re.compile(
    r"\b(?:diagram of|schematic of|flow ?chart|classification chart|classification table|text[- ]only graphic)\b",
    re.I,
)


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
    proposed = _get(panel, "proposed_findings")
    if proposed is None:
        proposed = []
    out = []
    for item in value or []:
        if isinstance(item, dict):
            out.extend((_text(item.get("finding_key")), _text(item.get("evidence"))))
        else:
            out.append(_text(item))
    if isinstance(proposed, list):
        out.extend(_text(item) for item in proposed)
    return [v for v in out if v]


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

    if _get(figure, "figure_is_compound") is True or _get(figure, "compound") is True:
        return "multi-panel collage; skip whole source"
    vision = _json_value(_get(figure, "vision_json"), {}) or {}
    if vision.get("figure_is_compound") is True or len(vision.get("panels") or []) > 1:
        return "multi-panel collage; skip whole source"
    if (_get(figure, "vision_panel_count") or 0) > 1:
        return "multi-panel collage; skip whole source"

    source_reason = source_exclusion_reason(figure, article)
    if source_reason:
        return source_reason
    caption = _text(_get(figure, "caption"))
    target = _text(_get(panel, "disease_key"))
    if target in _TARGET_CUES:
        target_in_caption = bool(_TARGET_CUES[target].search(caption))
        title = _text(_get(article, "title"))
        target_in_title = bool(_TARGET_CUES[target].search(title))
        other_pilot_disease = any(
            cue.search(caption) for key, cue in _TARGET_CUES.items() if key != target
        )
        if (_OTHER_DISEASE.search(caption) or other_pilot_disease) and not target_in_caption:
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

    # Veterinary terms in panel text can be absent from the parent caption.
    if _VETERINARY.search(panel_text + " " + evidence_text):
        return "veterinary image"
    return None
