"""Cheap, deterministic ranking of review articles for visual yield.

The ranker combines the retrieval evidence retained by stage 2 with article
metadata and, when available, JATS figure captions. It is deliberately a
heuristic: licensing and figure eligibility remain enforced by the existing
PMC/JATS parser. The functions are pure so ranking can be calibrated without
network or vision calls.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from . import diseases

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)
_MODALITY = {
    "clinical": re.compile(r"\b(clinical photograph|clinical photo|patient photograph|skin lesion|cutaneous|rash|papule|plaque)\b", re.I),
    "mri": re.compile(r"\b(mri|magnetic resonance|stir sequence)\b", re.I),
    "radiograph": re.compile(r"\b(radiograph|x-ray|radiographic|plain film)\b", re.I),
    "ct": re.compile(r"\b(ct scan|computed tomography)\b", re.I),
    "ultrasound": re.compile(r"\b(ultrasound|sonograph|ultrasonograph)\b", re.I),
    "histology": re.compile(r"\b(histolog|biopsy|histopatholog|immunofluorescen|hematoxylin|haematoxylin|h&e stain)\b", re.I),
}
_DIAGRAM = re.compile(r"\b(schematic|diagram|flowchart|algorithm|pathway|graphical abstract|illustration|model of pathogenesis)\b", re.I)
_MECHANISM = re.compile(r"\b(pathogenesis|molecular mechanism|signalling pathway|signaling pathway|mechanistic)\b", re.I)
_NONVISUAL_FOCUS = re.compile(
    r"\b(treatment|therapy|therapeutic|pharmacolog|drug|microbiot|mitochondri|"
    r"genetic|molecular|pathogenesis|immune mechanism|signaling pathway)\w*\b",
    re.I,
)
_EXPLICIT_IMAGE = re.compile(
    r"\b(photo(?:graph)?s?|images?|micrographs?|MRI|magnetic resonance|"
    r"radiographs?|x-rays?|CT scans?|computed tomography|ultrasound|"
    r"sonograms?|scans?|biops(?:y|ies)|H\s*&\s*E|hematoxylin|haematoxylin|"
    r"stain(?:ed|ing)?|tissue sections?)\b",
    re.I,
)
_PANEL_LABEL = re.compile(r"\([a-z0-9]+\)", re.I)
_ANATOMY = re.compile(
    r"\b(face|hand|finger|joint|skin|abdomen|chest|knee|scalp|nail|"
    r"back|spine|sacroiliac|muscle|eyelid|palm)\b",
    re.I,
)
_CLINICAL_MORPHOLOGY = re.compile(
    r"\b(rash|erythema|papules?|plaques?|ulcers?|lesions?|alopecia|"
    r"calcinosis|edema|oedema|atrophy|necrosis|poikiloderma)\b",
    re.I,
)
_LAB_ASSAY_ONLY = re.compile(r"\b(HEp-2 cells?|sera?|serum|ELISA|immunoblot|western blot)\b", re.I)
_FINDINGS = diseases.load_findings_vocab()


def _patient_panel_caption(caption: str) -> bool:
    """Recognize labeled clinical panels without matching distant text."""
    labels = list(_PANEL_LABEL.finditer(caption))
    if len(labels) < 2:
        return False
    for index, match in enumerate(labels):
        end = labels[index + 1].start() if index + 1 < len(labels) else len(caption)
        segment = caption[match.end() : end]
        if _ANATOMY.search(segment) and _CLINICAL_MORPHOLOGY.search(segment):
            return True
    return False


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(v) for v in value)
    return str(value)


def _evidence_text(item: Any) -> tuple[str, str, float]:
    if isinstance(item, Mapping):
        return (
            _text(item.get("text") or item.get("page_content") or item.get("passage")),
            _text(item.get("section") or item.get("section_title") or item.get("section_type")),
            _number(item.get("score"), 0.0),
        )
    return _text(item), "", 0.0


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _disease_terms(disease_key: str) -> list[str]:
    info = diseases.load_diseases().get(disease_key, {})
    return [str(v) for v in info.get("synonyms", []) if str(v)]


def _finding_terms(disease_key: str) -> list[str]:
    terms: list[str] = []
    for item in _FINDINGS:
        if disease_key not in item.get("disease_keys", []):
            continue
        terms.append(str(item.get("label", "")))
        terms.extend(str(v) for v in item.get("synonyms", []))
        # Keys often contain useful searchable phrases (si_bone_marrow_edema).
        terms.append(str(item.get("finding_key", "")).replace("_", " "))
    return [term for term in terms if len(term.strip()) > 2]


def _matches(text: str, terms: Iterable[str]) -> int:
    low = text.casefold()
    return sum(1 for term in set(terms) if term.casefold() in low)


def caption_signal(caption: str, disease_key: str, *, eligible: bool = True) -> dict[str, Any]:
    """Summarize figure-caption evidence without deciding legal eligibility."""
    caption = _text(caption)
    disease_hits = _matches(caption, _disease_terms(disease_key))
    finding_hits = _matches(caption, _finding_terms(disease_key))
    modalities = [name for name, pattern in _MODALITY.items() if pattern.search(caption)]
    diagram = bool(_DIAGRAM.search(caption))
    mechanism = bool(_MECHANISM.search(caption))
    explicit_image = bool(_EXPLICIT_IMAGE.search(caption))
    panel_anatomy = _patient_panel_caption(caption)
    lab_assay_only = bool(_LAB_ASSAY_ONLY.search(caption)) and not re.search(
        r"\b(biopsy|skin|muscle tissue|tissue section)\b", caption, re.I
    )
    image_context = explicit_image or panel_anatomy
    score = 0.0
    score += min(disease_hits, 2) * 2.0
    score += min(finding_hits, 4) * 2.5
    score += min(len(modalities), 2) * 2.0
    score += 1.0 if eligible else -12.0
    score += 4.0 if explicit_image else (1.5 if panel_anatomy else 0.0)
    score -= 5.0 if diagram else 0.0
    score -= 2.0 if mechanism and not (disease_hits or finding_hits or modalities) else 0.0
    score -= 4.0 if lab_assay_only else 0.0
    useful = bool(
        eligible
        and image_context
        and (disease_hits or finding_hits)
        and not diagram
        and not lab_assay_only
    )
    uncertain = bool(
        eligible and not useful and not diagram and (disease_hits or finding_hits)
    )
    if uncertain:
        score = min(score, 3.0)
    return {
        "score": score,
        "disease_hits": disease_hits,
        "finding_hits": finding_hits,
        "modalities": modalities,
        "diagram": diagram,
        "eligible": bool(eligible),
        "useful": useful,
        "uncertain": uncertain,
        "explicit_image": explicit_image,
        "panel_anatomy": panel_anatomy,
    }


def score_article(
    article: Mapping[str, Any],
    disease_key: str,
    *,
    captions: Sequence[Any] | None = None,
    coverage_gaps: Iterable[str] = (),
) -> dict[str, Any]:
    """Return a rank score and auditable feature breakdown.

    ``matched_passages`` can contain plain text or retrieval mappings with
    ``text``, ``section``/``section_title``, ``finding``, ``modality``, and
    ``score`` fields. Older rows with no evidence fall back to title, abstract,
    and ``retrieval_score``.
    """
    title = _text(article.get("title"))
    abstract = _text(article.get("abstract"))
    evidence = article.get("matched_passages") or article.get("retrieval_evidence") or []
    if isinstance(evidence, str):
        try:
            import json

            evidence = json.loads(evidence)
        except (TypeError, ValueError):
            evidence = [evidence]
    if not isinstance(evidence, (list, tuple)):
        evidence = [evidence]

    evidence_texts = [_evidence_text(v) for v in evidence]
    passage_text = " ".join(text for text, _, _ in evidence_texts)
    sections = " ".join(section for _, section, _ in evidence_texts)
    structured_terms = " ".join(
        _text(v.get(k)) for v in evidence if isinstance(v, Mapping)
        for k in ("finding", "modality", "query")
    )
    disease_terms = _disease_terms(disease_key)
    finding_terms = _finding_terms(disease_key)
    title_abstract = f"{title} {abstract}"
    score = _number(article.get("retrieval_score"), 0.0) * 10.0
    parts: dict[str, float] = {"retrieval": score}

    disease_title = _matches(title, disease_terms)
    disease_context = _matches(f"{abstract} {passage_text} {structured_terms}", disease_terms)
    finding_context = _matches(f"{title_abstract} {passage_text} {structured_terms}", finding_terms)
    visual_sections = _matches(sections, ("clinical", "imaging", "radiolog", "histolog", "patholog", "physical examination"))
    modalities = sum(bool(pattern.search(f"{title_abstract} {passage_text} {structured_terms}")) for pattern in _MODALITY.values())
    parts["disease_title"] = min(disease_title, 2) * 2.0
    parts["disease_context"] = min(disease_context, 3) * 2.0
    parts["finding_context"] = min(finding_context, 6) * 2.0
    parts["visual_section"] = min(visual_sections, 3) * 1.5
    parts["modality"] = min(modalities, 3) * 1.5

    # Findings mentioned in retrieval queries/passages that are still missing
    # from the library receive additional weight.
    gaps = {str(g).casefold().replace("_", " ") for g in coverage_gaps}
    uncovered_hits = [g for g in gaps if g and g in f"{title_abstract} {passage_text} {structured_terms}".casefold()]
    parts["coverage_gap"] = min(len(uncovered_hits), 5) * 2.5

    caption_scores = [caption_signal(_text(c.get("caption") if isinstance(c, Mapping) else c), disease_key,
                                     eligible=bool(c.get("eligible", True)) if isinstance(c, Mapping) else True)
                      for c in (captions or [])]
    # An article can rank well from one concrete relevant image even if its
    # title says "review" and discusses many mechanisms.
    useful_captions = [c for c in caption_scores if c["useful"]]
    best_caption = max((c["score"] for c in useful_captions), default=0.0)
    uncertain_captions = [c for c in caption_scores if c["uncertain"]]
    parts["caption_yield"] = (
        min(best_caption, 20.0)
        + min(max(len(useful_captions) - 1, 0), 4) * 1.5
        + min(len(uncertain_captions), 3) * 0.5
    )
    if captions is not None and not useful_captions:
        # A checked article with no eligible figure is a poor image source.
        # Keep uncertain captions available to the later figure triage.
        parts["caption_uncertainty"] = -4.0 if not captions else -2.0
    else:
        parts["caption_uncertainty"] = 0.0
    # A disease/finding mention in a treatment or mechanism review is weak
    # evidence for image yield. The penalty is soft: an actual image-bearing
    # caption can still lift a broad review into the selected batch.
    if not useful_captions and _NONVISUAL_FOCUS.search(title) and not modalities:
        parts["mechanism_only"] = -3.0
    elif not useful_captions and _MECHANISM.search(f"{title} {abstract}"):
        parts["mechanism_only"] = -2.0
    else:
        parts["mechanism_only"] = 0.0

    total = sum(parts.values())
    return {
        "score": total,
        "parts": parts,
        "useful_caption_count": len(useful_captions),
        "caption_signals": caption_scores,
        "uncovered_hits": uncovered_hits,
    }


def rank_articles(
    articles: Sequence[Mapping[str, Any]],
    disease_key: str,
    *,
    coverage_gaps: Iterable[str] = (),
) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    """Sort articles by visual-yield score with retrieval score as tie-break."""
    ranked = [
        (
            article,
            score_article(
                article,
                disease_key,
                captions=article.get("caption_candidates"),
                coverage_gaps=coverage_gaps,
            ),
        )
        for article in articles
    ]
    ranked.sort(
        key=lambda item: (
            item[1]["score"],
            _number(item[0].get("retrieval_score"), 0.0),
            str(item[0].get("pmcid", "")),
        ),
        reverse=True,
    )
    return ranked
