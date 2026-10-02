"""Patient age evidence shared by publication and viewer routing.

Model labels are not evidence. Numeric ages override labels; conflicting or
unattributed ages in a whole figure remain unknown rather than being guessed.
"""

import json
import re

_AGE = re.compile(r"\b(\d{1,3})\s*(?:[-–]\s*)?(years?|yrs?|months?|weeks?|days?)(?:\s*[-–]?\s*old)?\b", re.I)
_PATIENT = r"(?:patients?|children|child|boys?|girls?|men|women|male|female|infants?|neonates?)"
_GROUPS = {
    "child": r"(?:child|children|pediatric|paediatric|infant|newborn|neonate)",
    "adolescent": r"(?:adolescent|teenager|teenage)",
    "adult": r"(?:adult)",
    "older_adult": r"(?:older adult|elderly)",
}


def resolve_age(figure: dict) -> dict:
    age = _resolve_figure_age(figure)
    case_text = str(figure.get("case_age_text") or "")
    if age["age_group"] == "unknown" and case_text.strip():
        # Case reports state the patient's age in the abstract, not the
        # caption; the same conflict rules apply to that text.
        age = _resolve_figure_age({"caption": case_text})
    return age


def _resolve_figure_age(figure: dict) -> dict:
    caption = str(figure.get("caption") or figure.get("figure_caption") or "")
    mentions = figure.get("in_text_mentions_json", figure.get("in_text_mentions", []))
    if isinstance(mentions, str):
        try:
            mentions = json.loads(mentions)
        except ValueError:
            mentions = []
    # Figure captions take precedence; figure-linked mentions may fill an
    # absent age, but must never override a conflicting age in the caption.
    texts = [caption] if caption.strip() else [str(m) for m in mentions or []]
    text = " ".join(texts)
    unknown = {"age_group": "unknown", "evidence": None, "patient_age_years": None}
    # Do not assign the stated age of one patient to another unstated patient.
    if re.search(r"\b(?:another|different|second)\s+patient\b", text, re.I):
        return unknown
    ages = []
    for match in _AGE.finditer(text):
        # Restrict numbers to age expressions, excluding follow-up durations.
        if not re.search(r"old\b", match.group(), re.I):
            before = text[max(0, match.start() - 25):match.start()]
            if not re.search(r"\b(?:aged?|ages?)\s*[:=]?\s*$", before, re.I):
                continue
        number = int(match[1])
        unit = match[2].lower()
        years = number if unit.startswith(('year', 'yr')) else number / (12 if unit.startswith('month') else 52 if unit.startswith('week') else 365)
        if years > 120:
            return unknown
        group = 'child' if years < 12 else 'adolescent' if years < 18 else 'adult' if years < 65 else 'older_adult'
        ages.append((group, match.group(), years))
    if ages:
        if len({a[0] for a in ages}) != 1:
            return unknown
        return {"age_group": ages[0][0], "evidence": "; ".join(a[1] for a in ages),
                "patient_age_years": ages[0][2] if len({a[2] for a in ages}) == 1 else None}
    groups = []
    for group, words in _GROUPS.items():
        match = re.search(rf"\b{words}\s+{_PATIENT}\b|\b{_PATIENT}\s+(?:is|was|are|were)\s+(?:an?\s+)?{words}\b|\ban?\s+{words}\s+with\b", text, re.I)
        if match:
            groups.append((group, match.group()))
    # 'older adult patient' also matches 'adult patient'.
    if any(g[0] == 'older_adult' for g in groups):
        groups = [g for g in groups if g[0] != 'adult']
    if len(groups) != 1:
        if not groups and caption.strip() and mentions:
            return _resolve_figure_age({"caption": " ".join(str(m) for m in mentions)})
        return unknown
    return {"age_group": groups[0][0], "evidence": groups[0][1], "patient_age_years": None}
