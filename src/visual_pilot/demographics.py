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


def _unknown_age() -> dict:
    return {"age_group": "unknown", "evidence": None, "patient_age_years": None}


def _mentions(figure: dict) -> list:
    mentions = figure.get("in_text_mentions_json", figure.get("in_text_mentions", []))
    if isinstance(mentions, str):
        try:
            mentions = json.loads(mentions)
        except ValueError:
            mentions = []
    return mentions or []


def _years(number: int, unit: str) -> float:
    if unit.startswith(('year', 'yr')):
        return number
    return number / (12 if unit.startswith('month') else 52 if unit.startswith('week') else 365)


def _age_group(years: float) -> str:
    return 'child' if years < 12 else 'adolescent' if years < 18 else 'adult' if years < 65 else 'older_adult'


def _is_age_expression(text: str, match: re.Match) -> bool:
    """Restrict numbers to age expressions, excluding follow-up durations."""
    if re.search(r"old\b", match.group(), re.I):
        return True
    before = text[max(0, match.start() - 25):match.start()]
    return bool(re.search(r"\b(?:aged?|ages?)\s*[:=]?\s*$", before, re.I))


def _numeric_ages(text: str) -> list[tuple[str, str, float]] | None:
    """(group, evidence, years) per stated age; None when an age is implausible."""
    ages = []
    for match in _AGE.finditer(text):
        if not _is_age_expression(text, match):
            continue
        years = _years(int(match[1]), match[2].lower())
        if years > 120:
            return None
        ages.append((_age_group(years), match.group(), years))
    return ages


def _word_groups(text: str) -> list[tuple[str, str]]:
    """(group, evidence) for each age-group word naming the patient."""
    groups = []
    for group, words in _GROUPS.items():
        match = re.search(rf"\b{words}\s+{_PATIENT}\b|\b{_PATIENT}\s+(?:is|was|are|were)\s+(?:an?\s+)?{words}\b|\ban?\s+{words}\s+with\b", text, re.I)
        if match:
            groups.append((group, match.group()))
    # 'older adult patient' also matches 'adult patient'.
    if any(g[0] == 'older_adult' for g in groups):
        groups = [g for g in groups if g[0] != 'adult']
    return groups


def _resolve_figure_age(figure: dict) -> dict:
    caption = str(figure.get("caption") or figure.get("figure_caption") or "")
    mentions = _mentions(figure)
    # Figure captions take precedence; figure-linked mentions may fill an
    # absent age, but must never override a conflicting age in the caption.
    texts = [caption] if caption.strip() else [str(m) for m in mentions]
    text = " ".join(texts)
    # Do not assign the stated age of one patient to another unstated patient.
    if re.search(r"\b(?:another|different|second)\s+patient\b", text, re.I):
        return _unknown_age()
    ages = _numeric_ages(text)
    if ages is None or len({a[0] for a in ages}) > 1:
        return _unknown_age()
    if ages:
        return {"age_group": ages[0][0], "evidence": "; ".join(a[1] for a in ages),
                "patient_age_years": ages[0][2] if len({a[2] for a in ages}) == 1 else None}
    groups = _word_groups(text)
    if len(groups) == 1:
        return {"age_group": groups[0][0], "evidence": groups[0][1], "patient_age_years": None}
    if not groups and caption.strip() and mentions:
        return _resolve_figure_age({"caption": " ".join(str(m) for m in mentions)})
    return _unknown_age()
